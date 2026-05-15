#!/usr/bin/env python3
"""Layer-wise relation-aware GNN — per-layer fusion of interaction / miRNA-sim / mRNA-sim.

每一层同时跑三种关系的卷积，输入相同，输出用 gate 融合后再送入下一层。
这样消息可以在 interaction ↔ similarity 之间交替传播。

Usage:
  python scripts/multi_encoders.py \\
      --species human cow mouse worm --epochs 40 --patience 12 \\
      --mirna-sim-edges --mrna-sim-edges \\
      --run-root exp/exp4/multi_encoder/result
"""
from __future__ import annotations

import argparse, csv, gc, json, random, sys, time
from pathlib import Path

import numpy as np; import torch; import torch.nn.functional as F
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graphmmi import load_graph_bundle, pair_feature_dim, pair_feature_matrix, sample_negative_edges
from graphmmi.data import GraphBundle, positive_pair_set
from graphmmi.models import MultiEncoderPredictor

SPECIES_ORDER = ["human", "cow", "mouse", "worm"]
METRICS = ["auc", "aupr", "acc", "f1", "mcc"]


# ==================================================================
# helpers
# ==================================================================

def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def stable_seed(b: int, *p: str) -> int:
    v = b
    for part in p:
        for i, ch in enumerate(part): v += (i + 1) * ord(ch)
    return int(v % (2**31 - 1))

def avg_ranks(v: np.ndarray) -> np.ndarray:
    o = np.argsort(v, kind="mergesort"); r = np.empty(len(v), dtype=np.float64)
    sv = v[o]; start = 0
    while start < len(v):
        end = start + 1
        while end < len(v) and sv[end] == sv[start]: end += 1
        r[o[start:end]] = (start + 1 + end) / 2.0; start = end
    return r

def roc_auc_np(l: np.ndarray, s: np.ndarray) -> float:
    l = l.astype(np.int64); n_pos = int(l.sum()); n_neg = len(l) - n_pos
    if n_pos == 0 or n_neg == 0: return float("nan")
    ranks = avg_ranks(s)
    return (float(ranks[l == 1].sum()) - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)

def avg_prec_np(l: np.ndarray, s: np.ndarray) -> float:
    l = l.astype(np.int64); n_pos = int(l.sum())
    if n_pos == 0: return float("nan")
    o = np.argsort(-s, kind="mergesort"); sl = l[o]; tp = np.cumsum(sl)
    return float((tp / (np.arange(len(sl)) + 1) * sl).sum() / n_pos)

def bin_metrics(l: np.ndarray, s: np.ndarray, thr: float) -> dict:
    p = (s >= thr).astype(np.int64); l = l.astype(np.int64)
    tp = float(((p == 1) & (l == 1)).sum()); tn = float(((p == 0) & (l == 0)).sum())
    fp = float(((p == 1) & (l == 0)).sum()); fn = float(((p == 0) & (l == 1)).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    prec = tp / max(tp + fp, 1.0); rec = tp / max(tp + fn, 1.0)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    denom = np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-12))
    return {"acc": acc, "f1": f1, "mcc": (tp * tn - fp * fn) / denom}

def compute_metrics(labels: Tensor, logits: Tensor, thr: float) -> dict[str, float]:
    ln = labels.detach().cpu().numpy().astype(np.int64)
    sc = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
    out = {"auc": float(roc_auc_np(ln, sc)), "aupr": float(avg_prec_np(ln, sc))}
    out.update(bin_metrics(ln, sc, thr)); return out

def select_thr(labels: Tensor, logits: Tensor, metric: str, fb: float):
    ln = labels.detach().cpu().numpy().astype(np.int64)
    sc = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
    cands = np.unique(np.concatenate([sc, np.array([0.0, fb, 1.0])]))
    best_t, best_v = float(fb), -1e9; best_b = bin_metrics(ln, sc, best_t)
    for t in cands:
        cur = bin_metrics(ln, sc, float(t)); cv = cur[metric]
        if cv > best_v + 1e-12 or (abs(cv - best_v) <= 1e-12 and abs(float(t) - fb) < abs(best_t - fb)):
            best_t, best_v, best_b = float(t), cv, cur
    return best_t, best_b

def clone_sd(m: torch.nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

def compatible_sd(src_sd: dict[str, Tensor], tgt_m: torch.nn.Module) -> dict[str, Tensor]:
    tgt_sd = tgt_m.state_dict(); out = {}
    for k, v in src_sd.items():
        if k in tgt_sd and tuple(tgt_sd[k].shape) == tuple(v.shape): out[k] = v
    return out

def collect_mem(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()

def _edge_count(ei) -> int:
    return 0 if ei is None else int(ei.size(1))


# ==================================================================
# data loading
# ==================================================================

def load_graph(sp: str, args, device: torch.device) -> GraphBundle:
    path = args.processed_dir / sp / "graph_inputs.pt"
    g = load_graph_bundle(path, device=device, load_edge_attr=False)
    g.positive_pair_cache = positive_pair_set(g.all_positive_edge_index)
    return g

def get_branch_edges(g: GraphBundle, args, device: torch.device):
    """Return (int_ei, mirna_ei, mirna_ew, mrna_ei) from graph."""
    int_ei = g.edge_index
    mirna_ei = g.similarity_edge_index_mirna if (args.mirna_sim_edges and g.similarity_edge_index_mirna is not None and g.similarity_edge_index_mirna.numel()) else None
    mirna_ew = g.similarity_edge_weight_mirna if (mirna_ei is not None and g.similarity_edge_weight_mirna is not None) else None
    mrna_ei = g.similarity_edge_index_mrna if (args.mrna_sim_edges and g.similarity_edge_index_mrna is not None and g.similarity_edge_index_mrna.numel()) else None
    return int_ei, mirna_ei, mirna_ew, mrna_ei


# ==================================================================
# model
# ==================================================================

def build_model(g: GraphBundle, args, device: torch.device) -> MultiEncoderPredictor:
    return MultiEncoderPredictor(
        num_numeric_features=int(g.x.size(1)), num_nodes=int(g.x.size(0)),
        hidden_dim=args.hidden_dim,
        int_layers=args.int_layers,
        mirna_sim_layers=args.mirna_sim_layers,
        mrna_sim_layers=args.mrna_sim_layers,
        dropout=args.dropout, edge_attr_dim=pair_feature_dim(),
        int_conv=args.int_conv, mirna_sim_conv=args.mirna_sim_conv, mrna_sim_conv=args.mrna_sim_conv,
        gat_heads=args.gat_heads, gat_concat=args.gat_concat,
        id_embedding_dim=args.id_embedding_dim,
        type_embedding_dim=args.type_embedding_dim,
        species_embedding_dim=args.species_embedding_dim,
        decoder_layer_norm=args.decoder_layer_norm,
    ).to(device)


# ==================================================================
# training
# ==================================================================

@torch.no_grad()
def make_batch(g: GraphBundle, split: str, seed: int, neg_ratio: float,
               strategy: str, device: torch.device):
    gen = torch.Generator(device=device).manual_seed(seed)
    pos = g.split_pos_edge_index[split]
    neg = sample_negative_edges(pos, g.node_type, g.all_positive_edge_index,
                                neg_ratio=neg_ratio, strategy=strategy,
                                generator=gen, node_sequences=g.node_sequences,
                                blocked_pairs=g.positive_pair_cache)
    lei = torch.cat([pos, neg], dim=1)
    lab = torch.cat([torch.ones(pos.size(1), device=device), torch.zeros(neg.size(1), device=device)])
    return lei, lab, pair_feature_matrix(lei, g)


def train_one(
    model: MultiEncoderPredictor, g: GraphBundle, source: str,
    int_ei: Tensor, mirna_ei: Tensor | None, mirna_ew: Tensor | None, mrna_ei: Tensor | None,
    args, device: torch.device,
) -> tuple[dict[str, float], dict[str, Tensor]]:
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    best_sd = clone_sd(model); best_ep, best_aupr, best_auc = 0, -float("inf"), -float("inf"); stale = 0

    # fixed val/test batches — 不会每个 epoch 都变
    val_batch = make_batch(g, "val", stable_seed(args.seed, source, "val_fixed"),
                           args.eval_neg_ratio, args.neg_strategy, device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        lei, lab, ea = make_batch(g, "train", stable_seed(args.seed, source, "train", str(epoch)),
                                  args.neg_ratio, args.neg_strategy, device)
        opt.zero_grad(set_to_none=True)
        logits = model(g.x, g.node_type, g.species_id, int_ei, lei, edge_attr=ea,
                       mirna_edge_index=mirna_ei, mirna_edge_weight=mirna_ew, mrna_edge_index=mrna_ei)
        loss = F.binary_cross_entropy_with_logits(logits, lab)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0); opt.step()

        model.eval()
        with torch.no_grad():
            lei_v, lab_v, ea_v = val_batch
            logits_v = model(g.x, g.node_type, g.species_id, int_ei, lei_v, edge_attr=ea_v,
                             mirna_edge_index=mirna_ei, mirna_edge_weight=mirna_ew, mrna_edge_index=mrna_ei)
            val_aupr = float(avg_prec_np(lab_v.cpu().numpy().astype(np.int64),
                                         torch.sigmoid(logits_v).cpu().numpy().astype(np.float64)))
            val_auc = float(roc_auc_np(lab_v.cpu().numpy().astype(np.int64),
                                       torch.sigmoid(logits_v).cpu().numpy().astype(np.float64)))

        improved = val_aupr > best_aupr
        if improved:
            best_aupr, best_auc, best_ep, best_sd = val_aupr, val_auc, epoch, clone_sd(model); stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or improved:
            tag = " *" if improved else ""
            print(f"[{source}] e={epoch:03d} loss={loss.item():.4f} val_aupr={val_aupr:.4f} val_auc={val_auc:.4f}{tag}")
        if stale >= args.patience: break

    model.load_state_dict(best_sd)
    return {"best_epoch": best_ep, "best_val_aupr": best_aupr, "best_val_auc": best_auc}, best_sd


# ==================================================================
# main
# ==================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer-wise relation-aware GNN")
    p.add_argument("--species", nargs="+", default=["human"])
    p.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed/graph/random")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--device", default="cpu"); p.add_argument("--seed", type=int, default=42)
    # training
    p.add_argument("--epochs", type=int, default=60); p.add_argument("--patience", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--neg-strategy", default="endpoint_corrupt")
    p.add_argument("--neg-ratio", type=float, default=1.0); p.add_argument("--eval-neg-ratio", type=float, default=1.0)
    p.add_argument("--threshold-metric", default="mcc")
    p.add_argument("--eval-setting", choices=["strict_zero_shot", "calibrated_zero_shot"],
                   default="calibrated_zero_shot")
    # edges
    p.add_argument("--mirna-sim-edges", action="store_true"); p.add_argument("--mrna-sim-edges", action="store_true")
    # model
    p.add_argument("--int-layers", type=int, default=4)
    p.add_argument("--mirna-sim-layers", type=int, default=1)
    p.add_argument("--mrna-sim-layers", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--int-conv", default="graphsage")
    p.add_argument("--mirna-sim-conv", default="gatv2")
    p.add_argument("--mrna-sim-conv", default="graphsage")
    p.add_argument("--gat-heads", type=int, default=2); p.add_argument("--gat-concat", action="store_true")
    p.add_argument("--id-embedding-dim", type=int, default=0)
    p.add_argument("--type-embedding-dim", type=int, default=8)
    p.add_argument("--species-embedding-dim", type=int, default=0)
    p.add_argument("--decoder-layer-norm", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args(); device = torch.device(args.device)
    args.run_root.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    # save config
    (args.run_root / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))

    rows: list[dict] = []; source_states: dict[str, dict[str, Tensor]] = {}

    # ---- train on each source ----
    for source in args.species:
        g = load_graph(source, args, device)
        int_ei, mirna_ei, mirna_ew, mrna_ei = get_branch_edges(g, args, device)
        print(f"\n[load {source}] int={_edge_count(int_ei)} mirna_sim={_edge_count(mirna_ei)} mrna_sim={_edge_count(mrna_ei)}")
        model = build_model(g, args, device)
        info, best_sd = train_one(model, g, source, int_ei, mirna_ei, mirna_ew, mrna_ei, args, device)
        source_states[source] = best_sd
        print(f"[{source}] done: best_epoch={info['best_epoch']} best_val_aupr={info['best_val_aupr']:.4f}")
        del model, g; collect_mem(device)

    # ---- zero-shot eval on all (source, target) ----
    # Same-target eval negatives are FIXED (keyed by target, not source).
    eval_cache: dict[str, dict[str, tuple]] = {}  # target -> {split: batch_tensors}

    for target in args.species:
        g = load_graph(target, args, device)
        int_ei, mirna_ei, mirna_ew, mrna_ei = get_branch_edges(g, args, device)
        print(f"\n[eval target={target}] int={_edge_count(int_ei)} mirna_sim={_edge_count(mirna_ei)} mrna_sim={_edge_count(mrna_ei)}")

        # fixed val/test batches per target (same for all sources)
        eval_cache[target] = {}
        for split, tag in [("val", "thr_fixed"), ("test", "test_fixed")]:
            eval_cache[target][split] = make_batch(
                g, split, stable_seed(args.seed, target, tag),
                args.eval_neg_ratio, args.neg_strategy, device)

        for source in args.species:
            model = build_model(g, args, device)
            sd = compatible_sd(source_states[source], model)
            model.load_state_dict(sd, strict=False)
            model.eval()

            with torch.no_grad():
                lei_v, lab_v, ea_v = eval_cache[target]["val"]
                vl = model(g.x, g.node_type, g.species_id, int_ei, lei_v, edge_attr=ea_v,
                           mirna_edge_index=mirna_ei, mirna_edge_weight=mirna_ew, mrna_edge_index=mrna_ei)
                if args.eval_setting == "strict_zero_shot":
                    thr = 0.5
                else:
                    thr, _ = select_thr(lab_v, vl, args.threshold_metric, 0.5)

                lei_t, lab_t, ea_t = eval_cache[target]["test"]
                lt = model(g.x, g.node_type, g.species_id, int_ei, lei_t, edge_attr=ea_t,
                           mirna_edge_index=mirna_ei, mirna_edge_weight=mirna_ew, mrna_edge_index=mrna_ei)
                m = compute_metrics(lab_t, lt, thr)
            rows.append({"source": source, "target": target, **{k: m[k] for k in METRICS},
                         "loss": m.get("loss", float("nan")), "thr": thr})
            print(f"[{source}->{target}] AUC={m['auc']:.4f} AUPR={m['aupr']:.4f} F1={m['f1']:.4f}")
            del model; collect_mem(device)
        del g; collect_mem(device)

    # ---- save ----
    csv_path = args.run_root / "multi_encoder_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "target"] + METRICS + ["loss", "thr"])
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {csv_path}")
    print(f"Mean AUPR: {sum(r['aupr'] for r in rows)/len(rows):.4f}")


if __name__ == "__main__":
    main()
