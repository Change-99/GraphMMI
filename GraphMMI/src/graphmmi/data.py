from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass
class GraphBundle:
    x: Tensor
    node_type: Tensor
    species_id: Tensor
    edge_index: Tensor
    all_positive_edge_index: Tensor
    split_pos_edge_index: dict[str, Tensor]
    split_pos_edge_attr: dict[str, Tensor]
    node_ids: np.ndarray
    node_sequences: np.ndarray
    node_feature_names: np.ndarray
    edge_attr_names: np.ndarray


def load_graph_bundle(
    path: str | Path,
    device: str | torch.device = "cpu",
    load_edge_attr: bool = True,
) -> GraphBundle:
    path = Path(path)
    data: dict[str, Any]
    if path.suffix == ".pt":
        data = torch.load(path, map_location="cpu", weights_only=False)
    else:
        data = dict(np.load(path, allow_pickle=True))

    device = torch.device(device)
    split_pos_edge_index: dict[str, Tensor] = {}
    split_pos_edge_attr: dict[str, Tensor] = {}
    for split in ["train", "val", "test"]:
        split_pos_edge_index[split] = torch.as_tensor(data[f"{split}_pos_edge_index"], dtype=torch.long, device=device)
        if load_edge_attr:
            split_pos_edge_attr[split] = torch.as_tensor(data[f"{split}_pos_edge_attr"], dtype=torch.float32, device=device)
        else:
            split_pos_edge_attr[split] = torch.empty((split_pos_edge_index[split].size(1), 0), dtype=torch.float32, device=device)

    return GraphBundle(
        x=torch.as_tensor(data["x"], dtype=torch.float32, device=device),
        node_type=torch.as_tensor(data["node_type"], dtype=torch.long, device=device),
        species_id=torch.as_tensor(data["species_id"], dtype=torch.long, device=device),
        edge_index=torch.as_tensor(data["edge_index_train_pos_undirected"], dtype=torch.long, device=device),
        all_positive_edge_index=torch.as_tensor(data["all_positive_edge_index"], dtype=torch.long, device=device),
        split_pos_edge_index=split_pos_edge_index,
        split_pos_edge_attr=split_pos_edge_attr,
        node_ids=np.asarray(data["node_ids"]),
        node_sequences=np.asarray(data.get("node_sequences", np.array([""] * len(data["node_ids"])))),
        node_feature_names=np.asarray(data["node_feature_names"]),
        edge_attr_names=np.asarray(data["edge_attr_names"] if load_edge_attr else []),
    )


def positive_pair_set(edge_index: Tensor) -> set[tuple[int, int]]:
    edge_cpu = edge_index.detach().cpu().numpy()
    return {(int(src), int(dst)) for src, dst in edge_cpu.T}


def sample_negative_edges(
    pos_edge_index: Tensor,
    node_type: Tensor,
    all_positive_edge_index: Tensor,
    neg_ratio: float = 1.0,
    strategy: str = "endpoint_corrupt",
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample unobserved miRNA-mRNA pairs.

    Negatives are excluded from the full known-positive set, not only the train
    split. This keeps validation/test positives from being sampled as negatives.
    """
    if pos_edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=node_type.device)

    n_neg = int(round(pos_edge_index.size(1) * neg_ratio))
    if n_neg <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=node_type.device)

    mirna_nodes = torch.nonzero(node_type.eq(0), as_tuple=False).flatten()
    mrna_nodes = torch.nonzero(node_type.eq(1), as_tuple=False).flatten()
    if mirna_nodes.numel() == 0 or mrna_nodes.numel() == 0:
        raise ValueError("Both miRNA and mRNA nodes are required for negative sampling.")

    blocked = positive_pair_set(all_positive_edge_index)
    sampled: list[tuple[int, int]] = []
    local_seen: set[tuple[int, int]] = set()
    max_attempts = max(10000, n_neg * 200)
    attempts = 0

    while len(sampled) < n_neg and attempts < max_attempts:
        attempts += 1
        if strategy == "endpoint_corrupt":
            edge_idx = int(torch.randint(pos_edge_index.size(1), (1,), generator=generator, device=node_type.device).item())
            src = int(pos_edge_index[0, edge_idx].item())
            dst = int(pos_edge_index[1, edge_idx].item())
            if bool(torch.randint(2, (1,), generator=generator, device=node_type.device).item()):
                new_dst = int(mrna_nodes[torch.randint(mrna_nodes.numel(), (1,), generator=generator, device=node_type.device)].item())
                pair = (src, new_dst)
            else:
                new_src = int(mirna_nodes[torch.randint(mirna_nodes.numel(), (1,), generator=generator, device=node_type.device)].item())
                pair = (new_src, dst)
        elif strategy == "uniform":
            pair = (
                int(mirna_nodes[torch.randint(mirna_nodes.numel(), (1,), generator=generator, device=node_type.device)].item()),
                int(mrna_nodes[torch.randint(mrna_nodes.numel(), (1,), generator=generator, device=node_type.device)].item()),
            )
        else:
            raise ValueError(f"Unknown negative sampling strategy: {strategy}")

        if pair in blocked or pair in local_seen:
            continue
        sampled.append(pair)
        local_seen.add(pair)

    if len(sampled) != n_neg:
        raise RuntimeError(f"Failed to sample {n_neg} negatives; sampled {len(sampled)}")
    return torch.tensor(sampled, dtype=torch.long, device=node_type.device).t().contiguous()


def make_link_batch(
    pos_edge_index: Tensor,
    neg_edge_index: Tensor,
    pos_edge_attr: Tensor | None = None,
    use_edge_attr: bool = False,
) -> tuple[Tensor, Tensor, Tensor | None]:
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    labels = torch.cat(
        [
            torch.ones(pos_edge_index.size(1), dtype=torch.float32, device=pos_edge_index.device),
            torch.zeros(neg_edge_index.size(1), dtype=torch.float32, device=pos_edge_index.device),
        ],
        dim=0,
    )
    edge_attr = None
    if use_edge_attr:
        if pos_edge_attr is None:
            raise ValueError("pos_edge_attr is required when use_edge_attr=True.")
        zero_neg_attr = torch.zeros(
            (neg_edge_index.size(1), pos_edge_attr.size(1)),
            dtype=pos_edge_attr.dtype,
            device=pos_edge_attr.device,
        )
        edge_attr = torch.cat([pos_edge_attr, zero_neg_attr], dim=0)
    order = torch.randperm(labels.numel(), device=labels.device)
    edge_label_index = edge_label_index[:, order]
    labels = labels[order]
    if edge_attr is not None:
        edge_attr = edge_attr[order]
    return edge_label_index, labels, edge_attr


PAIR_FEATURE_NAMES = np.asarray(
    [
        "pair_mirna_log_len",
        "pair_mrna_log_len",
        "pair_log_len_ratio",
        "pair_log_len_absdiff",
        "pair_mirna_gc",
        "pair_mrna_gc",
        "pair_gc_absdiff",
        "pair_gc_product",
        "pair_seed_2_7_exact",
        "pair_seed_3_8_exact",
        "pair_seed_2_8_exact",
        "pair_seed_2_7_count_norm",
        "pair_seed_3_8_count_norm",
        "pair_seed_2_8_count_norm",
        "pair_seed_2_7_gc",
        "pair_seed_3_8_gc",
        "pair_seed_2_8_gc",
    ],
    dtype=str,
)

_COMPLEMENT = str.maketrans({"A": "U", "U": "A", "C": "G", "G": "C", "T": "A"})


def pair_feature_dim() -> int:
    return int(len(PAIR_FEATURE_NAMES))


def reverse_complement(seq: str) -> str:
    return str(seq).upper().replace("T", "U").translate(_COMPLEMENT)[::-1]


def gc_fraction(seq: str) -> float:
    valid = [base for base in str(seq).upper().replace("T", "U") if base in {"A", "C", "G", "U"}]
    if not valid:
        return 0.0
    return float((valid.count("G") + valid.count("C")) / len(valid))


def normalized_substring_count(pattern: str, text: str) -> float:
    if not pattern or not text or len(text) < len(pattern):
        return 0.0
    count = 0
    start = 0
    while True:
        idx = text.find(pattern, start)
        if idx < 0:
            break
        count += 1
        start = idx + 1
    return float(count / max(len(text) - len(pattern) + 1, 1))


def pair_feature_matrix(edge_index: Tensor, graph: GraphBundle) -> Tensor:
    """Compute label-safe pair features for positive and dynamic negative edges.

    These features use only the two endpoint sequences, so they are available for
    every candidate miRNA-mRNA pair and do not encode whether a pair was observed.
    """
    if edge_index.numel() == 0:
        return torch.empty((0, pair_feature_dim()), dtype=torch.float32, device=edge_index.device)

    pairs = edge_index.detach().cpu().numpy().T
    sequences = np.asarray(graph.node_sequences).astype(str)
    rows: list[list[float]] = []
    for src, dst in pairs:
        mirna_seq = sequences[int(src)].upper().replace("T", "U")
        mrna_seq = sequences[int(dst)].upper().replace("T", "U")
        mirna_len = len(mirna_seq)
        mrna_len = len(mrna_seq)
        mirna_log_len = float(np.log1p(mirna_len))
        mrna_log_len = float(np.log1p(mrna_len))
        mirna_gc = gc_fraction(mirna_seq)
        mrna_gc = gc_fraction(mrna_seq)

        seed_2_7 = reverse_complement(mirna_seq[1:7])
        seed_3_8 = reverse_complement(mirna_seq[2:8])
        seed_2_8 = reverse_complement(mirna_seq[1:8])
        seed_patterns = [seed_2_7, seed_3_8, seed_2_8]
        counts = [normalized_substring_count(seed, mrna_seq) for seed in seed_patterns]
        exact = [1.0 if count > 0.0 else 0.0 for count in counts]
        seed_gc = [gc_fraction(seed) for seed in [mirna_seq[1:7], mirna_seq[2:8], mirna_seq[1:8]]]

        rows.append(
            [
                mirna_log_len,
                mrna_log_len,
                mirna_log_len / max(mrna_log_len, 1e-6),
                abs(mirna_log_len - mrna_log_len),
                mirna_gc,
                mrna_gc,
                abs(mirna_gc - mrna_gc),
                mirna_gc * mrna_gc,
                *exact,
                *counts,
                *seed_gc,
            ]
        )
    return torch.tensor(rows, dtype=torch.float32, device=edge_index.device)
