# GraphMMI：基于图表示学习的 miRNA-mRNA 相互作用预测

本仓库用于本科毕业设计/科研实验：**基于图表示学习的微小 RNA 与编码 RNA 相互作用预测系统的设计与实现**。项目围绕 miRNA-mRNA interaction prediction 展开，比较传统表格特征模型与图神经网络模型在跨物种迁移学习场景下的表现。

当前仓库主要包含两个部分：

- `GraphMMI/`：本项目的核心代码，包含图数据构建、GraphSAGE/GATv2 训练、迁移学习实验、消融实验和简单前端原型。
- `TransferLearningMTI-baseline/`：原始 TransferLearningMTI 表格迁移学习 baseline 的复现/参考代码。

研究物种包括：

```text
human, cow, mouse, worm
```

核心任务是：给定 miRNA 与 mRNA 或 target site，预测二者是否存在相互作用，并分析跨物种迁移能力。

---

## 1. 项目特点

本项目包含以下实验主线：

1. **表格 baseline 复现**
   - ANN
   - XGBoost
   - 使用原始人工特征进行二分类和迁移学习

2. **图建模主线**
   - 将 miRNA 与 mRNA/target site 建成二部图或 target-site-level 图
   - positive interaction 作为真实边
   - negative interaction 在训练阶段动态采样
   - 使用 GraphSAGE 和 GATv2 进行链接预测

3. **跨物种迁移学习**
   - strict zero-shot
   - calibrated zero-shot
   - fine-tune
   - source-target transfer matrix

4. **图结构增强**
   - miRNA-miRNA sequence similarity edges
   - target-site similarity edges
   - target-site-aware node representation

5. **消融实验**
   - 相似边消融
   - encoder 层数消融
   - decoder 结构消融
   - negative sampling 策略消融

6. **系统原型**
   - `GraphMMI/ui/` 提供前端页面和本地服务，用于展示数据、训练配置、实验结果和预测演示。

---

## 2. 仓库结构

```text
GProject/
├── README.md
├── .gitignore
├── CLAUDE.md
├── run.ipynb
├── train_gnn_transfer.log
├── GraphMMI/
│   ├── requirements.txt
│   ├── AGENTS.md
│   ├── research.db
│   ├── src/
│   │   └── graphmmi/
│   │       ├── __init__.py
│   │       ├── data.py
│   │       └── models.py
│   ├── scripts/
│   │   ├── final_embedding.py
│   │   ├── train_gnn_transfer.py
│   │   ├── baseline_ann_xgb_transfer.py
│   │   ├── decoder_optimed.py
│   │   ├── multi_encoders.py
│   │   ├── pretrain_graphmae.py
│   │   ├── preprocess_graph_data.py
│   │   ├── embedding_optimed_v2.py
│   │   └── ...
│   ├── final_exp/
│   │   ├── exp1/
│   │   ├── exp2/
│   │   └── exp3/
│   ├── exp/
│   ├── ui/
│   │   ├── index.html
│   │   ├── app.js
│   │   ├── styles.css
│   │   └── server.py
│   ├── data/
│   └── runs/
└── TransferLearningMTI-baseline/
    ├── README.md
    ├── environment.yml
    ├── notebooks/
    ├── src/
    ├── data/
    └── models/
```

说明：

- `GraphMMI/src/graphmmi/` 是可复用核心模块。
- `GraphMMI/scripts/` 是实验脚本入口。
- `GraphMMI/final_exp/` 是论文最终实验脚本组织目录。
- `GraphMMI/exp/` 是早期探索和中间消融实验目录。
- `GraphMMI/data/`、`GraphMMI/runs/` 默认被 `.gitignore` 忽略，不建议直接上传大数据和训练输出。

---

## 3. 环境安装

建议使用 Python 3.10 或以上版本。

```bash
cd /mnt/d/Projects/GProject/GraphMMI

conda create -n graphmmi python=3.10
conda activate graphmmi

pip install -r requirements.txt
```

`GraphMMI/requirements.txt` 中包含：

```text
numpy
pandas
torch
matplotlib
tensorflow
xgboost
seaborn
```

其中：

- GNN 主线需要：`numpy`、`pandas`、`torch`、`matplotlib`
- baseline 复现需要：`tensorflow`、`xgboost`、`seaborn`

如果只运行 GNN，可以不关注 TensorFlow/XGBoost；如果要复现 baseline，则需要完整安装。

---

## 4. 数据准备

原始数据放在：

```text
GraphMMI/data/external/
```

期望文件格式：

```text
human1_pos.csv
human1_neg.csv
human2_pos.csv
human2_neg.csv
human3_pos.csv
human3_neg.csv
cow1_pos.csv
cow1_neg.csv
mouse1_pos.csv
mouse1_neg.csv
mouse2_pos.csv
mouse2_neg.csv
worm1_pos.csv
worm1_neg.csv
worm2_pos.csv
worm2_neg.csv
```

GNN 图建模默认只使用 `*_pos.csv` 构建正边；负样本不从 `*_neg.csv` 直接读取，而是在训练过程中动态采样。

`.gitignore` 已经忽略所有 `data/` 目录：

```text
data/
**/data/
```

因此同步到 GitHub 时，数据文件不会被上传。其他人在复现实验时，需要自行准备相同格式的数据。

---

## 5. 图建模数据格式

当前论文主线使用 target-site-level 图：

```bash
cd GraphMMI

python -u scripts/final_embedding.py \
  --node-mode target_site \
  --sim-mode topk \
  --mirna-sim-topk 5 \
  --mrna-sim-topk 5 \
  --output-dir data/processed/graph/final_target_site
```

生成目录：

```text
GraphMMI/data/processed/graph/final_target_site/
├── preprocess_summary.json
├── human/
│   ├── graph_inputs.pt
│   ├── graph_inputs.npz
│   ├── metadata.json
│   ├── nodes.csv
│   ├── positive_edges.csv
│   ├── train_pos_edges.csv
│   ├── val_pos_edges.csv
│   └── test_pos_edges.csv
├── cow/
├── mouse/
└── worm/
```

target-site 图的统计示例：

| 物种 | 节点总数 | miRNA 节点数 | target site 节点数 | 正样本边数 | 训练集 | 验证集 | 测试集 | miRNA 相似边 | target site 相似边 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| human | 9,641 | 605 | 9,036 | 9,097 | 6,550 | 728 | 1,819 | 5,262 | 71,588 |
| cow | 14,700 | 165 | 14,535 | 14,675 | 10,566 | 1,174 | 2,935 | 1,232 | 107,680 |
| mouse | 18,449 | 430 | 18,019 | 18,111 | 13,040 | 1,449 | 3,622 | 3,174 | 144,914 |
| worm | 2,267 | 122 | 2,145 | 2,166 | 1,560 | 173 | 433 | 1,012 | 17,524 |

---

## 6. 核心模型结构

核心模型位于：

```text
GraphMMI/src/graphmmi/models.py
```

主模型为 `GraphMMILinkPredictor`，整体结构如下：

```text
NodeInputEncoder
  -> GraphSAGEEncoder 或 GATv2Encoder
  -> LinkDecoder
  -> interaction logit
```

### 6.1 NodeInputEncoder

节点初始表示由以下部分组成：

```text
数值序列特征
+ node type embedding
+ 可选 ID embedding
+ 可选 species embedding
```

在 zero-shot 设置中，ID/species embedding 会被禁用，避免跨物种迁移时使用无生物学对齐意义的本地节点编号。

### 6.2 GraphSAGEEncoder

项目中自实现了 full-batch GraphSAGE，不依赖 PyTorch Geometric。聚合方式为邻居均值聚合，并支持：

- edge weight
- residual connection
- layer norm
- dropout

### 6.3 GATv2Encoder

项目中同样自实现了 GATv2 层，用 scatter 操作完成 attention 计算。GATv2 支持：

- multi-head attention
- edge weight bias
- residual connection
- layer norm
- dropout

### 6.4 LinkDecoder

边预测 decoder 使用源节点和目标节点 embedding 的组合：

```text
[z_src, z_dst, z_src * z_dst, |z_src - z_dst|, pair_features]
```

然后输入 MLP 输出 interaction logit。

pair features 由 miRNA 与 target/mRNA 序列实时计算，确保正负边使用同一套可计算特征，避免“正边有真实人工特征，负边填 0”导致的标签泄露。

---

## 7. 负采样策略

负采样在：

```text
GraphMMI/src/graphmmi/data.py
```

支持以下策略：

| 策略 | 含义 |
|---|---|
| `endpoint_corrupt` | 固定正边一端，随机替换另一端 |
| `random` / `uniform` | 从全部未观测 miRNA-target pair 中均匀采样 |
| `degree_aware` | 按节点度分布采样，使负样本更接近正样本的度分布 |
| `sequence_aware` | 根据 seed 或序列相似性构造更难的负样本 |

验证集和测试集默认使用 fixed negatives，保证不同模型、不同设置之间评估样本一致。

---

## 8. GNN 主实验

当前主训练脚本：

```text
GraphMMI/scripts/train_gnn_transfer.py
```

注意：当前最终实验要求先手动运行 `final_embedding.py`，再传入 `--skip-preprocess`。

### 8.1 运行 GraphSAGE 主实验

```bash
cd GraphMMI

python -u scripts/train_gnn_transfer.py \
  --species human cow mouse worm \
  --encoders graphsage \
  --settings strict_zero_shot finetune \
  --epochs 40 \
  --patience 8 \
  --num-layers 4 \
  --graphsage-hidden-dim 128 \
  --processed-dir data/processed/graph/final_target_site \
  --mirna-sim-edges --mrna-sim-edges \
  --skip-preprocess \
  --refresh-fixed-negatives \
  --run-root final_exp/exp2/result
```

### 8.2 运行 GATv2 主实验

```bash
cd GraphMMI

python -u scripts/train_gnn_transfer.py \
  --species human cow mouse worm \
  --encoders gatv2 \
  --settings strict_zero_shot finetune \
  --epochs 40 \
  --patience 8 \
  --num-layers 1 \
  --gatv2-hidden-dim 64 \
  --processed-dir data/processed/graph/final_target_site \
  --mirna-sim-edges --mrna-sim-edges \
  --skip-preprocess \
  --refresh-fixed-negatives \
  --run-root final_exp/exp2/result
```

也可以直接运行：

```bash
bash final_exp/exp2/run.sh
```

---

## 9. 迁移学习设置

训练脚本支持：

| setting | 含义 |
|---|---|
| `strict_zero_shot` | 不使用目标物种训练数据，阈值固定为 0.5 |
| `calibrated_zero_shot` | 不更新模型参数，但使用目标验证集选择阈值 |
| `finetune` | 从 source 初始化，在 target 训练集上微调，并用 target 验证集早停/选阈值 |

微调策略：

| 策略 | 含义 |
|---|---|
| `full` | 微调整个模型 |
| `last_layer` | 冻结大部分 encoder，只微调最后一层 GNN 和 decoder |
| `decoder` | 只微调 decoder |

---

## 10. Baseline 复现

baseline 脚本：

```text
GraphMMI/scripts/baseline_ann_xgb_transfer.py
```

该脚本复现传统表格机器学习设置：

- 使用 `*_pos.csv` 和 `*_neg.csv`
- 提取公共数值特征
- 训练 ANN 和 XGBoost
- 执行跨物种迁移评估
- 输出 transfer matrix、CSV 和 heatmap

检查数据：

```bash
cd GraphMMI
python scripts/baseline_ann_xgb_transfer.py --check-data
```

运行完整 baseline：

```bash
cd GraphMMI
python scripts/baseline_ann_xgb_transfer.py \
  --models ann xgb \
  --transfer-size 500 \
  --run-root runs/baseline_transfer
```

论文最终 baseline 脚本：

```bash
bash final_exp/exp1/A_baseline/run.sh
```

---

## 11. 最终实验目录

最终实验集中在：

```text
GraphMMI/final_exp/
```

### 11.1 Exp1：baseline 与 mRNA-level GNN

```text
final_exp/exp1/
├── A_baseline/
│   ├── run.sh
│   └── summary.py
└── B_mrna_gnn/
    └── run.sh
```

用于比较传统 ANN/XGB baseline 和早期 mRNA-level GNN。

### 11.2 Exp2：target-site-level 主模型

```text
final_exp/exp2/
├── run.sh
├── summary.py
├── summary_metrics.csv
└── result/
```

运行：

```bash
bash final_exp/exp2/run.sh
```

### 11.3 Exp3：消融实验

```text
final_exp/exp3/
├── A_edges/
│   └── run.sh
├── B_encoders/
│   └── run.sh
├── C_decoder/
│   └── run.sh
├── D_neg_sample/
│   └── run.sh
├── README.md
└── run_all.sh
```

运行全部消融：

```bash
bash final_exp/exp3/run_all.sh
```

单独运行：

```bash
bash final_exp/exp3/A_edges/run.sh
bash final_exp/exp3/B_encoders/run.sh
bash final_exp/exp3/C_decoder/run.sh
bash final_exp/exp3/D_neg_sample/run.sh
```

各组含义：

| 目录 | 实验 |
|---|---|
| `A_edges` | no-sim、miRNA-only、target-only、both-sim |
| `B_encoders` | GraphSAGE/GATv2 的 1-6 层深度消融 |
| `C_decoder` | baseline、residual、gated、bilinear、separated decoder 消融 |
| `D_neg_sample` | endpoint corrupt、degree-aware、sequence-aware 负采样消融 |

---

## 12. UI 系统原型

UI 位于：

```text
GraphMMI/ui/
```

包含：

```text
index.html
styles.css
app.js
server.py
```

启动：

```bash
cd GraphMMI
python ui/server.py
```

浏览器打开：

```text
http://127.0.0.1:8000
```

页面模块：

- 数据管理
- 模型训练
- 实验结果
- 可视化
- 预测演示
- API 与存储说明

UI 后端使用 Python 标准库 `http.server`，不依赖 FastAPI。

---

## 13. 输出文件说明

训练输出通常位于：

```text
GraphMMI/runs/
GraphMMI/final_exp/*/result/
```

典型输出：

```text
config.json
transfer_metrics.csv
transfer_metrics.json
heatmaps/
*.log
```

其中：

- `config.json`：记录实验参数
- `transfer_metrics.csv`：核心评估结果
- `transfer_metrics.json`：JSON 格式评估结果
- `heatmaps/`：source-target transfer matrix 可视化
- `*.log`：训练日志

---

## 14. GitHub 同步建议

当前 `.gitignore` 已忽略以下大文件/目录：

```text
data/
**/data/
*.pt
*.pth
*.ckpt
runs/
**/runs/
.mplconfig/
__pycache__/
```

建议上传到 GitHub 的内容：

- 代码：`GraphMMI/src/`
- 脚本：`GraphMMI/scripts/`
- 实验入口：`GraphMMI/final_exp/**/*.sh`
- 文档：`README.md`、`GraphMMI/scripts/*.md`、`GraphMMI/ui/README.md`
- UI 源码：`GraphMMI/ui/*.html/css/js/py`
- baseline 参考代码：`TransferLearningMTI-baseline/`

不建议上传：

- 原始数据 CSV
- 处理后的 `graph_inputs.pt`
- 模型权重
- 大量训练日志
- 大量 `runs/` 结果
- Python 缓存

如果需要公开某一组小规模结果，可以只上传关键 CSV，例如：

```text
summary_metrics.csv
decoder_ablation.csv
aupr_summary.csv
```

但要确认文件不包含隐私数据或过大中间文件。

---

## 15. 常用命令汇总

### 15.1 安装依赖

```bash
cd GraphMMI
pip install -r requirements.txt
```

### 15.2 构建 target-site 图

```bash
cd GraphMMI
python -u scripts/final_embedding.py \
  --node-mode target_site \
  --sim-mode topk \
  --mirna-sim-topk 5 \
  --mrna-sim-topk 5 \
  --output-dir data/processed/graph/final_target_site
```

### 15.3 运行主实验

```bash
cd GraphMMI
bash final_exp/exp2/run.sh
```

### 15.4 运行消融实验

```bash
cd GraphMMI
bash final_exp/exp3/run_all.sh
```

### 15.5 运行 baseline

```bash
cd GraphMMI
bash final_exp/exp1/A_baseline/run.sh
```

### 15.6 启动 UI

```bash
cd GraphMMI
python ui/server.py
```

---

## 16. 论文实验复现顺序建议

推荐顺序：

1. 准备 `GraphMMI/data/external/` 原始数据
2. 安装依赖
3. 运行 baseline
4. 构建 target-site 图
5. 运行 Exp2 主 GNN 实验
6. 运行 Exp3 消融实验
7. 用 summary 脚本整理结果
8. 启动 UI 查看系统原型

对应命令：

```bash
cd GraphMMI

# baseline
bash final_exp/exp1/A_baseline/run.sh

# main GNN experiment
bash final_exp/exp2/run.sh

# ablation experiments
bash final_exp/exp3/run_all.sh

# UI prototype
python ui/server.py
```

---

## 17. 备注

本仓库是研究型代码库，包含多轮实验探索痕迹。最终论文相关主线建议优先参考：

```text
GraphMMI/scripts/final_embedding.py
GraphMMI/scripts/train_gnn_transfer.py
GraphMMI/scripts/baseline_ann_xgb_transfer.py
GraphMMI/final_exp/
GraphMMI/ui/
```

早期脚本如 `preprocess_graph_data.py`、`train_gnn_transfer_v2.py`、`embedding_optimed_v2.py` 等保留用于对比和溯源，但最终实验以 `final_embedding.py` 和 `train_gnn_transfer.py` 为准。
