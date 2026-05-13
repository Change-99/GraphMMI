# 基础 ANN/XGBoost 模型复现

本文件夹将表格数据（Tabular）基准测试与未来的 GraphMMI GNN 流水线分开存放。

### 主脚本

```bash
python scripts/baseline_ann_xgb_transfer.py --models ann xgb --transfer-size 500

```

### 仅数据检查

在安装 TensorFlow 或 XGBoost 之前，可以使用此命令确认数据状态：

```bash
python scripts/baseline_ann_xgb_transfer.py --check-data

```

---

### 输出路径

输出结果将写入：

```text
runs/baseline_transfer/<时间戳>/
  metrics_long.csv
  run_summary.json
  matrices/
  heatmaps/
  models/

```

---

### 复现逻辑

该脚本将基准实验作为一个独立的表格任务进行复现，具体流程如下：

* **数据预处理**：处理 `data/external/` 路径下的 `*_pos.csv` 和 `*_neg.csv` 文件。
* **数据集整合**：将 8 个原始数据集整合为 4 个物种：**人类 (human)、奶牛 (cow)、小鼠 (mouse)、线虫 (worm)**。
* **模型训练**：训练基于源物种（Source-species）的 ANN 和 XGBoost 模型。
* **跨物种评估**：评估仅使用源物种训练的模型在跨物种预测上的表现。
* **迁移学习**：在目标物种（Target-species）的小规模子集上进行**微调（Fine-tuning）**。
* **结果可视化**：生成 4x4 的源-目标矩阵及相应的热图。

**注意**：该脚本不会导入 `TransferLearningMTI`，以确保基准实验与后续的 GraphSAGE/GNN 实现保持完全独立。