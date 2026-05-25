# GraphMMI UI Prototype

This folder contains a static prototype for thesis section 4.6, "系统设计与实现".

For a read-only preview, open `index.html` directly in a browser.

To make "启动训练任务" launch a real local training process, start the local
launcher from the GraphMMI repo root:

```bash
cd /mnt/d/Projects/GProject/GraphMMI
python ui/server.py
```

Then open:

```text
http://127.0.0.1:8000
```

The local launcher writes UI job logs to `ui/jobs/` and training outputs to
`runs/ui_experiments/`.

The page covers:

- 数据管理: species-level graph statistics from `data/processed/graph/final_target_site/*/metadata.json`.
- 模型训练: visual experiment configuration, generated command preview, and optional local job launch through `ui/server.py`.
- 实验结果: transfer summary, similarity-edge ablation, and decoder ablation tables.
- 可视化: GraphSAGE transfer heatmap and encoder-depth trend chart.
- 预测演示: single-pair mock prediction and Top-K result table.
- API 与存储: REST API groups and file-based result flow.

The implementation is dependency-free: plain HTML, CSS, JavaScript, and a
standard-library Python server.
