const species = ["human", "cow", "mouse", "worm"];

const datasets = [
  {
    species: "human",
    samples: "9,097",
    mirna: "605",
    target: "9,036",
    edges: "85,947",
    status: "processed/graph/final_target_site/human",
  },
  {
    species: "cow",
    samples: "14,675",
    mirna: "165",
    target: "14,535",
    edges: "123,587",
    status: "processed/graph/final_target_site/cow",
  },
  {
    species: "mouse",
    samples: "18,111",
    mirna: "430",
    target: "18,019",
    edges: "166,199",
    status: "processed/graph/final_target_site/mouse",
  },
  {
    species: "worm",
    samples: "2,166",
    mirna: "122",
    target: "2,145",
    edges: "20,702",
    status: "processed/graph/final_target_site/worm",
  },
];

const resultTables = {
  transfer: {
    columns: ["encoder", "setting", "AUPRavg", "AUPRdiag", "AUPRcross", "AUCavg", "F1avg", "MCCavg"],
    rows: [
      ["graphsage", "source_only", "0.8439", "0.8499", "0.8419", "0.8330", "0.5378", "0.4045"],
      ["graphsage", "transfer", "0.8433", "0.8414", "0.8440", "0.8411", "0.7444", "0.5084"],
      ["gatv2", "source_only", "0.8195", "0.8344", "0.8145", "0.8124", "0.5892", "0.4059"],
      ["gatv2", "transfer", "0.8415", "0.8376", "0.8428", "0.8360", "0.7192", "0.5045"],
    ],
  },
  edges: {
    columns: ["实验设置", "miRNA相似边", "target site相似边", "AUPRavg", "AUPRcross", "AUCavg"],
    rows: [
      ["no-sim", "×", "×", "0.8051", "0.8034", "0.7905"],
      ["miRNA-only", "√", "×", "0.8162", "0.8145", "0.8053"],
      ["target-only", "×", "√", "0.8600", "0.8586", "0.8517"],
      ["both-sim", "√", "√", "0.8623", "0.8653", "0.8602"],
    ],
  },
  decoder: {
    columns: ["decoder 结构", "主要设计", "test AUC", "test AUPR", "最佳 epoch"],
    rows: [
      ["baseline", "concat + MLP", "0.8672", "0.8727", "9"],
      ["residual", "MLP 中加入残差连接", "0.8472", "0.8632", "28"],
      ["gated", "pair feature 门控融合", "0.8524", "0.8633", "7"],
      ["bilinear", "双线性匹配项", "0.8461", "0.8559", "9"],
      ["separated", "节点特征与 pair feature 分别编码后融合", "0.8336", "0.8491", "9"],
    ],
  },
};

const heatmapData = {
  aupr: [
    [0.8564, 0.8788, 0.8607, 0.8152],
    [0.8370, 0.8698, 0.8378, 0.7966],
    [0.8605, 0.8830, 0.8472, 0.8114],
    [0.8384, 0.8727, 0.8357, 0.7924],
  ],
  auc: [
    [0.8552, 0.8704, 0.8519, 0.8061],
    [0.8317, 0.8590, 0.8294, 0.7870],
    [0.8526, 0.8742, 0.8367, 0.7991],
    [0.8270, 0.8629, 0.8258, 0.7804],
  ],
};

const trendData = [
  {
    encoder: "gatv2",
    values: [0.8392, 0.8662, 0.8678, 0.8555, 0.8483, 0.8569],
    best: "L3",
  },
  {
    encoder: "graphsage",
    values: [0.8195, 0.8558, 0.8568, 0.8595, 0.8466, 0.8552],
    best: "L4",
  },
];

const predictions = [
  ["hsa-miR-1", "target_000123", 0.923, 1],
  ["hsa-miR-21", "target_000298", 0.887, 1],
  ["hsa-miR-155", "target_001042", 0.841, 1],
  ["hsa-miR-7", "target_000516", 0.332, 0],
];

function renderDatasets() {
  const rows = document.querySelector("#datasetRows");
  rows.innerHTML = datasets
    .map(
      (row) => `
        <tr>
          <td><strong>${row.species}</strong></td>
          <td>${row.samples}</td>
          <td>${row.mirna}</td>
          <td>${row.target}</td>
          <td>${row.edges}</td>
          <td><span class="tag">${row.status}</span></td>
        </tr>
      `,
    )
    .join("");
}

function renderResultTable(type = "transfer") {
  const table = resultTables[type];
  document.querySelector("#resultHead").innerHTML = `
    <tr>${table.columns.map((column) => `<th>${column}</th>`).join("")}</tr>
  `;
  document.querySelector("#resultRows").innerHTML = table.rows
    .map(
      (row) => `
        <tr>${row.map((cell, index) => `<td>${index === 0 ? `<strong>${cell}</strong>` : cell}</td>`).join("")}</tr>
      `,
    )
    .join("");
}

function heatColor(value, min, max) {
  const ratio = (value - min) / (max - min || 1);
  const lightness = 92 - ratio * 36;
  const saturation = 34 + ratio * 24;
  return `hsl(173  ${saturation}% ${lightness}%)`;
}

function renderHeatmap(metric = "aupr") {
  const values = heatmapData[metric];
  const flat = values.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const cells = [`<div class="heatmap-label"></div>`];

  species.forEach((target) => cells.push(`<div class="heatmap-label">${target}</div>`));
  values.forEach((row, rowIndex) => {
    cells.push(`<div class="heatmap-label">${species[rowIndex]}</div>`);
    row.forEach((value, colIndex) => {
      cells.push(`
        <div class="heatmap-cell" style="background:${heatColor(value, min, max)}">
          <strong>${value.toFixed(4)}</strong>
          <span>${species[rowIndex]} -> ${species[colIndex]}</span>
        </div>
      `);
    });
  });
  document.querySelector("#heatmap").innerHTML = cells.join("");
}

function renderTrendChart() {
  const min = 0.80;
  const max = 0.875;
  const rows = trendData
    .map((series) => {
      const bars = series.values
        .map((value, index) => {
          const height = Math.max(18, ((value - min) / (max - min)) * 150);
          return `<div class="bar ${series.encoder}" style="height:${height}px" title="${series.encoder} L${index + 1}: ${value.toFixed(4)}"><span>L${index + 1}</span></div>`;
        })
        .join("");
      return `
        <div class="trend-row">
          <strong>${series.encoder}</strong>
          <div class="trend-bars">${bars}</div>
          <span class="tag">${series.best}</span>
        </div>
      `;
    })
    .join("");
  document.querySelector("#trendChart").innerHTML = rows;
}

function renderPredictions() {
  document.querySelector("#predictionRows").innerHTML = predictions
    .sort((a, b) => b[2] - a[2])
    .map(
      ([mirna, target, prob, label]) => `
        <tr>
          <td>${mirna}</td>
          <td>${target}</td>
          <td><strong>${prob.toFixed(3)}</strong></td>
          <td><span class="tag">${label}</span></td>
        </tr>
      `,
    )
    .join("");
}

function buildCommandPreview(data) {
  const model = data.model.toLowerCase();
  if (model === "ann" || model === "xgboost") {
    return `python -u scripts/baseline_ann_xgb_transfer.py \\
  --models ${model === "xgboost" ? "xgb" : "ann"} \\
  --species ${data.source} ${data.target} \\
  --transfer-size 500`;
  }
  return `python -u scripts/train_gnn_transfer.py \\
  --species ${data.source} ${data.target} \\
  --encoders ${model === "gatv2" ? "gatv2" : "graphsage"} \\
  --settings ${data.setting === "transfer" ? "finetune" : "strict_zero_shot"} \\
  --num-layers ${data.layers} \\
  --processed-dir data/processed/graph/final_target_site \\
  --neg-strategy ${data.negative}`;
}

function setTaskState(status, title, message) {
  document.querySelector("#taskState").innerHTML = `
    <span class="status-pill">${status}</span>
    <strong>${title}</strong>
    <p>${message}</p>
  `;
}

async function createExperiment(data) {
  const response = await fetch("/api/experiments", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    let message = "local launcher is not available";
    try {
      const payload = await response.json();
      message = payload.error || message;
    } catch {
      message = "请用 python ui/server.py 启动页面，普通 http.server 只能静态预览。";
    }
    throw new Error(message);
  }
  return response.json();
}

function pollJobStatus(experimentId) {
  const timer = window.setInterval(async () => {
    try {
      const response = await fetch(`/api/experiments/${experimentId}/status`);
      if (!response.ok) return;
      const job = await response.json();
      setTaskState(
        job.status,
        job.experiment_id,
        `pid=${job.pid} · log=${job.log_path} · run_root=${job.run_root}`,
      );
      if (job.status === "completed" || job.status === "failed") {
        window.clearInterval(timer);
      }
    } catch {
      window.clearInterval(timer);
    }
  }, 3000);
}

function bindNavigation() {
  document.querySelectorAll("[data-scroll]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelector(button.dataset.scroll)?.scrollIntoView({ behavior: "smooth" });
    });
  });

  const sections = [...document.querySelectorAll("main .section")];
  const links = [...document.querySelectorAll(".nav-link")];
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        links.forEach((link) => {
          link.classList.toggle("active", link.getAttribute("href") === `#${entry.target.id}`);
        });
      });
    },
    { rootMargin: "-40% 0px -55% 0px" },
  );
  sections.forEach((section) => observer.observe(section));
}

function bindResultTabs() {
  document.querySelectorAll("[data-result-table]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-result-table]").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      renderResultTable(button.dataset.resultTable);
    });
  });
}

function bindHeatmapTabs() {
  document.querySelectorAll("[data-heatmap-metric]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-heatmap-metric]").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      renderHeatmap(button.dataset.heatmapMetric);
    });
  });
}

function bindForms() {
  const form = document.querySelector("#experimentForm");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = Object.fromEntries(new FormData(form));
    document.querySelector("#commandPreview code").textContent = buildCommandPreview(data);
    setTaskState("submitting", "正在提交任务", "正在连接本地 UI 后端 /api/experiments。");

    try {
      const job = await createExperiment(data);
      document.querySelector("#commandPreview code").textContent = job.command_text || buildCommandPreview(data);
      setTaskState(
        job.status,
        job.experiment_id,
        `pid=${job.pid} · log=${job.log_path} · run_root=${job.run_root}`,
      );
      pollJobStatus(job.experiment_id);
    } catch (error) {
      setTaskState(
        "preview-only",
        "静态预览模式，未启动训练",
        `${error.message} 页面仍会生成命令，但浏览器不能直接执行本机 Python 脚本。`,
      );
    }
  });

  const predictForm = document.querySelector("#predictForm");
  predictForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const data = Object.fromEntries(new FormData(predictForm));
    const probability = Math.min(0.981, Math.max(0.104, (data.sequence.length % 17) / 20 + 0.42));
    const label = probability >= 0.5 ? 1 : 0;
    document.querySelector("#predictionSummary").innerHTML = `
      <span>${probability.toFixed(3)}</span>
      <strong>${label ? "高置信相互作用" : "低置信候选关系"}</strong>
      <small>${data.modelId} · ${data.species} · prediction_label = ${label}</small>
    `;
    predictions.unshift([data.mirna, data.target, probability, label]);
    renderPredictions();
  });
}

renderDatasets();
renderResultTable();
renderHeatmap();
renderTrendChart();
renderPredictions();
bindNavigation();
bindResultTabs();
bindHeatmapTabs();
bindForms();
