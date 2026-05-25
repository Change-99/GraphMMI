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

let runResultFiles = [];
let heatmapFiles = [];
let currentHeatmapMetric = "aupr";
let predictionOptions = { mirnas: [], targets: [] };
let datasetSummary = null;

function normalizePrediction(row) {
  if (Array.isArray(row)) {
    return {
      mirna_name: row[0],
      target_site_id: row[1],
      interaction_probability: row[2],
      prediction_label: row[3],
    };
  }
  return {
    mirna_name: row.mirna_name,
    target_site_id: row.target_site_id,
    interaction_probability: Number(row.interaction_probability),
    prediction_label: Number(row.prediction_label),
  };
}

function escapeAttribute(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString("en-US") : String(value);
}

function datasetFallbackPayload() {
  const rows = datasets.map((row) => ({
    species: row.species,
    samples: Number(row.samples.replaceAll(",", "")),
    nodes: Number(row.mirna.replaceAll(",", "")) + Number(row.target.replaceAll(",", "")),
    mirna: Number(row.mirna.replaceAll(",", "")),
    targets: Number(row.target.replaceAll(",", "")),
    positive_edges: Number(row.edges.replaceAll(",", "")),
    mirna_sim_edges: 0,
    target_sim_edges: 0,
    node_features: 98,
    edge_attr: 0,
    pair_feature_dim: 40,
    node_mode: "target_site",
    sim_mode: "topk",
    split: { train: 0, val: 0, test: 0 },
    diagnostic: {},
    files: [{ name: "graph path", path: row.status, exists: true }],
    status: "preview",
    root: row.status,
  }));
  const totals = rows.reduce(
    (acc, row) => {
      acc.samples += row.samples;
      acc.nodes += row.nodes;
      acc.mirna += row.mirna;
      acc.targets += row.targets;
      acc.positive_edges += row.positive_edges;
      acc.mirna_sim_edges += row.mirna_sim_edges;
      acc.target_sim_edges += row.target_sim_edges;
      return acc;
    },
    { samples: 0, nodes: 0, mirna: 0, targets: 0, positive_edges: 0, mirna_sim_edges: 0, target_sim_edges: 0 },
  );
  return {
    rows,
    totals,
    cards: [
      { label: "物种数", value: rows.length, hint: "human / cow / mouse / worm" },
      { label: "正样本总数", value: formatNumber(totals.samples), hint: "clean positive edges" },
      { label: "图节点总数", value: formatNumber(totals.nodes), hint: "miRNA + target site" },
      { label: "相似边总数", value: "待连接", hint: "启动本地后端后读取" },
    ],
    source: "内置演示数据",
  };
}

async function loadOverviewSummary() {
  const response = await fetch("/api/overview/summary");
  if (!response.ok) {
    throw new Error("overview API unavailable");
  }
  return response.json();
}

async function renderOverview() {
  try {
    const overview = await loadOverviewSummary();
    document.querySelector("#overviewCards").innerHTML = (overview.cards || [])
      .map(
        (card) => `
          <article class="metric-card">
            <span>${card.label}</span>
            <strong>${card.value}</strong>
            <small>${card.hint}</small>
          </article>
        `,
      )
      .join("");
    document.querySelector("#overviewModules").innerHTML = (overview.modules || [])
      .map(
        (module) => `
          <button class="module-item" type="button" data-scroll="#${module.page}">
            <span>${module.status}</span>
            <strong>${module.name}</strong>
            <small>${module.detail}</small>
          </button>
        `,
      )
      .join("");
    document.querySelector("#overviewBest").innerHTML = (overview.best || [])
      .map(
        (item) => `
          <div class="best-item">
            <span>${item.label}</span>
            <strong>${item.value}</strong>
            <small>${item.detail}</small>
          </div>
        `,
      )
      .join("");
    document.querySelector("#overviewRecent").innerHTML = (overview.recent || [])
      .map(
        (item) => `
          <div class="artifact-item">
            <strong>${item.label}</strong>
            <span>${item.mtime}</span>
            <small>${item.path}</small>
          </div>
        `,
      )
      .join("");
    document.querySelector("#overviewSource").textContent = `数据来源：${overview.source}`;
    bindDynamicScrollButtons(document.querySelector("#overviewModules"));
  } catch {
    document.querySelector("#overviewModules").innerHTML = `
      <div class="module-item static"><span>preview</span><strong>本地后端未连接</strong><small>启动 python ui/server.py 后显示真实状态</small></div>
    `;
    document.querySelector("#overviewBest").innerHTML = `
      <div class="best-item"><span>最高 AUPR</span><strong>0.8727</strong><small>decoder baseline, human</small></div>
      <div class="best-item"><span>最佳跨物种 AUPR</span><strong>0.8653</strong><small>both-sim edge ablation</small></div>
    `;
    document.querySelector("#overviewRecent").innerHTML = `
      <div class="artifact-item"><strong>静态预览模式</strong><span>未连接 API</span><small>启动本地 UI 服务后读取 runs 和 final_exp</small></div>
    `;
  }
}

async function loadDatasetSummary() {
  const response = await fetch("/api/datasets/summary");
  if (!response.ok) {
    throw new Error("dataset API unavailable");
  }
  return response.json();
}

function renderDatasetCards(cards) {
  document.querySelector("#datasetCards").innerHTML = cards
    .map(
      (card) => `
        <article class="dataset-card">
          <span>${card.label}</span>
          <strong>${card.value}</strong>
          <small>${card.hint}</small>
        </article>
      `,
    )
    .join("");
}

function selectedDatasetRows() {
  const selected = document.querySelector("#datasetSpeciesSelect")?.value || "all";
  const rows = datasetSummary?.rows || [];
  return selected === "all" ? rows : rows.filter((row) => row.species === selected);
}

async function renderDatasets() {
  try {
    datasetSummary = await loadDatasetSummary();
  } catch {
    datasetSummary = datasetFallbackPayload();
  }
  renderDatasetCards(datasetSummary.cards || []);
  const rows = document.querySelector("#datasetRows");
  rows.innerHTML = selectedDatasetRows()
    .map(
      (row) => `
        <tr>
          <td><strong>${row.species}</strong></td>
          <td>${formatNumber(row.samples)}</td>
          <td>${formatNumber(row.nodes)}</td>
          <td>${formatNumber(row.mirna)}</td>
          <td>${formatNumber(row.targets)}</td>
          <td>${formatNumber(row.mirna_sim_edges)}</td>
          <td>${formatNumber(row.target_sim_edges)}</td>
          <td>${row.node_features} / ${row.edge_attr}</td>
          <td><span class="tag">${row.status}</span></td>
        </tr>
      `,
    )
    .join("");
  renderDatasetDetail(selectedDatasetRows()[0] || datasetSummary.rows[0]);
  document.querySelector("#datasetSource").textContent = `数据来源：${datasetSummary.source}`;
}

function renderDatasetDetail(row) {
  if (!row) return;
  document.querySelector("#datasetDetailTitle").textContent = `${row.species} 数据详情`;
  const diag = row.diagnostic || {};
  const detailItems = [
    ["图构建方式", row.node_mode],
    ["相似边模式", `${row.sim_mode}, topK=${row.mirna_sim_topk || 0}/${row.target_sim_topk || 0}`],
    ["pair feature 维度", row.pair_feature_dim],
    ["原始样本行", formatNumber(row.raw_rows || row.samples)],
    ["重复 pair 去除", formatNumber(row.duplicate_removed || 0)],
    ["节点序列冲突去除", formatNumber(row.node_conflicts_removed || 0)],
    ["train/test pair 重叠", formatNumber(diag.pair_overlap_train_test || 0)],
    ["mRNA 层 train/test 重叠", `${formatNumber(diag.mrna_level_train_test_overlap || 0)} (${diag.mrna_level_overlap_ratio || 0})`],
  ];
  document.querySelector("#datasetDetailList").innerHTML = detailItems
    .map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
  renderSplitBars(row.split || {});
  document.querySelector("#datasetFileList").innerHTML = (row.files || [])
    .map(
      (file) => `
        <div class="file-item">
          <span class="${file.exists ? "file-ok" : "file-missing"}">${file.exists ? "ready" : "missing"}</span>
          <strong>${file.name}</strong>
          <small>${file.path}</small>
        </div>
      `,
    )
    .join("");
}

function renderSplitBars(split) {
  const total = (split.train || 0) + (split.val || 0) + (split.test || 0);
  const parts = [
    ["train", split.train || 0],
    ["val", split.val || 0],
    ["test", split.test || 0],
  ];
  document.querySelector("#datasetSplitBars").innerHTML = parts
    .map(([name, value]) => {
      const percent = total ? (value / total) * 100 : 0;
      return `
        <div class="split-row">
          <span>${name}</span>
          <div class="split-track"><i style="width:${percent}%"></i></div>
          <strong>${formatNumber(value)}</strong>
        </div>
      `;
    })
    .join("");
}

async function loadResultTable(type) {
  const response = await fetch(`/api/results/${type}`);
  if (!response.ok) {
    throw new Error("result API unavailable");
  }
  return response.json();
}

async function loadRunResultFiles() {
  const response = await fetch("/api/results/runs");
  if (!response.ok) {
    throw new Error("runs API unavailable");
  }
  const payload = await response.json();
  runResultFiles = payload.files || [];
  return runResultFiles;
}

function renderRunResultOptions(files) {
  const select = document.querySelector("#runResultSelect");
  select.innerHTML = files
    .map((file) => `<option value="${file.id}">${file.label}</option>`)
    .join("");
}

function setSelectOptions(select, values, selectedValue = "") {
  const current = selectedValue || select.value;
  select.innerHTML = values
    .map((value) => `<option value="${value}">${value}</option>`)
    .join("");
  if (values.includes(current)) {
    select.value = current;
  } else if (values.length) {
    select.value = values[0];
  }
}

async function loadSelectedRunResult() {
  const select = document.querySelector("#runResultSelect");
  if (!runResultFiles.length) {
    const files = await loadRunResultFiles();
    renderRunResultOptions(files);
  }
  if (!select.value && runResultFiles[0]) {
    select.value = runResultFiles[0].id;
  }
  if (!select.value) {
    return {
      columns: ["提示"],
      rows: [["未在 GraphMMI/runs 下找到 transfer_metrics.csv、decoder_ablation.csv 或 metrics_long.csv。"]],
      source: "GraphMMI/runs",
    };
  }
  const response = await fetch(`/api/results/runs?file=${encodeURIComponent(select.value)}`);
  if (!response.ok) {
    throw new Error("selected run result unavailable");
  }
  return response.json();
}

async function loadHeatmapFiles() {
  const response = await fetch("/api/visualization/heatmaps");
  if (!response.ok) {
    throw new Error("heatmap API unavailable");
  }
  const payload = await response.json();
  heatmapFiles = payload.files || [];
  return heatmapFiles;
}

function renderHeatmapFileOptions(files) {
  const select = document.querySelector("#heatmapFileSelect");
  const previous = select.value;
  select.innerHTML = files
    .map((file) => `<option value="${file.id}">${file.label}</option>`)
    .join("");
  if (files.some((file) => file.id === previous)) {
    select.value = previous;
  } else if (files.length) {
    select.value = files[0].id;
  }
}

async function loadSelectedHeatmap(metric = currentHeatmapMetric) {
  const fileSelect = document.querySelector("#heatmapFileSelect");
  const encoderSelect = document.querySelector("#heatmapEncoderSelect");
  const settingSelect = document.querySelector("#heatmapSettingSelect");
  if (!heatmapFiles.length) {
    const files = await loadHeatmapFiles();
    renderHeatmapFileOptions(files);
  }
  if (!fileSelect.value && heatmapFiles[0]) {
    fileSelect.value = heatmapFiles[0].id;
  }
  if (!fileSelect.value) {
    throw new Error("no heatmap files found");
  }
  const selectedFile = heatmapFiles.find((file) => file.id === fileSelect.value);
  if (selectedFile) {
    setSelectOptions(encoderSelect, selectedFile.encoders || [], encoderSelect.value);
    setSelectOptions(settingSelect, selectedFile.settings || [], settingSelect.value);
  }
  const params = new URLSearchParams({
    file: fileSelect.value,
    metric,
    encoder: encoderSelect.value,
    setting: settingSelect.value,
  });
  const response = await fetch(`/api/visualization/heatmaps?${params.toString()}`);
  if (!response.ok) {
    throw new Error("selected heatmap unavailable");
  }
  const payload = await response.json();
  setSelectOptions(encoderSelect, payload.encoders || [], payload.encoder);
  setSelectOptions(settingSelect, payload.settings || [], payload.setting);
  return payload;
}

async function renderResultTable(type = "transfer") {
  let table = resultTables[type];
  let source = "内置演示数据";
  const runControls = document.querySelector("#runResultControls");
  runControls.hidden = type !== "runs";
  try {
    const apiTable = type === "runs" ? await loadSelectedRunResult() : await loadResultTable(type);
    if (apiTable.columns && apiTable.rows) {
      table = apiTable;
      source = apiTable.source || "后端 API";
      if (apiTable.truncated) {
        source += `；仅显示前 ${apiTable.rows.length} / ${apiTable.total_rows} 行`;
      }
    }
  } catch {
    table = type === "runs"
      ? { columns: ["提示"], rows: [["启动 python ui/server.py 后可选择 GraphMMI/runs 下的真实结果 CSV。"]] }
      : resultTables[type];
    source = "内置演示数据；启动 python ui/server.py 后会读取真实 CSV";
  }
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
  document.querySelector("#resultSource").textContent = `数据来源：${source}`;
}

function heatColor(value, min, max) {
  const ratio = (value - min) / (max - min || 1);
  const lightness = 92 - ratio * 36;
  const saturation = 34 + ratio * 24;
  return `hsl(173  ${saturation}% ${lightness}%)`;
}

async function renderHeatmap(metric = "aupr") {
  currentHeatmapMetric = metric;
  let values = heatmapData[metric];
  let rowSpecies = species;
  let title = `GraphSAGE transfer 4×4 矩阵`;
  let source = "内置演示数据";
  try {
    const payload = await loadSelectedHeatmap(metric);
    values = payload.matrix || values;
    rowSpecies = payload.species || rowSpecies;
    title = `${payload.encoder || "selected"} ${payload.setting || ""} ${rowSpecies.length}×${rowSpecies.length} ${metric.toUpperCase()} 矩阵`;
    source = payload.source || "后端 API";
  } catch {
    source = "内置演示数据；启动 python ui/server.py 后可选择真实 transfer_metrics.csv";
  }
  const numericValues = values.flat().filter((value) => typeof value === "number" && Number.isFinite(value));
  const flat = numericValues.length ? numericValues : [0];
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const cells = [`<div class="heatmap-label"></div>`];

  rowSpecies.forEach((target) => cells.push(`<div class="heatmap-label">${target}</div>`));
  values.forEach((row, rowIndex) => {
    cells.push(`<div class="heatmap-label">${rowSpecies[rowIndex]}</div>`);
    row.forEach((value, colIndex) => {
      const hasValue = typeof value === "number" && Number.isFinite(value);
      cells.push(`
        <div class="heatmap-cell" style="background:${hasValue ? heatColor(value, min, max) : "hsl(270 18% 90%)"}">
          <strong>${hasValue ? value.toFixed(4) : "NA"}</strong>
          <span>${rowSpecies[rowIndex]} -> ${rowSpecies[colIndex]}</span>
        </div>
      `);
    });
  });
  document.querySelector("#heatmap").innerHTML = cells.join("");
  document.querySelector("#heatmapTitle").textContent = title;
  document.querySelector("#heatmapSource").textContent = `数据来源：${source}`;
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
    .map(normalizePrediction)
    .sort((a, b) => b.interaction_probability - a.interaction_probability)
    .map(
      (row) => `
        <tr>
          <td>${row.mirna_name}</td>
          <td>${row.target_site_id}</td>
          <td><strong>${row.interaction_probability.toFixed(3)}</strong></td>
          <td><span class="tag">${row.prediction_label}</span></td>
        </tr>
      `,
    )
    .join("");
}

async function loadPredictModels() {
  const response = await fetch("/api/predict/models");
  if (!response.ok) {
    throw new Error("predict model API unavailable");
  }
  return response.json();
}

function renderPredictModels(models) {
  const select = document.querySelector("#predictModelSelect");
  const previous = select.value;
  select.innerHTML = models
    .map((model) => `<option value="${model.id}">${model.label}</option>`)
    .join("");
  if (models.some((model) => model.id === previous)) {
    select.value = previous;
  }
}

async function initializePredictionApi() {
  try {
    const payload = await loadPredictModels();
    renderPredictModels(payload.models || []);
    await loadPredictionOptions(document.querySelector("#predictSpeciesSelect").value);
    const history = await fetch("/api/predict/history?limit=20");
    if (history.ok) {
      const table = await history.json();
      if (table.rows && table.rows.length) {
        predictions.splice(0, predictions.length, ...table.rows);
        renderPredictions();
        document.querySelector("#predictionSource").textContent = `数据来源：${table.source}`;
      }
    }
  } catch {
    document.querySelector("#predictionSource").textContent = "数据来源：前端演示数据；启动 python ui/server.py 后可调用预测 API";
  }
}

async function loadPredictionOptions(speciesName) {
  const response = await fetch(`/api/predict/options?species=${encodeURIComponent(speciesName)}`);
  if (!response.ok) {
    throw new Error("prediction option API unavailable");
  }
  const payload = await response.json();
  predictionOptions = {
    mirnas: payload.mirnas || [],
    targets: payload.targets || [],
  };
  renderPredictionOptions();
  document.querySelector("#predictionSource").textContent = `候选项来源：${payload.source}`;
}

function renderPredictionOptions() {
  const mirnaList = document.querySelector("#mirnaOptions");
  const targetList = document.querySelector("#targetOptions");
  mirnaList.innerHTML = predictionOptions.mirnas
    .map((item) => `<option value="${escapeAttribute(item.id)}" label="${escapeAttribute(item.name)}"></option>`)
    .join("");
  targetList.innerHTML = predictionOptions.targets
    .map((item) => `<option value="${escapeAttribute(item.id)}" label="${escapeAttribute(item.name)}"></option>`)
    .join("");
  const mirnaInput = document.querySelector("#mirnaInput");
  const targetInput = document.querySelector("#targetInput");
  if (!predictionOptions.mirnas.some((item) => item.id === mirnaInput.value || item.name === mirnaInput.value) && predictionOptions.mirnas[0]) {
    mirnaInput.value = predictionOptions.mirnas[0].id;
  }
  if (!predictionOptions.targets.some((item) => item.id === targetInput.value || item.name === targetInput.value) && predictionOptions.targets[0]) {
    targetInput.value = predictionOptions.targets[0].id;
  }
  fillSelectedTargetSequence();
}

function fillSelectedTargetSequence() {
  const targetValue = document.querySelector("#targetInput").value;
  const selected = predictionOptions.targets.find((item) => item.id === targetValue || item.name === targetValue);
  if (selected && selected.sequence) {
    document.querySelector("#targetSequenceInput").value = selected.sequence;
  }
}

async function postPrediction(endpoint, payload) {
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let message = "prediction API unavailable";
    try {
      const body = await response.json();
      message = body.error || message;
    } catch {
      message = "请使用 python ui/server.py 启动页面。";
    }
    throw new Error(message);
  }
  return response.json();
}

function setPredictionSummary(result) {
  const probability = Number(result.interaction_probability);
  const label = Number(result.prediction_label);
  document.querySelector("#predictionSummary").innerHTML = `
    <span>${probability.toFixed(3)}</span>
    <strong>${label ? "高置信相互作用" : "低置信候选关系"}</strong>
    <small>${result.model_id} · ${result.species} · prediction_label = ${label} · ${result.mode}</small>
  `;
  document.querySelector("#predictionSource").textContent = `数据来源：${result.result_file || "预测 API"}`;
}

function parseBatchPairs(text) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const [mirnaName, targetSiteId, ...sequenceParts] = line.split(",").map((item) => item.trim());
      return {
        mirna_name: mirnaName,
        target_site_id: targetSiteId,
        target_sequence: sequenceParts.join(","),
      };
    })
    .filter((row) => row.mirna_name && row.target_site_id && row.target_sequence);
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

function setLogOutput(text, hint = "") {
  const output = document.querySelector("#logOutput code");
  const panel = document.querySelector("#logOutput");
  const label = document.querySelector("#logHint");
  output.textContent = text || "暂无日志输出。";
  if (hint) label.textContent = hint;
  panel.scrollTop = panel.scrollHeight;
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

async function fetchJobLog(experimentId) {
  const response = await fetch(`/api/experiments/${experimentId}/log`);
  if (!response.ok) return "";
  const payload = await response.json();
  return payload.log || "";
}

function pollJobStatus(experimentId) {
  const timer = window.setInterval(async () => {
    try {
      const response = await fetch(`/api/experiments/${experimentId}/status`);
      if (!response.ok) return;
      const job = await response.json();
      const log = await fetchJobLog(experimentId);
      setTaskState(
        job.status,
        job.experiment_id,
        `pid=${job.pid} · log=${job.log_path} · run_root=${job.run_root}`,
      );
      setLogOutput(log, `状态：${job.status}`);
      if (job.status === "completed" || job.status === "failed") {
        window.clearInterval(timer);
      }
    } catch {
      window.clearInterval(timer);
    }
  }, 3000);
}

function showPage(pageId, pushState = true) {
  const pages = [...document.querySelectorAll(".page-section")];
  const hasPage = pages.some((page) => page.dataset.page === pageId);
  const nextPage = hasPage ? pageId : "overview";

  pages.forEach((page) => {
    page.hidden = page.dataset.page !== nextPage;
  });

  document.querySelectorAll("[data-page-link]").forEach((link) => {
    link.classList.toggle("active", link.dataset.pageLink === nextPage);
  });

  if (pushState && window.location.hash !== `#${nextPage}`) {
    window.history.pushState(null, "", `#${nextPage}`);
  }
  document.querySelector("#main")?.scrollIntoView({ block: "start" });
}

function bindNavigation() {
  bindDynamicScrollButtons(document);
  window.addEventListener("popstate", () => {
    showPage(window.location.hash.replace("#", "") || "overview", false);
  });

  showPage(window.location.hash.replace("#", "") || "overview", false);
}

function bindDynamicScrollButtons(root) {
  document.querySelectorAll("[data-scroll]").forEach((button) => {
    if (root !== document && !root.contains(button)) return;
    if (button.dataset.boundScroll === "true") return;
    button.dataset.boundScroll = "true";
    button.addEventListener("click", (event) => {
      event.preventDefault();
      showPage(button.dataset.scroll.replace("#", ""));
    });
  });

  document.querySelectorAll("[data-page-link]").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      showPage(link.dataset.pageLink);
    });
  });
}

function bindResultTabs() {
  document.querySelectorAll("[data-result-table]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-result-table]").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      renderResultTable(button.dataset.resultTable);
    });
  });
  document.querySelector("#runResultSelect").addEventListener("change", () => {
    renderResultTable("runs");
  });
  document.querySelector("#refreshRunResults").addEventListener("click", async () => {
    runResultFiles = [];
    await renderResultTable("runs");
  });
}

function rerenderDatasetRows() {
  if (!datasetSummary) return;
  const rows = document.querySelector("#datasetRows");
  rows.innerHTML = selectedDatasetRows()
    .map(
      (row) => `
        <tr>
          <td><strong>${row.species}</strong></td>
          <td>${formatNumber(row.samples)}</td>
          <td>${formatNumber(row.nodes)}</td>
          <td>${formatNumber(row.mirna)}</td>
          <td>${formatNumber(row.targets)}</td>
          <td>${formatNumber(row.mirna_sim_edges)}</td>
          <td>${formatNumber(row.target_sim_edges)}</td>
          <td>${row.node_features} / ${row.edge_attr}</td>
          <td><span class="tag">${row.status}</span></td>
        </tr>
      `,
    )
    .join("");
  renderDatasetDetail(selectedDatasetRows()[0] || datasetSummary.rows[0]);
}

function bindDatasetControls() {
  document.querySelector("#datasetSpeciesSelect").addEventListener("change", rerenderDatasetRows);
  document.querySelector("#refreshDatasets").addEventListener("click", async () => {
    await renderDatasets();
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
  document.querySelector("#heatmapFileSelect").addEventListener("change", () => {
    renderHeatmap(currentHeatmapMetric);
  });
  document.querySelector("#heatmapEncoderSelect").addEventListener("change", () => {
    renderHeatmap(currentHeatmapMetric);
  });
  document.querySelector("#heatmapSettingSelect").addEventListener("change", () => {
    renderHeatmap(currentHeatmapMetric);
  });
  document.querySelector("#refreshHeatmaps").addEventListener("click", async () => {
    heatmapFiles = [];
    await renderHeatmap(currentHeatmapMetric);
  });
}

function bindForms() {
  const form = document.querySelector("#experimentForm");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = Object.fromEntries(new FormData(form));
    document.querySelector("#commandPreview code").textContent = buildCommandPreview(data);
    setTaskState("submitting", "正在提交任务", "正在连接本地 UI 后端 /api/experiments。");
    setLogOutput("正在提交任务，等待后端创建日志文件...", "提交中");

    try {
      const job = await createExperiment(data);
      document.querySelector("#commandPreview code").textContent = job.command_text || buildCommandPreview(data);
      setTaskState(
        job.status,
        job.experiment_id,
        `pid=${job.pid} · log=${job.log_path} · run_root=${job.run_root}`,
      );
      setLogOutput(`任务已启动。\n日志文件：${job.log_path}\n等待训练脚本输出...`, "运行中");
      pollJobStatus(job.experiment_id);
    } catch (error) {
      setTaskState(
        "preview-only",
        "静态预览模式，未启动训练",
        `${error.message} 页面仍会生成命令，但浏览器不能直接执行本机 Python 脚本。`,
      );
      setLogOutput("没有本地后端日志。请使用 python ui/server.py 启动页面。", "未连接后端");
    }
  });

  const predictForm = document.querySelector("#predictForm");
  predictForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = Object.fromEntries(new FormData(predictForm));
    try {
      const result = await postPrediction("/api/predict/single", {
        model_id: data.modelId,
        species: data.species,
        mirna_name: data.mirna,
        target_site_id: data.target,
        target_sequence: data.sequence,
      });
      setPredictionSummary(result);
      predictions.unshift(result);
      renderPredictions();
    } catch (error) {
      const probability = Math.min(0.981, Math.max(0.104, (data.sequence.length % 17) / 20 + 0.42));
      const label = probability >= 0.5 ? 1 : 0;
      const result = {
        model_id: data.modelId,
        species: data.species,
        mirna_name: data.mirna,
        target_site_id: data.target,
        interaction_probability: probability,
        prediction_label: label,
        mode: "frontend_fallback",
      };
      setPredictionSummary({ ...result, result_file: error.message });
      predictions.unshift(result);
      renderPredictions();
    }
  });

  document.querySelector("#batchPredictButton").addEventListener("click", async () => {
    const data = Object.fromEntries(new FormData(predictForm));
    const items = parseBatchPairs(data.batchPairs || "");
    if (!items.length) {
      document.querySelector("#predictionSource").textContent = "批量输入格式：miRNA,target_site_id,target_sequence，每行一个候选 pair。";
      return;
    }
    try {
      const payload = await postPrediction("/api/predict/batch", {
        model_id: data.modelId,
        species: data.species,
        items,
      });
      predictions.splice(0, predictions.length, ...payload.results);
      setPredictionSummary(payload.results[0]);
      document.querySelector("#predictionSource").textContent = `数据来源：${payload.result_file}；batch=${payload.job_id}；count=${payload.count}`;
      renderPredictions();
    } catch (error) {
      document.querySelector("#predictionSource").textContent = `${error.message}；请使用 python ui/server.py 启动页面。`;
    }
  });

  document.querySelector("#predictSpeciesSelect").addEventListener("change", async (event) => {
    try {
      await loadPredictionOptions(event.target.value);
    } catch (error) {
      document.querySelector("#predictionSource").textContent = `${error.message}；无法读取该物种候选节点。`;
    }
  });

  document.querySelector("#targetInput").addEventListener("change", fillSelectedTargetSequence);
  document.querySelector("#targetInput").addEventListener("input", fillSelectedTargetSequence);
}

renderOverview();
renderDatasets();
renderResultTable();
renderHeatmap();
renderTrendChart();
renderPredictions();
initializePredictionApi();
bindNavigation();
bindDatasetControls();
bindResultTabs();
bindHeatmapTabs();
bindForms();
