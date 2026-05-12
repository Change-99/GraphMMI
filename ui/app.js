const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));
const svgNS = "http://www.w3.org/2000/svg";
const API_BASE = "/api";
const SPLIT = "cold_mirna";

const palette = {
  primary: "#6750A4",
  teal: "#006A60",
  amber: "#735C00",
  rose: "#7D5260",
};

const state = {
  species: ["human", "cow", "mouse", "worm"],
  activeSpecies: "human",
  runId: null,
  dashboard: null,
  transfer: null,
};

function createSvg(tag, attrs = {}, text = "") {
  const node = document.createElementNS(svgNS, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  if (text) node.textContent = text;
  return node;
}

function fmt(value, digits = 3) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "NA";
}

function formatNumber(value) {
  return Number(value || 0).toLocaleString("en-US");
}

function shortLabel(text) {
  const value = String(text || "");
  const parts = value.split("|");
  const label = parts[parts.length - 1] || value;
  return label.length > 18 ? `${label.slice(0, 16)}...` : label;
}

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status}: ${text}`);
  }
  return response.json();
}

function imageUrl(path) {
  return `${API_BASE}${path}`;
}

function loading(selector, text = "Loading real GNN data...") {
  const el = $(selector);
  if (el) el.innerHTML = `<div class="loading-note">${text}</div>`;
}

function showError(error) {
  $("#dashboard-stats").innerHTML = `<article class="stat-card rose"><span>API Error</span><strong>Failed</strong><small>${error.message}</small></article>`;
}

async function loadInitialData() {
  const runs = await api("/runs");
  state.runId = runs.default_transfer_run_id;
  const [dashboard, transfer] = await Promise.all([
    api("/dashboard/summary"),
    api(`/results/transfer?run_id=${encodeURIComponent(state.runId)}`),
  ]);
  state.dashboard = dashboard;
  state.transfer = transfer;
  state.species = dashboard.species || state.species;
}

function renderDashboard() {
  const totals = state.dashboard.totals;
  const best = state.dashboard.best_model;
  const stats = [
    { label: "miRNA 节点数", value: totals.mirna_nodes, note: state.species.join(" / "), tone: "" },
    { label: "mRNA 节点数", value: totals.mrna_nodes, note: "target genes", tone: "teal" },
    { label: "总节点数", value: totals.nodes, note: "bipartite graph nodes", tone: "amber" },
    { label: "真实正边数", value: totals.positive_edges, note: "validated positive edges", tone: "rose" },
    { label: "节点特征维度", value: totals.node_feature_dim, note: "sequence k-mer", tone: "teal" },
    { label: "当前最好模型", value: "GraphSAGE", note: `AUC ${fmt(best.mean_auc)} / AP ${fmt(best.mean_ap)}`, tone: "" },
  ];
  $("#dashboard-stats").innerHTML = stats
    .map(
      (stat) => `
        <article class="stat-card ${stat.tone}">
          <span>${stat.label}</span>
          <strong>${formatNumber(stat.value)}</strong>
          <small>${stat.note}</small>
        </article>
      `,
    )
    .join("");

  const colors = { AUC: palette.primary, AP: palette.teal, F1: palette.rose, ACC: palette.amber };
  $("#overview-bars").innerHTML = state.dashboard.overview_metrics
    .map(
      (item) => `
        <div class="bar-row">
          <span class="bar-label">${item.label}</span>
          <span class="bar-track"><i class="bar-fill" style="--value: ${item.value * 100}%; --bar-color: ${colors[item.label] || palette.primary};"></i></span>
          <span class="bar-value">${fmt(item.value)}</span>
        </div>
      `,
    )
    .join("");

  $("#species-donut").innerHTML = `
    <div class="donut">
      <div class="donut-center">
        <strong>${(totals.positive_edges / 1000).toFixed(1)}k</strong>
        <span>validated edges</span>
      </div>
    </div>
  `;
}

async function renderFeatureList() {
  const summary = await api(`/species/${state.activeSpecies}/summary?split=${SPLIT}`);
  const groups = [
    { name: "1-mer sequence composition", count: 4 },
    { name: "2-mer sequence composition", count: 16 },
    { name: "Sequence length", count: 1 },
    { name: "GC content", count: 1 },
    { name: "Pair edge attributes", count: summary.edge_feature_dim },
  ];
  $("#feature-list").innerHTML = groups
    .map((f) => `<div class="feature-item"><strong>${f.name}</strong><span>${f.count} dims</span></div>`)
    .join("");
}

async function renderSpeciesBars() {
  const summaries = await Promise.all(state.species.map((name) => api(`/species/${name}/summary?split=${SPLIT}`)));
  const maxEdges = Math.max(...summaries.map((item) => item.positive_edges));
  const colors = { human: palette.primary, cow: palette.rose, mouse: palette.teal, worm: palette.amber };
  $("#species-bars").innerHTML = summaries
    .map((item) => {
      const width = Math.round((item.positive_edges / maxEdges) * 100);
      return `
        <div class="hbar-row">
          <strong>${item.species}</strong>
          <span class="hbar-track"><i class="hbar-fill" style="--value: ${width}%; --bar-color: ${colors[item.species]};"></i></span>
          <span>${formatNumber(item.positive_edges)} edges</span>
        </div>
      `;
    })
    .join("");
}

async function renderBipartiteGraph(species = state.activeSpecies) {
  const svg = $("#bipartite-graph");
  svg.innerHTML = "";
  const [summary, graph] = await Promise.all([
    api(`/species/${species}/summary?split=${SPLIT}`),
    api(`/species/${species}/graph?split=${SPLIT}&limit_mirna=8&limit_edges=28`),
  ]);

  svg.appendChild(createSvg("text", { x: 54, y: 40, class: "chart-label" }, `miRNA nodes: ${formatNumber(summary.num_mirna_nodes)}`));
  svg.appendChild(createSvg("text", { x: 510, y: 40, class: "chart-label" }, `mRNA nodes: ${formatNumber(summary.num_mrna_nodes)}`));
  svg.appendChild(createSvg("text", { x: 292, y: 405, class: "chart-label" }, `validated edges: ${formatNumber(summary.positive_edges)}`));

  const mirnas = graph.nodes.filter((node) => node.type === "mirna").slice(0, 6);
  const mrnas = graph.nodes.filter((node) => node.type === "mrna").slice(0, 7);
  const mirnaIndex = new Map(mirnas.map((node, index) => [node.id, index]));
  const mrnaIndex = new Map(mrnas.map((node, index) => [node.id, index]));
  const leftX = 150;
  const rightX = 610;
  const topY = 82;

  graph.edges.forEach((edge, index) => {
    if (!mirnaIndex.has(edge.source_id) || !mrnaIndex.has(edge.target_id)) return;
    const y1 = topY + mirnaIndex.get(edge.source_id) * 52;
    const y2 = topY + mrnaIndex.get(edge.target_id) * 42;
    const path = `M ${leftX + 32} ${y1} C 305 ${y1}, 435 ${y2}, ${rightX - 32} ${y2}`;
    svg.appendChild(createSvg("path", { d: path, class: "graph-edge", opacity: String(0.35 + (index % 5) * 0.1) }));
  });

  mirnas.forEach((node, index) => {
    const y = topY + index * 52;
    svg.appendChild(createSvg("circle", { cx: leftX, cy: y, r: 23, class: "graph-node mirna" }));
    svg.appendChild(createSvg("text", { x: leftX - 104, y: y + 5, class: "graph-node-label" }, shortLabel(node.name)));
  });

  mrnas.forEach((node, index) => {
    const y = topY + index * 42;
    svg.appendChild(createSvg("circle", { cx: rightX, cy: y, r: 22, class: "graph-node mrna" }));
    svg.appendChild(createSvg("text", { x: rightX + 32, y: y + 5, class: "graph-node-label" }, shortLabel(node.name)));
  });
}

function setupSpeciesSwitch() {
  $$(".segment").forEach((button) => {
    button.addEventListener("click", async () => {
      const species = button.dataset.species;
      state.activeSpecies = species;
      $$(".segment").forEach((item) => item.classList.toggle("is-active", item === button));
      $("#active-species-chip").textContent = species;
      await renderBipartiteGraph(species);
      await renderFeatureList();
    });
  });
}

function renderModelCards() {
  const rows = state.transfer.rows;
  const mean = (field) => rows.reduce((sum, row) => sum + Number(row[field] || 0), 0) / rows.length;
  const cards = [
    { name: "GraphSAGE dynamic negatives", acc: mean("test_accuracy"), f1: mean("test_f1"), auc: mean("test_auc"), ap: mean("test_ap"), featured: true },
    { name: "Validation tuned threshold", acc: mean("val_best_accuracy"), f1: mean("val_best_f1"), auc: mean("source_best_val_auc"), ap: mean("test_ap") },
    { name: "Cold-miRNA transfer mean", acc: mean("test_accuracy"), f1: mean("test_f1"), auc: mean("test_auc"), ap: mean("test_ap") },
  ];
  $("#model-cards").innerHTML = cards
    .map(
      (model) => `
        <article class="model-card ${model.featured ? "featured" : ""}">
          <h3>${model.name}</h3>
          <dl>
            <div><dt>ACC</dt><dd>${fmt(model.acc, 2)}</dd></div>
            <div><dt>F1</dt><dd>${fmt(model.f1, 2)}</dd></div>
            <div><dt>AUC</dt><dd>${fmt(model.auc, 2)}</dd></div>
            <div><dt>AP</dt><dd>${fmt(model.ap, 2)}</dd></div>
          </dl>
        </article>
      `,
    )
    .join("");
}

function renderSameSpeciesChart() {
  const svg = $("#same-species-chart");
  svg.innerHTML = "";
  const width = 760;
  const height = 360;
  const margin = { top: 36, right: 34, bottom: 58, left: 54 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;
  const minY = 0.45;
  const maxY = 0.85;
  const rows = state.species.map((species) => state.transfer.rows.find((row) => row.source === species && row.target === species));
  const metrics = [
    { key: "test_auc", label: "AUC", color: palette.primary },
    { key: "test_ap", label: "AP", color: palette.teal },
    { key: "test_f1", label: "F1", color: palette.rose },
  ];

  [0.5, 0.6, 0.7, 0.8].forEach((tick) => {
    const y = margin.top + chartHeight - ((tick - minY) / (maxY - minY)) * chartHeight;
    svg.appendChild(createSvg("line", { x1: margin.left, y1: y, x2: width - margin.right, y2: y, class: "grid-line" }));
    svg.appendChild(createSvg("text", { x: 18, y: y + 4, class: "chart-label" }, tick.toFixed(1)));
  });
  svg.appendChild(createSvg("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, class: "axis-line" }));
  svg.appendChild(createSvg("line", { x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, class: "axis-line" }));

  const groupWidth = chartWidth / rows.length;
  rows.forEach((row, rowIndex) => {
    const groupX = margin.left + rowIndex * groupWidth;
    svg.appendChild(createSvg("text", { x: groupX + groupWidth / 2 - 20, y: height - 24, class: "chart-label" }, row.source));
    metrics.forEach((metric, metricIndex) => {
      const value = Number(row[metric.key]);
      const barHeight = Math.max(2, ((value - minY) / (maxY - minY)) * chartHeight);
      const x = groupX + 34 + metricIndex * 25;
      const y = margin.top + chartHeight - barHeight;
      svg.appendChild(createSvg("rect", { x, y, width: 18, height: barHeight, fill: metric.color, class: "chart-bar" }));
    });
  });

  metrics.forEach((metric, index) => {
    const x = width - 260 + index * 78;
    svg.appendChild(createSvg("rect", { x, y: 20, width: 14, height: 14, rx: 5, fill: metric.color }));
    svg.appendChild(createSvg("text", { x: x + 20, y: 32, class: "chart-label" }, metric.label));
  });
}

function heatColor(value) {
  if (value >= 0.72) return "#CCECE6";
  if (value >= 0.67) return "#D8F0C5";
  if (value >= 0.62) return "#E8DEF8";
  return "#FFD8E4";
}

function renderHeatmap(metric = "auc") {
  const species = state.transfer.species;
  const matrix = state.transfer.matrices[metric];
  const cells = [`<div class="heatmap-head">source</div>`];
  species.forEach((name) => cells.push(`<div class="heatmap-head">${name}</div>`));
  species.forEach((source, rowIndex) => {
    cells.push(`<div class="heatmap-head">${source}</div>`);
    matrix[rowIndex].forEach((value) => {
      cells.push(`<div class="heatmap-cell" style="--cell-color: ${heatColor(value)}">${fmt(value, 3)}</div>`);
    });
  });
  $("#heatmap").innerHTML = cells.join("");
}

function columnMean(field, target) {
  const rows = state.transfer.rows.filter((row) => row.target === target);
  return rows.reduce((sum, row) => sum + Number(row[field] || 0), 0) / rows.length;
}

function renderTransferChart() {
  const svg = $("#transfer-chart");
  svg.innerHTML = "";
  const width = 760;
  const height = 330;
  const margin = { top: 34, right: 34, bottom: 52, left: 54 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;
  const minY = 0.55;
  const maxY = 0.78;
  const curves = [
    { name: "AUC", color: palette.primary, values: state.species.map((target) => columnMean("test_auc", target)) },
    { name: "AP", color: palette.teal, values: state.species.map((target) => columnMean("test_ap", target)) },
    { name: "F1", color: palette.rose, values: state.species.map((target) => columnMean("test_f1", target)) },
  ];

  [0.55, 0.6, 0.65, 0.7, 0.75].forEach((tick) => {
    const y = margin.top + chartHeight - ((tick - minY) / (maxY - minY)) * chartHeight;
    svg.appendChild(createSvg("line", { x1: margin.left, y1: y, x2: width - margin.right, y2: y, class: "grid-line" }));
    svg.appendChild(createSvg("text", { x: 18, y: y + 4, class: "chart-label" }, tick.toFixed(2)));
  });
  svg.appendChild(createSvg("line", { x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, class: "axis-line" }));
  svg.appendChild(createSvg("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, class: "axis-line" }));

  state.species.forEach((label, index) => {
    const x = margin.left + (index / (state.species.length - 1)) * chartWidth;
    svg.appendChild(createSvg("text", { x: x - 20, y: height - 22, class: "chart-label" }, label));
  });

  curves.forEach((curve, curveIndex) => {
    const points = curve.values.map((value, index) => {
      const x = margin.left + (index / (curve.values.length - 1)) * chartWidth;
      const y = margin.top + chartHeight - ((value - minY) / (maxY - minY)) * chartHeight;
      return [x, y];
    });
    const path = points.map(([x, y], index) => `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
    svg.appendChild(createSvg("path", { d: path, stroke: curve.color, class: "line-curve" }));
    points.forEach(([x, y]) => svg.appendChild(createSvg("circle", { cx: x, cy: y, r: 5, fill: curve.color })));
    const legendX = width - 230 + curveIndex * 74;
    svg.appendChild(createSvg("circle", { cx: legendX, cy: 24, r: 6, fill: curve.color }));
    svg.appendChild(createSvg("text", { x: legendX + 12, y: 29, class: "chart-label" }, curve.name));
  });
}

function renderTopK(rows = []) {
  $("#topk-body").innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${row.mirna || ""}</td>
          <td>${shortLabel(row.mrna || row.mrna_id)}</td>
          <td><span class="prob-pill">${fmt(row.calibrated_score ?? row.score, 3)}</span></td>
          <td>${(row.calibrated_score ?? row.score) >= 0.5 ? "Positive interaction" : "Ranked candidate"}</td>
        </tr>
      `,
    )
    .join("");
}

async function loadDefaultTopK() {
  try {
    const result = await api(`/predict/topk?species=human&source_model=human&mirna=${encodeURIComponent("hsa-miR-21")}&k=5&run_id=${encodeURIComponent(state.runId)}`);
    renderTopK(result.items);
  } catch {
    renderTopK([]);
  }
}

function setupPrediction() {
  $("#predict-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const mirna = $("#mirna-input").value.trim() || "hsa-miR-21";
    const mrna = $("#mrna-input").value.trim() || "phosphatase";
    $("#probability-value").textContent = "...";
    $("#prediction-label").textContent = "Running GraphSAGE";
    $("#confidence-label").textContent = "Loading";
    $("#chosen-model").textContent = "GraphSAGE";
    try {
      const result = await api("/predict/pair", {
        method: "POST",
        body: JSON.stringify({ species: "human", source_model: "human", mirna, mrna, run_id: state.runId }),
      });
      if (!result.found) {
        $("#probability-value").textContent = "NA";
        $("#prediction-label").textContent = result.message;
        $("#confidence-label").textContent = "Node missing";
        return;
      }
      $("#probability-value").textContent = fmt(result.calibrated_score, 3);
      $("#prediction-label").textContent = result.label;
      $("#confidence-label").textContent = result.confidence;
      $("#chosen-model").textContent = `${result.source_model}->${result.target_species}`;
      const topk = await api(`/predict/topk?species=human&source_model=human&mirna=${encodeURIComponent(mirna)}&k=5&run_id=${encodeURIComponent(state.runId)}`);
      renderTopK(topk.items);
    } catch (error) {
      $("#probability-value").textContent = "ERR";
      $("#prediction-label").textContent = error.message;
      $("#confidence-label").textContent = "Failed";
    }
  });
}

async function setupNetworkControls() {
  const select = $("#network-mirna");
  const nodes = await api(`/species/human/nodes/search?split=${SPLIT}&kind=mirna&q=hsa&limit=20`);
  select.innerHTML = nodes.items.map((node) => `<option value="${node.name}">${shortLabel(node.name)}</option>`).join("");
  select.insertAdjacentHTML("afterbegin", `<option value="hsa-miR-21">hsa-miR-21</option>`);
  select.value = "hsa-miR-21";
  select.addEventListener("change", renderNetwork);
  $("#threshold-input").addEventListener("input", renderNetwork);
}

async function renderNetwork() {
  const svg = $("#interaction-network");
  const mirna = $("#network-mirna").value || "hsa-miR-21";
  const threshold = Number($("#threshold-input").value);
  svg.innerHTML = "";
  $("#threshold-value").textContent = threshold.toFixed(2);
  try {
    const topk = await api(`/predict/topk?species=human&source_model=human&mirna=${encodeURIComponent(mirna)}&k=12&run_id=${encodeURIComponent(state.runId)}`);
    drawNetwork(svg, mirna, topk.items.filter((item) => (item.calibrated_score ?? item.score) >= threshold).slice(0, 10), threshold);
  } catch (error) {
    svg.appendChild(createSvg("text", { x: 260, y: 280, class: "chart-label" }, error.message));
    $("#network-summary").innerHTML = `<div class="summary-row"><span>Status</span><strong>Failed</strong></div>`;
  }
}

function drawNetwork(svg, mirna, targets, threshold) {
  const center = { x: 450, y: 280 };
  const radiusX = 318;
  const radiusY = 190;
  svg.appendChild(createSvg("text", { x: 30, y: 40, class: "chart-label" }, `${mirna} ranked mRNA candidates`));
  if (!targets.length) {
    svg.appendChild(createSvg("text", { x: 322, y: 280, class: "chart-label" }, "当前阈值下暂无候选边"));
    $("#network-summary").innerHTML = `<div class="summary-row"><span>miRNA</span><strong>${mirna}</strong></div><div class="summary-row"><span>mRNA 节点</span><strong>0</strong></div>`;
    return;
  }
  const positioned = targets.map((target, index) => {
    const angle = -Math.PI / 2 + (index / targets.length) * Math.PI * 2;
    return { ...target, x: center.x + Math.cos(angle) * radiusX, y: center.y + Math.sin(angle) * radiusY };
  });
  positioned.forEach((target) => {
    const score = Number(target.calibrated_score ?? target.score);
    const path = `M ${center.x} ${center.y} Q ${(center.x + target.x) / 2} ${(center.y + target.y) / 2 - 36} ${target.x} ${target.y}`;
    svg.appendChild(createSvg("path", { d: path, class: "net-edge predicted", "stroke-width": (1.2 + score * 7).toFixed(1) }));
    svg.appendChild(createSvg("text", { x: (center.x + target.x) / 2 + 10, y: (center.y + target.y) / 2 - 10, class: "net-prob" }, fmt(score)));
  });
  const centerGroup = createSvg("g", { class: "net-node mirna" });
  centerGroup.appendChild(createSvg("circle", { cx: center.x, cy: center.y, r: 42 }));
  centerGroup.appendChild(createSvg("text", { x: center.x, y: center.y + 64, "text-anchor": "middle" }, shortLabel(mirna)));
  svg.appendChild(centerGroup);
  positioned.forEach((target) => {
    const group = createSvg("g", { class: "net-node mrna" });
    group.appendChild(createSvg("circle", { cx: target.x, cy: target.y, r: 30 }));
    group.appendChild(createSvg("text", { x: target.x, y: target.y + 48, "text-anchor": "middle" }, shortLabel(target.mrna)));
    svg.appendChild(group);
  });
  const best = positioned.reduce((top, item) => (Number(item.calibrated_score ?? item.score) > Number(top.calibrated_score ?? top.score) ? item : top), positioned[0]);
  $("#network-summary").innerHTML = `
    <div class="summary-row"><span>miRNA</span><strong>${shortLabel(mirna)}</strong></div>
    <div class="summary-row"><span>mRNA 节点</span><strong>${positioned.length}</strong></div>
    <div class="summary-row"><span>最高分数</span><strong>${shortLabel(best.mrna)} ${fmt(best.calibrated_score ?? best.score)}</strong></div>
    <div class="summary-row"><span>预测阈值</span><strong>${threshold.toFixed(2)}</strong></div>
  `;
}

function renderDiagnosticsGallery() {
  const gallery = $(".result-gallery");
  const run = encodeURIComponent(state.runId);
  gallery.innerHTML = `
    <figure><img src="${imageUrl(`/results/transfer/heatmap/auc?run_id=${run}`)}" alt="Transfer AUC heatmap" /><figcaption>Cold-miRNA AUC heatmap</figcaption></figure>
    <figure><img src="${imageUrl(`/results/transfer/heatmap/ap?run_id=${run}`)}" alt="Transfer AP heatmap" /><figcaption>Cold-miRNA AP heatmap</figcaption></figure>
    <figure><img src="${imageUrl(`/results/transfer/heatmap/f1?run_id=${run}`)}" alt="Transfer F1 heatmap" /><figcaption>Threshold-tuned F1 heatmap</figcaption></figure>
    <figure><img src="${imageUrl(`/diagnostics/human/human/plot/score_distribution?run_id=${run}`)}" alt="Score distribution" /><figcaption>Score distribution</figcaption></figure>
    <figure><img src="${imageUrl(`/diagnostics/human/human/plot/pr_curve?run_id=${run}`)}" alt="PR curve" /><figcaption>Validation PR curve</figcaption></figure>
    <figure><img src="${imageUrl(`/diagnostics/human/human/plot/calibration_curve?run_id=${run}`)}" alt="Calibration curve" /><figcaption>Calibration curve</figcaption></figure>
  `;
}

function setupNavigation() {
  $$(".nav-item").forEach((button) => {
    button.addEventListener("click", () => {
      const page = button.dataset.page;
      $$(".nav-item").forEach((item) => {
        item.classList.toggle("is-active", item === button);
        item.removeAttribute("aria-current");
      });
      button.setAttribute("aria-current", "page");
      $$(".page").forEach((section) => section.classList.toggle("is-active", section.id === page));
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  });
}

async function init() {
  setupNavigation();
  ["#dashboard-stats", "#overview-bars", "#species-bars", "#model-cards"].forEach((selector) => loading(selector));
  try {
    await loadInitialData();
    renderDashboard();
    await renderFeatureList();
    await renderSpeciesBars();
    await renderBipartiteGraph();
    setupSpeciesSwitch();
    renderModelCards();
    renderSameSpeciesChart();
    renderHeatmap("auc");
    renderTransferChart();
    renderDiagnosticsGallery();
    await loadDefaultTopK();
    setupPrediction();
    await setupNetworkControls();
    await renderNetwork();
  } catch (error) {
    console.error(error);
    showError(error);
  }
}

document.addEventListener("DOMContentLoaded", init);
