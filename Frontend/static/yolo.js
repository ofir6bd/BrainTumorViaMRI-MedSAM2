// YOLO tumour-segmentation viewer.
// Patient dropdown + slice slider (same controls as Explore), plus a "Slice summary"
// tab listing per-slice Dice (GET /yolo/api/dice). The segmentation image itself already
// shows predicted vs. expert mask overlays and the current slice's Dice score.

const Y = {
  patients: [],
  patientIdx: 0,
  patient: null,      // { patient_id, depth, slice_indices, best_slice_index, min_fg_voxels }
  weights: [],         // [{ filename, label, val_loss, test_dice, has_training_info }] best-first
  weightsFile: "",     // selected filename, "" = server default (best available)
  diceRows: {},        // `${patientIdx}::${weightsFile}` -> rows from /yolo/api/dice
  view: "segment",     // "segment" | "summary"
  trainingOpen: false,
  ready: false,
  showGT: true,
  showPred: true,
  showOverlap: true,
};

const yel = {};

async function initYolo() {
  if (Y.ready) return;
  Y.ready = true;
  yel.sel = document.getElementById("yoloPatient");
  yel.modelSel = document.getElementById("yoloModel");
  yel.zSlider = document.getElementById("yoloZ");
  yel.zLabel = document.getElementById("yoloZLabel");
  yel.sliceField = document.getElementById("yoloSliceField");
  yel.viewer = document.getElementById("yoloViewer");
  yel.status = document.getElementById("yoloStatus");
  yel.bestBtn = document.getElementById("yoloBest");
  yel.views = document.getElementById("yoloViews");
  yel.summaryPanel = document.getElementById("yoloSummaryPanel");
  yel.summaryWrap = document.getElementById("yoloSummaryTableWrap");
  yel.summaryChartWrap = document.getElementById("yoloSliceDiceChart");
  yel.main = document.querySelector("#yoloPanel main");
  yel.showTrainingBtn = document.getElementById("yoloShowTraining");
  yel.trainingPanel = document.getElementById("yoloTrainingPanel");
  yel.trainingWrap = document.getElementById("yoloTrainingWrap");
  yel.maskLegend = document.getElementById("yoloMaskLegend");

  Y.patients = await (await fetch("/yolo/api/patients")).json();
  yel.sel.innerHTML = "";
  if (!Y.patients.length) {
    yel.status.textContent = "No patients found in data/sample.";
    yel.viewer.removeAttribute("src");
    return;
  }
  for (const p of Y.patients) {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = p.label;
    yel.sel.appendChild(opt);
  }

  await loadWeightsList();

  yel.sel.addEventListener("change", (e) => loadPatient(Number(e.target.value)));
  yel.modelSel.addEventListener("change", (e) => {
    Y.weightsFile = e.target.value;
    renderSlice();
    if (Y.view === "summary") renderSummaryTable(Y.patientIdx);
    if (Y.trainingOpen) loadTrainingInfo();
  });
  yel.zSlider.addEventListener("input", renderSlice);
  yel.bestBtn.addEventListener("click", () => {
    yel.zSlider.value = bestZ();
    renderSlice();
  });
  yel.views.addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (!btn) return;
    setYoloView(btn.dataset.yview);
  });
  if (yel.showTrainingBtn) {
    yel.showTrainingBtn.addEventListener("click", toggleTrainingInfo);
  }
  if (yel.maskLegend) {
    yel.maskLegend.addEventListener("click", (e) => {
      const item = e.target.closest(".yolo-chart-legend-item");
      if (!item) return;
      const mask = item.dataset.mask;
      if (mask === "gt") Y.showGT = !Y.showGT;
      else if (mask === "pred") Y.showPred = !Y.showPred;
      else if (mask === "overlap") Y.showOverlap = !Y.showOverlap;
      else return;
      item.classList.toggle("off");
      renderSlice();
    });
  }

  await loadPatient(0);
}
window.initYolo = initYolo;

async function loadWeightsList() {
  if (!yel.modelSel) return;
  Y.weights = await (await fetch("/yolo/api/weights")).json();
  yel.modelSel.innerHTML = "";
  if (!Y.weights.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No trained weights found";
    yel.modelSel.appendChild(opt);
    yel.modelSel.disabled = true;
    Y.weightsFile = "";
    return;
  }
  yel.modelSel.disabled = false;
  for (const w of Y.weights) {
    const opt = document.createElement("option");
    opt.value = w.filename;
    opt.textContent = w.label;
    yel.modelSel.appendChild(opt);
  }
  // First entry is the best-by-val_loss model (see list_weight_files sort order).
  Y.weightsFile = Y.weights[0].filename;
  yel.modelSel.value = Y.weightsFile;
}

async function toggleTrainingInfo() {
  if (!yel.trainingPanel) return;
  Y.trainingOpen = !Y.trainingOpen;
  yel.trainingPanel.classList.toggle("hidden", !Y.trainingOpen);
  yel.showTrainingBtn.textContent = Y.trainingOpen ? "Hide training data" : "Show training data";
  if (Y.trainingOpen) await loadTrainingInfo();
}

async function loadTrainingInfo() {
  if (!yel.trainingWrap) return;
  if (!Y.weightsFile) {
    yel.trainingWrap.textContent = "No model selected.";
    return;
  }
  yel.trainingWrap.textContent = "Loading training data…";
  try {
    const resp = await fetch(`/yolo/api/weights/${encodeURIComponent(Y.weightsFile)}/info`);
    const meta = await resp.json();
    renderTrainingInfo(meta);
  } catch (e) {
    yel.trainingWrap.textContent = "Failed to load training data.";
  }
}

function _trainingTable(columns, rows, formatters) {
  const wrap = document.createElement("div");
  wrap.className = "yolo-training-table-wrap";
  const table = document.createElement("table");
  table.className = "yolo-summary-table";

  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  columns.forEach((c) => {
    const th = document.createElement("th");
    th.textContent = c.label;
    hr.appendChild(th);
  });
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((c) => {
      const td = document.createElement("td");
      td.textContent = formatters(row, c.key);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  return wrap;
}

// Minimal dependency-free SVG line chart (no external chart library / CDN needed).
// seriesList: [{ label, color, data: [{x, y}, ...], axis }, ...]. `axis` is optional and
// defaults to "left"; series with `axis: "right"` are scaled against their own min/max
// and plotted against a secondary y-axis drawn on the right edge of the chart (e.g. a
// tumour voxel count overlaid on a 0-1 Dice score). Returns null if there is no
// plottable data at all.
function _lineChartSVG(seriesList, opts = {}) {
  const width = opts.width || 580;
  const height = opts.height || 200;
  const padLeft = 54;
  const plottable = seriesList.filter((s) => s.data && s.data.length);
  if (!plottable.length) return null;
  const rightSeries = plottable.filter((s) => s.axis === "right");
  const leftSeries = plottable.filter((s) => s.axis !== "right");
  const hasRightAxis = rightSeries.length > 0;
  const padRight = hasRightAxis ? 60 : 12;
  const padTop = 12;
  const padBottom = 30;

  const allPoints = plottable.flatMap((s) => s.data);
  const xs = allPoints.map((p) => p.x);
  let xMin = Math.min(...xs), xMax = Math.max(...xs);
  if (xMax === xMin) xMax = xMin + 1;

  const leftPoints = leftSeries.flatMap((s) => s.data);
  let yMin = 0, yMax = 1;
  if (leftPoints.length) {
    const ys = leftPoints.map((p) => p.y);
    yMin = Math.min(...ys); yMax = Math.max(...ys);
    if (yMax === yMin) { yMax += 1; yMin -= 1; }
  }

  const rightPoints = rightSeries.flatMap((s) => s.data);
  let yMinR = 0, yMaxR = 1;
  if (rightPoints.length) {
    const ysR = rightPoints.map((p) => p.y);
    yMinR = Math.min(...ysR); yMaxR = Math.max(...ysR);
    if (yMaxR === yMinR) { yMaxR += 1; yMinR -= 1; }
  }

  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;
  const sx = (x) => padLeft + ((x - xMin) / (xMax - xMin)) * plotW;
  const sy = (y) => padTop + plotH - ((y - yMin) / (yMax - yMin)) * plotH;
  const syR = (y) => padTop + plotH - ((y - yMinR) / (yMaxR - yMinR)) * plotH;

  let svg = `<svg viewBox="0 0 ${width} ${height}" class="yolo-chart-svg" `
    + `style="height:${height}px" preserveAspectRatio="xMidYMid meet">`;

  const yTicks = [yMin, (yMin + yMax) / 2, yMax];
  yTicks.forEach((t) => {
    const y = sy(t);
    svg += `<line x1="${padLeft}" y1="${y.toFixed(1)}" x2="${padLeft + plotW}" `
      + `y2="${y.toFixed(1)}" class="yolo-chart-grid" />`;
    svg += `<text x="${padLeft - 6}" y="${(y + 3).toFixed(1)}" `
      + `class="yolo-chart-ticklabel yolo-chart-ylabel" text-anchor="end">${t.toFixed(2)}</text>`;
  });
  if (hasRightAxis) {
    const rColor = rightSeries[0].color;
    const rTicks = [yMinR, (yMinR + yMaxR) / 2, yMaxR];
    rTicks.forEach((t) => {
      const y = syR(t);
      svg += `<text x="${padLeft + plotW + 8}" y="${(y + 3).toFixed(1)}" `
        + `class="yolo-chart-ticklabel yolo-chart-ylabel" text-anchor="start" `
        + `style="fill:${rColor}">${Math.round(t).toLocaleString()}</text>`;
    });
  }
  const xTickCount = opts.xTickCount || 2;
  for (let i = 0; i <= xTickCount; i++) {
    const t = xMin + (i / xTickCount) * (xMax - xMin);
    const x = sx(t);
    svg += `<text x="${x.toFixed(1)}" y="${(padTop + plotH + 18).toFixed(1)}" `
      + `class="yolo-chart-ticklabel yolo-chart-xlabel" text-anchor="middle">${Math.round(t)}</text>`;
  }
  svg += `<line x1="${padLeft}" y1="${padTop}" x2="${padLeft}" y2="${padTop + plotH}" `
    + `class="yolo-chart-axis" />`;
  svg += `<line x1="${padLeft}" y1="${padTop + plotH}" x2="${padLeft + plotW}" `
    + `y2="${padTop + plotH}" class="yolo-chart-axis" />`;
  if (hasRightAxis) {
    svg += `<line x1="${padLeft + plotW}" y1="${padTop}" x2="${padLeft + plotW}" `
      + `y2="${padTop + plotH}" class="yolo-chart-axis" />`;
  }

  if (typeof opts.markerX === "number" && opts.markerX >= xMin && opts.markerX <= xMax) {
    const mx = sx(opts.markerX);
    svg += `<line x1="${mx.toFixed(1)}" y1="${padTop}" x2="${mx.toFixed(1)}" `
      + `y2="${padTop + plotH}" class="yolo-chart-mean-line" />`;
  }

  plottable.forEach((s, i) => {
    const scaleY = s.axis === "right" ? syR : sy;
    const d = s.data
      .map((p, j) => `${j === 0 ? "M" : "L"} ${sx(p.x).toFixed(1)} ${scaleY(p.y).toFixed(1)}`)
      .join(" ");
    svg += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" `
      + `class="yolo-series-line" data-series-idx="${i}" />`;
  });
  svg += `</svg>`;

  const wrap = document.createElement("div");
  wrap.className = "yolo-chart-wrap";
  wrap.innerHTML = svg;

  const legend = document.createElement("div");
  legend.className = "yolo-chart-legend";
  plottable.forEach((s, i) => {
    const item = document.createElement("span");
    item.className = "yolo-chart-legend-item";
    item.title = "Click to show/hide this series";
    const swatch = document.createElement("i");
    swatch.style.background = s.color;
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(s.axis === "right" ? `${s.label} (right axis)` : s.label));
    item.addEventListener("click", () => {
      const path = wrap.querySelector(`path[data-series-idx="${i}"]`);
      const nowOff = item.classList.toggle("off");
      if (path) path.style.display = nowOff ? "none" : "";
    });
    legend.appendChild(item);
  });
  wrap.appendChild(legend);
  return wrap;
}

// Minimal dependency-free SVG histogram. `values`: array of numbers (e.g. per-patient
// mean Dice). Bins the range [0, 1] (or the data's own min/max if `range` isn't given)
// into equal-width buckets and draws a bar chart, with an optional vertical marker line
// for the overall mean. Returns null if there are no values to plot.
function _histogramSVG(values, opts = {}) {
  const width = opts.width || 580;
  const height = opts.height || 200;
  const padLeft = 54;
  const padRight = 12;
  const padTop = 12;
  const padBottom = 38;
  const bins = opts.bins || 10;
  const color = opts.color || "#8b5cf6";
  const meanValue = opts.meanValue;

  const nums = values.filter((v) => typeof v === "number" && !Number.isNaN(v));
  if (!nums.length) return null;

  const [rangeMin, rangeMax] = opts.range || [0, 1];
  const binWidth = (rangeMax - rangeMin) / bins;
  const counts = new Array(bins).fill(0);
  nums.forEach((v) => {
    let idx = Math.floor((v - rangeMin) / binWidth);
    if (idx < 0) idx = 0;
    if (idx >= bins) idx = bins - 1;
    counts[idx] += 1;
  });
  const maxCount = Math.max(...counts, 1);

  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;
  const sx = (binIdx) => padLeft + (binIdx / bins) * plotW;
  const sy = (c) => padTop + plotH - (c / maxCount) * plotH;
  const barGap = 2;
  const barW = plotW / bins - barGap;

  let svg = `<svg viewBox="0 0 ${width} ${height}" class="yolo-chart-svg" `
    + `style="height:${height}px" preserveAspectRatio="xMidYMid meet">`;

  const yTickStep = 10;
  for (let t = 0; t <= maxCount + 1e-9; t += yTickStep) {
    const y = sy(t);
    svg += `<line x1="${padLeft}" y1="${y.toFixed(1)}" x2="${padLeft + plotW}" `
      + `y2="${y.toFixed(1)}" class="yolo-chart-grid" />`;
    svg += `<text x="${padLeft - 6}" y="${(y + 3).toFixed(1)}" `
      + `class="yolo-chart-ticklabel yolo-chart-ylabel" text-anchor="end">${Math.round(t)}</text>`;
  }
  if (maxCount % yTickStep !== 0) {
    const y = sy(maxCount);
    svg += `<line x1="${padLeft}" y1="${y.toFixed(1)}" x2="${padLeft + plotW}" `
      + `y2="${y.toFixed(1)}" class="yolo-chart-grid" />`;
    svg += `<text x="${padLeft - 6}" y="${(y + 3).toFixed(1)}" `
      + `class="yolo-chart-ticklabel yolo-chart-ylabel" text-anchor="end">${Math.round(maxCount)}</text>`;
  }

  counts.forEach((c, i) => {
    if (c <= 0) return;
    const x = sx(i) + barGap / 2;
    const y = sy(c);
    const h = padTop + plotH - y;
    svg += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" `
      + `height="${h.toFixed(1)}" fill="${color}" rx="2" />`;
  });

  for (let t = rangeMin; t <= rangeMax + 1e-9; t += 0.1) {
    const i = (t - rangeMin) / binWidth;
    const x = sx(i);
    svg += `<text x="${x.toFixed(1)}" y="${(padTop + plotH + 20).toFixed(1)}" `
      + `class="yolo-chart-ticklabel yolo-chart-xlabel" text-anchor="middle">${t.toFixed(2)}</text>`;
  }

  if (typeof meanValue === "number") {
    const mx = padLeft + ((meanValue - rangeMin) / (rangeMax - rangeMin)) * plotW;
    svg += `<line x1="${mx.toFixed(1)}" y1="${padTop}" x2="${mx.toFixed(1)}" `
      + `y2="${padTop + plotH}" class="yolo-chart-mean-line" />`;
  }

  svg += `<line x1="${padLeft}" y1="${padTop}" x2="${padLeft}" y2="${padTop + plotH}" `
    + `class="yolo-chart-axis" />`;
  svg += `<line x1="${padLeft}" y1="${padTop + plotH}" x2="${padLeft + plotW}" `
    + `y2="${padTop + plotH}" class="yolo-chart-axis" />`;
  svg += `</svg>`;

  const wrap = document.createElement("div");
  wrap.className = "yolo-chart-wrap";
  wrap.innerHTML = svg;

  if (typeof meanValue === "number") {
    const legend = document.createElement("div");
    legend.className = "yolo-chart-legend";
    const item = document.createElement("span");
    item.className = "yolo-chart-legend-item";
    const swatch = document.createElement("i");
    swatch.className = "yolo-chart-legend-dash";
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(`Overall mean Dice = ${meanValue.toFixed(4)}`));
    legend.appendChild(item);
    wrap.appendChild(legend);
  }
  return wrap;
}


function renderTrainingInfo(meta) {
  yel.trainingWrap.innerHTML = "";
  if (!meta || meta.available === false) {
    yel.trainingWrap.textContent = "No training data recorded for this checkpoint.";
    return;
  }

  const fmt = (v, d = 4) => (typeof v === "number" ? v.toFixed(d) : "—");

  const summary = document.createElement("div");
  summary.className = "yolo-training-grid";
  const summaryRows = [
    ["Trained", meta.timestamp ? meta.timestamp.replace("T", " ") : "—"],
    ["Epochs", meta.epochs ?? "—"],
    ["Image size", meta.imgsz ?? "—"],
    ["Batch size", meta.batch ?? "—"],
    ["Best val loss", fmt(meta.val_loss)],
    ["Test Dice (mean)", fmt(meta.test_dice)],
    ["Train patients", meta.n_train_patients ?? "—"],
    ["Val patients", meta.n_val_patients ?? "—"],
    ["Test patients", meta.n_test_patients ?? "—"],
    ["Data fraction", meta.data_fraction ?? "full"],
    ["Max patients", meta.max_patients ?? "—"],
  ];
  summaryRows.forEach(([k, v]) => {
    const item = document.createElement("div");
    item.className = "yolo-training-item";
    const kSpan = document.createElement("span");
    kSpan.className = "k";
    kSpan.textContent = k;
    const vSpan = document.createElement("span");
    vSpan.className = "v";
    vSpan.textContent = String(v);
    item.appendChild(kSpan);
    item.appendChild(vSpan);
    summary.appendChild(item);
  });
  yel.trainingWrap.appendChild(summary);

  const history = Array.isArray(meta.metrics_history) ? meta.metrics_history : [];
  if (history.length) {
    const epochs = history.map((r, i) => (typeof r.epoch === "number" ? r.epoch : i + 1));
    const cols0 = Object.keys(history[0] || {});
    const trainLossCols = cols0.filter((c) => c.startsWith("train/") && c.endsWith("_loss"));
    const valLossCols = cols0.filter((c) => c.startsWith("val/") && c.endsWith("_loss"));
    const sumCols = (row, cols) =>
      cols.reduce((s, c) => s + (typeof row[c] === "number" ? row[c] : 0), 0);

    const lossSeries = [];
    if (trainLossCols.length) {
      lossSeries.push({
        label: "Train loss", color: "#3b82f6",
        data: history.map((r, i) => ({ x: epochs[i], y: sumCols(r, trainLossCols) })),
      });
    }
    if (valLossCols.length) {
      lossSeries.push({
        label: "Val loss", color: "#ef4444",
        data: history.map((r, i) => ({ x: epochs[i], y: sumCols(r, valLossCols) })),
      });
    }
    const lossChart = lossSeries.length ? _lineChartSVG(lossSeries) : null;

    const mapCols = [
      ["metrics/mAP50(M)", "mAP50", "#10b981"],
      ["metrics/mAP50-95(M)", "mAP50-95", "#f59e0b"],
    ].filter(([key]) => cols0.includes(key));
    const mapSeries = mapCols.map(([key, label, color]) => ({
      label, color,
      data: history.map((r, i) => ({
        x: epochs[i], y: typeof r[key] === "number" ? r[key] : 0,
      })),
    }));
    const mapChart = mapSeries.length ? _lineChartSVG(mapSeries) : null;

    if (lossChart || mapChart) {
      const row = document.createElement("div");
      row.className = "yolo-chart-row";
      if (lossChart) {
        const col = document.createElement("div");
        col.className = "yolo-chart-col";
        const h4 = document.createElement("h4");
        h4.textContent = "Training curves — loss (train vs. val)";
        col.appendChild(h4);
        col.appendChild(lossChart);
        row.appendChild(col);
      }
      if (mapChart) {
        const colB = document.createElement("div");
        colB.className = "yolo-chart-col";
        const h4b = document.createElement("h4");
        h4b.textContent = "Training curves — mAP";
        colB.appendChild(h4b);
        colB.appendChild(mapChart);
        row.appendChild(colB);
      }
      yel.trainingWrap.appendChild(row);
    }


    const allCols = ["epoch", "train/box_loss", "train/seg_loss", "val/box_loss",
                     "val/seg_loss", "metrics/mAP50(M)", "metrics/mAP50-95(M)"];
    const cols = allCols.filter((c) => history.some((r) => c in r))
      .map((c) => ({ key: c, label: c }));
    const h4 = document.createElement("h4");
    h4.textContent = "Per-epoch metrics";
    yel.trainingWrap.appendChild(h4);
    yel.trainingWrap.appendChild(_trainingTable(cols, history, (row, key) => {
      const v = row[key];
      if (typeof v !== "number") return "";
      return key === "epoch" ? String(v) : v.toFixed(4);
    }));
  }

  const perPatient = Array.isArray(meta.per_patient_test_dice) ? meta.per_patient_test_dice : [];
  if (perPatient.length) {
    const h4 = document.createElement("h4");
    h4.textContent = "Per-patient test Dice";
    yel.trainingWrap.appendChild(h4);
    const cols = [
      { key: "patient_id", label: "patient" },
      { key: "mean_dice", label: "mean dice" },
      { key: "n_slices", label: "slices" },
    ];
    yel.trainingWrap.appendChild(_trainingTable(cols, perPatient, (row, key) => {
      const v = row[key];
      if (key === "mean_dice") return typeof v === "number" ? v.toFixed(4) : "—";
      return v === undefined || v === null ? "" : String(v);
    }));

    const diceValues = perPatient
      .map((r) => r.mean_dice)
      .filter((v) => typeof v === "number");
    const overallMeanDice = typeof meta.test_dice === "number"
      ? meta.test_dice
      : (diceValues.length ? diceValues.reduce((a, b) => a + b, 0) / diceValues.length : undefined);
    const histChart = _histogramSVG(diceValues, { meanValue: overallMeanDice, bins: 50, height: 400, width: 1160 });
    if (histChart) {
      const h4b = document.createElement("h4");
      h4b.textContent = "Distribution of mean Dice across test patients";
      yel.trainingWrap.appendChild(h4b);
      yel.trainingWrap.appendChild(histChart);
    }
  }
}

function bestZ() {
  if (!Y.patient) return 0;
  const idxList = Y.patient.slice_indices;
  if (!idxList.length) return 0;
  const pos = Math.min(Y.patient.best_slice_index, idxList.length - 1);
  return idxList[pos];
}

async function loadPatient(idx) {
  Y.patientIdx = idx;
  yel.sel.value = String(idx);
  yel.status.textContent = "Loading patient…";
  Y.patient = await (await fetch(`/yolo/api/patient/${idx}`)).json();

  yel.zSlider.max = Y.patient.depth - 1;
  yel.zSlider.value = bestZ();

  yel.status.textContent = "";
  renderSlice();
  if (Y.view === "summary") {
    await renderSummaryTable(idx);
  }
}

function renderSlice() {
  if (!Y.patient) return;
  const z = yel.zSlider.value;
  yel.zLabel.textContent = `z = ${z}`;
  yel.status.textContent = "Rendering…";
  yel.viewer.onload = () => (yel.status.textContent = "");
  yel.viewer.onerror = () => (yel.status.textContent = "Failed to render this slice.");
  const w = Y.weightsFile ? `&weights=${encodeURIComponent(Y.weightsFile)}` : "";
  const flags = `&gt=${Y.showGT ? 1 : 0}&pred=${Y.showPred ? 1 : 0}&overlap=${Y.showOverlap ? 1 : 0}`;
  yel.viewer.src = `/yolo/segment.png?id=${Y.patientIdx}&z=${z}${w}${flags}&_=${Date.now()}`;
}

function setYoloView(view) {
  if (!view || (view !== "segment" && view !== "summary")) return;
  Y.view = view;
  for (const btn of yel.views.querySelectorAll(".tab")) {
    btn.classList.toggle("active", btn.dataset.yview === view);
  }
  const summary = view === "summary";
  yel.summaryPanel.classList.toggle("hidden", !summary);
  if (yel.main) yel.main.classList.toggle("hidden", summary);
  if (yel.sliceField) yel.sliceField.classList.toggle("hidden", summary);
  if (summary) {
    renderSummaryTable(Y.patientIdx);
  }
}

async function loadDiceRows(idx) {
  const cacheKey = `${idx}::${Y.weightsFile}`;
  if (Y.diceRows[cacheKey] !== undefined) return Y.diceRows[cacheKey];
  const w = Y.weightsFile ? `&weights=${encodeURIComponent(Y.weightsFile)}` : "";
  const resp = await fetch(`/yolo/api/dice?id=${idx}${w}`);
  const payload = await resp.json();
  Y.diceRows[cacheKey] = Array.isArray(payload.rows) ? payload.rows : [];
  return Y.diceRows[cacheKey];
}

async function renderSummaryTable(idx) {
  if (!yel.summaryWrap) return;
  yel.summaryWrap.innerHTML = "";
  if (yel.summaryChartWrap) yel.summaryChartWrap.innerHTML = "";
  yel.status.textContent = "Loading slice summary… (running inference on all slices)";
  try {
    const rows = await loadDiceRows(idx);
    if (!rows.length) {
      yel.summaryWrap.textContent = "No processed slices for this patient.";
      yel.status.textContent = "";
      return;
    }

    const currentZ = Number(yel.zSlider.value);

    if (yel.summaryChartWrap) {
      const diceData = rows
        .filter((r) => typeof r.dice === "number")
        .map((r) => ({ x: r.z, y: r.dice }));
      const voxelData = rows
        .filter((r) => typeof r.tumor_voxels === "number")
        .map((r) => ({ x: r.z, y: r.tumor_voxels }));
      const predVoxelData = rows
        .filter((r) => typeof r.pred_tumor_voxels === "number")
        .map((r) => ({ x: r.z, y: r.pred_tumor_voxels }));
      const diceChart = _lineChartSVG(
        [
          { label: "Dice", color: "#3b82f6", data: diceData },
          { label: "GT tumor voxels", color: "#f59e0b", data: voxelData, axis: "right" },
          { label: "Predicted tumor voxels", color: "#22c55e", data: predVoxelData, axis: "right" },
        ],
        { height: 240, width: 900, xTickCount: 8, markerX: currentZ },
      );
      if (diceChart) {
        const h4 = document.createElement("h4");
        h4.textContent = "Dice per slice";
        yel.summaryChartWrap.appendChild(h4);
        yel.summaryChartWrap.appendChild(diceChart);
      }
    }

    const table = document.createElement("table");
    table.className = "yolo-summary-table";

    const thead = document.createElement("thead");
    const hr = document.createElement("tr");
    ["z", "dice", "GT voxels", "pred voxels"].forEach((c) => {
      const th = document.createElement("th");
      th.textContent = c;
      hr.appendChild(th);
    });
    thead.appendChild(hr);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    let activeRowEl = null;
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      if (row.z === currentZ) {
        tr.classList.add("active-row");
        activeRowEl = tr;
      }
      const tdZ = document.createElement("td");
      tdZ.textContent = String(row.z);
      tr.appendChild(tdZ);
      const tdDice = document.createElement("td");
      tdDice.textContent = (typeof row.dice === "number") ? row.dice.toFixed(3) : "";
      tr.appendChild(tdDice);
      const tdVox = document.createElement("td");
      tdVox.textContent = (typeof row.tumor_voxels === "number") ? row.tumor_voxels.toLocaleString() : "";
      tr.appendChild(tdVox);
      const tdPredVox = document.createElement("td");
      tdPredVox.textContent = (typeof row.pred_tumor_voxels === "number") ? row.pred_tumor_voxels.toLocaleString() : "";
      tr.appendChild(tdPredVox);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    yel.summaryWrap.appendChild(table);
    if (activeRowEl) {
      activeRowEl.scrollIntoView({ block: "center", behavior: "auto" });
    }
    yel.status.textContent = "";
  } catch (e) {
    yel.summaryWrap.textContent = "Failed to load slice summary.";
    yel.status.textContent = "";
  }
}
