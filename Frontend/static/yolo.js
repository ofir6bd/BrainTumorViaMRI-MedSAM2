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
  yel.main = document.querySelector("#yoloPanel main");
  yel.showTrainingBtn = document.getElementById("yoloShowTraining");
  yel.trainingPanel = document.getElementById("yoloTrainingPanel");
  yel.trainingWrap = document.getElementById("yoloTrainingWrap");

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
// seriesList: [{ label, color, data: [{x, y}, ...] }, ...]. Returns null if there is
// no plottable data at all.
function _lineChartSVG(seriesList, opts = {}) {
  const width = opts.width || 580;
  const height = opts.height || 200;
  const padLeft = 46;
  const padRight = 12;
  const padTop = 12;
  const padBottom = 26;

  const plottable = seriesList.filter((s) => s.data && s.data.length);
  const allPoints = plottable.flatMap((s) => s.data);
  if (!allPoints.length) return null;

  const xs = allPoints.map((p) => p.x);
  const ys = allPoints.map((p) => p.y);
  let xMin = Math.min(...xs), xMax = Math.max(...xs);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  if (xMax === xMin) xMax = xMin + 1;
  if (yMax === yMin) { yMax += 1; yMin -= 1; }

  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;
  const sx = (x) => padLeft + ((x - xMin) / (xMax - xMin)) * plotW;
  const sy = (y) => padTop + plotH - ((y - yMin) / (yMax - yMin)) * plotH;

  let svg = `<svg viewBox="0 0 ${width} ${height}" class="yolo-chart-svg" `
    + `preserveAspectRatio="xMidYMid meet">`;

  const yTicks = [yMin, (yMin + yMax) / 2, yMax];
  yTicks.forEach((t) => {
    const y = sy(t);
    svg += `<line x1="${padLeft}" y1="${y.toFixed(1)}" x2="${padLeft + plotW}" `
      + `y2="${y.toFixed(1)}" class="yolo-chart-grid" />`;
    svg += `<text x="${padLeft - 6}" y="${(y + 3).toFixed(1)}" `
      + `class="yolo-chart-ticklabel" text-anchor="end">${t.toFixed(2)}</text>`;
  });
  [xMin, xMax].forEach((t) => {
    const x = sx(t);
    svg += `<text x="${x.toFixed(1)}" y="${(padTop + plotH + 18).toFixed(1)}" `
      + `class="yolo-chart-ticklabel" text-anchor="middle">${Math.round(t)}</text>`;
  });
  svg += `<line x1="${padLeft}" y1="${padTop}" x2="${padLeft}" y2="${padTop + plotH}" `
    + `class="yolo-chart-axis" />`;
  svg += `<line x1="${padLeft}" y1="${padTop + plotH}" x2="${padLeft + plotW}" `
    + `y2="${padTop + plotH}" class="yolo-chart-axis" />`;

  plottable.forEach((s) => {
    const d = s.data
      .map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}`)
      .join(" ");
    svg += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" />`;
  });
  svg += `</svg>`;

  const wrap = document.createElement("div");
  wrap.className = "yolo-chart-wrap";
  wrap.innerHTML = svg;

  const legend = document.createElement("div");
  legend.className = "yolo-chart-legend";
  plottable.forEach((s) => {
    const item = document.createElement("span");
    item.className = "yolo-chart-legend-item";
    const swatch = document.createElement("i");
    swatch.style.background = s.color;
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(s.label));
    legend.appendChild(item);
  });
  wrap.appendChild(legend);
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
    if (lossChart) {
      const h4 = document.createElement("h4");
      h4.textContent = "Training curves — loss (train vs. val)";
      yel.trainingWrap.appendChild(h4);
      yel.trainingWrap.appendChild(lossChart);
    }

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
    if (mapChart) {
      const h4b = document.createElement("h4");
      h4b.textContent = "Training curves — mAP";
      yel.trainingWrap.appendChild(h4b);
      yel.trainingWrap.appendChild(mapChart);
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
  yel.viewer.src = `/yolo/segment.png?id=${Y.patientIdx}&z=${z}${w}&_=${Date.now()}`;
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
  yel.status.textContent = "Loading slice summary… (running inference on all slices)";
  try {
    const rows = await loadDiceRows(idx);
    if (!rows.length) {
      yel.summaryWrap.textContent = "No processed slices for this patient.";
      yel.status.textContent = "";
      return;
    }

    const currentZ = Number(yel.zSlider.value);

    const table = document.createElement("table");
    table.className = "yolo-summary-table";

    const thead = document.createElement("thead");
    const hr = document.createElement("tr");
    ["z", "dice"].forEach((c) => {
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
