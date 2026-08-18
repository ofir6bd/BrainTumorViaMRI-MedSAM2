// YOLO tumour-segmentation viewer.
// Patient dropdown + slice slider (same controls as Explore), plus a "Slice summary"
// tab listing per-slice Dice (GET /yolo/api/dice). The segmentation image itself already
// shows predicted vs. expert mask overlays and the current slice's Dice score.

const Y = {
  patients: [],
  patientIdx: 0,
  patient: null,      // { patient_id, depth, slice_indices, best_slice_index, min_fg_voxels }
  diceRows: {},        // patientIdx -> rows from /yolo/api/dice
  view: "segment",     // "segment" | "summary"
  ready: false,
};

const yel = {};

async function initYolo() {
  if (Y.ready) return;
  Y.ready = true;
  yel.sel = document.getElementById("yoloPatient");
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

  yel.sel.addEventListener("change", (e) => loadPatient(Number(e.target.value)));
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

  await loadPatient(0);
}
window.initYolo = initYolo;

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
  yel.viewer.src = `/yolo/segment.png?id=${Y.patientIdx}&z=${z}&_=${Date.now()}`;
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
  if (Y.diceRows[idx] !== undefined) return Y.diceRows[idx];
  const resp = await fetch(`/yolo/api/dice?id=${idx}`);
  const payload = await resp.json();
  Y.diceRows[idx] = Array.isArray(payload.rows) ? payload.rows : [];
  return Y.diceRows[idx];
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
