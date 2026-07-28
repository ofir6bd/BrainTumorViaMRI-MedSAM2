// FCM segmentation pipeline viewer.
// Shows every per-slice step image stacked on one page (one after another), each with an
// explanation caption. Whole volume Dice is shown permanently at the top.
// Navigation is per-slice (Prev / Next slice) and per-patient (Skip patient).

const A = {
  meta: null,          // /fcm/api/steps payload
  sliceSteps: [],      // per-slice steps in display order
  patients: [],
  patientIdx: 0,
  patient: null,       // { patient_id, depth, slice_indices, best_slice_index, ... }
  sliceIdx: 0,         // pointer into patient.slice_indices
  diceCache: {},       // patientIdx -> volume dice string
  rowsCache: {},       // patientIdx -> per-slice rows
  view: "pipeline",   // FCM segmentation sub-page view
  ready: false,
};

const el = {};

async function initFcmSegmentation() {
  if (A.ready) return;
  A.ready = true;
  el.sel = document.getElementById("fcmPatient");
  el.prev = document.getElementById("fcmPrevSlice");
  el.nextSlice = document.getElementById("fcmSkipSlice");
  el.skipPatient = document.getElementById("fcmSkipPatient");
  el.status = document.getElementById("fcmStatus");
  el.stack = document.getElementById("fcmStack");
  el.volDice = document.getElementById("fcmVolDice");
  el.sliceInfo = document.getElementById("fcmSliceInfo");
  el.views = document.getElementById("fcmViews");
  el.summaryPanel = document.getElementById("fcmSummaryPanel");
  el.summaryWrap = document.getElementById("fcmSummaryTableWrap");
  el.sliceButtons = document.querySelector("#fcmPanel .fcm-buttons");

  A.meta = await (await fetch("/fcm/api/steps")).json();
  A.sliceSteps = A.meta.steps.filter((s) => s.slice_based);
  A.patients = await (await fetch("/fcm/api/patients")).json();

  el.sel.innerHTML = "";
  if (!A.patients.length) {
    el.status.textContent = "No patients found in data/sample.";
    return;
  }
  for (const p of A.patients) {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = p.label;
    el.sel.appendChild(opt);
  }

  el.sel.addEventListener("change", (e) => loadPatient(Number(e.target.value)));
  el.prev.addEventListener("click", () => stepSlice(-1));
  el.nextSlice.addEventListener("click", () => stepSlice(1));
  el.skipPatient.addEventListener("click", () =>
    loadPatient((A.patientIdx + 1) % A.patients.length));
  el.views.addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (!btn) return;
    setFcmView(btn.dataset.aview);
  });

  await loadPatient(0);
}
window.initFcmSegmentation = initFcmSegmentation;

async function loadPatient(idx) {
  A.patientIdx = idx;
  el.sel.value = String(idx);
  el.status.textContent = "Loading patient…";
  A.patient = await (await fetch(`/fcm/api/patient/${idx}`)).json();
  A.sliceIdx = A.patient.best_slice_index || 0;
  el.status.textContent = "";
  renderStack();          // show slice images first…
  if (A.view === "slice-summary") {
    await renderSummaryTable(idx);
  }
  updateDice(idx);        // …then compute the (slower) whole-volume Dice
}

function nSlices() { return A.patient ? A.patient.slice_indices.length : 0; }

function stepSlice(delta) {
  const n = nSlices();
  if (!n) return;
  A.sliceIdx = Math.min(n - 1, Math.max(0, A.sliceIdx + delta));
  renderStack();
}

function setFcmView(view) {
  if (!view || (view !== "pipeline" && view !== "slice-summary")) return;
  A.view = view;
  for (const btn of el.views.querySelectorAll(".tab")) {
    btn.classList.toggle("active", btn.dataset.aview === view);
  }
  const summary = view === "slice-summary";
  el.summaryPanel.classList.toggle("hidden", !summary);
  el.stack.classList.toggle("hidden", summary);
  if (el.sliceButtons) el.sliceButtons.classList.toggle("hidden", summary);
  if (summary) {
    renderSummaryTable(A.patientIdx);
  }
}

async function updateDice(idx) {
  if (A.diceCache[idx] !== undefined) {
    el.volDice.textContent = A.diceCache[idx];
    return;
  }
  el.volDice.textContent = "computing…";
  el.volDice.classList.add("computing");
  try {
    const s = await (await fetch(`/fcm/api/summary/${idx}`)).json();
    const whole = (typeof s.volume_dice === "number") ? s.volume_dice.toFixed(3) : "n/a";
    const val = `${whole}`;
    A.diceCache[idx] = val;
    if (A.patientIdx === idx) {           // ignore if the user already switched patient
      el.volDice.textContent = val;
      el.volDice.classList.remove("computing");
    }
  } catch (e) {
    if (A.patientIdx === idx) {
      el.volDice.textContent = "error";
      el.volDice.classList.remove("computing");
    }
  }
}

function renderStack() {
  if (!A.patient) return;
  const n = nSlices();
  const zlist = A.patient.slice_indices;
  const z = n ? zlist[Math.min(A.sliceIdx, n - 1)] : 0;

  el.prev.disabled = A.sliceIdx <= 0;
  el.nextSlice.disabled = A.sliceIdx >= n - 1;
  el.sliceInfo.textContent = n
    ? `Slice #${A.sliceIdx + 1}/${n}`
    : "no slices";

  el.stack.innerHTML = "";
  if (!n) return;

  const ts = Date.now();
  A.sliceSteps.forEach((step, i) => {
    const fig = document.createElement("figure");
    fig.className = "fcm-fig";

    const h = document.createElement("h3");
    h.className = "fcm-fig-title";
    h.textContent = `${i + 1}. ${step.label}`;
    fig.appendChild(h);

    const img = document.createElement("img");
    img.alt = step.label;
    img.src = `/fcm/step.png?id=${A.patientIdx}&z=${z}&step=${step.id}&_=${ts}`;
    fig.appendChild(img);

    const cap = document.createElement("figcaption");
    cap.className = "fcm-fig-cap";
    cap.textContent = step.explanation || "";
    fig.appendChild(cap);

    el.stack.appendChild(fig);
  });
}

async function loadSliceRows(idx) {
  if (A.rowsCache[idx] !== undefined) return A.rowsCache[idx];
  const resp = await fetch(`/fcm/api/slices/${idx}`);
  const payload = await resp.json();
  A.rowsCache[idx] = Array.isArray(payload.rows) ? payload.rows : [];
  return A.rowsCache[idx];
}

function _isNum(v) {
  return typeof v === "number" && Number.isFinite(v);
}

function _fmt(v) {
  if (_isNum(v)) return Number.isInteger(v) ? String(v) : v.toFixed(3);
  return "";
}

async function renderSummaryTable(idx) {
  if (!el.summaryWrap) return;
  el.summaryWrap.innerHTML = "";
  el.status.textContent = "Loading slice summary…";
  try {
    const rows = await loadSliceRows(idx);
    if (!rows.length) {
      el.summaryWrap.textContent = "No processed slices for this patient.";
      el.status.textContent = "";
      return;
    }

    const zlist = (A.patient && Array.isArray(A.patient.slice_indices)) ? A.patient.slice_indices : [];
    const zToPos = new Map(zlist.map((zv, i) => [zv, i + 1]));
    const currentZ = zlist.length ? zlist[Math.min(A.sliceIdx, zlist.length - 1)] : null;

    const preferred = ["slice_no", "whole_dice"];
    const extra = Object.keys(rows[0]).filter((k) => !["slice_no", "z", "whole_dice"].includes(k));
    const numericExtras = extra.filter((k) => rows.some((r) => _isNum(r[k])));
    const columns = [...preferred, ...numericExtras];

    const table = document.createElement("table");
    table.className = "fcm-summary-table";

    const thead = document.createElement("thead");
    const hr = document.createElement("tr");
    columns.forEach((c) => {
      const th = document.createElement("th");
      th.textContent = c;
      hr.appendChild(th);
    });
    thead.appendChild(hr);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    let activeRowEl = null;
    rows.forEach((row) => {
      const zVal = _isNum(row.z) ? row.z : Number(row.z);
      const viewRow = { ...row, slice_no: zToPos.get(zVal) ?? "" };
      const tr = document.createElement("tr");
      if (currentZ !== null && zVal === currentZ) {
        tr.classList.add("active-row");
        activeRowEl = tr;
      }
      columns.forEach((c) => {
        const td = document.createElement("td");
        td.textContent = _fmt(viewRow[c]);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    el.summaryWrap.appendChild(table);
    if (activeRowEl) {
      activeRowEl.scrollIntoView({ block: "center", behavior: "auto" });
    }
    el.status.textContent = "";
  } catch (e) {
    el.summaryWrap.textContent = "Failed to load slice summary.";
    el.status.textContent = "";
  }
}
