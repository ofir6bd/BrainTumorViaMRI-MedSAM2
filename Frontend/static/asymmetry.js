// Asymmetry pipeline step-through viewer.
// "Next" walks the whole pipeline one render at a time: steps 1-4 are volume-level
// (Stage A), steps 5-10 repeat per qualifying slice, step 11 is the volume summary.

const A = {
  meta: null,          // { steps, first_slice_step, last_slice_step, summary_step }
  patients: [],
  patientIdx: 0,
  patient: null,       // { patient_id, depth, slice_indices, n_min_voxel }
  sliceIdx: 0,         // pointer into patient.slice_indices
  step: 1,
  z: 0,
  ready: false,
};

const el = {};

async function initAsymmetry() {
  if (A.ready) return;
  A.ready = true;
  el.sel = document.getElementById("asymPatient");
  el.next = document.getElementById("asymNext");
  el.skipSlice = document.getElementById("asymSkipSlice");
  el.skipPatient = document.getElementById("asymSkipPatient");
  el.restart = document.getElementById("asymRestart");
  el.status = document.getElementById("asymStatus");
  el.viewer = document.getElementById("asymViewer");
  el.stepChip = document.getElementById("asymStepChip");
  el.sliceChip = document.getElementById("asymSliceChip");
  el.stepTitle = document.getElementById("asymStepTitle");

  A.meta = await (await fetch("/asym/api/steps")).json();
  A.patients = await (await fetch("/asym/api/patients")).json();

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
  el.next.addEventListener("click", advance);
  el.skipSlice.addEventListener("click", skipSlice);
  el.skipPatient.addEventListener("click", skipPatient);
  el.restart.addEventListener("click", () => { A.sliceIdx = 0; A.step = 1; render(); });

  await loadPatient(0);
}
window.initAsymmetry = initAsymmetry;

async function loadPatient(idx) {
  A.patientIdx = idx;
  el.sel.value = String(idx);
  A.sliceIdx = 0;
  A.step = 1;
  el.status.textContent = "Loading patient…";
  A.patient = await (await fetch(`/asym/api/patient/${idx}`)).json();
  render();
}

const SUMMARY = () => A.meta.summary_step;      // 11
const FIRST_SLICE = () => A.meta.first_slice_step; // 5
const LAST_SLICE = () => A.meta.last_slice_step;   // 10

function nSlices() { return A.patient ? A.patient.slice_indices.length : 0; }

function advance() {
  const n = nSlices();
  const lastStageA = FIRST_SLICE() - 1;   // last volume-level (Stage A) step
  if (A.step <= lastStageA) {
    if (A.step === lastStageA) {
      A.step = n === 0 ? SUMMARY() : FIRST_SLICE();
      A.sliceIdx = 0;
    } else {
      A.step += 1;
    }
  } else if (A.step >= FIRST_SLICE() && A.step < LAST_SLICE()) {
    A.step += 1;
  } else if (A.step === LAST_SLICE()) {
    if (A.sliceIdx < n - 1) { A.sliceIdx += 1; A.step = FIRST_SLICE(); }
    else { A.step = SUMMARY(); }
  }
  // at SUMMARY: stay put
  render();
}

function skipSlice() {
  const n = nSlices();
  const lastStageA = FIRST_SLICE() - 1;
  if (n === 0) { A.step = SUMMARY(); return render(); }
  if (A.step <= lastStageA) { A.sliceIdx = 0; A.step = FIRST_SLICE(); }
  else if (A.sliceIdx < n - 1) { A.sliceIdx += 1; A.step = FIRST_SLICE(); }
  else { A.step = SUMMARY(); }
  render();
}

function skipPatient() {
  const next = (A.patientIdx + 1) % A.patients.length;
  loadPatient(next);
}

function render() {
  if (!A.patient) return;
  const zlist = A.patient.slice_indices;
  A.z = zlist.length ? zlist[Math.min(A.sliceIdx, zlist.length - 1)] : 0;

  const meta = A.meta.steps.find((s) => s.id === A.step) || A.meta.steps[0];
  el.stepChip.textContent = `#${A.step} – ${meta.label}`;
  el.stepTitle.textContent = meta.label;

  if (A.step === SUMMARY()) {
    el.sliceChip.textContent = "Volume summary";
  } else if (A.step >= FIRST_SLICE()) {
    el.sliceChip.textContent = `Slice z=${A.z}  (${A.sliceIdx + 1}/${zlist.length})`;
  } else {
    el.sliceChip.textContent = "Stage A (volume)";
  }

  const atSummary = A.step === SUMMARY();
  el.next.disabled = atSummary;
  el.skipSlice.disabled = atSummary;
  el.status.textContent = atSummary
    ? "Computing volume Dice… (first time may take a few seconds)"
    : "Rendering…";

  el.viewer.onload = () => {
    el.status.textContent = atSummary ? "Pipeline complete." : "";
  };
  el.viewer.onerror = () => (el.status.textContent = "Failed to render this step.");
  el.viewer.src = `/asym/step.png?id=${A.patientIdx}&z=${A.z}&step=${A.step}&_=${Date.now()}`;
}
