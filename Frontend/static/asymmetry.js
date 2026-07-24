// Asymmetry pipeline viewer.
// Shows every per-slice step image stacked on one page (one after another), each with an
// explanation caption. The whole-tumour (volume) Dice is shown permanently at the top.
// Navigation is per-slice (Prev / Next slice) and per-patient (Skip patient).

const A = {
  meta: null,          // /asym/api/steps payload
  sliceSteps: [],      // per-slice steps in display order
  patients: [],
  patientIdx: 0,
  patient: null,       // { patient_id, depth, slice_indices, best_slice_index, ... }
  sliceIdx: 0,         // pointer into patient.slice_indices
  diceCache: {},       // patientIdx -> volume dice string
  ready: false,
};

const el = {};

async function initAsymmetry() {
  if (A.ready) return;
  A.ready = true;
  el.sel = document.getElementById("asymPatient");
  el.prev = document.getElementById("asymPrevSlice");
  el.nextSlice = document.getElementById("asymSkipSlice");
  el.skipPatient = document.getElementById("asymSkipPatient");
  el.status = document.getElementById("asymStatus");
  el.stack = document.getElementById("asymStack");
  el.volDice = document.getElementById("asymVolDice");
  el.sliceInfo = document.getElementById("asymSliceInfo");

  A.meta = await (await fetch("/asym/api/steps")).json();
  A.sliceSteps = A.meta.steps.filter((s) => s.slice_based);
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
  el.prev.addEventListener("click", () => stepSlice(-1));
  el.nextSlice.addEventListener("click", () => stepSlice(1));
  el.skipPatient.addEventListener("click", () =>
    loadPatient((A.patientIdx + 1) % A.patients.length));

  await loadPatient(0);
}
window.initAsymmetry = initAsymmetry;

async function loadPatient(idx) {
  A.patientIdx = idx;
  el.sel.value = String(idx);
  el.status.textContent = "Loading patient…";
  A.patient = await (await fetch(`/asym/api/patient/${idx}`)).json();
  A.sliceIdx = A.patient.best_slice_index || 0;
  el.status.textContent = "";
  renderStack();          // show slice images first…
  updateDice(idx);        // …then compute the (slower) whole-volume Dice
}

function nSlices() { return A.patient ? A.patient.slice_indices.length : 0; }

function stepSlice(delta) {
  const n = nSlices();
  if (!n) return;
  A.sliceIdx = Math.min(n - 1, Math.max(0, A.sliceIdx + delta));
  renderStack();
}

async function updateDice(idx) {
  if (A.diceCache[idx] !== undefined) {
    el.volDice.textContent = A.diceCache[idx];
    return;
  }
  el.volDice.textContent = "computing…";
  el.volDice.classList.add("computing");
  try {
    const s = await (await fetch(`/asym/api/summary/${idx}`)).json();
    const val = (typeof s.volume_dice === "number") ? s.volume_dice.toFixed(3) : "n/a";
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
    ? `Slice z=${z}  ·  ${A.sliceIdx + 1} / ${n} processed slices`
    : "no processable slices";

  el.stack.innerHTML = "";
  if (!n) return;

  const ts = Date.now();
  A.sliceSteps.forEach((step, i) => {
    const fig = document.createElement("figure");
    fig.className = "asym-fig";

    const h = document.createElement("h3");
    h.className = "asym-fig-title";
    h.textContent = `${i + 1}. ${step.label}`;
    fig.appendChild(h);

    const img = document.createElement("img");
    img.alt = step.label;
    img.src = `/asym/step.png?id=${A.patientIdx}&z=${z}&step=${step.id}&_=${ts}`;
    fig.appendChild(img);

    const cap = document.createElement("figcaption");
    cap.className = "asym-fig-cap";
    cap.textContent = step.explanation || "";
    fig.appendChild(cap);

    el.stack.appendChild(fig);
  });
}
