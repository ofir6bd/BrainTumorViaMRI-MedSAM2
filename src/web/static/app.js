const patientSel = document.getElementById("patient");
const zSlider = document.getElementById("z");
const zLabel = document.getElementById("zlabel");
const sliceField = document.getElementById("sliceField");
const viewer = document.getElementById("viewer");
const statusEl = document.getElementById("status");
const bestBtn = document.getElementById("best");
const viewTabs = document.getElementById("views");

const VIEWS = {
  panels:     { slice: true,  url: (id, z) => `/panels.png?id=${id}&z=${z}` },
  modalities: { slice: true,  url: (id, z) => `/modalities.png?id=${id}&z=${z}` },
  bbox:       { slice: false, url: (id) => `/bbox.png?id=${id}` },
  scatter:    { slice: false, url: (id) => `/scatter.png?id=${id}` },
};

const current = { id: null, best: 0, view: "panels" };

async function loadPatients() {
  const list = await (await fetch("/api/patients")).json();
  patientSel.innerHTML = "";
  if (!list.length) {
    statusEl.textContent =
      "No patients found. Put data in data/dataset/training_data1_v2/ and restart run_web.bat.";
    viewer.removeAttribute("src");
    return;
  }
  for (const p of list) {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = p.label;
    patientSel.appendChild(opt);
  }
  setActiveTab("panels");
  await selectPatient(Number(list[0].id));
}

async function selectPatient(id) {
  current.id = id;
  statusEl.textContent = "Loading patient…";
  const info = await (await fetch(`/api/patient/${id}`)).json();
  current.best = info.best_slice;
  zSlider.max = info.depth - 1;
  zSlider.value = info.best_slice;
  render();
}

function setActiveTab(view) {
  current.view = view;
  for (const btn of viewTabs.querySelectorAll(".tab")) {
    btn.classList.toggle("active", btn.dataset.view === view);
  }
  sliceField.classList.toggle("hidden", !VIEWS[view].slice);
}

function render() {
  if (current.id === null) return;
  const view = VIEWS[current.view];
  const z = zSlider.value;
  zLabel.textContent = `z = ${z}`;
  statusEl.textContent = "Rendering…";
  viewer.onload = () => (statusEl.textContent = "");
  viewer.onerror = () => (statusEl.textContent = "Failed to render this view.");
  viewer.src = `${view.url(current.id, z)}&_=${Date.now()}`;
}

patientSel.addEventListener("change", (e) => selectPatient(Number(e.target.value)));
zSlider.addEventListener("input", render);
bestBtn.addEventListener("click", () => {
  zSlider.value = current.best;
  render();
});
viewTabs.addEventListener("click", (e) => {
  const btn = e.target.closest(".tab");
  if (!btn) return;
  setActiveTab(btn.dataset.view);
  render();
});

loadPatients();
