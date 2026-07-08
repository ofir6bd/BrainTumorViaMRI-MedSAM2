const patientSel = document.getElementById("patient");
const zSlider = document.getElementById("z");
const zLabel = document.getElementById("zlabel");
const viewer = document.getElementById("viewer");
const statusEl = document.getElementById("status");
const bestBtn = document.getElementById("best");
const toggle3dBtn = document.getElementById("toggle3d");

const current = { id: null, best: 0, mode: "2d" };

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
  await selectPatient(Number(list[0].id));
}

async function selectPatient(id) {
  current.id = id;
  statusEl.textContent = "Loading patient…";
  const info = await (await fetch(`/api/patient/${id}`)).json();
  current.best = info.best_slice;
  zSlider.max = info.depth - 1;
  zSlider.value = info.best_slice;
  showSlice();
}

function showSlice() {
  current.mode = "2d";
  const z = zSlider.value;
  zLabel.textContent = `z = ${z}`;
  statusEl.textContent = "";
  viewer.src = `/slice.png?id=${current.id}&z=${z}&t=${Date.now()}`;
}

function show3d() {
  current.mode = "3d";
  statusEl.textContent = "Rendering 3D scatter…";
  viewer.onload = () => (statusEl.textContent = "");
  viewer.src = `/scatter.png?id=${current.id}&t=${Date.now()}`;
}

patientSel.addEventListener("change", (e) => selectPatient(Number(e.target.value)));
zSlider.addEventListener("input", showSlice);
bestBtn.addEventListener("click", () => {
  zSlider.value = current.best;
  showSlice();
});
toggle3dBtn.addEventListener("click", () =>
  current.mode === "3d" ? showSlice() : show3d()
);

loadPatients();
