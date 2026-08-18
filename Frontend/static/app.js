const patientSel = document.getElementById("patient");
const zSlider = document.getElementById("z");
const zLabel = document.getElementById("zlabel");
const sliceField = document.getElementById("sliceField");
const viewer = document.getElementById("viewer");
const statusEl = document.getElementById("status");
const bestBtn = document.getElementById("best");
const viewTabs = document.getElementById("views");
const copyPageBtn = document.getElementById("copyPageBtn");
const copyPageStatus = document.getElementById("copyPageStatus");

const VIEWS = {
  panels:     { slice: true,  url: (id, z) => `/panels.png?id=${id}&z=${z}` },
  modalities: { slice: true,  url: (id, z) => `/modalities.png?id=${id}&z=${z}` },
  bbox:       { slice: false, url: (id) => `/bbox.png?id=${id}` },
  scatter:    { slice: false, url: (id) => `/scatter.png?id=${id}` },
  rgb:        { slice: true,  url: (id, z) => `/rgb.png?id=${id}&z=${z}` },
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

// ---- copy whole page as image ----
let _html2canvasPromise = null;

function setCopyStatus(msg) {
  if (copyPageStatus) copyPageStatus.textContent = msg || "";
}

function ensureHtml2Canvas() {
  if (window.html2canvas) return Promise.resolve(window.html2canvas);
  if (_html2canvasPromise) return _html2canvasPromise;

  _html2canvasPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js";
    script.async = true;
    script.onload = () => resolve(window.html2canvas);
    script.onerror = () => reject(new Error("Failed to load html2canvas"));
    document.head.appendChild(script);
  });

  return _html2canvasPromise;
}

async function copyPageAsImage() {
  if (!copyPageBtn) return;
  copyPageBtn.disabled = true;
  setCopyStatus("Preparing image…");

  try {
    if (!window.isSecureContext) {
      throw new Error("Clipboard image copy requires localhost/HTTPS context.");
    }
    if (!(navigator.clipboard && window.ClipboardItem)) {
      throw new Error("Clipboard image copy is not supported in this browser.");
    }

    const html2canvas = await ensureHtml2Canvas();
    const width = Math.max(document.documentElement.scrollWidth, document.body.scrollWidth, window.innerWidth);
    const height = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight, window.innerHeight);

    const canvas = await html2canvas(document.body, {
      backgroundColor: getComputedStyle(document.body).backgroundColor,
      useCORS: true,
      logging: false,
      scale: window.devicePixelRatio || 1,
      scrollX: 0,
      scrollY: 0,
      width,
      height,
      windowWidth: width,
      windowHeight: height,
    });

    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!blob) throw new Error("Failed to create image blob.");

    await navigator.clipboard.write([
      new ClipboardItem({ "image/png": blob }),
    ]);
    setCopyStatus("Copied. You can paste it now.");
  } catch (err) {
    setCopyStatus(err?.message || "Copy failed.");
  } finally {
    copyPageBtn.disabled = false;
  }
}

if (copyPageBtn) {
  copyPageBtn.addEventListener("click", copyPageAsImage);
}

// ---- sidebar mode switching (Explore / FCM Segmentation / YOLO Detection) ----
const sidenav = document.getElementById("sidenav");
const panels = {
  explore: document.getElementById("explorePanel"),
  "fcm-segmentation": document.getElementById("fcmPanel"),
  "yolo-detection": document.getElementById("yoloPanel"),
};
sidenav.addEventListener("click", (e) => {
  const btn = e.target.closest(".navbtn");
  if (!btn) return;
  const mode = btn.dataset.mode;
  for (const b of sidenav.querySelectorAll(".navbtn")) {
    b.classList.toggle("active", b === btn);
  }
  for (const [name, el] of Object.entries(panels)) {
    el.classList.toggle("hidden", name !== mode);
  }
  if (mode === "fcm-segmentation" && window.initFcmSegmentation) window.initFcmSegmentation();
  if (mode === "yolo-detection" && window.initYolo) window.initYolo();
});
