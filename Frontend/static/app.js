// App shell (sidebar + URL routing) and the Explore tab.
//
// Routes live in the hash so every view is a shareable link:
//   #analytics[?filters]         -> Analytics tab (dashboard.js owns the query part)
//   #explore/<patient_id>[/<z>]  -> Explore tab on that patient / slice

const patientSel = document.getElementById("patient");
const patientSearch = document.getElementById("patientSearch");
const patientCount = document.getElementById("patientCount");
const poolBadge = document.getElementById("poolBadge");
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

const POOL_ORDER = ["yolo_train", "yolo_val", "medsam2_train", "medsam2_val", "test"];

const current = { id: null, best: 0, view: "panels", patients: [], loaded: null };

function poolLabel(pool) {
  return pool || "other";
}

function fillPatientSelect(filter) {
  const q = (filter || "").trim().toLowerCase();
  const shown = current.patients.filter((p) => !q || p.label.toLowerCase().includes(q));
  patientSel.innerHTML = "";
  const groups = {};
  for (const p of shown) (groups[poolLabel(p.pool)] ||= []).push(p);
  const order = [...POOL_ORDER, ...Object.keys(groups).filter((k) => !POOL_ORDER.includes(k))];
  for (const pool of order) {
    if (!groups[pool]) continue;
    const og = document.createElement("optgroup");
    og.label = `${pool} (${groups[pool].length})`;
    for (const p of groups[pool]) {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.label;
      og.appendChild(opt);
    }
    patientSel.appendChild(og);
  }
  patientCount.textContent = q
    ? `${shown.length} of ${current.patients.length} match`
    : `${current.patients.length} patients`;
  if (current.id !== null && shown.some((p) => p.id === current.id)) {
    patientSel.value = String(current.id);
  }
  return shown;
}

function loadPatients() {
  if (!current.loaded) {
    current.loaded = fetch("/api/patients").then((r) => r.json()).then((list) => {
      current.patients = list;
      fillPatientSelect("");
      if (!list.length) {
        statusEl.textContent =
          "No patients found. Put data in data/dataset/<pool>/ (e.g. test/) and restart run_web.bat.";
        viewer.removeAttribute("src");
      }
      return list;
    });
  }
  return current.loaded;
}

async function selectPatient(id, z) {
  current.id = id;
  patientSel.value = String(id);
  const p = current.patients.find((x) => x.id === id);
  poolBadge.textContent = p ? poolLabel(p.pool) : "";
  poolBadge.dataset.pool = p ? p.pool || "" : "";
  statusEl.textContent = "Loading patient…";
  const info = await (await fetch(`/api/patient/${id}`)).json();
  current.best = info.best_slice;
  zSlider.max = info.depth - 1;
  zSlider.value = Number.isInteger(z) ? Math.max(0, Math.min(z, info.depth - 1)) : info.best_slice;
  render();
}

function setActiveTab(view) {
  current.view = view;
  for (const btn of viewTabs.querySelectorAll(".tab")) {
    btn.classList.toggle("active", btn.dataset.view === view);
  }
  sliceField.classList.toggle("hidden", !VIEWS[view].slice);
}

function exploreHash() {
  const p = current.patients.find((x) => x.id === current.id);
  return p ? `#explore/${p.label}/${zSlider.value}` : "#explore";
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
  if (location.hash.startsWith("#explore")) history.replaceState(null, "", exploreHash());
}

patientSel.addEventListener("change", (e) => selectPatient(Number(e.target.value)));
patientSearch.addEventListener("input", () => {
  const shown = fillPatientSelect(patientSearch.value);
  if (shown.length && !shown.some((p) => p.id === current.id)) selectPatient(shown[0].id);
});
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
setActiveTab("panels");

// ---- routing ----
const sidenav = document.getElementById("sidenav");
const panels = {
  analytics: document.getElementById("analyticsPanel"),
  explore: document.getElementById("explorePanel"),
};

function showMode(mode) {
  for (const b of sidenav.querySelectorAll(".navbtn")) {
    b.classList.toggle("active", b.dataset.mode === mode);
  }
  for (const [name, el] of Object.entries(panels)) el.classList.toggle("hidden", name !== mode);
}

async function route() {
  const hash = location.hash || "#analytics";
  if (hash.startsWith("#explore")) {
    showMode("explore");
    const [, pid, z] = hash.split("/");
    const list = await loadPatients();
    if (!list.length) return;
    const target = pid ? list.find((p) => p.label === decodeURIComponent(pid)) : null;
    const zi = z !== undefined && z !== "" ? parseInt(z, 10) : undefined;
    if (target && (target.id !== current.id || (zi !== undefined && zi !== Number(zSlider.value)))) {
      await selectPatient(target.id, zi);
    } else if (current.id === null) {
      await selectPatient(list[0].id);
    }
  } else {
    showMode("analytics");
    if (window.Dashboard) window.Dashboard.show();
  }
}

sidenav.addEventListener("click", (e) => {
  const btn = e.target.closest(".navbtn");
  if (!btn) return;
  if (btn.dataset.mode === "explore") location.hash = exploreHash();
  else location.hash = window.Dashboard ? window.Dashboard.hash() : "#analytics";
});
window.addEventListener("hashchange", route);

// Used by the Analytics tab to jump to a patient.
window.App = {
  openExplore(patientId, z) {
    location.hash = `#explore/${encodeURIComponent(patientId)}${Number.isInteger(z) ? `/${z}` : ""}`;
  },
};

document.addEventListener("keydown", (e) => {
  if (panels.explore.classList.contains("hidden")) return;
  const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement.tagName);
  if (e.key === "/" && !typing) {
    e.preventDefault();
    patientSearch.focus();
  } else if (e.key === "Escape" && document.activeElement === patientSearch) {
    patientSearch.value = "";
    fillPatientSelect("");
    patientSearch.blur();
  } else if ((e.key === "ArrowLeft" || e.key === "ArrowRight") && !typing) {
    const opts = [...patientSel.options];
    const i = opts.findIndex((o) => Number(o.value) === current.id);
    const next = opts[i + (e.key === "ArrowRight" ? 1 : -1)];
    if (next) selectPatient(Number(next.value));
  }
});

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

// dashboard.js loads after this file; route once everything is defined.
window.addEventListener("DOMContentLoaded", route);
