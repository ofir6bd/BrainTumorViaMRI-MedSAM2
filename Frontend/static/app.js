// App shell: sidebar + "Copy page image". The Analytics tab (dashboard.js) owns the page;
// its whole view lives in the URL hash (#analytics?...).

const copyPageBtn = document.getElementById("copyPageBtn");
const copyPageStatus = document.getElementById("copyPageStatus");

function route() {
  if (window.Dashboard) window.Dashboard.show();
}

document.getElementById("sidenav").addEventListener("click", (e) => {
  const btn = e.target.closest("button.navbtn"); // links (e.g. YOLO_finetune) just navigate
  if (btn && window.Dashboard) location.hash = window.Dashboard.hash();
});
window.addEventListener("hashchange", route);

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
