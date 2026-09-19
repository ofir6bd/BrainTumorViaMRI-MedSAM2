/* Analytics tab — dataset dashboard over every patient's real segmentation statistics.
 *
 * Data: GET /api/dashboard/data (computed server-side from the NIfTI files, cached in
 * outputs/dashboard/). Nothing here is estimated: a patient without a value for a metric
 * is left out of that chart and counted as such, never filled in.
 *
 * Cross-filtering: every filter (pool, label, side, search, histogram range, scatter
 * box, heatmap cell, shared-subject cell) narrows every chart, except that a chart never
 * filters itself — the histogram ignores its own range, the scatter its own box, and the
 * pool-level charts ignore the pool filter so the other pools stay visible for contrast.
 *
 * All state lives in the URL hash (#analytics?...), so "Copy link" reproduces the view.
 */
(function () {
  const $ = (id) => document.getElementById(id);
  const SVGNS = "http://www.w3.org/2000/svg";

  // Colour = model family (first three Cove slots: the only ones that stay apart when all
  // are on screen together). Train vs val is carried by fill vs ring / stripes, not hue.
  const FAMILY = {
    yolo: { label: "YOLO pools", color: "#3987e5" },
    medsam2: { label: "MedSAM2 pools", color: "#d95926" },
    test: { label: "Test", color: "#199e70" },
    other: { label: "Other", color: "#8a8aa0" },
  };
  const INK = { text: "#e8e8f0", muted: "#9a9ab0", grid: "#2c2c44", base: "#4a4a66",
                surface: "#1a1a2e", neutral: "#8f8fb8" };
  const STATUS = { good: "#0ca30c", critical: "#d03b3b" };
  const LEFT_T = 2 / 3;
  const RIGHT_T = 1 / 3;

  const pct = (v) => (v == null ? "–" : `${(v * 100).toFixed(v * 100 >= 10 ? 0 : 1)}%`);
  const METRICS = [
    { k: "wt", l: "Whole tumour (mL)", get: (r) => r.ml.WT, logDefault: true },
    { k: "tc", l: "Tumour core (mL)", get: (r) => r.ml.TC, logDefault: true },
    { k: "et", l: "Enhancing tumour ET (mL)", get: (r) => r.ml.ET, logDefault: true },
    { k: "netc", l: "Necrotic core NETC (mL)", get: (r) => r.ml.NETC, logDefault: true },
    { k: "snfh", l: "Oedema SNFH (mL)", get: (r) => r.ml.SNFH, logDefault: true },
    { k: "rc", l: "Resection cavity RC (mL)", get: (r) => r.ml.RC, logDefault: true },
    { k: "pct", l: "Tumour % of brain", get: (r) => r.wt_pct_brain, logDefault: true },
    { k: "brain", l: "Brain volume (mL)", get: (r) => r.brain_ml },
    { k: "slices", l: "Tumour slices", get: (r) => r.n_slices },
    { k: "peak", l: "Largest slice area (mm²)", get: (r) => r.peak_area_mm2 },
    { k: "ncc", l: "Connected components", get: (r) => r.n_cc, logDefault: true },
    { k: "lcc", l: "Largest component share", get: (r) => r.largest_cc_frac, fmt: pct },
    { k: "exlr", l: "Extent left–right (mm)", get: (r) => (r.extent_mm ? r.extent_mm[0] : null) },
    { k: "exap", l: "Extent front–back (mm)", get: (r) => (r.extent_mm ? r.extent_mm[1] : null) },
    { k: "exsi", l: "Extent up–down (mm)", get: (r) => (r.extent_mm ? r.extent_mm[2] : null) },
    { k: "left", l: "Share of tumour on left", get: (r) => r.left_frac, fmt: pct },
  ];
  const M = Object.fromEntries(METRICS.map((m) => [m.k, m]));

  const S = {
    loaded: false, loading: null, pollTimer: null,
    records: [], byId: {}, pools: [], poolMeta: {}, labels: [], manifest: null,
    subjectScans: {},
    filters: { pools: new Set(), labels: new Set(), side: "", q: "", range: null, sel: null,
               cell: null, shared: null },
    distMetric: "wt", distLog: true, histMetric: "wt", histLog: true,
    scX: "wt", scY: "et", scXLog: true, scYLog: true, plane: "axial",
    sort: { key: "wt", dir: -1 }, page: 0, pageSize: 50,
    open: null, drawerZ: null, tableRows: [],
    imageKinds: [], mod: "flair", overlay: true,
  };

  // ------------------------------------------------------------------ utilities
  function fmt(v) {
    if (v == null || Number.isNaN(v)) return "–";
    if (Number.isInteger(v)) return v.toLocaleString();
    const a = Math.abs(v);
    if (a >= 1000) return Math.round(v).toLocaleString();
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    return String(+v.toPrecision(2));
  }
  const fmtM = (m, v) => (m.fmt ? m.fmt(v) : fmt(v));
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
  const familyOf = (r) => (S.poolMeta[r.pool] ? S.poolMeta[r.pool].family : "other");
  const roleOf = (r) => (S.poolMeta[r.pool] ? S.poolMeta[r.pool].role : "train");
  const colorOf = (r) => FAMILY[familyOf(r)].color;
  function sideOf(r) {
    if (r.left_frac == null) return null;
    return r.left_frac > LEFT_T ? "left" : r.left_frac < RIGHT_T ? "right" : "bilateral";
  }
  function quantile(sorted, q) {
    if (!sorted.length) return null;
    const pos = (sorted.length - 1) * q;
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }
  function hash01(s) {  // deterministic jitter so dots never jump between renders
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
    return ((h >>> 0) % 10000) / 10000;
  }
  const TR = (log) => (log
    ? { f: (v) => Math.log10(1 + Math.max(0, v)), inv: (t) => Math.pow(10, t) - 1 }
    : { f: (v) => v, inv: (t) => t });
  function scale(d0, d1, r0, r1) {
    const k = (r1 - r0) / (d1 - d0 || 1);
    const s = (v) => r0 + (v - d0) * k;
    s.invert = (p) => d0 + (p - r0) / k;
    return s;
  }
  function niceTicks(lo, hi, n = 5) {
    const span = hi - lo || 1;
    const step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
    const err = span / n / step0;
    const step = step0 * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function rawTicks(tr, log, lo, hi) {  // tick positions in raw units, lo/hi in transformed units
    if (!log) return niceTicks(lo, hi);
    const rMax = tr.inv(hi);
    const rMin = tr.inv(lo);
    // log(1 + x) squeezes everything below 1 against the axis end, so ticks there would
    // collide with 0 — only use them when the whole range is that small.
    const floor = rMax < 3 ? 0.01 : 1;
    const out = [];
    if (rMin <= 0) out.push(0);
    for (let p = -2; Math.pow(10, p) <= rMax * 1.001; p++) {
      for (const m of [1, 3]) {
        const v = m * Math.pow(10, p);
        if (v >= floor && v >= rMin && v <= rMax * 1.001) out.push(+v.toPrecision(3));
      }
    }
    return out.length > 8 ? out.filter((v, i) => v === 0 || i % 2 === 0) : out;
  }
  function el(tag, attrs, parent) {
    const n = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v !== undefined && v !== null) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  }
  function txt(parent, x, y, s, attrs) {
    const t = el("text", { x, y, ...(attrs || {}) }, parent);
    t.textContent = s;
    return t;
  }
  function svgRoot(container, w, h, label) {
    container.innerHTML = "";
    const s = el("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h, class: "db-svg",
                          role: "img", "aria-label": label });
    container.appendChild(s);
    return s;
  }
  function stripes(svg, id, color) {
    const defs = el("defs", {}, svg);
    const p = el("pattern", { id, patternUnits: "userSpaceOnUse", width: 6, height: 6,
                              patternTransform: "rotate(45)" }, defs);
    el("rect", { width: 6, height: 6, fill: color, "fill-opacity": 0.18 }, p);
    el("rect", { width: 2.5, height: 6, fill: color }, p);
    return `url(#${id})`;
  }
  const widthOf = (node) => Math.max(280, node.clientWidth || 600);

  // tooltip
  const tip = $("dbTooltip");
  function showTip(html, evt) {
    tip.innerHTML = html;
    tip.classList.remove("hidden");
    const pad = 14;
    const r = tip.getBoundingClientRect();
    let x = evt.clientX + pad;
    let y = evt.clientY + pad;
    if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  }
  const hideTip = () => tip.classList.add("hidden");
  function hover(node, htmlFn) {
    node.addEventListener("mousemove", (e) => showTip(htmlFn(), e));
    node.addEventListener("mouseleave", hideTip);
  }
  const tipRow = (color, label, value) =>
    `<div class="tt-row"><i style="background:${color}"></i><span>${esc(label)}</span><b>${value}</b></div>`;
  const recTip = (r, extra) =>
    `<div class="tt-title">${esc(r.id)}</div>` +
    tipRow(colorOf(r), r.pool || "other", roleOf(r)) +
    tipRow(INK.neutral, "Whole tumour", `${fmt(r.ml.WT)} mL`) + (extra || "");

  // ------------------------------------------------------------------ filtering
  function passes(r, skip) {
    const f = S.filters;
    if (skip !== "pool" && f.pools.size && !f.pools.has(r.pool)) return false;
    for (const l of f.labels) if (!r.present.includes(l)) return false;
    if (f.side && sideOf(r) !== f.side) return false;
    if (f.q) {
      const q = f.q.toLowerCase();
      if (!r.id.toLowerCase().includes(q)) return false;
    }
    if (skip !== "range" && f.range) {
      const v = M[f.range.m].get(r);
      if (v == null || v < f.range.lo || v > f.range.hi) return false;
    }
    if (skip !== "sel" && f.sel) {
      const x = M[f.sel.x].get(r);
      const y = M[f.sel.y].get(r);
      if (x == null || y == null || x < f.sel.x0 || x > f.sel.x1 || y < f.sel.y0 || y > f.sel.y1) return false;
    }
    if (skip !== "cell" && f.cell) {
      const c = planeXY(r, f.cell.plane);
      if (!c || binOf(c[0]) !== f.cell.i || binOf(c[1]) !== f.cell.j) return false;
    }
    if (f.shared) {
      const set = S.sharedSubjects(f.shared[0], f.shared[1]);
      if (!set.has(r.subject) || (r.pool !== f.shared[0] && r.pool !== f.shared[1])) return false;
    }
    return true;
  }
  const filtered = (skip) => S.records.filter((r) => passes(r, skip));

  // ------------------------------------------------------------------ URL state
  function toHash() {
    const f = S.filters;
    const q = new URLSearchParams();
    if (f.pools.size) q.set("pools", [...f.pools].join(","));
    if (f.labels.size) q.set("labels", [...f.labels].join(","));
    if (f.side) q.set("side", f.side);
    if (f.q) q.set("q", f.q);
    if (f.range) q.set("range", `${f.range.m}:${f.range.lo}:${f.range.hi}`);
    if (f.sel) q.set("sel", [f.sel.x, f.sel.y, f.sel.x0, f.sel.x1, f.sel.y0, f.sel.y1].join(":"));
    if (f.cell) q.set("cell", `${f.cell.plane}:${f.cell.i}:${f.cell.j}`);
    if (f.shared) q.set("shared", f.shared.join(":"));
    q.set("dm", `${S.distMetric}:${+S.distLog}`);
    q.set("hm", `${S.histMetric}:${+S.histLog}`);
    q.set("sc", `${S.scX}:${+S.scXLog}:${S.scY}:${+S.scYLog}`);
    q.set("plane", S.plane);
    q.set("sort", `${S.sort.key}:${S.sort.dir}`);
    if (S.page) q.set("page", S.page);
    if (S.open) q.set("p", S.open);
    if (S.open && S.drawerZ != null) q.set("z", S.drawerZ);
    if (S.mod !== "flair") q.set("img", S.mod);
    if (!S.overlay) q.set("ov", "0");
    return `#analytics?${q.toString()}`;
  }
  function fromHash() {
    const h = location.hash;
    if (!h.startsWith("#analytics")) return;
    const q = new URLSearchParams(h.split("?")[1] || "");
    const f = S.filters;
    const list = (k) => (q.get(k) ? q.get(k).split(",").filter(Boolean) : []);
    f.pools = new Set(list("pools"));
    f.labels = new Set(list("labels"));
    f.side = q.get("side") || "";
    f.q = q.get("q") || "";
    const rng = (q.get("range") || "").split(":");
    f.range = rng.length === 3 && M[rng[0]] ? { m: rng[0], lo: +rng[1], hi: +rng[2] } : null;
    const sel = (q.get("sel") || "").split(":");
    f.sel = sel.length === 6 && M[sel[0]] && M[sel[1]]
      ? { x: sel[0], y: sel[1], x0: +sel[2], x1: +sel[3], y0: +sel[4], y1: +sel[5] } : null;
    const cell = (q.get("cell") || "").split(":");
    f.cell = cell.length === 3 ? { plane: cell[0], i: +cell[1], j: +cell[2] } : null;
    const sh = (q.get("shared") || "").split(":");
    f.shared = sh.length === 2 ? sh : null;
    const pair = (k, a, b) => {
      const v = (q.get(k) || "").split(":");
      if (M[v[0]]) { S[a] = v[0]; S[b] = v[1] === "1"; }
    };
    pair("dm", "distMetric", "distLog");
    pair("hm", "histMetric", "histLog");
    const sc = (q.get("sc") || "").split(":");
    if (sc.length === 4 && M[sc[0]] && M[sc[2]]) {
      S.scX = sc[0]; S.scXLog = sc[1] === "1"; S.scY = sc[2]; S.scYLog = sc[3] === "1";
    }
    if (["axial", "coronal", "sagittal"].includes(q.get("plane"))) S.plane = q.get("plane");
    const so = (q.get("sort") || "").split(":");
    if (so.length === 2) S.sort = { key: so[0], dir: +so[1] || 1 };
    S.page = +q.get("page") || 0;
    S.open = q.get("p") || null;
    S.drawerZ = q.get("z") != null ? +q.get("z") : null;
    S.mod = ["t1c", "t1", "t2", "flair", "sub", "all"].includes(q.get("img")) ? q.get("img") : "flair";
    S.overlay = q.get("ov") !== "0";
  }
  function syncHash() {
    if (!location.hash.startsWith("#analytics") && location.hash) return;
    history.replaceState(null, "", toHash());
  }

  // ------------------------------------------------------------------ loading
  async function poll() {
    clearTimeout(S.pollTimer);
    const st = await (await fetch("/api/dashboard/status")).json();
    const box = $("dbProgress");
    if (st.state === "running") {
      box.classList.remove("hidden");
      const p = st.total ? st.done / st.total : 0;
      $("dbProgressText").textContent =
        `Computing statistics from the NIfTI files: ${st.done} / ${st.total} patients (${Math.round(p * 100)}%)`;
      $("dbProgressFill").style.width = `${p * 100}%`;
      S.pollTimer = setTimeout(poll, 1000);
      return;
    }
    box.classList.add("hidden");
    await loadData();
  }

  async function loadData() {
    const d = await (await fetch("/api/dashboard/data")).json();
    S.records = d.records;
    S.byId = Object.fromEntries(d.records.map((r) => [r.id, r]));
    S.pools = d.pools;
    S.poolMeta = Object.fromEntries(d.pools.map((p) => [p.key, p]));
    S.labels = d.labels;
    S.imageKinds = d.image_kinds;
    S.manifest = d.manifest;
    S.subjectScans = {};
    for (const r of S.records) (S.subjectScans[r.subject] ||= []).push(r);
    const subjByPool = {};
    for (const r of S.records) (subjByPool[r.pool] ||= new Set()).add(r.subject);
    const sharedCache = {};
    S.subjByPool = subjByPool;
    S.sharedSubjects = (a, b) => {
      const key = `${a}|${b}`;
      if (!sharedCache[key]) {
        const A = subjByPool[a] || new Set();
        const B = subjByPool[b] || new Set();
        sharedCache[key] = new Set([...A].filter((s) => B.has(s)));
      }
      return sharedCache[key];
    };
    S.loaded = true;

    const errs = Object.entries(d.errors || {});
    const errBox = $("dbErrors");
    errBox.classList.toggle("hidden", !errs.length);
    errBox.innerHTML = errs.length
      ? `<b>${errs.length} patient(s) could not be read and are left out:</b> ` +
        errs.slice(0, 8).map(([id, e]) => `${esc(id)} (${esc(e)})`).join("; ") +
        (errs.length > 8 ? "; …" : "")
      : "";
    $("dbSub").textContent =
      `${d.records.length.toLocaleString()} of ${d.n_patients.toLocaleString()} patients from ${d.dataset_dir}` +
      (d.computed_at ? ` · statistics computed ${d.computed_at.replace("T", " ")}` : "");
    $("sidefoot").textContent = `${d.dataset_dir} · ${d.n_patients.toLocaleString()} patients`;
    buildControls();
    renderAll();
  }

  // ------------------------------------------------------------------ controls
  function buildControls() {
    for (const id of ["dbDistMetric", "dbHistMetric", "dbScX", "dbScY"]) {
      $(id).innerHTML = METRICS.map((m) => `<option value="${m.k}">${esc(m.l)}</option>`).join("");
    }
    // Pool chips
    $("dbPoolChips").innerHTML = S.pools.map((p) => {
      const c = FAMILY[p.family].color;
      const dot = p.role === "val"
        ? `<i class="chip-dot ring" style="border-color:${c}"></i>`
        : `<i class="chip-dot" style="background:${c}"></i>`;
      return `<button type="button" class="db-chip" data-pool="${p.key}" title="${esc(p.use)}">${dot}${esc(p.key)}</button>`;
    }).join("");
    $("dbLabelChips").innerHTML = S.labels.map((l) =>
      `<button type="button" class="db-chip" data-label="${l.name}"><i class="chip-dot sq" style="background:${l.color}"></i>${l.name}</button>`
    ).join("");
  }

  function syncControls() {
    const f = S.filters;
    for (const b of $("dbPoolChips").querySelectorAll(".db-chip")) {
      b.classList.toggle("on", f.pools.has(b.dataset.pool));
      b.setAttribute("aria-pressed", f.pools.has(b.dataset.pool));
    }
    for (const b of $("dbLabelChips").querySelectorAll(".db-chip")) {
      b.classList.toggle("on", f.labels.has(b.dataset.label));
      b.setAttribute("aria-pressed", f.labels.has(b.dataset.label));
    }
    $("dbSide").value = f.side;
    if (document.activeElement !== $("dbSearch")) $("dbSearch").value = f.q;
    $("dbDistMetric").value = S.distMetric;
    $("dbDistLog").checked = S.distLog;
    $("dbHistMetric").value = S.histMetric;
    $("dbHistLog").checked = S.histLog;
    $("dbScX").value = S.scX;
    $("dbScXLog").checked = S.scXLog;
    $("dbScY").value = S.scY;
    $("dbScYLog").checked = S.scYLog;
    $("dbPlane").value = S.plane;

    // Active (chart-made) filters as removable chips
    const chips = [];
    if (f.range) {
      chips.push(["range", `${M[f.range.m].l}: ${fmtM(M[f.range.m], f.range.lo)} – ${fmtM(M[f.range.m], f.range.hi)}`]);
    }
    if (f.sel) chips.push(["sel", `Scatter box: ${M[f.sel.x].l.split(" (")[0]} × ${M[f.sel.y].l.split(" (")[0]}`]);
    if (f.cell) chips.push(["cell", `Location cell (${f.cell.plane})`]);
    if (f.shared) chips.push(["shared", `Subjects in both ${f.shared[0]} and ${f.shared[1]}`]);
    $("dbActiveChips").innerHTML = chips.map(([k, l]) =>
      `<button type="button" class="db-chip on removable" data-clear="${k}" title="Remove this filter">${esc(l)} <span aria-hidden="true">&times;</span></button>`
    ).join("");
  }

  function update(opts) {
    if (!(opts && opts.keepPage)) S.page = 0;
    renderAll();
  }

  function resetFilters() {
    S.filters = { pools: new Set(), labels: new Set(), side: "", q: "", range: null, sel: null,
                  cell: null, shared: null };
    update();
  }

  // ------------------------------------------------------------------ renderers
  function renderAll() {
    if (!S.loaded) return;
    syncControls();
    const rows = filtered();
    const nSubj = new Set(rows.map((r) => r.subject)).size;
    $("dbCount").textContent =
      `${rows.length.toLocaleString()} of ${S.records.length.toLocaleString()} patients · ${nSubj.toLocaleString()} subjects`;
    renderKpis(rows);
    renderIntegrity();
    renderPoolChart();
    renderTimepoints(rows);
    renderDist();
    renderHist();
    renderLabelShare();
    renderLabelPresence();
    renderScatter();
    renderHeatmap();
    renderZProfile(rows);
    renderTable(rows);
    renderDrawer();
    syncHash();
  }

  function renderKpis(rows) {
    const all = S.records;
    const med = (rs, get) => quantile(rs.map(get).filter((v) => v != null).sort((a, b) => a - b), 0.5);
    const iqr = (rs, get) => {
      const v = rs.map(get).filter((x) => x != null).sort((a, b) => a - b);
      return v.length ? `${fmt(quantile(v, 0.25))}–${fmt(quantile(v, 0.75))}` : "–";
    };
    const share = (rs, test) => (rs.length ? rs.filter(test).length / rs.length : null);
    const isFiltered = rows.length !== all.length;
    const cards = [
      { l: "Patients", v: rows.length.toLocaleString(), s: isFiltered ? `of ${all.length.toLocaleString()}` : "all pools" },
      { l: "Subjects", v: new Set(rows.map((r) => r.subject)).size.toLocaleString(),
        s: `${(rows.length / Math.max(1, new Set(rows.map((r) => r.subject)).size)).toFixed(2)} scans each` },
      { l: "Median whole tumour", v: `${fmt(med(rows, (r) => r.ml.WT))} mL`,
        s: `IQR ${iqr(rows, (r) => r.ml.WT)}${isFiltered ? ` · all ${fmt(med(all, (r) => r.ml.WT))}` : ""}` },
      { l: "Median % of brain", v: `${fmt(med(rows, (r) => r.wt_pct_brain))}%`,
        s: isFiltered ? `all ${fmt(med(all, (r) => r.wt_pct_brain))}%` : `IQR ${iqr(rows, (r) => r.wt_pct_brain)}` },
      { l: "Post-op (has RC)", v: pct(share(rows, (r) => r.present.includes("RC"))),
        s: isFiltered ? `all ${pct(share(all, (r) => r.present.includes("RC")))}` : "resection cavity present" },
      { l: "Median components", v: fmt(med(rows, (r) => r.n_cc)),
        s: `${pct(share(rows, (r) => r.n_cc === 1))} single-piece` },
    ];
    $("dbKpis").innerHTML = cards.map((c) =>
      `<div class="db-kpi"><span class="db-kpi-l">${c.l}</span><span class="db-kpi-v">${c.v}</span><span class="db-kpi-s">${c.s}</span></div>`
    ).join("");
  }

  function renderIntegrity() {
    const pools = S.pools.map((p) => p.key);
    const fam = (k) => S.poolMeta[k].family;
    const mustBeZero = (a, b) => (fam(a) === "yolo" && fam(b) === "medsam2") || (fam(a) === "medsam2" && fam(b) === "yolo");
    // Same model, train vs val: allowed, but a subject on both sides makes val optimistic.
    const sameModel = (a, b) => fam(a) === fam(b) && fam(a) !== "test";
    let html = `<table class="db-leak-table"><thead><tr><th></th>${pools.map((p) => `<th>${esc(p)}</th>`).join("")}</tr></thead><tbody>`;
    let bad = 0;
    const warns = [];
    for (const a of pools) {
      html += `<tr><th>${esc(a)}</th>`;
      for (const b of pools) {
        const n = a === b ? (S.subjByPool[a] || new Set()).size : S.sharedSubjects(a, b).size;
        let cls = "neutral";
        let mark = "";
        if (a === b) cls = "diag";
        else if (mustBeZero(a, b)) {
          cls = n ? "bad" : "good";
          mark = n ? "✕ " : "✓ ";
          if (n) bad++;
        } else if (sameModel(a, b) && n) {
          cls = "warn";
          mark = "⚠ ";
          if (a < b) warns.push(`${a} ↔ ${b}: ${n}`);
        }
        const active = S.filters.shared && S.filters.shared[0] === a && S.filters.shared[1] === b;
        html += `<td class="leak-${cls}${active ? " active" : ""}" data-a="${a}" data-b="${b}" ${a !== b && n ? 'tabindex="0"' : ""}>${mark}${n}</td>`;
      }
      html += "</tr>";
    }
    html += "</tbody></table>";
    const verdict = bad
      ? `<p class="db-verdict bad">✕ ${bad / 2} YOLO↔MedSAM2 pool pair(s) share subjects — MedSAM2 would train on prompts YOLO has memorised.</p>`
      : `<p class="db-verdict good">✓ No subject is in both a YOLO pool and a MedSAM2 pool.</p>`;
    const warnLine = warns.length
      ? `<p class="db-verdict warn">⚠ Same person in a model's train and val pool (${warns.join(", ")} subjects) — that model's val score will be optimistic.</p>`
      : "";
    $("dbLeakMatrix").innerHTML = html + verdict + warnLine +
      `<p class="db-card-sub">Diagonal = subjects in the pool. Click a shared-subject cell to list those patients.</p>`;
    for (const td of $("dbLeakMatrix").querySelectorAll("td[tabindex]")) {
      const a = td.dataset.a;
      const b = td.dataset.b;
      const pick = () => {
        const cur = S.filters.shared;
        S.filters.shared = cur && cur[0] === a && cur[1] === b ? null : [a, b];
        update();
      };
      td.addEventListener("click", pick);
      td.addEventListener("keydown", (e) => { if (e.key === "Enter") pick(); });
      hover(td, () => {
        const subs = [...S.sharedSubjects(a, b)].sort();
        return `<div class="tt-title">${esc(a)} ∩ ${esc(b)}</div>${subs.length} shared subjects<br><span class="tt-muted">${esc(subs.slice(0, 6).join(", "))}${subs.length > 6 ? ", …" : ""}</span>`;
      });
    }

    const m = S.manifest;
    if (!m || !m.found) {
      $("dbManifest").innerHTML = `<p class="db-verdict bad">✕ No split_manifest.json found next to the pools.</p>`;
      return;
    }
    const diskCounts = {};
    for (const r of S.records) diskCounts[r.pool] = (diskCounts[r.pool] || 0) + 1;
    const rows = pools.map((p) => {
      const man = m.counts ? m.counts[p] : undefined;
      const ok = man === diskCounts[p];
      return `<tr><td>${esc(p)}</td><td>${man ?? "–"}</td><td>${diskCounts[p] || 0}</td><td class="${ok ? "db-ok" : "db-bad"}">${ok ? "✓" : "✕"}</td></tr>`;
    }).join("");
    const drift = m.moved.length + m.not_in_manifest.length + m.missing_on_disk.length;
    $("dbManifest").innerHTML =
      `<h4>Manifest vs disk</h4>` +
      `<table class="db-mini-table"><thead><tr><th>Pool</th><th>Manifest</th><th>On disk</th><th></th></tr></thead><tbody>${rows}</tbody></table>` +
      (drift
        ? `<p class="db-verdict bad">✕ ${m.moved.length} moved, ${m.not_in_manifest.length} not in manifest, ${m.missing_on_disk.length} missing on disk.` +
          (m.moved.length ? `<br><span class="tt-muted">${m.moved.slice(0, 4).map((x) => `${esc(x.id)}: ${esc(x.manifest)} → ${esc(x.disk)}`).join("; ")}</span>` : "") + `</p>`
        : `<p class="db-verdict good">✓ Every folder is where split_manifest.json says it is.</p>`) +
      (m.note ? `<p class="db-card-sub db-note">${esc(m.note)}</p>` : "");
  }

  function renderPoolChart() {
    const box = $("dbPoolChart");
    const W = widthOf(box);
    const rowH = 34;
    const H = S.pools.length * rowH + 10;
    const svg = svgRoot(box, W, H, "Patients per pool");
    const rows = filtered("pool");
    const L = 118;
    const R = 118;
    const full = {};
    const cur = {};
    const subj = {};
    for (const r of S.records) full[r.pool] = (full[r.pool] || 0) + 1;
    for (const r of rows) {
      cur[r.pool] = (cur[r.pool] || 0) + 1;
      (subj[r.pool] ||= new Set()).add(r.subject);
    }
    const max = Math.max(1, ...Object.values(full));
    const x = scale(0, max, L, W - R);
    const sel = S.filters.pools;
    S.pools.forEach((p, i) => {
      const y = 6 + i * rowH;
      const c = FAMILY[p.family].color;
      const fill = p.role === "val" ? stripes(svg, `pst-${p.key}`, c) : c;
      const g = el("g", { class: "db-clickable", opacity: sel.size && !sel.has(p.key) ? 0.35 : 1 }, svg);
      txt(g, L - 10, y + rowH / 2, p.key, { class: "db-lbl", "text-anchor": "end", "dominant-baseline": "middle" });
      el("rect", { x: L, y: y + 6, width: Math.max(0, x(full[p.key] || 0) - L), height: rowH - 14, rx: 3,
                   fill: c, "fill-opacity": 0.16 }, g);
      el("rect", { x: L, y: y + 6, width: Math.max(0, x(cur[p.key] || 0) - L), height: rowH - 14, rx: 3,
                   fill, stroke: p.role === "val" ? c : "none" }, g);
      if (sel.has(p.key)) el("rect", { x: L - 3, y: y + 3, width: x(full[p.key]) - L + 6, height: rowH - 8, rx: 5,
                                        fill: "none", stroke: INK.text, "stroke-width": 1.5 }, g);
      txt(g, W - R + 8, y + rowH / 2,
        `${(cur[p.key] || 0).toLocaleString()} · ${(subj[p.key] || new Set()).size} subj`,
        { class: "db-val", "dominant-baseline": "middle" });
      el("rect", { x: 0, y, width: W, height: rowH, fill: "transparent" }, g);
      g.addEventListener("click", () => {
        if (sel.has(p.key)) sel.delete(p.key); else sel.add(p.key);
        update();
      });
      hover(g, () => `<div class="tt-title">${esc(p.key)}</div>${esc(p.use)}<br>` +
        tipRow(c, "In current filter", (cur[p.key] || 0).toLocaleString()) +
        tipRow(INK.muted, "Whole pool", (full[p.key] || 0).toLocaleString()) +
        tipRow(INK.muted, "Subjects (filter)", (subj[p.key] || new Set()).size));
    });
  }

  function renderTimepoints(rows) {
    const box = $("dbTimepointChart");
    const W = widthOf(box);
    const H = 190;
    const svg = svgRoot(box, W, H, "Subjects by number of scans");
    const subs = new Set(rows.map((r) => r.subject));
    const counts = {};
    for (const s of subs) {
      const k = S.subjectScans[s].length;
      counts[k] = (counts[k] || 0) + 1;
    }
    const ks = Object.keys(counts).map(Number).sort((a, b) => a - b);
    if (!ks.length) { txt(svg, W / 2, H / 2, "No patients in filter", { class: "db-empty", "text-anchor": "middle" }); return; }
    const maxK = Math.max(...ks);
    const cats = Array.from({ length: maxK }, (_, i) => i + 1);
    const L = 44;
    const B = 34;
    const T = 16;
    const maxV = Math.max(...Object.values(counts));
    const y = scale(0, maxV, H - B, T);
    const bw = Math.min(56, (W - L - 10) / cats.length - 8);
    const x = (i) => L + 10 + i * ((W - L - 10) / cats.length) + ((W - L - 10) / cats.length - bw) / 2;
    for (const t of niceTicks(0, maxV, 4)) {
      el("line", { x1: L, x2: W - 4, y1: y(t), y2: y(t), class: "db-gridline" }, svg);
      txt(svg, L - 6, y(t), fmt(t), { class: "db-tick", "text-anchor": "end", "dominant-baseline": "middle" });
    }
    cats.forEach((k, i) => {
      const v = counts[k] || 0;
      const h = H - B - y(v);
      const g = el("g", {}, svg);
      if (v) el("path", { d: roundTop(x(i), y(v), bw, h), fill: INK.neutral }, g);
      txt(g, x(i) + bw / 2, y(v) - 5, v, { class: "db-val", "text-anchor": "middle" });
      txt(g, x(i) + bw / 2, H - B + 16, k, { class: "db-tick", "text-anchor": "middle" });
      el("rect", { x: x(i) - 4, y: T, width: bw + 8, height: H - B - T, fill: "transparent" }, g);
      hover(g, () => `<div class="tt-title">${k} scan${k > 1 ? "s" : ""} per subject</div>${v} subjects in the current filter`);
    });
    txt(svg, L + (W - L) / 2, H - 4, "scans per subject (whole dataset)", { class: "db-axis-title", "text-anchor": "middle" });
  }

  function roundTop(x, y, w, h) {  // bar with 4px rounded data-end, square baseline
    const r = Math.min(4, w / 2, h);
    return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
  }
  function roundRight(x, y, w, h) {
    const r = Math.min(4, h / 2, w);
    return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`;
  }

  // Two-sample Kolmogorov–Smirnov (asymptotic p-value, Numerical Recipes' Q_KS).
  function ks(a, b) {
    let i = 0;
    let j = 0;
    let d = 0;
    while (i < a.length && j < b.length) {
      const v = Math.min(a[i], b[j]);
      while (i < a.length && a[i] <= v) i++;
      while (j < b.length && b[j] <= v) j++;
      d = Math.max(d, Math.abs(i / a.length - j / b.length));
    }
    const en = Math.sqrt((a.length * b.length) / (a.length + b.length));
    const lam = (en + 0.12 + 0.11 / en) * d;
    let p = 0;
    let prev = 0;
    for (let k = 1; k <= 100; k++) {
      const term = 2 * (k % 2 ? 1 : -1) * Math.exp(-2 * k * k * lam * lam);
      p += term;
      if (Math.abs(term) <= 1e-3 * Math.abs(prev) || Math.abs(term) <= 1e-8 * p) return { d, p: Math.min(1, Math.max(0, p)) };
      prev = term;
    }
    return { d, p: 1 };
  }

  function renderDist() {
    const box = $("dbDistChart");
    const m = M[S.distMetric];
    const tr = TR(S.distLog);
    const rows = filtered("pool");
    const W = widthOf(box);
    const H = 300;
    const L = 58;
    const B = 34;
    const T = 12;
    const svg = svgRoot(box, W, H, `${m.l} by pool`);
    const byPool = {};
    let missing = 0;
    for (const r of rows) {
      const v = m.get(r);
      if (v == null) { missing++; continue; }
      (byPool[r.pool] ||= []).push({ r, v, t: tr.f(v) });
    }
    const allT = Object.values(byPool).flat().map((d) => d.t);
    if (!allT.length) { txt(svg, W / 2, H / 2, "No values in filter", { class: "db-empty", "text-anchor": "middle" }); $("dbKsTable").innerHTML = ""; return; }
    const lo = Math.min(...allT);
    const hi = Math.max(...allT);
    const pad = (hi - lo) * 0.04 || 1;
    const y = scale(lo - pad, hi + pad, H - B, T);
    for (const t of rawTicks(tr, S.distLog, lo - pad, hi + pad)) {
      const yy = y(tr.f(t));
      if (yy < T - 1 || yy > H - B + 1) continue;
      el("line", { x1: L, x2: W - 6, y1: yy, y2: yy, class: "db-gridline" }, svg);
      txt(svg, L - 6, yy, fmtM(m, t), { class: "db-tick", "text-anchor": "end", "dominant-baseline": "middle" });
    }
    const slot = (W - L - 6) / S.pools.length;
    const stats = {};
    S.pools.forEach((p, i) => {
      const d = (byPool[p.key] || []).slice().sort((a, b) => a.v - b.v);
      const cx = L + slot * i + slot / 2;
      const c = FAMILY[p.family].color;
      txt(svg, cx, H - B + 16, p.key, { class: "db-tick", "text-anchor": "middle" });
      txt(svg, cx, H - B + 29, `n=${d.length}`, { class: "db-tick dim", "text-anchor": "middle" });
      if (!d.length) return;
      const ts = d.map((x) => x.t);
      const q1 = quantile(ts, 0.25);
      const q2 = quantile(ts, 0.5);
      const q3 = quantile(ts, 0.75);
      const iqr = q3 - q1;
      const wLo = ts.find((t) => t >= q1 - 1.5 * iqr);
      const wHi = [...ts].reverse().find((t) => t <= q3 + 1.5 * iqr);
      stats[p.key] = { vals: d.map((x) => x.v), median: tr.inv(q2) };
      const bw = Math.min(46, slot * 0.36);
      // dots (behind the box), jittered deterministically within the slot
      const dots = el("g", {}, svg);
      for (const x of d) {
        const jx = cx + (hash01(x.r.id) - 0.5) * slot * 0.78;
        const open = S.open === x.r.id;
        const dot = el("circle", {
          cx: jx, cy: y(x.t), r: open ? 5 : 2.6,
          fill: p.role === "val" ? "none" : c, stroke: open ? INK.text : c,
          "stroke-width": open ? 2 : p.role === "val" ? 1.2 : 0, "fill-opacity": 0.55, class: "db-dot",
        }, dots);
        hover(dot, () => recTip(x.r, tipRow(INK.neutral, m.l, fmtM(m, x.v))));
        dot.addEventListener("click", () => openPatient(x.r.id));
      }
      el("line", { x1: cx, x2: cx, y1: y(wLo), y2: y(q1), stroke: INK.text, "stroke-width": 1.2 }, svg);
      el("line", { x1: cx, x2: cx, y1: y(q3), y2: y(wHi), stroke: INK.text, "stroke-width": 1.2 }, svg);
      el("rect", { x: cx - bw / 2, y: y(q3), width: bw, height: Math.max(1, y(q1) - y(q3)), rx: 3,
                   fill: INK.surface, "fill-opacity": 0.55, stroke: INK.text, "stroke-width": 1.2 }, svg);
      el("line", { x1: cx - bw / 2, x2: cx + bw / 2, y1: y(q2), y2: y(q2), stroke: c, "stroke-width": 3 }, svg);
      const hit = el("rect", { x: cx - bw / 2, y: y(q3), width: bw, height: Math.max(4, y(q1) - y(q3)), fill: "transparent" }, svg);
      hover(hit, () => `<div class="tt-title">${esc(p.key)} — ${esc(m.l)}</div>` +
        tipRow(c, "Median", fmtM(m, tr.inv(q2))) +
        tipRow(INK.muted, "Middle 50%", `${fmtM(m, tr.inv(q1))} – ${fmtM(m, tr.inv(q3))}`) +
        tipRow(INK.muted, "Patients", d.length));
    });
    txt(svg, 14, T + (H - B - T) / 2, `${m.l}${S.distLog ? " (log)" : ""}`,
      { class: "db-axis-title", "text-anchor": "middle", transform: `rotate(-90 14 ${T + (H - B - T) / 2})` });

    // KS table vs test
    const ref = stats.test;
    const alpha = 0.05 / Math.max(1, S.pools.length - 1);
    let t = `<table class="db-mini-table"><thead><tr><th>Pool</th><th>Median</th><th title="Largest gap between the two cumulative distributions">KS D</th><th>p</th></tr></thead><tbody>`;
    for (const p of S.pools) {
      const s = stats[p.key];
      if (!s) { t += `<tr><td>${esc(p.key)}</td><td colspan="3">no values</td></tr>`; continue; }
      if (p.key === "test") { t += `<tr><td>${esc(p.key)}</td><td>${fmtM(m, s.median)}</td><td colspan="2" class="dim">reference</td></tr>`; continue; }
      if (!ref) { t += `<tr><td>${esc(p.key)}</td><td>${fmtM(m, s.median)}</td><td colspan="2" class="dim">no test values</td></tr>`; continue; }
      const r = ks(s.vals.slice().sort((a, b) => a - b), ref.vals.slice().sort((a, b) => a - b));
      const sig = r.p < alpha;
      t += `<tr><td>${esc(p.key)}</td><td>${fmtM(m, s.median)}</td><td>${r.d.toFixed(3)}</td><td class="${sig ? "db-bad" : "db-ok"}">${sig ? "✕ " : "✓ "}${r.p < 0.001 ? "<0.001" : r.p.toFixed(3)}</td></tr>`;
    }
    t += `</tbody></table><p class="db-card-sub">✓ = no significant difference from test (p ≥ ${alpha.toFixed(4)}, Bonferroni).` +
      `${missing ? ` ${missing} patient(s) without a value are left out.` : ""}</p>`;
    $("dbKsTable").innerHTML = t;
  }

  function renderHist() {
    const box = $("dbHistChart");
    const m = M[S.histMetric];
    const tr = TR(S.histLog);
    const rows = filtered("range");
    const W = widthOf(box);
    const H = 230;
    const L = 52;
    const B = 40;
    const T = 14;
    const svg = svgRoot(box, W, H, `Histogram of ${m.l}`);
    const vals = rows.map((r) => ({ r, v: m.get(r) })).filter((d) => d.v != null);
    const allT = S.records.map((r) => m.get(r)).filter((v) => v != null).map(tr.f);
    if (!allT.length) return;
    const lo = Math.min(...allT);
    const hi = Math.max(...allT) || 1;
    const NB = 40;
    const bw = (hi - lo) / NB || 1;
    const fams = ["yolo", "medsam2", "test", "other"];
    const bins = Array.from({ length: NB }, () => ({ yolo: 0, medsam2: 0, test: 0, other: 0, n: 0 }));
    for (const d of vals) {
      const b = Math.min(NB - 1, Math.max(0, Math.floor((tr.f(d.v) - lo) / bw)));
      bins[b][familyOf(d.r)]++;
      bins[b].n++;
    }
    const maxN = Math.max(1, ...bins.map((b) => b.n));
    const x = scale(lo, lo + bw * NB, L, W - 8);
    const y = scale(0, maxN, H - B, T);
    for (const t of niceTicks(0, maxN, 4)) {
      el("line", { x1: L, x2: W - 8, y1: y(t), y2: y(t), class: "db-gridline" }, svg);
      txt(svg, L - 6, y(t), fmt(t), { class: "db-tick", "text-anchor": "end", "dominant-baseline": "middle" });
    }
    for (const t of rawTicks(tr, S.histLog, lo, lo + bw * NB)) {
      const xx = x(tr.f(t));
      if (xx < L - 1 || xx > W - 7) continue;
      el("line", { x1: xx, x2: xx, y1: H - B, y2: H - B + 4, stroke: INK.base }, svg);
      txt(svg, xx, H - B + 16, fmtM(m, t), { class: "db-tick", "text-anchor": "middle" });
    }
    el("line", { x1: L, x2: W - 8, y1: H - B, y2: H - B, stroke: INK.base }, svg);
    const gap = 1;
    bins.forEach((b, i) => {
      let base = H - B;
      const x0 = x(lo + i * bw) + gap;
      const w = Math.max(1, x(lo + (i + 1) * bw) - x(lo + i * bw) - 2 * gap);
      fams.forEach((f, k) => {
        if (!b[f]) return;
        const h = (H - B) - y(b[f]);
        const top = fams.slice(k + 1).every((g) => !b[g]);
        el(top ? "path" : "rect", top
          ? { d: roundTop(x0, base - h, w, h), fill: FAMILY[f].color }
          : { x: x0, y: base - h, width: w, height: Math.max(0, h - 1), fill: FAMILY[f].color }, svg);
        base -= h;
      });
    });
    txt(svg, L + (W - L) / 2, H - 6, `${m.l}${S.histLog ? " (log)" : ""}`, { class: "db-axis-title", "text-anchor": "middle" });

    // current range + brush
    const rng = S.filters.range && S.filters.range.m === S.histMetric ? S.filters.range : null;
    if (rng) {
      const a = x(tr.f(rng.lo));
      const b = x(tr.f(rng.hi));
      el("rect", { x: L, y: T, width: Math.max(0, a - L), height: H - B - T, fill: INK.surface, "fill-opacity": 0.72 }, svg);
      el("rect", { x: b, y: T, width: Math.max(0, W - 8 - b), height: H - B - T, fill: INK.surface, "fill-opacity": 0.72 }, svg);
      el("rect", { x: a, y: T, width: Math.max(1, b - a), height: H - B - T, fill: "none", stroke: INK.text, "stroke-dasharray": "4 3" }, svg);
    }
    const overlay = el("rect", { x: L, y: T, width: W - 8 - L, height: H - B - T, fill: "transparent", class: "db-brushable" }, svg);
    const brush = el("rect", { y: T, height: H - B - T, fill: INK.text, "fill-opacity": 0.12, stroke: INK.text, visibility: "hidden" }, svg);
    const toX = (e) => {
      const r = svg.getBoundingClientRect();
      return Math.max(L, Math.min(W - 8, ((e.clientX - r.left) / r.width) * W));
    };
    let start = null;
    overlay.addEventListener("pointerdown", (e) => {
      start = toX(e);
      overlay.setPointerCapture(e.pointerId);
      brush.setAttribute("visibility", "visible");
      brush.setAttribute("x", start);
      brush.setAttribute("width", 0);
    });
    overlay.addEventListener("pointermove", (e) => {
      const px = toX(e);
      if (start !== null) {
        brush.setAttribute("x", Math.min(start, px));
        brush.setAttribute("width", Math.abs(px - start));
        hideTip();
        return;
      }
      const i = Math.min(NB - 1, Math.max(0, Math.floor((x.invert(px) - lo) / bw)));
      const b = bins[i];
      showTip(`<div class="tt-title">${fmtM(m, tr.inv(lo + i * bw))} – ${fmtM(m, tr.inv(lo + (i + 1) * bw))}</div>` +
        ["yolo", "medsam2", "test"].map((f) => tipRow(FAMILY[f].color, FAMILY[f].label, b[f])).join("") +
        tipRow(INK.muted, "Total", b.n), e);
    });
    overlay.addEventListener("mouseleave", hideTip);
    overlay.addEventListener("pointerup", (e) => {
      if (start === null) return;
      const a = start;
      const b = toX(e);
      start = null;
      brush.setAttribute("visibility", "hidden");
      if (Math.abs(b - a) < 4) return;
      const v0 = tr.inv(x.invert(Math.min(a, b)));
      const v1 = tr.inv(x.invert(Math.max(a, b)));
      S.filters.range = { m: S.histMetric, lo: +v0.toPrecision(4), hi: +v1.toPrecision(4) };
      update();
    });
    overlay.addEventListener("dblclick", () => { S.filters.range = null; update(); });
  }

  function renderLabelShare() {
    const box = $("dbLabelShare");
    const rows = filtered("pool");
    const W = widthOf(box);
    const rowH = 30;
    const T = 26;
    const H = T + S.pools.length * rowH + 6;
    const svg = svgRoot(box, W, H, "Label share of whole tumour by pool");
    const L = 110;
    const R = 10;
    const x = scale(0, 1, L, W - R);
    // legend
    let lx = L;
    for (const l of S.labels) {
      el("rect", { x: lx, y: 4, width: 10, height: 10, rx: 2, fill: l.color }, svg);
      txt(svg, lx + 14, 13, l.name, { class: "db-tick" });
      lx += 62;
    }
    S.pools.forEach((p, i) => {
      const rs = rows.filter((r) => r.pool === p.key && r.ml.WT > 0);
      const y0 = T + i * rowH;
      txt(svg, L - 10, y0 + rowH / 2, p.key, { class: "db-lbl", "text-anchor": "end", "dominant-baseline": "middle" });
      if (!rs.length) { txt(svg, L, y0 + rowH / 2, "no patients", { class: "db-tick dim", "dominant-baseline": "middle" }); return; }
      const shares = S.labels.map((l) => rs.reduce((a, r) => a + r.ml[l.name] / r.ml.WT, 0) / rs.length);
      const tot = shares.reduce((a, b) => a + b, 0) || 1;
      let acc = 0;
      S.labels.forEach((l, k) => {
        const s = shares[k] / tot;
        const xa = x(acc);
        const w = Math.max(0, x(acc + s) - xa - (k < S.labels.length - 1 ? 2 : 0));
        const g = el("g", {}, svg);
        el(k === S.labels.length - 1 ? "path" : "rect", k === S.labels.length - 1
          ? { d: roundRight(xa, y0 + 5, w, rowH - 10), fill: l.color }
          : { x: xa, y: y0 + 5, width: w, height: rowH - 10, fill: l.color }, g);
        if (s >= 0.09) txt(g, xa + w / 2, y0 + rowH / 2, pct(s), { class: "db-inbar", "text-anchor": "middle", "dominant-baseline": "middle" });
        hover(g, () => `<div class="tt-title">${esc(p.key)} — ${l.name}</div>` +
          tipRow(l.color, "Mean share of WT", pct(s)) + tipRow(INK.muted, "Patients", rs.length));
        acc += s;
      });
    });
  }

  function renderLabelPresence() {
    const rows = filtered("pool");
    let html = `<table class="db-presence"><thead><tr><th></th>${S.labels.map((l) => `<th><i class="chip-dot sq" style="background:${l.color}"></i>${l.name}</th>`).join("")}</tr></thead><tbody>`;
    for (const p of S.pools) {
      const rs = rows.filter((r) => r.pool === p.key);
      html += `<tr><th>${esc(p.key)}</th>`;
      for (const l of S.labels) {
        const s = rs.length ? rs.filter((r) => r.present.includes(l.name)).length / rs.length : null;
        html += `<td title="${esc(p.key)}: ${s == null ? "no patients" : `${pct(s)} have ${l.name}`}"><div class="pres-bar"><span style="width:${(s || 0) * 100}%;background:${l.color}"></span></div><b>${pct(s)}</b></td>`;
      }
      html += "</tr>";
    }
    $("dbLabelPresence").innerHTML = html + "</tbody></table>";
  }

  function renderScatter() {
    const box = $("dbScatter");
    const mx = M[S.scX];
    const my = M[S.scY];
    const tx = TR(S.scXLog);
    const ty = TR(S.scYLog);
    const rows = filtered("sel");
    const W = widthOf(box);
    const H = 380;
    const L = 60;
    const B = 44;
    const T = 12;
    const R = 12;
    const svg = svgRoot(box, W, H, `${my.l} against ${mx.l}`);
    const pts = [];
    let missing = 0;
    for (const r of rows) {
      const a = mx.get(r);
      const b = my.get(r);
      if (a == null || b == null) { missing++; continue; }
      pts.push({ r, a, b, x: tx.f(a), y: ty.f(b) });
    }
    const allX = S.records.map((r) => mx.get(r)).filter((v) => v != null).map(tx.f);
    const allY = S.records.map((r) => my.get(r)).filter((v) => v != null).map(ty.f);
    const x0 = Math.min(...allX);
    const x1 = Math.max(...allX);
    const y0 = Math.min(...allY);
    const y1 = Math.max(...allY);
    const px = (x1 - x0) * 0.03 || 1;
    const py = (y1 - y0) * 0.04 || 1;
    const x = scale(x0 - px, x1 + px, L, W - R);
    const y = scale(y0 - py, y1 + py, H - B, T);
    for (const t of rawTicks(ty, S.scYLog, y0 - py, y1 + py)) {
      const yy = y(ty.f(t));
      if (yy < T - 1 || yy > H - B + 1) continue;
      el("line", { x1: L, x2: W - R, y1: yy, y2: yy, class: "db-gridline" }, svg);
      txt(svg, L - 6, yy, fmtM(my, t), { class: "db-tick", "text-anchor": "end", "dominant-baseline": "middle" });
    }
    for (const t of rawTicks(tx, S.scXLog, x0 - px, x1 + px)) {
      const xx = x(tx.f(t));
      if (xx < L - 1 || xx > W - R + 1) continue;
      el("line", { x1: xx, x2: xx, y1: T, y2: H - B, class: "db-gridline" }, svg);
      txt(svg, xx, H - B + 16, fmtM(mx, t), { class: "db-tick", "text-anchor": "middle" });
    }
    txt(svg, L + (W - L - R) / 2, H - 8, `${mx.l}${S.scXLog ? " (log)" : ""}`, { class: "db-axis-title", "text-anchor": "middle" });
    txt(svg, 14, T + (H - B - T) / 2, `${my.l}${S.scYLog ? " (log)" : ""}`,
      { class: "db-axis-title", "text-anchor": "middle", transform: `rotate(-90 14 ${T + (H - B - T) / 2})` });

    // current selection box (only drawn when it is on these axes)
    const sel = S.filters.sel;
    if (sel && sel.x === S.scX && sel.y === S.scY) {
      const ax = x(tx.f(sel.x0));
      const bx = x(tx.f(sel.x1));
      const ay = y(ty.f(sel.y1));
      const by = y(ty.f(sel.y0));
      el("rect", { x: ax, y: ay, width: Math.max(1, bx - ax), height: Math.max(1, by - ay),
                   fill: INK.text, "fill-opacity": 0.06, stroke: INK.text, "stroke-dasharray": "4 3" }, svg);
    }
    const overlay = el("rect", { x: L, y: T, width: W - R - L, height: H - B - T, fill: "transparent", class: "db-brushable" }, svg);
    const dots = el("g", {}, svg);
    // draw test last so the reference set stays on top
    const order = { medsam2: 0, yolo: 1, other: 2, test: 3 };
    pts.sort((a, b) => order[familyOf(a.r)] - order[familyOf(b.r)]);
    for (const p of pts) {
      const c = colorOf(p.r);
      const val = roleOf(p.r) === "val";
      const open = S.open === p.r.id;
      const dot = el("circle", {
        cx: x(p.x), cy: y(p.y), r: open ? 6 : 3.4,
        fill: val ? "none" : c, "fill-opacity": 0.7, stroke: open ? INK.text : val ? c : INK.surface,
        "stroke-width": open ? 2.2 : val ? 1.4 : 0.8, class: "db-dot",
      }, dots);
      hover(dot, () => recTip(p.r, tipRow(INK.neutral, mx.l, fmtM(mx, p.a)) + tipRow(INK.neutral, my.l, fmtM(my, p.b))));
      dot.addEventListener("click", () => openPatient(p.r.id));
    }
    if (missing) txt(svg, W - R, T + 10, `${missing} without a value not shown`, { class: "db-tick dim", "text-anchor": "end" });

    const brush = el("rect", { fill: INK.text, "fill-opacity": 0.1, stroke: INK.text, visibility: "hidden" }, svg);
    const toP = (e) => {
      const r = svg.getBoundingClientRect();
      return [Math.max(L, Math.min(W - R, ((e.clientX - r.left) / r.width) * W)),
              Math.max(T, Math.min(H - B, ((e.clientY - r.top) / r.height) * H))];
    };
    // Listen on the whole plot, not just the empty overlay: in dense areas a drag starts
    // on a dot. No pointer capture, so a plain click still reaches the dot underneath.
    let start = null;
    const inPlot = (e) => {
      const r = svg.getBoundingClientRect();
      const px = ((e.clientX - r.left) / r.width) * W;
      const py = ((e.clientY - r.top) / r.height) * H;
      return px >= L && px <= W - R && py >= T && py <= H - B;
    };
    const onMove = (e) => {
      const [cx, cy] = toP(e);
      brush.setAttribute("visibility", "visible");
      brush.setAttribute("x", Math.min(start[0], cx));
      brush.setAttribute("y", Math.min(start[1], cy));
      brush.setAttribute("width", Math.abs(cx - start[0]));
      brush.setAttribute("height", Math.abs(cy - start[1]));
      hideTip();
    };
    svg.addEventListener("pointerdown", (e) => {
      if (e.button !== 0 || !inPlot(e)) return;
      e.preventDefault();
      start = toP(e);
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp, { once: true });
    });
    function onUp(e) {
      window.removeEventListener("pointermove", onMove);
      if (!start) return;
      const [cx, cy] = toP(e);
      const s = start;
      start = null;
      brush.setAttribute("visibility", "hidden");
      if (Math.abs(cx - s[0]) < 5 || Math.abs(cy - s[1]) < 5) return;
      const v = (tr, t) => +tr.inv(t).toPrecision(4);
      S.filters.sel = {
        x: S.scX, y: S.scY,
        x0: v(tx, x.invert(Math.min(s[0], cx))), x1: v(tx, x.invert(Math.max(s[0], cx))),
        y0: v(ty, y.invert(Math.max(s[1], cy))), y1: v(ty, y.invert(Math.min(s[1], cy))),
      };
      update();
    }
    overlay.addEventListener("dblclick", () => { S.filters.sel = null; update(); });

    $("dbScatterLegend").innerHTML =
      ["yolo", "medsam2", "test"].map((f) =>
        `<span class="db-legend-item"><i class="chip-dot" style="background:${FAMILY[f].color}"></i>${FAMILY[f].label}</span>`).join("") +
      `<span class="db-legend-item"><i class="chip-dot" style="background:${INK.muted}"></i>train</span>` +
      `<span class="db-legend-item"><i class="chip-dot ring" style="border-color:${INK.muted}"></i>val</span>` +
      `<span class="db-legend-item dim">${pts.length.toLocaleString()} shown · drag to select · double-click to clear</span>`;
  }

  const NBIN = 12;
  const binOf = (v) => Math.min(NBIN - 1, Math.max(0, Math.floor(v * NBIN)));
  function planeXY(r, plane) {  // 0..1 in display space: x left->right, y bottom->top
    const c = r.centroid_rel;
    if (!c) return null;
    if (plane === "axial") return [c[0], c[1]];        // radiological: patient's right on the left
    if (plane === "coronal") return [c[0], c[2]];
    return [1 - c[1], c[2]];                            // sagittal: anterior on the left
  }
  const PLANE_AXES = {
    axial: { l: "R", r: "L", b: "posterior", t: "anterior" },
    coronal: { l: "R", r: "L", b: "inferior", t: "superior" },
    sagittal: { l: "anterior", r: "posterior", b: "inferior", t: "superior" },
  };

  function renderHeatmap() {
    const box = $("dbHeatmap");
    const rows = filtered("cell");
    const W = widthOf(box);
    const side = Math.min(W - 70, 330);
    const H = side + 60;
    const svg = svgRoot(box, W, H, `Tumour centre density, ${S.plane} plane`);
    const x0 = (W - side) / 2;
    const y0 = 20;
    const cell = side / NBIN;
    const grid = Array.from({ length: NBIN }, () => new Array(NBIN).fill(0));
    for (const r of rows) {
      const c = planeXY(r, S.plane);
      if (c) grid[binOf(c[0])][binOf(c[1])]++;
    }
    const max = Math.max(1, ...grid.flat());
    const lerp = (a, b, t) => Math.round(a + (b - a) * t);
    const lo = [32, 44, 84];
    const hi = [207, 227, 255];
    const color = (n) => {
      const t = Math.sqrt(n / max);
      return `rgb(${lerp(lo[0], hi[0], t)},${lerp(lo[1], hi[1], t)},${lerp(lo[2], hi[2], t)})`;
    };
    el("rect", { x: x0, y: y0, width: side, height: side, rx: 6, fill: "#14142a", stroke: INK.grid }, svg);
    for (let i = 0; i < NBIN; i++) {
      for (let j = 0; j < NBIN; j++) {
        const n = grid[i][j];
        const cx = x0 + i * cell;
        const cy = y0 + (NBIN - 1 - j) * cell;
        const active = S.filters.cell && S.filters.cell.plane === S.plane && S.filters.cell.i === i && S.filters.cell.j === j;
        const rect = el("rect", { x: cx + 1, y: cy + 1, width: cell - 2, height: cell - 2, rx: 2,
                                  fill: n ? color(n) : "transparent", stroke: active ? "#fff" : "none",
                                  "stroke-width": 2, class: n ? "db-clickable" : "" }, svg);
        if (!n) continue;
        hover(rect, () => `<div class="tt-title">${n} patient${n > 1 ? "s" : ""}</div><span class="tt-muted">cell ${i + 1},${j + 1} of ${NBIN}×${NBIN} · click to filter</span>`);
        rect.addEventListener("click", () => {
          S.filters.cell = active ? null : { plane: S.plane, i, j };
          update();
        });
      }
    }
    const ax = PLANE_AXES[S.plane];
    txt(svg, x0 - 6, y0 + side / 2, ax.l, { class: "db-lbl", "text-anchor": "end", "dominant-baseline": "middle" });
    txt(svg, x0 + side + 6, y0 + side / 2, ax.r, { class: "db-lbl", "dominant-baseline": "middle" });
    txt(svg, x0 + side / 2, y0 - 6, ax.t, { class: "db-tick", "text-anchor": "middle" });
    txt(svg, x0 + side / 2, y0 + side + 14, ax.b, { class: "db-tick", "text-anchor": "middle" });
    // colour key
    const ky = y0 + side + 30;
    const kw = Math.min(160, side * 0.6);
    const kx = x0 + (side - kw) / 2;
    for (let k = 0; k < 20; k++) {
      el("rect", { x: kx + (k * kw) / 20, y: ky, width: kw / 20 + 0.5, height: 8, fill: color((k / 19) ** 2 * max) }, svg);
    }
    txt(svg, kx - 6, ky + 7, "1", { class: "db-tick", "text-anchor": "end" });
    txt(svg, kx + kw + 6, ky + 7, `${max} patients`, { class: "db-tick" });
  }

  function renderZProfile(rows) {
    const box = $("dbZProfile");
    const W = widthOf(box);
    const H = 250;
    const L = 52;
    const B = 38;
    const T = 12;
    const R = 10;
    const svg = svgRoot(box, W, H, "Mean tumour area per axial slice by family");
    const fams = ["yolo", "medsam2", "test"];
    const n = Math.max(0, ...rows.map((r) => r.z_profile_mm2.length));
    const series = {};
    for (const f of fams) {
      const rs = rows.filter((r) => familyOf(r) === f);
      if (!rs.length) continue;
      const acc = new Array(n).fill(0);
      for (const r of rs) r.z_profile_mm2.forEach((v, i) => { acc[i] += v; });
      series[f] = { n: rs.length, mean: acc.map((v) => v / rs.length) };
    }
    const fs = Object.keys(series);
    $("dbZLegend").innerHTML = fs.map((f) =>
      `<span class="db-legend-item"><i class="chip-line${f === "medsam2" ? " dashed" : f === "test" ? " dotted" : ""}" style="border-color:${FAMILY[f].color}"></i>${FAMILY[f].label} (${series[f].n})</span>`).join("");
    if (!fs.length) { txt(svg, W / 2, H / 2, "No patients in filter", { class: "db-empty", "text-anchor": "middle" }); return; }
    const maxV = Math.max(1, ...fs.flatMap((f) => series[f].mean));
    const x = scale(0, n - 1, L, W - R);
    const y = scale(0, maxV * 1.05, H - B, T);
    for (const t of niceTicks(0, maxV * 1.05, 4)) {
      el("line", { x1: L, x2: W - R, y1: y(t), y2: y(t), class: "db-gridline" }, svg);
      txt(svg, L - 6, y(t), fmt(t), { class: "db-tick", "text-anchor": "end", "dominant-baseline": "middle" });
    }
    for (const t of niceTicks(0, n - 1, 6)) {
      txt(svg, x(t), H - B + 16, t, { class: "db-tick", "text-anchor": "middle" });
    }
    el("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, stroke: INK.base }, svg);
    txt(svg, L + (W - L - R) / 2, H - 6, "axial slice (inferior → superior)", { class: "db-axis-title", "text-anchor": "middle" });
    txt(svg, 14, T + (H - B - T) / 2, "mean area (mm²)",
      { class: "db-axis-title", "text-anchor": "middle", transform: `rotate(-90 14 ${T + (H - B - T) / 2})` });
    const dash = { yolo: null, medsam2: "7 4", test: "2 3" };
    for (const f of fs) {
      const d = series[f].mean.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
      el("path", { d, fill: "none", stroke: FAMILY[f].color, "stroke-width": 2, "stroke-linejoin": "round",
                   "stroke-linecap": "round", "stroke-dasharray": dash[f] }, svg);
    }
    const cross = el("line", { y1: T, y2: H - B, stroke: INK.muted, visibility: "hidden" }, svg);
    const marks = fs.map((f) => el("circle", { r: 4, fill: FAMILY[f].color, stroke: INK.surface, "stroke-width": 2, visibility: "hidden" }, svg));
    const ov = el("rect", { x: L, y: T, width: W - L - R, height: H - B - T, fill: "transparent" }, svg);
    ov.addEventListener("mousemove", (e) => {
      const r = svg.getBoundingClientRect();
      const i = Math.max(0, Math.min(n - 1, Math.round(x.invert(((e.clientX - r.left) / r.width) * W))));
      cross.setAttribute("x1", x(i));
      cross.setAttribute("x2", x(i));
      cross.setAttribute("visibility", "visible");
      fs.forEach((f, k) => {
        marks[k].setAttribute("cx", x(i));
        marks[k].setAttribute("cy", y(series[f].mean[i]));
        marks[k].setAttribute("visibility", "visible");
      });
      showTip(`<div class="tt-title">Slice ${i}</div>` +
        fs.map((f) => tipRow(FAMILY[f].color, FAMILY[f].label, `${fmt(series[f].mean[i])} mm²`)).join(""), e);
    });
    ov.addEventListener("mouseleave", () => {
      hideTip();
      cross.setAttribute("visibility", "hidden");
      marks.forEach((mk) => mk.setAttribute("visibility", "hidden"));
    });
  }

  // ------------------------------------------------------------------ table
  const SHORT = { wt: "WT", tc: "TC", et: "ET", netc: "NETC", snfh: "SNFH", rc: "RC" };
  const COLS = [
    { k: "id", l: "Patient", get: (r) => r.id, str: true },
    { k: "pool", l: "Pool", get: (r) => r.pool || "", str: true },
    ...Object.keys(SHORT).map((k) => ({ k, l: SHORT[k], get: M[k].get })),
    { k: "pct", l: "% brain", get: M.pct.get },
    { k: "slices", l: "Slices", get: M.slices.get },
    { k: "ncc", l: "Parts", get: M.ncc.get },
    { k: "side", l: "Side", get: (r) => sideOf(r) || "", str: true },
    { k: "labels", l: "Labels", get: (r) => r.present.join(" "), str: true },
  ];

  function sortRows(rows) {
    const col = COLS.find((c) => c.k === S.sort.key) || COLS[2];
    const dir = S.sort.dir;
    return rows.slice().sort((a, b) => {
      const va = col.get(a);
      const vb = col.get(b);
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      const c = col.str ? String(va).localeCompare(String(vb)) : va - vb;
      return c * dir || a.id.localeCompare(b.id);
    });
  }

  function renderTable(rows) {
    S.tableRows = sortRows(rows);
    const nPages = Math.max(1, Math.ceil(S.tableRows.length / S.pageSize));
    S.page = Math.min(S.page, nPages - 1);
    const start = S.page * S.pageSize;
    const page = S.tableRows.slice(start, start + S.pageSize);
    const head = COLS.map((c) => {
      const on = S.sort.key === c.k;
      const label = SHORT[c.k] ? `${SHORT[c.k]} <span class="dim">mL</span>` : c.l;
      return `<th data-k="${c.k}" class="${c.str ? "" : "num"}${on ? " sorted" : ""}" aria-sort="${on ? (S.sort.dir > 0 ? "ascending" : "descending") : "none"}" tabindex="0">${label}${on ? (S.sort.dir > 0 ? " ▲" : " ▼") : ""}</th>`;
    }).join("");
    const body = page.map((r) => {
      const fam = familyOf(r);
      const cells = COLS.map((c) => {
        if (c.k === "id") return `<td class="mono">${esc(r.id)}</td>`;
        if (c.k === "pool") {
          const col = FAMILY[fam].color;
          const dot = roleOf(r) === "val" ? `<i class="chip-dot ring" style="border-color:${col}"></i>` : `<i class="chip-dot" style="background:${col}"></i>`;
          return `<td>${dot}${esc(r.pool || "other")}</td>`;
        }
        if (c.k === "labels") {
          return `<td>${S.labels.map((l) => `<i class="lab-sq${r.present.includes(l.name) ? "" : " off"}" style="--c:${l.color}" title="${l.name}${r.present.includes(l.name) ? " present" : " absent"}"></i>`).join("")}</td>`;
        }
        if (c.k === "side") return `<td>${esc(sideOf(r) || "–")}</td>`;
        const v = c.get(r);
        return `<td class="num">${c.k === "pct" ? `${fmt(v)}%` : fmt(v)}</td>`;
      }).join("");
      return `<tr data-id="${esc(r.id)}" class="${S.open === r.id ? "active-row" : ""}" tabindex="0">${cells}</tr>`;
    }).join("");
    $("dbTableWrap").innerHTML = rows.length
      ? `<table class="db-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`
      : `<p class="db-empty-msg">No patients match these filters. <button type="button" id="dbEmptyReset">Reset filters</button></p>`;
    $("dbPager").innerHTML = rows.length
      ? `<span class="dim">${(start + 1).toLocaleString()}–${Math.min(start + S.pageSize, S.tableRows.length).toLocaleString()} of ${S.tableRows.length.toLocaleString()}</span>
         <button type="button" data-pg="-1" ${S.page ? "" : "disabled"} aria-label="Previous page">‹</button>
         <button type="button" data-pg="1" ${S.page < nPages - 1 ? "" : "disabled"} aria-label="Next page">›</button>
         <select id="dbPageSize" aria-label="Rows per page">${[25, 50, 100, 250].map((n) => `<option ${n === S.pageSize ? "selected" : ""}>${n}</option>`).join("")}</select>`
      : "";
    const er = $("dbEmptyReset");
    if (er) er.addEventListener("click", resetFilters);
  }

  // ------------------------------------------------------------------ drawer
  function openPatient(id, z) {
    S.open = id;
    S.drawerZ = Number.isInteger(z) ? z : null;
    renderAll();
  }
  function closeDrawer() {
    S.open = null;
    S.drawerZ = null;
    renderAll();
  }
  function stepPatient(d) {
    if (!S.open) return;
    const i = S.tableRows.findIndex((r) => r.id === S.open);
    const next = S.tableRows[i + d] || (i < 0 ? S.tableRows[0] : null);
    if (!next) return;
    const pos = S.tableRows.indexOf(next);
    S.page = Math.floor(pos / S.pageSize);
    openPatient(next.id);
  }
  const profileToZ = (r, i) => (r.axcodes[2] === "S" ? i : r.z_profile_mm2.length - 1 - i);

  function renderDrawer() {
    const d = $("dbDrawer");
    const r = S.open ? S.byId[S.open] : null;
    d.classList.toggle("open", !!r);
    d.setAttribute("aria-hidden", r ? "false" : "true");
    if (!r) return;
    const depth = r.shape[2];
    const z = S.drawerZ != null ? Math.max(0, Math.min(depth - 1, S.drawerZ)) : r.peak_z;
    const pool = S.poolMeta[r.pool];
    $("dbDrawerTitle").textContent = r.id;
    $("dbDrawerSub").textContent = `${r.pool || "other"}${pool ? ` · ${pool.use}` : ""} · subject ${r.subject}, timepoint ${r.timepoint}`;
    const pos = S.tableRows.findIndex((x) => x.id === r.id);
    $("dbPrev").disabled = pos <= 0;
    $("dbNext").disabled = pos < 0 || pos >= S.tableRows.length - 1;

    const rank = (get, rs) => {
      const v = get(r);
      const vals = rs.map(get).filter((x) => x != null);
      return vals.length ? vals.filter((x) => x < v).length / vals.length : null;
    };
    const inPool = S.records.filter((x) => x.pool === r.pool);
    const others = (S.subjectScans[r.subject] || []).filter((x) => x.id !== r.id)
      .sort((a, b) => a.timepoint.localeCompare(b.timepoint));
    const labelBars = S.labels.map((l) => {
      const v = r.ml[l.name];
      const s = r.ml.WT ? v / r.ml.WT : 0;
      return `<div class="dr-lab"><span><i class="chip-dot sq" style="background:${l.color}"></i>${l.name}</span>
        <div class="pres-bar"><span style="width:${s * 100}%;background:${l.color}"></span></div><b>${fmt(v)} mL</b></div>`;
    }).join("");
    const kv = [
      ["Whole tumour", `${fmt(r.ml.WT)} mL`, `larger than ${pct(rank((x) => x.ml.WT, inPool))} of ${r.pool} · ${pct(rank((x) => x.ml.WT, S.records))} of all`],
      ["Tumour core", `${fmt(r.ml.TC)} mL`],
      ["Brain", `${fmt(r.brain_ml)} mL`, `tumour = ${fmt(r.wt_pct_brain)}% of brain`],
      ["Tumour slices", `${r.n_slices}`, `largest slice ${fmt(r.peak_area_mm2)} mm² at z=${r.peak_z}`],
      ["Connected parts", `${r.n_cc}`, r.largest_cc_frac != null ? `largest part holds ${pct(r.largest_cc_frac)}` : ""],
      ["Extent", r.extent_mm ? `${r.extent_mm.map((v) => fmt(v)).join(" × ")} mm` : "–", "left–right × front–back × up–down"],
      ["Side", sideOf(r) || "–", r.left_frac != null ? `${pct(r.left_frac)} of tumour on the left` : ""],
      ["Volume", `${r.shape.join(" × ")}`, `${r.spacing_mm.join(" × ")} mm voxels · ${r.axcodes}`],
    ].map(([k, v, s]) => `<div class="dr-kv"><span>${k}</span><b>${esc(v)}</b>${s ? `<em>${esc(s)}</em>` : ""}</div>`).join("");

    const mods = [...S.imageKinds, { key: "all", label: "All" }];
    $("dbDrawerBody").innerHTML = `
      <div class="dr-thumb">
        <div class="dr-mods" role="group" aria-label="Image">${mods.map((k, i) => {
          const ok = k.key === "all" || r.images.includes(k.key);
          return `<button type="button" class="dr-mod" data-mod="${k.key}" ${ok ? "" : "disabled"} title="${esc(k.label)} (key ${i + 1})">${esc(k.label)}</button>`;
        }).join("")}</div>
        <div id="dbImages"></div>
        <div class="dr-thumb-ctrl">
          <input type="range" id="dbThumbZ" min="0" max="${depth - 1}" value="${z}" aria-label="Slice">
          <span class="zlabel" id="dbThumbZL">z = ${z}</span>
        </div>
        <div class="dr-btns">
          <button type="button" id="dbThumbPeak">Largest slice</button>
          <label class="db-toggle dr-ov" title="Show the segmentation labels on the image (key L)"><input type="checkbox" id="dbOverlay"> Labels</label>
        </div>
        <p class="db-card-sub" id="dbImagesCap"></p>
      </div>
      <h4>Tumour along the head</h4>
      <div id="dbDrawerProfile"></div>
      <h4>Labels</h4>
      <div class="dr-labs">${labelBars}</div>
      <h4>Measurements</h4>
      <div class="dr-kvs">${kv}</div>
      <h4>Other scans of this subject</h4>
      ${others.length
        ? `<div class="dr-others">${others.map((o) => `<button type="button" class="dr-other" data-id="${esc(o.id)}">
            <i class="chip-dot${roleOf(o) === "val" ? " ring" : ""}" style="${roleOf(o) === "val" ? "border-color" : "background"}:${colorOf(o)}"></i>
            <span class="mono">${esc(o.id)}</span><span class="dim">${esc(o.pool || "other")}</span><b>${fmt(o.ml.WT)} mL</b></button>`).join("")}</div>`
        : `<p class="db-card-sub">Only one scan of this subject in the dataset.</p>`}
    `;
    const setZ = (nz) => {
      S.drawerZ = nz;
      $("dbThumbZ").value = nz;
      $("dbThumbZL").textContent = `z = ${nz}`;
      drawImages();
      drawProfile(r, nz, setZ);
      syncHash();
    };
    $("dbThumbZ").addEventListener("input", (e) => setZ(+e.target.value));
    $("dbThumbPeak").addEventListener("click", () => setZ(r.peak_z));
    $("dbDrawerBody").querySelector(".dr-mods").addEventListener("click", (e) => {
      const b = e.target.closest(".dr-mod");
      if (b && !b.disabled) setImage(b.dataset.mod);
    });
    $("dbOverlay").addEventListener("change", (e) => setOverlay(e.target.checked));
    for (const b of $("dbDrawerBody").querySelectorAll(".dr-other")) b.addEventListener("click", () => openPatient(b.dataset.id));
    drawImages();
    drawProfile(r, z, setZ);
  }

  // The drawer's slice image(s): one modality / T1C - T1, or all five side by side.
  const IMAGE_NOTES = { sub: "T1C minus T1 (negative values set to 0): where the contrast agent enhanced." };
  function drawImages() {
    const r = S.open ? S.byId[S.open] : null;
    if (!r || !$("dbImages")) return;
    const z = S.drawerZ != null ? Math.max(0, Math.min(r.shape[2] - 1, S.drawerZ)) : r.peak_z;
    const all = S.mod === "all";
    const kinds = all ? r.images : [S.mod];
    const label = (k) => (S.imageKinds.find((x) => x.key === k) || { label: k }).label;
    for (const b of $("dbDrawerBody").querySelectorAll(".dr-mod")) {
      b.classList.toggle("on", b.dataset.mod === S.mod);
      b.setAttribute("aria-pressed", b.dataset.mod === S.mod);
    }
    $("dbOverlay").checked = S.overlay;
    const box = $("dbImages");
    box.className = all ? "dr-images grid" : "dr-images";
    if (!all && !r.images.includes(S.mod)) {
      box.innerHTML = `<p class="db-empty-msg">${esc(label(S.mod))} is not available for this patient.</p>`;
      $("dbImagesCap").textContent = "";
      return;
    }
    box.innerHTML = kinds.map((k) =>
      `<figure><img alt="${esc(label(k))} slice ${z} of ${esc(r.id)}${S.overlay ? " with label overlay" : ""}"
        src="/thumb.png?id=${r.idx}&z=${z}&mod=${k}&ov=${S.overlay ? 1 : 0}">${all ? `<figcaption>${esc(label(k))}</figcaption>` : ""}</figure>`
    ).join("");
    $("dbImagesCap").textContent = [
      all ? "All images at the same slice." : `${label(S.mod)}.`,
      !all && IMAGE_NOTES[S.mod] ? IMAGE_NOTES[S.mod] : "",
      "Radiological view (patient's right on the left).",
    ].filter(Boolean).join(" ");
  }
  function setImage(k) {
    S.mod = k;
    drawImages();
    syncHash();
  }
  function setOverlay(on) {
    S.overlay = on;
    drawImages();
    syncHash();
  }

  function drawProfile(r, z, setZ) {
    const box = $("dbDrawerProfile");
    const W = Math.max(260, box.clientWidth || 340);
    const H = 90;
    const svg = svgRoot(box, W, H, "Tumour area per slice for this patient");
    const prof = r.z_profile_mm2;
    const maxV = Math.max(1, ...prof);
    const x = scale(0, prof.length - 1, 4, W - 4);
    const y = scale(0, maxV, H - 16, 6);
    const area = `M${x(0)},${y(0)}` + prof.map((v, i) => `L${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("") + `L${x(prof.length - 1)},${y(0)}Z`;
    const c = colorOf(r);
    el("path", { d: area, fill: c, "fill-opacity": 0.12 }, svg);
    el("path", { d: prof.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(""), fill: "none", stroke: c, "stroke-width": 2 }, svg);
    const iz = profileToZ(r, z);
    el("line", { x1: x(iz), x2: x(iz), y1: 4, y2: H - 16, stroke: INK.text, "stroke-dasharray": "3 3" }, svg);
    el("circle", { cx: x(iz), cy: y(prof[iz] || 0), r: 4, fill: c, stroke: INK.surface, "stroke-width": 2 }, svg);
    txt(svg, 4, H - 3, "inferior", { class: "db-tick" });
    txt(svg, W - 4, H - 3, "superior", { class: "db-tick", "text-anchor": "end" });
    txt(svg, W / 2, H - 3, `${fmt(prof[iz] || 0)} mm² at z=${z}`, { class: "db-tick", "text-anchor": "middle" });
    const ov = el("rect", { x: 0, y: 0, width: W, height: H, fill: "transparent", class: "db-clickable" }, svg);
    const idxAt = (e) => {
      const b = svg.getBoundingClientRect();
      return Math.max(0, Math.min(prof.length - 1, Math.round(x.invert(((e.clientX - b.left) / b.width) * W))));
    };
    ov.addEventListener("mousemove", (e) => {
      const i = idxAt(e);
      showTip(`<div class="tt-title">z = ${profileToZ(r, i)}</div>${fmt(prof[i])} mm² · click to view`, e);
    });
    ov.addEventListener("mouseleave", hideTip);
    ov.addEventListener("click", (e) => setZ(profileToZ(r, idxAt(e))));
  }

  // ------------------------------------------------------------------ export
  function exportCsv() {
    const rows = S.tableRows;
    const cols = [
      ["patient_id", (r) => r.id], ["subject", (r) => r.subject], ["timepoint", (r) => r.timepoint],
      ["pool", (r) => r.pool], ...["WT", "TC", "NETC", "SNFH", "ET", "RC"].map((l) => [`${l}_ml`, (r) => r.ml[l]]),
      ["brain_ml", (r) => r.brain_ml], ["wt_pct_brain", (r) => r.wt_pct_brain], ["tumour_slices", (r) => r.n_slices],
      ["peak_z", (r) => r.peak_z], ["peak_area_mm2", (r) => r.peak_area_mm2], ["components", (r) => r.n_cc],
      ["largest_component_share", (r) => r.largest_cc_frac],
      ["extent_lr_mm", (r) => r.extent_mm && r.extent_mm[0]], ["extent_ap_mm", (r) => r.extent_mm && r.extent_mm[1]],
      ["extent_si_mm", (r) => r.extent_mm && r.extent_mm[2]],
      ["centre_left_rel", (r) => r.centroid_rel && r.centroid_rel[0]], ["centre_anterior_rel", (r) => r.centroid_rel && r.centroid_rel[1]],
      ["centre_superior_rel", (r) => r.centroid_rel && r.centroid_rel[2]], ["left_share", (r) => r.left_frac],
      ["side", (r) => sideOf(r)], ["labels_present", (r) => r.present.join(" ")],
    ];
    const cell = (v) => (v == null ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v);
    const csv = [cols.map((c) => c[0]).join(","), ...rows.map((r) => cols.map((c) => cell(c[1](r))).join(","))].join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = `brats_patients_${rows.length}.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  function flash(btn, msg) {
    const old = btn.textContent;
    btn.textContent = msg;
    setTimeout(() => { btn.textContent = old; }, 1400);
  }

  // ------------------------------------------------------------------ events
  $("dbPoolChips").addEventListener("click", (e) => {
    const b = e.target.closest(".db-chip");
    if (!b) return;
    const s = S.filters.pools;
    if (s.has(b.dataset.pool)) s.delete(b.dataset.pool); else s.add(b.dataset.pool);
    update();
  });
  $("dbLabelChips").addEventListener("click", (e) => {
    const b = e.target.closest(".db-chip");
    if (!b) return;
    const s = S.filters.labels;
    if (s.has(b.dataset.label)) s.delete(b.dataset.label); else s.add(b.dataset.label);
    update();
  });
  $("dbActiveChips").addEventListener("click", (e) => {
    const b = e.target.closest("[data-clear]");
    if (!b) return;
    S.filters[b.dataset.clear] = null;
    update();
  });
  $("dbSide").addEventListener("change", (e) => { S.filters.side = e.target.value; update(); });
  let searchT = null;
  $("dbSearch").addEventListener("input", (e) => {
    clearTimeout(searchT);
    searchT = setTimeout(() => { S.filters.q = e.target.value.trim(); update(); }, 150);
  });
  $("dbReset").addEventListener("click", resetFilters);
  const bind = (id, key, isCheck) => $(id).addEventListener("change", (e) => {
    S[key] = isCheck ? e.target.checked : e.target.value;
    renderAll();
  });
  $("dbDistMetric").addEventListener("change", (e) => {
    S.distMetric = e.target.value;
    S.distLog = !!M[S.distMetric].logDefault;
    renderAll();
  });
  $("dbHistMetric").addEventListener("change", (e) => {
    S.histMetric = e.target.value;
    S.histLog = !!M[S.histMetric].logDefault;
    renderAll();
  });
  $("dbScX").addEventListener("change", (e) => { S.scX = e.target.value; S.scXLog = !!M[S.scX].logDefault; renderAll(); });
  $("dbScY").addEventListener("change", (e) => { S.scY = e.target.value; S.scYLog = !!M[S.scY].logDefault; renderAll(); });
  bind("dbDistLog", "distLog", true);
  bind("dbHistLog", "histLog", true);
  bind("dbScXLog", "scXLog", true);
  bind("dbScYLog", "scYLog", true);
  bind("dbPlane", "plane", false);

  $("dbTableWrap").addEventListener("click", (e) => {
    const th = e.target.closest("th[data-k]");
    if (th) {
      const k = th.dataset.k;
      S.sort = S.sort.key === k ? { key: k, dir: -S.sort.dir } : { key: k, dir: COLS.find((c) => c.k === k).str ? 1 : -1 };
      update();
      return;
    }
    const tr = e.target.closest("tr[data-id]");
    if (tr) openPatient(tr.dataset.id);
  });
  $("dbTableWrap").addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    const th = e.target.closest("th[data-k]");
    const tr = e.target.closest("tr[data-id]");
    if (th) th.click();
    else if (tr) openPatient(tr.dataset.id);
  });
  $("dbPager").addEventListener("click", (e) => {
    const b = e.target.closest("[data-pg]");
    if (!b) return;
    S.page += +b.dataset.pg;
    update({ keepPage: true });
  });
  $("dbPager").addEventListener("change", (e) => {
    if (e.target.id !== "dbPageSize") return;
    S.pageSize = +e.target.value;
    update();
  });
  $("dbPrev").addEventListener("click", () => stepPatient(-1));
  $("dbNext").addEventListener("click", () => stepPatient(1));
  $("dbClose").addEventListener("click", closeDrawer);
  $("dbExport").addEventListener("click", exportCsv);
  $("dbShare").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(location.href);
      flash($("dbShare"), "Link copied");
    } catch {
      flash($("dbShare"), "Copy failed");
    }
  });
  $("dbRecompute").addEventListener("click", async () => {
    if (!confirm("Recompute statistics for every patient from the NIfTI files? This takes a few minutes.")) return;
    S.loaded = false;
    await fetch("/api/dashboard/recompute", { method: "POST" });
    poll();
  });

  document.addEventListener("keydown", (e) => {
    if ($("analyticsPanel").classList.contains("hidden")) return;
    const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement.tagName);
    if (e.key === "/" && !typing) {
      e.preventDefault();
      $("dbSearch").focus();
    } else if (e.key === "Escape") {
      if (document.activeElement === $("dbSearch")) {
        $("dbSearch").value = "";
        S.filters.q = "";
        $("dbSearch").blur();
        update();
      } else if (S.open) closeDrawer();
    } else if (!typing && S.open && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
      e.preventDefault();
      stepPatient(e.key === "ArrowRight" ? 1 : -1);
    } else if (!typing && S.open && /^[1-6]$/.test(e.key) && !e.ctrlKey && !e.metaKey) {
      const k = [...S.imageKinds.map((x) => x.key), "all"][+e.key - 1];
      const r = S.byId[S.open];
      if (k && (k === "all" || r.images.includes(k))) setImage(k);
    } else if (!typing && S.open && (e.key === "l" || e.key === "L") && !e.ctrlKey && !e.metaKey) {
      setOverlay(!S.overlay);
    } else if (!typing && (e.key === "r" || e.key === "R") && !e.ctrlKey && !e.metaKey) {
      resetFilters();
    }
  });

  let resizeT = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeT);
    resizeT = setTimeout(() => { if (!$("analyticsPanel").classList.contains("hidden")) renderAll(); }, 150);
  });

  window.Dashboard = {
    show() {
      fromHash();
      if (S.loaded) renderAll();
      else if (!S.loading) S.loading = poll();
    },
    hash: toHash,
  };
})();
