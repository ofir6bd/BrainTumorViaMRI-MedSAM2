/* Analytics — the dataset itself: every patient's real tumour statistics, cross-filtered.
 *
 * Data: /api/dashboard/data (measured server-side from each patient's -seg and FLAIR, cached).
 * A patient without a value for a measure is left out of that chart, never filled in.
 *
 * Cross-filtering: pool / parts / side / search, plus filters made on charts (a range on any
 * measure, a scatter box, a location cell, a shared-people cell, a parts combination). Every
 * chart shows the filtered patients, except that a chart never filters itself (a histogram
 * ignores its own range, the scatter its own box ...) and the pool charts ignore the pool
 * filter so the other pools stay visible for contrast. The whole view lives in the URL.
 */
(function () {
  "use strict";
  const K = Kit, $ = K.id, esc = K.esc, S_ = K.stats;

  // ------------------------------------------------------------------ measures
  const M = {};
  const METRICS = [
    ["wt", "Whole tumour (mL)", (r) => r.ml.WT, true],
    ["tc", "Tumour core (mL)", (r) => r.ml.TC, true],
    ["et", "Enhancing, ET (mL)", (r) => r.ml.ET, true],
    ["netc", "Dead core, NETC (mL)", (r) => r.ml.NETC, true],
    ["snfh", "Swelling, SNFH (mL)", (r) => r.ml.SNFH, true],
    ["rc", "Surgery cavity, RC (mL)", (r) => r.ml.RC, true],
    ["pct", "Tumour, % of brain", (r) => r.wt_pct_brain, true],
    ["etfrac", "ET share of tumour", (r) => (r.ml.WT ? r.ml.ET / r.ml.WT : null), false, K.pct],
    ["tcfrac", "Core share of tumour", (r) => (r.ml.WT ? r.ml.TC / r.ml.WT : null), false, K.pct],
    ["brain", "Brain (mL)", (r) => r.brain_ml, false],
    ["slices", "Slices with tumour", (r) => r.n_slices, false],
    ["peak", "Biggest slice (mm²)", (r) => r.peak_area_mm2, false],
    ["ncc", "Tumour pieces", (r) => r.n_cc, true],
    ["lcc", "Share in biggest piece", (r) => r.largest_cc_frac, false, K.pct],
    ["exlr", "Width L–R (mm)", (r) => (r.extent_mm ? r.extent_mm[0] : null), false],
    ["exap", "Length front–back (mm)", (r) => (r.extent_mm ? r.extent_mm[1] : null), false],
    ["exsi", "Height bottom–top (mm)", (r) => (r.extent_mm ? r.extent_mm[2] : null), false],
    ["left", "Share on the left", (r) => r.left_frac, false, K.pct],
  ].map(([k, l, get, log, f]) => (M[k] = { k, l, get, log, f: f || K.fmt }));
  const MINI = ["wt", "et", "snfh", "netc", "rc", "pct", "slices", "ncc", "brain", "left"];
  const CORR = ["wt", "tc", "et", "netc", "snfh", "rc", "brain", "slices", "peak", "ncc", "lcc", "left"];
  const SIDE_WORDS = { left: "left", right: "right", bilateral: "both sides" };
  const NB = 12;  // location heatmap bins

  // ------------------------------------------------------------------ state
  const S = {
    recs: [], byId: {}, pools: [], meta: {}, labels: [], kinds: [], manifest: null, n: 0,
    subjScans: {}, subjByPool: {},
    f: { pools: new Set(), labels: new Set(), side: "", q: "", ranges: {}, sel: null, cell: null, shared: null, combo: null },
    dm: "wt", dl: true, hm: "wt", hl: true, hmode: "hist", hby: "all", sx: "wt", sy: "et", sxl: true, syl: true,
    scol: "pool", trend: false, plane: "axial", punit: "scans",
    open: null, z: null, img: "flair", ov: true, cmp: [], order: [],
    charts: {}, tables: {},
  };
  const famOf = (r) => (S.meta[r.pool] ? S.meta[r.pool].family : "other");
  const roleOf = (r) => (S.meta[r.pool] ? S.meta[r.pool].role : "train");
  const colOf = (r) => K.FAMILY[famOf(r)];
  const sideOf = (r) => (r.left_frac == null ? null : r.left_frac > 2 / 3 ? "left" : r.left_frac < 1 / 3 ? "right" : "bilateral");
  const comboOf = (r) => (r.present.length ? r.present.join("+") : "none");
  const shared = (a, b) => {
    const A = S.subjByPool[a] || new Set(), B = S.subjByPool[b] || new Set();
    return new Set([...A].filter((s) => B.has(s)));
  };

  // ------------------------------------------------------------------ filtering
  function passes(r, skip) {
    const f = S.f;
    if (skip !== "pool" && f.pools.size && !f.pools.has(r.pool)) return false;
    for (const l of f.labels) if (!r.present.includes(l)) return false;
    if (f.side && sideOf(r) !== f.side) return false;
    if (f.q && !r.id.toLowerCase().includes(f.q.toLowerCase())) return false;
    for (const [m, [lo, hi]] of Object.entries(f.ranges)) {
      if (skip === `range:${m}`) continue;
      const v = M[m].get(r);
      if (v == null || v < lo || v > hi) return false;
    }
    if (skip !== "sel" && f.sel) {
      const x = M[f.sel.x].get(r), y = M[f.sel.y].get(r);
      if (x == null || y == null || x < f.sel.x0 || x > f.sel.x1 || y < f.sel.y0 || y > f.sel.y1) return false;
    }
    if (skip !== "cell" && f.cell) {
      const c = planeXY(r, f.cell.plane);
      if (!c || bin(c[0]) !== f.cell.i || bin(c[1]) !== f.cell.j) return false;
    }
    if (f.shared) {
      if (r.pool !== f.shared[0] && r.pool !== f.shared[1]) return false;
      if (f.shared[0] !== f.shared[1] && !shared(f.shared[0], f.shared[1]).has(r.subject)) return false;
    }
    if (skip !== "combo" && f.combo && comboOf(r) !== f.combo) return false;
    return true;
  }
  const rowsOf = (skip) => S.recs.filter((r) => passes(r, skip));

  // ------------------------------------------------------------------ URL
  function toHash() {
    const f = S.f;
    K.hashSet({
      pools: [...f.pools].join(","), labels: [...f.labels].join(","), side: f.side, q: f.q,
      ranges: Object.entries(f.ranges).map(([m, [a, b]]) => `${m}:${+a.toPrecision(4)}:${+b.toPrecision(4)}`).join(";"),
      sel: f.sel ? [f.sel.x, f.sel.y, f.sel.x0, f.sel.x1, f.sel.y0, f.sel.y1].map((v) => (typeof v === "number" ? +v.toPrecision(4) : v)).join(":") : "",
      cell: f.cell ? `${f.cell.plane}:${f.cell.i}:${f.cell.j}` : "", shared: f.shared ? f.shared.join(":") : "", combo: f.combo || "",
      dm: `${S.dm}:${+S.dl}`, hm: `${S.hm}:${+S.hl}:${S.hmode}:${S.hby}`, sc: `${S.sx}:${+S.sxl}:${S.sy}:${+S.syl}:${S.scol}:${+S.trend}`,
      plane: S.plane, pu: S.punit, p: S.open, z: S.open && S.z != null ? S.z : "", img: S.img !== "flair" ? S.img : "",
      ov: S.ov ? "" : "0", cmp: S.cmp.join(","),
    });
  }
  function fromHash() {
    const q = K.hashGet(), f = S.f;
    const list = (k) => (q[k] ? q[k].split(",").filter(Boolean) : []);
    f.pools = new Set(list("pools"));
    f.labels = new Set(list("labels"));
    f.side = q.side || "";
    f.q = q.q || "";
    f.ranges = {};
    for (const part of (q.ranges || "").split(";").filter(Boolean)) {
      const [m, a, b] = part.split(":");
      if (M[m]) f.ranges[m] = [+a, +b];
    }
    const sel = (q.sel || "").split(":");
    f.sel = sel.length === 6 && M[sel[0]] && M[sel[1]] ? { x: sel[0], y: sel[1], x0: +sel[2], x1: +sel[3], y0: +sel[4], y1: +sel[5] } : null;
    const cell = (q.cell || "").split(":");
    f.cell = cell.length === 3 ? { plane: cell[0], i: +cell[1], j: +cell[2] } : null;
    const sh = (q.shared || "").split(":");
    f.shared = sh.length === 2 ? sh : null;
    f.combo = q.combo || null;
    const dm = (q.dm || "").split(":"); if (M[dm[0]]) { S.dm = dm[0]; S.dl = dm[1] === "1"; }
    const hm = (q.hm || "").split(":"); if (M[hm[0]]) { S.hm = hm[0]; S.hl = hm[1] === "1"; S.hmode = hm[2] || "hist"; S.hby = hm[3] || "all"; }
    const sc = (q.sc || "").split(":");
    if (M[sc[0]] && M[sc[2]]) { S.sx = sc[0]; S.sxl = sc[1] === "1"; S.sy = sc[2]; S.syl = sc[3] === "1"; S.scol = sc[4] || "pool"; S.trend = sc[5] === "1"; }
    if (["axial", "coronal", "sagittal"].includes(q.plane)) S.plane = q.plane;
    S.punit = q.pu === "people" ? "people" : "scans";
    S.open = q.p || null;
    S.z = q.z !== undefined && q.z !== "" ? +q.z : null;
    S.img = q.img || "flair";
    S.ov = q.ov !== "0";
    S.cmp = list("cmp").slice(0, 4);
  }

  // ------------------------------------------------------------------ load
  async function poll() {
    let st;
    try { st = await K.api("/api/dashboard/status"); } catch (e) { K.fail(e); $("pageSub").textContent = e.message; return; }
    const box = $("progress");
    if (st.state === "running") {
      box.classList.remove("hidden");
      const p = st.total ? st.done / st.total : 0;
      $("progressText").textContent = `Measuring every patient from the scan files: ${st.done} of ${st.total} (${Math.round(p * 100)}%)`;
      $("progressFill").style.width = `${p * 100}%`;
      setTimeout(poll, 1000);
      return;
    }
    box.classList.add("hidden");
    await load();
  }
  async function load() {
    let d;
    try { d = await K.api("/api/dashboard/data"); } catch (e) { K.fail(e); return; }
    S.recs = d.records;
    S.byId = Object.fromEntries(d.records.map((r) => [r.id, r]));
    S.pools = d.pools;
    S.meta = Object.fromEntries(d.pools.map((p) => [p.key, p]));
    S.labels = d.labels.map((l) => l.name);
    S.kinds = d.image_kinds;
    S.manifest = d.manifest;
    S.n = d.n_patients;
    S.subjScans = {};
    S.subjByPool = {};
    for (const r of S.recs) {
      (S.subjScans[r.subject] ||= []).push(r);
      (S.subjByPool[r.pool] ||= new Set()).add(r.subject);
    }
    const errs = Object.entries(d.errors || {});
    $("errors").classList.toggle("hidden", !errs.length);
    $("errors").innerHTML = errs.length ? `<b>${errs.length} patient(s) could not be read and are left out:</b> ${errs.slice(0, 8).map(([id, e]) => `${esc(id)} (${esc(e)})`).join("; ")}` : "";
    $("pageSub").textContent = `${d.records.length.toLocaleString()} of ${d.n_patients.toLocaleString()} scans · ${Object.keys(S.subjScans).length.toLocaleString()} people · measured ${K.when(d.computed_at)} · ${d.dataset_dir}`;
    buildControls();
    fromHash();
    render();
  }

  // ------------------------------------------------------------------ controls
  function buildControls() {
    const opts = Object.values(M).map((m) => `<option value="${m.k}">${esc(m.l)}</option>`).join("");
    for (const id of ["distMetric", "histMetric", "scX", "scY"]) $(id).innerHTML = opts;
    $("poolChips").innerHTML = S.pools.map((p) => `<button class="chip" data-pool="${p.key}" title="${esc(p.use)}">
      <i class="dot ${p.role === "val" ? "ring" : ""}" style="${p.role === "val" ? "border-color" : "background"}:${K.FAMILY[p.family]}"></i>${esc(p.key)}</button>`).join("");
    $("labelChips").innerHTML = S.labels.map((l) => `<button class="chip" data-label="${l}" title="Only patients whose tumour has a ${esc(K.LABEL_WORDS[l])} part">
      <i class="dot sq" style="background:${K.LABELS[l]}"></i>${l} · ${esc(K.LABEL_WORDS[l])}</button>`).join("");
  }
  function syncControls() {
    const f = S.f;
    K.$$("#poolChips .chip").forEach((b) => b.classList.toggle("on", f.pools.has(b.dataset.pool)));
    K.$$("#labelChips .chip").forEach((b) => b.classList.toggle("on", f.labels.has(b.dataset.label)));
    K.$$("#sideSeg button").forEach((b) => b.classList.toggle("on", b.dataset.v === f.side));
    if (document.activeElement !== $("search")) $("search").value = f.q;
    $("distMetric").value = S.dm; $("distLog").checked = S.dl;
    $("histMetric").value = S.hm; $("histLog").checked = S.hl;
    K.$$("#histMode button").forEach((b) => b.classList.toggle("on", b.dataset.v === S.hmode));
    K.$$("#histBy button").forEach((b) => b.classList.toggle("on", b.dataset.v === S.hby));
    $("scX").value = S.sx; $("scXLog").checked = S.sxl; $("scY").value = S.sy; $("scYLog").checked = S.syl;
    $("scColor").value = S.scol; $("scTrend").checked = S.trend;
    K.$$("#planeSeg button").forEach((b) => b.classList.toggle("on", b.dataset.v === S.plane));
    K.$$("#poolUnit button").forEach((b) => b.classList.toggle("on", b.dataset.v === S.punit));
    const chips = [];
    for (const [m, [a, b]] of Object.entries(f.ranges)) chips.push([`range:${m}`, `${M[m].l}: ${M[m].f(a)} – ${M[m].f(b)}`]);
    if (f.sel) chips.push(["sel", `Box on ${M[f.sel.x].l} × ${M[f.sel.y].l}`]);
    if (f.cell) chips.push(["cell", `Location square (${f.cell.plane})`]);
    if (f.shared) chips.push(["shared", f.shared[0] === f.shared[1] ? `Pool ${f.shared[0]}` : `People in both ${f.shared[0]} and ${f.shared[1]}`]);
    if (f.combo) chips.push(["combo", `Parts exactly: ${f.combo}`]);
    $("activeChips").innerHTML = chips.map(([k, l]) => `<button class="chip on" data-clear="${k}" title="Remove this filter">${esc(l)}<span class="x">✕</span></button>`).join("");
    const n = rowsOf().length;
    $("count").textContent = `${n.toLocaleString()} of ${S.recs.length.toLocaleString()} scans`;
  }
  function reset() {
    S.f = { pools: new Set(), labels: new Set(), side: "", q: "", ranges: {}, sel: null, cell: null, shared: null, combo: null };
    update();
  }
  const update = () => { toHash(); render(); };

  // ------------------------------------------------------------------ render
  function render() {
    if (!S.recs.length) return;
    syncControls();
    const rows = rowsOf();
    renderKpis(rows);
    renderLeak(); renderManifest(); renderPools(); renderTimepoints(rows);
    renderDist(); renderHist(); renderMinis();
    renderShare(rows); renderPresence(rows); renderCombos();
    renderScatter(); renderCorr(rows);
    renderHeat(); renderZ(rows); renderSides(rows);
    renderOutliers(rows); renderTable(rows);
    renderTray(); renderDrawer();
  }

  function renderKpis(rows) {
    const all = S.recs;
    const med = (rs, m) => S_.median(rs.map(M[m].get));
    const share = (rs, fn) => (rs.length ? rs.filter(fn).length / rs.length : null);
    const people = new Set(rows.map((r) => r.subject)).size;
    const filtered = rows.length !== all.length;
    const rel = (a, b) => (!filtered || a == null || b == null || !b ? "" : ` · ${a >= b ? "▲" : "▼"} ${K.pct(Math.abs(a - b) / b, 0)} vs all`);
    const sides = ["left", "right", "bilateral"].map((s) => share(rows, (r) => sideOf(r) === s));
    const q = S_.summary(rows.map(M.wt.get));
    $("kpis").innerHTML = [
      K.kpi("Scans", rows.length.toLocaleString(), `${people.toLocaleString()} people · ${K.pct(rows.length / all.length, 0)} of the dataset`, { hero: true }),
      K.kpi("Median whole tumour", `${K.fmt(q.median)} mL`, `middle half ${K.fmt(q.q1)}–${K.fmt(q.q3)} mL${rel(q.median, med(all, "wt"))}`),
      K.kpi("Median enhancing (ET)", `${K.fmt(med(rows, "et"))} mL`, `core ${K.fmt(med(rows, "tc"))} mL${rel(med(rows, "et"), med(all, "et"))}`),
      K.kpi("Has a surgery cavity", K.pct(share(rows, (r) => r.present.includes("RC")), 0), "post-operative scans (RC)"),
      K.kpi("Several pieces", K.pct(share(rows, (r) => r.n_cc > 1), 0), `median ${K.fmt(med(rows, "ncc"))} piece(s)`),
      K.kpi("Tumour slices", K.fmt(med(rows, "slices")), `median per scan · biggest ${K.fmt(med(rows, "peak"))} mm²`),
      K.kpi("Side", `${K.pct(sides[0], 0)} / ${K.pct(sides[1], 0)}`, `left / right · ${K.pct(sides[2], 0)} both sides`),
      K.kpi("Tumour share of brain", K.pct((med(rows, "pct") || 0) / 100), `median · brain ${K.fmt(med(rows, "brain"))} mL`),
    ].join("");
  }

  // ------------------------------------------------------------------ split & integrity
  function renderLeak() {
    const pools = S.pools.map((p) => p.key);
    const fam = (k) => S.meta[k].family;
    const cross = (a, b) => (fam(a) === "yolo" && fam(b) === "medsam2") || (fam(a) === "medsam2" && fam(b) === "yolo");
    const same = (a, b) => fam(a) === fam(b) && fam(a) !== "test" && a !== b;
    const V = pools.map((a) => pools.map((b) => (a === b ? (S.subjByPool[a] || new Set()).size : shared(a, b).size)));
    let bad = 0; const warns = [];
    pools.forEach((a, i) => pools.forEach((b, j) => {
      if (i < j && cross(a, b) && V[i][j]) bad++;
      if (i < j && same(a, b) && V[i][j]) warns.push(`${a} ↔ ${b}: ${V[i][j]}`);
    }));
    const sel = S.f.shared ? [pools.indexOf(S.f.shared[0]), pools.indexOf(S.f.shared[1])] : null;
    S.charts.leak = Charts.heat($("leakChart"), {
      rows: pools, cols: pools, values: V, fmt: (v) => K.int(v), cell: 80, height: 22 + 52 * pools.length + 30, selected: sel, exportName: "people_shared_between_pools",
      color: "var(--neutral)",
      tip: (i, j) => {
        const a = pools[i], b = pools[j];
        if (i === j) return `${K.tipTitle(a)}${K.tipRow(K.color("var(--neutral)"), "people", V[i][j])}${K.tipRow(K.color("var(--neutral)"), "scans", S.recs.filter((r) => r.pool === a).length)}`;
        const subs = [...shared(a, b)].sort();
        const verdict = cross(a, b) ? (subs.length ? "✕ must be 0 (YOLO vs MedSAM2)" : "✓ none, as it must be") : same(a, b) ? (subs.length ? "⚠ train and val share people" : "✓ none") : "test overlap (known, kept)";
        return `${K.tipTitle(`${a} and ${b}`)}${K.tipRow(K.color("var(--neutral)"), "people in both", subs.length)}<div class="faint">${esc(verdict)}</div>
          <div class="faint">${esc(subs.slice(0, 6).join(", "))}${subs.length > 6 ? ", …" : ""}</div>`;
      },
      onClick: (i, j) => {
        const pair = [pools[i], pools[j]];
        S.f.shared = S.f.shared && S.f.shared[0] === pair[0] && S.f.shared[1] === pair[1] ? null : pair;
        update();
      },
    });
    $("leakVerdict").innerHTML = (bad ? `<div class="callout bad"><b>✕ ${bad} YOLO/MedSAM2 pool pair(s) share people.</b> MedSAM2 would learn from YOLO masks on people YOLO already knows.</div>`
      : `<div class="callout good"><b>✓ Nobody is in both a YOLO pool and a MedSAM2 pool.</b></div>`) +
      (warns.length ? `<div class="callout warn" style="margin-top:8px"><b>⚠ Train and val of one model share people</b> (${esc(warns.join(", "))}): that model's val score looks better than it is.</div>` : "") +
      `<p class="note">test shares some people with the training pools since the original split; measured harmless and kept so every test score stays comparable.</p>`;
  }
  function renderManifest() {
    const m = S.manifest;
    if (!m || !m.found) { $("manifest").innerHTML = `<div class="callout bad"><b>✕ split_manifest.json is missing.</b></div>`; return; }
    const disk = {}, ppl = {};
    for (const r of S.recs) { disk[r.pool] = (disk[r.pool] || 0) + 1; (ppl[r.pool] ||= new Set()).add(r.subject); }
    const drift = m.moved.length + m.not_in_manifest.length + m.missing_on_disk.length;
    const moves = (m.moves || []).slice().reverse();
    $("manifest").innerHTML = `
      <div class="tbl-wrap short"><table class="tbl"><thead><tr><th>Pool</th><th class="num">In the list</th><th class="num">In the folder</th><th class="num">People</th><th></th></tr></thead><tbody>
      ${S.pools.map((p) => { const a = m.counts ? m.counts[p.key] : null, b = disk[p.key] || 0;
        return `<tr><td><i class="dot ${p.role === "val" ? "ring" : ""}" style="${p.role === "val" ? "border-color" : "background"}:${K.FAMILY[p.family]}"></i> ${esc(p.key)}</td>
          <td class="num">${K.int(a)}</td><td class="num">${K.int(b)}</td><td class="num">${K.int((ppl[p.key] || new Set()).size)}</td><td>${a === b ? '<span class="good">✓</span>' : '<span class="bad">✕</span>'}</td></tr>`; }).join("")}
      </tbody></table></div>
      ${drift ? `<div class="callout bad" style="margin-top:8px"><b>✕ ${m.moved.length} moved, ${m.not_in_manifest.length} not in the list, ${m.missing_on_disk.length} missing from the folders.</b>
          ${m.moved.slice(0, 4).map((x) => `<div class="mono faint">${esc(x.id)}: ${esc(x.manifest)} → ${esc(x.disk)}</div>`).join("")}</div>`
        : `<div class="callout good" style="margin-top:8px"><b>✓ Every patient folder is where the list says.</b></div>`}
      <h4 style="margin:12px 0 6px">Re-splits</h4>
      ${moves.length ? moves.map((mv) => `<div class="row" style="font-size:.82rem;margin-bottom:4px"><span class="badge">${esc(mv.date)}</span>
          <b>${K.int(mv.scans || mv.n_ids)}</b> scans <span class="mono">${esc(mv.from)}</span> → <span class="mono">${esc(mv.to)}</span><span class="faint">seed ${esc(mv.seed)}</span></div>`).join("")
        : `<p class="note">None since the first split.</p>`}
      ${m.note ? `<details style="margin-top:8px"><summary class="dim" style="cursor:pointer">The note in the list</summary><p class="note">${esc(m.note)}</p></details>` : ""}`;
  }
  function renderPools() {
    const all = S.pools.map((p) => p.key);
    const inP = rowsOf("pool");
    const count = (rs, k) => (S.punit === "people" ? new Set(rs.filter((r) => r.pool === k).map((r) => r.subject)).size : rs.filter((r) => r.pool === k).length);
    const series = [
      { name: `whole pool`, color: "var(--faint)", values: all.map((k) => count(S.recs, k)) },
      { name: `your filters`, color: "var(--accent)", values: all.map((k) => count(inP, k)) },
    ];
    S.charts.pools = Charts.bars($("poolChart"), {
      cats: all, series, horizontal: true, yLabel: S.punit === "people" ? "People" : "Scans", exportName: "pool_sizes", legend: true,
      onClick: (ci) => { const k = all[ci]; S.f.pools = S.f.pools.size === 1 && S.f.pools.has(k) ? new Set() : new Set([k]); update(); },
    });
  }
  function renderTimepoints(rows) {
    const fams = ["yolo", "medsam2", "test"];
    const per = {};
    for (const r of rows) (per[famOf(r)] ||= {})[r.subject] = ((per[famOf(r)] || {})[r.subject] || 0) + 1;
    const maxN = Math.max(1, ...Object.values(per).flatMap((o) => Object.values(o)));
    const cats = Array.from({ length: Math.min(maxN, 8) }, (_, i) => (i + 1 === 8 && maxN > 8 ? "8+" : String(i + 1)));
    S.charts.tp = Charts.bars($("tpChart"), {
      cats, yLabel: "People", xLabel: "Scans per person", exportName: "scans_per_person",
      series: fams.map((f) => ({ name: f, color: K.FAMILY[f], values: cats.map((c, i) => Object.values(per[f] || {}).filter((n) => (c === "8+" ? n >= 8 : n === i + 1)).length) })),
    });
  }

  // ------------------------------------------------------------------ distributions
  function renderDist() {
    const m = M[S.dm];
    const rows = rowsOf("pool");
    const groups = S.pools.map((p) => ({ name: p.key, color: K.FAMILY[p.family], ring: p.role === "val",
      values: rows.filter((r) => r.pool === p.key).map((r) => ({ v: m.get(r), id: r.id, tip: tipFor(r, m) })).filter((x) => x.v != null) }));
    S.charts.dist = Charts.box($("distChart"), { groups, yLabel: m.l, log: S.dl, exportName: `pools_${m.k}`, height: 330, onClick: (v) => openPatient(v.id) });
    const test = rows.filter((r) => r.pool === "test").map(m.get);
    $("ksTable").innerHTML = `<div class="tbl-wrap short"><table class="tbl"><thead><tr><th>Pool</th><th class="num">n</th><th class="num">median</th><th class="num">KS D</th><th class="num">p</th><th>Like test?</th></tr></thead><tbody>
      ${S.pools.map((p) => {
        const v = rows.filter((r) => r.pool === p.key).map(m.get);
        const ks = p.key === "test" ? null : S_.ks(v, test);
        const verdict = !ks ? "—" : ks.p < 0.01 ? '<span class="bad">different</span>' : ks.p < 0.05 ? '<span class="warn">slightly</span>' : '<span class="good">alike</span>';
        return `<tr><td>${esc(p.key)}</td><td class="num">${S_.nums(v).length}</td><td class="num">${m.f(S_.median(v))}</td>
          <td class="num">${ks ? ks.d.toFixed(3) : "—"}</td><td class="num">${ks ? (ks.p < 0.001 ? "<0.001" : ks.p.toFixed(3)) : "—"}</td><td>${verdict}</td></tr>`;
      }).join("")}</tbody></table></div>
      <p class="note">D = biggest gap between the two cumulative curves (0 = identical). With hundreds of patients a small D can already be "different"; look at D, not only p.</p>`;
  }
  function renderHist() {
    const m = M[S.hm];
    const rows = rowsOf(`range:${m.k}`);
    const series = S.hby === "family"
      ? ["yolo", "medsam2", "test"].map((f) => ({ name: f, color: K.FAMILY[f], values: rows.filter((r) => famOf(r) === f).map(m.get) }))
      : [{ name: "patients", color: "var(--accent)", values: rows.map(m.get) }];
    const range = S.f.ranges[m.k] || null;
    if (S.hmode === "ecdf") {
      S.charts.hist = Charts.ecdf($("histChart"), { series, xLabel: m.l, exportName: `ecdf_${m.k}`, height: 280,
        markers: range ? [{ x: range[0], label: "from" }, { x: range[1], label: "to" }] : [] });
      return;
    }
    S.charts.hist = Charts.hist($("histChart"), {
      series, log: S.hl, bins: 40, xLabel: m.l, yLabel: S.hby === "family" ? "Share of the family" : "Patients", density: S.hby === "family",
      domainValues: S.recs.map(m.get), brush: true, range, height: 280, exportName: `hist_${m.k}`, overlay: false,
      markers: [{ x: S_.median(rows.map(m.get)), label: "median" }],
      onBrush: (lo, hi) => { if (lo == null) delete S.f.ranges[m.k]; else S.f.ranges[m.k] = [lo, hi]; update(); },
    });
  }
  function renderMinis() {
    const box = $("minis");
    if (!box.children.length) {
      box.innerHTML = MINI.map((k) => `<div class="mini"><h4><span>${esc(M[k].l)}</span><span class="x hidden" data-x="${k}">clear</span></h4><div data-mini="${k}"></div></div>`).join("");
      box.addEventListener("click", (e) => { const x = e.target.closest("[data-x]"); if (x) { delete S.f.ranges[x.dataset.x]; update(); } });
    }
    for (const k of MINI) {
      const m = M[k];
      const rows = rowsOf(`range:${k}`);
      K.$(`[data-x="${k}"]`, box).classList.toggle("hidden", !S.f.ranges[k]);
      Charts.hist(K.$(`[data-mini="${k}"]`, box), {
        series: [{ name: m.l, color: S.f.ranges[k] ? "var(--accent)" : "var(--neutral)", values: rows.map(m.get) }],
        log: m.log, bins: 24, height: 118, brush: true, range: S.f.ranges[k] || null, domainValues: S.recs.map(m.get), yLabel: " ",
        exportName: `mini_${k}`,
        onBrush: (lo, hi) => { if (lo == null) delete S.f.ranges[k]; else S.f.ranges[k] = [lo, hi]; update(); },
      });
    }
  }

  // ------------------------------------------------------------------ composition
  function renderShare(rows) {
    const pools = S.pools.map((p) => p.key);
    const by = (k) => rows.filter((r) => r.pool === k && r.ml.WT > 0);
    S.charts.share = Charts.bars($("shareChart"), {
      cats: pools, stacked: true, horizontal: true, yLabel: "Average share of the whole tumour", fmt: (v) => K.pct(v, 0), exportName: "tumour_makeup",
      series: S.labels.map((l) => ({ name: `${l} · ${K.LABEL_WORDS[l]}`, color: K.LABELS[l],
        values: pools.map((k) => { const rs = by(k); return rs.length ? S_.mean(rs.map((r) => r.ml[l] / r.ml.WT)) : null; }) })),
      yMax: 1,
    });
  }
  function renderPresence(rows) {
    const pools = S.pools.map((p) => p.key);
    S.charts.presence = Charts.bars($("presenceChart"), {
      cats: S.labels.map((l) => `${l} · ${K.LABEL_WORDS[l]}`), yLabel: "Share of patients with that part", fmt: (v) => K.pct(v, 0), yMax: 1, exportName: "part_presence",
      series: S.pools.map((p) => ({ name: p.key, color: K.FAMILY[p.family], stripes: p.role === "val",
        values: S.labels.map((l) => { const rs = rows.filter((r) => r.pool === p.key); return rs.length ? rs.filter((r) => r.present.includes(l)).length / rs.length : null; }) })),
      onClick: (ci) => { const l = S.labels[ci]; S.f.labels.has(l) ? S.f.labels.delete(l) : S.f.labels.add(l); update(); },
    });
  }
  function renderCombos() {
    const rows = rowsOf("combo");
    const cnt = {};
    for (const r of rows) cnt[comboOf(r)] = (cnt[comboOf(r)] || 0) + 1;
    const combos = Object.entries(cnt).sort((a, b) => b[1] - a[1]);
    S.charts.combo = Charts.bars($("comboChart"), {
      cats: combos.map(([c]) => (c === S.f.combo ? `▶ ${c}` : c)), yLabel: "Patients", exportName: "part_combinations", showValues: true,
      series: [{ name: "patients", color: "var(--accent)", values: combos.map(([, n]) => n) }],
      onClick: (ci) => { const c = combos[ci][0]; S.f.combo = S.f.combo === c ? null : c; update(); },
    });
  }

  // ------------------------------------------------------------------ relationships
  function tipFor(r, m) {
    return `${K.tipTitle(r.id)}${K.tipRow(K.color(colOf(r)), r.pool, roleOf(r))}${K.tipRow(K.color("var(--neutral)"), "Whole tumour", `${K.fmt(r.ml.WT)} mL`)}${
      m && m.k !== "wt" ? K.tipRow(K.color("var(--neutral)"), m.l, m.f(m.get(r))) : ""}${K.tipRow(K.color("var(--neutral)"), "parts", r.present.join(", ") || "none")}
      <div class="faint">click to open</div>`;
  }
  function colourBy(r) {
    switch (S.scol) {
      case "side": return { c: { left: "var(--yolo)", right: "var(--medsam2)", bilateral: "var(--test)" }[sideOf(r)] || "var(--neutral)", g: SIDE_WORDS[sideOf(r)] || "unknown" };
      case "cc": return { c: r.n_cc > 1 ? "var(--medsam2)" : "var(--yolo)", g: r.n_cc > 1 ? "several pieces" : "one piece" };
      case "rc": return { c: r.present.includes("RC") ? "var(--rc)" : "var(--neutral)", g: r.present.includes("RC") ? "has cavity" : "no cavity" };
      case "tp": { const n = +r.timepoint - 100; return { c: K.PALETTE[Math.min(n, 6)], g: `scan ${n + 1}` }; }
      default: return { c: colOf(r), g: r.pool, ring: roleOf(r) === "val" };
    }
  }
  function renderScatter() {
    const mx = M[S.sx], my = M[S.sy];
    const rows = rowsOf("sel");
    const groups = {};
    const points = rows.map((r) => {
      const cb = colourBy(r);
      groups[cb.g] = { name: cb.g, color: cb.c, ring: cb.ring };
      return { x: mx.get(r), y: my.get(r), id: r.id, color: cb.c, ring: cb.ring, group: cb.g, tip: tipFor(r, my) };
    });
    S.charts.scatter = Charts.scatter($("scatter"), {
      points, xLabel: mx.l, yLabel: my.l, logX: S.sxl, logY: S.syl, trend: S.trend, height: 380, exportName: `${mx.k}_vs_${my.k}`,
      legend: Object.values(groups), highlight: S.open,
      box: S.f.sel && S.f.sel.x === S.sx && S.f.sel.y === S.sy ? S.f.sel : null,
      onClick: (p) => openPatient(p.id),
      onSelect: (box) => { S.f.sel = box ? { x: S.sx, y: S.sy, ...box } : null; update(); },
    });
  }
  function renderCorr(rows) {
    const V = CORR.map((a) => CORR.map((b) => (a === b ? 1 : S_.spearman(rows.map(M[a].get), rows.map(M[b].get)))));
    S.charts.corr = Charts.heat($("corrChart"), {
      rows: CORR.map((k) => M[k].l), cols: CORR.map((k) => M[k].l), values: V, min: -1, max: 1, diverging: true, cell: 40,
      posColor: "var(--yolo)", negColor: "var(--medsam2)", fmt: (v) => (v == null ? "–" : v.toFixed(2)), exportName: "spearman_matrix",
      tip: (i, j) => `${K.tipTitle(`${M[CORR[i]].l} × ${M[CORR[j]].l}`)}${K.tipRow(K.color("var(--accent)"), "Spearman ρ", V[i][j] == null ? "–" : V[i][j].toFixed(3))}<div class="faint">click to plot this pair</div>`,
      onClick: (i, j) => { S.sx = CORR[j]; S.sy = CORR[i]; S.sxl = M[S.sx].log; S.syl = M[S.sy].log; update(); $("scatter").scrollIntoView({ behavior: "smooth", block: "center" }); },
    });
  }

  // ------------------------------------------------------------------ location
  const bin = (v) => Math.min(NB - 1, Math.max(0, Math.floor(v * NB)));
  function planeXY(r, plane) {  // 0..1 in display space
    const c = r.centroid_rel;
    if (!c) return null;
    if (plane === "axial") return [c[0], c[1]];
    if (plane === "coronal") return [c[0], c[2]];
    return [1 - c[1], c[2]];
  }
  const AXES = {
    axial: { x: ["patient's right", "left"], y: ["back", "front"] },
    coronal: { x: ["patient's right", "left"], y: ["bottom", "top"] },
    sagittal: { x: ["front", "back"], y: ["bottom", "top"] },
  };
  function renderHeat() {
    const rows = rowsOf("cell");
    const grid = Array.from({ length: NB }, () => new Array(NB).fill(0));
    const ids = Array.from({ length: NB }, () => Array.from({ length: NB }, () => []));
    for (const r of rows) { const c = planeXY(r, S.plane); if (c) { grid[bin(c[1])][bin(c[0])]++; ids[bin(c[1])][bin(c[0])].push(r.id); } }
    const ax = AXES[S.plane];
    const rowsL = Array.from({ length: NB }, (_, i) => (i === 0 ? ax.y[1] : i === NB - 1 ? ax.y[0] : ""));
    const colsL = Array.from({ length: NB }, (_, j) => (j === 0 ? ax.x[0] : j === NB - 1 ? ax.x[1] : ""));
    const V = grid.slice().reverse();  // top row = front / top
    const sel = S.f.cell && S.f.cell.plane === S.plane ? [NB - 1 - S.f.cell.j, S.f.cell.i] : null;
    S.charts.heat = Charts.heat($("heatChart"), {
      rows: rowsL, cols: colsL, values: V, cell: 30, selected: sel, rotateCols: false, showValues: true, fmt: (v) => (v ? String(v) : ""), exportName: `tumour_centres_${S.plane}`,
      color: "var(--medsam2)",
      tip: (i, j) => { const list = ids[NB - 1 - i][j]; return `${K.tipTitle(`${list.length} patients`)}<div class="faint">${esc(list.slice(0, 5).join(", "))}${list.length > 5 ? ", …" : ""}</div><div class="faint">click to keep them</div>`; },
      onClick: (i, j) => {
        const cell = { plane: S.plane, i: j, j: NB - 1 - i };
        S.f.cell = S.f.cell && S.f.cell.plane === cell.plane && S.f.cell.i === cell.i && S.f.cell.j === cell.j ? null : cell;
        update();
      },
    });
  }
  function renderZ(rows) {
    const fams = ["yolo", "medsam2", "test"];
    const N = 60;  // resample every profile to 60 points, bottom -> top of its own tumour-bearing head
    const series = [], bands = [];
    for (const f of fams) {
      const rs = rows.filter((r) => famOf(r) === f && r.z_profile_mm2 && r.z_profile_mm2.length);
      if (!rs.length) continue;
      const cols = Array.from({ length: N }, () => []);
      for (const r of rs) {
        const p = r.z_profile_mm2;
        for (let i = 0; i < N; i++) cols[i].push(p[Math.min(p.length - 1, Math.round((i / (N - 1)) * (p.length - 1)))]);
      }
      const med = cols.map((c) => S_.median(c)), q1 = cols.map((c) => S_.quantile(S_.sorted(c), 0.25)), q3 = cols.map((c) => S_.quantile(S_.sorted(c), 0.75));
      series.push({ name: `${f} median`, color: K.FAMILY[f], points: med.map((v, i) => [(i / (N - 1)) * 100, v]) });
      series.push({ name: `${f} top of middle half`, color: K.FAMILY[f], dash: "3 3", width: 1, points: q3.map((v, i) => [(i / (N - 1)) * 100, v]) });
      void q1; void bands;
    }
    S.charts.z = Charts.line($("zChart"), { series, xLabel: "Position in the head, bottom → top (%)", yLabel: "Tumour area (mm²)", height: 250, exportName: "tumour_area_along_head",
      xFmt: (v) => `${Math.round(v)}%`, legend: true });
  }
  function renderSides(rows) {
    const sides = ["right", "bilateral", "left"];
    S.charts.side = Charts.bars($("sideChart"), {
      cats: S.pools.map((p) => p.key), stacked: true, horizontal: true, yLabel: "Share of patients", fmt: (v) => K.pct(v, 0), yMax: 1, exportName: "tumour_side",
      series: sides.map((s, i) => ({ name: SIDE_WORDS[s], color: ["var(--medsam2)", "var(--test)", "var(--yolo)"][i],
        values: S.pools.map((p) => { const rs = rows.filter((r) => r.pool === p.key && sideOf(r)); return rs.length ? rs.filter((r) => sideOf(r) === s).length / rs.length : null; }) })),
      height: 170,
    });
  }

  // ------------------------------------------------------------------ outliers
  function renderOutliers(rows) {
    const keys = ["wt", "et", "netc", "snfh", "brain", "slices", "peak", "ncc"];
    const stats = {};
    // Robust centre and spread per measure, over the patients who HAVE that part: a 0 mL part
    // means "absent", not "extremely small", and mixing the zeros in collapses the spread.
    for (const k of keys) {
      const v = S.recs.map(M[k].get).filter((x) => x != null && x > 0).map((x) => Math.log10(1 + x));
      const med = S_.median(v), mad = Math.max(0.05, S_.median(v.map((x) => Math.abs(x - med))) * 1.4826);
      stats[k] = { med, mad };
    }
    const PIECES_P99 = S_.quantile(S_.sorted(S.recs.map((r) => r.n_cc)), 0.99);
    const out = [];
    for (const r of rows) {
      const why = [];
      let score = 0;
      for (const k of keys) {
        const v = M[k].get(r);
        if (v == null || v <= 0) continue;
        const z = (Math.log10(1 + v) - stats[k].med) / stats[k].mad;
        if (Math.abs(z) > 3.5) { why.push(`${M[k].l} ${z > 0 ? "very high" : "very low"} (${M[k].f(v)})`); score = Math.max(score, Math.abs(z)); }
      }
      if (!r.ml.WT) { why.push("no tumour marked at all"); score = Math.max(score, 9); }
      if (r.n_cc > PIECES_P99) why.push(`${r.n_cc} separate tumour pieces (more than 99% of scans)`);
      if (why.length) out.push({ ...r, why: why.join(" · "), score: score || 3.5 });
    }
    const cols = [
      { k: "id", label: "Patient", html: (v) => `<span class="mono">${esc(v)}</span>` },
      { k: "pool", label: "Pool" },
      { k: "why", label: "Why it is listed" },
      { k: "wt", label: "Whole tumour (mL)", num: true, get: (r) => r.ml.WT, fmt: K.fmt },
      { k: "score", label: "How far out", num: true, fmt: (v) => v.toFixed(1), title: "Largest robust z-score among the measures" },
    ];
    if (!S.tables.out) S.tables.out = K.table($("outTable"), cols, { rows: out, key: (r) => r.id, onRow: (r) => openPatient(r.id), exportName: "unusual_patients", pageSize: 25, sort: { k: "score", dir: -1 }, short: true });
    else S.tables.out.update(out);
  }

  // ------------------------------------------------------------------ patient table
  function renderTable(rows) {
    const maxWT = Math.max(1, ...S.recs.map((r) => r.ml.WT));
    const cols = [
      { k: "id", label: "Patient", html: (v, r) => `${S.cmp.includes(v) ? "★ " : ""}<span class="mono">${esc(v)}</span>` },
      { k: "pool", label: "Pool", html: (v, r) => `<i class="dot ${roleOf(r) === "val" ? "ring" : ""}" style="${roleOf(r) === "val" ? "border-color" : "background"}:${colOf(r)}"></i> ${esc(v)}` },
      { k: "timepoint", label: "Scan", num: true, title: "Timepoint suffix (100 = first scan)" },
      { k: "wt", label: "Whole (mL)", num: true, get: (r) => r.ml.WT, fmt: K.fmt, bar: { max: maxWT, color: "var(--accent)" } },
      { k: "tc", label: "Core (mL)", num: true, get: (r) => r.ml.TC, fmt: K.fmt },
      { k: "et", label: "ET (mL)", num: true, get: (r) => r.ml.ET, fmt: K.fmt, bar: { max: maxWT, color: "var(--et)" } },
      { k: "netc", label: "NETC (mL)", num: true, get: (r) => r.ml.NETC, fmt: K.fmt, bar: { max: maxWT, color: "var(--netc)" } },
      { k: "snfh", label: "SNFH (mL)", num: true, get: (r) => r.ml.SNFH, fmt: K.fmt, bar: { max: maxWT, color: "var(--snfh)" } },
      { k: "rc", label: "RC (mL)", num: true, get: (r) => r.ml.RC, fmt: K.fmt, bar: { max: maxWT, color: "var(--rc)" } },
      { k: "pct", label: "% brain", num: true, get: (r) => r.wt_pct_brain, fmt: K.fmt },
      { k: "slices", label: "Slices", num: true, get: (r) => r.n_slices },
      { k: "ncc", label: "Pieces", num: true, get: (r) => r.n_cc },
      { k: "side", label: "Side", get: (r) => SIDE_WORDS[sideOf(r)] || "–" },
      { k: "parts", label: "Parts", get: (r) => r.present.join(" ") },
      { k: "brain", label: "Brain (mL)", num: true, get: (r) => r.brain_ml, fmt: K.fmt, hidden: true },
      { k: "peak", label: "Biggest slice (mm²)", num: true, get: (r) => r.peak_area_mm2, fmt: K.fmt, hidden: true },
      { k: "size", label: "Size (mm)", get: (r) => (r.extent_mm ? r.extent_mm.map((v) => K.fmt(v)).join("×") : "–"), hidden: true },
    ];
    if (!S.tables.pat) {
      S.tables.pat = K.table($("patTable"), cols, { rows, key: (r) => r.id, onRow: (r) => openPatient(r.id), exportName: "patients",
        sort: { k: "wt", dir: -1 }, selected: S.open, pageSize: 50 });
    } else { S.tables.pat.state.selected = S.open; S.tables.pat.update(rows); }
    S.order = S.tables.pat.rows().map((r) => r.id);
  }

  // ------------------------------------------------------------------ drawer
  function openPatient(id, z) {
    if (!S.byId[id]) return;
    S.open = id;
    S.z = z != null ? z : null;
    toHash();
    renderDrawer();
    if (S.tables.pat) { S.tables.pat.state.selected = id; S.tables.pat.render(); }
    if (S.charts.scatter) S.charts.scatter.update({ highlight: id });
  }
  function closeDrawer() { S.open = null; stopCine(); toHash(); renderDrawer(); }
  function step(d) {
    const order = S.order.length ? S.order : S.recs.map((r) => r.id);
    const i = order.indexOf(S.open);
    const n = order[i + d];
    if (n) openPatient(n);
  }
  const profToZ = (r, i) => (r.axcodes[2] === "S" ? i : r.z_profile_mm2.length - 1 - i);
  const zToProf = (r, z) => (r.axcodes[2] === "S" ? z : r.z_profile_mm2.length - 1 - z);
  let cine = null;
  function stopCine() { if (cine) { clearInterval(cine); cine = null; const b = $("drPlay"); if (b) b.textContent = "▶ Play"; } }
  function renderDrawer() {
    const d = $("drawer");
    const r = S.open ? S.byId[S.open] : null;
    d.classList.toggle("open", !!r);
    d.setAttribute("aria-hidden", r ? "false" : "true");
    if (!r) return;
    const depth = r.shape[2];
    const z = S.z != null ? K.clamp(S.z, 0, depth - 1) : r.peak_z;
    const pool = S.meta[r.pool];
    $("drTitle").textContent = r.id;
    $("drSub").textContent = `${r.pool}${pool ? ` — ${pool.use}` : ""} · person ${r.subject}, scan ${r.timepoint}`;
    $("drPin").textContent = S.cmp.includes(r.id) ? "★ In compare" : "☆ Compare";
    $("drResults").href = `/results/#p=${encodeURIComponent(r.id)}`;
    const pos = S.order.indexOf(r.id);
    $("drPrev").disabled = pos <= 0;
    $("drNext").disabled = pos < 0 || pos >= S.order.length - 1;
    const pr = (get, rs) => { const v = get(r); const vals = rs.map(get).filter((x) => x != null); return v == null || !vals.length ? null : vals.filter((x) => x < v).length / vals.length; };
    const inPool = S.recs.filter((x) => x.pool === r.pool);
    const others = (S.subjScans[r.subject] || []).slice().sort((a, b) => a.timepoint.localeCompare(b.timepoint));
    const kinds = [...S.kinds, { key: "all", label: "All" }];
    const kv = [
      ["Whole tumour", `${K.fmt(r.ml.WT)} mL`, `bigger than ${K.pct(pr((x) => x.ml.WT, inPool), 0)} of ${r.pool} · ${K.pct(pr((x) => x.ml.WT, S.recs), 0)} of all`],
      ["Tumour core", `${K.fmt(r.ml.TC)} mL`, "dead core + enhancing + cavity"],
      ["Brain", `${K.fmt(r.brain_ml)} mL`, `tumour = ${K.fmt(r.wt_pct_brain)}% of it`],
      ["Slices with tumour", `${r.n_slices}`, `biggest ${K.fmt(r.peak_area_mm2)} mm² at z = ${r.peak_z}`],
      ["Pieces", `${r.n_cc}`, r.largest_cc_frac != null ? `biggest holds ${K.pct(r.largest_cc_frac, 0)}` : ""],
      ["Size", r.extent_mm ? `${r.extent_mm.map((v) => K.fmt(v)).join(" × ")} mm` : "–", "width × length × height"],
      ["Side", SIDE_WORDS[sideOf(r)] || "–", r.left_frac != null ? `${K.pct(r.left_frac, 0)} on the left` : ""],
      ["Scan", r.shape.join(" × "), `${r.spacing_mm.join(" × ")} mm · ${r.axcodes}`],
    ].map(([k, v, s]) => `<div class="kpi"><span class="l">${k}</span><span class="v" style="font-size:1.05rem">${esc(v)}</span><span class="s">${esc(s)}</span></div>`).join("");
    $("drBody").innerHTML = `
      <div class="row"><div class="seg" id="drMods">${kinds.map((k, i) => `<button data-mod="${k.key}" ${k.key === "all" || r.images.includes(k.key) ? "" : "disabled"} title="${esc(k.label)} (${i + 1})">${esc(k.label)}</button>`).join("")}</div>
        <label class="inline"><input type="checkbox" id="drOv" ${S.ov ? "checked" : ""}> tumour parts (L)</label></div>
      <div id="drImgs"></div>
      <div class="row">
        <button class="small" id="drPlay" title="Play through the tumour slices (P)">▶ Play</button>
        <button class="small" id="drPeak" title="Jump to the slice with the most tumour">Biggest slice</button>
        <input type="range" id="drZ" min="0" max="${depth - 1}" value="${z}" style="flex:1" aria-label="Slice">
        <b class="mono" id="drZL">z = ${z}</b>
      </div>
      <div class="row" style="gap:12px">${S.labels.map((l) => `<span class="lg" style="--c:${K.LABELS[l]}">${l} · ${K.LABEL_WORDS[l]}</span>`).join("")}</div>
      <div><h3>Tumour area on each slice</h3><div id="drProf"></div></div>
      <div><h3>Tumour parts</h3><div id="drParts"></div></div>
      <div class="kpis">${kv}</div>
      <div><h3>This person's scans</h3>${others.length > 1 ? `<div id="drTime"></div>` : `<p class="note">Only one scan of this person.</p>`}
        <div class="chips" style="margin-top:6px">${others.map((o) => `<button class="chip ${o.id === r.id ? "on" : ""}" data-open="${esc(o.id)}"><i class="dot" style="background:${colOf(o)}"></i>${esc(o.timepoint)} · ${esc(o.pool)} · ${K.fmt(o.ml.WT)} mL</button>`).join("")}</div></div>`;
    const setZ = (nz) => {
      S.z = K.clamp(nz, 0, depth - 1);
      $("drZ").value = S.z;
      $("drZL").textContent = `z = ${S.z}`;
      drawImgs(r);
      if (S.charts.prof) S.charts.prof.update({ markers: [{ x: S.z, label: `z ${S.z}`, color: "var(--accent)", solid: true }] });
      toHash();
    };
    S.setZ = setZ;
    $("drZ").addEventListener("input", (e) => setZ(+e.target.value));
    $("drPeak").addEventListener("click", () => setZ(r.peak_z));
    $("drPlay").addEventListener("click", () => {
      if (cine) return stopCine();
      const tz = r.z_profile_mm2.map((v, i) => [v, profToZ(r, i)]).filter(([v]) => v > 0).map(([, zz]) => zz).sort((a, b) => a - b);
      if (!tz.length) return;
      let k = Math.max(0, tz.indexOf(S.z != null ? S.z : r.peak_z));
      $("drPlay").textContent = "■ Stop";
      cine = setInterval(() => { k = (k + 1) % tz.length; setZ(tz[k]); }, 220);
    });
    $("drMods").addEventListener("click", (e) => { const b = e.target.closest("button[data-mod]"); if (b && !b.disabled) { S.img = b.dataset.mod; drawImgs(r); toHash(); } });
    $("drOv").addEventListener("change", (e) => { S.ov = e.target.checked; drawImgs(r); toHash(); });
    K.$$("[data-open]", $("drBody")).forEach((b) => b.addEventListener("click", () => openPatient(b.dataset.open)));
    drawImgs(r);
    S.charts.prof = Charts.line($("drProf"), {
      series: [{ name: "tumour area", color: "var(--accent)", area: true, points: r.z_profile_mm2.map((v, i) => [profToZ(r, i), v]).sort((a, b) => a[0] - b[0]) }],
      xLabel: "Slice (z)", yLabel: "mm²", height: 170, legend: false, zoom: false, exportName: `${r.id}_profile`,
      markers: [{ x: z, label: `z ${z}`, color: "var(--accent)", solid: true }], onClick: (x) => setZ(Math.round(x)),
    });
    void zToProf;
    Charts.bars($("drParts"), { cats: S.labels.map((l) => `${l} · ${K.LABEL_WORDS[l]}`), horizontal: true, yLabel: "mL", exportName: `${r.id}_parts`,
      series: [{ name: "volume", color: "var(--neutral)", values: S.labels.map((l) => r.ml[l]) }], height: 150 });
    // colour each part bar with its own label colour
    K.$$("#drParts rect.hot").forEach((rect, i) => rect.setAttribute("fill", K.color(K.LABELS[S.labels[i]])));
    if (others.length > 1) {
      Charts.line($("drTime"), {
        series: ["WT", "TC", "ET"].map((k, i) => ({ name: k === "WT" ? "whole" : k === "TC" ? "core" : "enhancing", color: ["var(--accent)", "var(--medsam2)", "var(--et)"][i], dots: true,
          points: others.map((o, j) => [j + 1, o.ml[k]]) })),
        xLabel: "Scan number", yLabel: "mL", height: 180, xFmt: (v) => String(Math.round(v)), zoom: false, exportName: `${r.subject}_timeline`,
        markers: [{ x: others.findIndex((o) => o.id === r.id) + 1, label: "this scan" }],
        onClick: (x) => { const o = others[Math.round(x) - 1]; if (o && o.id !== r.id) openPatient(o.id); },
      });
    }
  }
  function drawImgs(r) {
    const box = $("drImgs");
    if (!box) return;
    const z = S.z != null ? S.z : r.peak_z;
    K.$$("#drMods button").forEach((b) => b.classList.toggle("on", b.dataset.mod === S.img));
    const src = (k) => `/thumb.png?pid=${encodeURIComponent(r.id)}&z=${z}&mod=${k}&ov=${S.ov ? 1 : 0}`;
    if (S.img === "all") {
      box.innerHTML = `<div class="thumbs">${S.kinds.filter((k) => r.images.includes(k.key)).map((k) =>
        `<figure data-k="${k.key}"><img src="${src(k.key)}" alt="${esc(k.label)} slice ${z}" loading="lazy"><figcaption>${esc(k.label)}</figcaption></figure>`).join("")}</div>`;
      K.$$("figure", box).forEach((f) => f.addEventListener("click", () => { S.img = f.dataset.k; drawImgs(r); toHash(); }));
    } else {
      const img = K.$("img", box);
      if (img && box.dataset.mode === "one") { img.src = src(S.img); img.alt = `${S.img} slice ${z}`; }
      else { box.dataset.mode = "one"; box.innerHTML = `<div class="imgbox" style="max-width:420px;margin:auto"><img src="${src(S.img)}" alt="${S.img} slice ${z}"></div>`; }
      // warm the next slices, so Play and the slider feel instant
      for (const dz of [1, 2]) { const i = new Image(); i.src = `/thumb.png?pid=${encodeURIComponent(r.id)}&z=${Math.min(r.shape[2] - 1, z + dz)}&mod=${S.img}&ov=${S.ov ? 1 : 0}`; }
      return;
    }
    box.dataset.mode = "all";
  }

  // ------------------------------------------------------------------ compare tray
  function togglePin(id) {
    if (S.cmp.includes(id)) S.cmp = S.cmp.filter((x) => x !== id);
    else if (S.cmp.length >= 4) return K.toast("Up to 4 patients can be compared.");
    else S.cmp.push(id);
    toHash(); renderTray(); renderDrawer();
    if (S.tables.pat) S.tables.pat.render();
  }
  function renderTray() {
    $("tray").classList.toggle("hidden", !S.cmp.length);
    $("trayChips").innerHTML = S.cmp.map((id) => `<button class="chip on" data-rm="${esc(id)}">${esc(id)}<span class="x">✕</span></button>`).join("");
  }
  function openCompare() {
    const rs = S.cmp.map((id) => S.byId[id]).filter(Boolean);
    if (!rs.length) return;
    const rowsDef = [
      ["Pool", (r) => r.pool], ["Whole tumour (mL)", (r) => K.fmt(r.ml.WT)], ["Core (mL)", (r) => K.fmt(r.ml.TC)], ["Enhancing (mL)", (r) => K.fmt(r.ml.ET)],
      ["Dead core (mL)", (r) => K.fmt(r.ml.NETC)], ["Swelling (mL)", (r) => K.fmt(r.ml.SNFH)], ["Cavity (mL)", (r) => K.fmt(r.ml.RC)],
      ["% of brain", (r) => K.fmt(r.wt_pct_brain)], ["Slices with tumour", (r) => r.n_slices], ["Pieces", (r) => r.n_cc],
      ["Side", (r) => SIDE_WORDS[sideOf(r)] || "–"], ["Size (mm)", (r) => (r.extent_mm ? r.extent_mm.map((v) => K.fmt(v)).join(" × ") : "–")],
    ];
    const html = `<div class="thumbs" style="grid-template-columns:repeat(${rs.length},1fr)">${rs.map((r) =>
      `<figure><img src="/thumb.png?pid=${encodeURIComponent(r.id)}&z=${r.peak_z}&mod=flair&ov=1" alt=""><figcaption class="mono">${esc(r.id)} · z ${r.peak_z}</figcaption></figure>`).join("")}</div>
      <div class="tbl-wrap" style="margin-top:12px"><table class="tbl"><thead><tr><th></th>${rs.map((r) => `<th class="mono">${esc(r.id)}</th>`).join("")}</tr></thead><tbody>
      ${rowsDef.map(([l, g]) => `<tr><td class="dim">${esc(l)}</td>${rs.map((r) => `<td class="num">${esc(g(r))}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
      <div id="cmpProf" style="margin-top:12px"></div>`;
    const m = K.modal("Compare patients", html, { wide: true });
    Charts.line(K.$("#cmpProf", m.el), {
      series: rs.map((r, i) => ({ name: r.id, color: K.PALETTE[i], points: r.z_profile_mm2.map((v, j) => [(j / Math.max(1, r.z_profile_mm2.length - 1)) * 100, v]) })),
      xLabel: "Position in the head, bottom → top (%)", yLabel: "Tumour area (mm²)", height: 240, xFmt: (v) => `${Math.round(v)}%`, exportName: "compare_profiles",
    });
  }

  // ------------------------------------------------------------------ export
  function exportCsv() {
    const rows = rowsOf();
    const cols = [{ k: "id" }, { k: "pool" }, { k: "subject" }, { k: "timepoint" },
      ...Object.values(M).map((m) => ({ label: m.k, raw: m.get })), { label: "parts", raw: (r) => r.present.join(" ") }, { label: "side", raw: sideOf }];
    K.download(`brats_patients_${rows.length}.csv`, K.csv(rows, cols), "text/csv");
    K.toast(`Saved ${rows.length} patients.`, "good");
  }

  // ------------------------------------------------------------------ wiring
  function wire() {
    $("poolChips").addEventListener("click", (e) => { const b = e.target.closest("[data-pool]"); if (!b) return; const k = b.dataset.pool; S.f.pools.has(k) ? S.f.pools.delete(k) : S.f.pools.add(k); update(); });
    $("labelChips").addEventListener("click", (e) => { const b = e.target.closest("[data-label]"); if (!b) return; const k = b.dataset.label; S.f.labels.has(k) ? S.f.labels.delete(k) : S.f.labels.add(k); update(); });
    $("sideSeg").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { S.f.side = b.dataset.v; update(); } });
    $("search").addEventListener("input", K.debounce((e) => { S.f.q = e.target.value.trim(); update(); }, 200));
    $("activeChips").addEventListener("click", (e) => {
      const b = e.target.closest("[data-clear]"); if (!b) return;
      const k = b.dataset.clear;
      if (k.startsWith("range:")) delete S.f.ranges[k.slice(6)]; else S.f[k] = null;
      update();
    });
    $("resetBtn").addEventListener("click", reset);
    $("minisClear").addEventListener("click", () => { S.f.ranges = {}; update(); });
    const bindSel = (id, fn) => $(id).addEventListener("change", (e) => { fn(e.target); update(); });
    bindSel("distMetric", (t) => { S.dm = t.value; S.dl = M[t.value].log; });
    bindSel("distLog", (t) => { S.dl = t.checked; });
    bindSel("histMetric", (t) => { S.hm = t.value; S.hl = M[t.value].log; });
    bindSel("histLog", (t) => { S.hl = t.checked; });
    bindSel("scX", (t) => { S.sx = t.value; S.sxl = M[t.value].log; });
    bindSel("scY", (t) => { S.sy = t.value; S.syl = M[t.value].log; });
    bindSel("scXLog", (t) => { S.sxl = t.checked; });
    bindSel("scYLog", (t) => { S.syl = t.checked; });
    bindSel("scColor", (t) => { S.scol = t.value; });
    bindSel("scTrend", (t) => { S.trend = t.checked; });
    const seg = (id, key) => $(id).addEventListener("click", (e) => { const b = e.target.closest("button[data-v]"); if (b) { S[key] = b.dataset.v; update(); } });
    seg("histMode", "hmode"); seg("histBy", "hby"); seg("planeSeg", "plane"); seg("poolUnit", "punit");
    $("drClose").addEventListener("click", closeDrawer);
    $("drPrev").addEventListener("click", () => step(-1));
    $("drNext").addEventListener("click", () => step(1));
    $("drPin").addEventListener("click", () => S.open && togglePin(S.open));
    $("trayChips").addEventListener("click", (e) => { const b = e.target.closest("[data-rm]"); if (b) togglePin(b.dataset.rm); });
    $("trayOpen").addEventListener("click", openCompare);
    $("trayClear").addEventListener("click", () => { S.cmp = []; toHash(); renderTray(); renderDrawer(); });
    $("csvBtn").addEventListener("click", exportCsv);
    $("recomputeBtn").addEventListener("click", async () => {
      if (!confirm("Measure every patient again from the scan files? It takes a few minutes; the page updates when done.")) return;
      try { await K.post("/api/dashboard/recompute"); poll(); } catch (e) { K.fail(e); }
    });
    window.addEventListener("hashchange", () => { fromHash(); render(); });

    K.key("/", "Search patients", () => $("search").focus());
    K.key("r", "Reset every filter", reset);
    K.key("escape", "Close the patient", () => S.open && closeDrawer());
    K.key("arrowleft", "Previous patient (drawer open)", () => S.open && step(-1));
    K.key("arrowright", "Next patient (drawer open)", () => S.open && step(1));
    K.key("[", "Previous slice", () => S.open && S.setZ((S.z != null ? S.z : S.byId[S.open].peak_z) - 1));
    K.key("]", "Next slice", () => S.open && S.setZ((S.z != null ? S.z : S.byId[S.open].peak_z) + 1));
    K.key("p", "Play / stop the slices", () => S.open && $("drPlay") && $("drPlay").click());
    K.key("l", "Tumour parts on / off", () => { if (S.open) { S.ov = !S.ov; renderDrawer(); toHash(); } });
    K.key("c", "Add the open patient to compare", () => S.open && togglePin(S.open));
    ["flair", "t1c", "t1", "t2", "sub", "all"].forEach((k, i) => K.key(String(i + 1), `Image: ${k}`, () => { if (S.open) { S.img = k; renderDrawer(); toHash(); } }));
    K.command("Save the filtered patients as CSV", exportCsv, { group: "Analytics" });
    K.command("Reset every filter", reset, { group: "Analytics", hint: "R" });
    K.command("Compare the pinned patients", openCompare, { group: "Analytics" });
    K.provider((q) => S.recs.filter((r) => r.id.toLowerCase().includes(q)).slice(0, 12)
      .map((r) => ({ label: `Open patient ${r.id}`, hint: `${r.pool} · ${K.fmt(r.ml.WT)} mL`, group: "Patients", run: () => openPatient(r.id) })));
  }

  document.addEventListener("DOMContentLoaded", () => { wire(); poll(); });
})();
