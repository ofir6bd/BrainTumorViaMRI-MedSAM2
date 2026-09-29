/* Results — every scored YOLO and MedSAM2 run, joined patient by patient with the dataset.
 *
 * Scores: /results/api/scores/<model>/<run>/<split> (what each run saved when it was scored;
 * no GPU). Patients: /api/dashboard/data. The join is by patient id, so every score can be
 * split by what the tumour looks like — the "where does it fail" questions.
 *
 * Pools: `test` is scored by both models. `medsam2_val` is MedSAM2's check pool (with the YOLO
 * hint it was given); `yolo_val` is YOLO's. Nothing is recomputed here except summaries of the
 * saved per-patient numbers (means, bootstrap ranges, paired tests).
 */
(function () {
  "use strict";
  const K = Kit, $ = K.id, esc = K.esc, St = K.stats;
  const SPLIT = { test: { yolo: "test", medsam2: "test" }, medsam2_val: { yolo: null, medsam2: "val" }, yolo_val: { yolo: "val", medsam2: null } };
  const COL = { yolo: "var(--yolo)", hint: "var(--neutral)", med: "var(--medsam2)" };
  const NAME = { yolo: "YOLO", hint: "YOLO hint", med: "MedSAM2" };

  const S = {
    src: null, recs: {}, cache: {},
    pool: "test", yolo: null, med: null, yt: null, mt: null, ref: "hint",
    f: { parts: new Set(), group: null, range: null, box: null, q: "" },
    strat: "size", sizeModel: "med", ab: "medsam2", abA: null, abB: null,
    rows: [], open: null, z: null, pics: false, charts: {}, tables: {}, order: [],
  };

  // ------------------------------------------------------------------ data
  async function scores(model, run, split) {
    if (!run || !split) return null;
    const key = `${model}:${run}:${split}`;
    if (!S.cache[key]) {
      S.cache[key] = K.api(`/results/api/scores/${model}/${encodeURIComponent(run)}/${split}`).catch((e) => { delete S.cache[key]; throw e; });
    }
    return S.cache[key];
  }
  const runOf = (model, id) => (S.src[model] || []).find((r) => r.id === id);
  const hasSplit = (model, id, split) => { const r = runOf(model, id); return !!(r && split && r.splits.includes(split)); };

  async function build() {
    const sp = SPLIT[S.pool];
    const [ys, ms] = await Promise.all([
      hasSplit("yolo", S.yolo, sp.yolo) ? scores("yolo", S.yolo, sp.yolo) : null,
      hasSplit("medsam2", S.med, sp.medsam2) ? scores("medsam2", S.med, sp.medsam2) : null,
    ]).catch((e) => { K.fail(e); return [null, null]; });
    S.ys = ys; S.ms = ms;
    if (ys && (S.yt == null || S.yt >= ys.thresholds.length)) S.yt = ys.at;
    if (ms && (S.mt == null || S.mt >= ms.thresholds.length)) S.mt = ms.at;
    const by = {};
    if (ys) for (const r of ys.rows) (by[r.id] ||= { id: r.id }).y = r;
    if (ms) for (const r of ms.rows) (by[r.id] ||= { id: r.id }).m = r;
    S.rows = Object.values(by).map((o) => {
      const rec = S.recs[o.id] || null;
      const yolo = o.y ? o.y.dice[S.yt] : null;
      const med = o.m ? o.m.dice[S.mt] : null;
      const hint = o.m ? o.m.hint : null;
      const ref = S.ref === "yolo" ? yolo : hint;
      return { id: o.id, rec, yolo, med, hint, delta: med != null && ref != null ? med - ref : null, y: o.y, m: o.m,
               wt: rec ? rec.ml.WT : null, gt: (o.m || o.y).gt };
    });
    renderSources();
    render();
  }

  // ------------------------------------------------------------------ groups (the "where" split)
  function edges(vals, k = 5) { const s = St.sorted(vals); return Array.from({ length: k - 1 }, (_, i) => St.quantile(s, (i + 1) / k)); }
  function grouper() {
    const rows = S.rows;
    const fifths = (get, unit) => {
      const e = edges(rows.map(get));
      const lab = (i) => (i === 0 ? `< ${K.fmt(e[0])}${unit}` : i === e.length ? `≥ ${K.fmt(e[e.length - 1])}${unit}` : `${K.fmt(e[i - 1])}–${K.fmt(e[i])}${unit}`);
      return { order: Array.from({ length: e.length + 1 }, (_, i) => lab(i)), of: (r) => { const v = get(r); if (v == null) return null; let i = 0; while (i < e.length && v >= e[i]) i++; return lab(i); } };
    };
    switch (S.strat) {
      case "size": return fifths((r) => r.wt, " mL");
      case "slices": return fifths((r) => (r.rec ? r.rec.n_slices : null), "");
      case "parts": {
        const of = (r) => (r.rec ? (r.rec.present.join("+") || "none") : null);
        const cnt = {}; rows.forEach((r) => { const g = of(r); if (g) cnt[g] = (cnt[g] || 0) + 1; });
        return { order: Object.keys(cnt).sort((a, b) => cnt[b] - cnt[a]), of };
      }
      case "rc": return { order: ["has a cavity", "no cavity"], of: (r) => (r.rec ? (r.rec.present.includes("RC") ? "has a cavity" : "no cavity") : null) };
      case "et": return { order: ["has enhancing", "no enhancing"], of: (r) => (r.rec ? (r.rec.present.includes("ET") ? "has enhancing" : "no enhancing") : null) };
      case "side": return { order: ["left", "both sides", "right"], of: (r) => { const f = r.rec && r.rec.left_frac; return f == null ? null : f > 2 / 3 ? "left" : f < 1 / 3 ? "right" : "both sides"; } };
      case "pieces": return { order: ["1 piece", "2–3 pieces", "4+ pieces"], of: (r) => (r.rec ? (r.rec.n_cc <= 1 ? "1 piece" : r.rec.n_cc <= 3 ? "2–3 pieces" : "4+ pieces") : null) };
      case "height": return { order: ["bottom third", "middle third", "top third"], of: (r) => { const c = r.rec && r.rec.centroid_rel; return !c ? null : c[2] < 1 / 3 ? "bottom third" : c[2] < 2 / 3 ? "middle third" : "top third"; } };
      default: return { order: [], of: () => null };
    }
  }

  // ------------------------------------------------------------------ filtering
  function passes(r, skip) {
    const f = S.f;
    if (f.q && !r.id.toLowerCase().includes(f.q.toLowerCase())) return false;
    for (const p of f.parts) if (!r.rec || !r.rec.present.includes(p)) return false;
    if (skip !== "group" && f.group && S.G.of(r) !== f.group) return false;
    if (skip !== "range" && f.range && (r.delta == null || r.delta < f.range[0] || r.delta > f.range[1])) return false;
    if (skip !== "box" && f.box) {
      const x = r[S.ref === "yolo" ? "yolo" : "hint"], y = r.med;
      if (x == null || y == null || x < f.box.x0 || x > f.box.x1 || y < f.box.y0 || y > f.box.y1) return false;
    }
    return true;
  }
  const rowsOf = (skip) => S.rows.filter((r) => passes(r, skip));

  // ------------------------------------------------------------------ URL
  function toHash() {
    const f = S.f;
    K.hashSet({ pool: S.pool, y: S.yolo, m: S.med, yt: S.yt, mt: S.mt, ref: S.ref, strat: S.strat, sm: S.sizeModel,
      parts: [...f.parts].join(","), g: f.group || "", range: f.range ? f.range.map((v) => v.toFixed(4)).join(":") : "",
      box: f.box ? [f.box.x0, f.box.x1, f.box.y0, f.box.y1].map((v) => v.toFixed(3)).join(":") : "", q: f.q,
      ab: S.ab, a: S.abA, b: S.abB, p: S.open, z: S.open && S.z != null ? S.z : "" });
  }
  function fromHash() {
    const q = K.hashGet();
    if (SPLIT[q.pool]) S.pool = q.pool;
    if (q.y) S.yolo = q.y;
    if (q.m) S.med = q.m;
    S.yt = q.yt !== undefined && q.yt !== "" ? +q.yt : S.yt;
    S.mt = q.mt !== undefined && q.mt !== "" ? +q.mt : S.mt;
    S.ref = q.ref === "yolo" ? "yolo" : "hint";
    if (q.strat) S.strat = q.strat;
    if (q.sm) S.sizeModel = q.sm;
    S.f.parts = new Set((q.parts || "").split(",").filter(Boolean));
    S.f.group = q.g || null;
    const rg = (q.range || "").split(":");
    S.f.range = rg.length === 2 ? [+rg[0], +rg[1]] : null;
    const bx = (q.box || "").split(":");
    S.f.box = bx.length === 4 ? { x0: +bx[0], x1: +bx[1], y0: +bx[2], y1: +bx[3] } : null;
    S.f.q = q.q || "";
    if (q.ab) S.ab = q.ab;
    if (q.a) S.abA = q.a;
    if (q.b) S.abB = q.b;
    S.open = q.p || null;
    S.z = q.z !== undefined && q.z !== "" ? +q.z : null;
  }

  // ------------------------------------------------------------------ controls
  function runOptions(model, split) {
    return (S.src[model] || []).map((r) => {
      const ev = r[split === "val" ? "val" : "test"];
      const ok = split && r.splits.includes(split);
      return `<option value="${esc(r.id)}" ${ok ? "" : "disabled"}>${esc(r.id)}${r.smoke ? " (quick)" : ""} · ${ok && ev ? `3D Dice ${K.f4(ev.dice3d_mean)}` : "not scored here"}</option>`;
    }).join("");
  }
  function renderSources() {
    const sp = SPLIT[S.pool];
    K.$$("#poolSeg button").forEach((b) => b.classList.toggle("on", b.dataset.v === S.pool));
    $("yoloRun").innerHTML = sp.yolo ? runOptions("yolo", sp.yolo) : `<option>not scored on ${S.pool}</option>`;
    $("yoloRun").disabled = !sp.yolo;
    if (sp.yolo) $("yoloRun").value = S.yolo;
    $("medRun").innerHTML = sp.medsam2 ? runOptions("medsam2", sp.medsam2) : `<option>not scored on ${S.pool}</option>`;
    $("medRun").disabled = !sp.medsam2;
    if (sp.medsam2) $("medRun").value = S.med;
    $("yoloThr").innerHTML = S.ys ? S.ys.thresholds.map((t, i) => `<option value="${i}">${t}${i === S.ys.at ? " (reported)" : ""}</option>`).join("") : "";
    $("yoloThr").disabled = !S.ys;
    if (S.ys) $("yoloThr").value = S.yt;
    $("medThr").innerHTML = S.ms ? S.ms.thresholds.map((t, i) => `<option value="${i}">${t}${i === S.ms.at ? " (reported)" : ""}</option>`).join("") : "";
    $("medThr").disabled = !S.ms;
    if (S.ms) $("medThr").value = S.mt;
    $("pairX").innerHTML = `<option value="hint" ${S.ms ? "" : "disabled"}>vs its own YOLO hint (same slices)</option><option value="yolo" ${S.ys ? "" : "disabled"}>vs the YOLO run chosen above</option>`;
    $("pairX").value = S.ref;
    const notes = [];
    const mr = runOf("medsam2", S.med);
    if (S.ms && S.ys && mr && mr.yolo_run && mr.yolo_run !== S.yolo)
      notes.push(`MedSAM2 run <b>${esc(S.med)}</b> was given hints by YOLO run <b>${esc(mr.yolo_run)}</b>, not by <b>${esc(S.yolo)}</b> chosen above. “vs its own YOLO hint” is the fair pairing.`);
    if (S.pool !== "test") notes.push(`<b>${esc(S.pool)}</b> is a check pool: good for choosing settings, but only <b>test</b> is the real result.`);
    if (S.ms && S.mt !== S.ms.at) notes.push("MedSAM2 is shown at a cut-off other than the one it reports.");
    if (S.ys && S.yt !== S.ys.at) notes.push("YOLO is shown at a confidence other than the one it reports.");
    $("note").classList.toggle("hidden", !notes.length);
    $("note").innerHTML = notes.join("<br>");
  }
  function syncChips() {
    const parts = ["NETC", "SNFH", "ET", "RC"];
    const chips = parts.map((p) => `<button class="chip ${S.f.parts.has(p) ? "on" : ""}" data-part="${p}"><i class="dot sq" style="background:${K.LABELS[p]}"></i>has ${p}</button>`);
    if (S.f.group) chips.push(`<button class="chip on" data-clear="group">group: ${esc(S.f.group)}<span class="x">✕</span></button>`);
    if (S.f.range) chips.push(`<button class="chip on" data-clear="range">change ${K.signed(S.f.range[0], 3)} … ${K.signed(S.f.range[1], 3)}<span class="x">✕</span></button>`);
    if (S.f.box) chips.push(`<button class="chip on" data-clear="box">box on the scatter<span class="x">✕</span></button>`);
    $("filterChips").innerHTML = chips.join("");
    if (document.activeElement !== $("search")) $("search").value = S.f.q;
    $("count").textContent = `${rowsOf().length} of ${S.rows.length} patients`;
  }

  // ------------------------------------------------------------------ render
  function render() {
    S.G = grouper();
    syncChips();
    const rows = rowsOf();
    renderKpis(rows);
    renderBoard();
    renderPair(); renderDelta(); renderEcdf(rows); renderSweeps(rows);
    renderStrat(); renderSize(rows); renderSliceErr(rows);
    renderTable(rows);
    renderAB();
    renderDrawer();
  }
  const ci = (v) => { const c = St.bootCI(v); return c ? `95% range ${K.f3(c[0])}–${K.f3(c[1])}` : ""; };
  function renderKpis(rows) {
    const y = rows.map((r) => r.yolo), m = rows.map((r) => r.med), h = rows.map((r) => r.hint), d = rows.map((r) => r.delta);
    const w = St.wilcoxon(d);
    const helped = d.filter((x) => x != null && x > 0.01).length, hurt = d.filter((x) => x != null && x < -0.01).length;
    const worst = rows.filter((r) => r.delta != null).sort((a, b) => a.delta - b.delta)[0];
    const k = [K.kpi("Patients", rows.length.toLocaleString(), `pool ${esc(S.pool)}`, { hero: true })];
    if (S.ms) k.push(K.kpi("MedSAM2 3D Dice", K.f4(St.mean(m)), `median ${K.f4(St.median(m))} · ${ci(m)}`, { cls: "", title: "Mean 3D Dice per patient" }));
    if (S.ms) k.push(K.kpi("YOLO hint 3D Dice", K.f4(St.mean(h)), `median ${K.f4(St.median(h))} · the hint MedSAM2 got`));
    if (S.ys) k.push(K.kpi("YOLO 3D Dice", K.f4(St.mean(y)), `median ${K.f4(St.median(y))} · ${ci(y)}`));
    if (S.ms && St.nums(d).length) {
      k.push(K.kpi(`Change vs ${S.ref === "yolo" ? "YOLO" : "hint"}`, K.deltaHtml(St.mean(d)), `median ${K.signed(St.median(d))} · ${ci(d)}`, { cls: St.mean(d) >= 0 ? "" : "" }));
      k.push(K.kpi("Is the change real?", w ? St.pfmt(w.p) : "–", w ? `Wilcoxon signed-rank, n = ${w.n}` : "too few patients"));
      k.push(K.kpi("Helped / hurt", `${helped} / ${hurt}`, `by more than 0.01 · ${rows.length - helped - hurt} about the same`));
      if (worst) k.push(K.kpi("Worst patient", K.deltaHtml(worst.delta, 3), esc(worst.id), { click: worst.id }));
    }
    $("kpis").innerHTML = k.join("");
    K.$$("#kpis [data-click]").forEach((el) => el.addEventListener("click", () => openPatient(el.dataset.click)));
  }

  function renderBoard() {
    const rows = [
      ...S.src.yolo.map((r) => ({ ...r, kind: "yolo" })),
      ...S.src.medsam2.map((r) => ({ ...r, kind: "medsam2" })),
    ];
    const cols = [
      { k: "kind", label: "Model", html: (v) => `<span class="badge" style="color:${v === "yolo" ? "var(--yolo)" : "var(--medsam2)"};border-color:currentColor">${v === "yolo" ? "YOLO" : "MedSAM2"}</span>` },
      { k: "id", label: "Run", html: (v, r) => `<span class="mono">${esc(v)}</span>${r.smoke ? ' <span class="badge">quick</span>' : ""}${(r.kind === "yolo" ? S.yolo : S.med) === v ? " ◀" : ""}` },
      { k: "label", label: "Setting", get: (r) => (r.kind === "yolo" ? r.arch : r.label.split(" · ")[1]) },
      { k: "val", label: "Val 3D Dice", num: true, get: (r) => (r.val ? r.val.dice3d_mean : null), fmt: K.f4, title: "yolo_val for YOLO, medsam2_val for MedSAM2" },
      { k: "test", label: "Test 3D Dice", num: true, get: (r) => (r.test ? r.test.dice3d_mean : null), fmt: K.f4, bar: { max: 1, color: (v, r) => (r.kind === "yolo" ? "var(--yolo)" : "var(--medsam2)") } },
      { k: "gain", label: "Test gain vs hint", num: true, get: (r) => (r.test && r.test.delta_mean != null ? r.test.delta_mean : null), html: (v) => (v == null ? "–" : K.deltaHtml(v)) },
      { k: "hh", label: "Helped / hurt", get: (r) => (r.test && r.test.helped != null ? `${r.test.helped} / ${r.test.hurt}` : "–") },
      { k: "hint", label: "Hint YOLO", get: (r) => r.yolo_run || "", hidden: true },
      { k: "created", label: "Created", get: (r) => K.when(r.created) },
    ];
    if (!S.tables.board) S.tables.board = K.table($("board"), cols, { rows, key: (r) => `${r.kind}:${r.id}`, exportName: "all_runs", short: true, pageSize: 25,
      sort: { k: "test", dir: -1 }, onRow: (r) => { if (r.kind === "yolo") S.yolo = r.id; else S.med = r.id; S.yt = S.mt = null; toHash(); build(); } });
    else S.tables.board.render();
  }

  function renderPair() {
    const rows = rowsOf("box");
    const refK = S.ref === "yolo" ? "yolo" : "hint";
    const pts = rows.filter((r) => r[refK] != null && r.med != null).map((r) => ({
      x: r[refK], y: r.med, id: r.id, group: r.delta > 0.01 ? "helped" : r.delta < -0.01 ? "hurt" : "about the same",
      color: r.delta > 0.01 ? "var(--good)" : r.delta < -0.01 ? "var(--bad)" : "var(--neutral)", tip: tipFor(r) }));
    S.charts.pair = Charts.scatter($("pairChart"), {
      points: pts, xLabel: `${NAME[refK]} 3D Dice`, yLabel: "MedSAM2 3D Dice", diagonal: true, xMin: 0, xMax: 1, yMin: 0, yMax: 1, height: 360,
      legend: [{ name: "helped", color: "var(--good)" }, { name: "about the same", color: "var(--neutral)" }, { name: "hurt", color: "var(--bad)" }],
      box: S.f.box, highlight: S.open, exportName: "medsam2_vs_reference", empty: "Choose a pool that MedSAM2 was scored on.",
      onClick: (p) => openPatient(p.id), onSelect: (b) => { S.f.box = b; update(); },
    });
  }
  function renderDelta() {
    const rows = rowsOf("range");
    S.charts.delta = Charts.hist($("deltaChart"), {
      series: [{ name: "patients", color: "var(--medsam2)", values: rows.map((r) => r.delta) }], bins: 41, brush: true, range: S.f.range,
      xLabel: `Change in 3D Dice (MedSAM2 − ${S.ref === "yolo" ? "YOLO" : "hint"})`, yLabel: "Patients", height: 360, exportName: "change_histogram",
      markers: [{ x: 0, label: "no change", color: "var(--faint)" }, { x: St.mean(rows.map((r) => r.delta)) || 0, label: "mean", color: "var(--medsam2)" }],
      empty: "No MedSAM2 scores on this pool.",
      onBrush: (lo, hi) => { S.f.range = lo == null ? null : [lo, hi]; update(); },
    });
  }
  function renderEcdf(rows) {
    const series = [];
    if (S.ys) series.push({ name: "YOLO", color: COL.yolo, values: rows.map((r) => r.yolo) });
    if (S.ms) series.push({ name: "YOLO hint", color: COL.hint, values: rows.map((r) => r.hint), dash: "5 4" });
    if (S.ms) series.push({ name: "MedSAM2", color: COL.med, values: rows.map((r) => r.med) });
    S.charts.ecdf = Charts.ecdf($("ecdfChart"), { series, xLabel: "3D Dice", height: 300, exportName: "dice_ecdf" });
  }
  function renderSweeps(rows) {
    const ids = new Set(rows.map((r) => r.id));
    const sweep = (sc, key) => sc.thresholds.map((t, i) => [t, St.mean(sc.rows.filter((r) => ids.has(r.id)).map((r) => r.dice[i]))]);
    Charts.line($("sweepY"), { series: S.ys ? [{ name: "YOLO", color: COL.yolo, points: sweep(S.ys), dots: true }] : [], xLabel: "YOLO confidence", yLabel: "Mean 3D Dice",
      markers: S.ys ? [{ x: S.ys.thresholds[S.yt], label: "in use" }] : [], height: 240, legend: false, empty: "YOLO was not scored on this pool.", exportName: "yolo_sweep" });
    Charts.line($("sweepM"), { series: S.ms ? [{ name: "MedSAM2", color: COL.med, points: sweep(S.ms), dots: true }] : [], xLabel: "MedSAM2 cut-off (logit)", yLabel: "Mean 3D Dice",
      markers: S.ms ? [{ x: S.ms.thresholds[S.mt], label: "in use" }] : [], height: 240, legend: false, empty: "MedSAM2 was not scored on this pool.", exportName: "medsam2_sweep" });
  }

  function renderStrat() {
    const rows = rowsOf("group");
    const G = S.G;
    const cats = G.order.filter((g) => rows.some((r) => G.of(r) === g));
    const models = [S.ys && "yolo", S.ms && "hint", S.ms && "med"].filter(Boolean);
    const vals = (g, k) => rows.filter((r) => G.of(r) === g).map((r) => r[k]);
    const series = models.map((k) => ({
      name: NAME[k], color: COL[k], stripes: k === "hint",
      values: cats.map((g) => St.mean(vals(g, k))),
      err: cats.map((g) => St.bootCI(vals(g, k), St.mean, 400)),
      tips: cats.map((g) => K.tipRow(K.color("var(--faint)"), "patients", St.nums(vals(g, k)).length)),
    }));
    S.charts.strat = Charts.bars($("stratChart"), {
      cats: cats.map((g) => `${g === S.f.group ? "▶ " : ""}${g} (n=${rows.filter((r) => G.of(r) === g).length})`), series, yLabel: "Mean 3D Dice",
      yMin: 0, yMax: 1, fmt: K.f3, height: 300, exportName: `dice_by_${S.strat}`, legend: true,
      onClick: (ci) => { const g = cats[ci]; S.f.group = S.f.group === g ? null : g; update(); },
    });
    $("stratTable").innerHTML = `<div class="tbl-wrap short"><table class="tbl"><thead><tr><th>Group</th><th class="num">n</th>
      ${models.map((k) => `<th class="num">${NAME[k]}</th>`).join("")}${S.ms ? `<th class="num">Change</th><th class="num">helped / hurt</th><th class="num">p (Wilcoxon)</th>` : ""}</tr></thead><tbody>
      ${cats.map((g) => {
        const rs = rows.filter((r) => G.of(r) === g);
        const d = rs.map((r) => r.delta);
        const w = St.wilcoxon(d);
        return `<tr><td>${esc(g)}</td><td class="num">${rs.length}</td>${models.map((k) => `<td class="num">${K.f4(St.mean(rs.map((r) => r[k])))}</td>`).join("")}
          ${S.ms ? `<td class="num">${K.deltaHtml(St.mean(d))}</td><td class="num">${d.filter((x) => x > 0.01).length} / ${d.filter((x) => x < -0.01).length}</td><td class="num">${w ? (w.p < 0.001 ? "<0.001" : w.p.toFixed(3)) : "–"}</td>` : ""}</tr>`;
      }).join("")}</tbody></table></div>`;
  }
  function renderSize(rows) {
    const k = S.sizeModel;
    const get = (r) => (k === "yolo" ? r.yolo : k === "hint" ? r.hint : k === "delta" ? r.delta : r.med);
    S.charts.size = Charts.scatter($("sizeChart"), {
      points: rows.filter((r) => r.wt != null && get(r) != null).map((r) => ({ x: r.wt, y: get(r), id: r.id, color: k === "delta" ? (r.delta >= 0 ? "var(--good)" : "var(--bad)") : COL[k === "delta" ? "med" : k], tip: tipFor(r) })),
      xLabel: "Whole tumour (mL, log)", yLabel: k === "delta" ? "Change in 3D Dice" : `${NAME[k]} 3D Dice`, logX: true, trend: true, height: 320,
      hlines: k === "delta" ? [{ y: 0 }] : [], highlight: S.open, exportName: `dice_vs_size_${k}`, onClick: (p) => openPatient(p.id),
    });
  }
  function renderSliceErr(rows) {
    const models = [S.ys && ["yolo", (r) => r.y], S.ms && ["med", (r) => r.m]].filter(Boolean);
    Charts.bars($("sliceErrChart"), {
      cats: ["missed tumour slices", "false slices"], yLabel: "Slices", showValues: true, exportName: "slice_errors",
      series: models.map(([k, get]) => ({ name: NAME[k], color: COL[k], values: [St.sum(rows.map((r) => (get(r) ? get(r).missed_slices : 0))), St.sum(rows.map((r) => (get(r) ? get(r).false_slices : 0)))] })),
      height: 260,
    });
  }

  function tipFor(r) {
    const c = (k) => K.color(COL[k]);
    return `${K.tipTitle(r.id)}${r.rec ? K.tipRow(K.color("var(--faint)"), "whole tumour", `${K.fmt(r.wt)} mL`) : ""}${
      r.yolo != null ? K.tipRow(c("yolo"), "YOLO", K.f4(r.yolo)) : ""}${r.hint != null ? K.tipRow(c("hint"), "hint", K.f4(r.hint)) : ""}${
      r.med != null ? K.tipRow(c("med"), "MedSAM2", K.f4(r.med)) : ""}${r.delta != null ? K.tipRow(c("med"), "change", K.signed(r.delta)) : ""}<div class="faint">click to open</div>`;
  }

  function renderTable(rows) {
    const cols = [
      { k: "id", label: "Patient", html: (v) => `<span class="mono">${esc(v)}</span>` },
      { k: "wt", label: "Tumour (mL)", num: true, fmt: K.fmt },
      { k: "parts", label: "Parts", get: (r) => (r.rec ? r.rec.present.join(" ") : "–") },
      { k: "yolo", label: "YOLO", num: true, fmt: K.f4, bar: { max: 1, color: "var(--yolo)" } },
      { k: "hint", label: "Hint", num: true, fmt: K.f4 },
      { k: "med", label: "MedSAM2", num: true, fmt: K.f4, bar: { max: 1, color: "var(--medsam2)" } },
      { k: "delta", label: "Change", num: true, html: (v) => K.deltaHtml(v), raw: (r) => r.delta },
      { k: "missY", label: "YOLO missed sl.", num: true, get: (r) => (r.y ? r.y.missed_slices : null) },
      { k: "missM", label: "MedSAM2 missed sl.", num: true, get: (r) => (r.m ? r.m.missed_slices : null) },
      { k: "falseM", label: "MedSAM2 false sl.", num: true, get: (r) => (r.m ? r.m.false_slices : null), hidden: true },
      { k: "anchor", label: "Start slice", num: true, get: (r) => (r.m ? r.m.anchor_z : null), hidden: true },
      { k: "tslices", label: "Tumour slices", num: true, get: (r) => (r.m || r.y).tumour_slices, hidden: true },
    ];
    if (!S.tables.pat) S.tables.pat = K.table($("patTable"), cols, { rows, key: (r) => r.id, onRow: (r) => openPatient(r.id), exportName: `results_${S.pool}`,
      sort: { k: S.ms ? "delta" : "yolo", dir: 1 }, selected: S.open });
    else { S.tables.pat.state.selected = S.open; S.tables.pat.update(rows); }
    S.order = S.tables.pat.rows().map((r) => r.id);
  }

  // ------------------------------------------------------------------ run vs run
  async function renderAB() {
    K.$$("#abModel button").forEach((b) => b.classList.toggle("on", b.dataset.v === S.ab));
    const split = SPLIT[S.pool][S.ab];
    const opts = runOptions(S.ab, split);
    $("abA").innerHTML = opts; $("abB").innerHTML = opts;
    const ok = (S.src[S.ab] || []).filter((r) => split && r.splits.includes(split)).map((r) => r.id);
    if (!ok.includes(S.abA)) S.abA = ok[0] || null;
    if (!ok.includes(S.abB)) S.abB = ok[1] || ok[0] || null;
    if (S.abA) $("abA").value = S.abA;
    if (S.abB) $("abB").value = S.abB;
    if (!split || !S.abA || !S.abB) { $("abKpis").innerHTML = `<p class="note">${S.ab === "yolo" ? "YOLO" : "MedSAM2"} has no scored runs on ${esc(S.pool)}.</p>`; $("abScatter").innerHTML = $("abHist").innerHTML = ""; return; }
    let a, b;
    try { [a, b] = await Promise.all([scores(S.ab, S.abA, split), scores(S.ab, S.abB, split)]); } catch (e) { K.fail(e); return; }
    const keep = new Set(rowsOf().map((r) => r.id));
    const bm = Object.fromEntries(b.rows.map((r) => [r.id, r.dice[b.at]]));
    const pairs = a.rows.filter((r) => keep.has(r.id) && bm[r.id] != null).map((r) => ({ id: r.id, a: r.dice[a.at], b: bm[r.id] }));
    const d = pairs.map((p) => p.b - p.a);
    const w = St.wilcoxon(d), c = St.bootCI(d);
    $("abKpis").innerHTML = [
      K.kpi("Patients in both", pairs.length),
      K.kpi(`A · ${S.abA}`, K.f4(St.mean(pairs.map((p) => p.a))), "mean 3D Dice"),
      K.kpi(`B · ${S.abB}`, K.f4(St.mean(pairs.map((p) => p.b))), "mean 3D Dice"),
      K.kpi("B minus A", K.deltaHtml(St.mean(d)), c ? `95% range ${K.signed(c[0], 4)} … ${K.signed(c[1], 4)}` : "", { hero: true }),
      K.kpi("Real or noise?", w ? St.pfmt(w.p) : "–", c && c[0] <= 0 && c[1] >= 0 ? "the range includes 0: noise" : "the range excludes 0"),
      K.kpi("B better / worse", `${d.filter((x) => x > 0.005).length} / ${d.filter((x) => x < -0.005).length}`, "by more than 0.005"),
    ].join("");
    Charts.scatter($("abScatter"), { points: pairs.map((p) => ({ x: p.a, y: p.b, id: p.id, color: p.b >= p.a ? "var(--good)" : "var(--bad)" })), diagonal: true,
      xLabel: `A · ${S.abA}`, yLabel: `B · ${S.abB}`, xMin: 0, xMax: 1, yMin: 0, yMax: 1, height: 300, onClick: (p) => openPatient(p.id), exportName: "run_a_vs_b" });
    Charts.hist($("abHist"), { series: [{ name: "patients", color: "var(--accent)", values: d }], bins: 41, xLabel: "B − A (3D Dice)", yLabel: "Patients", height: 300,
      markers: [{ x: 0, label: "no change", color: "var(--faint)" }], exportName: "run_b_minus_a" });
  }

  // ------------------------------------------------------------------ patient drawer
  function openPatient(id) { S.open = id; S.z = null; S.pics = false; toHash(); renderDrawer(); if (S.tables.pat) { S.tables.pat.state.selected = id; S.tables.pat.render(); } }
  function closeDrawer() { S.open = null; toHash(); renderDrawer(); }
  function step(d) { const i = S.order.indexOf(S.open); const n = S.order[i + d]; if (n) openPatient(n); }
  let drawnFor = null;
  async function renderDrawer() {
    const dr = $("drawer");
    const r = S.open ? S.rows.find((x) => x.id === S.open) : null;
    dr.classList.toggle("open", !!S.open);
    dr.setAttribute("aria-hidden", S.open ? "false" : "true");
    if (!S.open) { drawnFor = null; return; }
    $("drTitle").textContent = S.open;
    $("drData").href = `/#p=${encodeURIComponent(S.open)}`;
    const pos = S.order.indexOf(S.open);
    $("drPrev").disabled = pos <= 0; $("drNext").disabled = pos < 0 || pos >= S.order.length - 1;
    if (!r) { $("drSub").textContent = `Not scored on ${S.pool} by the chosen runs.`; $("drBody").innerHTML = ""; return; }
    $("drSub").textContent = `${r.rec ? `${r.rec.pool} · ${K.fmt(r.wt)} mL · parts ${r.rec.present.join(", ") || "none"}` : ""}`;
    const key = `${S.open}|${S.yolo}|${S.med}|${S.pool}`;
    if (drawnFor === key) return updatePics(r);
    drawnFor = key;
    const sp = SPLIT[S.pool];
    const q = new URLSearchParams({ pid: S.open });
    if (S.ys) { q.set("yolo", S.yolo); q.set("yolo_split", sp.yolo); }
    if (S.ms) { q.set("medsam2", S.med); q.set("medsam2_split", sp.medsam2); }
    $("drBody").innerHTML = `<div class="kpis">${[
      r.yolo != null && K.kpi("YOLO", K.f4(r.yolo), `${r.y.missed_slices} missed · ${r.y.false_slices} false slices`),
      r.hint != null && K.kpi("YOLO hint", K.f4(r.hint), "what MedSAM2 was given"),
      r.med != null && K.kpi("MedSAM2", K.f4(r.med), `${r.m.missed_slices} missed · ${r.m.false_slices} false slices`, { hero: true }),
      r.delta != null && K.kpi("Change", K.deltaHtml(r.delta), `vs ${S.ref === "yolo" ? "YOLO" : "hint"}`),
    ].filter(Boolean).join("")}</div>
      <div><h3>Slice by slice</h3><div id="drProf" class="skeleton"></div></div>
      <div><div class="row"><h3 style="flex:1">Pictures</h3><button class="small" id="drPics" title="Runs the models on this slice (uses the graphics card)">${S.pics ? "Hide model pictures" : "Show model pictures (GPU)"}</button></div>
        <div id="drImgs"></div>
        <div class="row"><input type="range" id="drZ" min="0" max="0" style="flex:1" aria-label="Slice"><b class="mono" id="drZL"></b></div>
        <div class="row" style="gap:12px"><span class="lg" style="--c:var(--tp)">found</span><span class="lg" style="--c:var(--fn)">missed</span><span class="lg" style="--c:var(--fp)">drawn, not tumour</span></div></div>`;
    $("drPics").addEventListener("click", () => { S.pics = !S.pics; $("drPics").textContent = S.pics ? "Hide model pictures" : "Show model pictures (GPU)"; updatePics(r); });
    let d;
    try { d = await K.api(`/results/api/patient?${q}`); } catch (e) { $("drProf").classList.remove("skeleton"); $("drProf").innerHTML = `<p class="note">${esc(e.message)}</p>`; return; }
    S.slices = d;
    const base = d.series[0];
    const zs = base.z;
    const dice = (s) => s.z.map((z, i) => [z, s.gt[i] + s.pred[i] ? (2 * s.inter[i]) / (s.gt[i] + s.pred[i]) : null]);
    const colK = { yolo: "yolo", hint: "hint", medsam2: "med" };
    const series = d.series.map((s) => ({ name: `Dice · ${NAME[colK[s.model]]}`, color: COL[colK[s.model]], points: dice(s), dash: s.model === "hint" ? "5 4" : null, width: 1.8 }));
    series.push({ name: "expert tumour px", color: "var(--gt)", axis: "right", area: true, points: zs.map((z, i) => [z, base.gt[i]]), width: 1.2 });
    const anchor = (d.series.find((s) => s.anchor_z) || {}).anchor_z;
    $("drProf").classList.remove("skeleton");
    const peak = zs[base.gt.indexOf(Math.max(...base.gt))];
    if (S.z == null) S.z = peak;
    S.charts.prof = Charts.line($("drProf"), { series, xLabel: "Slice (z)", yLabel: "Dice on the slice", y2Label: "Expert tumour pixels", yMin: 0, yMax: 1, height: 280, zoom: true,
      markers: [{ x: S.z, label: `z ${S.z}`, color: "var(--accent)", solid: true }, ...(anchor ? anchor.map((z) => ({ x: z, label: "MedSAM2 start", color: "var(--medsam2)" })) : [])],
      exportName: `${S.open}_slices`, onClick: (x) => setZ(Math.round(x)) });
    const zr = $("drZ");
    zr.min = zs[0]; zr.max = zs[zs.length - 1]; zr.value = S.z;
    zr.addEventListener("input", () => setZ(+zr.value));
    updatePics(r);
  }
  function setZ(z) {
    const zs = S.slices ? S.slices.series[0].z : [];
    if (zs.length) z = zs.reduce((b, v) => (Math.abs(v - z) < Math.abs(b - z) ? v : b), zs[0]);
    S.z = z;
    if ($("drZ")) $("drZ").value = z;
    if (S.charts.prof) S.charts.prof.update({ markers: [{ x: z, label: `z ${z}`, color: "var(--accent)", solid: true }] });
    toHash();
    updatePics(S.rows.find((x) => x.id === S.open));
  }
  const updatePics = K.debounce((r) => {
    const box = $("drImgs");
    if (!box || !r || S.z == null) return;
    $("drZL").textContent = `z = ${S.z}`;
    const sp = SPLIT[S.pool];
    const figs = [`<figure><img src="/thumb.png?pid=${encodeURIComponent(S.open)}&z=${S.z}&mod=flair&ov=1" alt="FLAIR with expert labels"><figcaption>Expert labels (no model)</figcaption></figure>`];
    if (S.pics && S.ms) figs.push(`<figure style="grid-column:span 3"><img src="/medsam2/api/runs/${encodeURIComponent(S.med)}/slice.png?patient=${encodeURIComponent(S.open)}&z=${S.z}&panels=frame,prompt,yolo,medsam2" alt="MedSAM2 panels"><figcaption>MedSAM2 run ${esc(S.med)}: frame · YOLO hint · YOLO mask · MedSAM2</figcaption></figure>`);
    if (S.pics && S.ys) figs.push(`<figure style="grid-column:span 2"><img src="/finetune/api/runs/${encodeURIComponent(S.yolo)}/predict.png?split=${sp.yolo}&patient=${encodeURIComponent(S.open)}&z=${S.z}&conf=${S.ys.thresholds[S.yt]}" alt="YOLO panels"><figcaption>YOLO run ${esc(S.yolo)} at conf ${S.ys.thresholds[S.yt]}</figcaption></figure>`);
    box.innerHTML = `<div class="thumbs" style="grid-template-columns:repeat(4,1fr)">${figs.join("")}</div>${S.pics ? '<p class="note">Model pictures run the models on this one slice; the first one for a patient takes a few seconds.</p>' : ""}`;
    K.$$("img", box).forEach((im) => im.addEventListener("error", () => { im.replaceWith(K.h(`<p class="note">This picture could not be made (the run's weights may be missing).</p>`)); }));
    K.$$("figure", box).forEach((f) => f.addEventListener("click", () => K.lightbox(K.$("img", f).src, K.$("figcaption", f).textContent)));
  }, 150);

  // ------------------------------------------------------------------ wiring
  const update = () => { toHash(); render(); };
  function wire() {
    $("poolSeg").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; S.pool = b.dataset.v; S.f.group = S.f.range = S.f.box = null; toHash(); build(); });
    $("yoloRun").addEventListener("change", (e) => { S.yolo = e.target.value; S.yt = null; toHash(); build(); });
    $("medRun").addEventListener("change", (e) => { S.med = e.target.value; S.mt = null; toHash(); build(); });
    $("yoloThr").addEventListener("change", (e) => { S.yt = +e.target.value; toHash(); build(); });
    $("medThr").addEventListener("change", (e) => { S.mt = +e.target.value; toHash(); build(); });
    $("pairX").addEventListener("change", (e) => { S.ref = e.target.value; S.f.box = S.f.range = null; toHash(); build(); });
    $("strat").addEventListener("change", (e) => { S.strat = e.target.value; S.f.group = null; update(); });
    $("sizeModel").addEventListener("change", (e) => { S.sizeModel = e.target.value; update(); });
    $("abModel").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { S.ab = b.dataset.v; S.abA = S.abB = null; toHash(); renderAB(); } });
    $("abA").addEventListener("change", (e) => { S.abA = e.target.value; toHash(); renderAB(); });
    $("abB").addEventListener("change", (e) => { S.abB = e.target.value; toHash(); renderAB(); });
    $("filterChips").addEventListener("click", (e) => {
      const p = e.target.closest("[data-part]");
      if (p) { const k = p.dataset.part; S.f.parts.has(k) ? S.f.parts.delete(k) : S.f.parts.add(k); return update(); }
      const c = e.target.closest("[data-clear]");
      if (c) { S.f[c.dataset.clear] = null; update(); }
    });
    $("search").addEventListener("input", K.debounce((e) => { S.f.q = e.target.value.trim(); update(); }, 200));
    $("resetBtn").addEventListener("click", () => { S.f = { parts: new Set(), group: null, range: null, box: null, q: "" }; update(); });
    $("csvBtn").addEventListener("click", () => {
      const rows = rowsOf();
      K.download(`results_${S.pool}_${rows.length}.csv`, K.csv(rows, [
        { label: "patient", raw: (r) => r.id }, { label: "pool", raw: (r) => (r.rec ? r.rec.pool : "") }, { label: "whole_tumour_ml", raw: (r) => r.wt },
        { label: "parts", raw: (r) => (r.rec ? r.rec.present.join(" ") : "") }, { label: `yolo_${S.yolo || ""}`, raw: (r) => r.yolo },
        { label: "yolo_hint", raw: (r) => r.hint }, { label: `medsam2_${S.med || ""}`, raw: (r) => r.med }, { label: "change", raw: (r) => r.delta }]), "text/csv");
    });
    $("drClose").addEventListener("click", closeDrawer);
    $("drPrev").addEventListener("click", () => step(-1));
    $("drNext").addEventListener("click", () => step(1));
    window.addEventListener("hashchange", () => { const was = `${S.pool}|${S.yolo}|${S.med}|${S.yt}|${S.mt}|${S.ref}`; fromHash();
      if (`${S.pool}|${S.yolo}|${S.med}|${S.yt}|${S.mt}|${S.ref}` !== was) build(); else render(); });
    K.key("/", "Search patients", () => $("search").focus());
    K.key("escape", "Close the patient", () => S.open && closeDrawer());
    K.key("arrowleft", "Previous patient", () => S.open && step(-1));
    K.key("arrowright", "Next patient", () => S.open && step(1));
    K.key("[", "Previous slice", () => S.open && S.z != null && setZ(S.z - 1));
    K.key("]", "Next slice", () => S.open && S.z != null && setZ(S.z + 1));
    K.key("m", "Show / hide model pictures", () => S.open && $("drPics") && $("drPics").click());
    K.provider((q) => S.rows.filter((r) => r.id.toLowerCase().includes(q)).slice(0, 10)
      .map((r) => ({ label: `Open scores of ${r.id}`, hint: r.med != null ? `MedSAM2 ${K.f3(r.med)}` : `YOLO ${K.f3(r.yolo)}`, group: "Patients", run: () => openPatient(r.id) })));
    for (const [v, l] of [["size", "tumour size"], ["parts", "parts"], ["side", "side"], ["pieces", "pieces"], ["height", "position"]])
      K.command(`Split scores by ${l}`, () => { S.strat = v; $("strat").value = v; update(); $("secWhere").scrollIntoView({ behavior: "smooth" }); }, { group: "Results" });
  }

  async function boot() {
    wire();
    try {
      const [src, data] = await Promise.all([K.api("/results/api/sources"), K.api("/api/dashboard/data")]);
      S.src = src;
      S.recs = Object.fromEntries(data.records.map((r) => [r.id, r]));
    } catch (e) { K.fail(e); $("pageSub").textContent = e.message; return; }
    const best = (list, split) => (list.filter((r) => !r.smoke && r.splits.includes(split))
      .sort((a, b) => ((b[split] || {}).dice3d_mean || 0) - ((a[split] || {}).dice3d_mean || 0))[0] || {}).id;
    S.yolo = best(S.src.yolo, "test") || best(S.src.yolo, "val");
    S.med = best(S.src.medsam2, "test") || best(S.src.medsam2, "val");
    fromHash();
    $("strat").value = S.strat;
    $("sizeModel").value = S.sizeModel;
    $("pageSub").textContent = `${S.src.yolo.length} scored YOLO run(s) · ${S.src.medsam2.length} scored MedSAM2 run(s) · joined with ${Object.keys(S.recs).length.toLocaleString()} measured scans`;
    await build();
  }
  document.addEventListener("DOMContentLoaded", boot);
})();
