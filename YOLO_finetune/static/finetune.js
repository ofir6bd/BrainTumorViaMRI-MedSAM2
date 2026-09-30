/* YOLO_finetune page — start runs, follow the live one, analyse and compare any run.
 *
 * Everything comes from the run folders (status.json, results.csv, dice_check.csv,
 * eval/*.json); nothing is estimated except the clearly labelled time left. The view
 * (run, pool, confidence, compared runs, chart group, viewer patient / slice) lives in the URL.
 */
(function () {
  "use strict";
  const K = Kit, $ = K.id, esc = K.esc, St = K.stats;
  const BASE = document.body.dataset.base;  // "/finetune/"
  const LIVE = ["queued", "running"];
  const WORD = { queued: "starting", running: "running", done: "finished", failed: "failed", stopped: "stopped", interrupted: "cut off" };
  const CMP = ["var(--yolo)", "var(--medsam2)", "var(--test)", "var(--et)"];
  const badge = (st) => `<span class="badge ${st || ""}">${esc(WORD[st] || st || "–")}</span>`;

  const S = {
    ov: null, runs: [], active: null, sel: null, run: null, cache: {}, cmp: new Set(), group: "loss", split: "test", conf: null,
    evalData: {}, range: null, vModel: null, vSplit: null, vPatient: null, vZ: null, vConf: null, vSlices: null, profile: null,
    pool: {}, logRun: null, logOffset: 0, logLines: [], tables: {}, charts: {},
  };
  const api = (p, o) => K.api(BASE + p, o);
  const runApi = (id, p) => api(`api/runs/${encodeURIComponent(id)}${p || ""}`);

  // ------------------------------------------------------------------ URL
  function toHash() {
    K.hashSet({ run: S.sel, split: S.split !== "test" ? S.split : "", conf: S.conf, group: S.group !== "loss" ? S.group : "",
      cmp: [...S.cmp].join(","), model: S.vModel && S.vModel !== S.sel ? S.vModel : "", pool: S.vSplit && S.vSplit !== S.split ? S.vSplit : "",
      patient: S.vPatient, z: S.vZ, vconf: S.vConf });
  }
  function fromHash() {
    const q = K.hashGet();
    S.sel = q.run || null;
    S.split = q.split === "val" ? "val" : "test";
    S.conf = q.conf ? +q.conf : null;
    S.group = ["loss", "mask", "box", "lr"].includes(q.group) ? q.group : "loss";
    S.cmp = new Set((q.cmp || "").split(",").filter(Boolean));
    S.vModel = q.model || null;
    S.vSplit = q.pool === "val" || q.pool === "test" ? q.pool : null;
    S.vPatient = q.patient || null;
    S.vZ = q.z ? +q.z : null;
    S.vConf = q.vconf ? +q.vconf : null;
  }

  // ------------------------------------------------------------------ overview + runs
  async function loadOverview() {
    const ov = await api("api/overview");
    S.ov = ov; S.runs = ov.runs; S.active = ov.active;
    $("pageSub").innerHTML = `${esc(ov.config.model)} · train <code>${esc(ov.config.data.train_pool)}</code> (${ov.pools.train}) · val <code>${esc(ov.config.data.val_pool)}</code> (${ov.pools.val}) · test <code>${esc(ov.config.data.test_pool)}</code> (${ov.pools.test}) · ${ov.runs.length} run(s)`;
    for (const n of K.$$("[data-cfg]")) n.textContent = n.dataset.cfg.split(".").reduce((o, k) => (o == null ? o : o[k]), ov.config);
    $("newRunBtn").disabled = !!S.active;
    $("newRunBtn").title = S.active ? `Run ${S.active} is still going` : "Start a fine-tune (N)";
    renderRuns();
    if (!S.sel && S.runs.length) selectRun(S.active || S.runs[0].id, true);
  }
  function renderRuns() {
    const cols = [
      { k: "cmp", label: "⊞", nosort: true, title: "Tick to draw this run's curves with the others", html: (v, r) => `<input type="checkbox" class="cmp" data-id="${esc(r.id)}" ${S.cmp.has(r.id) ? "checked" : ""} aria-label="Compare ${esc(r.id)}">` },
      { k: "id", label: "Run", html: (v, r) => `<span class="mono">${esc(v)}</span>${r.smoke ? ' <span class="badge">quick</span>' : ""}${v === S.sel ? " ◀" : ""}` },
      { k: "state", label: "Status", html: (v, r) => `${badge(v)}${r.error ? ` <span class="bad" title="${esc(r.error)}">!</span>` : ""}` },
      { k: "model", label: "Model" },
      { k: "min_mask_area", label: "Min piece", num: true, title: "data.min_mask_area" },
      { k: "rounds", label: "Rounds", num: true, get: (r) => r.epochs_done, html: (v, r) => `${v}/${r.epochs || r.epochs_cfg}` },
      { k: "map", label: "Best val mAP", num: true, get: (r) => (r.best ? r.best.map5095_m : null), html: (v, r) => (v == null ? "–" : `${K.f3(v)} <span class="faint">@${r.best.epoch}</span>`) },
      { k: "val", label: "Val 3D Dice", num: true, get: (r) => (r.val ? r.val.dice3d_mean : null), fmt: K.f4 },
      { k: "test", label: "Test 3D Dice", num: true, get: (r) => (r.test ? r.test.dice3d_mean : null), fmt: K.f4, bar: { max: 1, color: "var(--yolo)" } },
      { k: "t2d", label: "Test 2D Dice", num: true, get: (r) => (r.test ? r.test.tumour_slice_dice : null), fmt: K.f4, hidden: true },
      { k: "conf", label: "Conf", num: true, get: (r) => (r.test ? r.test.conf : null), hidden: true },
      { k: "time", label: "Train time", num: true, get: (r) => r.train_seconds, fmt: K.secs },
      { k: "created", label: "Started", get: (r) => K.when(r.created) },
    ];
    if (!S.tables.runs) {
      S.tables.runs = K.table($("runsTable"), cols, { rows: S.runs, key: (r) => r.id, exportName: "yolo_runs", short: true, pageSize: 25,
        onRow: (r, e) => { if (!e.target.closest(".cmp")) selectRun(r.id); }, selected: S.sel });
      $("runsTable").addEventListener("change", (e) => {
        const cb = e.target.closest(".cmp");
        if (!cb) return;
        if (cb.checked && S.cmp.size >= 4) { cb.checked = false; return K.toast("Up to 4 runs at a time."); }
        cb.checked ? S.cmp.add(cb.dataset.id) : S.cmp.delete(cb.dataset.id);
        toHash(); renderCurves(); cmpNote();
      });
    } else { S.tables.runs.state.selected = S.sel; S.tables.runs.update(S.runs); }
    cmpNote();
  }
  function cmpNote() { $("cmpNote").textContent = S.cmp.size ? `Ticked: ${[...S.cmp].join(", ")}` : ""; }

  async function loadRun(id) { const r = await runApi(id); S.cache[id] = r; return r; }
  async function selectRun(id, keep) {
    if (S.sel !== id && !keep) { S.vPatient = S.vZ = S.vConf = null; S.conf = null; }
    S.sel = id; S.evalData = {}; S.range = null;
    toHash();
    if (S.tables.runs) { S.tables.runs.state.selected = id; S.tables.runs.render(); }
    try { S.run = await loadRun(id); } catch (e) { K.fail(e); return; }
    renderAnalysis();
  }

  // ------------------------------------------------------------------ run summary
  function bestEpoch(res) {
    const f = res["metrics/mAP50-95(M)"];
    if (!f || !f.length) return null;
    let bi = 0; f.forEach((v, i) => { if (v != null && v > f[bi]) bi = i; });
    return res.epoch[bi];
  }
  function renderAnalysis() {
    const r = S.run;
    if (!r) return;
    $("analysis").classList.remove("hidden");
    $("selId").textContent = r.id;
    $("selSub").innerHTML = `${badge(r.state)} ${esc(r.model)} · image ${r.imgsz} · ${r.batch} slices per step · ${r.epochs_done} round(s) · picks by <b>${esc((r.config.select || {}).by || "map")}</b>${r.smoke ? " · quick test" : ""}${r.error ? ` · <span class="bad">${esc(r.error)}</span>` : ""}`;
    const live = LIVE.includes(r.state);
    $("resumeBtn").disabled = live || !!S.active || !r.has_last || r.state === "done";
    $("reevalBtn").disabled = live || !!S.active || !r.has_best;
    $("dlRounds").disabled = !r.results || !r.results.epoch;
    $("toResults").href = `/results/#pool=test&y=${encodeURIComponent(r.id)}`;
    const t = r.eval && r.eval.test, v = r.eval && r.eval.val;
    const res = r.results || {};
    $("kpis").innerHTML = [
      K.kpi("Test 3D Dice", t ? K.f4(t.dice3d_mean) : "–", t ? `median ${K.f4(t.dice3d_median)} · ${t.patients} patients · conf ${t.conf}` : "after the final check", { hero: true }),
      K.kpi("Val 3D Dice", v ? K.f4(v.dice3d_mean) : "–", v ? `best conf on val: ${v.best_conf}` : ""),
      K.kpi("Test 2D Dice", t ? K.f4(t.tumour_slice_dice) : "–", "slices with tumour"),
      K.kpi("Tumour slices found", t ? K.pct(t.sensitivity) : "–", t ? `empty slices kept empty ${K.pct(t.specificity)}` : ""),
      K.kpi("Best val mAP", r.best ? K.f3(r.best.map5095_m) : "–", r.best ? `mask mAP50-95, round ${r.best.epoch}` : "", { spark: res["metrics/mAP50-95(M)"], sparkColor: "var(--yolo)" }),
      r.dice_check ? K.kpi("Best round (3D Dice check)", K.f4(Math.max(...r.dice_check.dice3d)), `round ${r.dice_check.epoch[r.dice_check.dice3d.indexOf(Math.max(...r.dice_check.dice3d))]}`, { spark: r.dice_check.dice3d, sparkColor: "var(--test)" }) : "",
      K.kpi("Training time", K.secs(r.train_seconds), r.epochs_done ? `${K.secs(r.train_seconds / r.epochs_done)} per round` : ""),
      K.kpi("Extras when scoring", t ? `${t.tta_flip ? "mirror" : "no mirror"} · ${t.min_component || 0}` : "–", "mirror TTA · smallest piece kept (voxels)"),
    ].join("");
    renderCurves(); renderDataset(); renderEval(); renderPlots();
  }

  // ------------------------------------------------------------------ curves
  const GROUPS = {
    loss: { y: "Error (lower is better)", charts: [["box", "Box error — where the tumour is"], ["seg", "Mask error — the outline"], ["cls", "Class error — tumour or not"], ["dfl", "Box-edge error"]]
      .map(([k, t]) => ({ t, s: [{ col: `train/${k}_loss`, name: "training slices", color: "var(--yolo)" }, { col: `val/${k}_loss`, name: "val slices", color: "var(--medsam2)", dash: "6 4" }], cmp: `val/${k}_loss` })) },
    mask: { y: "Score (0–1)", charts: [["precision", "Precision — drawn tumours that are real"], ["recall", "Recall — real tumours that were drawn"], ["mAP50", "mAP50"], ["mAP50-95", "mAP50-95 (strict)"]]
      .map(([k, t]) => ({ t, s: [{ col: `metrics/${k}(M)`, name: `mask ${k}`, color: "var(--yolo)" }], cmp: `metrics/${k}(M)`, y01: true })) },
    box: { y: "Score (0–1)", charts: [["precision", "Precision"], ["recall", "Recall"], ["mAP50", "mAP50"], ["mAP50-95", "mAP50-95 (strict)"]]
      .map(([k, t]) => ({ t, s: [{ col: `metrics/${k}(B)`, name: `box ${k}`, color: "var(--yolo)" }], cmp: `metrics/${k}(B)`, y01: true })) },
    lr: { y: "Learning rate", charts: [{ t: "Learning rate per weight group", s: [0, 1, 2].map((i) => ({ col: `lr/pg${i}`, name: `group ${i}`, color: CMP[i], dash: i ? "5 4" : null })), cmp: "lr/pg0" }] },
  };
  async function renderCurves() {
    K.$$("#groupSeg button").forEach((b) => b.classList.toggle("on", b.dataset.v === S.group));
    const g = GROUPS[S.group];
    const ids = [...S.cmp];
    const comparing = ids.length >= 2;
    const runs = comparing ? await Promise.all(ids.map((id) => S.cache[id] || loadRun(id).catch(() => null))) : [S.run];
    const valid = runs.filter((r) => r && r.results && r.results.epoch && r.results.epoch.length);
    $("curves").innerHTML = g.charts.map((c, i) => `<div class="${g.charts.length === 1 ? "s12" : "s6"}"><h3>${esc(c.t)}</h3><div id="curve${i}"></div></div>`).join("");
    $("curvesSub").textContent = comparing ? `Comparing ${valid.length} runs (${ids.join(", ")}); for errors, each run's val line.` : "One point per round. Hover one chart to read all of them at the same round; drag to zoom.";
    if (!valid.length) { $("curves").innerHTML = `<div class="s12 empty">No round has finished yet.</div>`; $("curveNote").classList.add("hidden"); return; }
    g.charts.forEach((c, i) => {
      const series = comparing
        ? valid.map((r, k) => ({ name: r.id, color: CMP[k], dash: k ? ["6 4", "2 3", "8 3 2 3"][k - 1] : null, points: r.results.epoch.map((e, j) => [e, (r.results[c.cmp] || [])[j]]) }))
        : c.s.map((s) => ({ ...s, points: valid[0].results.epoch.map((e, j) => [e, (valid[0].results[s.col] || [])[j]]) }));
      const be = comparing ? null : bestEpoch(valid[0].results);
      S.charts[`c${i}`] = Charts.line($(`curve${i}`), { series, sync: "yolo-rounds", height: 220, xName: "round", xLabel: "Round (epoch)", yLabel: g.y,
        yMin: c.y01 ? 0 : undefined, markers: be ? [{ x: be, label: `best mAP: ${be}` }] : [], exportName: `${S.sel}_${S.group}_${i}`,
        yFmt: S.group === "lr" ? (v) => v.toExponential(1) : undefined });
    });
    const r0 = valid[0], res = r0.results;
    let note = "";
    if (!comparing && res["val/seg_loss"]) {
      const vl = res["val/seg_loss"];
      let mi = 0; vl.forEach((v, i) => { if (v != null && v < vl[mi]) mi = i; });
      const since = res.epoch.length - 1 - mi;
      note = `<b>Mask error on val slices was lowest at round ${res.epoch[mi]}</b> (${K.fmt(vl[mi])})${since ? `, ${since} round(s) before the last. Rising after that means YOLO starts memorising its training slices.` : " — the latest round."}`;
    }
    if (!comparing && r0.dice_check) note += `${note ? "<br>" : ""}<b>3D Dice check</b> after each round: best ${K.f4(Math.max(...r0.dice_check.dice3d))} — this run keeps that round (<code>best_dice.pt</code>).`;
    $("curveNote").innerHTML = note;
    $("curveNote").classList.toggle("hidden", !note);
  }

  // ------------------------------------------------------------------ dataset
  function renderDataset() {
    const m = S.run.dataset;
    if (!m) { $("dataSub").textContent = "Shown once the run has made (or reused) its pictures."; ["dataBars", "dataTable", "areaBars"].forEach((id) => { $(id).innerHTML = ""; }); return; }
    $("dataSub").innerHTML = `<code>dataset/${esc(m.key)}</code>, made ${esc(K.when(m.created))}. One picture per axial slice with at least ${m.data.min_fg_voxels} brain pixels; tumour pieces under ${m.data.min_mask_area} px are left out of the answers.`;
    const sp = Object.entries(m.splits);
    Charts.bars($("dataBars"), { cats: sp.map(([k]) => k), stacked: true, horizontal: true, yLabel: "Slices", exportName: "slices_per_pool", legend: true,
      series: [{ name: "with tumour", color: "var(--yolo)", values: sp.map(([, s]) => s.positive) }, { name: "no tumour", color: "var(--faint)", values: sp.map(([, s]) => s.negative) }], height: 130 });
    $("dataTable").innerHTML = `<div class="tbl-wrap short"><table class="tbl"><thead><tr><th>Pool</th><th class="num">Patients</th><th class="num">Slices</th><th class="num">With tumour</th><th class="num">Pieces drawn</th><th class="num">Typical piece</th></tr></thead><tbody>
      ${sp.map(([k, s]) => `<tr><td>${k}</td><td class="num">${s.patients}</td><td class="num">${s.slices.toLocaleString()}</td><td class="num">${K.pct(s.positive / s.slices, 0)}</td><td class="num">${s.instances.toLocaleString()}</td><td class="num">${s.area_median != null ? `${K.fmt(s.area_median)} px` : "–"}</td></tr>`).join("")}</tbody></table></div>`;
    const b = m.area_bins;
    const lab = (i) => (i === b.length - 1 ? `≥ ${b[i].toLocaleString()}` : `${b[i].toLocaleString()}–${b[i + 1].toLocaleString()}`);
    Charts.bars($("areaBars"), { cats: m.splits.train.area_hist.map((_, i) => lab(i)), horizontal: true, yLabel: "Slices", exportName: "tumour_area_per_slice", legend: true,
      series: sp.map(([k, s], i) => ({ name: k, color: i ? "var(--medsam2)" : "var(--yolo)", values: s.area_hist })) });
  }

  // ------------------------------------------------------------------ evaluation
  const evalData = (split) => (S.evalData[split] ||= runApi(S.sel, `/eval/${split}`).catch(() => null));
  function patientRows(data, k) {
    return data.patients.map((p) => {
      const [inter, pred, gt] = p.sweep[k];
      const s = p.slices;
      let tp = 0, fn = 0, fp = 0;
      s.gt.forEach((g, i) => { if (g && s.pred[i]) tp++; else if (g) fn++; else if (s.pred[i]) fp++; });
      return { id: p.id, dice: pred + gt ? (2 * inter) / (pred + gt) : 1, gt_ml: gt / 1000, pred_ml: pred / 1000, tumour_slices: tp + fn, tp, fn, fp, maxconf: Math.max(0, ...s.conf) };
    });
  }
  async function renderEval() {
    K.$$("#splitSeg button").forEach((b) => b.classList.toggle("on", b.dataset.v === S.split));
    const r = S.run, sum = r.eval && r.eval[S.split];
    const clear = (msg) => { $("evalKpis").innerHTML = `<p class="empty">${msg}</p>`; ["sweepChart", "diceHist", "diceScatter", "sizeBars", "confusion", "patTable", "sweepNote"].forEach((id) => { $(id).innerHTML = ""; }); renderViewer(); };
    if (!sum) return clear(LIVE.includes(r.state) ? "The final check runs after training." : "This run has no final check on this pool yet.");
    const data = await evalData(S.split);
    if (!data) return clear("The result files were not found.");
    if (S.conf == null || !data.thresholds.includes(S.conf)) S.conf = sum.conf;
    $("confSel").innerHTML = data.thresholds.map((t) => `<option value="${t}" ${t === S.conf ? "selected" : ""}>${t}${t === sum.conf ? " (setting)" : ""}</option>`).join("");
    const k = data.thresholds.indexOf(S.conf);
    const rows = patientRows(data, k);
    S.evalRows = rows;
    const d3 = rows.map((x) => x.dice), c = St.bootCI(d3);
    const col = S.split === "test" ? "var(--test)" : "var(--yolo)";
    $("evalKpis").innerHTML = [
      K.kpi(`3D Dice · ${S.split}`, K.f4(St.mean(d3)), `median ${K.f4(St.median(d3))} · ${c ? `95% range ${K.f3(c[0])}–${K.f3(c[1])}` : ""}`, { hero: true }),
      K.kpi("Patients", rows.length, `${rows.filter((x) => x.dice < 0.5).length} below 0.5`),
      K.kpi("2D Dice", K.f4(sum.tumour_slice_dice), "slices with tumour (at the setting)"),
      K.kpi("Tumour slices found", K.pct(sum.sensitivity), `empty kept empty ${K.pct(sum.specificity)}`),
      K.kpi("Confidence shown", S.conf, `setting ${sum.conf} · best here ${sum.best_conf}`),
      K.kpi("Missed / false slices", `${St.sum(rows.map((x) => x.fn)).toLocaleString()} / ${St.sum(rows.map((x) => x.fp)).toLocaleString()}`, "over all patients"),
    ].join("");
    const series = ["val", "test"].filter((s) => r.eval[s]).map((s) => ({ name: `${s} pool`, color: s === "test" ? "var(--test)" : "var(--yolo)", dash: s === "val" ? "6 4" : null, dots: true, points: r.eval[s].sweep.map((p) => [p.conf, p.dice3d_mean]) }));
    const vb = r.eval.val && r.eval.val.best_conf;
    Charts.line($("sweepChart"), { series, height: 240, xName: "confidence", xLabel: "Confidence (lower = YOLO draws more)", yLabel: "Mean 3D Dice", exportName: `${S.sel}_sweep`,
      markers: [{ x: sum.conf, label: "setting" }, ...(vb != null && vb !== sum.conf ? [{ x: vb, label: "best on val", color: "var(--yolo)" }] : []), ...(S.conf !== sum.conf ? [{ x: S.conf, label: "shown", color: "var(--accent)" }] : [])],
      onClick: (x) => { S.conf = x; toHash(); renderEval(); } });
    $("sweepNote").innerHTML = r.eval.val && r.eval.test ? `Best on val: <b>${vb}</b>. Choose the confidence on val, never on test. Click a point to show that confidence.` : "";
    Charts.hist($("diceHist"), { series: [{ name: "patients", color: col, values: d3 }], domain: [0, 1], bins: 25, xLabel: "3D Dice", yLabel: "Patients", brush: true, range: S.range,
      markers: [{ x: St.mean(d3), label: "mean" }], height: 240, exportName: `${S.sel}_dice_hist`,
      onBrush: (lo, hi) => { S.range = lo == null ? null : [lo, hi]; renderPat(); } });
    Charts.scatter($("diceScatter"), { points: rows.map((x) => ({ x: x.gt_ml, y: x.dice, id: x.id, color: col,
      tip: `${K.tipTitle(x.id)}${K.tipRow(K.color(col), "3D Dice", K.f4(x.dice))}${K.tipRow(K.color("var(--faint)"), "expert", `${K.fmt(x.gt_ml)} mL`)}${K.tipRow(K.color("var(--faint)"), "YOLO drew", `${K.fmt(x.pred_ml)} mL`)}` })),
      logX: true, yMin: 0, yMax: 1, trend: true, xLabel: "Expert tumour (mL, log)", yLabel: "3D Dice", height: 280, highlight: S.vPatient, exportName: `${S.sel}_dice_vs_size`,
      onClick: (p) => openViewer(p.id) });
    const buckets = [[0, 10], [10, 30], [30, 60], [60, 100], [100, Infinity]];
    const inB = buckets.map(([a, b]) => rows.filter((x) => x.gt_ml >= a && x.gt_ml < b).map((x) => x.dice));
    Charts.bars($("sizeBars"), { cats: buckets.map(([a, b], i) => `${b === Infinity ? `≥ ${a}` : `${a}–${b}`} mL (n=${inB[i].length})`), yLabel: "Mean 3D Dice", yMin: 0, yMax: 1, fmt: K.f3, showValues: true,
      series: [{ name: "mean 3D Dice", color: col, values: inB.map((v) => St.mean(v)), err: inB.map((v) => St.bootCI(v, St.mean, 400)) }], height: 200, exportName: `${S.sel}_dice_by_size` });
    const dd = sum.detection;
    $("confusion").innerHTML = `<div class="tbl-wrap short"><table class="tbl"><thead><tr><th></th><th class="num">YOLO drew</th><th class="num">drew nothing</th></tr></thead><tbody>
      <tr><td>has tumour</td><td class="num good">${dd.tp.toLocaleString()} found</td><td class="num bad">${dd.fn.toLocaleString()} missed</td></tr>
      <tr><td>no tumour</td><td class="num bad">${dd.fp.toLocaleString()} false</td><td class="num good">${dd.tn.toLocaleString()} empty</td></tr></tbody></table></div>`;
    renderPat();
    renderViewer();
  }
  function renderPat() {
    const rows = (S.evalRows || []).filter((x) => !S.range || (x.dice >= S.range[0] && x.dice <= S.range[1]));
    const cols = [
      { k: "id", label: "Patient", html: (v) => `<span class="mono">${esc(v)}</span>` },
      { k: "dice", label: "3D Dice", num: true, fmt: K.f4, bar: { max: 1, color: "var(--yolo)" } },
      { k: "gt_ml", label: "Expert (mL)", num: true, fmt: K.fmt },
      { k: "pred_ml", label: "YOLO drew (mL)", num: true, fmt: K.fmt },
      { k: "tumour_slices", label: "Tumour slices", num: true },
      { k: "tp", label: "Found", num: true }, { k: "fn", label: "Missed", num: true }, { k: "fp", label: "False", num: true },
      { k: "maxconf", label: "Top conf", num: true, fmt: K.f2 },
    ];
    if (!S.tables.pat) S.tables.pat = K.table($("patTable"), cols, { rows, key: (r) => r.id, onRow: (r) => openViewer(r.id), exportName: `${S.sel}_${S.split}_patients`, sort: { k: "dice", dir: 1 }, selected: S.vPatient });
    else { S.tables.pat.state.selected = S.vPatient; S.tables.pat.update(rows); }
  }

  // ------------------------------------------------------------------ viewer
  const modelRuns = () => S.runs.filter((r) => r.has_best);
  function renderViewer() {
    const runs = modelRuns();
    const on = runs.length > 0;
    ["vModel", "vSplit", "vPatient", "vZ", "vConf", "vPeak", "vRun"].forEach((id) => { $(id).disabled = !on; });
    if (!on) { $("vNote").textContent = "Shown once a run has a trained model."; return; }
    if (!S.vModel || !runs.some((r) => r.id === S.vModel)) S.vModel = runs.some((r) => r.id === S.sel) ? S.sel : runs[0].id;
    if (!S.vSplit) S.vSplit = S.split;
    if (S.vConf == null) S.vConf = S.conf != null ? S.conf : S.ov.config.evaluate.conf;
    $("vModel").innerHTML = runs.map((r) => `<option value="${esc(r.id)}" ${r.id === S.vModel ? "selected" : ""}>${esc(r.id)} · ${esc(r.model)}${r.test ? ` · test ${K.f3(r.test.dice3d_mean)}` : ""}</option>`).join("");
    $("vSplit").value = S.vSplit;
    $("vConf").value = S.vConf;
    loadPatients();
  }
  async function loadPatients() {
    const ids = await (S.pool[S.vSplit] ||= api(`api/pool_patients?split=${S.vSplit}`).then((d) => d.patients).catch(() => []));
    if (!ids.length) return;
    if (!S.vPatient || !ids.includes(S.vPatient)) {
      const worst = S.evalRows && S.split === S.vSplit ? [...S.evalRows].sort((a, b) => a.dice - b.dice)[0] : null;
      S.vPatient = worst && ids.includes(worst.id) ? worst.id : ids[0];
      S.vZ = null;
    }
    $("vPatient").innerHTML = ids.map((id) => `<option ${id === S.vPatient ? "selected" : ""}>${esc(id)}</option>`).join("");
    showSlice(); drawProfile();
  }
  const pkey = () => `${S.vModel}|${S.vSplit}|${S.vPatient}|${S.vConf}`;
  const showSlice = K.debounce(async () => {
    if (!S.vModel || !S.vPatient) return;
    $("vConfL").textContent = (+S.vConf).toFixed(2);
    toHash();
    const q = `split=${S.vSplit}&patient=${encodeURIComponent(S.vPatient)}${S.vZ == null ? "" : `&z=${S.vZ}`}&conf=${S.vConf}`;
    $("vLoading").classList.remove("hidden");
    const img = $("vImg");
    img.onload = () => $("vLoading").classList.add("hidden");
    img.onerror = () => { $("vLoading").textContent = "This slice could not be drawn."; };
    img.src = `${BASE}api/runs/${encodeURIComponent(S.vModel)}/predict.png?${q}`;
    img.alt = `${S.vPatient} slice ${S.vZ}`;
    try {
      const st = await api(`api/runs/${encodeURIComponent(S.vModel)}/predict.json?${q}`);
      S.vSlices = st.brain_slices; S.vZ = st.z;
      const i = S.vSlices.indexOf(st.z);
      $("vZ").max = S.vSlices.length - 1; $("vZ").value = Math.max(0, i);
      $("vZl").textContent = `z = ${st.z}`;
      $("vStats").innerHTML = [
        K.kpi("2D Dice here", st.gt || st.pred ? K.f3(st.dice) : "–", st.gt || st.pred ? "overlap on this slice" : "no tumour, nothing drawn"),
        K.kpi("Expert", `${st.gt.toLocaleString()} px`, ""),
        K.kpi("YOLO drew", `${st.pred.toLocaleString()} px`, `${st.inter.toLocaleString()} px right`),
        K.kpi("Shapes", st.confs.length, st.confs.length ? `sure: ${st.confs.join(", ")}` : `none at ≥ ${S.vConf}`),
      ].join("");
      drawProfile(); toHash();
    } catch (e) { $("vStats").innerHTML = `<p class="bad">${esc(e.message)}</p>`; $("vLoading").classList.add("hidden"); }
  }, 120);
  async function runPatient() {
    if (!S.vModel || !S.vPatient) return;
    const key = pkey();
    $("vRun").disabled = true;
    $("vNote").textContent = `Running ${S.vPatient} through ${S.vModel}…`;
    try {
      const d = await api(`api/runs/${encodeURIComponent(S.vModel)}/profile.json?split=${S.vSplit}&patient=${encodeURIComponent(S.vPatient)}&conf=${S.vConf}`);
      S.profile = { key, data: d };
      const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
      const tum = d.dice.filter((_, i) => d.gt[i] > 0);
      $("vNote").innerHTML = `<b>${esc(S.vPatient)}</b>: 3D Dice <b>${K.f4(d.dice3d)}</b> · mean slice Dice ${K.f4(mean(d.dice))} (all ${d.z.length} slices), ${K.f4(mean(tum))} (${tum.length} tumour slices) · expert ${K.fmt(d.gt_total / 1000)} mL · YOLO ${K.fmt(d.pred_total / 1000)} mL. A slice where both are empty scores 1.`;
      drawProfile();
    } catch (e) { $("vNote").innerHTML = `<span class="bad">${esc(e.message)}</span>`; } finally { $("vRun").disabled = false; }
  }
  function drawProfile() {
    const p = S.profile && S.profile.key === pkey() ? S.profile.data : null;
    if (!p) { $("vProfile").innerHTML = `<p class="empty">Press <b>Whole patient</b> to run every slice.</p>`; return; }
    S.charts.prof = Charts.line($("vProfile"), { height: 260, xLabel: "Slice (z)", yLabel: "Dice on the slice", y2Label: "Tumour pixels", y2Min: 0, yMin: 0, yMax: 1, exportName: `${S.vPatient}_profile`,
      series: [{ name: "Dice", color: "var(--yolo)", points: p.z.map((z, i) => [z, p.dice[i]]) },
               { name: "expert pixels", color: "var(--gt)", axis: "right", area: true, points: p.z.map((z, i) => [z, p.gt[i]]) },
               { name: "YOLO pixels", color: "var(--medsam2)", axis: "right", dash: "5 4", points: p.z.map((z, i) => [z, p.pred[i]]) }],
      markers: [{ x: S.vZ, label: `z ${S.vZ}`, color: "var(--accent)", solid: true }], onClick: (z) => { S.vZ = z; showSlice(); } });
  }
  function openViewer(id) { S.vPatient = id; S.vSplit = S.split; S.vZ = null; S.vConf = S.conf; renderViewer(); $("secViewer").scrollIntoView({ behavior: "smooth" }); }

  // ------------------------------------------------------------------ plots
  function renderPlots() {
    const r = S.run;
    const order = (f) => (f === "results.png" ? 0 : /^Mask/.test(f) ? 1 : /confusion/.test(f) ? 2 : /^Box/.test(f) ? 3 : /^val_/.test(f) ? 4 : 5);
    const files = [...(r.plots || [])].sort((a, b) => order(a) - order(b) || a.localeCompare(b));
    $("plots").innerHTML = files.length ? files.map((f) => `<figure data-f="${esc(f)}"><img loading="lazy" src="${BASE}api/runs/${encodeURIComponent(r.id)}/file/${encodeURIComponent(f)}" alt="${esc(f)}"><figcaption>${esc(f)}</figcaption></figure>`).join("")
      : `<p class="empty">Saved when training finishes.</p>`;
  }

  // ------------------------------------------------------------------ live
  function renderLive(run) {
    const st = run ? run.status : null;
    const live = st && LIVE.includes(run.state);
    $("stopBtn").disabled = !live;
    $("liveId").textContent = run ? run.id : "";
    if (!run) { $("liveSub").textContent = "Nothing is running."; K.$$("#stepper li").forEach((li) => { li.className = ""; K.$("i", li).style.width = "0"; K.$("em", li).textContent = ""; }); $("liveKpis").innerHTML = ""; return; }
    const stages = ["dataset", "train", "evaluate", "done"];
    const cur = run.state === "done" ? 3 : Math.max(0, stages.indexOf(st.stage));
    const prog = { dataset: st.dataset_total ? st.dataset_done / st.dataset_total : 0,
      train: st.epochs ? ((st.epoch || 1) - 1 + (st.batches ? (st.batch || 0) / st.batches : 0)) / st.epochs : 0,
      evaluate: st.eval_total ? st.eval_done / st.eval_total : 0, done: run.state === "done" ? 1 : 0 };
    const txt = { dataset: st.dataset_total ? `${st.dataset_done} of ${st.dataset_total} patients` : "",
      train: st.epochs ? `round ${st.epoch} of ${st.epochs}${st.batches && cur === 1 ? ` · step ${st.batch}/${st.batches}` : ""}` : "",
      evaluate: st.eval_total ? `${st.eval_done} of ${st.eval_total}${st.eval_split ? ` (${st.eval_split})` : ""}` : "", done: run.state === "done" ? K.when(st.finished) : "" };
    K.$$("#stepper li").forEach((li, i) => {
      const k = li.dataset.stage, bad = !live && run.state !== "done" && i === cur;
      li.className = i < cur || run.state === "done" ? "done" : i === cur ? (bad ? "bad" : "cur") : "";
      K.$("i", li).style.width = `${(i < cur ? 1 : i === cur ? prog[k] : 0) * 100}%`;
      K.$("em", li).textContent = txt[k];
    });
    const res = run.results || {}, times = res.time || [];
    const per = times.length ? times[times.length - 1] / times.length : null;
    const left = st.epochs && per && cur === 1 ? (st.epochs - (st.epoch || 1) + 1 - (st.batches ? (st.batch || 0) / st.batches : 0)) * per : null;
    const be = bestEpoch(res);
    const since = be != null && res.epoch ? res.epoch[res.epoch.length - 1] - be : null;
    const dc = run.dice_check;
    $("liveSub").innerHTML = `${badge(run.state)} ${esc(run.model)}${run.smoke ? " · quick test" : ""} · started ${esc(K.when(st.started))} (${K.ago(st.started)})${run.error ? ` · <span class="bad">${esc(run.error)}</span>` : ""}`;
    $("liveKpis").innerHTML = [
      K.kpi("Time per round", per ? K.secs(per) : "–", `${times.length} round(s) done`),
      K.kpi("Time left", left != null ? `≈ ${K.secs(left)}` : "–", left != null ? "a guess; it can stop earlier" : ""),
      K.kpi("Best val mAP", be != null ? K.f3(Math.max(...res["metrics/mAP50-95(M)"].filter((v) => v != null))) : "–", be != null ? `round ${be}` : "", { spark: res["metrics/mAP50-95(M)"], sparkColor: "var(--yolo)" }),
      K.kpi("Rounds since best mAP", since != null ? since : "–", `stops at ${run.config.train.patience}`),
      dc ? K.kpi("3D Dice check", K.f4(dc.dice3d[dc.dice3d.length - 1]), `best ${K.f4(Math.max(...dc.dice3d))}`, { spark: dc.dice3d, sparkColor: "var(--test)" }) : "",
      st.dataset_total && cur === 0 ? K.kpi("Pictures", `${st.dataset_done}/${st.dataset_total}`, "patients made into pictures") : "",
    ].join("");
    const pts = (col) => (res.epoch || []).map((e, j) => [e, (res[col] || [])[j]]);
    Charts.line($("liveChart"), { height: 230, xLabel: "Round", yLabel: "Mask mAP50-95", y2Label: "val mask error", zoom: false, exportName: `${run.id}_live`,
      series: [{ name: "mask mAP50-95", color: "var(--yolo)", points: pts("metrics/mAP50-95(M)"), dots: true },
               ...(dc ? [{ name: "3D Dice check", color: "var(--test)", points: dc.epoch.map((e, i) => [e, dc.dice3d[i]]), dots: true }] : []),
               { name: "val mask error", color: "var(--medsam2)", axis: "right", dash: "5 4", points: pts("val/seg_loss") }],
      empty: "Waiting for the first round." });
  }

  // log
  const BAR = /━|it\/s|s\/it|\d+%\s*[━─╸]/;
  async function pollLog(id) {
    if (S.logRun !== id) { S.logRun = id; S.logOffset = 0; S.logLines = []; }
    const r = await runApi(id, `/log?offset=${S.logOffset}`);
    if (r.offset < S.logOffset) S.logLines = [];
    S.logOffset = r.offset;
    if (r.text) {
      for (const raw of r.text.replace(/\x1b\[[0-9;]*[A-Za-z]/g, "").split("\n")) {
        const parts = raw.split("\r").filter((x) => x.trim());
        if (parts.length) S.logLines.push(parts[parts.length - 1]);
      }
      if (S.logLines.length > 4000) S.logLines = S.logLines.slice(-4000);
    }
    renderLog();
  }
  function renderLog() {
    const f = $("logFilter").value.trim().toLowerCase(), bars = $("logBars").checked;
    const lines = S.logLines.filter((l) => (bars || !BAR.test(l)) && (!f || l.toLowerCase().includes(f))).slice(-1500);
    $("log").innerHTML = lines.map((l) => {
      const cls = /Traceback|Error|error:|failed/i.test(l) ? "err" : /\[eval\]|\[check\]|done|best/i.test(l) ? "ok" : /^=====/.test(l) ? "hi" : "";
      return cls ? `<span class="${cls}">${esc(l)}</span>` : esc(l);
    }).join("\n") || (S.logRun ? "(nothing yet)" : "");
    if ($("logFollow").checked) $("log").scrollTop = $("log").scrollHeight;
  }

  // ------------------------------------------------------------------ start dialog
  function openStart() {
    const ov = S.ov;
    if (!ov) return;
    const c = ov.config;
    const f = (id, label, v, extra = "", hint = "") => `<label class="field"><span>${label}</span><input id="${id}" type="number" value="${v}" ${extra}>${hint ? `<span class="hint">${hint}</span>` : ""}</label>`;
    const html = `<form id="startForm" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px">
      <label class="field"><span>Model</span><select id="fModel">${ov.models.map((m) => `<option ${m === c.model ? "selected" : ""}>${esc(m)}</option>`).join("")}</select></label>
      ${f("fEpochs", "Rounds (most)", c.train.epochs, `min="1" max="1000"`)}
      ${f("fPatience", "Patience", c.train.patience, `min="0" max="1000"`, "stop after this many rounds with no gain")}
      ${f("fImgsz", "Picture size", c.train.imgsz, `step="32" min="64" max="2048"`, "multiple of 32")}
      ${f("fBatch", "Slices per step", c.train.batch, `min="1" max="256"`)}
      ${f("fMinMask", "Smallest tumour piece (px)", c.data.min_mask_area, `min="0" max="5000"`, "changes the pictures → rebuild")}
      ${f("fMinFg", "Least brain on a slice (px)", c.data.min_fg_voxels, `min="0" max="50000"`, "changes the pictures → rebuild")}
      ${f("fMosaic", "Mosaic", c.train.mosaic, `step="0.1" min="0" max="1"`, "0 = off (4 brains glued together)")}
      ${f("fScale", "Zoom ±", c.train.scale, `step="0.05" min="0" max="0.9"`)}
      ${f("fDegrees", "Rotation ± (°)", c.train.degrees, `step="1" min="0" max="45"`)}
      ${f("fFlip", "Left-right flip", c.train.fliplr, `step="0.1" min="0" max="1"`)}
      <label class="field"><span>Keep which round</span><select id="fSelect"><option value="map" ${(c.select || {}).by !== "dice3d" ? "selected" : ""}>best mask mAP (Ultralytics)</option><option value="dice3d" ${(c.select || {}).by === "dice3d" ? "selected" : ""}>best 3D Dice check</option></select></label>
      <label class="inline"><input type="checkbox" id="fCos" ${c.train.cos_lr ? "checked" : ""}> cosine learning rate</label>
      <label class="inline"><input type="checkbox" id="fSmoke"> quick test (2 patients, 2 rounds)</label>
      <p class="callout" id="estimate" style="grid-column:1/-1;margin:0"></p>
      <p class="bad" id="formErr" style="grid-column:1/-1;margin:0"></p>
      <div class="row" style="grid-column:1/-1"><button class="primary big" type="submit">▶ Start</button><span class="note">Runs as its own process — closing this page does not stop it.</span></div>
    </form>
    <details style="margin-top:12px"><summary class="dim" style="cursor:pointer">All settings — YOLO_finetune/config.yaml</summary><pre class="yaml" style="margin-top:8px">${esc(ov.config_text)}</pre></details>`;
    const m = K.modal("Start a YOLO fine-tune", html, { wide: true });
    const form = K.$("#startForm", m.el);
    const est = () => {
      const p = ov.pools, smoke = K.$("#fSmoke", m.el).checked, n = (k) => (smoke ? Math.min(2, p[k]) : p[k]);
      const rebuild = +K.$("#fMinMask", m.el).value !== c.data.min_mask_area || +K.$("#fMinFg", m.el).value !== c.data.min_fg_voxels;
      K.$("#estimate", m.el).innerHTML = `Trains on <b>${n("train")}</b> patients of <code>${esc(c.data.train_pool)}</code>, picks a round on <b>${n("val")}</b> of <code>${esc(c.data.val_pool)}</code>, scores val and <b>${n("test")}</b> of <code>${esc(c.data.test_pool)}</code>. ${smoke ? "Quick test: 2 rounds." : `Up to <b>${esc(K.$("#fEpochs", m.el).value)}</b> rounds.`}${rebuild ? " <b>The pictures will be built again</b> (a few minutes)." : ""}`;
    };
    form.addEventListener("input", est);
    est();
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const g = (id) => K.$(`#${id}`, m.el);
      const body = { model: g("fModel").value, smoke: g("fSmoke").checked, select_by: g("fSelect").value, cos_lr: g("fCos").checked,
        overrides: { "train.epochs": +g("fEpochs").value, "train.patience": +g("fPatience").value, "train.imgsz": +g("fImgsz").value, "train.batch": +g("fBatch").value,
          "data.min_mask_area": +g("fMinMask").value, "data.min_fg_voxels": +g("fMinFg").value, "train.mosaic": +g("fMosaic").value,
          "train.scale": +g("fScale").value, "train.degrees": +g("fDegrees").value, "train.fliplr": +g("fFlip").value } };
      if (body.overrides["train.imgsz"] % 32) { g("formErr").textContent = "Picture size must be a multiple of 32."; return; }
      if (!confirm(`Start ${body.smoke ? "a quick test" : `${body.model}, up to ${body.overrides["train.epochs"]} rounds`}? It uses the GPU until it finishes or you press Stop.`)) return;
      try {
        const r = await api("api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
        m.close();
        K.toast(`Started run <b>${esc(r.id)}</b>.`, "good");
        S.active = r.id;
        await loadOverview();
        selectRun(r.id);
        $("secLive").scrollIntoView({ behavior: "smooth" });
      } catch (err) { g("formErr").textContent = err.message; }
    });
  }

  // ------------------------------------------------------------------ settings diff between ticked runs
  async function diffSettings() {
    const ids = [...S.cmp].slice(0, 4);
    if (ids.length < 2) return K.toast("Tick at least two runs first.");
    const runs = await Promise.all(ids.map((id) => S.cache[id] || loadRun(id)));
    const flat = (o, p = "") => Object.entries(o || {}).flatMap(([k, v]) => (v && typeof v === "object" && !Array.isArray(v) ? flat(v, `${p}${k}.`) : [[`${p}${k}`, JSON.stringify(v)]]));
    const maps = runs.map((r) => Object.fromEntries(flat(r.config).filter(([k]) => !k.startsWith("run."))));
    const keys = [...new Set(maps.flatMap((m) => Object.keys(m)))].filter((k) => new Set(maps.map((m) => m[k])).size > 1).sort();
    K.modal("Settings that differ", keys.length ? `<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Setting</th>${ids.map((id) => `<th class="mono">${esc(id)}</th>`).join("")}</tr></thead><tbody>
      ${keys.map((k) => `<tr><td class="mono">${esc(k)}</td>${maps.map((m) => `<td class="mono">${esc(m[k] ?? "–")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
      <p class="note">Test 3D Dice: ${runs.map((r) => `${esc(r.id)} ${r.test ? K.f4(r.test.dice3d_mean) : "–"}`).join(" · ")}</p>` : `<p>These runs used the same settings.</p>`, { wide: true });
  }

  // ------------------------------------------------------------------ polling
  let pollT = null;
  async function poll() {
    clearTimeout(pollT);
    try {
      const wasActive = S.active;
      await loadOverview();
      if (wasActive && !S.active) K.toast(`Run <b>${esc(wasActive)}</b> has ended.`, "good", 9000);
      const liveId = S.active || (S.sel && S.runs.some((r) => r.id === S.sel) ? S.sel : null);
      if (liveId) {
        const run = await loadRun(liveId);
        renderLive(run);
        await pollLog(liveId);
        if (liveId === S.sel) {
          const prev = S.run;
          S.run = run;
          if (!prev || prev.epochs_done !== run.epochs_done || prev.state !== run.state || !!prev.eval !== !!run.eval) {
            if (prev && !!prev.eval !== !!run.eval) S.evalData = {};
            renderAnalysis();
          }
        }
      } else renderLive(null);
      document.title = S.active ? `● YOLO_finetune · BraTS viewer` : `YOLO_finetune · BraTS viewer`;
    } catch (e) { console.error(e); }
    pollT = setTimeout(poll, document.hidden ? 15000 : S.active ? 2500 : 8000);
  }

  // ------------------------------------------------------------------ wiring
  function wire() {
    $("newRunBtn").addEventListener("click", openStart);
    $("diffBtn").addEventListener("click", diffSettings);
    $("groupSeg").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { S.group = b.dataset.v; toHash(); renderCurves(); } });
    $("splitSeg").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b && b.dataset.v !== S.split) { S.split = b.dataset.v; S.conf = null; S.range = null; S.vPatient = null; toHash(); renderEval(); } });
    $("confSel").addEventListener("change", (e) => { S.conf = +e.target.value; toHash(); renderEval(); });
    $("cfgBtn").addEventListener("click", () => S.run && K.modal(`Settings of ${S.run.id}`, `<pre class="yaml">${esc(S.run.config_text)}</pre>`, { wide: true }));
    $("dlRounds").addEventListener("click", () => { location.href = `${BASE}api/runs/${encodeURIComponent(S.sel)}/file/results.csv?download=1`; });
    const act = async (kind, q) => {
      if (!confirm(q)) return;
      try { await api(`api/runs/${encodeURIComponent(S.sel)}/${kind}`, { method: "POST" }); K.toast("Started.", "good"); poll(); } catch (e) { K.fail(e); }
    };
    $("resumeBtn").addEventListener("click", () => act("resume", "Continue training from where it stopped (last.pt)?"));
    $("reevalBtn").addEventListener("click", () => act("evaluate", "Score the chosen checkpoint again on val and test?"));
    $("stopBtn").addEventListener("click", async () => {
      const id = S.active;
      if (!id || !confirm(`Stop run ${id}? You can continue it later with Resume.`)) return;
      try { await api(`api/runs/${encodeURIComponent(id)}/stop`, { method: "POST" }); K.toast(`Stopped ${esc(id)}.`); } catch (e) { K.fail(e); }
      poll();
    });
    $("logFilter").addEventListener("input", renderLog);
    $("logBars").addEventListener("change", renderLog);
    $("logDl").addEventListener("click", () => K.download(`${S.logRun || "run"}_log.txt`, S.logLines.join("\n")));
    $("plots").addEventListener("click", (e) => { const f = e.target.closest("[data-f]"); if (f) K.lightbox(`${BASE}api/runs/${encodeURIComponent(S.sel)}/file/${encodeURIComponent(f.dataset.f)}`, f.dataset.f); });
    $("vModel").addEventListener("change", (e) => { S.vModel = e.target.value; S.vZ = null; showSlice(); drawProfile(); });
    $("vSplit").addEventListener("change", (e) => { S.vSplit = e.target.value; S.vPatient = null; S.vZ = null; loadPatients(); });
    $("vPatient").addEventListener("change", (e) => { S.vPatient = e.target.value; S.vZ = null; showSlice(); drawProfile(); });
    $("vZ").addEventListener("input", (e) => { if (S.vSlices) { S.vZ = S.vSlices[+e.target.value]; showSlice(); } });
    $("vConf").addEventListener("input", (e) => { S.vConf = +(+e.target.value).toFixed(2); showSlice(); drawProfile(); });
    $("vRun").addEventListener("click", runPatient);
    $("vPeak").addEventListener("click", async () => {
      if (!(S.profile && S.profile.key === pkey())) await runPatient();
      const d = S.profile && S.profile.key === pkey() ? S.profile.data : null;
      if (d) { S.vZ = d.z[d.gt.indexOf(Math.max(...d.gt))]; showSlice(); }
    });
    K.key("n", "Start a new run", () => !$("newRunBtn").disabled && openStart());
    K.key("arrowleft", "Previous slice", () => { if (!S.vSlices) return; const i = S.vSlices.indexOf(S.vZ) - 1; if (i >= 0) { S.vZ = S.vSlices[i]; showSlice(); } });
    K.key("arrowright", "Next slice", () => { if (!S.vSlices) return; const i = S.vSlices.indexOf(S.vZ) + 1; if (i < S.vSlices.length) { S.vZ = S.vSlices[i]; showSlice(); } });
    K.key("[", "Previous patient", () => stepPatient(-1));
    K.key("]", "Next patient", () => stepPatient(1));
    K.command("Start a new YOLO run", openStart, { group: "YOLO", hint: "N" });
    K.command("Compare settings of the ticked runs", diffSettings, { group: "YOLO" });
    K.provider((q) => S.runs.filter((r) => r.id.includes(q)).slice(0, 8).map((r) => ({ label: `Open run ${r.id}`, hint: r.test ? `test ${K.f4(r.test.dice3d_mean)}` : WORD[r.state], group: "Runs", run: () => selectRun(r.id) })));
    window.addEventListener("hashchange", () => { const was = S.sel; fromHash(); if (S.sel && S.sel !== was) selectRun(S.sel, true); });
  }
  function stepPatient(d) {
    const opts = [...$("vPatient").options].map((o) => o.value);
    const i = opts.indexOf(S.vPatient) + d;
    if (i >= 0 && i < opts.length) { S.vPatient = opts[i]; S.vZ = null; $("vPatient").value = opts[i]; showSlice(); drawProfile(); }
  }

  document.addEventListener("DOMContentLoaded", () => {
    wire();
    fromHash();
    if (S.sel) selectRun(S.sel, true).catch(() => { S.sel = null; });
    poll();
  });
})();
