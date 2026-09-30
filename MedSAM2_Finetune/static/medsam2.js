/* MedSAM2_Finetune page — start runs, follow the live one, and analyse what MedSAM2 added
 * on top of its YOLO hint, patient by patient and slice by slice.
 *
 * Everything comes from the run folders (status.json, results.csv, eval/*.json). The start
 * dialog is built from the server's own list of editable settings (routes.EDITABLE), so it
 * can never offer a setting the server would refuse. The view lives in the URL.
 */
(function () {
  "use strict";
  const K = Kit, $ = K.id, esc = K.esc, St = K.stats;
  const BASE = document.body.dataset.base;  // "/medsam2/"
  const LIVE = ["queued", "running"];
  const WORD = { queued: "starting", running: "running", done: "finished", failed: "failed", stopped: "stopped", interrupted: "cut off" };
  const CMP = ["var(--medsam2)", "var(--yolo)", "var(--test)", "var(--et)"];
  const badge = (st) => `<span class="badge ${st || ""}">${esc(WORD[st] || st || "–")}</span>`;
  // plain words for the settings the start dialog offers
  const LABELS = {
    "prompt.yolo_run": "YOLO that makes the hints", "prompt.variant": "Hint style", "prompt.yolo_conf": "Lowest YOLO confidence kept",
    "prompt.score_min": "Blob counts as YOLO's answer at", "prompt.logit_scale": "How loud the hint is", "prompt.empty_logit": "'Nothing here' value",
    "augment.p": "Hints damaged on purpose", "model.checkpoint": "Starting checkpoint", "data.max_train_patients": "Training patients (empty = all)",
    "data.val_patients": "Patients in the round check", "data.clips_per_patient": "Clips per patient", "data.tumour_clip_fraction": "Share of clips on tumour",
    "data.min_fg_voxels": "Least brain on a slice", "video.num_frames": "Slices per clip", "video.reverse_fraction": "Share read backwards",
    "video.batch": "Clips per step", "train.epochs": "Rounds", "train.patience": "Stop after N quiet rounds", "train.accum": "Steps added up",
    "train.lr": "Learning rate", "train.memory_lr": "Memory learning rate", "train.vision_lr": "Image-encoder learning rate", "train.unfreeze": "Which weights may change",
    "train.dice_weight": "Overlap weight", "train.bce_weight": "Pixel weight", "train.iou_weight": "Quality-head weight", "train.obj_weight": "Object-head weight",
    "train.fliplr": "Left-right flip", "train.workers": "Reader processes", "evaluate.mask_threshold": "Tumour cut-off (logit)", "evaluate.min_component": "Smallest piece kept (voxels)",
  };
  const GROUP = { prompt: "The YOLO hint", augment: "The YOLO hint", model: "Model", data: "Who it learns from", video: "Clips", train: "How it learns", evaluate: "When running" };

  const S = {
    ov: null, runs: [], active: null, sel: null, run: null, cache: {}, cmp: new Set(), split: "test", evalData: {}, range: null,
    pSplit: "test", pPatient: null, which: "best.pt", prof: null, z: null, patients: {}, logRun: null, logOffset: 0, tables: {}, charts: {},
  };
  const api = (p, o) => K.api(BASE + p, o);
  const runApi = (id, p) => api(`api/runs/${encodeURIComponent(id)}${p || ""}`);

  // ------------------------------------------------------------------ URL
  function toHash() {
    K.hashSet({ run: S.sel, split: S.split !== "test" ? S.split : "", cmp: [...S.cmp].join(","), pool: S.pSplit, patient: S.pPatient, which: S.which !== "best.pt" ? S.which : "", z: S.z });
  }
  function fromHash() {
    const q = K.hashGet();
    S.sel = q.run || null;
    S.split = q.split === "val" ? "val" : "test";
    S.cmp = new Set((q.cmp || "").split(",").filter(Boolean));
    S.pSplit = ["test", "val", "train"].includes(q.pool) ? q.pool : "test";
    S.pPatient = q.patient || null;
    S.which = q.which === "last.pt" ? "last.pt" : "best.pt";
    S.z = q.z ? +q.z : null;
  }

  // ------------------------------------------------------------------ overview + runs
  async function loadOverview() {
    const ov = await api("api/overview");
    S.ov = ov; S.runs = ov.runs; S.active = ov.active;
    const c = ov.config;
    $("pageSub").innerHTML = `${esc(c.model.checkpoint)} · hints from YOLO <code>${esc(c.prompt.yolo_run || "best")}</code> · train <code>${esc(c.data.train_pool)}</code> (${ov.pools.train}) · check <code>${esc(c.data.val_pool)}</code> (${ov.pools.val}) · test (${ov.pools.test}) · ${ov.runs.length} run(s)`;
    for (const n of K.$$("[data-cfg]")) n.textContent = n.dataset.cfg.split(".").reduce((o, k) => (o == null ? o : o[k]), c);
    $("newRunBtn").disabled = !!S.active;
    renderRuns();
    if (!S.sel && S.runs.length) selectRun(S.active || S.runs[0].id, true);
  }
  function renderRuns() {
    const cols = [
      { k: "cmp", label: "⊞", nosort: true, html: (v, r) => `<input type="checkbox" class="cmp" data-id="${esc(r.id)}" ${S.cmp.has(r.id) ? "checked" : ""} aria-label="Compare ${esc(r.id)}">` },
      { k: "id", label: "Run", html: (v, r) => `<span class="mono">${esc(v)}</span>${r.smoke ? ' <span class="badge">quick</span>' : ""}${v === S.sel ? " ◀" : ""}` },
      { k: "state", label: "Status", html: (v, r) => `${badge(v)}${r.error ? ` <span class="bad" title="${esc(r.error)}">!</span>` : ""}` },
      { k: "unfreeze", label: "Weights changed" },
      { k: "augment", label: "Hints damaged", num: true, fmt: (v) => (v == null ? "–" : K.f2(v)) },
      { k: "yolo_run", label: "Hint YOLO", html: (v) => `<span class="mono">${esc(v || "–")}</span>`, hidden: true },
      { k: "rounds", label: "Rounds", num: true, get: (r) => r.epochs_done, html: (v, r) => `${v}/${r.epochs_cfg}` },
      { k: "best", label: "Best check", num: true, get: (r) => (r.best ? r.best.val_dice3d : null), html: (v, r) => (v == null ? "–" : `${K.f4(v)} <span class="faint">@${r.best.epoch}</span>`) },
      { k: "val", label: "Val 3D Dice", num: true, get: (r) => (r.val ? r.val.dice3d_mean : null), fmt: K.f4 },
      { k: "test", label: "Test 3D Dice", num: true, get: (r) => (r.test ? r.test.dice3d_mean : null), fmt: K.f4, bar: { max: 1, color: "var(--medsam2)" } },
      { k: "gain", label: "Test gain", num: true, get: (r) => (r.test ? r.test.delta_mean : null), html: (v) => (v == null ? "–" : K.deltaHtml(v)) },
      { k: "hh", label: "Helped / hurt", get: (r) => (r.test ? `${r.test.helped} / ${r.test.hurt}` : "–") },
      { k: "time", label: "Train time", num: true, get: (r) => r.train_seconds, fmt: K.secs },
      { k: "created", label: "Started", get: (r) => K.when(r.created) },
    ];
    if (!S.tables.runs) {
      S.tables.runs = K.table($("runsTable"), cols, { rows: S.runs, key: (r) => r.id, exportName: "medsam2_runs", short: true, pageSize: 25, selected: S.sel,
        onRow: (r, e) => { if (!e.target.closest(".cmp")) selectRun(r.id); } });
      $("runsTable").addEventListener("change", (e) => {
        const cb = e.target.closest(".cmp");
        if (!cb) return;
        if (cb.checked && S.cmp.size >= 4) { cb.checked = false; return K.toast("Up to 4 runs at a time."); }
        cb.checked ? S.cmp.add(cb.dataset.id) : S.cmp.delete(cb.dataset.id);
        toHash(); renderCurves(); $("cmpNote").textContent = S.cmp.size ? `Ticked: ${[...S.cmp].join(", ")}` : "";
      });
    } else { S.tables.runs.state.selected = S.sel; S.tables.runs.update(S.runs); }
  }
  async function loadRun(id) { const r = await runApi(id); S.cache[id] = r; return r; }
  async function selectRun(id, keep) {
    if (S.sel !== id && !keep) { S.prof = null; S.z = null; }
    S.sel = id; S.evalData = {}; S.range = null;
    toHash();
    if (S.tables.runs) { S.tables.runs.state.selected = id; S.tables.runs.render(); }
    try { S.run = await loadRun(id); } catch (e) { K.fail(e); return; }
    renderAnalysis();
  }

  // ------------------------------------------------------------------ analysis
  function renderAnalysis() {
    const r = S.run;
    if (!r) return;
    $("analysis").classList.remove("hidden");
    $("selId").textContent = r.id;
    const c = r.config;
    $("selSub").innerHTML = `${badge(r.state)} ${esc(c.model.checkpoint)} · ${esc(c.train.unfreeze)} · hints from YOLO <code>${esc(c.prompt.yolo_run)}</code> at conf ${c.prompt.yolo_conf} · ${r.epochs_done} round(s)${r.smoke ? " · quick test" : ""}${r.error ? ` · <span class="bad">${esc(r.error)}</span>` : ""}`;
    $("toResults").href = `/results/#pool=test&m=${encodeURIComponent(r.id)}`;
    const t = r.eval && r.eval.test, v = r.eval && r.eval.val, res = r.results || {};
    $("kpis").innerHTML = [
      K.kpi("Test 3D Dice", t ? K.f4(t.dice3d_mean) : "–", t ? `YOLO hint ${K.f4(t.yolo_dice3d_mean)} · ${t.patients} patients` : "after the final check", { hero: true }),
      K.kpi("Test gain over the hint", t ? K.deltaHtml(t.delta_mean) : "–", t ? `median ${K.signed(t.delta_median)} · helped ${t.helped} / hurt ${t.hurt}` : ""),
      K.kpi("Val 3D Dice", v ? K.f4(v.dice3d_mean) : "–", v ? `hint ${K.f4(v.yolo_dice3d_mean)} · best cut-off ${v.best_threshold}` : ""),
      K.kpi("Best round check", r.best ? K.f4(r.best.val_dice3d) : "–", r.best ? `round ${r.best.epoch} · ${c.data.val_patients} check patients` : "", { spark: res.val_dice3d, sparkColor: "var(--medsam2)" }),
      K.kpi("Tumour slices found", t ? K.pct(t.sensitivity) : "–", t ? `empty kept empty ${K.pct(t.specificity)}` : ""),
      K.kpi("Training time", K.secs(r.train_seconds), r.epochs_done ? `${K.secs(r.train_seconds / r.epochs_done)} per round` : ""),
    ].join("");
    renderCurves(); renderEval();
    loadPatients();
  }
  async function renderCurves() {
    const ids = [...S.cmp];
    const comparing = ids.length >= 2;
    const runs = comparing ? await Promise.all(ids.map((id) => S.cache[id] || loadRun(id).catch(() => null))) : [S.run];
    const valid = runs.filter((r) => r && r.results && r.results.epoch && r.results.epoch.length);
    const charts = [["val_dice3d", "Check score (3D Dice on the check pool)", "3D Dice"], ["loss", "Total loss", "Loss"], ["dice", "Overlap loss (Dice part)", "Loss"],
                    ["bce", "Pixel loss (BCE part)", "Loss"], ["obj", "Is there tumour? (object head)", "Loss"], ["lr", "Learning rate", "Step size"]];
    $("curves").innerHTML = charts.map((c, i) => `<div class="s6"><h3>${esc(c[1])}</h3><div id="mc${i}"></div></div>`).join("");
    $("curvesSub").textContent = comparing ? `Comparing ${valid.length} runs: ${ids.join(", ")}.` : "One point per round; hover one chart to read all of them; drag to zoom.";
    if (!valid.length) { $("curves").innerHTML = `<div class="s12 empty">No round has finished yet.</div>`; return; }
    charts.forEach(([col, , y], i) => {
      const series = valid.map((r, k) => ({ name: comparing ? r.id : col, color: comparing ? CMP[k] : i ? "var(--medsam2)" : "var(--test)", dash: k ? ["6 4", "2 3", "8 3 2 3"][k - 1] : null, dots: true,
        points: r.results.epoch.map((e, j) => [e, (r.results[col] || [])[j]]) }));
      const best = !comparing && S.run.best ? [{ x: S.run.best.epoch, label: `best: ${S.run.best.epoch}` }] : [];
      Charts.line($(`mc${i}`), { series, sync: "med-rounds", height: 210, xName: "round", xLabel: "Round", yLabel: y, markers: best, legend: comparing, exportName: `${S.sel}_${col}`,
        yFmt: col === "lr" ? (v) => v.toExponential(1) : undefined });
    });
  }
  const evalData = (split) => (S.evalData[split] ||= runApi(S.sel, `/eval/${split}`).catch(() => null));
  async function renderEval() {
    K.$$("#splitSeg button").forEach((b) => b.classList.toggle("on", b.dataset.v === S.split));
    const r = S.run, ev = r.eval, sum = ev && ev[S.split];
    const clear = (msg) => { $("evalKpis").innerHTML = `<p class="empty">${msg}</p>`; ["sweepChart", "confusion", "pairChart", "deltaChart", "sizeGain", "scoreGain", "patTable"].forEach((id) => { $(id).innerHTML = ""; }); };
    if (!sum) return clear(LIVE.includes(r.state) ? "The final check runs after training." : "This run has not been scored on this pool.");
    const data = await evalData(S.split);
    if (!data) return clear("The result files were not found.");
    const recs = data.patients;
    S.recs = recs;
    const d = recs.map((p) => p.delta), c = St.bootCI(d), w = St.wilcoxon(d);
    $("evalKpis").innerHTML = [
      K.kpi(`MedSAM2 · ${S.split}`, K.f4(sum.dice3d_mean), `median ${K.f4(sum.dice3d_median)} · ${sum.patients} patients`, { hero: true }),
      K.kpi("YOLO hint alone", K.f4(sum.yolo_dice3d_mean), `median ${K.f4(sum.yolo_dice3d_median)}`),
      K.kpi("Gain", K.deltaHtml(sum.delta_mean), c ? `95% range ${K.signed(c[0])} … ${K.signed(c[1])}` : ""),
      K.kpi("Is the gain real?", w ? St.pfmt(w.p) : "–", w ? `Wilcoxon, n = ${w.n}` : ""),
      K.kpi("Helped / hurt", `${sum.helped} / ${sum.hurt}`, `${sum.unchanged} about the same (±0.01)`),
      K.kpi("Tumour-slice Dice", K.f4(sum.tumour_slice_dice), `old-style ${K.f4(sum.legacy_slice_dice)} · YOLO ${K.f4(sum.yolo_legacy_slice_dice)}`),
      K.kpi("Cut-off", sum.threshold, `best here ${sum.best_threshold}${ev.val ? ` · best on val ${ev.val.best_threshold}` : ""}`),
    ].join("");
    const series = ["val", "test"].filter((s) => ev[s]).map((s) => ({ name: `${s} pool`, color: s === "test" ? "var(--test)" : "var(--medsam2)", dash: s === "val" ? "6 4" : null, dots: true, points: ev[s].sweep.map((p) => [p.threshold, p.dice3d_mean]) }));
    Charts.line($("sweepChart"), { series, height: 240, xLabel: "Cut-off on MedSAM2's logit (0 = probability 0.5)", yLabel: "Mean 3D Dice", exportName: `${S.sel}_sweep`,
      markers: [{ x: sum.threshold, label: "setting" }, ...(ev.val && ev.val.best_threshold !== sum.threshold ? [{ x: ev.val.best_threshold, label: "best on val", color: "var(--medsam2)" }] : [])] });
    const dd = sum.detection;
    $("confusion").innerHTML = `<div class="tbl-wrap short"><table class="tbl"><thead><tr><th></th><th class="num">MedSAM2 drew</th><th class="num">drew nothing</th></tr></thead><tbody>
      <tr><td>has tumour</td><td class="num good">${dd.tp.toLocaleString()} found</td><td class="num bad">${dd.fn.toLocaleString()} missed</td></tr>
      <tr><td>no tumour</td><td class="num bad">${dd.fp.toLocaleString()} false</td><td class="num good">${dd.tn.toLocaleString()} empty</td></tr></tbody></table></div>`;
    $("confNote").textContent = `Found ${K.pct(sum.sensitivity)} of tumour slices; left ${K.pct(sum.specificity)} of empty slices empty. Every brain slice of every ${S.split} patient, counted once.`;
    renderHelp();
  }
  function renderHelp() {
    const recs = S.recs || [];
    const col = (x) => (x > 0.01 ? "var(--good)" : x < -0.01 ? "var(--bad)" : "var(--neutral)");
    const tip = (p) => `${K.tipTitle(p.id)}${K.tipRow(K.color("var(--neutral)"), "YOLO hint", K.f4(p.yolo_dice3d))}${K.tipRow(K.color("var(--medsam2)"), "MedSAM2", K.f4(p.dice3d))}${K.tipRow(K.color(col(p.delta)), "change", K.signed(p.delta))}${K.tipRow(K.color("var(--faint)"), "tumour", `${K.fmt(p.gt_total / 1000)} mL`)}`;
    Charts.scatter($("pairChart"), { points: recs.map((p) => ({ x: p.yolo_dice3d, y: p.dice3d, id: p.id, color: col(p.delta), tip: tip(p) })), diagonal: true, xMin: 0, xMax: 1, yMin: 0, yMax: 1,
      xLabel: "YOLO hint 3D Dice", yLabel: "MedSAM2 3D Dice", height: 320, highlight: S.pPatient, exportName: `${S.sel}_pairs`, onClick: (p) => openPatient(p.id),
      legend: [{ name: "helped", color: "var(--good)" }, { name: "about the same", color: "var(--neutral)" }, { name: "hurt", color: "var(--bad)" }] });
    Charts.hist($("deltaChart"), { series: [{ name: "patients", color: "var(--medsam2)", values: recs.map((p) => p.delta) }], bins: 41, xLabel: "Change in 3D Dice (MedSAM2 − hint)", yLabel: "Patients",
      height: 320, brush: true, range: S.range, markers: [{ x: 0, label: "no change", color: "var(--faint)" }, { x: St.mean(recs.map((p) => p.delta)), label: "mean", color: "var(--medsam2)" }],
      exportName: `${S.sel}_change`, onBrush: (lo, hi) => { S.range = lo == null ? null : [lo, hi]; renderPatTable(); } });
    const buckets = [[0, 10], [10, 30], [30, 60], [60, 100], [100, Infinity]];
    const byB = buckets.map(([a, b]) => recs.filter((p) => p.gt_total / 1000 >= a && p.gt_total / 1000 < b));
    Charts.bars($("sizeGain"), { cats: buckets.map(([a, b], i) => `${b === Infinity ? `≥ ${a}` : `${a}–${b}`} mL (n=${byB[i].length})`), yLabel: "Mean 3D Dice", yMin: 0, yMax: 1, fmt: K.f3, height: 260, legend: true, exportName: `${S.sel}_gain_by_size`,
      series: [{ name: "YOLO hint", color: "var(--neutral)", stripes: true, values: byB.map((g) => St.mean(g.map((p) => p.yolo_dice3d))), err: byB.map((g) => St.bootCI(g.map((p) => p.yolo_dice3d), St.mean, 400)) },
               { name: "MedSAM2", color: "var(--medsam2)", values: byB.map((g) => St.mean(g.map((p) => p.dice3d))), err: byB.map((g) => St.bootCI(g.map((p) => p.dice3d), St.mean, 400)) }] });
    Charts.scatter($("scoreGain"), { points: recs.map((p) => ({ x: St.mean(p.score.filter((s) => s > 0)) || 0, y: p.delta, id: p.id, color: col(p.delta), tip: tip(p) })), trend: true, hlines: [{ y: 0 }],
      xLabel: "YOLO's mean blob score on slices it drew", yLabel: "Change in 3D Dice", height: 260, exportName: `${S.sel}_gain_vs_score`, onClick: (p) => openPatient(p.id) });
    renderPatTable();
  }
  function renderPatTable() {
    const rows = (S.recs || []).filter((p) => !S.range || (p.delta >= S.range[0] && p.delta <= S.range[1]));
    const cols = [
      { k: "id", label: "Patient", html: (v) => `<span class="mono">${esc(v)}</span>` },
      { k: "yolo_dice3d", label: "YOLO hint", num: true, fmt: K.f4 },
      { k: "dice3d", label: "MedSAM2", num: true, fmt: K.f4, bar: { max: 1, color: "var(--medsam2)" } },
      { k: "delta", label: "Change", num: true, html: (v) => K.deltaHtml(v), raw: (r) => r.delta },
      { k: "gt_total", label: "Tumour (mL)", num: true, get: (p) => p.gt_total / 1000, fmt: K.fmt },
      { k: "anchor", label: "Start slice", num: true, get: (p) => (p.anchor_z || [])[0] },
      { k: "missed", label: "Missed slices", num: true, get: (p) => p.gt.filter((g, i) => g && !p.pred[i]).length },
      { k: "false", label: "False slices", num: true, get: (p) => p.gt.filter((g, i) => !g && p.pred[i]).length },
    ];
    if (!S.tables.pat) S.tables.pat = K.table($("patTable"), cols, { rows, key: (p) => p.id, onRow: (p) => openPatient(p.id), exportName: `${S.sel}_${S.split}_patients`, sort: { k: "delta", dir: 1 }, selected: S.pPatient });
    else { S.tables.pat.state.selected = S.pPatient; S.tables.pat.update(rows); }
  }

  // ------------------------------------------------------------------ one patient + slices
  async function loadPatients() {
    const ids = await (S.patients[S.pSplit] ||= api(`api/pool_patients?split=${S.pSplit}`).then((d) => d.patients).catch(() => []));
    if (!S.pPatient || !ids.includes(S.pPatient)) S.pPatient = null;
    $("pSplit").value = S.pSplit;
    $("pWhich").value = S.which;
    $("pPatient").innerHTML = ids.map((id) => `<option ${id === S.pPatient ? "selected" : ""}>${esc(id)}</option>`).join("");
    if (S.pPatient && (!S.prof || S.prof.id !== S.pPatient)) showPatient();
  }
  function openPatient(id) {
    S.pSplit = S.split; S.pPatient = id; S.z = null;
    loadPatients().then(showPatient);
    $("secPatient").scrollIntoView({ behavior: "smooth" });
  }
  async function showPatient() {
    S.pPatient = $("pPatient").value || S.pPatient;
    if (!S.pPatient || !S.sel) return;
    toHash();
    $("profNote").textContent = "Working…";
    let p = null;
    if (S.which === "best.pt" && (S.pSplit === "test" || S.pSplit === "val")) p = await runApi(S.sel, `/eval/${S.pSplit}?patient=${encodeURIComponent(S.pPatient)}`).catch(() => null);
    if (!p) {
      try { p = await runApi(S.sel, `/profile.json?patient=${encodeURIComponent(S.pPatient)}&which=${S.which}`); }
      catch (e) { $("profNote").innerHTML = `<span class="bad">${esc(e.message)}</span>`; return; }
    }
    S.prof = p;
    const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
    const tum = (xs) => xs.filter((_, i) => p.gt[i] > 0);
    $("profNote").innerHTML = `${p.stored ? "From the stored results of the last scoring pass (no GPU)." : "Worked out just now on the graphics card."}
      Mean slice Dice over all ${p.z.length} slices: MedSAM2 <b>${K.f4(mean(p.dice))}</b>, YOLO hint ${K.f4(mean(p.yolo_dice))};
      over the ${tum(p.dice).length} tumour slices: MedSAM2 <b>${K.f4(mean(tum(p.dice)))}</b>, YOLO hint ${K.f4(mean(tum(p.yolo_dice)))}. A slice where both are empty scores 1.`;
    $("profKpis").innerHTML = [
      K.kpi("MedSAM2 3D Dice", K.f4(p.dice3d), `${esc(p.id)}`, { hero: true }),
      K.kpi("YOLO hint", K.f4(p.yolo_dice3d), "same slices"),
      K.kpi("Change", K.deltaHtml(p.delta), ""),
      K.kpi("Expert tumour", `${K.fmt(p.gt_total / 1000)} mL`, `${p.gt.filter((g) => g).length} slices`),
      K.kpi("Start slice", (p.anchor_z || []).join(", ") || "–", "segmented without memory"),
    ].join("");
    const peak = p.z[p.gt.indexOf(Math.max(...p.gt))];
    if (S.z == null || !p.z.includes(S.z)) S.z = peak;
    drawProfile();
    setupSlices();
  }
  function drawProfile() {
    const p = S.prof;
    if (!p) return;
    S.charts.prof = Charts.line($("profChart"), { height: 290, xLabel: "Slice (z)", yLabel: "Dice on the slice", y2Label: "Tumour pixels", y2Min: 0, yMin: 0, yMax: 1, exportName: `${p.id}_slices`,
      series: [{ name: "Dice · MedSAM2", color: "var(--medsam2)", points: p.z.map((z, i) => [z, p.dice[i]]) },
               { name: "Dice · YOLO hint", color: "var(--yolo)", points: p.z.map((z, i) => [z, p.yolo_dice[i]]) },
               { name: "expert pixels", color: "var(--gt)", axis: "right", area: true, points: p.z.map((z, i) => [z, p.gt[i]]) },
               { name: "MedSAM2 pixels", color: "var(--medsam2)", axis: "right", dash: "2 3", width: 1.2, points: p.z.map((z, i) => [z, p.pred[i]]) },
               { name: "YOLO pixels", color: "var(--yolo)", axis: "right", dash: "2 3", width: 1.2, points: p.z.map((z, i) => [z, p.yolo[i]]) }],
      markers: [{ x: S.z, label: `z ${S.z}`, color: "var(--accent)", solid: true }, ...(p.anchor_z || []).map((z) => ({ x: z, label: "start", color: "var(--medsam2)" }))],
      onClick: (x) => setZ(Math.round(x)) });
  }
  function setupSlices() {
    const p = S.prof;
    $("zRange").min = 0; $("zRange").max = p.z.length - 1; $("zRange").value = Math.max(0, p.z.indexOf(S.z));
    loadSlice();
  }
  function setZ(z) {
    const p = S.prof;
    if (!p) return;
    S.z = p.z.reduce((b, v) => (Math.abs(v - z) < Math.abs(b - z) ? v : b), p.z[0]);
    $("zRange").value = p.z.indexOf(S.z);
    if (S.charts.prof) S.charts.prof.update({ markers: [{ x: S.z, label: `z ${S.z}`, color: "var(--accent)", solid: true }, ...(p.anchor_z || []).map((zz) => ({ x: zz, label: "start", color: "var(--medsam2)" }))] });
    toHash();
    loadSlice();
  }
  const panels = () => K.$$(".panel").filter((c) => c.checked).map((c) => c.value);
  const lg = (c, t) => `<span class="lg" style="--c:${c}">${t}</span>`;
  const RESULT = `${lg("var(--tp)", "found")} ${lg("var(--fn)", "missed")} ${lg("var(--fp)", "drawn, not tumour")}`;
  const PANEL_KEY = {
    frame: ["The picture", `${lg("#e33", "T1C")} ${lg("#3c3", "T2")} ${lg("#46f", "FLAIR")}, mixed as colours`],
    prompt: ["YOLO's hint (on FLAIR)", `how sure YOLO is: ${lg("#7a3fa0", "barely")} ${lg("#f08a00", "maybe")} ${lg("#e33", "sure")} · none = FLAIR only`],
    yolo: ["YOLO's own mask", RESULT],
    medsam2: ["MedSAM2's answer", RESULT],
  };
  function panelKey() {
    const ps = panels();
    $("panelKey").style.gridTemplateColumns = `repeat(${ps.length || 1}, minmax(0, 1fr))`;
    $("panelKey").innerHTML = ps.map((p) => `<div><b>${PANEL_KEY[p][0]}</b><br>${PANEL_KEY[p][1]}</div>`).join("");
  }
  const loadSlice = K.debounce(async () => {
    if (!S.prof || S.z == null) return;
    panelKey();
    const q = `patient=${encodeURIComponent(S.prof.id)}&which=${S.which}&z=${S.z}`;
    $("zLabel").textContent = `z = ${S.z}`;
    $("sliceLoading").classList.remove("hidden");
    const img = $("sliceImg");
    img.onload = () => $("sliceLoading").classList.add("hidden");
    img.onerror = () => { $("sliceLoading").textContent = "This slice could not be drawn."; };
    img.src = `${BASE}api/runs/${encodeURIComponent(S.sel)}/slice.png?${q}&panels=${panels().join(",")}`;
    try {
      const i = await runApi(S.sel, `/slice.json?${q}`);
      $("sliceSide").innerHTML = [
        K.kpi("Expert", `${i.gt.toLocaleString()} px`, `slice ${i.z}`),
        K.kpi("MedSAM2", K.f3(i.medsam2.dice), `${i.medsam2.px.toLocaleString()} px drawn · ${i.medsam2.inter.toLocaleString()} right`, { hero: true }),
        K.kpi("YOLO", K.f3(i.yolo.dice), `${i.yolo.px.toLocaleString()} px drawn · ${i.yolo.inter.toLocaleString()} right`),
        K.kpi("YOLO's best blob", K.f2(i.yolo_score), `${i.blobs} blob(s)`),
        K.kpi("MedSAM2 “tumour here?”", K.f2(i.obj_score), "above 0 = yes"),
        K.kpi("This slice", i.is_anchor ? "start slice" : "hint + memory", `start at z = ${(i.anchor_z || []).join(", ")}`),
      ].join("");
    } catch (e) { $("sliceSide").innerHTML = `<p class="bad">${esc(e.message)}</p>`; }
  }, 150);
  let cine = null;
  function play() {
    if (cine) { clearInterval(cine); cine = null; $("zPlay").textContent = "▶ Play"; return; }
    const p = S.prof;
    if (!p) return;
    const tz = p.z.filter((z, i) => p.gt[i] || p.pred[i]);
    if (!tz.length) return;
    let k = Math.max(0, tz.indexOf(S.z));
    $("zPlay").textContent = "■ Stop";
    cine = setInterval(() => { k = (k + 1) % tz.length; setZ(tz[k]); }, 700);
  }

  // ------------------------------------------------------------------ live
  function renderLive(run) {
    const st = run ? run.status : null;
    const live = st && LIVE.includes(run.state);
    $("stopBtn").disabled = !live;
    $("resumeBtn").disabled = !run || live || !!S.active || !run.has_last || run.state === "done";
    $("rescoreBtn").disabled = !run || live || !!S.active || !run.has_best;
    $("liveId").textContent = run ? run.id : "";
    if (!run) { $("liveSub").textContent = "Nothing is running."; $("liveKpis").innerHTML = ""; return; }
    const stages = ["prompts", "train", "evaluate", "done"];
    const cur = run.state === "done" ? 3 : Math.max(0, stages.indexOf(st.stage || "prompts"));
    const prog = { prompts: st.prompt_total ? st.prompt_done / st.prompt_total : 0, train: st.epochs ? ((st.epoch || 1) - 1 + (st.batches ? (st.batch || 0) / st.batches : 0)) / st.epochs : 0,
      evaluate: st.eval_total ? st.eval_done / st.eval_total : 0, done: run.state === "done" ? 1 : 0 };
    const txt = { prompts: st.prompt_total ? `${st.prompt_done} of ${st.prompt_total} patients` : "", train: st.epochs ? `round ${st.epoch} of ${st.epochs}${st.batches && cur === 1 ? ` · step ${st.batch}/${st.batches}` : ""}` : "",
      evaluate: st.eval_total ? `${st.eval_done} of ${st.eval_total}${st.eval_split ? ` (${st.eval_split})` : ""}` : "", done: run.state === "done" ? K.when(st.finished) : "" };
    K.$$("#stepper li").forEach((li, i) => {
      const k = li.dataset.stage, bad = !live && run.state !== "done" && i === cur;
      li.className = i < cur || run.state === "done" ? "done" : i === cur ? (bad ? "bad" : "cur") : "";
      K.$("i", li).style.width = `${(i < cur ? 1 : i === cur ? prog[k] : 0) * 100}%`;
      K.$("em", li).textContent = txt[k];
    });
    const res = run.results || {};
    const left = st.epoch_time && st.epochs && cur === 1 ? (st.epochs - (st.epoch || 1) + 1 - (st.batches ? (st.batch || 0) / st.batches : 0)) * st.epoch_time : null;
    $("liveSub").innerHTML = `${badge(run.state)} weights: ${esc(run.config.train.unfreeze)} · hints from <code>${esc(run.config.prompt.yolo_run)}</code>${run.smoke ? " · quick test" : ""} · started ${esc(K.when(st.started))} (${K.ago(st.started)})${st.error ? ` · <span class="bad">${esc(st.error)}</span>` : ""}`;
    $("liveKpis").innerHTML = [
      K.kpi("Time per round", st.epoch_time ? K.secs(st.epoch_time) : "–", `${(res.epoch || []).length} round(s) done`),
      K.kpi("Time left", left != null ? `≈ ${K.secs(left)}` : "–", left != null ? "a guess; it can stop earlier" : ""),
      K.kpi("Best check score", st.best_val_dice3d != null ? K.f4(st.best_val_dice3d) : "–", st.val_dice3d != null ? `last ${K.f4(st.val_dice3d)}` : "", { spark: res.val_dice3d, sparkColor: "var(--medsam2)" }),
      K.kpi("Rounds since the best", st.since_best != null ? st.since_best : "–", `stops at ${run.config.train.patience}`),
      K.kpi("Loss now", st.loss != null ? K.f4(st.loss) : "–", "lower is better"),
      K.kpi("Clips per round", st.train_clips ? st.train_clips.toLocaleString() : "–", st.train_patients ? `${st.train_patients} patients` : ""),
    ].join("");
    Charts.line($("liveChart"), { height: 230, xLabel: "Round", yLabel: "3D Dice (check)", y2Label: "loss", zoom: false, exportName: `${run.id}_live`, empty: "Waiting for the first round.",
      series: [{ name: "check 3D Dice", color: "var(--medsam2)", dots: true, points: (res.epoch || []).map((e, j) => [e, res.val_dice3d[j]]) },
               { name: "loss", color: "var(--faint)", axis: "right", dash: "5 4", points: (res.epoch || []).map((e, j) => [e, res.loss[j]]) }] });
  }
  async function pollLog(id) {
    if (S.logRun !== id) { S.logRun = id; S.logOffset = 0; $("log").textContent = ""; S.logText = ""; }
    const r = await runApi(id, `/log?offset=${S.logOffset}`);
    if (r.offset < S.logOffset) S.logText = "";
    S.logOffset = r.offset;
    if (r.text) S.logText = ((S.logText || "") + r.text).split("\n").filter((l) => !/Warning|warnings\.warn|forward_call|scaled_dot|Falling back/.test(l)).slice(-3000).join("\n");
    renderLog();
  }
  function renderLog() {
    const f = $("logFilter").value.trim().toLowerCase();
    const lines = (S.logText || "").split("\n").filter((l) => !f || l.toLowerCase().includes(f)).slice(-1500);
    $("log").innerHTML = lines.map((l) => {
      const cls = /Traceback|Error|failed/i.test(l) ? "err" : /\[eval\]|\(best\)/.test(l) ? "ok" : /^=====|\[run\]/.test(l) ? "hi" : "";
      return cls ? `<span class="${cls}">${esc(l)}</span>` : esc(l);
    }).join("\n") || "(nothing yet)";
    if ($("logFollow").checked) $("log").scrollTop = $("log").scrollHeight;
  }

  // ------------------------------------------------------------------ start dialog (built from routes.EDITABLE)
  function openStart() {
    const ov = S.ov;
    if (!ov) return;
    const cfg = ov.config;
    const val = (k) => k.split(".").reduce((o, p) => (o == null ? o : o[p]), cfg);
    const groups = {};
    for (const [k, spec] of Object.entries(ov.editable)) (groups[GROUP[k.split(".")[0]] || "Other"] ||= []).push([k, spec]);
    const field = ([k, spec]) => {
      const v = val(k);
      const id = `f_${k.replace(/\./g, "_")}`;
      const rb = ov.rebuilds.includes(k) ? ' <span class="warn" title="Changing this rebuilds the YOLO hints">⟳</span>' : "";
      if (spec.kind === "choice") {
        const opts = spec.choices.map((c) => `<option value="${esc(c)}" ${String(c) === String(v ?? "") ? "selected" : ""}>${esc(c === "" ? "best scoring (automatic)" : c)}</option>`).join("");
        return `<label class="field"><span>${esc(LABELS[k] || k)}${rb}</span><select id="${id}" data-k="${k}">${opts}</select><span class="hint mono">${esc(k)}</span></label>`;
      }
      return `<label class="field"><span>${esc(LABELS[k] || k)}${rb}</span><input id="${id}" data-k="${k}" type="${spec.kind === "int?" ? "text" : "number"}" step="any" value="${v == null ? "" : v}" ${spec.kind !== "int?" ? `min="${spec.min}" max="${spec.max}"` : `placeholder="all"`}>
        <span class="hint mono">${esc(k)}</span></label>`;
    };
    const html = `<form id="startForm">${Object.entries(groups).map(([g, fs]) => `<h4 style="margin:10px 0 8px">${esc(g)}</h4><div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px">${fs.map(field).join("")}</div>`).join("")}
      <div class="row" style="margin-top:12px"><label class="inline"><input type="checkbox" id="fSmoke"> quick test (2 patients, 1 round)</label></div>
      <p class="callout" id="estimate" style="margin-top:10px"></p><p class="bad" id="formErr"></p>
      <div class="row"><button class="primary big" type="submit">▶ Start</button><span class="note">Runs as its own process — closing this page does not stop it.</span></div></form>
      <details style="margin-top:12px"><summary class="dim" style="cursor:pointer">Why each default is what it is — MedSAM2_Finetune/config.yaml</summary><pre class="yaml" id="cfgText" style="margin-top:8px"></pre></details>`;
    const m = K.modal("Start a MedSAM2 fine-tune", html, { wide: true });
    const changed = () => K.$$("[data-k]", m.el).filter((el) => String(el.value) !== String(val(el.dataset.k) ?? "")).map((el) => el.dataset.k);
    const est = () => {
      const ch = changed();
      const n = +(K.$('[data-k="data.max_train_patients"]', m.el) || {}).value || ov.pools.train;
      const clips = n * +(K.$('[data-k="data.clips_per_patient"]', m.el) || { value: 1 }).value;
      K.$("#estimate", m.el).innerHTML = `About <b>${n}</b> patients per round = <b>${clips.toLocaleString()}</b> clips of ${esc((K.$('[data-k="video.num_frames"]', m.el) || {}).value)} slices. ` +
        (ch.length ? `Changed: ${ch.map((k) => `<code>${esc(k)}</code>`).join(", ")}.` : "Everything as in config.yaml.") +
        (ch.some((k) => ov.rebuilds.includes(k)) ? " <b>The YOLO hints will be worked out again</b> for every patient used." : "");
    };
    m.el.addEventListener("input", est);
    est();
    K.$("#startForm", m.el).addEventListener("submit", async (e) => {
      e.preventDefault();
      const overrides = {};
      for (const k of changed()) overrides[k] = K.$(`[data-k="${k}"]`, m.el).value;
      const smoke = K.$("#fSmoke", m.el).checked;
      if (!confirm(`Start ${smoke ? "a quick test" : "a MedSAM2 fine-tune"}${Object.keys(overrides).length ? ` with ${Object.keys(overrides).length} changed setting(s)` : ""}? It uses the GPU until it finishes or you press Stop.`)) return;
      try {
        const r = await api("api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ overrides, smoke }) });
        m.close(); K.toast(`Started run <b>${esc(r.id)}</b>.`, "good");
        await loadOverview(); selectRun(r.id); $("secLive").scrollIntoView({ behavior: "smooth" });
      } catch (err) { K.$("#formErr", m.el).textContent = err.message; }
    });
    K.$("#cfgText", m.el).textContent = ov.config_text || "";
  }
  async function diffSettings() {
    const ids = [...S.cmp].slice(0, 4);
    if (ids.length < 2) return K.toast("Tick at least two runs first.");
    const runs = await Promise.all(ids.map((id) => S.cache[id] || loadRun(id)));
    const flat = (o, p = "") => Object.entries(o || {}).flatMap(([k, v]) => (v && typeof v === "object" && !Array.isArray(v) ? flat(v, `${p}${k}.`) : [[`${p}${k}`, JSON.stringify(v)]]));
    const maps = runs.map((r) => Object.fromEntries(flat(r.config).filter(([k]) => !k.startsWith("run."))));
    const keys = [...new Set(maps.flatMap((m) => Object.keys(m)))].filter((k) => new Set(maps.map((m) => m[k])).size > 1).sort();
    K.modal("Settings that differ", keys.length ? `<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Setting</th>${ids.map((id) => `<th class="mono">${esc(id)}</th>`).join("")}</tr></thead><tbody>
      ${keys.map((k) => `<tr><td class="mono">${esc(k)}</td>${maps.map((m) => `<td class="mono">${esc(m[k] ?? "–")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
      <p class="note">Test 3D Dice: ${runs.map((r) => `${esc(r.id)} ${r.test ? K.f4(r.test.dice3d_mean) : "–"}`).join(" · ")}. For a paired test of two runs use the Results page (Run against run).</p>` : "<p>These runs used the same settings.</p>", { wide: true });
  }

  // ------------------------------------------------------------------ polling
  let pollT = null;
  async function poll() {
    clearTimeout(pollT);
    try {
      const was = S.active;
      await loadOverview();
      if (was && !S.active) K.toast(`Run <b>${esc(was)}</b> has ended.`, "good", 9000);
      const liveId = S.active || S.sel;
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
      }
      document.title = S.active ? "● MedSAM2_Finetune · BraTS viewer" : "MedSAM2_Finetune · BraTS viewer";
    } catch (e) { console.error(e); }
    pollT = setTimeout(poll, document.hidden ? 15000 : S.active ? 3000 : 9000);
  }

  // ------------------------------------------------------------------ wiring
  function wire() {
    $("newRunBtn").addEventListener("click", openStart);
    $("diffBtn").addEventListener("click", diffSettings);
    $("splitSeg").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b && b.dataset.v !== S.split) { S.split = b.dataset.v; S.range = null; toHash(); renderEval(); } });
    $("cfgBtn").addEventListener("click", () => S.run && K.modal(`Settings of ${S.run.id}`, `<pre class="yaml">${esc(JSON.stringify(S.run.config, null, 2))}</pre>`, { wide: true }));
    $("csvBtn").addEventListener("click", () => {
      const rows = S.recs || [];
      K.download(`${S.sel}_${S.split}_patients.csv`, K.csv(rows, [{ k: "id", label: "patient" }, { k: "yolo_dice3d", label: "yolo_hint_dice3d" }, { k: "dice3d", label: "medsam2_dice3d" },
        { k: "delta", label: "change" }, { label: "gt_ml", raw: (p) => p.gt_total / 1000 }, { label: "start_slice", raw: (p) => (p.anchor_z || [])[0] }]), "text/csv");
    });
    const act = async (kind, q) => {
      if (!confirm(q)) return;
      try { await api(`api/runs/${encodeURIComponent(S.sel)}/${kind}`, { method: "POST" }); K.toast("Started.", "good"); poll(); } catch (e) { K.fail(e); }
    };
    $("resumeBtn").addEventListener("click", () => act("resume", "Continue training from last.pt?"));
    $("rescoreBtn").addEventListener("click", () => act("evaluate", "Score best.pt again on the whole val and test pools?"));
    $("stopBtn").addEventListener("click", async () => {
      const id = S.active;
      if (!id || !confirm(`Stop run ${id}? You can continue it later with Resume.`)) return;
      try { await api(`api/runs/${encodeURIComponent(id)}/stop`, { method: "POST" }); K.toast(`Stopped ${esc(id)}.`); } catch (e) { K.fail(e); }
      poll();
    });
    $("logFilter").addEventListener("input", renderLog);
    $("logDl").addEventListener("click", () => K.download(`${S.logRun || "run"}_log.txt`, S.logText || ""));
    $("pSplit").addEventListener("change", (e) => { S.pSplit = e.target.value; S.pPatient = null; loadPatients(); });
    $("pWhich").addEventListener("change", (e) => { S.which = e.target.value; showPatient(); });
    $("pPatient").addEventListener("change", (e) => { S.pPatient = e.target.value; S.z = null; toHash(); });
    $("pRun").addEventListener("click", () => { S.z = null; showPatient(); });
    $("zRange").addEventListener("input", (e) => S.prof && setZ(S.prof.z[+e.target.value]));
    $("zPlay").addEventListener("click", play);
    K.$$(".panel").forEach((c) => c.addEventListener("change", loadSlice));
    K.key("n", "Start a new run", () => !$("newRunBtn").disabled && openStart());
    K.key("arrowleft", "Previous slice", () => S.prof && setZ(S.prof.z[Math.max(0, S.prof.z.indexOf(S.z) - 1)]));
    K.key("arrowright", "Next slice", () => S.prof && setZ(S.prof.z[Math.min(S.prof.z.length - 1, S.prof.z.indexOf(S.z) + 1)]));
    K.key("p", "Play / stop the slices", play);
    K.key("[", "Previous patient", () => stepPatient(-1));
    K.key("]", "Next patient", () => stepPatient(1));
    K.command("Start a new MedSAM2 run", openStart, { group: "MedSAM2", hint: "N" });
    K.command("Compare settings of the ticked runs", diffSettings, { group: "MedSAM2" });
    K.provider((q) => [
      ...S.runs.filter((r) => r.id.includes(q)).slice(0, 6).map((r) => ({ label: `Open run ${r.id}`, hint: r.test ? `test ${K.f4(r.test.dice3d_mean)}` : WORD[r.state], group: "Runs", run: () => selectRun(r.id) })),
      ...(S.recs || []).filter((p) => p.id.toLowerCase().includes(q)).slice(0, 8).map((p) => ({ label: `Open patient ${p.id}`, hint: `change ${K.signed(p.delta, 3)}`, group: "Patients", run: () => openPatient(p.id) })),
    ]);
    window.addEventListener("hashchange", () => { const was = S.sel; fromHash(); if (S.sel && S.sel !== was) selectRun(S.sel, true); });
  }
  function stepPatient(d) {
    const opts = [...$("pPatient").options].map((o) => o.value);
    const i = opts.indexOf(S.pPatient) + d;
    if (i >= 0 && i < opts.length) { S.pPatient = opts[i]; $("pPatient").value = opts[i]; S.z = null; showPatient(); }
  }

  document.addEventListener("DOMContentLoaded", () => {
    wire();
    fromHash();
    if (S.sel) selectRun(S.sel, true).catch(() => { S.sel = null; });
    poll();
  });
})();
