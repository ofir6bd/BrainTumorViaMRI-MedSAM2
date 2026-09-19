/* YOLO_finetune page: start runs, follow the live one, analyse any run.
 * Everything shown comes from the run folders (status.json, results.csv, eval/*.json) —
 * nothing is estimated except the clearly labelled time-to-finish.
 * State that makes a view (run, split, patient, slice, threshold, compare set, curve group)
 * lives in the URL hash, so "Copy link" reproduces it. */
(function () {
  const $ = (id) => document.getElementById(id);
  const { fmt, esc, row } = window.Charts;
  const BASE = document.body.dataset.base; // "/finetune/" — this page is a blueprint of the main viewer
  const C = { val: "#3987e5", test: "#199e70", train: "#3987e5", valLoss: "#d95926", cmp: ["#3987e5", "#d95926", "#199e70"] };
  const STATE_LABEL = { queued: "starting", running: "running", done: "done", failed: "failed", stopped: "stopped", interrupted: "interrupted" };
  const LIVE = ["queued", "running"];

  const S = {
    ov: null, runs: [], active: null, sel: null, run: null, runCache: {}, cmp: new Set(),
    group: "loss", split: "test", evalData: {}, patSort: { k: "dice", dir: 1 }, patQ: "", patRange: null,
    vPatient: null, vZ: null, vConf: null, logRun: null, logOffset: 0, logLines: [],
  };

  const api = async (url, opt) => {
    const r = await fetch(url, opt);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.error || `${r.status} ${r.statusText}`);
    return body;
  };
  const pct = (v, d = 1) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);
  const dice = (inter, pred, gt) => (pred + gt ? (2 * inter) / (pred + gt) : 1);
  const dur = (s) => {
    if (s == null) return "–";
    s = Math.round(s);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    return h ? `${h} h ${m} min` : m ? `${m} min ${s % 60} s` : `${s} s`;
  };
  const since = (iso) => (iso ? (Date.now() - new Date(iso).getTime()) / 1000 : null);
  const badge = (state) => `<span class="badge st-${state || "none"}">${esc(STATE_LABEL[state] || state || "–")}</span>`;
  const kpi = (label, value, sub, cls) =>
    `<div class="kpi ${cls || ""}"><span class="kpi-l">${label}</span><span class="kpi-v">${value}</span>${sub ? `<span class="kpi-s">${sub}</span>` : ""}</div>`;

  // ------------------------------------------------------------------ URL state
  function toHash() {
    const q = new URLSearchParams();
    if (S.sel) q.set("run", S.sel);
    if (S.split !== "test") q.set("split", S.split);
    if (S.group !== "loss") q.set("group", S.group);
    if (S.cmp.size) q.set("cmp", [...S.cmp].join(","));
    if (S.vPatient) q.set("patient", S.vPatient);
    if (S.vZ != null) q.set("z", S.vZ);
    if (S.vConf != null) q.set("conf", S.vConf);
    return `#${q}`;
  }
  function fromHash() {
    const q = new URLSearchParams(location.hash.slice(1));
    S.sel = q.get("run") || null;
    S.split = q.get("split") === "val" ? "val" : "test";
    S.group = ["loss", "mask", "box", "lr"].includes(q.get("group")) ? q.get("group") : "loss";
    S.cmp = new Set((q.get("cmp") || "").split(",").filter(Boolean));
    S.vPatient = q.get("patient") || null;
    S.vZ = q.get("z") != null ? +q.get("z") : null;
    S.vConf = q.get("conf") != null ? +q.get("conf") : null;
  }
  const syncHash = () => history.replaceState(null, "", toHash());

  // ------------------------------------------------------------------ overview + start form
  function highlightYaml(text) {
    return text.split("\n").map((ln) => {
      const i = ln.indexOf("#");
      const code = i >= 0 ? ln.slice(0, i) : ln;
      const cmt = i >= 0 ? `<span class="y-c">${esc(ln.slice(i))}</span>` : "";
      const m = code.match(/^(\s*)([\w.-]+)(:)(.*)$/);
      const body = m ? `${m[1]}<span class="y-k">${esc(m[2])}</span>:${m[4] ? `<span class="y-v">${esc(m[4])}</span>` : ""}` : esc(code);
      return body + cmt;
    }).join("\n");
  }

  function fillForm(cfg) {
    $("fModel").value = cfg.model;
    $("fEpochs").value = cfg.train.epochs;
    $("fPatience").value = cfg.train.patience;
    $("fImgsz").value = cfg.train.imgsz;
    $("fBatch").value = cfg.train.batch;
    $("fSmoke").checked = false;
    updateEstimate();
  }

  function updateEstimate() {
    if (!S.ov) return;
    const p = S.ov.pools;
    const smoke = $("fSmoke").checked;
    const n = (k) => (smoke ? Math.min(2, p[k]) : p[k]);
    $("estimate").innerHTML = `Trains on <b>${n("train")}</b> patients of <code>${esc(S.ov.config.data.train_pool)}</code>, picks the best epoch on <b>${n("val")}</b> of <code>${esc(S.ov.config.data.val_pool)}</code>, then scores val and <b>${n("test")}</b> of <code>${esc(S.ov.config.data.test_pool)}</code>. ` +
      (smoke ? "Smoke test: 2 epochs." : `Up to <b>${esc($("fEpochs").value)}</b> epochs, stopping early after <b>${esc($("fPatience").value)}</b> without improvement.`);
  }

  async function loadOverview() {
    const ov = await api(BASE + "api/overview");
    const first = !S.ov;
    S.ov = ov;
    S.runs = ov.runs;
    S.active = ov.active;
    if (first) {
      $("fModel").innerHTML = ov.models.map((m) => `<option>${esc(m)}</option>`).join("");
      for (const [key, id] of [["train.epochs", "fEpochs"], ["train.patience", "fPatience"], ["train.imgsz", "fImgsz"], ["train.batch", "fBatch"]]) {
        $(id).min = ov.editable[key].min;
        $(id).max = ov.editable[key].max;
      }
      fillForm(ov.config);
      $("configText").innerHTML = highlightYaml(ov.config_text);
      for (const n of document.querySelectorAll("[data-cfg]")) {  // numbers quoted in "How it works"
        n.textContent = n.dataset.cfg.split(".").reduce((o, k) => o[k], ov.config);
      }
    }
    $("topSub").innerHTML = `${esc(ov.config.model)} on <code>${esc(ov.config.data.train_pool)}</code> (${ov.pools.train}) · val <code>${esc(ov.config.data.val_pool)}</code> (${ov.pools.val}) · test <code>${esc(ov.config.data.test_pool)}</code> (${ov.pools.test}) · ${ov.runs.length} run${ov.runs.length === 1 ? "" : "s"}`;
    $("startBtn").disabled = !!S.active;
    $("startBtn").title = S.active ? `Run ${S.active} is still going` : "";
    renderRuns();
    if (!S.sel && S.runs.length) selectRun((S.active || S.runs[0].id), true);
  }

  $("startForm").addEventListener("input", updateEstimate);
  $("resetForm").addEventListener("click", () => fillForm(S.ov.config));
  $("copyCfg").addEventListener("click", () => navigator.clipboard.writeText(S.ov.config_text).then(() => flash($("copyCfg"), "Copied")));
  $("startForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    $("formErr").textContent = "";
    const body = {
      model: $("fModel").value, smoke: $("fSmoke").checked,
      overrides: { "train.epochs": +$("fEpochs").value, "train.patience": +$("fPatience").value,
                   "train.imgsz": +$("fImgsz").value, "train.batch": +$("fBatch").value },
    };
    if (body.overrides["train.imgsz"] % 32) {
      $("formErr").textContent = "Image size must be a multiple of 32.";
      $("fImgsz").focus();
      return;
    }
    const what = body.smoke ? "a smoke test (2 patients per pool, 2 epochs)"
      : `${body.model}, up to ${body.overrides["train.epochs"]} epochs, imgsz ${body.overrides["train.imgsz"]}, batch ${body.overrides["train.batch"]}`;
    if (!confirm(`Start ${what}?\n\nIt uses the GPU until it finishes or you press Stop.`)) return;
    $("startBtn").disabled = true;
    try {
      const r = await api(BASE + "api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      S.active = r.id;
      await loadOverview();
      selectRun(r.id);
      $("secLive").scrollIntoView({ behavior: "smooth" });
    } catch (err) {
      $("formErr").textContent = err.message;
      $("startBtn").disabled = !!S.active;
    }
  });

  // ------------------------------------------------------------------ runs table
  function renderRuns() {
    if (!S.runs.length) {
      $("runsTable").innerHTML = `<p class="empty">No runs yet — start one above.</p>`;
      return;
    }
    const rows = S.runs.map((r) => {
      const t = r.test;
      return `<tr data-id="${esc(r.id)}" class="${r.id === S.sel ? "sel" : ""}" tabindex="0">
        <td><input type="checkbox" class="cmp" data-id="${esc(r.id)}" ${S.cmp.has(r.id) ? "checked" : ""} aria-label="Compare ${esc(r.id)}"></td>
        <td class="mono">${esc(r.id)}${r.smoke ? ' <span class="badge smoke">smoke</span>' : ""}</td>
        <td>${badge(r.state)}${r.error ? ` <span class="err" title="${esc(r.error)}">!</span>` : ""}</td>
        <td>${esc(r.model)}</td>
        <td class="num">${r.epochs_done}/${r.epochs || r.epochs_cfg}</td>
        <td class="num">${r.best ? `${fmt(r.best.map5095_m)} <span class="dim">@${r.best.epoch}</span>` : "–"}</td>
        <td class="num">${t ? `<b>${t.dice3d_mean.toFixed(4)}</b>` : "–"}</td>
        <td class="num">${t ? t.tumour_slice_dice.toFixed(4) : "–"}</td>
        <td class="num">${t ? t.legacy_slice_dice.toFixed(4) : "–"}</td>
        <td class="num">${dur(r.train_seconds)}</td>
        <td>${esc((r.created || "").replace("T", " "))}</td></tr>`;
    }).join("");
    $("runsTable").innerHTML = `<table class="tbl"><thead><tr><th title="Compare">&#8645;</th><th>Run</th><th>State</th><th>Model</th>
      <th class="num">Epochs</th><th class="num" title="Best mask mAP50-95 on val">Best mAP50-95 (M)</th>
      <th class="num" title="Mean 3D Dice per test patient">Test 3D Dice</th><th class="num" title="Mean Dice over test slices that have tumour">Tumour-slice Dice</th>
      <th class="num" title="Old YOLO/ metric: mean over all brain slices, an empty-vs-empty slice counts 1.0">Legacy slice Dice</th>
      <th class="num">Train time</th><th>Created</th></tr></thead><tbody>${rows}</tbody></table>`;
  }
  $("runsTable").addEventListener("click", (e) => {
    const cb = e.target.closest(".cmp");
    if (cb) {
      if (cb.checked && S.cmp.size >= 3) {
        cb.checked = false;
        alert("Compare up to 3 runs at a time.");
        return;
      }
      if (cb.checked) S.cmp.add(cb.dataset.id); else S.cmp.delete(cb.dataset.id);
      syncHash();
      renderCurves();
      return;
    }
    const tr = e.target.closest("tr[data-id]");
    if (tr) selectRun(tr.dataset.id);
  });
  $("runsTable").addEventListener("keydown", (e) => {
    const tr = e.target.closest("tr[data-id]");
    if (e.key === "Enter" && tr) selectRun(tr.dataset.id);
  });

  // ------------------------------------------------------------------ run selection
  async function loadRun(id) {
    const run = await api(`${BASE}api/runs/${encodeURIComponent(id)}`);
    S.runCache[id] = run;
    return run;
  }

  async function selectRun(id, keepViewer) {
    if (S.sel !== id && !keepViewer) {
      S.vPatient = null;
      S.vZ = null;
      S.vConf = null;
    }
    S.sel = id;
    S.evalData = {};
    S.patRange = null;
    syncHash();
    renderRuns();
    S.run = await loadRun(id);
    renderAnalysis();
  }

  function renderAnalysis() {
    const r = S.run;
    if (!r) return;
    $("analysis").classList.remove("hidden");
    $("selId").textContent = r.id;
    $("selSub").innerHTML = `${badge(r.state)} ${esc(r.model)} · imgsz ${r.imgsz} · batch ${r.batch} · ` +
      `${r.epochs_done} epoch${r.epochs_done === 1 ? "" : "s"} trained${r.smoke ? " · smoke test" : ""}` +
      (r.error ? ` · <span class="err">${esc(r.error)}</span>` : "");
    const live = LIVE.includes(r.state);
    $("resumeBtn").disabled = live || !!S.active || !r.has_last || r.state === "done";
    $("reevalBtn").disabled = live || !!S.active || !r.has_best;
    $("dlResults").disabled = !r.results || !r.results.epoch;
    $("runCfg").innerHTML = highlightYaml(r.config_text);

    const t = r.eval && r.eval.test;
    const v = r.eval && r.eval.val;
    const best = r.best;
    $("kpis").innerHTML = [
      kpi("Test 3D Dice", t ? t.dice3d_mean.toFixed(4) : "–", t ? `median ${t.dice3d_median.toFixed(4)} · ${t.patients} patients` : "after evaluation", "hero"),
      kpi("Tumour-slice Dice", t ? t.tumour_slice_dice.toFixed(4) : "–", "test slices that have tumour"),
      kpi("Legacy slice Dice", t ? t.legacy_slice_dice.toFixed(4) : "–", "old YOLO/ metric · was 0.8662"),
      kpi("Slice sensitivity", t ? pct(t.sensitivity) : "–", t ? `specificity ${pct(t.specificity)}` : ""),
      kpi("Best mAP50-95 (mask)", best ? fmt(best.map5095_m) : "–", best ? `epoch ${best.epoch} · mAP50 ${fmt(best.map50_m)}` : "val, during training"),
      kpi("Val-chosen threshold", v ? v.best_conf : "–", v ? `config uses ${v.conf}` : ""),
      kpi("Training time", dur(r.train_seconds), r.results.time ? `${dur(r.train_seconds / r.epochs_done)} per epoch` : ""),
    ].join("");
    renderCurves();
    renderDataset();
    renderEval();
    renderPlots();
  }

  $("showRunCfg").addEventListener("click", () => $("runCfg").classList.toggle("hidden"));
  $("dlResults").addEventListener("click", () => { location.href = `${BASE}api/runs/${encodeURIComponent(S.sel)}/file/results.csv?download=1`; });
  $("resumeBtn").addEventListener("click", () => runAction("resume", "Resume training from last.pt?"));
  $("reevalBtn").addEventListener("click", () => runAction("evaluate", "Evaluate best.pt again on val and test?"));
  async function runAction(kind, question) {
    if (!confirm(question)) return;
    try {
      await api(`${BASE}api/runs/${encodeURIComponent(S.sel)}/${kind}`, { method: "POST" });
      await loadOverview();
      poll();
    } catch (err) {
      alert(err.message);
    }
  }

  // ------------------------------------------------------------------ curves
  const GROUPS = {
    loss: { label: "Losses", charts: ["box", "seg", "cls", "dfl"].map((k) => ({
      title: `${k} loss`, series: [{ col: `train/${k}_loss`, name: "train", color: C.train }, { col: `val/${k}_loss`, name: "val", color: C.valLoss, dash: "6 4" }],
      cmpCol: `val/${k}_loss` })) },
    mask: { label: "Mask metrics", charts: ["precision", "recall", "mAP50", "mAP50-95"].map((k) => ({
      title: `mask ${k}`, series: [{ col: `metrics/${k}(M)`, name: `mask ${k}`, color: C.train }], cmpCol: `metrics/${k}(M)`, y01: true })) },
    box: { label: "Box metrics", charts: ["precision", "recall", "mAP50", "mAP50-95"].map((k) => ({
      title: `box ${k}`, series: [{ col: `metrics/${k}(B)`, name: `box ${k}`, color: C.train }], cmpCol: `metrics/${k}(B)`, y01: true })) },
    lr: { label: "Learning rate", charts: [{ title: "learning rate", series: [0, 1, 2].map((i) => ({ col: `lr/pg${i}`, name: `group ${i}`, color: C.cmp[i], dash: i ? "5 4" : null })), cmpCol: "lr/pg0" }] },
  };
  $("curveGroups").innerHTML = Object.entries(GROUPS).map(([k, g]) => `<button type="button" class="chip" data-group="${k}">${g.label}</button>`).join("");
  $("curveGroups").addEventListener("click", (e) => {
    const b = e.target.closest("[data-group]");
    if (!b) return;
    S.group = b.dataset.group;
    syncHash();
    renderCurves();
  });

  function bestEpoch(res) {
    const f = res["metrics/mAP50-95(M)"];
    if (!f || !f.length) return null;
    let bi = 0;
    f.forEach((v, i) => { if (v != null && v > f[bi]) bi = i; });
    return res.epoch[bi];
  }

  async function renderCurves() {
    for (const b of $("curveGroups").querySelectorAll(".chip")) b.classList.toggle("on", b.dataset.group === S.group);
    const g = GROUPS[S.group];
    const box = $("curves");
    const cmpIds = [...S.cmp];
    const comparing = cmpIds.length >= 2;
    const runs = comparing ? await Promise.all(cmpIds.map((id) => S.runCache[id] || loadRun(id).catch(() => null))) : [S.run];
    const valid = runs.filter((r) => r && r.results && r.results.epoch && r.results.epoch.length);
    box.innerHTML = g.charts.map((_, i) => `<div class="curve"><h4>${esc(g.charts[i].title)}</h4><div id="curve${i}"></div></div>`).join("");
    $("curvesSub").textContent = comparing
      ? `Comparing ${valid.length} runs — ${cmpIds.join(", ")}. Val curves where train/val exist.`
      : "Per epoch, from Ultralytics' results.csv. Hover to read every chart at the same epoch.";
    if (!valid.length) {
      box.innerHTML = `<p class="empty">No epochs finished yet.</p>`;
      $("curveNote").textContent = "";
      return;
    }
    const sync = { listeners: new Set() };
    g.charts.forEach((ch, i) => {
      const series = comparing
        ? valid.map((r, k) => ({ name: r.id, color: C.cmp[k], dash: k === 1 ? "6 4" : k === 2 ? "2 3" : null,
                                 points: r.results.epoch.map((e, j) => [e, (r.results[ch.cmpCol] || [])[j]]) }))
        : ch.series.map((s) => ({ ...s, points: valid[0].results.epoch.map((e, j) => [e, (valid[0].results[s.col] || [])[j]]) }));
      const be = comparing ? null : bestEpoch(valid[0].results);
      Charts.line($(`curve${i}`), { series, sync, height: 180, xName: "epoch", xLabel: "epoch",
        yMin: ch.y01 ? 0 : undefined, markers: be ? [{ x: be, label: `best ${be}` }] : [],
        yFmt: S.group === "lr" ? (v) => v.toExponential(1) : undefined, left: S.group === "lr" ? 58 : 50 });
    });
    const legend = comparing ? valid.map((r, k) => `<span class="lgd"><i style="border-color:${C.cmp[k]};border-top-style:${k === 1 ? "dashed" : k === 2 ? "dotted" : "solid"}"></i>${esc(r.id)}</span>`).join("")
      : g.charts[0].series.map((s) => `<span class="lgd"><i style="border-color:${s.color};border-top-style:${s.dash ? "dashed" : "solid"}"></i>${esc(s.name)}</span>`).join("");
    const res = valid[0].results;
    let note = "";
    if (!comparing && res["val/seg_loss"]) {
      const vl = res["val/seg_loss"];
      let mi = 0;
      vl.forEach((v, i) => { if (v != null && v < vl[mi]) mi = i; });
      const since = res.epoch.length - 1 - mi;
      note = `Val seg loss is lowest at epoch ${res.epoch[mi]} (${fmt(vl[mi])})` +
        (since ? `, ${since} epoch${since > 1 ? "s" : ""} ago — rising since then means the model is starting to memorise the training slices.` : " — the latest epoch.");
    }
    $("curveNote").innerHTML = `<span class="lgds">${legend}</span> ${esc(note)}`;
  }

  // ------------------------------------------------------------------ dataset
  function renderDataset() {
    const m = S.run.dataset;
    if (!m) {
      $("dataSub").textContent = "Available once the run has built (or reused) its dataset.";
      ["dataBars", "dataTable", "areaBars"].forEach((id) => { $(id).innerHTML = ""; });
      return;
    }
    $("dataSub").innerHTML = `<code>dataset/${esc(m.key)}</code>, built ${esc(m.created.replace("T", " "))}. One PNG per axial slice with at least ${m.data.min_fg_voxels} brain voxels; tumour pieces under ${m.data.min_mask_area} px left out of the labels.`;
    const items = [];
    for (const [split, s] of Object.entries(m.splits)) {
      items.push({ label: `${split} · tumour`, value: s.positive, color: C[split] || C.train, tip: `<div class="tt-title">${split}</div>${s.positive.toLocaleString()} slices with tumour` });
      items.push({ label: `${split} · none`, value: s.negative, color: "#5a5a78", tip: `<div class="tt-title">${split}</div>${s.negative.toLocaleString()} slices without tumour (empty label)` });
    }
    Charts.bars($("dataBars"), { items, labelWidth: 110 });
    $("dataTable").innerHTML = `<table class="tbl small"><thead><tr><th>Split</th><th class="num">Patients</th><th class="num">Slices</th><th class="num">With tumour</th><th class="num">Tumour outlines</th><th class="num">Median size</th></tr></thead><tbody>` +
      Object.entries(m.splits).map(([k, s]) => `<tr><td>${k}</td><td class="num">${s.patients}</td><td class="num">${s.slices.toLocaleString()}</td><td class="num">${pct(s.positive / s.slices, 0)}</td><td class="num">${s.instances.toLocaleString()}</td><td class="num">${s.area_median != null ? `${fmt(s.area_median)} px` : "–"}</td></tr>`).join("") + "</tbody></table>";
    const bins = m.area_bins;
    const lab = (i) => (i === bins.length - 1 ? `≥ ${bins[i].toLocaleString()}` : `${bins[i].toLocaleString()}–${bins[i + 1].toLocaleString()}`);
    const tr = m.splits.train;
    const va = m.splits.val;
    Charts.bars($("areaBars"), {
      items: tr.area_hist.map((c, i) => ({ label: lab(i), value: c, color: C.train,
        text: `${c.toLocaleString()}${va ? ` · val ${va.area_hist[i].toLocaleString()}` : ""}`,
        tip: `<div class="tt-title">${lab(i)} px</div>${row(C.train, "train slices", c.toLocaleString())}${va ? row("#5a5a78", "val slices", va.area_hist[i].toLocaleString()) : ""}` })),
      labelWidth: 100, valueWidth: 110, rowH: 24,
    });
  }

  // ------------------------------------------------------------------ evaluation
  async function evalData(split) {
    if (!S.evalData[split]) S.evalData[split] = api(`${BASE}api/runs/${encodeURIComponent(S.sel)}/eval/${split}`).catch(() => null);
    return S.evalData[split];
  }

  function patientRows(data, conf) {
    const k = data.thresholds.indexOf(conf);
    return data.patients.map((p) => {
      const [inter, pred, gt] = p.sweep[k];
      const s = p.slices;
      let tp = 0; let fn = 0; let fp = 0;
      s.gt.forEach((g, i) => { if (g && s.pred[i]) tp++; else if (g) fn++; else if (s.pred[i]) fp++; });
      return { id: p.id, dice: dice(inter, pred, gt), gt_ml: gt / 1000, pred_ml: pred / 1000, slices: s.z.length,
               tumour_slices: tp + fn, tp, fn, fp, maxconf: Math.max(0, ...s.conf) };
    });
  }

  async function renderEval() {
    for (const b of $("splitChips").querySelectorAll(".chip")) b.classList.toggle("on", b.dataset.split === S.split);
    const r = S.run;
    const sum = r.eval && r.eval[S.split];
    const clear = (msg) => {
      $("evalKpis").innerHTML = `<p class="empty">${msg}</p>`;
      ["sweepChart", "diceHist", "diceScatter", "sizeBars", "confusion", "patTable", "sweepNote"].forEach((id) => { $(id).innerHTML = ""; });
      $("patCount").textContent = "";
      renderViewer(null);
    };
    if (!sum) return clear(LIVE.includes(r.state) ? "Evaluation runs after training." : "This run has no evaluation.");
    const data = await evalData(S.split);
    if (!data) return clear("Evaluation files not found.");
    const conf = sum.conf;
    const rows = patientRows(data, conf);
    const col = C[S.split];
    $("evalKpis").innerHTML = [
      kpi(`${S.split} 3D Dice`, sum.dice3d_mean.toFixed(4), `median ${sum.dice3d_median.toFixed(4)} · ${sum.patients} patients`, "hero"),
      kpi("Tumour-slice Dice", sum.tumour_slice_dice.toFixed(4), "slices that have tumour"),
      kpi("Legacy slice Dice", sum.legacy_slice_dice.toFixed(4), "empty-vs-empty slice = 1.0"),
      kpi("Sensitivity", pct(sum.sensitivity), "tumour slices found"),
      kpi("Specificity", pct(sum.specificity), "tumour-free slices left empty"),
      kpi("Threshold", conf, `best on ${S.split}: ${sum.best_conf}`),
    ].join("");

    // sweep: val + test from the summaries
    const series = ["val", "test"].filter((s) => r.eval[s]).map((s) => ({ name: s, color: C[s], dash: s === "val" ? "6 4" : null, dots: true,
      points: r.eval[s].sweep.map((p) => [p.conf, p.dice3d_mean]) }));
    const vb = r.eval.val && r.eval.val.best_conf;
    Charts.line($("sweepChart"), { series, height: 210, xName: "threshold", xLabel: "confidence threshold", yLabel: "mean 3D Dice",
      markers: [{ x: conf, label: "config" }, ...(vb != null && vb !== conf ? [{ x: vb, label: "val best", color: C.val }] : [])] });
    if (r.eval.val && r.eval.test) {
      const at = (s, c) => (r.eval[s].sweep.find((p) => p.conf === c) || {}).dice3d_mean;
      $("sweepNote").innerHTML = `val is highest at <b>${vb}</b>; test at that threshold: <b>${fmt(at("test", vb))}</b> vs <b>${fmt(at("test", conf))}</b> at the configured ${conf}. ` +
        `Pick the threshold on val, never on test.`;
    }

    Charts.histogram($("diceHist"), { values: rows.map((x) => x.dice), lo: 0, hi: 1, bins: 20, color: col, unit: "patients",
      xLabel: "3D Dice", markers: [{ x: sum.dice3d_mean, label: "mean" }],
      onClick: (lo, hi) => { S.patRange = [lo, hi]; renderPatTable(rows); $("patTable").scrollIntoView({ behavior: "smooth", block: "center" }); } });

    Charts.scatter($("diceScatter"), { points: rows.map((x) => ({ x: x.gt_ml, y: x.dice, color: col, id: x.id, big: x.id === S.vPatient,
      tip: `<div class="tt-title">${esc(x.id)}</div>${row(col, "3D Dice", fmt(x.dice))}${row("#9a9ab0", "tumour", `${fmt(x.gt_ml)} mL`)}${row("#9a9ab0", "predicted", `${fmt(x.pred_ml)} mL`)}` })),
      logX: true, yMin: 0, yMax: 1, xLabel: "tumour volume (mL, log)", yLabel: "3D Dice", onClick: (p) => openViewer(p.id) });

    const buckets = [[0, 10], [10, 30], [30, 60], [60, 100], [100, Infinity]];
    Charts.bars($("sizeBars"), { max: 1, labelWidth: 100, items: buckets.map(([a, b]) => {
      const inB = rows.filter((x) => x.gt_ml >= a && x.gt_ml < b);
      const m = inB.length ? inB.reduce((s, x) => s + x.dice, 0) / inB.length : 0;
      const label = b === Infinity ? `≥ ${a} mL` : `${a}–${b} mL`;
      return { label, value: m, color: col, text: inB.length ? `${m.toFixed(3)} · n=${inB.length}` : "n=0",
               tip: `<div class="tt-title">${label}</div>${inB.length} patients, mean 3D Dice ${inB.length ? m.toFixed(4) : "–"}` };
    }) });

    const d = sum.detection;
    $("confusion").innerHTML = `<table class="confusion"><thead><tr><th></th><th>predicted tumour</th><th>predicted none</th></tr></thead><tbody>
      <tr><th>has tumour</th><td class="ok">${d.tp.toLocaleString()}<em>found</em></td><td class="bad">${d.fn.toLocaleString()}<em>missed</em></td></tr>
      <tr><th>no tumour</th><td class="bad">${d.fp.toLocaleString()}<em>false alarm</em></td><td class="ok">${d.tn.toLocaleString()}<em>correctly empty</em></td></tr></tbody></table>`;

    S.evalRows = rows;
    renderPatTable(rows);
    renderViewer(data);
  }

  const PCOLS = [["id", "Patient", true], ["dice", "3D Dice"], ["gt_ml", "Tumour mL"], ["pred_ml", "Predicted mL"],
                 ["tumour_slices", "Tumour slices"], ["tp", "Found"], ["fn", "Missed"], ["fp", "False"], ["maxconf", "Top conf"]];
  function renderPatTable(rows) {
    const q = S.patQ.toLowerCase();
    let list = rows.filter((x) => !q || x.id.toLowerCase().includes(q));
    if (S.patRange) list = list.filter((x) => x.dice >= S.patRange[0] && x.dice < S.patRange[1] + (S.patRange[1] >= 1 ? 1e-9 : 0));
    const { k, dir } = S.patSort;
    const isStr = k === "id";
    list.sort((a, b) => (isStr ? a[k].localeCompare(b[k]) : a[k] - b[k]) * dir);
    $("patRange").classList.toggle("hidden", !S.patRange);
    if (S.patRange) $("patRange").innerHTML = `3D Dice ${S.patRange[0].toFixed(2)}–${S.patRange[1].toFixed(2)} <button type="button" aria-label="Clear range" id="clearRange">&times;</button>`;
    $("patCount").textContent = `${list.length} of ${rows.length}`;
    $("patTable").innerHTML = `<table class="tbl"><thead><tr>${PCOLS.map(([c, l, s]) =>
      `<th data-k="${c}" class="${s ? "" : "num"}${k === c ? " sorted" : ""}" tabindex="0">${l}${k === c ? (dir > 0 ? " ▲" : " ▼") : ""}</th>`).join("")}</tr></thead><tbody>` +
      list.map((x) => `<tr data-id="${esc(x.id)}" class="${x.id === S.vPatient ? "sel" : ""}" tabindex="0"><td class="mono">${esc(x.id)}</td>
        <td class="num"><span class="dbar" style="--v:${x.dice}"></span>${x.dice.toFixed(3)}</td><td class="num">${fmt(x.gt_ml)}</td><td class="num">${fmt(x.pred_ml)}</td>
        <td class="num">${x.tumour_slices}</td><td class="num">${x.tp}</td><td class="num ${x.fn ? "warn" : ""}">${x.fn}</td><td class="num ${x.fp ? "warn" : ""}">${x.fp}</td><td class="num">${x.maxconf.toFixed(2)}</td></tr>`).join("") +
      "</tbody></table>";
    const cr = $("clearRange");
    if (cr) cr.addEventListener("click", () => { S.patRange = null; renderPatTable(rows); });
  }
  $("patTable").addEventListener("click", (e) => {
    const th = e.target.closest("th[data-k]");
    if (th) {
      const k = th.dataset.k;
      S.patSort = S.patSort.k === k ? { k, dir: -S.patSort.dir } : { k, dir: k === "id" ? 1 : k === "dice" ? 1 : -1 };
      renderPatTable(S.evalRows);
      return;
    }
    const tr = e.target.closest("tr[data-id]");
    if (tr) openViewer(tr.dataset.id);
  });
  $("patTable").addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    const th = e.target.closest("th[data-k]");
    const tr = e.target.closest("tr[data-id]");
    if (th) th.click(); else if (tr) openViewer(tr.dataset.id);
  });
  let patT = null;
  $("patSearch").addEventListener("input", (e) => {
    clearTimeout(patT);
    patT = setTimeout(() => { S.patQ = e.target.value.trim(); if (S.evalRows) renderPatTable(S.evalRows); }, 120);
  });
  $("splitChips").addEventListener("click", (e) => {
    const b = e.target.closest("[data-split]");
    if (!b || b.dataset.split === S.split) return;
    S.split = b.dataset.split;
    S.vPatient = null;
    S.vZ = null;
    S.patRange = null;
    syncHash();
    renderEval();
  });
  $("dlEval").addEventListener("click", () => {
    if (!S.evalRows) return;
    const head = PCOLS.map((c) => c[0]);
    const csv = [head.join(","), ...S.evalRows.map((x) => head.map((k) => (typeof x[k] === "number" ? +x[k].toFixed(6) : x[k])).join(","))].join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = `${S.sel}_${S.split}_patients.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  // ------------------------------------------------------------------ prediction viewer
  let viewerData = null;
  function renderViewer(data) {
    viewerData = data;
    const sel = $("vPatient");
    const on = !!data && S.run && S.run.has_best;
    for (const id of ["vPatient", "vZ", "vConf", "vPeak"]) $(id).disabled = !on;
    if (!on) {
      sel.innerHTML = "";
      $("vImg").removeAttribute("src");
      $("vStatus").textContent = "Available once the run has been evaluated.";
      $("vStats").innerHTML = "";
      $("vProfile").innerHTML = "";
      return;
    }
    const ids = data.patients.map((p) => p.id).sort();
    sel.innerHTML = ids.map((id) => `<option>${esc(id)}</option>`).join("");
    if (!S.vPatient || !ids.includes(S.vPatient)) {
      // default: the worst patient at the configured threshold — the most instructive one
      S.vPatient = S.evalRows ? [...S.evalRows].sort((a, b) => a.dice - b.dice)[0].id : ids[0];
      S.vZ = null;
    }
    if (S.vConf == null) S.vConf = S.run.eval[S.split].conf;
    sel.value = S.vPatient;
    $("vConf").value = S.vConf;
    showSlice();
  }

  function currentPatient() {
    return viewerData && viewerData.patients.find((p) => p.id === S.vPatient);
  }

  let predT = null;
  function showSlice() {
    const p = currentPatient();
    if (!p) return;
    const zs = p.slices.z;
    if (S.vZ == null || !zs.includes(S.vZ)) {
      let bi = 0;
      p.slices.gt.forEach((g, i) => { if (g > p.slices.gt[bi]) bi = i; });
      S.vZ = zs[bi];
    }
    const i = zs.indexOf(S.vZ);
    $("vZ").min = 0;
    $("vZ").max = zs.length - 1;
    $("vZ").value = i;
    $("vZl").textContent = `z = ${S.vZ}`;
    $("vConfL").textContent = S.vConf.toFixed(2);
    syncHash();
    drawProfile(p);
    for (const tr of $("patTable").querySelectorAll("tr[data-id]")) tr.classList.toggle("sel", tr.dataset.id === S.vPatient);
    clearTimeout(predT);
    predT = setTimeout(async () => {
      const q = `split=${S.split}&patient=${encodeURIComponent(S.vPatient)}&z=${S.vZ}&conf=${S.vConf}`;
      $("vStatus").textContent = "Predicting…";
      const img = $("vImg");
      img.onload = () => { $("vStatus").textContent = ""; };
      img.onerror = () => { $("vStatus").textContent = "Could not render this slice."; };
      img.src = `${BASE}api/runs/${encodeURIComponent(S.sel)}/predict.png?${q}`;
      img.alt = `${S.vPatient} slice ${S.vZ}: model input and prediction vs expert mask`;
      try {
        const st = await api(`${BASE}api/runs/${encodeURIComponent(S.sel)}/predict.json?${q}`);
        $("vStats").innerHTML = [
          kpi("Slice Dice", st.gt || st.pred ? st.dice.toFixed(3) : "–", st.gt || st.pred ? "" : "no tumour, none predicted"),
          kpi("Expert", `${st.gt.toLocaleString()} px`),
          kpi("Predicted", `${st.pred.toLocaleString()} px`, `${st.inter.toLocaleString()} px overlap`),
          kpi("Instances", st.confs.length, st.confs.length ? `conf ${st.confs.join(", ")}` : `none ≥ ${S.vConf}`),
        ].join("");
      } catch (err) {
        $("vStats").innerHTML = `<p class="err">${esc(err.message)}</p>`;
      }
    }, 120);
  }

  function drawProfile(p) {
    const s = p.slices;
    const pts = s.z.map((z, i) => [z, s.gt[i] || s.pred[i] ? dice(s.inter[i], s.pred[i], s.gt[i]) : null]);
    const area = s.z.map((z, i) => [z, s.gt[i] / Math.max(1, Math.max(...s.gt))]);
    Charts.line($("vProfile"), { height: 170, xName: "z", xLabel: "axial slice", yMin: 0, yMax: 1,
      series: [{ name: "slice Dice", color: C[S.split], points: pts }, { name: "expert tumour (scaled)", color: "#5a5a78", dash: "4 3", points: area }],
      markers: [{ x: S.vZ, label: `z ${S.vZ}` }], onClick: (z) => { S.vZ = z; showSlice(); } });
  }

  function openViewer(id) {
    S.vPatient = id;
    S.vZ = null;
    $("vPatient").value = id;
    showSlice();
    renderEval();
    $("secViewer").scrollIntoView({ behavior: "smooth" });
  }
  $("vPatient").addEventListener("change", (e) => { S.vPatient = e.target.value; S.vZ = null; showSlice(); });
  $("vZ").addEventListener("input", (e) => { const p = currentPatient(); if (p) { S.vZ = p.slices.z[+e.target.value]; showSlice(); } });
  $("vConf").addEventListener("input", (e) => { S.vConf = +(+e.target.value).toFixed(2); showSlice(); });
  $("vPeak").addEventListener("click", () => { S.vZ = null; showSlice(); });

  // ------------------------------------------------------------------ plots
  function renderPlots() {
    const r = S.run;
    const order = (f) => (f === "results.png" ? 0 : /^Mask/.test(f) ? 1 : /confusion/.test(f) ? 2 : /^Box/.test(f) ? 3 : /^val_/.test(f) ? 4 : 5);
    const files = [...r.plots].sort((a, b) => order(a) - order(b) || a.localeCompare(b));
    $("plots").innerHTML = files.length
      ? files.map((f) => `<figure><button type="button" data-f="${esc(f)}"><img loading="lazy" src="${BASE}api/runs/${encodeURIComponent(r.id)}/file/${encodeURIComponent(f)}" alt="${esc(f)}"></button><figcaption>${esc(f)}</figcaption></figure>`).join("")
      : `<p class="empty">Ultralytics saves its plots at the end of training.</p>`;
  }
  $("plots").addEventListener("click", (e) => {
    const b = e.target.closest("[data-f]");
    if (!b) return;
    $("lightImg").src = `${BASE}api/runs/${encodeURIComponent(S.sel)}/file/${encodeURIComponent(b.dataset.f)}`;
    $("lightImg").alt = b.dataset.f;
    $("lightbox").showModal();
  });

  // ------------------------------------------------------------------ live run
  function renderLive(run) {
    const st = run ? run.status : null;
    const live = st && LIVE.includes(run.state);
    $("stopBtn").disabled = !live;
    $("liveId").textContent = run ? run.id : "";
    if (!run) {
      $("liveSub").textContent = "No run is going.";
      $("stepper").querySelectorAll("li").forEach((li) => { li.className = ""; li.querySelector("i").style.width = "0"; li.querySelector("em").textContent = ""; });
      $("liveKpis").innerHTML = "";
      return;
    }
    const stages = ["dataset", "train", "evaluate", "done"];
    const cur = run.state === "done" ? 3 : Math.max(0, stages.indexOf(st.stage));
    const prog = {
      dataset: st.dataset_total ? st.dataset_done / st.dataset_total : 0,
      train: st.epochs ? ((st.epoch || 1) - 1 + (st.batches ? (st.batch || 0) / st.batches : 0)) / st.epochs : 0,
      evaluate: st.eval_total ? st.eval_done / st.eval_total : 0,
      done: run.state === "done" ? 1 : 0,
    };
    const text = {
      dataset: st.dataset_total ? `${st.dataset_done} / ${st.dataset_total} patients` : "",
      train: st.epochs ? `epoch ${st.epoch} / ${st.epochs}${st.batches && cur === 1 ? ` · batch ${st.batch} / ${st.batches}` : ""}` : "",
      evaluate: st.eval_total ? `${st.eval_done} / ${st.eval_total} patients${st.eval_split ? ` (${st.eval_split})` : ""}` : "",
      done: run.state === "done" ? (st.finished || "").replace("T", " ") : "",
    };
    $("stepper").querySelectorAll("li").forEach((li, i) => {
      const k = li.dataset.stage;
      const failed = !live && run.state !== "done" && i === cur;
      li.className = i < cur || run.state === "done" ? "done" : i === cur ? (failed ? "bad" : "cur") : "";
      li.querySelector("i").style.width = `${(i < cur ? 1 : i === cur ? prog[k] : 0) * 100}%`;
      li.querySelector("em").textContent = text[k];
    });
    const res = run.results || {};
    const times = res.time || [];
    const perEpoch = times.length ? times[times.length - 1] / times.length : null;
    const left = st.epochs && perEpoch && cur === 1 ? (st.epochs - (st.epoch || 1) + 1 - (st.batches ? (st.batch || 0) / st.batches : 0)) * perEpoch : null;
    const be = bestEpoch(res);
    const sinceBest = be != null && res.epoch ? res.epoch[res.epoch.length - 1] - be : null;
    const patience = run.config.train.patience;
    $("liveSub").innerHTML = `${badge(run.state)} ${esc(run.model)}${run.smoke ? " · smoke test" : ""} · started ${esc((st.started || "").replace("T", " "))}` +
      (run.error ? ` · <span class="err">${esc(run.error)}</span>` : "");
    $("liveKpis").innerHTML = [
      kpi("Elapsed", live ? dur(since(st.started)) : dur(since(st.started) - since(st.finished || st.updated))),
      kpi("Time per epoch", perEpoch ? dur(perEpoch) : "–", times.length ? `${times.length} done` : ""),
      kpi("Training left", left != null ? `≤ ${dur(left)}` : "–", left != null ? "estimate; early stop can end it sooner" : ""),
      kpi("Best mAP50-95 (M)", be != null ? fmt(Math.max(...res["metrics/mAP50-95(M)"].filter((v) => v != null))) : "–", be != null ? `epoch ${be}` : ""),
      kpi("Since best", sinceBest != null ? `${sinceBest} epoch${sinceBest === 1 ? "" : "s"}` : "–", sinceBest != null ? `stops at ${patience} (patience)` : "",
          sinceBest != null && patience && sinceBest >= patience * 0.7 ? "warnk" : ""),
    ].join("");
  }

  $("stopBtn").addEventListener("click", async () => {
    const id = S.active;
    if (!id || !confirm(`Stop run ${id}? Training can be resumed later from its last.pt.`)) return;
    try {
      await api(`${BASE}api/runs/${encodeURIComponent(id)}/stop`, { method: "POST" });
    } catch (err) {
      alert(err.message);
    }
    await loadOverview();
    poll();
  });

  // log
  const BAR_RE = /━|it\/s|s\/it|\d+%\s*[━─╸]/;
  async function pollLog(id) {
    if (S.logRun !== id) {
      S.logRun = id;
      S.logOffset = 0;
      S.logLines = [];
    }
    const r = await api(`${BASE}api/runs/${encodeURIComponent(id)}/log?offset=${S.logOffset}`);
    if (r.offset < S.logOffset) S.logLines = [];
    S.logOffset = r.offset;
    if (r.text) {
      // Progress bars rewrite their line with \r: keep only the final state of each line.
      for (const raw of r.text.replace(/\x1b\[[0-9;]*[A-Za-z]/g, "").split("\n")) {
        const parts = raw.split("\r").filter((x) => x.trim());
        if (parts.length) S.logLines.push(parts[parts.length - 1]);
      }
      if (S.logLines.length > 4000) S.logLines = S.logLines.slice(-4000);
    }
    renderLog();
  }
  function renderLog() {
    const f = $("logFilter").value.trim().toLowerCase();
    const bars = $("logBars").checked;
    const lines = S.logLines.filter((l) => (bars || !BAR_RE.test(l)) && (!f || l.toLowerCase().includes(f)));
    const box = $("log");
    box.textContent = lines.slice(-1500).join("\n") || (S.logRun ? "(no output yet)" : "");
    if ($("logFollow").checked) box.scrollTop = box.scrollHeight;
  }
  $("logFilter").addEventListener("input", renderLog);
  $("logBars").addEventListener("change", renderLog);

  // gpu
  async function pollGpu() {
    try {
      const g = await api(BASE + "api/gpu");
      $("gpu").innerHTML = g.available
        ? `<b>${esc(g.name)}</b><span>GPU ${fmt(g.util)}%</span><span>${fmt(g.mem_used / 1024)} / ${fmt(g.mem_total / 1024)} GB</span><span>${fmt(g.temp)} °C</span>${g.power != null ? `<span>${fmt(g.power)} W</span>` : ""}` +
          `<span class="gbar" title="GPU utilisation"><i style="width:${g.util || 0}%"></i></span>`
        : "GPU readout unavailable";
    } catch { /* keep the last reading */ }
  }

  // ------------------------------------------------------------------ polling loop
  let pollT = null;
  async function poll() {
    clearTimeout(pollT);
    try {
      await loadOverview();
      const liveId = S.active || (S.sel && S.runs.find((r) => r.id === S.sel) ? S.sel : null);
      if (liveId) {
        const run = await loadRun(liveId);
        renderLive(run);
        await pollLog(liveId);
        if (liveId === S.sel) {
          const prev = S.run;
          S.run = run;
          // redraw the analysis when something new landed (an epoch, a stage, a state)
          if (!prev || prev.epochs_done !== run.epochs_done || prev.state !== run.state || !!prev.eval !== !!run.eval) {
            if (!!prev && !!prev.eval !== !!run.eval) S.evalData = {};
            renderAnalysis();
          }
        }
      } else {
        renderLive(null);
      }
    } catch (err) {
      console.error(err);
    }
    pollT = setTimeout(poll, S.active ? 2000 : 6000);
  }

  // ------------------------------------------------------------------ keys, links
  function flash(btn, msg) {
    const old = btn.textContent;
    btn.textContent = msg;
    setTimeout(() => { btn.textContent = old; }, 1300);
  }
  $("copyLink").addEventListener("click", () => navigator.clipboard.writeText(location.href).then(() => flash($("copyLink"), "Link copied")));

  document.addEventListener("keydown", (e) => {
    const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement.tagName) && document.activeElement.type !== "range" && document.activeElement.type !== "checkbox";
    if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "/") {
      e.preventDefault();
      $("patSearch").focus();
    } else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      const p = currentPatient();
      if (!p) return;
      const i = p.slices.z.indexOf(S.vZ) + (e.key === "ArrowRight" ? 1 : -1);
      if (i >= 0 && i < p.slices.z.length) { e.preventDefault(); S.vZ = p.slices.z[i]; showSlice(); }
    } else if (e.key === "[" || e.key === "]") {
      const opts = [...$("vPatient").options].map((o) => o.value);
      const i = opts.indexOf(S.vPatient) + (e.key === "]" ? 1 : -1);
      if (i >= 0 && i < opts.length) { S.vPatient = opts[i]; S.vZ = null; $("vPatient").value = opts[i]; showSlice(); }
    }
  });

  let resizeT = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeT);
    resizeT = setTimeout(renderAnalysis, 200);
  });

  fromHash();
  if (S.sel) selectRun(S.sel, true).catch(() => { S.sel = null; });
  $("analysis").classList.add("hidden");
  setInterval(pollGpu, 3000);
  pollGpu();
  poll();
})();
