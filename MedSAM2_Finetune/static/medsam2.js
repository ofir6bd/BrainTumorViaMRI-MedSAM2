/* MedSAM2_Finetune page.
   Charts come from the YOLO_finetune chart kit (window.Charts) — one copy, one look.
   Every number on this page is measured; nothing here invents or smooths data. */
(() => {
  "use strict";
  const BASE = document.body.dataset.base.replace(/\/$/, "");
  const $ = (id) => document.getElementById(id);
  const api = (path) => fetch(BASE + path).then(async (r) => {
    const body = await r.json().catch(() => ({ error: r.statusText }));
    if (!r.ok) throw new Error(body.error || r.statusText);
    return body;
  });
  const post = (path, body) => fetch(BASE + path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  }).then(async (r) => {
    const out = await r.json().catch(() => ({ error: r.statusText }));
    if (!r.ok) throw new Error(out.error || r.statusText);
    return out;
  });
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
  const f4 = (v) => (v == null ? "—" : Number(v).toFixed(4));
  const f2 = (v) => (v == null ? "—" : Number(v).toFixed(2));
  const pct = (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%");
  const secs = (s) => (s == null ? "—" : s < 90 ? `${Math.round(s)} s`
    : s < 5400 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${(s / 3600).toFixed(1)} h`);
  const COL = { medsam2: "#3987e5", yolo: "#e0803c", gt: "#8fe08f", gain: "#2fa84f", loss: "#d03b3b",
                muted: "#7f7f9c", val: "#9085e9", anchor: "#c08bff" };

  const S = {           // everything the page knows, in one place
    overview: null, runId: null, run: null, compare: new Set(), split: "test",
    patients: [], patient: null, slice: null, logOffset: 0, profile: null,
  };

  /* ------------------------------------------------------------------ hash state */
  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    S.runId = p.get("run") || S.runId;
    S.split = p.get("pool") || S.split;
    S.patient = p.get("patient") || S.patient;
  }
  function writeHash() {
    const p = new URLSearchParams();
    if (S.runId) p.set("run", S.runId);
    if (S.split) p.set("pool", S.split);
    if (S.patient) p.set("patient", S.patient);
    history.replaceState(null, "", "#" + p.toString());
  }

  /* ------------------------------------------------------------------ the start form */
  const FIELDS = {
    "prompt.yolo_run": "fYoloRun", "prompt.variant": "fVariant", "prompt.yolo_conf": "fYoloConf",
    "prompt.score_min": "fScoreMin", "prompt.logit_scale": "fLogitScale",
    "prompt.empty_logit": "fEmptyLogit",
    "anchors.count": "fAnchorCount", "anchors.min_gap": "fAnchorGap",
    "hitl.rounds": "fHitlRounds",
    "model.checkpoint": "fCheckpoint",
    "data.max_train_patients": "fMaxTrain", "data.val_patients": "fValPatients",
    "data.clips_per_patient": "fClips", "data.tumour_clip_fraction": "fTumourClips",
    "data.min_fg_voxels": "fMinFg",
    "video.num_frames": "fNumFrames", "video.reverse_fraction": "fReverse",
    "video.batch": "fVideoBatch",
    "train.unfreeze": "fUnfreeze", "train.epochs": "fEpochs", "train.patience": "fPatience",
    "train.accum": "fAccum", "train.lr": "fLr", "train.memory_lr": "fMemoryLr",
    "train.vision_lr": "fVisionLr", "train.dice_weight": "fDiceW", "train.bce_weight": "fBceW",
    "train.fliplr": "fFliplr", "train.workers": "fWorkers",
    "evaluate.mask_threshold": "fThreshold",
  };
  const cfgValue = (cfg, dotted) => dotted.split(".").reduce((n, k) => (n == null ? n : n[k]), cfg);

  function fillForm(ov) {
    const runs = ov.yolo_runs.map((r) => ({
      value: r.id,
      label: `${r.id} — ${r.test_dice3d != null ? "test 3D Dice " + f4(r.test_dice3d) : "not scored"}`,
    }));
    runs.unshift({ value: "", label: "best scoring run (automatic)" });
    fillSelect($("fYoloRun"), runs);
    fillSelect($("fVariant"), Object.keys(ov.variants).map((k) => ({ value: k, label: k })));
    fillSelect($("fCheckpoint"), ov.checkpoints.map((k) => ({ value: k, label: k })));
    fillSelect($("fUnfreeze"), Object.keys(ov.unfreeze).map((k) => ({ value: k, label: k })));
    resetForm();
    $("fVariant").addEventListener("change", showVariantNote);
    Object.values(FIELDS).forEach((id) => $(id).addEventListener("change", showEstimate));
    showVariantNote();
  }
  function fillSelect(sel, items) {
    sel.innerHTML = items.map((i) => `<option value="${esc(i.value)}">${esc(i.label)}</option>`).join("");
  }
  function resetForm() {
    const cfg = S.overview.config;
    for (const [key, id] of Object.entries(FIELDS)) {
      const v = cfgValue(cfg, key);
      $(id).value = v == null ? "" : v;
    }
    showVariantNote();
    showEstimate();
  }
  function showVariantNote() {
    const ov = S.overview;
    const v = $("fVariant").value;
    $("variantNote").innerHTML = `<b>${esc(v)}</b> — ${esc(ov.variants[v] || "")}`;
  }
  function showEstimate() {
    const ov = S.overview;
    const changed = Object.entries(FIELDS).filter(([key, id]) => {
      const now = $(id).value;
      const was = cfgValue(ov.config, key);
      return String(was == null ? "" : was) !== String(now);
    }).map(([key]) => key);
    const rebuild = changed.filter((k) => ov.rebuilds.includes(k));
    const n = Number($("fMaxTrain").value || ov.pools.train || 0);
    const clips = n * Number($("fClips").value || 1);
    const checks = Number($("fValPatients").value || ov.pools.val || 0);
    const parts = [`About <b>${n}</b> patients per round = <b>${clips}</b> clips of
      ${esc($("fNumFrames").value)} slices. The check after each round propagates
      <b>${checks}</b> whole patients, so it is the slow part.`];
    if (rebuild.length) {
      parts.push(`<b>The hints will be worked out again</b> (you changed ${rebuild.map(esc).join(", ")}) —
        that is a YOLO pass over every slice of every patient used, which takes a while the first time.`);
    }
    $("startEstimate").innerHTML = parts.join(" ");
  }
  function overrides() {
    const out = {};
    for (const [key, id] of Object.entries(FIELDS)) {
      const el = $(id);
      const was = cfgValue(S.overview.config, key);
      const now = el.value;
      if (String(was == null ? "" : was) !== String(now)) out[key] = now;
    }
    return out;
  }

  /* ------------------------------------------------------------------ runs table */
  const RUN_COLS = [
    ["compare", "", (r) => `<input type="checkbox" class="cmp" data-id="${esc(r.id)}" ${S.compare.has(r.id) ? "checked" : ""}>`],
    ["id", "Run", (r) => `<code>${esc(r.id)}</code>${r.smoke ? ' <span class="badge smoke">quick</span>' : ""}`],
    ["state", "Status", (r) => `<span class="badge ${stateClass(r.state)}">${esc(stateWord(r.state))}</span>`],
    ["variant", "Prompt style", (r) => esc(r.variant)],
    ["yolo_run", "From YOLO", (r) => `<code>${esc(r.yolo_run)}</code>`],
    ["unfreeze", "Weights changed", (r) => esc(r.unfreeze)],
    ["hitl", "HITL rounds", (r) => `${r.rounds || "—"}`],
    ["rounds", "Training rounds", (r) => `${r.epochs_done}/${r.epochs_cfg}`, true],
    ["best", "Best check score", (r) => (r.best ? `${f4(r.best.val_dice3d)} <span class="dim">@${r.best.epoch}</span>` : "—"), true],
    ["test", "Test 3D Dice", (r) => (r.test ? `<b>${f4(r.test.dice3d_mean)}</b>` : "—"), true],
    ["yolo", "YOLO alone", (r) => (r.test ? f4(r.test.yolo_dice3d_mean) : "—"), true],
    ["delta", "Change", (r) => (r.test ? deltaCell(r.test.delta_mean) : "—"), true],
    ["time", "Training time", (r) => secs(r.train_seconds), true],
    ["created", "Started", (r) => esc((r.created || "").replace("T", " ")), true],
  ];
  const stateWord = (s) => ({ done: "finished", running: "running", queued: "starting",
    failed: "failed", stopped: "stopped", interrupted: "cut off" }[s] || s || "—");
  const stateClass = (s) => ({ done: "st-done", running: "st-running", failed: "st-failed",
    stopped: "st-stopped", interrupted: "st-failed" }[s] || "");
  const deltaCell = (d) => d == null ? "—"
    : `<span style="color:${d >= 0 ? COL.gain : COL.loss}">${d >= 0 ? "+" : ""}${f4(d)}</span>`;

  function renderRuns() {
    const tbl = $("runsTbl");
    tbl.querySelector("thead").innerHTML = "<tr>" + RUN_COLS
      .map(([, label, , num]) => `<th class="${num ? "num" : ""}">${esc(label)}</th>`).join("") + "</tr>";
    tbl.querySelector("tbody").innerHTML = S.overview.runs.map((r) => `<tr data-id="${esc(r.id)}"
      class="${r.id === S.runId ? "sel" : ""}">` + RUN_COLS
      .map(([, , cell, num]) => `<td class="${num ? "num" : ""}">${cell(r)}</td>`).join("") + "</tr>").join("")
      || `<tr><td colspan="${RUN_COLS.length}" class="dim">No runs yet — start one above.</td></tr>`;
    tbl.querySelectorAll("tbody tr").forEach((tr) => tr.addEventListener("click", (e) => {
      if (e.target.classList.contains("cmp")) return;
      if (tr.dataset.id) selectRun(tr.dataset.id);
    }));
    tbl.querySelectorAll(".cmp").forEach((box) => box.addEventListener("change", () => {
      if (box.checked && S.compare.size >= 3) { box.checked = false; return; }
      box.checked ? S.compare.add(box.dataset.id) : S.compare.delete(box.dataset.id);
      drawCurves();
    }));
  }

  /* ------------------------------------------------------------------ live run */
  const STAGES = [["prompts", "1. Work out the hints"], ["train", "2. Train"],
                  ["evaluate", "3. Final check"], ["done", "4. Done"]];

  function renderLive(run) {
    const st = (run && run.status) || {};
    const live = ["queued", "running"].includes(st.state);
    $("liveId").textContent = run ? run.id : "";
    $("liveSub").innerHTML = run
      ? `<span class="badge ${stateClass(st.state)}">${esc(stateWord(st.state))}</span>
         ${esc(run.checkpoint)} · prompt <b>${esc(run.variant)}</b> from YOLO <code>${esc(run.yolo_run)}</code>
         · started ${esc((st.started || run.created || "").replace("T", " "))}
         ${st.error ? `<span class="err">${esc(st.error)}</span>` : ""}`
      : "No run yet.";
    const at = STAGES.findIndex(([k]) => k === (st.stage || "prompts"));
    const finished = st.state === "done";
    $("stepper").innerHTML = STAGES.map(([key, label], i) => {
      // A finished run has every stage behind it, including "Done" itself — without this the
      // last bar sits at the half-way default, because no branch below fills it in.
      let pctDone = finished ? 100 : i < at ? 100 : i > at ? 0 : 50;
      let sub = "";
      if (key === "prompts" && st.prompt_total) {
        pctDone = i === at ? (100 * st.prompt_done) / st.prompt_total : pctDone;
        sub = `${st.prompt_done} of ${st.prompt_total} patients`;
      } else if (key === "train" && st.epochs) {
        pctDone = i === at ? (100 * (st.epoch - 1 + (st.batch || 0) / (st.batches || 1))) / st.epochs : pctDone;
        sub = `round ${st.epoch} of ${st.epochs}${st.batches ? ` · step ${st.batch} of ${st.batches}` : ""}`;
      } else if (key === "evaluate" && st.eval_total) {
        pctDone = i === at ? (100 * st.eval_done) / st.eval_total : pctDone;
        sub = `${st.eval_done} of ${st.eval_total} patients (${esc(st.eval_split || "")})`;
      }
      const bad = st.state === "failed" || st.state === "interrupted";
      const cls = bad && i === at ? "bad" : finished || i < at ? "done" : i === at ? "cur" : "";
      return `<li class="${cls}"><b>${esc(label)}</b>
        <div class="step-bar"><i style="width:${Math.max(0, Math.min(100, pctDone))}%"></i></div>
        <em>${esc(sub)}</em></li>`;
    }).join("");

    const k = [];
    if (st.started) {
      const end = st.finished ? new Date(st.finished) : new Date();
      k.push(["Time so far", secs((end - new Date(st.started)) / 1000)]);
    }
    if (st.epoch_time) k.push(["Time per round", secs(st.epoch_time), `${st.epoch || 0} rounds done`]);
    if (st.epoch_time && st.epochs && st.epoch) {
      k.push(["Training time left", secs(st.epoch_time * (st.epochs - st.epoch)), "a guess; it can stop earlier"]);
    }
    if (st.loss != null) k.push(["Loss now", f4(st.loss), "lower is better"]);
    if (st.best_val_dice3d != null) k.push(["Best check score", f4(st.best_val_dice3d), "3D Dice on the check pool"]);
    if (st.since_best != null) k.push(["Rounds since the best", st.since_best, `stops at ${run.config ? run.config.train.patience : "—"}`]);
    if (st.train_clips) k.push(["Clips per round", st.train_clips.toLocaleString(), `${st.train_patients} patients`]);
    $("liveKpis").innerHTML = k.map(([l, v, s]) => kpi(l, v, s)).join("");
    $("btnStop").disabled = !live;
    $("btnResume").disabled = !run || live || !run.has_last;
    $("btnRescore").disabled = !run || live || !run.has_best;
  }
  const kpi = (label, value, sub, cls) => `<div class="kpi ${cls || ""}"><span class="kpi-l">${esc(label)}</span>
    <span class="kpi-v">${value}</span>${sub ? `<span class="kpi-s">${esc(sub)}</span>` : ""}</div>`;

  async function pullLog(force) {
    if (!S.runId) return;
    const r = await api(`/api/runs/${S.runId}/log?offset=${force ? 0 : S.logOffset}`).catch(() => null);
    if (!r) return;
    const pre = $("log");
    if (force) pre.textContent = "";
    S.logOffset = r.offset;
    if (r.text) {
      const filter = $("logFilter").value.trim();
      const text = filter ? r.text.split("\n").filter((l) => l.includes(filter)).join("\n") : r.text;
      pre.textContent = (force ? "" : pre.textContent) + text;
      if ($("logFollow").checked) pre.scrollTop = pre.scrollHeight;
    }
    if (!pre.textContent) pre.textContent = "(nothing yet)";
  }

  /* ------------------------------------------------------------------ training curves */
  function drawCurves() {
    const ids = S.compare.size ? [...S.compare] : (S.runId ? [S.runId] : []);
    Promise.all(ids.map((id) => api(`/api/runs/${id}`).catch(() => null))).then((runs) => {
      const got = runs.filter((r) => r && r.results && r.results.epoch);
      const box = $("curves");
      box.innerHTML = "";
      if (!got.length) { box.innerHTML = '<p class="dim">No rounds finished yet.</p>'; return; }
      const palette = [COL.medsam2, COL.yolo, COL.val];
      const chart = (title, cols, yLabel, extra) => {
        const div = document.createElement("div");
        div.innerHTML = `<h3>${esc(title)}</h3>`;
        box.appendChild(div);
        const target = document.createElement("div");
        div.appendChild(target);
        const series = [];
        got.forEach((r, i) => cols.forEach(([key, name], j) => {
          if (!r.results[key]) return;
          series.push({
            name: got.length > 1 ? `${r.id.slice(-6)} ${name}` : name,
            color: cols.length > 1 ? [COL.medsam2, COL.yolo, COL.gt, COL.val][j] : palette[i % 3],
            dash: got.length > 1 && i ? (i === 1 ? "4 3" : "1 3") : null,
            points: r.results.epoch.map((e, n) => [e, r.results[key][n]]),
          });
        }));
        Charts.line(target, { series, height: 230, xLabel: "Round (epoch)", yLabel, ...(extra || {}) });
      };
      chart("Check score after each round", [["val_dice3d", "3D Dice on the check pool"]], "3D Dice (higher is better)");
      chart("Total loss", [["loss", "loss"]], "Loss (lower is better)");
      chart("What the loss is made of", [["dice", "overlap (Dice)"], ["bce", "per pixel (BCE)"],
        ["iou", "quality head"], ["obj", "is there tumour"]], "Loss part");
      chart("Learning rate", [["lr", "learning rate"]], "Step size");
      chart("Seconds per round", [["time", "seconds"]], "Seconds");
    });
  }

  /* ------------------------------------------------------------------ results */
  function renderEval(run) {
    $("evalRunId").textContent = run ? run.id : "";
    const ev = run && run.eval;
    const test = ev && ev.test;
    const kp = $("evalKpis");
    if (!test) {
      kp.innerHTML = '<p class="dim">This run has not been scored yet. Press “Score again” once it has a best round.</p>';
      $("chSweep").innerHTML = "";
      $("confusion").innerHTML = "";
      return;
    }
    kp.innerHTML = [
      kpi("3D Dice — MedSAM2", `<b>${f4(test.dice3d_mean)}</b>`, `${test.patients} test patients · median ${f4(test.dice3d_median)}`, "hero"),
      kpi("3D Dice — YOLO alone", f4(test.yolo_dice3d_mean), `median ${f4(test.yolo_dice3d_median)}`),
      kpi("Change", deltaCell(test.delta_mean), `median ${test.delta_median >= 0 ? "+" : ""}${f4(test.delta_median)}`,
        test.delta_mean >= 0 ? "" : "warnk"),
      kpi("Patients helped", `${test.helped}`, `${test.hurt} hurt · ${test.unchanged} unchanged`),
      kpi("Tumour-slice Dice", f4(test.tumour_slice_dice), "slices that really have tumour"),
      kpi("Old-style slice Dice", f4(test.legacy_slice_dice), `YOLO alone: ${f4(test.yolo_legacy_slice_dice)}`),
      kpi("Found a tumour slice", pct(test.sensitivity), "of slices with tumour"),
      kpi("Left empty slices empty", pct(test.specificity), "of slices without tumour"),
      kpi("Cut-off used", test.threshold, `best here: ${test.best_threshold}${ev.val ? ` · best on check: ${ev.val.best_threshold}` : ""}`),
      kpi("Anchors per patient", f2(test.anchors_mean), `${f2(test.rounds_mean)} correction rounds`),
      kpi("Best round count", test.best_round ?? "—",
          ev.val && ev.val.best_round ? `best on the check pool: ${ev.val.best_round}` : "on this pool"),
    ].join("");

    const series = [{ name: "test", color: COL.medsam2, points: test.sweep.map((s) => [s.threshold, s.dice3d_mean]) }];
    if (ev.val) series.push({ name: "check (val)", color: COL.val, dash: "4 3",
      points: ev.val.sweep.map((s) => [s.threshold, s.dice3d_mean]) });
    const markers = [{ x: test.threshold, label: "setting" }];
    if (ev.val) markers.push({ x: ev.val.best_threshold, label: "best on check" });
    $("chSweep").innerHTML = "";
    Charts.line($("chSweep"), { series, markers, height: 250,
      xLabel: "Cut-off on MedSAM2's confidence (logit; 0 = probability 0.5)",
      yLabel: "Mean 3D Dice per patient" });

    const d = test.detection;
    $("confusion").innerHTML = `<table class="confusion"><tr><th></th><th>MedSAM2 drew something</th><th>MedSAM2 drew nothing</th></tr>
      <tr><th>Tumour is there</th><td class="ok">${d.tp.toLocaleString()}<em>found</em></td>
        <td class="bad">${d.fn.toLocaleString()}<em>missed</em></td></tr>
      <tr><th>No tumour</th><td class="bad">${d.fp.toLocaleString()}<em>false alarm</em></td>
        <td class="ok">${d.tn.toLocaleString()}<em>correctly empty</em></td></tr></table>`;
  }

  /* ------------------------------------------------------------------ HITL rounds */
  function renderRounds() {
    const ev = S.run && S.run.eval;
    const part = ev && ev[S.split];
    $("chRounds").innerHTML = "";
    $("chAnchors").innerHTML = "";
    if (!part || !part.by_round || !part.by_round.length) return;
    const cfg = (S.run && S.run.config) || {};
    const setting = cfg.hitl && cfg.hitl.rounds;
    const markers = [];
    if (part.best_round) markers.push({ x: part.best_round, label: "best here", color: COL.gain });
    if (setting && setting !== part.best_round) markers.push({ x: setting, label: "setting" });
    const series = [{ name: `MedSAM2 (${part.patients} patients)`, color: COL.medsam2, dots: true,
                      points: part.by_round.map((r) => [r.round, r.dice3d_mean]) }];
    if (part.yolo_dice3d_mean != null) {
      series.push({ name: "YOLO alone", color: COL.yolo, dash: "4 3",
                    points: part.by_round.map((r) => [r.round, part.yolo_dice3d_mean]) });
    }
    Charts.line($("chRounds"), {
      height: 280, markers, series,
      xLabel: "Rounds setting  (round N = N anchors, with anchors.count = 1)",
      yLabel: `Mean 3D Dice over all ${part.patients} patients`,
    });
    Charts.histogram($("chAnchors"), {
      values: (part.patient_scores || []).map((r) => r.anchors), bins: 10, height: 260,
      xLabel: "Anchors used", yLabel: "Patients", color: COL.val,
    });
  }

  /* ------------------------------------------------------------------ did it help */
  function scores() {
    const ev = S.run && S.run.eval;
    const part = ev && ev[S.split];
    return (part && part.patient_scores) || [];
  }

  function renderHelp() {
    const rows = scores();
    const paired = $("chPaired");
    const delta = $("chDelta");
    paired.innerHTML = delta.innerHTML = "";
    if (!rows.length) {
      paired.innerHTML = '<p class="dim">Nothing scored on this pool yet.</p>';
      $("bestTbl").querySelector("tbody").innerHTML = "";
      $("worstTbl").querySelector("tbody").innerHTML = "";
      $("chBySize").innerHTML = $("chByAnchors").innerHTML = "";
      return;
    }
    // The grey "no change" line: points where MedSAM2 scored exactly what YOLO scored.
    const diagonal = Array.from({ length: 51 }, (_, i) => ({ x: i / 50, y: i / 50, color: COL.muted }));
    const dots = rows.map((r) => ({
      x: r.yolo_dice3d, y: r.dice3d, color: r.delta >= 0 ? COL.gain : COL.loss, id: r.id,
      tip: `<div class="tt-title">${esc(r.id)}</div>
        ${Charts.row(COL.yolo, "YOLO alone", f4(r.yolo_dice3d))}
        ${Charts.row(COL.medsam2, "MedSAM2", f4(r.dice3d))}
        ${Charts.row(r.delta >= 0 ? COL.gain : COL.loss, "change", (r.delta >= 0 ? "+" : "") + f4(r.delta))}`,
    }));
    Charts.scatter(paired, { points: [...diagonal, ...dots], height: 300, yMin: 0, yMax: 1,
      xLabel: "3D Dice — YOLO alone", yLabel: "3D Dice — after MedSAM2",
      onClick: (p) => { if (p.id) { S.patient = p.id; $("pSplit").value = S.split; loadPatients().then(runProfile); } } });
    Charts.histogram(delta, { values: rows.map((r) => r.delta), bins: 31, height: 300,
      xLabel: "Change in 3D Dice (MedSAM2 minus YOLO)", yLabel: "Patients", color: COL.medsam2 });

    const table = (el, list) => {
      el.querySelector("thead").innerHTML = `<tr><th>Patient</th><th class="num">YOLO</th>
        <th class="num">MedSAM2</th><th class="num">Change</th><th class="num">Tumour (voxels)</th></tr>`;
      el.querySelector("tbody").innerHTML = list.map((r) => `<tr data-id="${esc(r.id)}">
        <td><code>${esc(r.id)}</code></td><td class="num">${f4(r.yolo_dice3d)}</td>
        <td class="num">${f4(r.dice3d)}</td><td class="num">${deltaCell(r.delta)}</td>
        <td class="num">${r.gt_total.toLocaleString()}</td></tr>`).join("");
      el.querySelectorAll("tbody tr").forEach((tr) => tr.addEventListener("click", () => {
        S.patient = tr.dataset.id; $("pSplit").value = S.split; loadPatients().then(runProfile);
      }));
    };
    const sorted = [...rows].sort((a, b) => b.delta - a.delta);
    table($("bestTbl"), sorted.slice(0, 10));
    table($("worstTbl"), sorted.slice(-10).reverse());

    $("chBySize").innerHTML = $("chByAnchors").innerHTML = "";
    Charts.scatter($("chBySize"), {
      points: rows.filter((r) => r.gt_total > 0).map((r) => ({
        x: r.gt_total, y: r.delta, color: r.delta >= 0 ? COL.gain : COL.loss,
        tip: `<div class="tt-title">${esc(r.id)}</div>${Charts.row(COL.medsam2, "change", f4(r.delta))}
              ${Charts.row(COL.gt, "tumour voxels", r.gt_total.toLocaleString())}` })),
      logX: true, height: 260, xLabel: "Tumour size (voxels, log scale)", yLabel: "Change in 3D Dice",
    });
    Charts.scatter($("chByAnchors"), {
      points: rows.map((r) => ({
        x: r.anchors, y: r.delta, color: r.delta >= 0 ? COL.gain : COL.loss,
        tip: `<div class="tt-title">${esc(r.id)}</div>${Charts.row(COL.val, "anchors", r.anchors)}
              ${Charts.row(COL.yolo, "rounds", r.rounds)}
              ${Charts.row(COL.medsam2, "change", f4(r.delta))}` })),
      height: 260, xLabel: "Anchors the patient ended up with", yLabel: "Change in 3D Dice",
    });
  }

  function downloadCsv() {
    const rows = scores();
    if (!rows.length) return;
    const head = ["patient", "yolo_dice3d", "medsam2_dice3d", "delta", "gt_voxels",
                  "mean_yolo_score", "anchors", "rounds"];
    const body = rows.map((r) => [r.id, r.yolo_dice3d, r.dice3d, r.delta, r.gt_total,
                                  r.score_mean, r.anchors, r.rounds].join(","));
    const blob = new Blob([[head.join(","), ...body].join("\n")], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${S.runId}_${S.split}_per_patient.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  /* ------------------------------------------------------------------ one patient */
  async function loadPatients() {
    const split = $("pSplit").value;
    const r = await api(`/api/pool_patients?split=${split}`).catch(() => ({ patients: [] }));
    S.patients = r.patients;
    const keep = S.patients.includes(S.patient) ? S.patient : S.patients[0];
    S.patient = keep;
    fillSelect($("pPatient"), S.patients.map((p) => ({ value: p, label: p })));
    $("pPatient").value = keep || "";
    writeHash();
  }

  async function runProfile() {
    if (!S.runId || !S.patient) return;
    const which = $("pWhich").value;
    const split = $("pSplit").value;
    $("profileNote").textContent = "Working…";
    let prof = null;
    if (which === "best.pt" && (split === "test" || split === "val")) {
      prof = await api(`/api/runs/${S.runId}/eval/${split}?patient=${encodeURIComponent(S.patient)}`).catch(() => null);
    }
    if (!prof) {
      prof = await api(`/api/runs/${S.runId}/profile.json?patient=${encodeURIComponent(S.patient)}&which=${which}`)
        .catch((e) => { $("profileNote").textContent = e.message; return null; });
    }
    if (!prof) return;
    S.profile = prof;
    drawProfile(prof);
    loadSlice(prof.z[argmax(prof.gt)] ?? prof.z[0]);
  }
  const argmax = (a) => a.reduce((best, v, i) => (v > a[best] ? i : best), 0);

  function drawProfile(p) {
    const box = $("chProfile");
    box.innerHTML = "";
    Charts.dualLine(box, {
      height: 330,
      xLabel: "Slice number (z)", leftLabel: "Dice on this slice", rightLabel: "Tumour pixels on this slice",
      leftMin: 0, leftMax: 1,
      series: [
        { name: "Dice — MedSAM2", color: COL.medsam2, points: p.z.map((z, i) => [z, p.dice[i]]) },
        { name: "Dice — YOLO", color: COL.yolo, dash: "4 3", points: p.z.map((z, i) => [z, p.yolo_dice[i]]) },
        { name: "pixels — expert", color: COL.gt, axis: "right", points: p.z.map((z, i) => [z, p.gt[i]]) },
        { name: "pixels — MedSAM2", color: "#6fb3ff", axis: "right", points: p.z.map((z, i) => [z, p.pred[i]]) },
        { name: "pixels — YOLO", color: "#f0b27a", axis: "right", dash: "4 3", points: p.z.map((z, i) => [z, p.yolo[i]]) },
      ],
      marker: S.slice ? S.slice.z : null,
      onClick: (x) => loadSlice(Math.round(x)),
      markers: (p.anchor_z || []).map((z, i) => ({ x: z, label: `anchor ${i + 1}`, color: COL.anchor })),
    });
    renderRoundsTable(p);
    $("profileNote").innerHTML = `<b>${esc(p.id)}</b> — 3D Dice: MedSAM2 <b>${f4(p.dice3d)}</b>,
      YOLO ${f4(p.yolo_dice3d)} (${deltaCell(p.delta)}). ${p.stored ? "From the stored results of the last scoring pass."
      : "Worked out just now on the graphics card."} The dashed purple lines are the slices that
      were prompted (the anchors); everything else was reached by memory. Click the chart to jump to a slice.`;
  }

  function renderRoundsTable(p) {
    const el = $("roundsTbl");
    const rounds = p.rounds || [];
    el.querySelector("thead").innerHTML = `<tr><th>Round</th><th>Anchor slices (z)</th>
      <th class="num">3D Dice</th><th class="num">Same as the round before</th></tr>`;
    el.querySelector("tbody").innerHTML = rounds.map((r) => `<tr>
      <td>${r.round}</td><td><code>${(r.z || []).join(", ")}</code></td>
      <td class="num">${f4(r.dice3d)}</td>
      <td class="num">${r.same_as_previous == null ? "—" : f4(r.same_as_previous)}</td></tr>`).join("");
    $("roundsWrap").classList.toggle("hidden", !rounds.length);
  }

  /* ------------------------------------------------------------------ slice viewer */
  const panels = () => [...document.querySelectorAll(".panel")].filter((c) => c.checked).map((c) => c.value);

  async function loadSlice(z) {
    if (!S.runId || !S.patient) return;
    const which = $("pWhich").value;
    const q = `patient=${encodeURIComponent(S.patient)}&which=${which}${z != null ? `&z=${z}` : ""}`;
    const info = await api(`/api/runs/${S.runId}/slice.json?${q}`).catch((e) => {
      $("sliceSide").innerHTML = `<p class="err">${esc(e.message)}</p>`; return null;
    });
    if (!info) return;
    S.slice = info;
    const range = $("zRange");
    range.min = 0;
    range.max = info.brain_slices.length - 1;
    range.value = info.brain_slices.indexOf(info.z);
    $("zLabel").textContent = `z = ${info.z}`;
    $("sliceImg").src = `${BASE}/api/runs/${S.runId}/slice.png?${q}&panels=${panels().join(",")}&_=${Date.now()}`;
    $("sliceSide").innerHTML = `<h3>Slice ${info.z}</h3>` + [
      kpi("Expert tumour pixels", info.gt.toLocaleString()),
      kpi("MedSAM2", `${f4(info.medsam2.dice)}`, `${info.medsam2.px.toLocaleString()} px drawn · ${info.medsam2.inter.toLocaleString()} right`),
      kpi("YOLO alone", `${f4(info.yolo.dice)}`, `${info.yolo.px.toLocaleString()} px drawn · ${info.yolo.inter.toLocaleString()} right`),
      kpi("YOLO's best blob score", f2(info.yolo_score), `${info.blobs} blob(s)`),
      kpi("MedSAM2 “is there tumour”", f2(info.obj_score), "above 0 means yes"),
      kpi("This slice", info.is_anchor ? "is an anchor" : "was reached by memory",
          `anchors at z = ${(info.anchor_z || []).join(", ")}`),
    ].join("");
    if (S.profile) drawProfile(S.profile);
  }

  function stepSlice(d) {
    if (!S.slice) return;
    const i = S.slice.brain_slices.indexOf(S.slice.z) + d;
    if (i >= 0 && i < S.slice.brain_slices.length) loadSlice(S.slice.brain_slices[i]);
  }
  function stepPatient(d) {
    const i = S.patients.indexOf(S.patient) + d;
    if (i >= 0 && i < S.patients.length) {
      S.patient = S.patients[i];
      $("pPatient").value = S.patient;
      writeHash();
      runProfile();
    }
  }

  /* ------------------------------------------------------------------ wiring */
  async function selectRun(id) {
    S.runId = id;
    S.logOffset = 0;
    writeHash();
    renderRuns();
    await loadRun();
  }

  async function loadRun() {
    if (!S.runId) { renderLive(null); renderEval(null); return; }
    S.run = await api(`/api/runs/${S.runId}`).catch(() => null);
    renderLive(S.run);
    renderEval(S.run);
    renderRounds();
    renderHelp();
    drawCurves();
    await pullLog(true);
  }

  async function refresh() {
    S.overview = await api("/api/overview");
    $("topSub").innerHTML = `Pools: <b>${S.overview.pools.train}</b> train ·
      <b>${S.overview.pools.val}</b> check · <b>${S.overview.pools.test}</b> test patients ·
      ${S.overview.yolo_runs.length} YOLO run(s) available · ${S.overview.runs.length} fine-tune(s) here`;
    const c = S.overview.config;
    $("howNote").innerHTML = `Right now: prompts come from YOLO run
      <code>${esc(c.prompt.yolo_run || "the best scoring one")}</code>, style
      <b>${esc(c.prompt.variant)}</b>; the first <b>${c.anchors.count}</b> anchor(s) are chosen by
      the most confident slice, then up to <b>${c.hitl.rounds}</b> correction round(s);
      MedSAM2 starts from <code>${esc(c.model.checkpoint)}</code> and
      only <b>${esc(c.train.unfreeze)}</b> may change.`;
    document.querySelectorAll("[data-cfg]").forEach((el) => {
      el.textContent = el.dataset.cfg.split(".").reduce((n, k) => (n == null ? n : n[k]), c);
    });
    renderRuns();
    if (!S.runId && S.overview.runs.length) S.runId = S.overview.active || S.overview.runs[0].id;
  }

  async function boot() {
    readHash();
    await refresh();
    fillForm(S.overview);
    await loadRun();
    await loadPatients();

    $("startForm").addEventListener("submit", (e) => { e.preventDefault(); start(false); });
    $("btnSmoke").addEventListener("click", () => start(true));
    $("btnReset").addEventListener("click", resetForm);
    $("btnStop").addEventListener("click", () => act("stop"));
    $("btnResume").addEventListener("click", () => act("resume"));
    $("btnRescore").addEventListener("click", () => act("evaluate"));
    $("btnCsv").addEventListener("click", downloadCsv);
    $("helpSplit").addEventListener("change", () => { S.split = $("helpSplit").value; writeHash(); renderRounds(); renderHelp(); });
    $("pSplit").addEventListener("change", () => loadPatients());
    $("pPatient").addEventListener("change", () => { S.patient = $("pPatient").value; writeHash(); });
    $("btnProfile").addEventListener("click", runProfile);
    $("pWhich").addEventListener("change", runProfile);
    $("logFilter").addEventListener("input", () => pullLog(true));
    $("zRange").addEventListener("input", () => {
      if (S.slice) loadSlice(S.slice.brain_slices[Number($("zRange").value)]);
    });
    document.querySelectorAll(".panel").forEach((c) => c.addEventListener("change", () => loadSlice(S.slice && S.slice.z)));
    document.addEventListener("keydown", (e) => {
      if (/input|select|textarea/i.test(e.target.tagName)) return;
      if (e.key === "ArrowLeft") stepSlice(-1);
      else if (e.key === "ArrowRight") stepSlice(1);
      else if (e.key === "[") stepPatient(-1);
      else if (e.key === "]") stepPatient(1);
    });

    setInterval(tick, 2000);
    setInterval(gpu, 3000);
    setInterval(refresh, 15000);
    gpu();
  }

  async function start(smoke) {
    $("startErr").textContent = "";
    try {
      const r = await post("/api/runs", { overrides: overrides(), smoke });
      await refresh();
      await selectRun(r.id);
      $("secLive").scrollIntoView({ behavior: "smooth" });
    } catch (e) {
      $("startErr").textContent = e.message;
    }
  }

  async function act(what) {
    if (!S.runId) return;
    try {
      await post(`/api/runs/${S.runId}/${what}`);
      await refresh();
      await loadRun();
    } catch (e) {
      $("liveSub").innerHTML += ` <span class="err">${esc(e.message)}</span>`;
    }
  }

  async function tick() {
    if (!S.runId) return;
    const run = await api(`/api/runs/${S.runId}`).catch(() => null);
    if (!run) return;
    const wasLive = S.run && ["queued", "running"].includes((S.run.status || {}).state);
    const nowLive = ["queued", "running"].includes((run.status || {}).state);
    const rounds = S.run ? S.run.epochs_done : -1;
    S.run = run;
    renderLive(run);
    if (nowLive) await pullLog(false);
    if (run.epochs_done !== rounds) drawCurves();
    if (wasLive && !nowLive) { renderEval(run); renderRounds(); renderHelp(); await refresh(); }
  }

  async function gpu() {
    const g = await api("/api/gpu").catch(() => ({ available: false }));
    $("gpu").innerHTML = g.available
      ? `<span class="dim">${esc(g.name)}</span> <div class="gbar"><i style="width:${g.util || 0}%"></i></div>
         <span>${g.util || 0}%</span> <span class="dim">${Math.round(g.mem_used || 0)} / ${Math.round(g.mem_total || 0)} MB</span>`
      : '<span class="dim">no GPU reading</span>';
  }

  boot().catch((e) => { $("topSub").innerHTML = `<span class="err">${esc(e.message)}</span>`; });
})();
