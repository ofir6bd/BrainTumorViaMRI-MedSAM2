/* GT vs YOLO tab — MedSAM2 prompted with the expert mask vs with the YOLO mask.
 *
 * Selecting a patient starts the run (both arms, up to 7 anchors each). That is real
 * GPU work — tens of seconds — and the request blocks until it finishes, so the status
 * line says what is happening. Results are cached server-side under outputs/gtvsyolo/,
 * so a patient you have already run comes back instantly and is marked in the dropdown.
 */
(function () {
  const $ = (id) => document.getElementById(id);
  const els = {
    patient: $("gvPatient"),
    model: $("gvModel"),
    z: $("gvZ"),
    zLabel: $("gvZLabel"),
    best: $("gvBest"),
    rerun: $("gvRerun"),
    status: $("gvStatus"),
    viewer: $("gvViewer"),
    views: $("gvViews"),
    slicePanel: $("gvSlicePanel"),
    summaryPanel: $("gvSummaryPanel"),
    arms: $("gvArms"),
    anchors: $("gvAnchors"),
    sliceChart: $("gvSliceChart"),
    roundChart: $("gvRoundChart"),
    table: $("gvTableWrap"),
  };
  if (!els.patient) return;

  const COLORS = { gt: "#40a6ff", yolo: "#ff5a26", truth: "#26ff26" };
  const state = { id: null, weights: "", result: null, view: "segment" };

  // ---------------------------------------------------------------- chart
  function lineChart(series, opts = {}) {
    const width = opts.width || 620, height = opts.height || 210;
    const padL = 54, padR = 14, padT = 12, padB = 30;
    const pts = series.flatMap((s) => s.data);
    if (!pts.length) return "";
    const xs = pts.map((p) => p.x);
    let xMin = Math.min(...xs), xMax = Math.max(...xs);
    if (xMax === xMin) xMax = xMin + 1;
    const yMin = 0, yMax = 1;
    const plotW = width - padL - padR, plotH = height - padT - padB;
    const sx = (x) => padL + ((x - xMin) / (xMax - xMin)) * plotW;
    const sy = (y) => padT + plotH - ((y - yMin) / (yMax - yMin)) * plotH;

    let svg = `<svg viewBox="0 0 ${width} ${height}" class="yolo-chart-svg" `
      + `style="height:${height}px" preserveAspectRatio="xMidYMid meet">`;
    [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
      const y = sy(t);
      svg += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${padL + plotW}" y2="${y.toFixed(1)}" class="yolo-chart-grid" />`
        + `<text x="${padL - 6}" y="${(y + 3).toFixed(1)}" class="yolo-chart-ticklabel yolo-chart-ylabel" text-anchor="end">${t.toFixed(2)}</text>`;
    });
    const ticks = opts.xTickCount || 4;
    for (let i = 0; i <= ticks; i++) {
      const t = xMin + (i / ticks) * (xMax - xMin);
      svg += `<text x="${sx(t).toFixed(1)}" y="${(padT + plotH + 18).toFixed(1)}" class="yolo-chart-ticklabel yolo-chart-xlabel" text-anchor="middle">${Math.round(t)}</text>`;
    }
    svg += `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" class="yolo-chart-axis" />`
      + `<line x1="${padL}" y1="${padT + plotH}" x2="${padL + plotW}" y2="${padT + plotH}" class="yolo-chart-axis" />`;

    (opts.markers || []).forEach((m) => {
      if (m.x < xMin || m.x > xMax) return;
      const x = sx(m.x);
      svg += `<line x1="${x.toFixed(1)}" y1="${padT}" x2="${x.toFixed(1)}" y2="${padT + plotH}" `
        + `stroke="${m.color || "#8a8aa0"}" stroke-width="1" stroke-dasharray="3 3" opacity="0.85" />`;
      if (m.label) {
        svg += `<text x="${x.toFixed(1)}" y="${(padT + 10).toFixed(1)}" class="yolo-chart-ticklabel" `
          + `text-anchor="middle" style="fill:${m.color || "#8a8aa0"}">${m.label}</text>`;
      }
    });

    series.forEach((s) => {
      if (!s.data.length) return;
      const d = s.data.map((p, j) => `${j === 0 ? "M" : "L"} ${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}`).join(" ");
      svg += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" class="yolo-series-line" />`;
      if (s.dots) {
        s.data.forEach((p) => {
          svg += `<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="3.5" fill="${s.color}" />`;
        });
      }
    });
    svg += `</svg>`;

    const legend = series.map((s) =>
      `<span class="yolo-chart-legend-item"><i style="background:${s.color}"></i>${s.label}</span>`).join("");
    return `<div class="yolo-chart-wrap">${svg}</div>`
      + `<div class="yolo-chart-legend">${legend}</div>`
      + (opts.caption ? `<p class="gv-caption">${opts.caption}</p>` : "");
  }

  // ---------------------------------------------------------------- render
  function armCard(arm, a, comparable) {
    const cut = a.anchors_used > comparable
      ? `<span class="gv-note">compare head-to-head only up to ${comparable} anchor${comparable === 1 ? "" : "s"}</span>`
      : "";
    return `<div class="gv-card" style="border-color:${COLORS[arm]}">
      <h4 style="color:${COLORS[arm]}">${a.label}</h4>
      <p class="gv-dice">${a.final_dice.toFixed(4)}<span>Dice (3D, vs GT WT)</span></p>
      <dl>
        <dt>Anchor slices used</dt><dd><b>${a.anchors_used}</b> / ${a.anchors_available}</dd>
        <dt>Slice numbers</dt><dd class="gv-zlist">${a.anchor_z.length ? a.anchor_z.join(", ") : "&mdash;"}</dd>
        <dt>Stopped because</dt><dd>${a.stop_reason}</dd>
        <dt>Runtime</dt><dd>${a.seconds.toFixed(1)} s</dd>
      </dl>${cut}</div>`;
  }

  function renderSummary(r) {
    const gt = r.arms.gt, yolo = r.arms.yolo;
    els.arms.innerHTML = armCard("gt", gt, r.comparable_rounds) + armCard("yolo", yolo, r.comparable_rounds);

    const empties = new Set(r.yolo_empty_anchors);
    const chips = r.schedule.map((z, i) => {
      const usedGt = gt.anchor_z.includes(z), usedYolo = yolo.anchor_z.includes(z);
      const cls = ["gv-chip"];
      if (!usedGt && !usedYolo) cls.push("gv-chip-unused");
      if (empties.has(z)) cls.push("gv-chip-empty");
      const marks = [usedGt ? `<i style="background:${COLORS.gt}"></i>` : "",
        usedYolo ? `<i style="background:${COLORS.yolo}"></i>` : ""].join("");
      const title = `anchor #${i + 1} — z=${z}`
        + (usedGt ? " · used by GT arm" : " · not reached by GT arm")
        + (usedYolo ? " · used by YOLO arm" : " · not reached by YOLO arm")
        + (empties.has(z) ? " · YOLO found no tumour here (no mask)" : "");
      return `<span class="${cls.join(" ")}" title="${title}">z=${z}${marks}</span>`;
    }).join("");
    els.anchors.innerHTML = `<h4>Anchor schedule <span class="gv-sub">fixed from GT geometry before either arm ran &mdash; both arms are offered these slices in this order</span></h4>`
      + `<div class="gv-chips">${chips}</div>`
      + `<p class="gv-caption">Filled dot = that arm actually used the slice. `
      + `${r.yolo_empty_anchors.length ? `YOLO had no mask on z = ${r.yolo_empty_anchors.join(", ")} — that arm got no prompt there.` : "YOLO produced a mask on every anchor it reached."}</p>`;

    els.sliceChart.innerHTML = `<h4>Per-slice Dice</h4>` + lineChart([
      { label: "MedSAM2 ← GT mask", color: COLORS.gt, data: r.slices.map((s) => ({ x: s.z, y: s.dice_gt })) },
      { label: "MedSAM2 ← YOLO mask", color: COLORS.yolo, data: r.slices.map((s) => ({ x: s.z, y: s.dice_yolo })) },
    ], {
      markers: r.schedule.map((z, i) => ({ x: z, label: `a${i + 1}`, color: "#8a8aa0" })),
      caption: "x = slice (z), y = Dice against the expert whole-tumour mask. Dashed lines are the anchor slices.",
    });

    els.roundChart.innerHTML = `<h4>Convergence</h4>` + lineChart([
      { label: "MedSAM2 ← GT mask", color: COLORS.gt, dots: true, data: gt.rounds.map((x) => ({ x: x.anchors_used, y: x.dice })) },
      { label: "MedSAM2 ← YOLO mask", color: COLORS.yolo, dots: true, data: yolo.rounds.map((x) => ({ x: x.anchors_used, y: x.dice })) },
    ], {
      xTickCount: Math.max(1, Math.max(gt.anchors_used, yolo.anchors_used) - 1),
      caption: `x = number of anchor slices given, y = whole-volume Dice. Equal-anchor comparison is valid up to ${r.comparable_rounds}.`,
    });

    const rows = r.slices.map((s) => {
      const d = s.dice_yolo - s.dice_gt;
      const cls = d > 0.001 ? "gv-pos" : d < -0.001 ? "gv-neg" : "";
      const anchor = r.schedule.includes(s.z) ? ` <span class="gv-anchor-tag">anchor</span>` : "";
      return `<tr><td>${s.z}${anchor}</td><td>${s.gt_voxels.toLocaleString()}</td>`
        + `<td>${s.dice_gt.toFixed(4)}</td><td>${s.dice_yolo.toFixed(4)}</td>`
        + `<td class="${cls}">${d >= 0 ? "+" : ""}${d.toFixed(4)}</td></tr>`;
    }).join("");
    const mean = (k) => r.slices.length ? r.slices.reduce((a, s) => a + s[k], 0) / r.slices.length : 0;
    els.table.innerHTML = `<table class="yolo-summary-table">
      <thead><tr><th>Slice z</th><th>GT voxels</th><th>Dice (GT mask)</th><th>Dice (YOLO mask)</th><th>Δ</th></tr></thead>
      <tbody>${rows}</tbody>
      <tfoot><tr><td>mean over ${r.slices.length} slices</td><td></td>
        <td>${mean("dice_gt").toFixed(4)}</td><td>${mean("dice_yolo").toFixed(4)}</td>
        <td>${(mean("dice_yolo") - mean("dice_gt")).toFixed(4)}</td></tr></tfoot></table>`;
  }

  function refreshImage() {
    if (state.id === null || !state.result) return;
    const z = Number(els.z.value);
    const anchorIdx = state.result.schedule.indexOf(z);
    els.zLabel.textContent = `z = ${z}` + (anchorIdx >= 0 ? `  ·  anchor #${anchorIdx + 1}` : "");
    els.viewer.src = `/gtvsyolo/segment.png?id=${state.id}&z=${z}`
      + `&weights=${encodeURIComponent(state.weights)}&_=${Date.now()}`;
  }

  // ---------------------------------------------------------------- run
  async function run(force) {
    if (state.id === null) return;
    const label = els.patient.options[els.patient.selectedIndex].textContent;
    els.status.textContent = `Running MedSAM2 twice for ${label.replace(" ✓", "")} — GT-mask-prompted and YOLO-mask-prompted, up to 7 anchors each. This takes tens of seconds and the page will not update until it finishes…`;
    els.status.classList.add("gv-running");
    els.viewer.removeAttribute("src");
    els.arms.innerHTML = els.anchors.innerHTML = els.sliceChart.innerHTML = "";
    els.roundChart.innerHTML = els.table.innerHTML = "";
    try {
      const url = `/gtvsyolo/api/patient/${state.id}?weights=${encodeURIComponent(state.weights)}`
        + (force ? "&force=1" : "");
      const res = await fetch(url);
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
      state.result = data;
      els.z.min = 0;
      els.z.max = Math.max(0, data.depth - 1);
      els.z.value = data.best_slice_index;
      els.status.classList.remove("gv-running");
      els.status.textContent = `${data.patient_id} — GT mask: Dice ${data.arms.gt.final_dice.toFixed(4)} `
        + `(${data.arms.gt.anchors_used}/${data.arms.gt.anchors_available} anchors) · `
        + `YOLO mask: Dice ${data.arms.yolo.final_dice.toFixed(4)} `
        + `(${data.arms.yolo.anchors_used}/${data.arms.yolo.anchors_available} anchors) · `
        + `${data.total_seconds.toFixed(1)} s total`;
      renderSummary(data);
      refreshImage();
      loadPatients(els.patient.value);
    } catch (err) {
      els.status.classList.remove("gv-running");
      els.status.textContent = `Failed: ${err.message}`;
    }
  }

  async function loadPatients(keep) {
    const list = await (await fetch(`/gtvsyolo/api/patients?weights=${encodeURIComponent(state.weights)}`)).json();
    els.patient.innerHTML = "";
    for (const p of list) {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.cached ? `${p.label} ✓` : p.label;
      els.patient.appendChild(opt);
    }
    if (keep !== undefined) els.patient.value = keep;
  }

  async function loadWeights() {
    const list = await (await fetch("/gtvsyolo/api/weights")).json();
    els.model.innerHTML = "";
    for (const w of list) {
      const opt = document.createElement("option");
      opt.value = w.filename;
      opt.textContent = w.label;
      els.model.appendChild(opt);
    }
    state.weights = list.length ? list[0].filename : "";
  }

  // ---------------------------------------------------------------- wiring
  els.patient.addEventListener("change", () => {
    state.id = Number(els.patient.value);
    run(false);
  });
  els.model.addEventListener("change", async () => {
    state.weights = els.model.value;
    await loadPatients();
    state.id = Number(els.patient.value);
    run(false);
  });
  els.z.addEventListener("input", refreshImage);
  els.best.addEventListener("click", () => {
    if (!state.result) return;
    els.z.value = state.result.best_slice_index;
    refreshImage();
  });
  els.rerun.addEventListener("click", () => run(true));
  els.views.addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (!btn) return;
    state.view = btn.dataset.gview;
    for (const b of els.views.querySelectorAll(".tab")) b.classList.toggle("active", b === btn);
    els.slicePanel.classList.toggle("hidden", state.view !== "segment");
    els.summaryPanel.classList.toggle("hidden", state.view !== "summary");
  });

  (async function init() {
    await loadWeights();
    await loadPatients();
    if (els.patient.options.length) {
      state.id = Number(els.patient.value);
      // Show the pre-selected patient straight away if its result is already on disk;
      // never kick off a fresh multi-arm run just because the tab was opened.
      const cached = els.patient.options[els.patient.selectedIndex].textContent.endsWith("✓");
      if (cached) {
        run(false);
      } else {
        els.status.textContent = "Select a patient to run the comparison "
          + "(a ✓ marks patients already computed — those load instantly).";
      }
    } else {
      els.status.textContent = "No test-split patients found.";
    }
  })();
})();
