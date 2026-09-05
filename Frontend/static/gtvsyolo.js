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
    search: $("gvPatientSearch"),
    count: $("gvPatientCount"),
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

  const COLORS = {
    gt: "#40a6ff", yolo: "#ff5a26", truth: "#26ff26",
    gtVox: "#8fd0ff", yoloVox: "#ffb08a",   // voxel curves: paler arm colours
  };
  const state = { id: null, weights: "", result: null, view: "segment", patients: [] };

  // ---------------------------------------------------------------- chart
  /* Two-axis line chart. Series carry `axis: "right"` to hang off the right-hand scale
     (voxel counts, which share no units with Dice) and `dash` for a dashed stroke.
     Hidden series are excluded from the axis ranges, so hiding the big voxel curves
     rescales what is left instead of leaving it squashed. */
  function chartSVG(series, opts) {
    const width = opts.width || 620, height = opts.height || 210;
    const visible = series.filter((s) => !s.hidden && s.data && s.data.length);
    const right = visible.filter((s) => s.axis === "right");
    const padL = 54, padR = right.length ? 62 : 14, padT = 12, padB = 30;
    const pts = visible.flatMap((s) => s.data);
    if (!pts.length) return "";

    const xs = pts.map((p) => p.x);
    let xMin = Math.min(...xs), xMax = Math.max(...xs);
    if (xMax === xMin) xMax = xMin + 1;
    const plotW = width - padL - padR, plotH = height - padT - padB;

    // Left axis is Dice, pinned to 0-1 so runs stay visually comparable.
    const sx = (x) => padL + ((x - xMin) / (xMax - xMin)) * plotW;
    const sy = (y) => padT + plotH - y * plotH;
    let rMax = 1;
    if (right.length) {
      rMax = Math.max(...right.flatMap((s) => s.data.map((p) => p.y)), 1);
    }
    const syR = (y) => padT + plotH - (y / rMax) * plotH;

    let svg = '<svg viewBox="0 0 ' + width + ' ' + height + '" class="yolo-chart-svg" '
      + 'style="height:' + height + 'px" preserveAspectRatio="xMidYMid meet">';
    [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
      const y = sy(t);
      svg += '<line x1="' + padL + '" y1="' + y.toFixed(1) + '" x2="' + (padL + plotW)
        + '" y2="' + y.toFixed(1) + '" class="yolo-chart-grid" />'
        + '<text x="' + (padL - 6) + '" y="' + (y + 3).toFixed(1)
        + '" class="yolo-chart-ticklabel yolo-chart-ylabel" text-anchor="end">'
        + t.toFixed(2) + '</text>';
    });
    if (right.length) {
      const rColor = right[0].color;
      [0, 0.5, 1].forEach((f) => {
        svg += '<text x="' + (padL + plotW + 8) + '" y="' + (syR(rMax * f) + 3).toFixed(1)
          + '" class="yolo-chart-ticklabel yolo-chart-ylabel" text-anchor="start" style="fill:'
          + rColor + '">' + Math.round(rMax * f).toLocaleString() + '</text>';
      });
      svg += '<line x1="' + (padL + plotW) + '" y1="' + padT + '" x2="' + (padL + plotW)
        + '" y2="' + (padT + plotH) + '" class="yolo-chart-axis" />';
    }
    /* Either a fixed step along x (opts.xTickStep, e.g. every 5 slices) or a count of
       evenly spaced ticks across the range. */
    const xTicks = [];
    if (opts.xTickStep) {
      const step = opts.xTickStep;
      for (let t = Math.ceil(xMin / step) * step; t <= xMax; t += step) xTicks.push(t);
    } else {
      const ticks = opts.xTickCount || 4;
      for (let i = 0; i <= ticks; i++) xTicks.push(xMin + (i / ticks) * (xMax - xMin));
    }
    // A dense tick row needs a smaller label or the numbers run into each other.
    const xLabelStyle = opts.xLabelSize ? ' style="font-size:' + opts.xLabelSize + 'px"' : "";
    xTicks.forEach((t) => {
      svg += '<text x="' + sx(t).toFixed(1) + '" y="' + (padT + plotH + 18).toFixed(1)
        + '" class="yolo-chart-ticklabel yolo-chart-xlabel" text-anchor="middle"'
        + xLabelStyle + '>' + Math.round(t) + '</text>';
    });
    svg += '<line x1="' + padL + '" y1="' + padT + '" x2="' + padL + '" y2="' + (padT + plotH)
      + '" class="yolo-chart-axis" />'
      + '<line x1="' + padL + '" y1="' + (padT + plotH) + '" x2="' + (padL + plotW)
      + '" y2="' + (padT + plotH) + '" class="yolo-chart-axis" />';

    (opts.markers || []).forEach((m) => {
      if (m.x < xMin || m.x > xMax) return;
      const x = sx(m.x), color = m.color || "#8a8aa0";
      svg += '<line x1="' + x.toFixed(1) + '" y1="' + padT + '" x2="' + x.toFixed(1)
        + '" y2="' + (padT + plotH) + '" stroke="' + color
        + '" stroke-width="1" stroke-dasharray="3 3" opacity="0.85" />';
      if (m.label) {
        svg += '<text x="' + x.toFixed(1) + '" y="' + (padT + 10).toFixed(1)
          + '" class="yolo-chart-ticklabel" text-anchor="middle" style="fill:' + color + '">'
          + m.label + '</text>';
      }
    });

    visible.forEach((s) => {
      const scale = s.axis === "right" ? syR : sy;
      const d = s.data.map((p, j) => (j === 0 ? "M" : "L") + " " + sx(p.x).toFixed(1)
        + " " + scale(p.y).toFixed(1)).join(" ");
      svg += '<path d="' + d + '" fill="none" stroke="' + s.color + '" stroke-width="'
        + (s.dash ? 1.5 : 2) + '" '
        + (s.dash ? 'stroke-dasharray="' + s.dash + '" ' : "")
        + 'class="yolo-series-line" />';
      if (s.dots) {
        s.data.forEach((p) => {
          svg += '<circle cx="' + sx(p.x).toFixed(1) + '" cy="' + scale(p.y).toFixed(1)
            + '" r="3.5" fill="' + s.color + '" />';
        });
      }
    });
    return svg + "</svg>";
  }

  /* Draw into `container` and keep the series list live: clicking a legend entry toggles
     that series and redraws (which also rescales the axes). */
  function renderChart(container, title, series, opts = {}) {
    const live = series.map((s) => Object.assign({ hidden: false }, s));
    function draw() {
      const legend = live.map((s, i) =>
        '<span class="yolo-chart-legend-item' + (s.hidden ? " gv-legend-off" : "")
        + '" data-idx="' + i + '" title="Click to show/hide this series">'
        + '<i style="background:' + s.color + '"></i>' + s.label + "</span>").join("");
      container.innerHTML = "<h4>" + title + "</h4>"
        + '<div class="yolo-chart-wrap">' + chartSVG(live, opts) + "</div>"
        + '<div class="yolo-chart-legend">' + legend + "</div>"
        + (opts.caption ? '<p class="gv-caption">' + opts.caption + "</p>" : "");
      container.querySelectorAll(".yolo-chart-legend-item").forEach((el) => {
        el.addEventListener("click", () => {
          const i = Number(el.dataset.idx);
          live[i].hidden = !live[i].hidden;
          draw();
        });
      });
    }
    draw();
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

    renderChart(els.sliceChart, "Per-slice Dice and tumour voxels", [
      // Dice is null on slices with neither ground truth nor prediction — nothing to
      // score there, so those points are dropped rather than drawn as a perfect 1.0.
      { label: "Dice \u2014 MedSAM2 \u2190 GT mask", color: COLORS.gt,
        data: r.slices.filter((s) => s.dice_gt !== null).map((s) => ({ x: s.z, y: s.dice_gt })) },
      { label: "Dice \u2014 MedSAM2 \u2190 YOLO mask", color: COLORS.yolo,
        data: r.slices.filter((s) => s.dice_yolo !== null).map((s) => ({ x: s.z, y: s.dice_yolo })) },
      { label: "GT tumour voxels (right)", color: COLORS.truth, axis: "right", dash: "5 3",
        data: r.slices.map((s) => ({ x: s.z, y: s.gt_voxels })) },
      { label: "Segmented voxels, GT arm (right)", color: COLORS.gtVox, axis: "right", dash: "2 3",
        data: r.slices.map((s) => ({ x: s.z, y: s.voxels_gt_arm })) },
      { label: "Segmented voxels, YOLO arm (right)", color: COLORS.yoloVox, axis: "right", dash: "2 3",
        data: r.slices.map((s) => ({ x: s.z, y: s.voxels_yolo_arm })) },
    ], {
      width: 1240, xTickStep: 5, xLabelSize: 10,
      markers: r.schedule.map((z, i) => ({ x: z, label: "a" + (i + 1), color: "#8a8aa0" })),
      caption: "x = every slice in the volume. Left axis: Dice against the expert "
        + "whole-tumour mask (slices with neither tumour nor prediction are unscored "
        + "and left blank). "
        + "Right axis: tumour voxels on that slice \u2014 expert, and as segmented by each "
        + "arm, so over- and under-segmentation is visible where Dice alone is ambiguous. "
        + "Dotted verticals are the anchor slices. Click a legend entry to hide a series.",
    });

    renderChart(els.roundChart, "Convergence", [
      { label: "MedSAM2 \u2190 GT mask", color: COLORS.gt, dots: true,
        data: gt.rounds.map((x) => ({ x: x.anchors_used, y: x.dice })) },
      { label: "MedSAM2 \u2190 YOLO mask", color: COLORS.yolo, dots: true,
        data: yolo.rounds.map((x) => ({ x: x.anchors_used, y: x.dice })) },
    ], {
      xTickCount: Math.max(1, Math.max(gt.anchors_used, yolo.anchors_used) - 1),
      caption: "x = number of anchor slices given, y = whole-volume Dice. "
        + "Equal-anchor comparison is valid up to " + r.comparable_rounds + ".",
    });

    const fmt = (v) => (v === null ? "&mdash;" : v.toFixed(4));
    const rows = r.slices.map((s) => {
      const scored = s.dice_gt !== null && s.dice_yolo !== null;
      const d = scored ? s.dice_yolo - s.dice_gt : null;
      const cls = d === null ? "" : d > 0.001 ? "gv-pos" : d < -0.001 ? "gv-neg" : "";
      const anchor = r.schedule.includes(s.z) ? ` <span class="gv-anchor-tag">anchor</span>` : "";
      const blank = s.gt_voxels === 0 && !s.voxels_gt_arm && !s.voxels_yolo_arm
        ? " gv-row-empty" : "";
      return `<tr class="${blank.trim()}"><td>${s.z}${anchor}</td>`
        + `<td>${s.gt_voxels.toLocaleString()}</td>`
        + `<td>${fmt(s.dice_gt)}</td><td>${fmt(s.dice_yolo)}</td>`
        + `<td class="${cls}">${d === null ? "&mdash;" : (d >= 0 ? "+" : "") + d.toFixed(4)}</td></tr>`;
    }).join("");
    // Means are over scored slices only; averaging in the empty background would
    // silently drag both arms towards each other.
    const scoredRows = r.slices.filter((s) => s.dice_gt !== null || s.dice_yolo !== null);
    const mean = (k) => {
      const vals = scoredRows.map((s) => s[k]).filter((v) => v !== null);
      return vals.length ? vals.reduce((a, v) => a + v, 0) / vals.length : 0;
    };
    els.table.innerHTML = `<table class="yolo-summary-table">
      <thead><tr><th>Slice z</th><th>GT voxels</th><th>Dice (GT mask)</th><th>Dice (YOLO mask)</th><th>Δ</th></tr></thead>
      <tbody>${rows}</tbody>
      <tfoot><tr><td>mean over ${scoredRows.length} scored of ${r.slices.length} slices</td><td></td>
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

  /* Rebuild the dropdown: sorted by patient id (not the split's shuffled order) and
     narrowed to the search box. The selected patient is always kept in the list even if
     it does not match the filter, so filtering never blanks the selection — and options
     are only rewritten, never selected, so this can never fire `change` and kick off a
     run. */
  function renderPatientOptions(keep) {
    const q = (els.search.value || "").trim().toLowerCase();
    const current = keep !== undefined && keep !== null
      ? String(keep)
      : (state.id === null ? null : String(state.id));
    const matches = state.patients.filter((p) => !q || p.label.toLowerCase().includes(q));
    const shown = matches.slice();
    if (current !== null && !shown.some((p) => String(p.id) === current)) {
      const kept = state.patients.find((p) => String(p.id) === current);
      if (kept) shown.unshift(kept);
    }
    els.patient.innerHTML = "";
    for (const p of shown) {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.cached ? `${p.label} ✓` : p.label;
      els.patient.appendChild(opt);
    }
    if (current !== null) els.patient.value = current;
    els.count.textContent = q
      ? `${matches.length} / ${state.patients.length}`
      : `${state.patients.length} patients`;
  }

  async function loadPatients(keep) {
    const list = await (await fetch(`/gtvsyolo/api/patients?weights=${encodeURIComponent(state.weights)}`)).json();
    list.sort((a, b) => a.label.localeCompare(b.label, undefined, { numeric: true }));
    state.patients = list;
    renderPatientOptions(keep);
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
  els.search.addEventListener("input", () => renderPatientOptions());
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
