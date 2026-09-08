/* Inference tests tab — YOLO confidence variants as MedSAM2 prompts.
 *
 * A patient is eleven MedSAM2 runs, so the sweep is driven from here one variant at a
 * time: request O, draw its row, request GT, draw that row, and so on. Each request stays
 * short enough to survive a browser timeout, the table fills in while the GPU works, and
 * Stop actually stops (the run in flight finishes and is cached, nothing after it starts).
 *
 * Results are cached server-side under outputs/yolomedsam2test/, so a variant already run
 * comes back instantly and is ticked in the dropdown.
 *
 * Charts are drawn into a wide viewBox with `height:auto`, so they scale to the full
 * width of the panel rather than letterboxing inside a fixed box. Every axis carries its
 * own title — a Dice axis says "Dice", a slice axis says which slice — and slice charts
 * span the whole volume (0..depth-1) rather than only the slices that happen to score, so
 * two variants are always read against the same x range.
 */
(function () {
  const $ = (id) => document.getElementById(id);
  const els = {
    patient: $("itPatient"), search: $("itPatientSearch"), count: $("itPatientCount"),
    model: $("itModel"), runAll: $("itRunAll"), stop: $("itStop"), rerun: $("itRerun"),
    runAllPatients: $("itRunAllPatients"),
    views: $("itViews"), status: $("itStatus"),
    variantsPanel: $("itVariantsPanel"), varTable: $("itVarTable"),
    roundChart: $("itRoundChart"), sliceChart: $("itSliceChart"),
    deltaChart: $("itDeltaChart"), volumeChart: $("itVolumeChart"),
    slicePanel: $("itSlicePanel"), variantSel: $("itVariant"),
    z: $("itZ"), zLabel: $("itZLabel"), best: $("itBest"), viewer: $("itViewer"),
    sliceField: $("itSliceField"),
    probePanel: $("itProbePanel"), probeChart: $("itProbeChart"),
    probeScatter: $("itProbeScatter"), probeBins: $("itProbeBins"),
    probeHist: $("itProbeHist"),
    probeStats: $("itProbeStats"),
    aggPanel: $("itAggPanel"), aggTable: $("itAggTable"), aggChart: $("itAggChart"),
    aggDelta: $("itAggDelta"), aggSpread: $("itAggSpread"), aggAnchors: $("itAggAnchors"),
    aggRefresh: $("itAggRefresh"),
  };
  if (!els.patient) return;

  const state = {
    id: null, weights: "", view: "variants", patients: [], variants: [],
    results: {}, running: false, cancel: false, depth: 0, best: 0, selected: "O",
    zTouched: false, aggregate: null,
  };

  const fmt = (v, n = 4) => (v === null || v === undefined ? "—" : Number(v).toFixed(n));
  const signed = (v) => {
    if (v === null || v === undefined) return "—";
    const s = (v >= 0 ? "+" : "") + v.toFixed(4);
    return '<span class="' + (v > 1e-6 ? "gv-pos" : v < -1e-6 ? "gv-neg" : "") + '">'
      + s + "</span>";
  };
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");

  /* The "what is different" cell: the change in plain words, then the question that
     change is there to answer. Two lines, because "mask amplitude set by the slice's
     area-weighted confidence" tells a reader who already knows the code nothing they
     did not know, and everyone else nothing at all. */
  /* ID and label in one cell. They were two columns saying the same thing, and the
     width they cost came straight out of the prose column next to them. */
  const nameCell = (v) => '<td class="it-name"><b style="color:' + v.color + '">'
    + esc(v.id) + "</b><span>" + esc(v.label) + "</span></td>";

  const whatCell = (v) => '<td class="it-what"><span class="it-change">'
    + esc(v.changes) + '</span>'
    + (v.why ? '<span class="it-why">' + esc(v.why) + "</span>" : "") + "</td>";

  // ---------------------------------------------------------------- charts
  // One coordinate system for every chart. The SVG is stretched to the panel width by CSS
  // (`height:auto`), so these are layout units, not pixels — a wide box keeps the tick
  // labels from colliding once a 180-slice axis is drawn across it.
  const VB_W = 1200;
  const PAD = { l: 96, r: 44, t: 22, b: 66 };
  const RIGHT_PAD = 104;

  /* Line/scatter chart.
     opts: xLabel, yLabel, yRightLabel — axis titles, always drawn.
           xMin/xMax  — pin the x range (slice charts pass the whole volume).
           yMin/yMax  — pin the y range; a yMin below 0 draws a zero line.
           A series carries axis:"right" for the second scale, scatter:true for points
           only, dash for a dashed stroke, wide for a heavier stroke.
     Hidden series are excluded from the ranges, so switching one off rescales what is
     left instead of leaving it squashed. */
  function chartSVG(series, opts) {
    const height = opts.height || 340;
    const visible = series.filter((s) => !s.hidden && s.data && s.data.length);
    const right = visible.filter((s) => s.axis === "right");
    const padR = right.length ? RIGHT_PAD : PAD.r;
    const pts = visible.flatMap((s) => s.data);
    if (!pts.length) return '<p class="gv-caption">nothing to plot yet</p>';

    const xs = pts.map((p) => p.x);
    const xMin = opts.xMin !== undefined ? opts.xMin : Math.min(...xs);
    let xMax = opts.xMax !== undefined ? opts.xMax : Math.max(...xs);
    if (xMax === xMin) xMax = xMin + 1;

    const left = visible.filter((s) => s.axis !== "right");
    const lys = left.flatMap((s) => s.data.map((p) => p.y));
    let yMin = opts.yMin !== undefined ? opts.yMin : 0;
    let yMax = opts.yMax !== undefined ? opts.yMax : 1;
    if (opts.yAuto && lys.length) {
      // Symmetric around zero for difference charts, so "better" and "worse" are the
      // same visual distance from the line.
      const m = Math.max(...lys.map(Math.abs), 1e-4) * 1.15;
      yMin = -m; yMax = m;
    }

    const plotW = VB_W - PAD.l - padR, plotH = height - PAD.t - PAD.b;
    const sx = (x) => PAD.l + ((x - xMin) / (xMax - xMin)) * plotW;
    const sy = (y) => PAD.t + plotH - ((y - yMin) / (yMax - yMin)) * plotH;
    const rMax = right.length
      ? Math.max(...right.flatMap((s) => s.data.map((p) => p.y)), 1) : 1;
    const syR = (y) => PAD.t + plotH - (y / rMax) * plotH;

    let svg = '<svg viewBox="0 0 ' + VB_W + " " + height + '" class="yolo-chart-svg '
      + 'it-chart-svg" preserveAspectRatio="xMidYMid meet">';

    // horizontal grid + left tick labels
    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
      const v = yMin + (i / yTicks) * (yMax - yMin), y = sy(v);
      svg += '<line x1="' + PAD.l + '" y1="' + y.toFixed(1) + '" x2="' + (PAD.l + plotW)
        + '" y2="' + y.toFixed(1) + '" class="yolo-chart-grid" />'
        + '<text x="' + (PAD.l - 10) + '" y="' + (y + 5).toFixed(1)
        + '" class="it-tick" text-anchor="end">'
        // `?? 2`, not `|| 2` — a voxel-count axis passes 0 decimals, which is falsy.
        + (Math.abs(v) < 1e-9 ? "0" : v.toFixed(opts.yDecimals ?? 2)) + "</text>";
    }
    if (yMin < 0 && yMax > 0) {
      svg += '<line x1="' + PAD.l + '" y1="' + sy(0).toFixed(1) + '" x2="'
        + (PAD.l + plotW) + '" y2="' + sy(0).toFixed(1) + '" class="it-zeroline" />';
    }

    // x tick labels
    const ticks = opts.xTickCount || 10;
    for (let i = 0; i <= ticks; i++) {
      const t = xMin + (i / ticks) * (xMax - xMin);
      svg += '<text x="' + sx(t).toFixed(1) + '" y="' + (PAD.t + plotH + 24).toFixed(1)
        + '" class="it-tick" text-anchor="middle">'
        + (opts.xInt === false ? t.toFixed(2) : Math.round(t)) + "</text>";
    }

    // axes
    svg += '<line x1="' + PAD.l + '" y1="' + PAD.t + '" x2="' + PAD.l + '" y2="'
      + (PAD.t + plotH) + '" class="yolo-chart-axis" />'
      + '<line x1="' + PAD.l + '" y1="' + (PAD.t + plotH) + '" x2="' + (PAD.l + plotW)
      + '" y2="' + (PAD.t + plotH) + '" class="yolo-chart-axis" />';

    // axis titles — the thing that says what the numbers mean
    svg += '<text x="' + (PAD.l + plotW / 2).toFixed(1) + '" y="' + (height - 16)
      + '" class="it-axis-title" text-anchor="middle">' + esc(opts.xLabel || "") + "</text>"
      + '<text transform="translate(26,' + (PAD.t + plotH / 2).toFixed(1)
      + ') rotate(-90)" class="it-axis-title" text-anchor="middle">'
      + esc(opts.yLabel || "") + "</text>";

    if (right.length) {
      const rc = right[0].color;
      for (let i = 0; i <= 2; i++) {
        const v = (i / 2) * rMax;
        svg += '<text x="' + (PAD.l + plotW + 12) + '" y="' + (syR(v) + 5).toFixed(1)
          + '" class="it-tick" text-anchor="start" style="fill:' + rc + '">'
          + Math.round(v).toLocaleString() + "</text>";
      }
      svg += '<line x1="' + (PAD.l + plotW) + '" y1="' + PAD.t + '" x2="'
        + (PAD.l + plotW) + '" y2="' + (PAD.t + plotH) + '" class="yolo-chart-axis" />'
        + '<text transform="translate(' + (VB_W - 14) + ","
        + (PAD.t + plotH / 2).toFixed(1) + ') rotate(-90)" class="it-axis-title" '
        + 'text-anchor="middle" style="fill:' + rc + '">'
        + esc(opts.yRightLabel || "") + "</text>";
    }

    (opts.markers || []).forEach((m) => {
      if (m.x < xMin || m.x > xMax) return;
      const x = sx(m.x);
      svg += '<line x1="' + x.toFixed(1) + '" y1="' + PAD.t + '" x2="' + x.toFixed(1)
        + '" y2="' + (PAD.t + plotH) + '" stroke="' + (m.color || "#8a8aa0")
        + '" stroke-width="1.2" stroke-dasharray="4 4" opacity="0.8" />';
    });

    visible.forEach((s) => {
      const scale = s.axis === "right" ? syR : sy;
      // A histogram series: one rect per bin, `barWidth` given in data units so the bars
      // stay aligned to the axis whatever the bin count.
      if (s.bars) {
        const w = Math.max(1, sx(xMin + (s.barWidth || 0.02)) - sx(xMin) - 1);
        const base = sy(Math.max(yMin, 0));
        s.data.forEach((p) => {
          const y = scale(p.y);
          svg += '<rect x="' + (sx(p.x) - w / 2).toFixed(1) + '" y="' + y.toFixed(1)
            + '" width="' + w.toFixed(1) + '" height="'
            + Math.max(0, base - y).toFixed(1) + '" fill="' + s.color
            + '" opacity="' + (s.fillOpacity || 0.75) + '" />';
          // The count above the bar, turned on its side: most of these bins are a few
          // pixels tall next to the spike at 1.0, and a bar you cannot see is a bar you
          // cannot read.
          if (s.labels && p.y > 0) {
            const lx = sx(p.x).toFixed(1), ly = (y - 6).toFixed(1);
            svg += '<text transform="translate(' + lx + "," + ly + ') rotate(-90)" '
              + 'class="it-tick" text-anchor="start" style="fill:' + s.color + '">'
              + p.y.toLocaleString() + "</text>";
          }
        });
        return;
      }
      if (!s.scatter) {
        const d = s.data.map((p, j) => (j === 0 ? "M" : "L") + " " + sx(p.x).toFixed(1)
          + " " + scale(p.y).toFixed(1)).join(" ");
        svg += '<path d="' + d + '" fill="none" stroke="' + s.color + '" stroke-width="'
          + (s.wide ? 3 : 2) + '" stroke-linejoin="round" '
          + (s.dash ? 'stroke-dasharray="7 4" ' : "") + "/>";
      }
      if (s.dots || s.scatter) {
        s.data.forEach((p) => {
          svg += '<circle cx="' + sx(p.x).toFixed(1) + '" cy="' + scale(p.y).toFixed(1)
            + '" r="' + (s.scatter ? 4 : 4.5) + '" fill="' + s.color + '" '
            + (s.scatter ? 'opacity="0.65" ' : "") + "/>";
        });
      }
    });
    return svg + "</svg>";
  }

  /* Draw into `container`, keeping the series list live so a legend click toggles one
     series and redraws (which also rescales the axes). */
  function renderChart(container, title, series, opts = {}) {
    const live = series.map((s) => Object.assign({}, s));
    function draw() {
      const legend = live.map((s, i) =>
        '<span class="yolo-chart-legend-item' + (s.hidden ? " gv-legend-off" : "")
        + '" data-idx="' + i + '" title="Click to show/hide">'
        + '<i style="background:' + s.color + '"></i>' + esc(s.label) + "</span>").join("");
      container.innerHTML = "<h4>" + esc(title) + "</h4>"
        + '<div class="yolo-chart-wrap">' + chartSVG(live, opts) + "</div>"
        + '<div class="yolo-chart-legend">' + legend + "</div>"
        + (opts.caption ? '<p class="gv-caption">' + opts.caption + "</p>" : "");
      container.querySelectorAll(".yolo-chart-legend-item").forEach((el) => {
        el.addEventListener("click", () => {
          live[Number(el.dataset.idx)].hidden = !live[Number(el.dataset.idx)].hidden;
          draw();
        });
      });
    }
    draw();
  }

  /* Horizontal bars — for comparing eleven scalars, where a line chart would imply an
     ordering between variants that does not exist. `diverging` puts the zero in the
     middle so gains and losses read as direction, not just length. */
  function barChart(container, title, rows, opts = {}) {
    const max = Math.max(...rows.map((r) => Math.abs(r.value || 0)), opts.min || 1e-4);
    const bars = rows.map((r) => {
      const v = r.value || 0, pct = ((Math.abs(v) / max) * 100).toFixed(1);
      const fill = '<span class="it-bar-fill" style="width:' + pct + "%;background:"
        + r.color + '"></span>';
      const track = opts.diverging
        ? '<span class="it-bar-track it-bar-diverging">'
          + '<span class="it-bar-half it-bar-neg">' + (v < 0 ? fill : "") + "</span>"
          + '<span class="it-bar-half it-bar-pos">' + (v >= 0 ? fill : "") + "</span>"
          + "</span>"
        : '<span class="it-bar-track">' + fill + "</span>";
      return '<div class="it-bar-row"><span class="it-bar-label">' + esc(r.label)
        + "</span>" + track + '<span class="it-bar-value' + (opts.diverging
          ? (v > 1e-6 ? " gv-pos" : v < -1e-6 ? " gv-neg" : "") : "") + '">'
        + (r.text !== undefined ? r.text : fmt(r.value)) + "</span></div>";
    }).join("");
    container.innerHTML = "<h4>" + esc(title) + "</h4>"
      + '<div class="it-bars">' + bars + "</div>"
      + '<p class="it-bar-axis">' + esc(opts.xLabel || "") + "</p>"
      + (opts.caption ? '<p class="gv-caption">' + opts.caption + "</p>" : "");
  }

  // ---------------------------------------------------------------- data
  const api = (path) => fetch(path).then((r) => r.json());
  const q = (extra) => {
    const p = new URLSearchParams();
    if (state.weights) p.set("weights", state.weights);
    Object.entries(extra || {}).forEach(([k, v]) => p.set(k, v));
    return p.toString();
  };

  function setStatus(text, cls) {
    els.status.textContent = text;
    els.status.className = "status" + (cls ? " " + cls : "");
  }

  async function loadVariants() {
    state.variants = await api("/inftest/api/variants");
    els.variantSel.innerHTML = state.variants
      .map((v) => '<option value="' + v.id + '">' + v.id + " · " + v.label + "</option>")
      .join("");
    els.variantSel.value = state.selected;
  }

  async function loadModels() {
    const list = await api("/inftest/api/weights");
    els.model.innerHTML = list
      .map((w) => '<option value="' + w.filename + '">' + w.label + "</option>").join("");
    state.weights = list.length ? list[0].filename : "";
    els.model.value = state.weights;
  }

  async function loadPatients() {
    state.patients = await api("/inftest/api/patients?" + q());
    renderPatientOptions();
  }

  function renderPatientOptions() {
    const term = (els.search.value || "").trim().toLowerCase();
    const shown = state.patients.filter((p) => !term
      || p.label.toLowerCase().includes(term));
    els.patient.innerHTML = shown.map((p) => {
      const tick = p.done === p.total ? "✓ " : p.done ? "· " : "";
      return '<option value="' + p.id + '">' + tick + p.label
        + (p.done ? " (" + p.done + "/" + p.total + ")" : "") + "</option>";
    }).join("");
    els.count.textContent = shown.length + " of " + state.patients.length
      + " test patients";
    if (shown.length) {
      const keep = shown.find((p) => p.id === state.id);
      els.patient.value = String(keep ? keep.id : shown[0].id);
    }
  }

  // ---------------------------------------------------------------- sweep
  async function runVariant(vid, force) {
    const r = await api("/inftest/api/run/" + state.id + "?"
      + q({ variant: vid, force: force ? "1" : "" }));
    if (r.error) throw new Error(vid + ": " + r.error);
    state.results[vid] = r;
    state.depth = r.depth;
    if (r.best_slice_index !== undefined) state.best = r.best_slice_index;
    return r;
  }

  function lockControls(on) {
    els.runAll.disabled = on;
    els.rerun.disabled = on;
    els.runAllPatients.disabled = on;
    els.stop.classList.toggle("hidden", !on);
  }

  async function sweep(force) {
    if (state.running) return;
    state.running = true;
    state.cancel = false;
    lockControls(true);
    const ids = state.variants.map((v) => v.id);
    try {
      for (let i = 0; i < ids.length; i++) {
        if (state.cancel) { setStatus("Stopped after " + i + " variant(s)."); break; }
        setStatus("Running " + ids[i] + " (" + (i + 1) + "/" + ids.length
          + ") — MedSAM2 propagation, this takes tens of seconds…", "gv-running");
        await runVariant(ids[i], force);
        renderVariants();
      }
      if (!state.cancel) setStatus("All " + ids.length + " variants done.");
    } catch (e) {
      setStatus(String(e.message || e), "gv-running");
    } finally {
      state.running = false;
      lockControls(false);
      loadPatients();
      syncSlice();
    }
  }

  // The cohort sweep. Deliberately never forces: it fills in what is missing, so it can
  // be stopped and pressed again tomorrow without throwing away a night of GPU work.
  // Patients already complete are skipped from the cached counts alone, with no request
  // at all — loading a patient's volumes to be told there is nothing to do costs seconds
  // each, and there are 324 of them.
  async function sweepAllPatients() {
    if (state.running) return;
    const todo = state.patients.filter((p) => p.done < p.total);
    if (!todo.length) {
      setStatus("Every patient already has every variant cached.");
      return;
    }
    const jobs = todo.reduce((n, p) => n + (p.total - p.done), 0);
    if (!confirm("Run the missing variants for " + todo.length + " patient(s) — "
      + jobs + " runs in all. This is hours of GPU work. It can be stopped at any "
      + "point and resumed later, keeping whatever finished.")) return;

    state.running = true;
    state.cancel = false;
    lockControls(true);
    let done = 0;
    try {
      for (let i = 0; i < todo.length; i++) {
        if (state.cancel) break;
        const p = todo[i];
        const st = await api("/inftest/api/status/" + p.id + "?" + q());
        const missing = state.variants.map((v) => v.id)
          .filter((id) => !(st.cached || []).includes(id));
        for (const vid of missing) {
          if (state.cancel) break;
          setStatus("Patient " + (i + 1) + "/" + todo.length + " · " + p.label
            + " — " + vid + " (" + (done + 1) + "/" + jobs + " runs)", "gv-running");
          const r = await api("/inftest/api/run/" + p.id + "?" + q({ variant: vid }));
          // One patient's failure must not end the sweep — record it and carry on.
          if (r.error) setStatus(p.label + " · " + vid + ": " + r.error, "gv-running");
          if (p.id === state.id && !r.error) state.results[vid] = r;
          done++;
        }
        if (p.id === state.id) renderVariants();
        await loadPatients();
      }
      setStatus(state.cancel
        ? "Stopped after " + done + " of " + jobs + " runs. Press again to resume."
        : "Cohort sweep finished: " + done + " runs over " + todo.length + " patient(s).");
    } catch (e) {
      setStatus(String(e.message || e), "gv-running");
    } finally {
      state.running = false;
      lockControls(false);
      await loadPatients();
      syncSlice();
    }
  }

  // ---------------------------------------------------------------- views
  function variantMeta(vid) {
    return state.variants.find((v) => v.id === vid) || { id: vid, label: vid };
  }

  function renderVariants() {
    const ref = state.results.O;
    const rows = state.variants.map((v) => {
      const r = state.results[v.id];
      if (!r) {
        return '<tr class="gv-row-empty">' + nameCell(v) + whatCell(v)
          + '<td colspan="5">not run</td></tr>';
      }
      const delta = ref ? r.final_dice - ref.final_dice : null;
      const empty = (r.empty_anchors || []).length;
      return "<tr" + (v.id === state.selected ? ' class="active-row"' : "") + ">"
        + nameCell(v) + whatCell(v)
        + "<td>" + fmt(r.final_dice) + "</td>"
        + "<td>" + (v.id === "O" ? "—" : signed(delta)) + "</td>"
        + "<td>" + r.anchors_used + " / " + r.anchors_available
        + (empty ? ' <span class="gv-anchor-tag" title="anchors where the prompt was '
          + 'empty or gated off">' + empty + " empty</span>" : "") + "</td>"
        + "<td>" + r.stop_reason + "</td>"
        + "<td>" + r.seconds.toFixed(1) + "s</td></tr>";
    }).join("");

    els.varTable.innerHTML = '<table class="yolo-summary-table"><thead><tr>'
      + "<th>Variant</th><th>What is different, and what it tests</th><th>Dice</th>"
      + "<th>&Delta; vs O</th><th>Anchors</th><th>Stopped because</th><th>Time</th>"
      + "</tr></thead><tbody>" + rows + "</tbody></table>";

    renderRoundChart();
    renderSliceChart();
    renderDeltaChart();
    renderVolumeChart();
  }

  const done = () => state.variants.filter((v) => state.results[v.id]);
  const sliceSpan = () => ({ xMin: 0, xMax: Math.max(1, state.depth - 1) });

  function renderRoundChart() {
    const series = done().map((v) => ({
      label: v.id, color: v.color, dots: true,
      wide: v.id === "O" || v.id === "GT", dash: v.id === "GT",
      data: (state.results[v.id].rounds || [])
        .map((r) => ({ x: r.anchors_used, y: r.dice })),
    }));
    if (!series.length) { els.roundChart.innerHTML = ""; return; }
    const maxA = Math.max(...series.flatMap((s) => s.data.map((p) => p.x)), 1);
    renderChart(els.roundChart, "Dice against number of anchors", series, {
      xLabel: "Anchors used (HITL rounds)", yLabel: "Dice (whole tumour, volume)",
      xMin: 1, xMax: maxA, xTickCount: maxA - 1, height: 340,
      caption: "Each point is one HITL round. Early-stopping variants end where their "
        + "curve stops; F and G run the whole budget, so their tails show exactly what "
        + "the stop rule would have discarded.",
    });
  }

  function renderSliceChart() {
    // Eleven curves over ~180 slices is unreadable, so only the reference, the ceiling
    // and the selected variant are on by default — the rest are one legend click away.
    const shown = new Set(["O", "GT", state.selected]);
    const series = done().map((v) => ({
      label: v.id, color: v.color, hidden: !shown.has(v.id),
      wide: v.id === "O", dash: v.id === "GT",
      data: (state.results[v.id].slices || []).filter((s) => s.dice !== null)
        .map((s) => ({ x: s.z, y: s.dice })),
    }));
    const ref = state.results.O;
    if (ref) {
      series.push({
        label: "GT tumour voxels", color: "#5a5a80", axis: "right",
        data: ref.slices.map((s) => ({ x: s.z, y: s.gt_voxels })),
      });
    }
    if (!series.length) { els.sliceChart.innerHTML = ""; return; }
    renderChart(els.sliceChart, "Dice slice by slice", series, Object.assign({
      xLabel: "Slice z (whole volume, 0 to " + Math.max(0, state.depth - 1) + ")",
      yLabel: "Dice on that slice", yRightLabel: "GT tumour voxels",
      xTickCount: 12, height: 360,
      markers: (ref ? ref.schedule : []).map((z) => ({ x: z, color: "#8a8aa0" })),
      caption: "Dashed verticals are the anchor slices. The x axis spans the whole volume, "
        + "so a gap means neither the ground truth nor that variant had anything on those "
        + "slices — nothing to score.",
    }, sliceSpan()));
  }

  function renderDeltaChart() {
    const ref = state.results.O;
    if (!ref) { els.deltaChart.innerHTML = ""; return; }
    const refBy = new Map(ref.slices.map((s) => [s.z, s.dice]));
    const shown = new Set(["GT", state.selected === "O" ? "G" : state.selected]);
    const series = done().filter((v) => v.id !== "O").map((v) => ({
      label: v.id, color: v.color, hidden: !shown.has(v.id), dash: v.id === "GT",
      data: (state.results[v.id].slices || []).filter(
        (s) => s.dice !== null && refBy.get(s.z) !== null
              && refBy.get(s.z) !== undefined)
        .map((s) => ({ x: s.z, y: s.dice - refBy.get(s.z) })),
    })).filter((s) => s.data.length);
    if (!series.length) { els.deltaChart.innerHTML = ""; return; }
    renderChart(els.deltaChart, "Where each variant gains or loses against the reference",
      series, Object.assign({
        xLabel: "Slice z (whole volume)",
        yLabel: "Dice difference vs O  (+ better, − worse)",
        yAuto: true, yDecimals: 3, xTickCount: 12, height: 340,
        markers: ref.schedule.map((z) => ({ x: z, color: "#8a8aa0" })),
        caption: "Above the zero line the variant beats the reference on that slice, below "
          + "it loses. A volume-level Dice difference of near zero can still hide large "
          + "gains and losses that cancel out — this is where you see them.",
      }, sliceSpan()));
  }

  function renderVolumeChart() {
    const ref = state.results.O;
    if (!ref) { els.volumeChart.innerHTML = ""; return; }
    const shown = new Set(["GT", state.selected]);
    const series = [{
      label: "ground truth", color: "#26ff26", wide: true,
      data: ref.slices.map((s) => ({ x: s.z, y: s.gt_voxels })),
    }].concat(done().map((v) => ({
      label: v.id, color: v.color, hidden: !shown.has(v.id) && v.id !== "O",
      data: (state.results[v.id].slices || []).map((s) => ({ x: s.z, y: s.voxels })),
    })));
    const peak = Math.max(...series.flatMap((s) => s.data.map((p) => p.y)), 1);
    renderChart(els.volumeChart, "Predicted tumour area against the truth, slice by slice",
      series, Object.assign({
        xLabel: "Slice z (whole volume)", yLabel: "Tumour voxels on that slice",
        yMin: 0, yMax: peak, yDecimals: 0, xTickCount: 12, height: 340,
        markers: ref.schedule.map((z) => ({ x: z, color: "#8a8aa0" })),
        caption: "A curve above the green line is over-segmenting that slice, below it is "
          + "under-segmenting. Dice alone cannot tell you which way a variant is wrong.",
      }, sliceSpan()));
  }

  function syncSlice() {
    els.z.max = String(Math.max(0, state.depth - 1));
    // Land on the fullest tumour slice rather than z=0, which is empty skull on every
    // patient. `zTouched` keeps a slice the user picked from being snapped away.
    if (!state.zTouched || Number(els.z.value) > state.depth - 1) {
      els.z.value = String(state.best);
      els.zLabel.textContent = "z = " + els.z.value;
    }
    if (!state.results[state.selected]) {
      els.viewer.removeAttribute("src");
      return;
    }
    els.viewer.src = "/inftest/segment.png?"
      + q({ id: state.id, z: els.z.value, variant: state.selected })
      + "&_=" + Date.now();
  }

  // ---------------------------------------------------------------- probe
  async function loadProbe(force) {
    setStatus("Running the YOLO-only confidence probe…", "gv-running");
    const p = await api("/inftest/api/probe/" + state.id + "?"
      + q({ force: force ? "1" : "" }));
    if (p.error) { setStatus(p.error, "gv-running"); return; }
    setStatus("Probe done.");

    const rows = p.rows.filter((r) => r.conf !== null);
    const depth = state.depth || Math.max(...p.rows.map((r) => r.z), 1) + 1;
    renderChart(els.probeChart, "YOLO confidence and YOLO Dice, slice by slice", [
      { label: "confidence (area-weighted)", color: "#ffd23f",
        data: rows.map((r) => ({ x: r.z, y: r.conf })) },
      { label: "YOLO Dice", color: "#ff5a26",
        data: rows.map((r) => ({ x: r.z, y: r.dice })) },
      { label: "GT tumour voxels", color: "#5a5a80", axis: "right",
        data: p.rows.map((r) => ({ x: r.z, y: r.gt_voxels })) },
    ], {
      xLabel: "Slice z (whole volume, 0 to " + (depth - 1) + ")",
      yLabel: "Confidence / Dice  (0 to 1)", yRightLabel: "GT tumour voxels",
      xMin: 0, xMax: depth - 1, xTickCount: 12, height: 340,
      caption: "Where the two lines move together, confidence is telling the truth about "
        + "quality. Slices where YOLO found nothing have no confidence and are dropped.",
    });

    /* One cloud per granularity, each scored against the Dice of the thing its number
       describes — a slice for B and C2, a single lump for C1. Same axes, so the three
       are read against each other: whichever tilts upwards most steeply is the level at
       which YOLO's confidence actually knows something. */
    /* B and C1 only. Both plot the same number — YOLO's detection score — at two zoom
       levels, so they share an axis honestly. C2's number comes from a different head
       entirely (per-pixel mask probability, saturated near 1.0) and belongs in the
       histogram below, not overlaid on a detection-score axis. */
    const insts = (p.instance_rows || []);
    const scatterSeries = [
      { label: "B · slice confidence (per slice)", color: "#4ade80", scatter: true,
        data: rows.map((r) => ({ x: r.conf, y: r.dice })) },
    ];
    if (insts.length) {
      scatterSeries.push({
        label: "C1 · lump confidence (per lump)", color: "#c084fc", scatter: true,
        data: insts.map((r) => ({ x: r.conf, y: r.dice })),
      });
    }
    renderChart(els.probeScatter, "Detection confidence against Dice · B and C1",
      scatterSeries, {
        xLabel: "YOLO detection confidence  ·  B: the slice's area-weighted average  "
          + "·  C1: that one lump's own score",
        yLabel: "YOLO Dice  (of the slice, or of that lump for C1)",
        xMin: 0, xMax: 1, xInt: false, xTickCount: 10, height: 360,
        caption: "B is literally the area-weighted mean of the C1 points on the same "
          + "slice, so on a slice with one lump they are the same point. A cloud with no "
          + "upward tilt means confidence does not predict quality at that granularity. "
          + "C1's points are lumps, not slices, so a small lump can score 0 while the "
          + "slice around it does well.",
      });

    /* C2's actual material: every per-pixel probability in the patient, not the per-slice
       average the scatter shows. The scatter can only say where a slice's average landed;
       this says what the values it averaged look like. */
    const hist = p.prob_hist;
    if (hist && hist.edges) {
      const w = hist.edges[1] - hist.edges[0];
      const mid = hist.inside.map((_, i) => (hist.edges[i] + hist.edges[i + 1]) / 2);
      /* One series, because C2 does not split these: the whole map is handed over and
         MedSAM2 applies the 0.5 cut itself. Splitting by YOLO's own mask would draw a
         line the variant never sees. */
      const total = hist.inside.map((v, i) => v + hist.fringe[i]);
      const peak = Math.max(...total, 1);
      renderChart(els.probeHist, "C2 · every per-pixel probability YOLO produced", [
        { label: "voxels handed to MedSAM2", color: "#f472b6",
          bars: true, barWidth: w, labels: true,
          data: mid.map((x, i) => ({ x, y: total[i] })) },
      ], {
        xLabel: "Per-pixel probability", yLabel: "Voxels in that bin (whole patient)",
        xMin: 0, xMax: 1, xInt: false, xTickCount: 10, yMin: 0,
        // Headroom for the count printed above the tallest bar.
        yMax: peak * 1.18,
        yDecimals: 0, height: 380,
        caption: "Every voxel carrying a value, whichever side of 0.5 it falls — C2 sends "
          + "the lot and lets MedSAM2 decide, so the mask is not drawn here. Voxels "
          + "outside every detection box are exactly 0 and are left out; they would be one "
          + "bar many times taller than the chart. Read the counts: nearly everything "
          + "piles up against 1.0 or against 0, so C2's \"per-pixel confidence\" is close "
          + "to binary already, with far less gradation to prompt with than the idea "
          + "suggests.",
      });
    }

    // Binned means make the trend legible where the raw cloud does not.
    const bins = [];
    for (let lo = 0.2; lo < 1.0; lo += 0.1) {
      const inBin = rows.filter((r) => r.conf >= lo && r.conf < lo + 0.1);
      if (!inBin.length) continue;
      bins.push({
        label: lo.toFixed(1) + "–" + (lo + 0.1).toFixed(1) + "  (n=" + inBin.length + ")",
        color: "#c084fc",
        value: inBin.reduce((a, r) => a + r.dice, 0) / inBin.length,
      });
    }
    barChart(els.probeBins, "Mean YOLO Dice per confidence band", bins, {
      xLabel: "bar length = mean Dice in that confidence band (0 to 1)",
      caption: "If these rise from top to bottom, confidence is calibrated against "
        + "quality on this patient. `n` is how many slices fell in the band.",
    });

    const flat = (v) => (v === null || v === undefined ? "—" : v.toFixed(3));
    const g = p.granularity || {};
    const gChip = (id, what) => {
      const c = g[id];
      if (!c) return "";
      return '<span class="gv-chip" title="' + what + ' — Pearson is the straight-line '
        + 'correlation, Spearman the rank one: does more confidence mean better, '
        + 'whatever the shape">' + id + ' <b>r=' + flat(c.pearson) + "</b> ρ="
        + flat(c.spearman) + " <i>n=" + c.n + "</i></span>";
    };
    els.probeStats.innerHTML = '<div class="gv-chips">'
      + '<span class="gv-chip">scored slices <b>' + p.n_scored + "</b></span>"
      + gChip("B", "One confidence per slice, against that slice's Dice")
      + gChip("C1", "One confidence per detected lump, against that lump's own Dice")
      + gChip("C2", "Mean per-pixel probability inside the mask, against the slice's Dice")
      + '<span class="gv-chip">detection threshold <b>' + p.conf_threshold
      + "</b></span></div>";
  }

  // ---------------------------------------------------------------- aggregate
  async function loadAggregate() {
    setStatus("Reading every cached result…", "gv-running");
    const a = await api("/inftest/api/aggregate?" + q());
    state.aggregate = a;
    setStatus(a.n_patients_complete + " patient(s) have all variants; "
      + a.n_patients_any + " have at least one.");

    const rows = a.variants.map((v) => {
      const meta = variantMeta(v.id);
      return "<tr>" + nameCell(Object.assign({}, meta, v)) + whatCell(v)
        + "<td>" + v.n + "</td><td>" + fmt(v.mean_dice) + "</td>"
        + "<td>" + fmt(v.median_dice) + "</td><td>" + fmt(v.std_dice) + "</td>"
        + "<td>" + (v.id === "O" ? "—" : signed(v.mean_delta_vs_ref)) + "</td>"
        + "<td>" + (v.id === "O" ? "—" : v.wins_vs_ref + " / " + v.losses_vs_ref)
        + "</td><td>" + fmt(v.mean_anchors, 1) + "</td>"
        + "<td>" + fmt(v.mean_seconds, 1) + "s</td></tr>";
    }).join("");
    els.aggTable.innerHTML = '<table class="yolo-summary-table"><thead><tr>'
      + "<th>Variant</th><th>What is different, and what it tests</th>"
      + "<th>n</th><th>Mean Dice</th>"
      + "<th>Median</th><th>SD</th><th>&Delta; vs O</th><th>Win/Loss</th>"
      + "<th>Anchors</th><th>Time</th></tr></thead><tbody>" + rows + "</tbody></table>";

    const scored = a.variants.filter((v) => v.n > 0);
    barChart(els.aggChart, "Mean Dice over the patients that ran every variant",
      scored.map((v) => ({
        label: v.id + " · " + v.label, color: variantMeta(v.id).color,
        value: v.mean_dice, text: fmt(v.mean_dice),
      })), {
        xLabel: "bar length = mean volume Dice (0 to 1)",
        caption: "Compared on the same patients only — a patient is left out of every "
          + "column until it has finished all eleven variants.",
      });

    barChart(els.aggDelta, "Mean change against the reference",
      scored.filter((v) => v.id !== "O").map((v) => ({
        label: v.id + " · " + v.label, color: variantMeta(v.id).color,
        value: v.mean_delta_vs_ref,
        text: (v.mean_delta_vs_ref >= 0 ? "+" : "") + fmt(v.mean_delta_vs_ref),
      })), {
        diverging: true,
        xLabel: "bar direction = mean Dice change vs O  (right better, left worse)",
        caption: "The mean can hide a variant that wins big on a few patients and loses "
          + "on the rest — read it next to the Win/Loss column and the spread below.",
      });

    barChart(els.aggAnchors, "Mean anchors used",
      scored.map((v) => ({
        label: v.id + " · " + v.label, color: variantMeta(v.id).color,
        value: v.mean_anchors, text: fmt(v.mean_anchors, 1),
      })), {
        xLabel: "bar length = mean number of anchors before that variant stopped",
        caption: "A confidence variant that also ran more anchors than the reference has "
          + "two things changing at once — its Dice gain is not purely the prompt.",
      });

    // Spread: one point per patient, sorted so the hard patients sit on the left. The
    // mean says which variant is better on average; this says whether it is ever worse.
    const complete = a.patients.filter((p) => p.complete && p.dice.O !== undefined)
      .sort((x, y) => x.dice.O - y.dice.O);
    if (complete.length) {
      const shown = new Set(["GT", "G", "F"]);
      const series = a.variants.filter((v) => v.n > 0 && v.id !== "O").map((v) => ({
        label: v.id, color: variantMeta(v.id).color, hidden: !shown.has(v.id),
        dots: true, dash: v.id === "GT",
        data: complete.map((p, i) => ({ x: i, y: (p.dice[v.id] ?? 0) - p.dice.O })),
      }));
      renderChart(els.aggSpread, "Per-patient change against the reference", series, {
        xLabel: "Patient (sorted by reference Dice, hardest on the left)",
        yLabel: "Dice change vs O  (+ better, − worse)",
        yAuto: true, yDecimals: 3, xMin: 0, xMax: Math.max(1, complete.length - 1),
        xTickCount: Math.min(complete.length - 1, 20), height: 340,
        caption: "Every point above zero is a patient the variant improved. A variant "
          + "whose points all sit above the line is a safe change; one that scatters "
          + "either side is a gamble that happens to average out. Patients, left to "
          + "right: " + complete.map((p) => esc(p.patient_id.replace("BraTS-GLI-", "")))
            .join(", ") + ".",
      });
    }
  }

  function showView(name) {
    state.view = name;
    els.views.querySelectorAll(".tab").forEach((b) =>
      b.classList.toggle("active", b.dataset.iview === name));
    els.variantsPanel.classList.toggle("hidden", name !== "variants");
    els.slicePanel.classList.toggle("hidden", name !== "slice");
    els.probePanel.classList.toggle("hidden", name !== "probe");
    els.aggPanel.classList.toggle("hidden", name !== "aggregate");
    els.sliceField.classList.toggle("hidden", name !== "slice");
    if (name === "slice") syncSlice();
    if (name === "probe" && !els.probeStats.innerHTML) loadProbe(false);
    if (name === "aggregate") loadAggregate();
  }

  // ---------------------------------------------------------------- wiring
  async function selectPatient(id) {
    state.id = id;
    state.results = {};
    state.zTouched = false;
    [els.varTable, els.roundChart, els.sliceChart, els.deltaChart, els.volumeChart,
     els.probeStats, els.probeChart, els.probeScatter, els.probeHist, els.probeBins]
      .forEach((el) => { el.innerHTML = ""; });
    const status = await api("/inftest/api/status/" + id + "?" + q());
    const cached = status.cached || [];
    setStatus(cached.length
      ? "Loading " + cached.length + " cached variant(s)…"
      : "Nothing cached for this patient — press Run all variants.");
    for (const vid of cached) {
      try { await runVariant(vid, false); } catch (e) { /* reported below */ }
    }
    renderVariants();
    syncSlice();
    if (cached.length) {
      setStatus(cached.length + " of " + state.variants.length
        + " variants cached." + (cached.length < state.variants.length
          ? " Press Run all variants for the rest." : ""));
    }
  }

  els.patient.addEventListener("change", () => selectPatient(Number(els.patient.value)));
  els.search.addEventListener("input", renderPatientOptions);
  els.model.addEventListener("change", async () => {
    state.weights = els.model.value;
    await loadPatients();
    selectPatient(Number(els.patient.value));
  });
  els.runAll.addEventListener("click", () => sweep(false));
  els.runAllPatients.addEventListener("click", sweepAllPatients);
  els.rerun.addEventListener("click", () => {
    if (confirm("Discard the cached results for this patient and run all "
      + state.variants.length + " variants again? That is several minutes of GPU work."))
      sweep(true);
  });
  els.stop.addEventListener("click", () => {
    state.cancel = true;
    setStatus("Stopping after the current variant…", "gv-running");
  });
  els.views.addEventListener("click", (e) => {
    const b = e.target.closest(".tab");
    if (b) showView(b.dataset.iview);
  });
  els.variantSel.addEventListener("change", () => {
    state.selected = els.variantSel.value;
    renderVariants();
    syncSlice();
  });
  els.z.addEventListener("input", () => {
    state.zTouched = true;
    els.zLabel.textContent = "z = " + els.z.value;
    syncSlice();
  });
  els.best.addEventListener("click", () => {
    state.zTouched = false;
    els.z.value = String(state.best);
    els.zLabel.textContent = "z = " + els.z.value;
    syncSlice();
  });
  els.aggRefresh.addEventListener("click", () => loadAggregate());

  let started = false;
  window.initInferenceTest = async function () {
    if (started) return;
    started = true;
    await loadVariants();
    await loadModels();
    await loadPatients();
    showView("variants");
    if (els.patient.value) selectPatient(Number(els.patient.value));
  };
})();
