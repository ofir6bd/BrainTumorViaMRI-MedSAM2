/* Charts — the one chart library every page uses (plain SVG, no dependencies).
 *
 * Every chart: redraws on resize and on theme change, names both axes, has a hover tooltip,
 * and a small toolbar (PNG / SVG / CSV of exactly the data drawn). Colours may be CSS
 * variables ("var(--yolo)"); they are resolved when drawing, so exports keep them.
 *
 *   Charts.line(el, {series:[{name, color, points:[[x,y]], dash, dots, area, axis:"right", step}],
 *                    xLabel, yLabel, y2Label, yMin, yMax, logY, markers:[{x, label, color}],
 *                    bands:[{x0, x1, color, label}], hlines:[{y, label, color}], sync:"group",
 *                    zoom:true, onClick(x), xFmt, yFmt, height})
 *   Charts.bars(el, {cats, series:[{name, color, values, err:[[lo,hi]], stripes}], stacked, horizontal,
 *                    yLabel, xLabel, onClick(ci, si), fmt, showValues, yMin, yMax, height})
 *   Charts.hist(el, {series:[{name, color, values}], bins, log, xLabel, yLabel, brush, range,
 *                    onBrush(lo, hi | null), density, markers, height})
 *   Charts.scatter(el, {points:[{x, y, color, ring, r, id, tip}], xLabel, yLabel, logX, logY, xMin, xMax,
 *                    yMin, yMax, diagonal, trend, legend, onClick(p), onSelect(box|null, pts), box,
 *                    selected:Set, highlight, height})
 *   Charts.box(el, {groups:[{name, color, ring, values:[{v, id, tip}]}], yLabel, log, onClick(p), height})
 *   Charts.heat(el, {rows, cols, values, min, max, diverging, fmt, onClick(i, j), tip(i, j), height})
 *   Charts.ecdf(el, {series:[{name, color, values}], xLabel, log, height})
 */
(function () {
  "use strict";
  const K = window.Kit;
  const NS = "http://www.w3.org/2000/svg";
  const C = K.color;
  const Ch = {};

  // ------------------------------------------------------------------ svg + scale helpers
  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v !== undefined && v !== null && v !== false) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  }
  function text(parent, x, y, s, attrs) {
    const t = el("text", { x, y, fill: C("var(--muted)"), "font-size": 11, ...(attrs || {}) }, parent);
    t.textContent = s;
    return t;
  }
  function lin(d0, d1, r0, r1) {
    const k = (r1 - r0) / ((d1 - d0) || 1);
    const s = (v) => r0 + (v - d0) * k;
    s.invert = (p) => d0 + (p - r0) / k;
    s.d = [d0, d1];
    return s;
  }
  // log(1 + v): keeps zeros, spreads the small end — the dataset has many 0 mL labels
  const LOG = { f: (v) => Math.log10(1 + Math.max(0, v)), inv: (t) => Math.pow(10, t) - 1 };
  const ID = { f: (v) => v, inv: (v) => v };
  function niceTicks(lo, hi, n = 5) {
    const span = hi - lo || 1;
    const step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
    const err = span / n / step0;
    const step = step0 * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
    const out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function logTicks(lo, hi) {  // lo/hi in raw units
    const out = [];
    if (lo <= 0) out.push(0);
    const floor = hi < 3 ? 0.01 : 1;
    for (let p = -2; Math.pow(10, p) <= hi * 1.001; p++) for (const m of [1, 3]) {
      const v = m * Math.pow(10, p);
      if (v >= floor && v >= lo && v <= hi * 1.001) out.push(+v.toPrecision(3));
    }
    return out.length > 8 ? out.filter((v, i) => v === 0 || i % 2 === 0) : out;
  }
  const tfmt = (v) => {
    const a = Math.abs(v);
    if (a >= 10000) return `${+(v / 1000).toPrecision(3)}k`;
    if (a >= 100 || Number.isInteger(v)) return String(Math.round(v * 100) / 100);
    return String(+v.toPrecision(3));
  };
  function extent(vals, pad = 0) {
    const v = vals.filter((x) => x != null && Number.isFinite(x));
    if (!v.length) return [0, 1];
    let lo = Math.min(...v), hi = Math.max(...v);
    if (lo === hi) { lo -= 0.5; hi += 0.5; }
    const p = (hi - lo) * pad;
    return [lo - p, hi + p];
  }

  // ------------------------------------------------------------------ mount: container, tools, resize, theme
  function mount(container, opts, draw) {
    container.classList.add("chart");
    container.innerHTML = "";
    const state = { opts, zoom: null, hidden: new Set((opts.series || []).filter((s) => s.hidden).map((s) => s.name)) };
    const legendEl = document.createElement("div");
    legendEl.className = "legend";
    const holder = document.createElement("div");
    const tools = K.h(`<div class="chart-tools" role="toolbar" aria-label="Chart tools">
        <button class="small ghost" data-x="png" title="Save as picture (PNG)">PNG</button>
        <button class="small ghost" data-x="svg" title="Save as SVG">SVG</button>
        <button class="small ghost" data-x="csv" title="Save the data drawn here (CSV)">CSV</button></div>`);
    container.append(tools, legendEl, holder);
    const handle = {
      container, state, holder, legendEl,
      redraw() {
        const w = Math.max(260, container.clientWidth || 600);
        holder.innerHTML = "";
        state.w = w;
        state.rows = [];  // CSV rows collected while drawing: [{...}]
        if (!draw(handle, w)) holder.innerHTML = `<div class="chart-empty">${K.esc(opts.empty || "Nothing to show yet.")}</div>`;
      },
      svg: () => K.$("svg", holder),
      update(newOpts) { Object.assign(opts, newOpts); handle.redraw(); },
    };
    tools.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-x]");
      if (!b) return;
      const name = (opts.exportName || opts.yLabel || "chart").replace(/[^\w.-]+/g, "_").slice(0, 60);
      if (b.dataset.x === "csv") {
        const rows = state.rows || [];
        if (!rows.length) return K.toast("This chart has no table data.");
        const keys = [...new Set(rows.flatMap((r) => Object.keys(r)))];
        K.download(`${name}.csv`, K.csv(rows, keys.map((k) => ({ k, label: k }))), "text/csv");
      } else exportSvg(handle.svg(), name, b.dataset.x);
    });
    let lastW = 0;
    const ro = new ResizeObserver(() => {
      const w = container.clientWidth;
      if (Math.abs(w - lastW) > 4) { lastW = w; handle.redraw(); }
    });
    ro.observe(container);
    window.addEventListener("themechange", () => handle.redraw());
    lastW = container.clientWidth;
    handle.redraw();
    return handle;
  }

  function exportSvg(svg, name, kind) {
    if (!svg) return;
    const clone = svg.cloneNode(true);
    clone.setAttribute("xmlns", NS);
    const bg = C("var(--panel)");
    const r = el("rect", { x: 0, y: 0, width: svg.getAttribute("width"), height: svg.getAttribute("height"), fill: bg });
    clone.insertBefore(r, clone.firstChild);
    const src = new XMLSerializer().serializeToString(clone);
    if (kind === "svg") return K.download(`${name}.svg`, src, "image/svg+xml");
    const img = new Image();
    const scale = 2;
    img.onload = () => {
      const cv = document.createElement("canvas");
      cv.width = svg.width.baseVal.value * scale;
      cv.height = svg.height.baseVal.value * scale;
      const ctx = cv.getContext("2d");
      ctx.scale(scale, scale);
      ctx.drawImage(img, 0, 0);
      cv.toBlob((b) => K.download(`${name}.png`, b));
    };
    img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(src)}`;
  }

  function legend(handle, items, onToggle) {
    const L = handle.legendEl;
    if (!items || !items.length) { L.innerHTML = ""; return; }
    L.innerHTML = items.map((it) => `<span data-n="${K.esc(it.name)}" class="${handle.state.hidden.has(it.name) ? "off" : ""}"
      title="Click to hide / show">${it.kind === "pt" ? `<i class="pt" style="${it.ring ? `border:2px solid ${C(it.color)};background:none` : `background:${C(it.color)}`}"></i>`
      : it.kind === "box" ? `<i class="box" style="background:${C(it.color)}"></i>` : it.dash ? `<i style="width:18px;background:repeating-linear-gradient(90deg,${C(it.color)} 0 3px,transparent 3px 6px)"></i>`
      : `<i style="background:${C(it.color)}"></i>`}${K.esc(it.name)}</span>`).join("");
    if (onToggle) L.onclick = (e) => {
      const s = e.target.closest("span[data-n]");
      if (!s) return;
      const n = s.dataset.n;
      handle.state.hidden.has(n) ? handle.state.hidden.delete(n) : handle.state.hidden.add(n);
      onToggle();
    };
  }

  // axes: bottom + left (+ optional right). b = {L, R, T, B, W, H}
  function frame(svg, b, x, y, o) {
    const g = el("g", {}, svg);
    const yt = o.yTicks || niceTicks(y.d[0], y.d[1], Math.max(3, Math.round((b.H - b.T - b.B) / 55)));
    for (const t of yt) {
      const py = y(o.yT ? o.yT.f(t) : t);
      if (py < b.T - 1 || py > b.H - b.B + 1) continue;
      el("line", { x1: b.L, x2: b.W - b.R, y1: py, y2: py, stroke: C("var(--grid)") }, g);
      text(g, b.L - 6, py + 4, (o.yFmt || tfmt)(t), { "text-anchor": "end" });
    }
    const xt = o.xTicks || niceTicks(x.d[0], x.d[1], Math.max(3, Math.round((b.W - b.L - b.R) / 80)));
    for (const t of xt) {
      const px = x(o.xT ? o.xT.f(t) : t);
      if (px < b.L - 1 || px > b.W - b.R + 1) continue;
      if (o.xGrid) el("line", { x1: px, x2: px, y1: b.T, y2: b.H - b.B, stroke: C("var(--grid)") }, g);
      text(g, px, b.H - b.B + 15, (o.xFmt || tfmt)(t), { "text-anchor": "middle" });
    }
    el("line", { x1: b.L, x2: b.W - b.R, y1: b.H - b.B, y2: b.H - b.B, stroke: C("var(--axis)") }, g);
    if (o.xLabel) text(g, b.L + (b.W - b.L - b.R) / 2, b.H - 4, o.xLabel, { "text-anchor": "middle", "font-weight": 600 });
    if (o.yLabel) {
      const cy = b.T + (b.H - b.B - b.T) / 2;
      text(g, 11, cy, o.yLabel, { "text-anchor": "middle", "font-weight": 600, transform: `rotate(-90 11 ${cy})` });
    }
    return g;
  }
  function root(handle, w, h, label) {
    return el("svg", { width: w, height: h, viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": label || "chart" }, handle.holder);
  }
  function pointer(svg, e) {
    const r = svg.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  // ------------------------------------------------------------------ crosshair sync between charts
  const SYNC = {};
  Ch.sync = (group, x, from) => { for (const h of SYNC[group] || []) if (h !== from && h.cross) h.cross(x); };

  // ------------------------------------------------------------------ LINE
  Ch.line = (container, opts) => mount(container, opts, (H, w) => {
    const o = H.state.opts;
    const series = (o.series || []).filter((s) => s.points && s.points.length);
    legend(H, o.legend === false ? [] : series.map((s) => ({ name: s.name, color: s.color, dash: s.dash })), () => H.redraw());
    const vis = series.filter((s) => !H.state.hidden.has(s.name));
    if (!series.length) return false;
    const h = o.height || 260;
    const hasRight = vis.some((s) => s.axis === "right");
    const b = { L: 54, R: hasRight ? 54 : 16, T: 10, B: o.xLabel ? 36 : 22, W: w, H: h };
    const yT = o.logY ? LOG : ID;
    const all = vis.flatMap((s) => s.points);
    let [x0, x1] = o.xDomain || extent(all.map((p) => p[0]));
    if (H.state.zoom) [x0, x1] = H.state.zoom;
    const leftPts = vis.filter((s) => s.axis !== "right").flatMap((s) => s.points).filter((p) => p[0] >= x0 && p[0] <= x1);
    let [y0, y1] = extent(leftPts.map((p) => yT.f(p[1])), 0.06);
    if (o.yMin != null) y0 = yT.f(o.yMin);
    if (o.yMax != null) y1 = yT.f(o.yMax);
    for (const hl of o.hlines || []) { y0 = Math.min(y0, yT.f(hl.y)); y1 = Math.max(y1, yT.f(hl.y)); }
    const x = lin(x0, x1, b.L, w - b.R), y = lin(y0, y1, h - b.B, b.T);
    const rightPts = vis.filter((s) => s.axis === "right").flatMap((s) => s.points);
    const [r0, r1] = extent(rightPts.map((p) => p[1]), 0.06);
    const y2 = lin(o.y2Min != null ? o.y2Min : Math.min(0, r0), r1, h - b.B, b.T);
    const svg = root(H, w, h, o.yLabel);
    const clipId = K.uid("clip");
    el("rect", { x: b.L, y: b.T, width: w - b.L - b.R, height: h - b.T - b.B }, el("clipPath", { id: clipId }, el("defs", {}, svg)));
    for (const band of o.bands || []) {
      const bx0 = x(Math.max(band.x0, x0)), bx1 = x(Math.min(band.x1, x1));
      if (bx1 > bx0) {
        el("rect", { x: bx0, y: b.T, width: bx1 - bx0, height: h - b.T - b.B, fill: C(band.color || "var(--accent)"), "fill-opacity": 0.08 }, svg);
        if (band.label) text(svg, bx0 + 4, b.T + 12, band.label, { "font-size": 10 });
      }
    }
    frame(svg, b, x, y, { ...o, yT, yTicks: o.logY ? logTicks(yT.inv(y0), yT.inv(y1)) : null, xGrid: false });
    if (hasRight) {
      for (const t of niceTicks(y2.d[0], y2.d[1], 4)) text(svg, w - b.R + 6, y2(t) + 4, tfmt(t), { "text-anchor": "start" });
      if (o.y2Label) { const cy = b.T + (h - b.B - b.T) / 2; text(svg, w - 10, cy, o.y2Label, { "text-anchor": "middle", "font-weight": 600, transform: `rotate(90 ${w - 10} ${cy})` }); }
    }
    for (const hl of o.hlines || []) {
      const py = y(yT.f(hl.y));
      el("line", { x1: b.L, x2: w - b.R, y1: py, y2: py, stroke: C(hl.color || "var(--faint)"), "stroke-dasharray": "5 4" }, svg);
      if (hl.label) text(svg, w - b.R - 4, py - 4, hl.label, { "text-anchor": "end", "font-size": 10 });
    }
    const plot = el("g", { "clip-path": `url(#${clipId})` }, svg);
    for (const s of vis) {
      const ys = s.axis === "right" ? (v) => y2(v) : (v) => y(yT.f(v));
      const pts = s.points.filter((p) => p[1] != null && Number.isFinite(p[1]));
      let d = "";
      pts.forEach((p, i) => {
        const px = x(p[0]), py = ys(p[1]);
        if (s.step && i) d += `H${px}V${py}`; else d += `${i ? "L" : "M"}${px},${py}`;
      });
      if (s.area && pts.length) {
        el("path", { d: `${d}L${x(pts[pts.length - 1][0])},${h - b.B}L${x(pts[0][0])},${h - b.B}Z`, fill: C(s.color), "fill-opacity": 0.12 }, plot);
      }
      el("path", { d, fill: "none", stroke: C(s.color), "stroke-width": s.width || 2, "stroke-dasharray": s.dash || null,
                   "stroke-linejoin": "round" }, plot);
      if (s.dots || pts.length < 30) for (const p of pts) el("circle", { cx: x(p[0]), cy: ys(p[1]), r: s.dots ? 3 : 2, fill: C(s.color) }, plot);
      for (const p of pts) H.state.rows.push({ series: s.name, x: p[0], y: p[1] });
    }
    (o.markers || []).forEach((m, mi) => {
      if (m.x < x0 || m.x > x1) return;
      const px = x(m.x);
      el("line", { x1: px, x2: px, y1: b.T, y2: h - b.B, stroke: C(m.color || "var(--faint)"), "stroke-dasharray": m.solid ? null : "4 3", "stroke-width": 1.4 }, svg);
      if (m.label) text(svg, px + 3, b.T + 10 + (mi % 3) * 12, m.label, { "font-size": 10, fill: C(m.color || "var(--muted)") });
    });
    // hover crosshair + tooltip, optional sync, click, brush-zoom
    const cross = el("line", { y1: b.T, y2: h - b.B, stroke: C("var(--text)"), "stroke-opacity": 0.35, visibility: "hidden" }, svg);
    const hit = el("rect", { x: b.L, y: b.T, width: w - b.L - b.R, height: h - b.T - b.B, fill: "transparent" }, svg);
    const allX = [...new Set(vis.flatMap((s) => s.points.map((p) => p[0])))].sort((a, c) => a - c);
    const nearestX = (vx) => allX.reduce((best, v) => (Math.abs(v - vx) < Math.abs(best - vx) ? v : best), allX[0]);
    H.cross = (vx) => {
      if (vx == null || vx < x0 || vx > x1) { cross.setAttribute("visibility", "hidden"); return; }
      cross.setAttribute("x1", x(vx)); cross.setAttribute("x2", x(vx)); cross.setAttribute("visibility", "visible");
    };
    if (o.sync) { SYNC[o.sync] = (SYNC[o.sync] || []).filter((c) => c !== H && c.container.isConnected); SYNC[o.sync].push(H); }
    let drag = null;
    const brush = el("rect", { y: b.T, height: h - b.T - b.B, class: "brush", fill: C("var(--accent)"), "fill-opacity": 0.14, visibility: "hidden" }, svg);
    hit.addEventListener("mousemove", (e) => {
      const [px] = pointer(svg, e);
      const vx = nearestX(x.invert(px));
      H.cross(vx);
      if (o.sync) Ch.sync(o.sync, vx, H);
      if (drag) {
        brush.setAttribute("x", Math.min(drag, px)); brush.setAttribute("width", Math.abs(px - drag)); brush.setAttribute("visibility", "visible");
      }
      const rows = vis.map((s) => {
        const p = s.points.find((q) => q[0] === vx);
        return p && p[1] != null ? K.tipRow(C(s.color), s.name, (o.yFmt || K.fmt)(p[1])) : "";
      }).join("");
      K.tip(`${K.tipTitle(`${o.xName || o.xLabel || "x"}: ${(o.xFmt || K.fmt)(vx)}`)}${rows}${o.tipExtra ? o.tipExtra(vx) : ""}`, e);
    });
    hit.addEventListener("mouseleave", () => { K.untip(); H.cross(null); if (o.sync) Ch.sync(o.sync, null, H); });
    hit.addEventListener("mousedown", (e) => { if (o.zoom !== false) { drag = pointer(svg, e)[0]; e.preventDefault(); } });
    hit.addEventListener("mouseup", (e) => {
      const [px] = pointer(svg, e);
      if (drag != null && Math.abs(px - drag) > 8) {
        const a = x.invert(Math.min(drag, px)), c = x.invert(Math.max(drag, px));
        H.state.zoom = [a, c];
        drag = null;
        H.redraw();
        return;
      }
      drag = null;
      brush.setAttribute("visibility", "hidden");
      if (o.onClick) o.onClick(nearestX(x.invert(px)));
    });
    if (o.onClick) hit.style.cursor = "pointer";
    if (H.state.zoom) {
      const btn = K.h(`<button class="small zoom-reset">Reset zoom</button>`);
      btn.addEventListener("click", () => { H.state.zoom = null; H.redraw(); });
      H.holder.appendChild(btn);
    }
    return true;
  });

  // ------------------------------------------------------------------ BARS
  Ch.bars = (container, opts) => mount(container, opts, (H, w) => {
    const o = H.state.opts;
    const cats = o.cats || [];
    const series = o.series || [];
    if (!cats.length || !series.length) return false;
    legend(H, series.length > 1 || o.legend ? series.map((s) => ({ name: s.name, color: s.color, kind: "box" })) : [], () => H.redraw());
    const vis = series.filter((s) => !H.state.hidden.has(s.name));
    const horiz = !!o.horizontal;
    const catW = horiz ? 0 : Math.max(...cats.map((c) => String(c).length)) * 6.2;
    const rotate = !horiz && catW > (w - 80) / cats.length;
    const h = o.height || (horiz ? Math.max(120, cats.length * (vis.length > 1 ? 34 : 26) + 50) : 250);
    const labW = horiz ? Math.min(220, Math.max(...cats.map((c) => String(c).length)) * 6.4 + 12) : 0;
    const b = { L: horiz ? labW : 54, R: horiz ? 60 : 16, T: 10, B: horiz ? (o.xLabel || o.yLabel ? 38 : 22) : (rotate ? 70 : o.xLabel ? 40 : 26), W: w, H: h };
    const tickFmt = o.tickFmt || o.fmt || null;
    const totals = cats.map((_, ci) => (o.stacked ? vis.reduce((s, se) => s + (se.values[ci] || 0), 0) : Math.max(0, ...vis.map((se) => (se.err && se.err[ci] ? se.err[ci][1] : se.values[ci] || 0)))));
    const lows = cats.map((_, ci) => Math.min(0, ...vis.map((se) => se.values[ci] || 0)));
    const vmax = o.yMax != null ? o.yMax : Math.max(...totals) * 1.08 || 1;
    const vmin = o.yMin != null ? o.yMin : Math.min(0, ...lows);
    const svg = root(H, w, h, o.yLabel);
    const defs = el("defs", {}, svg);
    const fillOf = (s) => {
      if (!s.stripes) return C(s.color);
      const id = K.uid("st");
      const p = el("pattern", { id, patternUnits: "userSpaceOnUse", width: 6, height: 6, patternTransform: "rotate(45)" }, defs);
      el("rect", { width: 6, height: 6, fill: C(s.color), "fill-opacity": 0.22 }, p);
      el("rect", { width: 2.6, height: 6, fill: C(s.color) }, p);
      return `url(#${id})`;
    };
    const fills = vis.map(fillOf);
    const fmtV = o.fmt || K.fmt;
    if (horiz) {
      const x = lin(vmin, vmax, b.L, w - b.R);
      const band = (h - b.T - b.B) / cats.length;
      frame(svg, b, x, lin(0, 1, h - b.B, b.T), { ...o, yTicks: [], xGrid: true, xLabel: o.yLabel || o.xLabel, yLabel: null, xFmt: tickFmt });
      cats.forEach((c, ci) => {
        text(svg, b.L - 6, b.T + band * (ci + 0.5) + 4, String(c), { "text-anchor": "end", fill: C("var(--text)") });
        let acc = 0;
        vis.forEach((s, si) => {
          const v = s.values[ci];
          if (v == null) return;
          const bh = o.stacked ? band * 0.62 : (band * 0.7) / vis.length;
          const by = b.T + band * ci + band * 0.15 + (o.stacked ? 0 : si * bh);
          const xa = x(o.stacked ? acc : Math.min(0, v)), xb = x(o.stacked ? acc + v : Math.max(0, v));
          const r = el("rect", { x: xa, y: by, width: Math.max(0.5, xb - xa), height: bh - 1, rx: 2, fill: fills[si], class: "hot" }, svg);
          acc += o.stacked ? v : 0;
          if (o.showValues !== false && !o.stacked) text(svg, xb + 4, by + bh / 2 + 4, fmtV(v), { "font-size": 10 });
          K.hover(r, () => `${K.tipTitle(String(c))}${K.tipRow(C(s.color), s.name, fmtV(v))}${s.err && s.err[ci] ? K.tipRow(C("var(--faint)"), "95% range", `${fmtV(s.err[ci][0])} – ${fmtV(s.err[ci][1])}`) : ""}${s.tips ? s.tips[ci] || "" : ""}`);
          if (o.onClick) r.addEventListener("click", () => o.onClick(ci, si));
          if (s.err && s.err[ci]) {
            const cy = by + bh / 2;
            el("line", { x1: x(s.err[ci][0]), x2: x(s.err[ci][1]), y1: cy, y2: cy, stroke: C("var(--text)"), "stroke-width": 1.2 }, svg);
          }
          H.state.rows.push({ category: c, series: s.name, value: v });
        });
      });
      return true;
    }
    const y = lin(vmin, vmax, h - b.B, b.T);
    const band = (w - b.L - b.R) / cats.length;
    frame(svg, b, lin(0, 1, b.L, w - b.R), y, { ...o, xTicks: [], yFmt: tickFmt });
    cats.forEach((c, ci) => {
      const cx = b.L + band * (ci + 0.5);
      if (rotate) text(svg, cx, h - b.B + 12, String(c), { "text-anchor": "end", transform: `rotate(-38 ${cx} ${h - b.B + 12})` });
      else text(svg, cx, h - b.B + 15, String(c), { "text-anchor": "middle" });
      let acc = 0;
      vis.forEach((s, si) => {
        const v = s.values[ci];
        if (v == null) return;
        const bw = o.stacked ? band * 0.62 : (band * 0.72) / vis.length;
        const bx = b.L + band * ci + band * (o.stacked ? 0.19 : 0.14) + (o.stacked ? 0 : si * bw);
        const ya = y(o.stacked ? acc + v : Math.max(0, v)), yb = y(o.stacked ? acc : Math.min(0, v));
        const r = el("rect", { x: bx, y: ya, width: Math.max(1, bw - 2), height: Math.max(0.5, yb - ya), rx: 2, fill: fills[si], class: "hot",
                               stroke: s.outline ? C(s.color) : null }, svg);
        acc += o.stacked ? v : 0;
        if (s.err && s.err[ci]) {
          const ex = bx + (bw - 2) / 2;
          el("line", { x1: ex, x2: ex, y1: y(s.err[ci][0]), y2: y(s.err[ci][1]), stroke: C("var(--text)"), "stroke-width": 1.2 }, svg);
          for (const e2 of s.err[ci]) el("line", { x1: ex - 4, x2: ex + 4, y1: y(e2), y2: y(e2), stroke: C("var(--text)"), "stroke-width": 1.2 }, svg);
        }
        if (o.showValues && !o.stacked && bw > 22) text(svg, bx + (bw - 2) / 2, ya - 4, fmtV(v), { "text-anchor": "middle", "font-size": 10 });
        K.hover(r, () => `${K.tipTitle(String(c))}${K.tipRow(C(s.color), s.name, fmtV(v))}${s.err && s.err[ci] ? K.tipRow(C("var(--faint)"), "95% range", `${fmtV(s.err[ci][0])} – ${fmtV(s.err[ci][1])}`) : ""}${s.tips ? s.tips[ci] || "" : ""}`);
        if (o.onClick) r.addEventListener("click", () => o.onClick(ci, si));
        H.state.rows.push({ category: c, series: s.name, value: v });
      });
    });
    for (const hl of o.hlines || []) {
      const py = y(hl.y);
      el("line", { x1: b.L, x2: w - b.R, y1: py, y2: py, stroke: C(hl.color || "var(--faint)"), "stroke-dasharray": "5 4" }, svg);
      if (hl.label) text(svg, w - b.R - 4, py - 4, hl.label, { "text-anchor": "end", "font-size": 10 });
    }
    return true;
  });

  // ------------------------------------------------------------------ HISTOGRAM (with brush)
  Ch.hist = (container, opts) => mount(container, opts, (H, w) => {
    const o = H.state.opts;
    const series = (o.series || []).filter((s) => s.values && s.values.some((v) => v != null));
    if (!series.length) return false;
    legend(H, series.length > 1 ? series.map((s) => ({ name: s.name, color: s.color, kind: "box" })) : [], () => H.redraw());
    const vis = series.filter((s) => !H.state.hidden.has(s.name));
    const T = o.log ? LOG : ID;
    const all = (o.domainValues || vis.flatMap((s) => s.values)).filter((v) => v != null && Number.isFinite(v)).map(T.f);
    let [lo, hi] = o.domain ? o.domain.map(T.f) : extent(all);
    if (!o.domain && lo === hi) hi = lo + 1;
    const nb = o.bins || 30;
    const step = (hi - lo) / nb;
    const counts = vis.map((s) => {
      const c = new Array(nb).fill(0);
      for (const v0 of s.values) {
        if (v0 == null || !Number.isFinite(v0)) continue;
        const i = K.clamp(Math.floor((T.f(v0) - lo) / step), 0, nb - 1);
        c[i] += 1;
      }
      return o.density ? c.map((v) => v / (s.values.filter((x) => x != null).length || 1)) : c;
    });
    const h = o.height || 220;
    const b = { L: 50, R: 14, T: 10, B: o.xLabel ? 38 : 24, W: w, H: h };
    const ymax = Math.max(1e-9, ...counts.flat()) * 1.08;
    const x = lin(lo, hi, b.L, w - b.R), y = lin(0, ymax, h - b.B, b.T);
    const svg = root(H, w, h, o.xLabel);
    frame(svg, b, x, y, { ...o, xT: T, xTicks: o.log ? logTicks(T.inv(lo), T.inv(hi)) : null, yLabel: o.yLabel || (o.density ? "Share" : "Count"),
                          yFmt: o.density ? (v) => K.pct(v, 0) : null });
    const sel = o.range ? [T.f(o.range[0]), T.f(o.range[1])] : null;
    counts.forEach((c, si) => {
      const s = vis[si];
      c.forEach((v, i) => {
        if (!v) return;
        const a = lo + i * step, z = a + step;
        const inSel = !sel || (z > sel[0] && a < sel[1]);
        const bw = (x(z) - x(a)) / (vis.length > 1 && !o.overlay ? vis.length : 1);
        const bx = x(a) + (vis.length > 1 && !o.overlay ? si * bw : 0);
        const r = el("rect", { x: bx + 0.5, y: y(v), width: Math.max(1, bw - 1), height: h - b.B - y(v), fill: C(s.color),
                               "fill-opacity": (inSel ? 0.85 : 0.22) * (o.overlay && vis.length > 1 ? 0.6 : 1), class: "hot" }, svg);
        K.hover(r, () => `${K.tipTitle(`${K.fmt(T.inv(a))} – ${K.fmt(T.inv(z))}`)}${K.tipRow(C(s.color), s.name, o.density ? K.pct(v) : K.int(v))}`);
        if (o.onBarClick) r.addEventListener("click", () => o.onBarClick(T.inv(a), T.inv(z), s));
        H.state.rows.push({ series: s.name, from: T.inv(a), to: T.inv(z), value: v });
      });
    });
    (o.markers || []).forEach((m, mi) => {
      const px = x(T.f(m.x));
      if (px < b.L || px > w - b.R) return;
      el("line", { x1: px, x2: px, y1: b.T, y2: h - b.B, stroke: C(m.color || "var(--text)"), "stroke-dasharray": "4 3", "stroke-width": 1.5 }, svg);
      if (m.label) text(svg, px + 3, b.T + 10 + mi * 12, m.label, { "font-size": 10, fill: C(m.color || "var(--muted)") });
    });
    if (sel) {
      el("rect", { x: x(Math.max(lo, sel[0])), y: b.T, width: Math.max(1, x(Math.min(hi, sel[1])) - x(Math.max(lo, sel[0]))), height: h - b.T - b.B,
                   fill: "none", stroke: C("var(--accent)"), "stroke-dasharray": "4 3" }, svg);
    }
    if (o.brush) {
      const hit = el("rect", { x: b.L, y: b.T, width: w - b.L - b.R, height: h - b.T - b.B, fill: "transparent", style: "cursor:crosshair" }, svg);
      const br = el("rect", { y: b.T, height: h - b.T - b.B, fill: C("var(--accent)"), "fill-opacity": 0.15, visibility: "hidden" }, svg);
      let d0 = null;
      hit.addEventListener("mousedown", (e) => { d0 = pointer(svg, e)[0]; e.preventDefault(); });
      hit.addEventListener("mousemove", (e) => {
        const px = pointer(svg, e)[0];
        const v = T.inv(x.invert(px));
        const bi = K.clamp(Math.floor((x.invert(px) - lo) / step), 0, nb - 1);
        K.tip(`${K.tipTitle(`${K.fmt(T.inv(lo + bi * step))} – ${K.fmt(T.inv(lo + (bi + 1) * step))}`)}${vis.map((s, si) => K.tipRow(C(s.color), s.name,
          o.density ? K.pct(counts[si][bi]) : K.int(counts[si][bi]))).join("")}<div class="faint" style="margin-top:4px">Drag to select a range · double-click to clear</div>`, e);
        if (d0 != null) { br.setAttribute("x", Math.min(d0, px)); br.setAttribute("width", Math.abs(px - d0)); br.setAttribute("visibility", "visible"); }
        void v;
      });
      hit.addEventListener("mouseleave", K.untip);
      hit.addEventListener("mouseup", (e) => {
        const px = pointer(svg, e)[0];
        if (d0 != null && Math.abs(px - d0) > 5) {
          o.onBrush && o.onBrush(T.inv(x.invert(Math.min(d0, px))), T.inv(x.invert(Math.max(d0, px))));
        } else if (d0 != null && o.onBarClick) {
          const bi = K.clamp(Math.floor((x.invert(px) - lo) / step), 0, nb - 1);
          o.onBarClick(T.inv(lo + bi * step), T.inv(lo + (bi + 1) * step));
        }
        d0 = null;
        br.setAttribute("visibility", "hidden");
      });
      hit.addEventListener("dblclick", () => o.onBrush && o.onBrush(null));
    }
    return true;
  });

  // ------------------------------------------------------------------ SCATTER (box select, click, trend, diagonal)
  Ch.scatter = (container, opts) => mount(container, opts, (H, w) => {
    const o = H.state.opts;
    const pts0 = (o.points || []).filter((p) => p.x != null && p.y != null && Number.isFinite(p.x) && Number.isFinite(p.y));
    legend(H, (o.legend || []).map((l) => ({ ...l, kind: "pt" })), () => H.redraw());
    const pts = pts0.filter((p) => !p.group || !H.state.hidden.has(p.group));
    if (!pts0.length) return false;
    const XT = o.logX ? LOG : ID, YT = o.logY ? LOG : ID;
    const h = o.height || 320;
    const b = { L: 56, R: 16, T: 12, B: o.xLabel ? 40 : 24, W: w, H: h };
    let [x0, x1] = extent(pts0.map((p) => XT.f(p.x)), 0.04);
    let [y0, y1] = extent(pts0.map((p) => YT.f(p.y)), 0.05);
    if (o.xMin != null) x0 = XT.f(o.xMin); if (o.xMax != null) x1 = XT.f(o.xMax);
    if (o.yMin != null) y0 = YT.f(o.yMin); if (o.yMax != null) y1 = YT.f(o.yMax);
    if (H.state.zoom) [x0, x1, y0, y1] = H.state.zoom;
    const x = lin(x0, x1, b.L, w - b.R), y = lin(y0, y1, h - b.B, b.T);
    const svg = root(H, w, h, o.yLabel);
    frame(svg, b, x, y, { ...o, xT: XT, yT: YT, xGrid: true,
                          xTicks: o.logX ? logTicks(XT.inv(x0), XT.inv(x1)) : null, yTicks: o.logY ? logTicks(YT.inv(y0), YT.inv(y1)) : null });
    const clipId = K.uid("clip");
    el("rect", { x: b.L, y: b.T, width: w - b.L - b.R, height: h - b.T - b.B }, el("clipPath", { id: clipId }, el("defs", {}, svg)));
    const g = el("g", { "clip-path": `url(#${clipId})` }, svg);
    if (o.diagonal) {
      const lo = Math.max(x0, y0), hi = Math.min(x1, y1);
      el("line", { x1: x(lo), y1: y(lo), x2: x(hi), y2: y(hi), stroke: C("var(--faint)"), "stroke-dasharray": "5 4" }, g);
    }
    for (const hl of o.hlines || []) el("line", { x1: b.L, x2: w - b.R, y1: y(YT.f(hl.y)), y2: y(YT.f(hl.y)), stroke: C(hl.color || "var(--faint)"), "stroke-dasharray": "5 4" }, g);
    for (const vl of o.vlines || []) el("line", { x1: x(XT.f(vl.x)), x2: x(XT.f(vl.x)), y1: b.T, y2: h - b.B, stroke: C(vl.color || "var(--faint)"), "stroke-dasharray": "5 4" }, g);
    if (o.trend && pts.length > 2) {
      const lr = K.stats.linreg(pts.map((p) => XT.f(p.x)), pts.map((p) => YT.f(p.y)));
      if (lr) el("line", { x1: x(x0), y1: y(lr.slope * x0 + lr.intercept), x2: x(x1), y2: y(lr.slope * x1 + lr.intercept),
                           stroke: C("var(--text)"), "stroke-opacity": 0.5, "stroke-width": 1.5 }, g);
    }
    const sel = o.selected;
    const anySel = sel && sel.size;
    const drawn = [];
    for (const p of pts) {
      const on = !anySel || sel.has(p.id);
      const cx = x(XT.f(p.x)), cy = y(YT.f(p.y));
      const hl = o.highlight && p.id === o.highlight;
      const c = el("circle", { cx, cy, r: hl ? 7 : p.r || 3.6,
        fill: p.ring ? "none" : C(p.color || "var(--accent)"), stroke: p.ring || hl ? C(hl ? "var(--text)" : p.color) : null, "stroke-width": p.ring ? 1.6 : hl ? 2 : null,
        "fill-opacity": on ? 0.78 : 0.12, "stroke-opacity": on ? 1 : 0.2, class: "hot" }, g);
      c.__p = p;
      drawn.push([cx, cy, p]);
      H.state.rows.push({ id: p.id, x: p.x, y: p.y, group: p.group });
    }
    if (o.box) {
      const [bx0, bx1] = [x(XT.f(o.box.x0)), x(XT.f(o.box.x1))], [by0, by1] = [y(YT.f(o.box.y1)), y(YT.f(o.box.y0))];
      el("rect", { x: bx0, y: by0, width: bx1 - bx0, height: by1 - by0, fill: C("var(--accent)"), "fill-opacity": 0.08, stroke: C("var(--accent)"), "stroke-dasharray": "4 3" }, svg);
    }
    // interaction layer: nearest point hover, click, drag-box select, shift+drag zoom
    const hit = el("rect", { x: b.L, y: b.T, width: w - b.L - b.R, height: h - b.T - b.B, fill: "transparent" }, svg);
    const br = el("rect", { fill: C("var(--accent)"), "fill-opacity": 0.12, stroke: C("var(--accent)"), "stroke-dasharray": "3 3", visibility: "hidden" }, svg);
    const nearest = (px, py) => {
      let best = null, bd = 144;
      for (const [cx, cy, p] of drawn) { const d = (cx - px) ** 2 + (cy - py) ** 2; if (d < bd) { bd = d; best = p; } }
      return best;
    };
    let d0 = null;
    hit.addEventListener("mousedown", (e) => { d0 = pointer(svg, e); e.preventDefault(); });
    hit.addEventListener("mousemove", (e) => {
      const [px, py] = pointer(svg, e);
      if (d0) {
        br.setAttribute("x", Math.min(d0[0], px)); br.setAttribute("y", Math.min(d0[1], py));
        br.setAttribute("width", Math.abs(px - d0[0])); br.setAttribute("height", Math.abs(py - d0[1])); br.setAttribute("visibility", "visible");
        return;
      }
      const p = nearest(px, py);
      hit.style.cursor = p && o.onClick ? "pointer" : "crosshair";
      if (p) K.tip(p.tip || `${K.tipTitle(p.id || "")}${K.tipRow(C(p.color), o.xLabel || "x", K.fmt(p.x))}${K.tipRow(C(p.color), o.yLabel || "y", K.fmt(p.y))}`, e);
      else K.untip();
      if (o.onHover) o.onHover(p);
    });
    hit.addEventListener("mouseleave", () => { K.untip(); if (o.onHover) o.onHover(null); });
    hit.addEventListener("mouseup", (e) => {
      const [px, py] = pointer(svg, e);
      if (d0 && (Math.abs(px - d0[0]) > 6 || Math.abs(py - d0[1]) > 6)) {
        const X0 = XT.inv(x.invert(Math.min(d0[0], px))), X1 = XT.inv(x.invert(Math.max(d0[0], px)));
        const Y0 = YT.inv(y.invert(Math.max(d0[1], py))), Y1 = YT.inv(y.invert(Math.min(d0[1], py)));
        if (e.shiftKey || !o.onSelect) {
          H.state.zoom = [XT.f(X0), XT.f(X1), YT.f(Y0), YT.f(Y1)];
          d0 = null;
          H.redraw();
          return;
        }
        const inside = pts.filter((p) => p.x >= X0 && p.x <= X1 && p.y >= Y0 && p.y <= Y1);
        o.onSelect({ x0: X0, x1: X1, y0: Y0, y1: Y1 }, inside);
      } else {
        const p = nearest(px, py);
        if (p && o.onClick) o.onClick(p);
      }
      d0 = null;
      br.setAttribute("visibility", "hidden");
    });
    hit.addEventListener("dblclick", () => { if (H.state.zoom) { H.state.zoom = null; H.redraw(); } else if (o.onSelect) o.onSelect(null, []); });
    if (o.stat !== false && pts.length > 2) {
      const rho = K.stats.spearman(pts.map((p) => p.x), pts.map((p) => p.y));
      if (rho != null) text(svg, w - b.R - 4, b.T + 12, `Spearman ρ = ${rho.toFixed(2)} · n = ${pts.length}`, { "text-anchor": "end", "font-size": 10 });
    }
    if (H.state.zoom) {
      const btn = K.h(`<button class="small zoom-reset">Reset zoom</button>`);
      btn.addEventListener("click", () => { H.state.zoom = null; H.redraw(); });
      H.holder.appendChild(btn);
    }
    return true;
  });

  // ------------------------------------------------------------------ BOX + STRIP
  Ch.box = (container, opts) => mount(container, opts, (H, w) => {
    const o = H.state.opts;
    const groups = (o.groups || []).filter((g) => g.values && g.values.length);
    if (!groups.length) return false;
    const T = o.log ? LOG : ID;
    const h = o.height || 300;
    const b = { L: 56, R: 12, T: 10, B: 40, W: w, H: h };
    const all = groups.flatMap((g) => g.values.map((v) => T.f(v.v)));
    const [lo, hi] = extent(all, 0.04);
    const y = lin(o.yMin != null ? T.f(o.yMin) : lo, o.yMax != null ? T.f(o.yMax) : hi, h - b.B, b.T);
    const band = (w - b.L - b.R) / groups.length;
    const svg = root(H, w, h, o.yLabel);
    frame(svg, b, lin(0, 1, b.L, w - b.R), y, { ...o, yT: T, xTicks: [], yTicks: o.log ? logTicks(T.inv(y.d[0]), T.inv(y.d[1])) : null, xLabel: null });
    const hit = [];
    groups.forEach((g, gi) => {
      const cx = b.L + band * (gi + 0.5);
      text(svg, cx, h - b.B + 15, g.name, { "text-anchor": "middle", fill: C("var(--text)") });
      text(svg, cx, h - b.B + 28, `n = ${g.values.length}`, { "text-anchor": "middle", "font-size": 10 });
      const vals = g.values.map((v) => v.v).filter((v) => v != null).sort((a, c) => a - c);
      const q = (p) => K.stats.quantile(vals, p);
      const [q1, md, q3] = [q(0.25), q(0.5), q(0.75)];
      const iqr = q3 - q1;
      const wlo = vals.find((v) => v >= q1 - 1.5 * iqr), whi = [...vals].reverse().find((v) => v <= q3 + 1.5 * iqr);
      const bw = Math.min(70, band * 0.42);
      const col = C(g.color);
      el("line", { x1: cx, x2: cx, y1: y(T.f(wlo)), y2: y(T.f(whi)), stroke: col, "stroke-width": 1.3 }, svg);
      const bx = el("rect", { x: cx - bw / 2, y: y(T.f(q3)), width: bw, height: Math.max(1, y(T.f(q1)) - y(T.f(q3))),
        fill: g.ring ? "none" : col, "fill-opacity": 0.18, stroke: col, "stroke-width": 1.5, rx: 3 }, svg);
      el("line", { x1: cx - bw / 2, x2: cx + bw / 2, y1: y(T.f(md)), y2: y(T.f(md)), stroke: col, "stroke-width": 2.5 }, svg);
      const mean = K.stats.mean(vals);
      el("path", { d: `M${cx - 4},${y(T.f(mean))}h8M${cx},${y(T.f(mean)) - 4}v8`, stroke: C("var(--text)"), "stroke-width": 1.3 }, svg);
      K.hover(bx, () => `${K.tipTitle(g.name)}${K.tipRow(col, "median", K.fmt(md))}${K.tipRow(col, "middle half", `${K.fmt(q1)} – ${K.fmt(q3)}`)}${K.tipRow(col, "mean (+)", K.fmt(mean))}${K.tipRow(col, "patients", vals.length)}`);
      if (o.strip !== false) {
        for (const v of g.values) {
          if (v.v == null) continue;
          const jit = (hash01(v.id || String(v.v)) - 0.5) * bw * 1.35;
          const c = el("circle", { cx: cx + jit, cy: y(T.f(v.v)), r: 2.3, fill: g.ring ? "none" : col, stroke: g.ring ? col : null, "fill-opacity": 0.5, "stroke-opacity": 0.8, class: "hot" }, svg);
          hit.push([c, v, g]);
          H.state.rows.push({ group: g.name, id: v.id, value: v.v });
        }
      }
    });
    for (const [c, v, g] of hit) {
      K.hover(c, () => v.tip || `${K.tipTitle(v.id || "")}${K.tipRow(C(g.color), g.name, K.fmt(v.v))}`);
      if (o.onClick) c.addEventListener("click", () => o.onClick(v, g));
    }
    for (const hl of o.hlines || []) {
      const py = y(T.f(hl.y));
      el("line", { x1: b.L, x2: w - b.R, y1: py, y2: py, stroke: C(hl.color || "var(--faint)"), "stroke-dasharray": "5 4" }, svg);
      if (hl.label) text(svg, w - b.R - 4, py - 4, hl.label, { "text-anchor": "end", "font-size": 10 });
    }
    return true;
  });
  function hash01(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
    return ((h >>> 0) % 10000) / 10000;
  }

  // ------------------------------------------------------------------ HEATMAP
  Ch.heat = (container, opts) => mount(container, opts, (H, w) => {
    const o = H.state.opts;
    const R = o.rows || [], Cc = o.cols || [], V = o.values || [];
    if (!R.length || !Cc.length) return false;
    const labW = Math.min(170, Math.max(...R.map((r) => String(r).length)) * 6.4 + 10);
    const colLab = Math.max(...Cc.map((c) => String(c).length)) * 6;
    const rotate = o.rotateCols !== false && colLab > (w - labW) / Cc.length - 4;
    const cell = Math.max(12, Math.min(o.cell || 60, (w - labW - 10) / Cc.length));
    const top = rotate ? Math.min(120, colLab * 0.75 + 12) : 22;
    const h = o.height || top + cell * R.length + 30;
    const cellH = (h - top - 30) / R.length;
    const flat = V.flat().filter((v) => v != null && Number.isFinite(v));
    const lo = o.min != null ? o.min : Math.min(...flat), hi = o.max != null ? o.max : Math.max(...flat);
    const svg = root(H, w, h, o.title);
    const col = (v) => {
      if (v == null || !Number.isFinite(v)) return C("var(--panel-2)");
      if (o.diverging) {
        const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
        const t = K.clamp(v / m, -1, 1);
        return mix(C("var(--panel-2)"), C(t >= 0 ? (o.posColor || "var(--yolo)") : (o.negColor || "var(--medsam2)")), Math.abs(t));
      }
      const t = hi > lo ? K.clamp((v - lo) / (hi - lo), 0, 1) : 0;
      return mix(C("var(--panel-2)"), C(o.color || "var(--accent)"), 0.08 + 0.92 * t);
    };
    Cc.forEach((c, j) => {
      const cx = labW + cell * (j + 0.5);
      if (rotate) text(svg, cx, top - 6, String(c), { "text-anchor": "start", transform: `rotate(-45 ${cx} ${top - 6})`, "font-size": 10 });
      else text(svg, cx, top - 7, String(c), { "text-anchor": "middle", "font-size": 10 });
    });
    R.forEach((r, i) => {
      text(svg, labW - 6, top + cellH * (i + 0.5) + 4, String(r), { "text-anchor": "end", fill: C("var(--text)") });
      Cc.forEach((c, j) => {
        const v = V[i] ? V[i][j] : null;
        const rect = el("rect", { x: labW + cell * j + 1, y: top + cellH * i + 1, width: cell - 2, height: cellH - 2, rx: 3, fill: col(v), class: "hot",
          stroke: o.selected && o.selected[0] === i && o.selected[1] === j ? C("var(--text)") : null, "stroke-width": 2 }, svg);
        if (o.showValues !== false && cell > 30 && cellH > 16 && v != null) {
          text(svg, labW + cell * (j + 0.5), top + cellH * (i + 0.5) + 4, (o.fmt || K.fmt)(v),
               { "text-anchor": "middle", "font-size": 10, fill: ink(col(v)) });
        }
        K.hover(rect, () => (o.tip ? o.tip(i, j) : `${K.tipTitle(`${r} × ${c}`)}${K.tipRow(col(v), "value", (o.fmt || K.fmt)(v))}`));
        if (o.onClick) rect.addEventListener("click", () => o.onClick(i, j));
        H.state.rows.push({ row: r, col: c, value: v });
      });
    });
    // colour key
    const kx = labW, ky = h - 16, kw = Math.min(220, cell * Cc.length);
    const gid = K.uid("g");
    const grad = el("linearGradient", { id: gid }, el("defs", {}, svg));
    const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
    for (let t = 0; t <= 1.0001; t += 0.25) el("stop", { offset: `${t * 100}%`, "stop-color": col(o.diverging ? -m + 2 * m * t : lo + (hi - lo) * t) }, grad);
    el("rect", { x: kx, y: ky, width: kw, height: 7, fill: `url(#${gid})`, rx: 2 }, svg);
    text(svg, kx, ky - 3, (o.fmt || K.fmt)(o.diverging ? -Math.max(Math.abs(lo), Math.abs(hi)) : lo), { "font-size": 10 });
    text(svg, kx + kw, ky - 3, (o.fmt || K.fmt)(o.diverging ? Math.max(Math.abs(lo), Math.abs(hi)) : hi), { "font-size": 10, "text-anchor": "end" });
    return true;
  });
  // dark or light text, whichever reads better on this fill
  function ink(fill) {
    const m = /rgb\((\d+),(\d+),(\d+)\)/.exec(fill || "");
    if (!m) return C("var(--text)");
    const [r, g, b] = m.slice(1).map((v) => +v / 255);
    return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 ? "#12131f" : "#f4f5fb";
  }
  function mix(a, b, t) {
    const p = (c) => { c = c.replace("#", ""); if (c.length === 3) c = c.split("").map((x) => x + x).join(""); return [0, 2, 4].map((i) => parseInt(c.slice(i, i + 2), 16)); };
    try {
      const A = p(a), B = p(b);
      return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(",")})`;
    } catch (e) { return b; }
  }
  Ch.mix = mix;

  // ------------------------------------------------------------------ ECDF
  Ch.ecdf = (container, opts) => {
    const series = (opts.series || []).map((s) => {
      const v = K.stats.sorted(s.values);
      return { ...s, step: true, points: v.map((x, i) => [x, (i + 1) / v.length]) };
    });
    return Ch.line(container, { ...opts, series, yMin: 0, yMax: 1, yLabel: opts.yLabel || "Share of patients at or below", yFmt: (v) => K.pct(v, 0), zoom: true });
  };

  window.Charts = Ch;
})();
