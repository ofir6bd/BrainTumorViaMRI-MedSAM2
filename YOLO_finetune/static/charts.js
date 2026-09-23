/* Small SVG chart kit for the YOLO_finetune page: line (with synced crosshair),
 * histogram, scatter and horizontal bars, plus one shared tooltip. No dependencies.
 * Every chart draws into a container at the container's own pixel width. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const INK = { muted: "#9a9ab0", grid: "#26263c", base: "#4a4a66", text: "#e8e8f0", surface: "#1a1a2e" };

  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v !== undefined && v !== null) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  }
  function txt(parent, x, y, s, attrs) {
    const t = el("text", { x, y, ...(attrs || {}) }, parent);
    t.textContent = s;
    return t;
  }
  function root(container, h, label) {
    container.innerHTML = "";
    const w = Math.max(240, container.clientWidth || 480);
    const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h, class: "ch-svg", role: "img", "aria-label": label || "" });
    container.appendChild(svg);
    return { svg, w, h };
  }
  function scale(d0, d1, r0, r1) {
    const k = (r1 - r0) / (d1 - d0 || 1);
    const s = (v) => r0 + (v - d0) * k;
    s.invert = (p) => d0 + (p - r0) / k;
    return s;
  }
  function ticks(lo, hi, n = 5) {
    const span = hi - lo || 1;
    const s0 = Math.pow(10, Math.floor(Math.log10(span / n)));
    const e = span / n / s0;
    const step = s0 * (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1);
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function fmt(v) {
    if (v == null || Number.isNaN(v)) return "–";
    if (Number.isInteger(v)) return v.toLocaleString();
    const a = Math.abs(v);
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    return String(+v.toPrecision(3));
  }
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

  // ---- tooltip
  const tip = document.createElement("div");
  tip.className = "ch-tip hidden";
  tip.setAttribute("role", "tooltip");
  document.addEventListener("DOMContentLoaded", () => document.body.appendChild(tip));
  function showTip(html, e) {
    tip.innerHTML = html;
    tip.classList.remove("hidden");
    const r = tip.getBoundingClientRect();
    let x = e.clientX + 14;
    let y = e.clientY + 14;
    if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
    if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  }
  const hideTip = () => tip.classList.add("hidden");
  const row = (color, label, value, dash) =>
    `<div class="tt-row"><i style="background:${dash ? "none" : color};${dash ? `border-top:2px dashed ${color};height:0` : ""}"></i><span>${esc(label)}</span><b>${value}</b></div>`;

  function axes(svg, x, y, box, opt) {
    const { L, R, T, B, w, h } = box;
    for (const t of opt.yTickValues || ticks(y.lo, y.hi, opt.yTicks || 4)) {
      const yy = y(t);
      el("line", { x1: L, x2: w - R, y1: yy, y2: yy, class: "ch-grid" }, svg);
      txt(svg, L - 6, yy, (opt.yFmt || fmt)(t), { class: "ch-tick", "text-anchor": "end", "dominant-baseline": "middle" });
    }
    for (const t of opt.xTickValues || ticks(x.lo, x.hi, opt.xTicks || 6)) {
      const xx = x(t);
      if (xx < L - 1 || xx > w - R + 1) continue;
      el("line", { x1: xx, x2: xx, y1: h - B, y2: h - B + 4, stroke: INK.base }, svg);
      txt(svg, xx, h - B + 16, (opt.xFmt || fmt)(t), { class: "ch-tick", "text-anchor": "middle" });
    }
    el("line", { x1: L, x2: w - R, y1: h - B, y2: h - B, stroke: INK.base }, svg);
    if (opt.xLabel) txt(svg, L + (w - L - R) / 2, h - 4, opt.xLabel, { class: "ch-axis", "text-anchor": "middle" });
    if (opt.yLabel) txt(svg, 12, T + (h - B - T) / 2, opt.yLabel,
      { class: "ch-axis", "text-anchor": "middle", transform: `rotate(-90 12 ${T + (h - B - T) / 2})` });
  }

  /* A row of legend keys at the top of a chart: line keys (solid / dashed) or squares. */
  function legend(svg, x, y, items) {
    let lx = x;
    for (const it of items) {
      if (it.line) el("line", { x1: lx, x2: lx + 18, y1: y - 4, y2: y - 4, stroke: it.color, "stroke-width": 2, "stroke-dasharray": it.dash || null }, svg);
      else el("rect", { x: lx + 4, y: y - 9, width: 10, height: 10, rx: 2, fill: it.color }, svg);
      txt(svg, lx + 23, y, it.name, { class: "ch-tick" });
      lx += 34 + it.name.length * 6.1;
    }
  }

  /* Line chart. series: [{name, color, dash, points:[[x,y],...]}]. markers: [{x, label}].
   * With more than one series a legend is drawn on top. xLabel / yLabel name the axes.
   * sync: a shared {listeners:Set} so hovering one chart moves the crosshair on all. */
  function line(container, opt) {
    const { svg, w, h } = root(container, opt.height || 200, opt.title);
    const withLegend = opt.series.length > 1;
    const box = { L: opt.left || 54, R: 12, T: withLegend ? 26 : 10, B: opt.xLabel ? 36 : 24, w, h };
    if (withLegend) legend(svg, box.L, 13, opt.series.map((s) => ({ name: s.name, color: s.color, dash: s.dash, line: true })));
    const pts = opt.series.flatMap((s) => s.points.filter((p) => p[1] != null && Number.isFinite(p[1])));
    if (!pts.length) {
      txt(svg, w / 2, h / 2, opt.empty || "No data yet", { class: "ch-empty", "text-anchor": "middle" });
      return { move() {} };
    }
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    let ylo = opt.yMin != null ? opt.yMin : Math.min(...ys);
    let yhi = opt.yMax != null ? opt.yMax : Math.max(...ys);
    if (opt.yMin == null) ylo -= (yhi - ylo) * 0.06 || 0.05;
    if (opt.yMax == null) yhi += (yhi - ylo) * 0.06 || 0.05;
    const x = scale(Math.min(...xs), Math.max(...xs), box.L, w - box.R);
    const y = scale(ylo, yhi, h - box.B, box.T);
    x.lo = Math.min(...xs); x.hi = Math.max(...xs); y.lo = ylo; y.hi = yhi;
    // Whole-number x values (rounds, slices) get whole-number ticks, never 1.20 / 1.40.
    const intX = xs.every(Number.isInteger);
    const xTickValues = opt.xTickValues
      || (intX ? [...new Set(ticks(x.lo, x.hi, opt.xTicks || 6).map(Math.round))].filter((t) => t >= x.lo && t <= x.hi) : null);
    axes(svg, x, y, box, { ...opt, xTickValues });
    for (const m of opt.markers || []) {
      const xx = x(m.x);
      el("line", { x1: xx, x2: xx, y1: box.T, y2: h - box.B, stroke: m.color || INK.muted, "stroke-dasharray": "3 3" }, svg);
      if (m.label) txt(svg, xx + 4, box.T + 10, m.label, { class: "ch-tick" });
    }
    for (const s of opt.series) {
      const p = s.points.filter((q) => q[1] != null && Number.isFinite(q[1]));
      if (!p.length) continue;
      el("path", { d: p.map((q, i) => `${i ? "L" : "M"}${x(q[0]).toFixed(1)},${y(q[1]).toFixed(1)}`).join(""),
                   fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round",
                   "stroke-linecap": "round", "stroke-dasharray": s.dash || null }, svg);
      if (p.length === 1 || s.dots) {
        for (const q of p) el("circle", { cx: x(q[0]), cy: y(q[1]), r: 3.5, fill: s.color, stroke: INK.surface, "stroke-width": 1.5 }, svg);
      }
    }
    const cross = el("line", { y1: box.T, y2: h - box.B, stroke: INK.muted, visibility: "hidden" }, svg);
    const dots = opt.series.map((s) => el("circle", { r: 4, fill: s.color, stroke: INK.surface, "stroke-width": 2, visibility: "hidden" }, svg));
    const allX = [...new Set(xs)].sort((a, b) => a - b);
    const nearest = (xv) => allX.reduce((b, v) => (Math.abs(v - xv) < Math.abs(b - xv) ? v : b), allX[0]);
    function move(xv, e) {
      if (xv == null) {
        cross.setAttribute("visibility", "hidden");
        dots.forEach((d) => d.setAttribute("visibility", "hidden"));
        return;
      }
      const xn = nearest(xv);
      cross.setAttribute("x1", x(xn));
      cross.setAttribute("x2", x(xn));
      cross.setAttribute("visibility", "visible");
      let html = `<div class="tt-title">${esc(opt.xName || "x")} ${fmt(xn)}</div>`;
      opt.series.forEach((s, i) => {
        const q = s.points.find((p) => p[0] === xn);
        if (!q || q[1] == null) { dots[i].setAttribute("visibility", "hidden"); return; }
        dots[i].setAttribute("cx", x(xn));
        dots[i].setAttribute("cy", y(q[1]));
        dots[i].setAttribute("visibility", "visible");
        html += row(s.color, s.name, (opt.yFmt || fmt)(q[1]), s.dash);
      });
      if (e) showTip(html, e);
    }
    const ov = el("rect", { x: box.L, y: box.T, width: w - box.L - box.R, height: h - box.T - box.B, fill: "transparent" }, svg);
    ov.addEventListener("mousemove", (e) => {
      const r = svg.getBoundingClientRect();
      const xv = x.invert(((e.clientX - r.left) / r.width) * w);
      if (opt.sync) opt.sync.listeners.forEach((f) => f !== move && f(nearest(xv)));
      move(xv, e);
    });
    ov.addEventListener("mouseleave", () => {
      hideTip();
      if (opt.sync) opt.sync.listeners.forEach((f) => f(null));
      move(null);
    });
    if (opt.onClick) {
      ov.style.cursor = "pointer";
      ov.addEventListener("click", (e) => {
        const r = svg.getBoundingClientRect();
        opt.onClick(nearest(x.invert(((e.clientX - r.left) / r.width) * w)));
      });
    }
    if (opt.sync) opt.sync.listeners.add(move);
    return { move };
  }

  /* Two-scale line chart: series with axis "left" (e.g. Dice 0–1) and axis "right"
   * (e.g. pixel counts). Each axis is named and its ticks take the colour of the series
   * that belong to it, so it is clear which line is read on which side.
   * series: [{name, color, dash, axis, points:[[x,y],...]}]; marker: {x, label}. */
  function dualLine(container, opt) {
    const { svg, w, h } = root(container, opt.height || 260, opt.title);
    const box = { L: 54, R: 60, T: 26, B: 40, w, h };
    const left = opt.series.filter((s) => s.axis !== "right");
    const right = opt.series.filter((s) => s.axis === "right");
    legend(svg, box.L, 13, opt.series.map((s) => ({ name: s.name, color: s.color, dash: s.dash, line: true })));
    const all = opt.series.flatMap((s) => s.points.filter((p) => p[1] != null && Number.isFinite(p[1])));
    if (!all.length) {
      txt(svg, w / 2, h / 2, opt.empty || "No data yet", { class: "ch-empty", "text-anchor": "middle" });
      return;
    }
    const xs = all.map((p) => p[0]);
    const x = scale(Math.min(...xs), Math.max(...xs), box.L, w - box.R);
    const span = (ss, min0) => {
      const v = ss.flatMap((s) => s.points.map((p) => p[1])).filter((q) => q != null && Number.isFinite(q));
      return [min0 ? 0 : Math.min(...v, 0), Math.max(...v, min0 ? 1e-9 : 1) * 1.05];
    };
    const [lLo, lHi] = opt.leftMin != null ? [opt.leftMin, opt.leftMax] : span(left, true);
    const [rLo, rHi] = span(right, true);
    const yL = scale(lLo, lHi, h - box.B, box.T);
    const yR = scale(rLo, rHi, h - box.B, box.T);
    const leftColor = left.length ? left[0].color : INK.muted;
    const rightColor = right.length ? right[0].color : INK.muted;
    for (const t of ticks(lLo, lHi, 4)) {
      el("line", { x1: box.L, x2: w - box.R, y1: yL(t), y2: yL(t), class: "ch-grid" }, svg);
      txt(svg, box.L - 6, yL(t), (opt.leftFmt || fmt)(t), { class: "ch-tick", fill: leftColor, "text-anchor": "end", "dominant-baseline": "middle" });
    }
    if (right.length) {
      for (const t of ticks(rLo, rHi, 4)) {
        txt(svg, w - box.R + 8, yR(t), fmt(t), { class: "ch-tick", fill: rightColor, "dominant-baseline": "middle" });
      }
      txt(svg, w - 10, box.T + (h - box.B - box.T) / 2, opt.rightLabel || "",
        { class: "ch-axis", fill: rightColor, "text-anchor": "middle", transform: `rotate(90 ${w - 10} ${box.T + (h - box.B - box.T) / 2})` });
    }
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const xTicks = xs.every(Number.isInteger)
      ? [...new Set(ticks(xMin, xMax, 6).map(Math.round))]
      : ticks(xMin, xMax, 6);
    for (const t of xTicks) {
      if (t < xMin || t > xMax) continue;
      el("line", { x1: x(t), x2: x(t), y1: h - box.B, y2: h - box.B + 4, stroke: INK.base }, svg);
      txt(svg, x(t), h - box.B + 16, fmt(t), { class: "ch-tick", "text-anchor": "middle" });
    }
    el("line", { x1: box.L, x2: w - box.R, y1: h - box.B, y2: h - box.B, stroke: INK.base }, svg);
    txt(svg, box.L + (w - box.L - box.R) / 2, h - 4, opt.xLabel || "", { class: "ch-axis", "text-anchor": "middle" });
    txt(svg, 12, box.T + (h - box.B - box.T) / 2, opt.leftLabel || "",
      { class: "ch-axis", fill: leftColor, "text-anchor": "middle", transform: `rotate(-90 12 ${box.T + (h - box.B - box.T) / 2})` });
    // Vertical markers, drawn before the series so the data stays on top. Labels alternate
    // height so markers a few x apart do not write over each other.
    (opt.markers || []).forEach((m, i) => {
      if (m.x == null || m.x < xMin || m.x > xMax) return;
      const mx = x(m.x);
      el("line", { x1: mx, x2: mx, y1: box.T, y2: h - box.B, stroke: m.color || INK.muted,
                   "stroke-dasharray": "3 3" }, svg);
      if (m.label) txt(svg, mx + 4, box.T + 10 + (i % 2) * 11, m.label, { class: "ch-tick", fill: m.color || null });
    });
    const yOf = (s) => (s.axis === "right" ? yR : yL);
    for (const s of opt.series) {
      const p = s.points.filter((q) => q[1] != null && Number.isFinite(q[1]));
      if (!p.length) continue;
      el("path", { d: p.map((q, i) => `${i ? "L" : "M"}${x(q[0]).toFixed(1)},${yOf(s)(q[1]).toFixed(1)}`).join(""),
                   fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round",
                   "stroke-linecap": "round", "stroke-dasharray": s.dash || null }, svg);
    }
    if (opt.marker && opt.marker.x != null) {
      const mx = x(opt.marker.x);
      el("line", { x1: mx, x2: mx, y1: box.T, y2: h - box.B, stroke: "#e24b4a", "stroke-width": 1.5, "stroke-dasharray": "5 4" }, svg);
      if (opt.marker.label) txt(svg, mx + 5, box.T + 11, opt.marker.label, { class: "ch-tick" });
    }
    const cross = el("line", { y1: box.T, y2: h - box.B, stroke: INK.muted, visibility: "hidden" }, svg);
    const dots = opt.series.map((s) => el("circle", { r: 4, fill: s.color, stroke: INK.surface, "stroke-width": 2, visibility: "hidden" }, svg));
    const allX = [...new Set(xs)].sort((a, b) => a - b);
    const nearest = (xv) => allX.reduce((b, v) => (Math.abs(v - xv) < Math.abs(b - xv) ? v : b), allX[0]);
    const xAt = (e) => {
      const r = svg.getBoundingClientRect();
      return nearest(x.invert(((e.clientX - r.left) / r.width) * w));
    };
    const ov = el("rect", { x: box.L, y: box.T, width: w - box.L - box.R, height: h - box.T - box.B,
                            fill: "transparent", class: opt.onClick ? "ch-dot" : "" }, svg);
    ov.addEventListener("mousemove", (e) => {
      const xn = xAt(e);
      cross.setAttribute("x1", x(xn));
      cross.setAttribute("x2", x(xn));
      cross.setAttribute("visibility", "visible");
      let html = `<div class="tt-title">${esc(opt.xName || "x")} ${fmt(xn)}</div>`;
      opt.series.forEach((s, i) => {
        const q = s.points.find((p) => p[0] === xn);
        if (!q || q[1] == null) { dots[i].setAttribute("visibility", "hidden"); return; }
        dots[i].setAttribute("cx", x(xn));
        dots[i].setAttribute("cy", yOf(s)(q[1]));
        dots[i].setAttribute("visibility", "visible");
        html += row(s.color, s.name, fmt(q[1]), s.dash);
      });
      showTip(html + (opt.onClick ? `<div class="tt-muted">click to show this slice</div>` : ""), e);
    });
    ov.addEventListener("mouseleave", () => {
      hideTip();
      cross.setAttribute("visibility", "hidden");
      dots.forEach((d) => d.setAttribute("visibility", "hidden"));
    });
    if (opt.onClick) ov.addEventListener("click", (e) => opt.onClick(xAt(e)));
  }

  /* Histogram over [lo, hi] in `bins` equal bins; each bar shows its count on top.
   * markers: [{x, label, color}]. */
  function histogram(container, opt) {
    const { svg, w, h } = root(container, opt.height || 190, opt.title);
    const box = { L: 54, R: 12, T: 20, B: opt.xLabel ? 36 : 24, w, h };
    const vals = opt.values.filter((v) => v != null && Number.isFinite(v));
    if (!vals.length) {
      txt(svg, w / 2, h / 2, opt.empty || "No data yet", { class: "ch-empty", "text-anchor": "middle" });
      return;
    }
    const lo = opt.lo != null ? opt.lo : Math.min(...vals);
    const hi = opt.hi != null ? opt.hi : Math.max(...vals);
    const n = opt.bins || 20;
    const bw = (hi - lo) / n || 1;
    const counts = new Array(n).fill(0);
    for (const v of vals) counts[Math.min(n - 1, Math.max(0, Math.floor((v - lo) / bw)))]++;
    const x = scale(lo, hi, box.L, w - box.R);
    const y = scale(0, Math.max(...counts) * 1.08, h - box.B, box.T);
    x.lo = lo; x.hi = hi; y.lo = 0; y.hi = Math.max(...counts) * 1.08;
    // Counts are whole numbers: never label the axis 0.5 / 1.50.
    const maxC = Math.max(...counts);
    const step = Math.max(1, Math.ceil(maxC / 4));
    const yTickValues = maxC <= 8 ? Array.from({ length: maxC + 1 }, (_, i) => i)
      : Array.from({ length: Math.floor(maxC / step) + 1 }, (_, i) => i * step);
    axes(svg, x, y, box, { ...opt, yTickValues, yFmt: (v) => fmt(v) });
    counts.forEach((c, i) => {
      const x0 = x(lo + i * bw) + 1;
      const ww = Math.max(1, x(lo + (i + 1) * bw) - x(lo + i * bw) - 2);
      const g = el("g", {}, svg);
      if (c) {
        const hh = h - box.B - y(c);
        const r = Math.min(3, ww / 2, hh);
        el("path", { d: `M${x0},${h - box.B}V${y(c) + r}Q${x0},${y(c)} ${x0 + r},${y(c)}H${x0 + ww - r}Q${x0 + ww},${y(c)} ${x0 + ww},${y(c) + r}V${h - box.B}Z`,
                     fill: opt.color || "#3987e5" }, g);
        if (ww >= 11) txt(g, x0 + ww / 2, y(c) - 4, c, { class: "ch-val", "text-anchor": "middle" });
      }
      el("rect", { x: x0, y: box.T, width: ww, height: h - box.T - box.B, fill: "transparent" }, g);
      g.addEventListener("mousemove", (e) => showTip(`<div class="tt-title">${(opt.xFmt || fmt)(lo + i * bw)} – ${(opt.xFmt || fmt)(lo + (i + 1) * bw)}</div>${c} ${esc(opt.unit || "items")}`, e));
      g.addEventListener("mouseleave", hideTip);
      if (opt.onClick) {
        g.style.cursor = "pointer";
        g.addEventListener("click", () => opt.onClick(lo + i * bw, lo + (i + 1) * bw));
      }
    });
    for (const m of opt.markers || []) {
      if (m.x == null) continue;
      const xx = x(m.x);
      el("line", { x1: xx, x2: xx, y1: box.T, y2: h - box.B, stroke: m.color || INK.text, "stroke-width": 1.5, "stroke-dasharray": "4 3" }, svg);
      txt(svg, xx + 4, box.T + 10, m.label, { class: "ch-tick" });
    }
  }

  /* Scatter. points: [{x, y, color, ring, id, tip}]; logX: log10 x axis. */
  function scatter(container, opt) {
    const { svg, w, h } = root(container, opt.height || 280, opt.title);
    const box = { L: 50, R: 12, T: 12, B: 38, w, h };
    const pts = opt.points.filter((p) => p.x != null && p.y != null && (!opt.logX || p.x > 0));
    if (!pts.length) {
      txt(svg, w / 2, h / 2, opt.empty || "No data yet", { class: "ch-empty", "text-anchor": "middle" });
      return;
    }
    const fx = opt.logX ? Math.log10 : (v) => v;
    const xs = pts.map((p) => fx(p.x));
    const ys = pts.map((p) => p.y);
    const xlo = Math.min(...xs); const xhi = Math.max(...xs);
    const ylo = opt.yMin != null ? opt.yMin : Math.min(...ys);
    const yhi = opt.yMax != null ? opt.yMax : Math.max(...ys);
    const px = (xhi - xlo) * 0.04 || 1;
    const x = scale(xlo - px, xhi + px, box.L, w - box.R);
    const y = scale(ylo, yhi, h - box.B, box.T);
    x.lo = xlo - px; x.hi = xhi + px; y.lo = ylo; y.hi = yhi;
    const xTickValues = opt.logX
      ? [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000].map(Math.log10).filter((t) => t >= x.lo && t <= x.hi)
      : null;
    axes(svg, x, y, box, { ...opt, xTickValues, xFmt: opt.logX ? (t) => fmt(+Math.pow(10, t).toPrecision(2)) : opt.xFmt });
    for (const p of pts) {
      const d = el("circle", { cx: x(fx(p.x)), cy: y(p.y), r: p.big ? 6 : 3.6, fill: p.ring ? "none" : p.color,
                               "fill-opacity": 0.75, stroke: p.big ? INK.text : p.ring ? p.color : INK.surface,
                               "stroke-width": p.big ? 2 : p.ring ? 1.5 : 0.8, class: "ch-dot" }, svg);
      if (p.tip) {
        d.addEventListener("mousemove", (e) => showTip(p.tip, e));
        d.addEventListener("mouseleave", hideTip);
      }
      if (opt.onClick) d.addEventListener("click", () => opt.onClick(p));
    }
  }

  /* Horizontal bars. items: [{label, value, color, text, tip}], max optional.
   * xLabel names the bar length (with ticks under the bars), yLabel names the rows. */
  function bars(container, opt) {
    const rowH = opt.rowH || 28;
    const bottom = opt.xLabel ? 40 : 8;
    const { svg, w, h } = root(container, opt.items.length * rowH + 4 + bottom, opt.title);
    const L = (opt.labelWidth || 120) + (opt.yLabel ? 22 : 0);
    const R = opt.valueWidth || 70;
    const max = opt.max || Math.max(1e-9, ...opt.items.map((i) => i.value || 0));
    const x = scale(0, max, L, w - R);
    const yEnd = 4 + opt.items.length * rowH;
    if (opt.xLabel) {
      for (const t of ticks(0, max, 4)) {
        el("line", { x1: x(t), x2: x(t), y1: yEnd, y2: yEnd + 4, stroke: INK.base }, svg);
        txt(svg, x(t), yEnd + 16, (opt.xFmt || fmt)(t), { class: "ch-tick", "text-anchor": "middle" });
      }
      txt(svg, L + (w - L - R) / 2, h - 4, opt.xLabel, { class: "ch-axis", "text-anchor": "middle" });
    }
    if (opt.yLabel) {
      txt(svg, 12, yEnd / 2, opt.yLabel, { class: "ch-axis", "text-anchor": "middle", transform: `rotate(-90 12 ${yEnd / 2})` });
    }
    opt.items.forEach((it, i) => {
      const y0 = 4 + i * rowH;
      const g = el("g", {}, svg);
      txt(g, L - 8, y0 + rowH / 2, it.label, { class: "ch-lbl", "text-anchor": "end", "dominant-baseline": "middle" });
      el("rect", { x: L, y: y0 + 6, width: w - R - L, height: rowH - 12, rx: 3, fill: "#14142a" }, g);
      const ww = Math.max(0, x(it.value || 0) - L);
      if (ww) el("path", { d: `M${L},${y0 + 6}H${L + ww - Math.min(3, ww)}Q${L + ww},${y0 + 6} ${L + ww},${y0 + 6 + Math.min(3, ww)}V${y0 + rowH - 6 - Math.min(3, ww)}Q${L + ww},${y0 + rowH - 6} ${L + ww - Math.min(3, ww)},${y0 + rowH - 6}H${L}Z`,
                           fill: it.color || "#3987e5" }, g);
      txt(g, w - R + 8, y0 + rowH / 2, it.text != null ? it.text : fmt(it.value), { class: "ch-val", "dominant-baseline": "middle" });
      if (it.tip) {
        g.addEventListener("mousemove", (e) => showTip(it.tip, e));
        g.addEventListener("mouseleave", hideTip);
      }
    });
  }

  window.Charts = { line, dualLine, histogram, scatter, bars, fmt, esc, row, showTip, hideTip };
})();
