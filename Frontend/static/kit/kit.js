/* Kit — the parts every page of the viewer shares.
 *
 * Utilities, API calls, theme, tooltip, toasts, modal, drawer, command palette (Ctrl+K),
 * keyboard help (?), URL-hash state, saved views, the sortable / searchable / exportable
 * data table, section navigation with scroll-spy, KPI tiles, statistics and downloads.
 * Pages use `Kit.*`; charts live in charts.js (`Charts.*`), which builds on this file.
 * Every number the pages show comes from the server — nothing here invents data.
 */
(function () {
  "use strict";
  const K = {};

  // ------------------------------------------------------------------ basics
  K.$ = (sel, root) => (root || document).querySelector(sel);
  K.$$ = (sel, root) => [...(root || document).querySelectorAll(sel)];
  K.id = (id) => document.getElementById(id);
  K.esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
  K.clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  K.debounce = (fn, ms = 200) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
  K.throttle = (fn, ms = 100) => { let last = 0, t; return (...a) => { const now = Date.now(); clearTimeout(t);
    if (now - last >= ms) { last = now; fn(...a); } else t = setTimeout(() => { last = Date.now(); fn(...a); }, ms - (now - last)); }; };
  K.h = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; };
  K.uid = (() => { let n = 0; return (p = "u") => `${p}${++n}`; })();

  // numbers
  K.fmt = (v, digits) => {
    if (v == null || Number.isNaN(v)) return "–";
    if (digits != null) return Number(v).toFixed(digits);
    if (Number.isInteger(v)) return v.toLocaleString();
    const a = Math.abs(v);
    if (a >= 1000) return Math.round(v).toLocaleString();
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    if (a === 0) return "0";
    return String(+v.toPrecision(2));
  };
  K.f4 = (v) => (v == null ? "–" : Number(v).toFixed(4));
  K.f3 = (v) => (v == null ? "–" : Number(v).toFixed(3));
  K.f2 = (v) => (v == null ? "–" : Number(v).toFixed(2));
  K.pct = (v, d = 1) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);
  K.signed = (v, d = 4) => (v == null ? "–" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`);
  K.int = (v) => (v == null ? "–" : Math.round(v).toLocaleString());
  K.secs = (s) => (s == null ? "–" : s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`
    : `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`);
  K.when = (iso) => (iso ? String(iso).replace("T", " ").slice(0, 16) : "–");
  K.ago = (iso) => {
    if (!iso) return "";
    const s = (Date.now() - new Date(iso).getTime()) / 1000;
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.round(s / 60)} min ago`;
    if (s < 86400) return `${Math.round(s / 3600)} h ago`;
    return `${Math.round(s / 86400)} d ago`;
  };
  K.deltaHtml = (d, digits = 4) => (d == null ? "–"
    : `<span class="${d > 0 ? "good" : d < 0 ? "bad" : "dim"}">${K.signed(d, digits)}</span>`);

  // ------------------------------------------------------------------ api
  K.api = async (url, opts) => {
    let r;
    try {
      r = await fetch(url, opts);
    } catch (e) {
      throw new Error("The server is not answering. Is the viewer still running?");
    }
    const text = await r.text();
    let body;
    try { body = text ? JSON.parse(text) : {}; } catch (e) {
      throw new Error(r.ok ? "The server sent something that is not JSON." : `Server error ${r.status}`);
    }
    if (!r.ok) throw new Error(body.error || `Server error ${r.status}`);
    return body;
  };
  K.post = (url, body) => K.api(url, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}) });

  // ------------------------------------------------------------------ theme / colours
  const THEME_KEY = "brats-theme";
  K.theme = () => document.documentElement.dataset.theme || "dark";
  K.setTheme = (t) => {
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem(THEME_KEY, t); } catch (e) { /* private mode */ }
    window.dispatchEvent(new CustomEvent("themechange", { detail: t }));
  };
  K.toggleTheme = () => K.setTheme(K.theme() === "dark" ? "light" : "dark");
  // Resolve "var(--yolo)" (or a plain colour) to a real colour, for SVG attributes and PNGs.
  K.color = (c) => {
    if (!c || !String(c).startsWith("var(")) return c;
    const name = String(c).slice(4, -1).trim();
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
  };
  K.FAMILY = { yolo: "var(--yolo)", medsam2: "var(--medsam2)", test: "var(--test)", other: "var(--neutral)" };
  K.LABELS = { NETC: "var(--netc)", SNFH: "var(--snfh)", ET: "var(--et)", RC: "var(--rc)" };
  K.LABEL_WORDS = { NETC: "dead core", SNFH: "swelling", ET: "enhancing", RC: "surgery cavity",
                    WT: "whole tumour", TC: "tumour core" };
  // categorical fallback for more than three series (checked on both themes)
  K.PALETTE = ["var(--yolo)", "var(--medsam2)", "var(--test)", "var(--et)", "var(--rc)", "var(--netc)", "var(--neutral)"];

  // ------------------------------------------------------------------ tooltip
  let tipEl;
  K.tip = (html, evt) => {
    if (!tipEl) { tipEl = document.createElement("div"); tipEl.className = "tip hidden"; tipEl.setAttribute("role", "tooltip"); document.body.appendChild(tipEl); }
    tipEl.innerHTML = html;
    tipEl.classList.remove("hidden");
    const pad = 14, r = tipEl.getBoundingClientRect();
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
    tipEl.style.left = `${Math.max(4, x)}px`;
    tipEl.style.top = `${Math.max(4, y)}px`;
  };
  K.untip = () => tipEl && tipEl.classList.add("hidden");
  K.hover = (node, fn) => {
    node.addEventListener("mousemove", (e) => { const h = fn(e); if (h) K.tip(h, e); else K.untip(); });
    node.addEventListener("mouseleave", K.untip);
  };
  K.tipTitle = (s) => `<div class="tt">${K.esc(s)}</div>`;
  K.tipRow = (color, label, value) =>
    `<div class="tr"><i style="background:${color}"></i><span>${K.esc(label)}</span><b>${value}</b></div>`;

  // ------------------------------------------------------------------ toasts
  let toastBox;
  K.toast = (msg, kind = "", ms = 4200) => {
    if (!toastBox) { toastBox = document.createElement("div"); toastBox.className = "toasts"; toastBox.setAttribute("aria-live", "polite"); document.body.appendChild(toastBox); }
    const t = document.createElement("div");
    t.className = `toast ${kind}`;
    t.innerHTML = msg;
    toastBox.appendChild(t);
    setTimeout(() => t.remove(), ms);
    return t;
  };
  K.fail = (e) => K.toast(K.esc(e && e.message ? e.message : e), "bad", 7000);

  // ------------------------------------------------------------------ modal / lightbox
  K.modal = (title, body, { wide = false, onClose } = {}) => {
    const m = K.h(`<div class="modal" role="dialog" aria-modal="true"><div class="box ${wide ? "wide" : ""}">
      <div class="box-head"><h3 style="flex:1">${K.esc(title)}</h3><button class="icon ghost" aria-label="Close">✕</button></div>
      <div class="box-body"></div></div></div>`);
    const b = K.$(".box-body", m);
    if (typeof body === "string") b.innerHTML = body; else if (body) b.appendChild(body);
    const close = () => { m.remove(); document.removeEventListener("keydown", onKey, true); onClose && onClose(); };
    const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } };
    m.addEventListener("click", (e) => { if (e.target === m) close(); });
    K.$(".box-head button", m).addEventListener("click", close);
    document.addEventListener("keydown", onKey, true);
    document.body.appendChild(m);
    return { el: m, body: b, close };
  };
  K.lightbox = (src, title) => K.modal(title || "Picture", `<img src="${K.esc(src)}" alt="" style="width:100%;border-radius:8px">`, { wide: true });

  // ------------------------------------------------------------------ fullscreen card
  K.fullscreen = (card) => {
    if (card.classList.contains("is-full")) {
      card.classList.remove("is-full");
      K.$(".backdrop") && K.$(".backdrop").remove();
    } else {
      const bd = document.createElement("div");
      bd.className = "backdrop";
      bd.addEventListener("click", () => K.fullscreen(card));
      document.body.appendChild(bd);
      card.classList.add("is-full");
    }
    window.dispatchEvent(new Event("resize"));
  };
  document.addEventListener("keydown", (e) => {
    const full = K.$(".card.is-full");
    if (e.key === "Escape" && full && !K.$(".modal")) K.fullscreen(full);
  });

  // ------------------------------------------------------------------ hash state
  // #k=v&k2=v2 — every page keeps its whole view in the URL, so "Copy link" reproduces it.
  K.hashGet = () => Object.fromEntries(new URLSearchParams(location.hash.replace(/^#/, "")));
  K.hashSet = (obj, push = false) => {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(obj)) if (v !== undefined && v !== null && v !== "") q.set(k, v);
    const h = `#${q.toString()}`;
    if (h !== location.hash) (push ? history.pushState : history.replaceState).call(history, null, "", h);
  };
  K.copyLink = async () => {
    try { await navigator.clipboard.writeText(location.href); K.toast("Link copied — it opens this exact view.", "good"); }
    catch (e) { K.modal("Copy this link", `<input type="text" value="${K.esc(location.href)}" style="width:100%" readonly>`); }
  };

  // ------------------------------------------------------------------ saved views (per page, this browser)
  const viewsKey = () => `brats-views:${location.pathname}`;
  K.views = {
    list() { try { return JSON.parse(localStorage.getItem(viewsKey()) || "[]"); } catch (e) { return []; } },
    save(name) {
      const v = K.views.list().filter((x) => x.name !== name);
      v.unshift({ name, hash: location.hash, at: new Date().toISOString() });
      try { localStorage.setItem(viewsKey(), JSON.stringify(v.slice(0, 30))); } catch (e) { /* ignore */ }
    },
    remove(name) {
      try { localStorage.setItem(viewsKey(), JSON.stringify(K.views.list().filter((x) => x.name !== name))); } catch (e) { /* ignore */ }
    },
    menu(anchorBtn, apply) {
      const close = () => menu.remove();
      const menu = K.h(`<div class="menu"></div>`);
      const render = () => {
        const list = K.views.list();
        menu.innerHTML = `<button data-act="save">＋ Save this view…</button>` +
          (list.length ? list.map((v) => `<button data-name="${K.esc(v.name)}">★ ${K.esc(v.name)}<span class="faint" style="margin-left:auto">${K.ago(v.at)}</span></button>`).join("") +
            `<button data-act="clear" class="faint">Delete a saved view…</button>` : `<div class="faint" style="padding:6px 8px;font-size:.8rem">No saved views yet.</div>`);
      };
      render();
      menu.addEventListener("click", (e) => {
        const b = e.target.closest("button");
        if (!b) return;
        if (b.dataset.act === "save") {
          const name = prompt("Name for this view:");
          if (name) { K.views.save(name.trim()); K.toast(`Saved view “${K.esc(name)}”.`, "good"); }
        } else if (b.dataset.act === "clear") {
          const name = prompt(`Delete which view? (${K.views.list().map((v) => v.name).join(", ")})`);
          if (name) K.views.remove(name.trim());
        } else if (b.dataset.name) {
          const v = K.views.list().find((x) => x.name === b.dataset.name);
          if (v) { history.replaceState(null, "", v.hash || "#"); apply && apply(); window.dispatchEvent(new HashChangeEvent("hashchange")); }
        }
        close();
      });
      const wrap = anchorBtn.parentElement;
      wrap.style.position = "relative";
      wrap.appendChild(menu);
      setTimeout(() => document.addEventListener("click", function off(ev) {
        if (!menu.contains(ev.target)) { close(); document.removeEventListener("click", off); }
      }), 0);
    },
  };

  // ------------------------------------------------------------------ downloads
  K.download = (name, content, mime = "text/plain") => {
    const blob = content instanceof Blob ? content : new Blob([content], { type: mime });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  };
  K.csv = (rows, cols) => {
    const q = (v) => { if (v == null) return ""; const s = String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
    return [cols.map((c) => q(c.label || c.k)).join(","), ...rows.map((r) => cols.map((c) => q(c.raw ? c.raw(r) : c.get ? c.get(r) : r[c.k])).join(","))].join("\n");
  };

  // ------------------------------------------------------------------ keys + help
  const KEYS = [];
  const typing = (e) => /input|select|textarea/i.test(e.target.tagName) || e.target.isContentEditable;
  K.key = (combo, desc, fn, group = "This page") => KEYS.push({ combo, desc, fn, group });
  const match = (e, combo) => {
    const parts = combo.toLowerCase().split("+");
    const key = parts.pop();
    const want = { ctrl: parts.includes("ctrl"), shift: parts.includes("shift"), alt: parts.includes("alt") };
    if ((e.ctrlKey || e.metaKey) !== want.ctrl || e.altKey !== want.alt) return false;
    if (want.shift && !e.shiftKey) return false;
    return e.key.toLowerCase() === key || (key === "?" && e.key === "?") || e.code.toLowerCase() === `key${key}`;
  };
  document.addEventListener("keydown", (e) => {
    if (K.$(".modal")) return;
    for (const k of KEYS) {
      const isCtrl = k.combo.toLowerCase().startsWith("ctrl+");
      if ((isCtrl || !typing(e)) && match(e, k.combo)) { e.preventDefault(); k.fn(e); return; }
    }
  });
  K.showKeys = () => {
    const groups = {};
    for (const k of KEYS) (groups[k.group] ||= []).push(k);
    K.modal("Keyboard shortcuts", Object.entries(groups).map(([g, ks]) =>
      `<h4 style="margin:6px 0 8px">${K.esc(g)}</h4><div class="keys-grid">${ks.map((k) =>
        `<div><span>${K.esc(k.desc)}</span><span>${k.combo.split("+").map((p) => `<kbd>${K.esc(p)}</kbd>`).join(" ")}</span></div>`).join("")}</div>`).join(""), { wide: true });
  };

  // ------------------------------------------------------------------ command palette
  const COMMANDS = [];
  const PROVIDERS = [];
  K.command = (label, run, { group = "Actions", hint = "" } = {}) => COMMANDS.push({ label, run, group, hint });
  K.provider = (fn) => PROVIDERS.push(fn); // fn(query) -> [{label, run, group, hint}]
  K.palette = () => {
    const m = K.h(`<div class="modal" role="dialog" aria-modal="true" aria-label="Command palette"><div class="box palette">
      <input type="text" placeholder="Type a command, a section, or a patient id…" aria-label="Search">
      <ul role="listbox"></ul></div></div>`);
    const input = K.$("input", m), ul = K.$("ul", m);
    let items = [], at = 0;
    const close = () => m.remove();
    const render = () => {
      const q = input.value.trim().toLowerCase();
      const base = COMMANDS.filter((c) => !q || c.label.toLowerCase().includes(q) || c.group.toLowerCase().includes(q));
      const extra = q ? PROVIDERS.flatMap((p) => p(q) || []) : [];
      items = [...base, ...extra].slice(0, 60);
      at = Math.min(at, Math.max(0, items.length - 1));
      let grp = null;
      ul.innerHTML = items.map((c, i) => {
        const head = c.group !== grp ? `<li class="grp">${K.esc((grp = c.group))}</li>` : "";
        return `${head}<li data-i="${i}" class="${i === at ? "on" : ""}" role="option">${K.esc(c.label)}<span class="k">${K.esc(c.hint || "")}</span></li>`;
      }).join("") || `<li class="grp">Nothing matches.</li>`;
    };
    const run = (i) => { const c = items[i]; if (c) { close(); c.run(); } };
    input.addEventListener("input", () => { at = 0; render(); });
    input.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") { at = Math.min(items.length - 1, at + 1); render(); e.preventDefault(); }
      else if (e.key === "ArrowUp") { at = Math.max(0, at - 1); render(); e.preventDefault(); }
      else if (e.key === "Enter") run(at);
      else if (e.key === "Escape") close();
    });
    ul.addEventListener("click", (e) => { const li = e.target.closest("li[data-i]"); if (li) run(+li.dataset.i); });
    m.addEventListener("click", (e) => { if (e.target === m) close(); });
    document.body.appendChild(m);
    render();
    input.focus();
  };

  // ------------------------------------------------------------------ section nav (sidebar TOC + scroll spy)
  K.toc = () => {
    const nav = K.id("toc");
    const secs = K.$$("[data-toc]");
    if (!nav || !secs.length) return;
    nav.innerHTML = secs.map((s) => `<a href="#" data-to="${s.id}">${K.esc(s.dataset.toc)}</a>`).join("");
    nav.addEventListener("click", (e) => {
      const a = e.target.closest("a[data-to]");
      if (!a) return;
      e.preventDefault();
      K.id(a.dataset.to).scrollIntoView({ behavior: "smooth", block: "start" });
    });
    for (const s of secs) K.command(`Go to: ${s.dataset.toc}`, () => s.scrollIntoView({ behavior: "smooth" }), { group: "Sections" });
    // the current section = the last one whose top has passed the sticky bars
    const spy = K.throttle(() => {
      let cur = secs[0];
      for (const s of secs) if (s.getBoundingClientRect().top < 150) cur = s;
      if (innerHeight + scrollY >= document.body.scrollHeight - 4) cur = secs[secs.length - 1];
      K.$$("a", nav).forEach((a) => a.classList.toggle("on", a.dataset.to === cur.id));
    }, 120);
    addEventListener("scroll", spy, { passive: true });
    spy();
  };

  // ------------------------------------------------------------------ KPI tile
  K.kpi = (label, value, sub, o = {}) =>
    `<div class="kpi ${o.hero ? "hero" : ""} ${o.cls || ""}" ${o.click ? `data-click="${K.esc(o.click)}" tabindex="0"` : ""} title="${K.esc(o.title || "")}">
      <span class="l">${K.esc(label)}</span><span class="v">${value}</span>
      ${sub ? `<span class="s">${sub}</span>` : ""}${o.spark ? K.sparkSvg(o.spark, o.sparkColor) : ""}</div>`;
  K.sparkSvg = (vals, color = "var(--accent)", w = 64, h = 20) => {
    const v = vals.filter((x) => x != null);
    if (v.length < 2) return "";
    const lo = Math.min(...v), hi = Math.max(...v), span = hi - lo || 1;
    const pts = vals.map((x, i) => (x == null ? null : `${(i / (vals.length - 1)) * w},${h - ((x - lo) / span) * h}`)).filter(Boolean);
    return `<svg class="spark" width="${w}" height="${h}" aria-hidden="true"><polyline points="${pts.join(" ")}" fill="none" stroke="${color}" stroke-width="1.5"/></svg>`;
  };

  // ------------------------------------------------------------------ stats
  const S = {};
  S.nums = (a) => a.filter((v) => v != null && Number.isFinite(v));
  S.sum = (a) => a.reduce((s, v) => s + v, 0);
  S.mean = (a) => { const v = S.nums(a); return v.length ? S.sum(v) / v.length : null; };
  S.quantile = (sorted, q) => {
    if (!sorted.length) return null;
    const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  };
  S.sorted = (a) => S.nums(a).slice().sort((x, y) => x - y);
  S.median = (a) => S.quantile(S.sorted(a), 0.5);
  S.std = (a) => { const v = S.nums(a); if (v.length < 2) return null; const m = S.mean(v); return Math.sqrt(S.sum(v.map((x) => (x - m) ** 2)) / (v.length - 1)); };
  S.sem = (a) => { const v = S.nums(a); const sd = S.std(v); return sd == null ? null : sd / Math.sqrt(v.length); };
  S.summary = (a) => { const s = S.sorted(a); return { n: s.length, mean: S.mean(s), median: S.quantile(s, 0.5), q1: S.quantile(s, 0.25),
    q3: S.quantile(s, 0.75), min: s[0] ?? null, max: s[s.length - 1] ?? null, sd: S.std(s) }; };
  // deterministic PRNG, so bootstrap intervals do not wobble between redraws
  S.rng = (seed = 12345) => () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  S.bootCI = (a, stat = S.mean, n = 1000, alpha = 0.05) => {
    const v = S.nums(a);
    if (v.length < 3) return null;
    const r = S.rng(v.length * 7919), out = [];
    for (let b = 0; b < n; b++) { const s = new Array(v.length); for (let i = 0; i < v.length; i++) s[i] = v[(r() * v.length) | 0]; out.push(stat(s)); }
    out.sort((x, y) => x - y);
    return [S.quantile(out, alpha / 2), S.quantile(out, 1 - alpha / 2)];
  };
  // standard normal CDF
  S.phi = (z) => { const t = 1 / (1 + 0.2316419 * Math.abs(z)); const d = 0.3989423 * Math.exp(-z * z / 2);
    const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))); return z > 0 ? 1 - p : p; };
  // Wilcoxon signed-rank (normal approximation, ties averaged, zeros dropped). Two-sided p.
  S.wilcoxon = (diffs) => {
    const d = S.nums(diffs).filter((x) => x !== 0);
    const n = d.length;
    if (n < 6) return null;
    const idx = d.map((v, i) => [Math.abs(v), i]).sort((a, b) => a[0] - b[0]);
    const rank = new Array(n);
    for (let i = 0; i < n;) { let j = i; while (j + 1 < n && idx[j + 1][0] === idx[i][0]) j++;
      for (let k = i; k <= j; k++) rank[idx[k][1]] = (i + j) / 2 + 1; i = j + 1; }
    const wPlus = S.sum(d.map((v, i) => (v > 0 ? rank[i] : 0)));
    const mu = n * (n + 1) / 4, sd = Math.sqrt(n * (n + 1) * (2 * n + 1) / 24);
    const z = (wPlus - mu) / sd;
    return { n, w: wPlus, z, p: 2 * (1 - S.phi(Math.abs(z))) };
  };
  // two-sample Kolmogorov–Smirnov: D and asymptotic p
  S.ks = (a, b) => {
    const x = S.sorted(a), y = S.sorted(b);
    if (x.length < 3 || y.length < 3) return null;
    let i = 0, j = 0, d = 0;
    while (i < x.length && j < y.length) {
      const v = Math.min(x[i], y[j]);
      while (i < x.length && x[i] <= v) i++;
      while (j < y.length && y[j] <= v) j++;
      d = Math.max(d, Math.abs(i / x.length - j / y.length));
    }
    const ne = (x.length * y.length) / (x.length + y.length), lam = (Math.sqrt(ne) + 0.12 + 0.11 / Math.sqrt(ne)) * d;
    let p = 0;
    for (let k = 1; k <= 100; k++) p += 2 * (k % 2 ? 1 : -1) * Math.exp(-2 * k * k * lam * lam);
    return { d, p: K.clamp(p, 0, 1) };
  };
  S.ranks = (a) => { const idx = a.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]); const r = new Array(a.length);
    for (let i = 0; i < a.length;) { let j = i; while (j + 1 < a.length && idx[j + 1][0] === idx[i][0]) j++;
      for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2 + 1; i = j + 1; } return r; };
  S.pearson = (x, y) => {
    const pairs = x.map((v, i) => [v, y[i]]).filter(([a, b]) => a != null && b != null && Number.isFinite(a) && Number.isFinite(b));
    if (pairs.length < 3) return null;
    const mx = S.mean(pairs.map((p) => p[0])), my = S.mean(pairs.map((p) => p[1]));
    let sxy = 0, sxx = 0, syy = 0;
    for (const [a, b] of pairs) { sxy += (a - mx) * (b - my); sxx += (a - mx) ** 2; syy += (b - my) ** 2; }
    return sxx && syy ? sxy / Math.sqrt(sxx * syy) : null;
  };
  S.spearman = (x, y) => {
    const pairs = x.map((v, i) => [v, y[i]]).filter(([a, b]) => a != null && b != null && Number.isFinite(a) && Number.isFinite(b));
    if (pairs.length < 3) return null;
    return S.pearson(S.ranks(pairs.map((p) => p[0])), S.ranks(pairs.map((p) => p[1])));
  };
  S.linreg = (x, y) => {
    const pairs = x.map((v, i) => [v, y[i]]).filter(([a, b]) => a != null && b != null);
    if (pairs.length < 3) return null;
    const mx = S.mean(pairs.map((p) => p[0])), my = S.mean(pairs.map((p) => p[1]));
    let sxy = 0, sxx = 0;
    for (const [a, b] of pairs) { sxy += (a - mx) * (b - my); sxx += (a - mx) ** 2; }
    const slope = sxx ? sxy / sxx : 0;
    return { slope, intercept: my - slope * mx };
  };
  S.pfmt = (p) => (p == null ? "–" : p < 0.001 ? "p < 0.001" : `p = ${p.toFixed(3)}`);
  K.stats = S;

  // ------------------------------------------------------------------ data table
  /* cols: [{k, label, get(r), fmt(v, r), num, sort(r) , bar:{max, color}, hidden, title, html}]
     opts: {rows, key, onRow, pageSize, search, exportName, selected, sort:{k, dir}, toolbar} */
  K.table = (container, cols, opts = {}) => {
    const st = { rows: opts.rows || [], q: "", sort: opts.sort || null, page: 0, size: opts.pageSize || 50,
                 hidden: new Set(cols.filter((c) => c.hidden).map((c) => c.k)), selected: opts.selected || null };
    container.innerHTML = `
      <div class="tbl-bar">
        ${opts.search !== false ? `<input type="search" placeholder="Search…" aria-label="Search the table" style="min-width:180px">` : ""}
        <span class="dim count"></span><span class="spacer"></span>
        <span class="colpick"><button class="small" data-act="cols" title="Show or hide columns">Columns ▾</button></span>
        <button class="small" data-act="csv" title="Save the rows you see (after search) as CSV">CSV</button>
      </div>
      <div class="tbl-wrap ${opts.short ? "short" : ""}"><table class="tbl"><thead></thead><tbody></tbody></table></div>
      <div class="tbl-foot"></div>`;
    const input = K.$("input[type=search]", container);
    const thead = K.$("thead", container), tbody = K.$("tbody", container), foot = K.$(".tbl-foot", container);
    const val = (c, r) => (c.get ? c.get(r) : r[c.k]);
    const visible = () => cols.filter((c) => !st.hidden.has(c.k));
    const view = () => {
      let rows = st.rows;
      if (st.q) {
        const q = st.q.toLowerCase();
        rows = rows.filter((r) => cols.some((c) => { const v = val(c, r); return v != null && String(v).toLowerCase().includes(q); }));
      }
      if (st.sort) {
        const c = cols.find((x) => x.k === st.sort.k);
        if (c) {
          const g = c.sort || ((r) => val(c, r));
          rows = rows.slice().sort((a, b) => {
            const x = g(a), y = g(b);
            if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
            return (typeof x === "string" ? x.localeCompare(y) : x - y) * st.sort.dir;
          });
        }
      }
      return rows;
    };
    const render = () => {
      const rows = view();
      const pages = Math.max(1, Math.ceil(rows.length / st.size));
      st.page = K.clamp(st.page, 0, pages - 1);
      const vis = visible();
      thead.innerHTML = `<tr>${vis.map((c) => `<th class="${c.num ? "num" : ""} ${c.nosort ? "" : "sortable"}" data-k="${c.k}" title="${K.esc(c.title || "")}">${K.esc(c.label)}${
        st.sort && st.sort.k === c.k ? `<span class="arr">${st.sort.dir > 0 ? "▲" : "▼"}</span>` : ""}</th>`).join("")}</tr>`;
      const slice = rows.slice(st.page * st.size, (st.page + 1) * st.size);
      tbody.innerHTML = slice.map((r) => {
        const key = opts.key ? opts.key(r) : "";
        return `<tr class="${opts.onRow ? "click" : ""} ${st.selected != null && key === st.selected ? "sel" : ""}" data-key="${K.esc(key)}">${vis.map((c) => {
          const v = val(c, r);
          const shown = c.html ? c.html(v, r) : c.fmt ? K.esc(c.fmt(v, r)) : K.esc(v == null ? "–" : v);
          let bar = "";
          if (c.bar && v != null) {
            const max = typeof c.bar.max === "function" ? c.bar.max() : c.bar.max || 1;
            bar = `<i style="width:${K.clamp(Math.abs(v) / max, 0, 1) * 100}%;background:${c.bar.color ? (typeof c.bar.color === "function" ? c.bar.color(v, r) : c.bar.color) : "var(--accent)"}"></i>`;
          }
          return `<td class="${c.num ? "num" : ""} ${bar ? "cellbar" : ""}">${bar}<span style="position:relative">${shown}</span></td>`;
        }).join("")}</tr>`;
      }).join("") || `<tr><td colspan="${vis.length}" class="empty">No rows.</td></tr>`;
      const cnt = K.$(".count", container);
      if (cnt) cnt.textContent = `${rows.length.toLocaleString()} of ${st.rows.length.toLocaleString()} rows`;
      foot.innerHTML = pages > 1 ? `<button class="small" data-pg="-1" ${st.page ? "" : "disabled"}>‹ Prev</button>
        <span>Page ${st.page + 1} of ${pages}</span><button class="small" data-pg="1" ${st.page < pages - 1 ? "" : "disabled"}>Next ›</button>
        <span class="spacer"></span><label class="inline">Rows <select data-size>${[25, 50, 100, 250, 1000].map((n) => `<option ${n === st.size ? "selected" : ""}>${n}</option>`).join("")}</select></label>` : "";
    };
    thead.addEventListener("click", (e) => {
      const th = e.target.closest("th.sortable");
      if (!th) return;
      const k = th.dataset.k;
      const c = cols.find((x) => x.k === k);
      st.sort = st.sort && st.sort.k === k ? { k, dir: -st.sort.dir } : { k, dir: c && c.num ? -1 : 1 };
      opts.onSort && opts.onSort(st.sort);
      render();
    });
    tbody.addEventListener("click", (e) => {
      const tr = e.target.closest("tr[data-key]");
      if (tr && opts.onRow) { const r = view().find((x) => (opts.key ? opts.key(x) : "") === tr.dataset.key); if (r) opts.onRow(r, e); }
    });
    if (opts.onHover) {
      tbody.addEventListener("mouseover", (e) => { const tr = e.target.closest("tr[data-key]"); if (tr) opts.onHover(tr.dataset.key); });
      tbody.addEventListener("mouseleave", () => opts.onHover(null));
    }
    foot.addEventListener("click", (e) => { const b = e.target.closest("[data-pg]"); if (b) { st.page += +b.dataset.pg; render(); } });
    foot.addEventListener("change", (e) => { if (e.target.matches("[data-size]")) { st.size = +e.target.value; st.page = 0; render(); } });
    if (input) input.addEventListener("input", K.debounce(() => { st.q = input.value.trim(); st.page = 0; render(); }, 150));
    container.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-act]");
      if (!b) return;
      if (b.dataset.act === "csv") {
        const vis = visible();
        K.download(`${opts.exportName || "table"}.csv`, K.csv(view(), vis.map((c) => ({ label: c.label, raw: (r) => (c.raw ? c.raw(r) : val(c, r)) }))), "text/csv");
      } else if (b.dataset.act === "cols") {
        const wrap = b.parentElement;
        if (K.$(".menu", wrap)) { K.$(".menu", wrap).remove(); return; }
        const menu = K.h(`<div class="menu">${cols.map((c) => `<label><input type="checkbox" data-c="${c.k}" ${st.hidden.has(c.k) ? "" : "checked"}> ${K.esc(c.label)}</label>`).join("")}</div>`);
        menu.addEventListener("change", (ev) => { const k = ev.target.dataset.c; ev.target.checked ? st.hidden.delete(k) : st.hidden.add(k); render(); });
        wrap.appendChild(menu);
        setTimeout(() => document.addEventListener("click", function off(ev) { if (!wrap.contains(ev.target)) { menu.remove(); document.removeEventListener("click", off); } }), 0);
      }
    });
    render();
    return {
      update(rows) { st.rows = rows; render(); },
      select(key) { st.selected = key; render(); },
      search(q) { st.q = q; if (input) input.value = q; st.page = 0; render(); },
      rows: () => view(),
      state: st,
      render,
    };
  };

  // ------------------------------------------------------------------ GPU readout (sidebar)
  K.gpu = (el) => {
    if (!el) return;
    const tick = async () => {
      if (document.hidden) return;
      try {
        const g = await K.api("/finetune/api/gpu");
        el.innerHTML = g.available
          ? `<span>${K.esc(g.name || "GPU")}</span><div class="bar"><i style="width:${g.util || 0}%"></i></div>
             <span>${g.util || 0}% busy · ${Math.round((g.mem_used || 0) / 1024 * 10) / 10} / ${Math.round((g.mem_total || 0) / 1024)} GB${g.temp ? ` · ${g.temp}°C` : ""}</span>`
          : `<span>No GPU reading</span>`;
      } catch (e) { el.innerHTML = `<span>GPU: no answer</span>`; }
    };
    tick();
    setInterval(tick, 4000);
  };

  // ------------------------------------------------------------------ polling that pauses in background tabs
  K.every = (fn, ms) => {
    let t = null;
    const loop = async () => { if (!document.hidden) { try { await fn(); } catch (e) { /* next tick */ } } t = setTimeout(loop, ms); };
    t = setTimeout(loop, ms);
    return () => clearTimeout(t);
  };

  // ------------------------------------------------------------------ shell wiring (runs on every page)
  K.boot = () => {
    K.key("ctrl+k", "Open the command palette", K.palette, "Everywhere");
    K.key("?", "Show these shortcuts", K.showKeys, "Everywhere");
    K.key("t", "Switch light / dark theme", K.toggleTheme, "Everywhere");
    K.key("g", "Go to the next page", () => {
      const links = K.$$(".nav a"); const i = links.findIndex((a) => a.getAttribute("aria-current") === "page");
      if (links.length) location.href = links[(i + 1) % links.length].href;
    }, "Everywhere");
    for (const a of K.$$(".nav a")) K.command(`Open page: ${a.textContent.trim()}`, () => { location.href = a.href; }, { group: "Pages" });
    K.command("Switch light / dark theme", K.toggleTheme, { group: "View", hint: "T" });
    K.command("Copy a link to this view", K.copyLink, { group: "View" });
    K.command("Print / save as PDF", () => print(), { group: "View" });
    K.command("Keyboard shortcuts", K.showKeys, { group: "View", hint: "?" });
    const th = K.id("themeBtn");
    if (th) th.addEventListener("click", K.toggleTheme);
    const pb = K.id("paletteBtn");
    if (pb) pb.addEventListener("click", K.palette);
    const kb = K.id("keysBtn");
    if (kb) kb.addEventListener("click", K.showKeys);
    const lb = K.id("linkBtn");
    if (lb) lb.addEventListener("click", K.copyLink);
    K.gpu(K.id("gpuMini"));
    K.toc();
    // card tools: [data-full] expands its card
    document.addEventListener("click", (e) => {
      const b = e.target.closest("[data-full]");
      if (b) K.fullscreen(b.closest(".card"));
    });
  };

  window.Kit = K;
  document.addEventListener("DOMContentLoaded", () => K.boot());
})();
