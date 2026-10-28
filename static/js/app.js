/* NeuronScope 前端：层选择 + 神经元矩阵画布 + 编辑面板 + SSE 对话 */
"use strict";

const $ = (id) => document.getElementById(id);
const el = (tag, cls) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  return n;
};

/* 颜色只来自 token：JS 里通过 getComputedStyle 解析 CSS 变量 */
const colorCache = new Map();
function cssColor(name) {
  if (colorCache.has(name)) return colorCache.get(name);
  const probe = el("span");
  probe.style.color = `var(${name})`;
  probe.style.display = "none";
  document.body.appendChild(probe);
  const c = getComputedStyle(probe).color;
  probe.remove();
  colorCache.set(name, c);
  return c;
}

const S = {
  ready: false,
  numLayers: 0,
  interSize: 0,
  edits: {},        // {"12": {"4521": {"op":"mul"|"add", "value": number}}}
  editMode: "mul",  // 详情面板当前编辑模式：mul=乘法倍率，add=加性偏移
  layer: 0,
  mode: "fit",      // fit | zoom
  probe: null,      // {"0": Uint8Array, ...}
  sel: new Set(),
  lastClick: -1,
  hover: -1,
  flash: null,      // {n} 定位高亮（十字线），短暂显示后清除
  busy: false,
  ctrl: null,
};

/* ---------------- 基础 API ---------------- */

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch { /* 保留状态码信息 */ }
    throw new Error(msg);
  }
  return res.json();
}

function toast(msg) {
  const box = el("div", "toast");
  box.textContent = msg;
  $("toasts").appendChild(box);
  setTimeout(() => box.remove(), 5000);
}

/* ---------------- 信息轮询与总览 ---------------- */

function editCount() {
  return Object.values(S.edits).reduce((a, m) => a + Object.keys(m).length, 0);
}

function editLabel(e) {
  if (!e) return "未编辑";
  if (e.op === "add") {
    return `${e.value >= 0 ? "+" : ""}${Number(e.value).toFixed(2)}（加性）`;
  }
  if (e.value === 0) return "已消融 ×0";
  if (e.value < 0) return `×${Number(e.value).toFixed(2)}（翻转）`;
  return `×${Number(e.value).toFixed(2)}`;
}

function editColorName(e) {
  if (e.op === "add") return "--color-ok";
  if (e.value === 0) return "--color-danger";
  if (e.value < 0) return "--color-warn";
  return "--color-accent";
}

function applyInfo(info) {
  S.ready = info.ready;
  if (info.num_layers) S.numLayers = info.num_layers;
  if (info.intermediate_size) S.interSize = info.intermediate_size;
  S.edits = info.edits || {};
  const chip = $("model-chip");
  if (info.ready) {
    chip.removeAttribute("aria-busy");
    chip.textContent = "模型就绪";
    chip.classList.add("text-ok");
    $("dim-chip").textContent =
      `${info.num_layers} 层 × ${info.intermediate_size} 神经元 · ${info.device} · ${info.dtype}`;
    $("dim-chip").classList.remove("hidden");
  } else {
    chip.textContent = "模型加载中…";
  }
  $("edit-count").textContent = editCount();
}

async function pollInfo() {
  try {
    applyInfo(await api("/api/info"));
    if (!S.ready) setTimeout(pollInfo, 1500);
    else onReady();
  } catch {
    setTimeout(pollInfo, 3000);
  }
}

function onReady() {
  $("btn-probe").disabled = false;
  $("btn-send").disabled = false;
  $("locate-layer").max = String(S.numLayers - 1);
  $("locate-neuron").max = String(S.interSize - 1);
  $("matrix-empty").classList.add("hidden");
  buildRail();
  layoutMatrix();
  drawMatrix();
  updateDetail();
  setLayer(0);
}

/* ---------------- 层轨道 ---------------- */

function buildRail() {
  const rail = $("layer-rail");
  rail.textContent = "";
  for (let l = 0; l < S.numLayers; l++) {
    const b = el("button", "layer-btn");
    b.type = "button";
    b.dataset.layer = l;
    b.textContent = l;
    b.title = `第 ${l} 层`;
    b.addEventListener("click", () => setLayer(l));
    rail.appendChild(b);
  }
  refreshRail();
}

function refreshRail() {
  for (const b of $("layer-rail").children) {
    const l = Number(b.dataset.layer);
    const has = (S.edits[l] && Object.keys(S.edits[l]).length > 0) || false;
    let dot = b.querySelector(".dot");
    if (has && !dot) {
      dot = el("span", "dot absolute right-0.5 top-0.5 h-1.5 w-1.5 rounded-full bg-danger");
      b.appendChild(dot);
    } else if (!has && dot) {
      dot.remove();
    }
  }
}

function setLayer(l) {
  S.layer = l;
  S.sel.clear();
  S.lastClick = -1;
  S.hover = -1;
  for (const b of $("layer-rail").children) {
    b.setAttribute("aria-pressed", String(Number(b.dataset.layer) === l));
    if (Number(b.dataset.layer) === l) b.scrollIntoView({ block: "nearest", inline: "nearest" });
  }
  $("layer-desc").textContent = `第 ${l} 层 · ${S.interSize} 神经元`;
  $("matrix").setAttribute("aria-label", `第 ${l} 层，共 ${S.interSize} 个神经元的激活矩阵`);
  hideTooltip();
  layoutMatrix();
  drawMatrix();
  updateDetail();
}

/* ---------------- 矩阵画布 ---------------- */

const geo = { cols: 0, rows: 0, cell: 0, w: 0, h: 0, mode: "" };

function layoutMatrix(force = false) {
  if (!S.interSize) return;
  const cv = $("matrix");
  const wrap = $("matrix-wrap");
  // 适配模式画布必然放得下，禁用滚动以消除"滚动条出现→容器变小→重算→再变化"的反馈循环
  wrap.style.overflow = S.mode === "fit" ? "clip" : "auto";
  const pad = 32; // p-4 左右
  const availW = Math.max(80, wrap.clientWidth - pad);
  const availH = Math.max(80, wrap.clientHeight - 32);
  const N = S.interSize;
  let cols, cell;
  if (S.mode === "zoom") {
    cell = 16;
    cols = Math.max(8, Math.floor(availW / cell));
  } else {
    cols = Math.max(1, Math.round(Math.sqrt((N * availW) / Math.max(availH, 1))));
    cell = Math.min(availW / cols, availH / Math.ceil(N / cols));
  }
  const rows = Math.ceil(N / cols);
  // 画布 CSS 尺寸取整，杜绝亚像素溢出触发滚动条
  const w = Math.floor(cols * cell);
  const h = Math.floor(rows * cell);
  const unchanged =
    !force &&
    geo.mode === S.mode &&
    geo.cols === cols &&
    geo.rows === rows &&
    Math.abs(geo.cell - cell) < 0.01 &&
    geo.w === w &&
    geo.h === h;
  geo.mode = S.mode;
  geo.cols = cols;
  geo.rows = rows;
  geo.cell = cell;
  geo.w = w;
  geo.h = h;
  if (unchanged) return;
  const dpr = window.devicePixelRatio || 1;
  cv.style.width = `${geo.w}px`;
  cv.style.height = `${geo.h}px`;
  cv.style.marginInline = S.mode === "fit" ? "auto" : "0";
  cv.width = Math.round(geo.w * dpr);
  cv.height = Math.round(geo.h * dpr);
  drawMatrix();
}

function strokeCell(ctx, n, width, color) {
  const cx = (n % geo.cols) * geo.cell;
  const cy = Math.floor(n / geo.cols) * geo.cell;
  ctx.lineWidth = width;
  ctx.strokeStyle = color;
  ctx.strokeRect(cx + 1, cy + 1, geo.cell - 2, geo.cell - 2);
}

function drawMatrix() {
  if (!S.interSize || !geo.cols) return;
  const cv = $("matrix");
  const ctx = cv.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, geo.w, geo.h);
  const pd = S.probe ? S.probe[String(S.layer)] : null;
  const le = S.edits[S.layer] || S.edits[String(S.layer)] || {};
  const gap = geo.cell >= 6 ? 1 : 0.5;
  const size = geo.cell - gap;
  const idle = cssColor("--color-raise");
  const cSelect = cssColor("--color-select");
  const heat = [];
  for (let i = 0; i <= 8; i++) heat.push(cssColor(`--heat-${i}`));
  for (let n = 0; n < S.interSize; n++) {
    const cx = (n % geo.cols) * geo.cell;
    const cy = Math.floor(n / geo.cols) * geo.cell;
    let fill = idle;
    if (pd) fill = heat[Math.min(8, Math.round((pd[n] * 8) / 255))];
    ctx.fillStyle = fill;
    ctx.fillRect(cx, cy, size, size);
    const e = le[n];
    if (e) {
      ctx.fillStyle = cssColor(editColorName(e));
      ctx.fillRect(cx, cy, size, size);
    }
  }
  if (S.hover >= 0 && !S.sel.has(S.hover)) strokeCell(ctx, S.hover, 1, cssColor("--color-ink"));
  // 选中环画在格外圈，格内的编辑色（红/青）保持完整可见
  ctx.strokeStyle = cSelect;
  ctx.lineWidth = 1.5;
  for (const n of S.sel) {
    const cx = (n % geo.cols) * geo.cell;
    const cy = Math.floor(n / geo.cols) * geo.cell;
    ctx.strokeRect(cx - 1.25, cy - 1.25, geo.cell + 2.5, geo.cell + 2.5);
  }
  if (S.flash && S.flash.n >= 0) {
    const cx = (S.flash.n % geo.cols) * geo.cell + geo.cell / 2;
    const cy = Math.floor(S.flash.n / geo.cols) * geo.cell + geo.cell / 2;
    ctx.strokeStyle = cSelect;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, cy);
    ctx.lineTo(geo.w, cy);
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, geo.h);
    ctx.stroke();
  }
}

function neuronAt(offsetX, offsetY) {
  const col = Math.floor(offsetX / geo.cell);
  const row = Math.floor(offsetY / geo.cell);
  if (col < 0 || row < 0 || col >= geo.cols || row >= geo.rows) return -1;
  const n = row * geo.cols + col;
  return n < S.interSize ? n : -1;
}

function hideTooltip() {
  $("tooltip").classList.add("hidden");
}

function showTooltip(n, clientX, clientY) {
  const tt = $("tooltip");
  const wrap = $("matrix-wrap");
  const wr = wrap.getBoundingClientRect();
  const pd = S.probe ? S.probe[String(S.layer)] : null;
  const le = S.edits[S.layer] || S.edits[String(S.layer)] || {};
  const s = le[n];
  const lines = [`#${n}`];
  if (pd) lines.push(`激活 ${((pd[n] / 255) * 100).toFixed(0)}%（层内相对）`);
  lines.push(editLabel(s));
  tt.textContent = lines.join(" · ");
  tt.classList.remove("hidden");
  const x = Math.min(clientX - wr.left + 14, wr.width - tt.offsetWidth - 8);
  const y = Math.min(clientY - wr.top + 14, wr.height - tt.offsetHeight - 8);
  tt.style.left = `${Math.max(4, x)}px`;
  tt.style.top = `${Math.max(4, y)}px`;
}

function bindMatrix() {
  const cv = $("matrix");
  cv.addEventListener("pointermove", (e) => {
    const n = neuronAt(e.offsetX, e.offsetY);
    if (n !== S.hover) {
      S.hover = n;
      drawMatrix();
    }
    if (n >= 0) showTooltip(n, e.clientX, e.clientY);
    else hideTooltip();
  });
  cv.addEventListener("pointerleave", () => {
    S.hover = -1;
    hideTooltip();
    drawMatrix();
  });
  cv.addEventListener("click", (e) => {
    const n = neuronAt(e.offsetX, e.offsetY);
    if (n < 0) return;
    if (e.ctrlKey || e.metaKey) {
      if (S.sel.has(n)) S.sel.delete(n);
      else S.sel.add(n);
      S.lastClick = n;
    } else if (e.shiftKey && S.lastClick >= 0) {
      S.sel.clear();
      const [a, b] = [Math.min(S.lastClick, n), Math.max(S.lastClick, n)];
      for (let i = a; i <= b; i++) S.sel.add(i);
    } else {
      S.sel.clear();
      S.sel.add(n);
      S.lastClick = n;
    }
    drawMatrix();
    updateDetail();
  });
  cv.addEventListener("dblclick", (e) => {
    const n = neuronAt(e.offsetX, e.offsetY);
    if (n < 0) return;
    const le = S.edits[S.layer] || S.edits[String(S.layer)] || {};
    if (!le[n]) applyEdits([n], "mul", 0);
    else removeEdits([n]);
  });
  cv.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      S.sel.clear();
      drawMatrix();
      updateDetail();
    }
  });
  let roScheduled = false;
  new ResizeObserver(() => {
    if (roScheduled) return;
    roScheduled = true;
    requestAnimationFrame(() => {
      roScheduled = false;
      layoutMatrix();
    });
  }).observe($("matrix-wrap"));
}

/* ---------------- 编辑操作 ---------------- */

function afterEditsChanged() {
  $("edit-count").textContent = editCount();
  refreshRail();
  drawMatrix();
  updateDetail();
}

async function applyEdits(neurons, op, value) {
  try {
    await api("/api/edits/batch", {
      method: "POST",
      body: JSON.stringify({
        layer: S.layer,
        items: neurons.map((n) => ({ neuron: n, op, value })),
      }),
    });
    const m = S.edits[S.layer] || (S.edits[S.layer] = {});
    for (const n of neurons) m[n] = { op, value };
    afterEditsChanged();
  } catch (e) {
    toast(`编辑失败：${e.message}`);
  }
}

async function removeEdits(neurons) {
  try {
    await api("/api/edits/batch", {
      method: "POST",
      body: JSON.stringify({
        layer: S.layer,
        items: neurons.map((n) => ({ neuron: n, op: "mul", value: null })),
      }),
    });
    const m = S.edits[S.layer] || {};
    for (const n of neurons) delete m[n];
    if (!Object.keys(m).length) delete S.edits[S.layer];
    afterEditsChanged();
  } catch (e) {
    toast(`清除失败：${e.message}`);
  }
}

async function clearLayer(layer) {
  try {
    await api(`/api/edits/${layer}`, { method: "DELETE" });
    delete S.edits[layer];
    afterEditsChanged();
  } catch (e) {
    toast(`清空失败：${e.message}`);
  }
}

async function clearAll() {
  try {
    await api("/api/edits", { method: "DELETE" });
    S.edits = {};
    afterEditsChanged();
    toast("已清空全部编辑");
  } catch (e) {
    toast(`清空失败：${e.message}`);
  }
}

/* ---------------- 确认弹窗（清空全部编辑） ---------------- */

let confirmReturnFocus = null;

function openConfirmClear() {
  if (editCount() === 0) {
    toast("当前没有任何编辑，无需清空");
    return;
  }
  $("confirm-clear-msg").textContent =
    `将清除全部 ${editCount()} 处编辑，矩阵标记一并消失，edits.json 会同步写回。确定清空吗？`;
  confirmReturnFocus = document.activeElement;
  $("confirm-clear").classList.remove("hidden");
  $("app-root").inert = true;
  $("confirm-clear-cancel").focus();
}

function closeConfirmClear() {
  $("confirm-clear").classList.add("hidden");
  $("app-root").inert = false;
  if (confirmReturnFocus && typeof confirmReturnFocus.focus === "function") {
    confirmReturnFocus.focus();
  }
  confirmReturnFocus = null;
}

function bindConfirmClear() {
  $("btn-clear-all").addEventListener("click", openConfirmClear);
  $("confirm-clear-cancel").addEventListener("click", closeConfirmClear);
  $("confirm-clear-backdrop").addEventListener("click", closeConfirmClear);
  $("confirm-clear-ok").addEventListener("click", async () => {
    closeConfirmClear();
    await clearAll();
  });
  document.addEventListener("keydown", (e) => {
    const dlg = $("confirm-clear");
    if (dlg.classList.contains("hidden")) return;
    if (e.key === "Escape") {
      e.preventDefault();
      closeConfirmClear();
    } else if (e.key === "Tab") {
      const ring = [$("confirm-clear-cancel"), $("confirm-clear-ok")];
      const idx = ring.indexOf(document.activeElement);
      e.preventDefault();
      const next = e.shiftKey
        ? ring[(idx - 1 + ring.length) % ring.length]
        : ring[(idx + 1) % ring.length];
      next.focus();
    }
  });
}

/* ---------------- 定位 ---------------- */

function scrollToNeuron(n) {
  const wrap = $("matrix-wrap");
  const x = (n % geo.cols) * geo.cell;
  const y = Math.floor(n / geo.cols) * geo.cell;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  wrap.scrollTo({
    left: Math.max(0, x + geo.cell / 2 - wrap.clientWidth / 2),
    top: Math.max(0, y + geo.cell / 2 - wrap.clientHeight / 2),
    behavior: reduced ? "auto" : "smooth",
  });
}

function locateNeuron() {
  const layerRaw = $("locate-layer").value.trim();
  const neuronRaw = $("locate-neuron").value.trim();
  if (layerRaw === "" || neuronRaw === "") {
    toast("请先输入层号和神经元编号，再点定位");
    return;
  }
  const L = Number(layerRaw);
  const N = Number(neuronRaw);
  if (!Number.isInteger(L) || L < 0 || L >= S.numLayers) {
    toast(`层号超出范围：应在 0 到 ${S.numLayers - 1} 之间，请改正后重试`);
    return;
  }
  if (!Number.isInteger(N) || N < 0 || N >= S.interSize) {
    toast(`神经元编号超出范围：应在 0 到 ${S.interSize - 1} 之间，请改正后重试`);
    return;
  }
  setLayer(L);
  S.sel.add(N);
  S.lastClick = N;
  scrollToNeuron(N);
  S.flash = { n: N };
  drawMatrix();
  updateDetail();
  clearTimeout(S.flashTimer);
  S.flashTimer = setTimeout(() => {
    S.flash = null;
    drawMatrix();
  }, 1600);
}

/* ---------------- 详情面板 ---------------- */

function updateDetail() {
  const body = $("detail-body");
  body.textContent = "";
  if (!S.interSize) {
    body.textContent = "模型加载完成后可在此编辑所选神经元。";
    return;
  }
  if (S.sel.size === 0) {
    body.textContent =
      "点击矩阵中的神经元开始编辑；Ctrl+单击多选，Shift+单击范围选择，双击快速消融/恢复，Esc 取消选择。也可以在上方输入层号和神经元编号后点「定位」直达。";
    const entries = [];
    for (const [l, neurons] of Object.entries(S.edits)) {
      for (const [n, e] of Object.entries(neurons)) entries.push({ l: Number(l), n: Number(n), e });
    }
    if (entries.length) {
      const label = el("p", "mt-2 text-xs text-muted");
      label.textContent = `当前全部编辑（${entries.length} 处，点击跳转定位）：`;
      body.appendChild(label);
      const box = el("div", "mt-1 flex max-h-28 flex-wrap gap-1.5 overflow-y-auto");
      for (const { l, n, e } of entries.sort((a, b) => a.l - b.l || a.n - b.n)) {
        const chip = el("button", "btn btn-ghost btn-sm-mode");
        chip.type = "button";
        chip.textContent = `L${l}#${n} ${editLabel(e)}`;
        chip.title = "点击定位并选中该神经元";
        chip.addEventListener("click", () => {
          $("locate-layer").value = String(l);
          $("locate-neuron").value = String(n);
          locateNeuron();
        });
        box.appendChild(chip);
      }
      body.appendChild(box);
    }
    return;
  }

  const ids = [...S.sel].sort((a, b) => a - b);
  const le = S.edits[S.layer] || S.edits[String(S.layer)] || {};

  if (ids.length === 1) {
    if (document.activeElement !== $("locate-layer")) $("locate-layer").value = String(S.layer);
    if (document.activeElement !== $("locate-neuron")) $("locate-neuron").value = String(ids[0]);
  }

  const head = el("p", "text-sm text-ink");
  const b = el("b", "font-mono");
  b.textContent = String(ids.length);
  head.append("已选 ", b, " 个神经元 · 第 ", document.createTextNode(String(S.layer)), " 层：");
  const code = el("span", "font-mono text-xs text-muted");
  const preview = ids.slice(0, 10).map((n) => `#${n}`).join(" ");
  code.textContent = preview + (ids.length > 10 ? ` … 共 ${ids.length} 个` : "");
  head.appendChild(code);
  body.appendChild(head);

  const mode = S.editMode;
  const current = ids.length === 1 ? le[ids[0]] : null;
  const base = current && current.op === mode ? current.value : mode === "mul" ? 1 : 0;
  const clampSlider = (v) =>
    mode === "mul" ? Math.min(3, Math.max(0, v)) : Math.min(5, Math.max(-5, v));
  const fmt = (v) =>
    mode === "mul" ? `×${Number(v).toFixed(2)}` : `${v >= 0 ? "+" : ""}${Number(v).toFixed(2)}`;

  const row = el("div", "mt-2 flex flex-wrap items-center gap-2");
  const mkMode = (m, label) => {
    const btn = el("button", "btn btn-ghost btn-sm-mode");
    btn.type = "button";
    btn.textContent = label;
    btn.setAttribute("aria-pressed", String(mode === m));
    btn.addEventListener("click", () => {
      S.editMode = m;
      updateDetail();
    });
    return btn;
  };
  row.append(mkMode("mul", "乘 ×"), mkMode("add", "加 +"));

  const lab = el("label", "text-xs text-muted");
  lab.setAttribute("for", "scale-slider");
  lab.textContent = mode === "mul" ? "缩放倍率" : "加性偏移";
  const slider = el("input");
  slider.id = "scale-slider";
  slider.type = "range";
  slider.min = mode === "mul" ? "0" : "-5";
  slider.max = mode === "mul" ? "3" : "5";
  slider.step = "0.05";
  slider.value = String(clampSlider(base));
  slider.className = "w-52";
  const num = el("input", "input-num w-24");
  num.type = "number";
  num.id = "scale-num";
  num.min = "-100";
  num.max = "100";
  num.step = "0.05";
  num.value = Number(base).toFixed(2);
  num.setAttribute("aria-label", "精确编辑数值（-100 到 100）");
  const val = el("span", "font-mono text-sm");
  val.textContent = fmt(base);
  slider.addEventListener("input", () => {
    num.value = Number(slider.value).toFixed(2);
    val.textContent = fmt(slider.value);
  });
  const commit = (raw) => {
    const v = Number(raw);
    if (Number.isNaN(v) || v < -100 || v > 100) {
      toast("数值超出范围：应在 -100 到 100 之间，请改正后重试");
      return;
    }
    if (mode === "add" && v === 0) {
      removeEdits(ids);
      return;
    }
    applyEdits(ids, mode, v);
  };
  slider.addEventListener("change", () => commit(slider.value));
  num.addEventListener("change", () => commit(num.value));
  num.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commit(num.value);
    }
  });
  row.append(lab, slider, num, val);
  body.appendChild(row);

  const acts = el("div", "mt-2 flex flex-wrap items-center gap-2");
  const mk = (label, fn, cls = "btn btn-ghost") => {
    const btn = el("button", cls);
    btn.type = "button";
    btn.textContent = label;
    btn.addEventListener("click", fn);
    return btn;
  };
  if (mode === "mul") {
    acts.append(
      mk("置零消融", () => applyEdits(ids, "mul", 0)),
      mk("减半", () => applyEdits(ids, "mul", Number((base / 2).toFixed(2)))),
      mk("复原 ×1", () => applyEdits(ids, "mul", 1)),
      mk("加倍", () => applyEdits(ids, "mul", Number(Math.min(100, base * 2).toFixed(2)))),
    );
  } else {
    acts.append(
      mk("+1", () => applyEdits(ids, "add", Number((base + 1).toFixed(2)))),
      mk("−1", () => applyEdits(ids, "add", Number((base - 1).toFixed(2)))),
      mk("反转符号", () => applyEdits(ids, "add", Number((-base).toFixed(2)))),
    );
  }
  acts.append(mk("清除编辑", () => removeEdits(ids), "btn btn-danger"));
  body.appendChild(acts);

  const tools = el("div", "mt-3 flex flex-wrap items-center gap-2");
  const clrLayer = mk(`清空第 ${S.layer} 层编辑`, () => clearLayer(S.layer), "btn btn-danger");
  tools.append(clrLayer);
  body.appendChild(tools);
}

/* ---------------- 探测 ---------------- */

async function runProbe() {
  if (S.busy) return;
  const text = $("probe-text").value.trim();
  if (!text) {
    toast("探测文本为空，请先输入一段话");
    return;
  }
  const btn = $("btn-probe");
  btn.disabled = true;
  btn.textContent = "探测中…";
  try {
    const r = await api("/api/probe", {
      method: "POST",
      body: JSON.stringify({ text }),
    });
    S.probe = r.data;
    drawMatrix();
    toast(`探测完成：${r.tokens} 个 token，已更新 ${Object.keys(r.data).length} 层热区`);
  } catch (e) {
    toast(`探测失败：${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "探测激活";
  }
}

/* ---------------- 对话 ---------------- */

function addTurn(label) {
  $("chat-empty")?.classList.add("hidden");
  const root = el("article", "mt-4 min-w-0 border-t border-border pt-4 first:mt-0 first:border-t-0 first:pt-0");
  const head = el("div", "font-mono text-xs text-muted");
  head.textContent = label;
  const think = el("details", "mt-1");
  think.open = false;
  const sum = el("summary", "cursor-pointer select-none text-xs text-muted");
  sum.textContent = "思考过程";
  const thinkBody = el("div", "mt-1 whitespace-pre-wrap break-words text-sm leading-relaxed text-muted");
  think.append(sum, thinkBody);
  const answer = el("div", "mt-2 whitespace-pre-wrap break-words text-base leading-relaxed");
  const stats = el("div", "mt-2 font-mono text-xs text-muted");
  root.append(head, think, answer, stats);
  return { root, head, think, thinkBody, answer, stats };
}

function renderStream(block, full, phase) {
  const idx = full.indexOf("</think>");
  const thinkText = idx >= 0 ? full.slice(0, idx).trim() : full;
  const answerText = idx >= 0 ? full.slice(idx + 8).trim() : "";
  if (thinkText) {
    block.think.classList.remove("hidden");
    block.thinkBody.textContent = thinkText;
    block.think.open = phase === "think";
  }
  if (answerText) block.answer.textContent = answerText;
  return answerText;
}

async function streamInto(block, useEdits, userText) {
  const payload = {
    messages: [...history, { role: "user", content: userText }],
    use_edits: useEdits,
    max_new_tokens: Number($("max-tokens").value) || 1024,
  };
  const t = $("temperature").value;
  if (t !== "") payload.temperature = Number(t);

  S.ctrl = new AbortController();
  $("btn-stop").classList.remove("hidden");
  setBusy(true);
  let full = "";
  let phase = "think";
  let stat = null;
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: S.ctrl.signal,
    });
    if (!res.ok || !res.body) {
      let msg = `${res.status} ${res.statusText}`;
      try {
        const j = await res.json();
        if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
      } catch { /* 保留状态码 */ }
      throw new Error(msg);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        if (!raw.startsWith("data: ")) continue;
        const evt = JSON.parse(raw.slice(6));
        if (evt.t === "d") {
          full += evt.x;
          phase = full.includes("</think>") ? "answer" : "think";
          renderStream(block, full, phase);
        } else if (evt.t === "done") {
          stat = evt;
        } else if (evt.t === "err") {
          throw new Error(evt.msg);
        }
      }
    }
    const answerText = renderStream(block, full, "answer");
    if (stat) {
      block.stats.textContent = stat.aborted
        ? "已中止"
        : `${stat.chunks} tokens · ${stat.tps} tok/s · ${stat.sec}s`;
    }
    return answerText;
  } catch (e) {
    const aborted = e.name === "AbortError";
    const partial = renderStream(block, full, "answer");
    block.stats.textContent = aborted ? "已中止" : `出错：${e.message}`;
    block.stats.classList.toggle("text-danger", !aborted);
    if (!aborted) toast(`生成失败：${e.message}`);
    return partial;
  } finally {
    $("btn-stop").classList.add("hidden");
    setBusy(false);
    S.ctrl = null;
  }
}

function setBusy(v) {
  S.busy = v;
  $("btn-send").disabled = v || !S.ready;
  $("chat-input").disabled = false;
  $("chat-hint").textContent = v ? "生成中…" : "";
}

let history = [];

async function send() {
  if (S.busy || !S.ready) return;
  const text = $("chat-input").value.trim();
  if (!text) {
    toast("消息为空，请输入内容");
    return;
  }
  $("chat-input").value = "";
  history.push({ role: "user", content: text });
  addTurn("用户").root.append(document.createTextNode(text));
  const messagesBox = $("messages");
  messagesBox.scrollTop = messagesBox.scrollHeight;

  if ($("compare").checked) {
    const grid = el("div", "mt-4 grid min-w-0 gap-4 border-t border-border pt-4 lg:grid-cols-2");
    const b1 = addTurn("模型 · 基线");
    const b2 = addTurn("模型 · 应用编辑");
    b1.root.classList.remove("mt-4", "border-t", "border-border", "pt-4", "first:mt-0", "first:border-t-0", "first:pt-0");
    b2.root.classList.remove("mt-4", "border-t", "border-border", "pt-4", "first:mt-0", "first:border-t-0", "first:pt-0");
    grid.append(b1.root, b2.root);
    $("messages").appendChild(grid);
    messagesBox.scrollTop = messagesBox.scrollHeight;
    const a1 = await streamInto(b1, false, text);
    const a2 = await streamInto(b2, true, text);
    history.push({ role: "assistant", content: a2 || a1 });
  } else {
    const block = addTurn($("use-edits").checked ? "模型 · 应用编辑" : "模型 · 基线");
    $("messages").appendChild(block.root);
    messagesBox.scrollTop = messagesBox.scrollHeight;
    const answer = await streamInto(block, $("use-edits").checked, text);
    history.push({ role: "assistant", content: answer });
  }
  messagesBox.scrollTop = messagesBox.scrollHeight;
}

function stopGen() {
  if (S.ctrl) S.ctrl.abort();
}

/* ---------------- 绑定与启动 ---------------- */

function bindControls() {
  $("btn-probe").addEventListener("click", runProbe);
  bindConfirmClear();
  $("btn-send").addEventListener("click", (e) => {
    e.preventDefault();
    send();
  });
  $("btn-stop").addEventListener("click", stopGen);
  $("chat-form").addEventListener("submit", (e) => e.preventDefault());
  $("chat-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  $("mode-fit").addEventListener("click", () => setMode("fit"));
  $("mode-zoom").addEventListener("click", () => setMode("zoom"));
  $("locate-form").addEventListener("submit", (e) => {
    e.preventDefault();
    locateNeuron();
  });
}

function setMode(m) {
  S.mode = m;
  $("mode-fit").setAttribute("aria-pressed", String(m === "fit"));
  $("mode-zoom").setAttribute("aria-pressed", String(m === "zoom"));
  layoutMatrix();
}

bindMatrix();
bindControls();
pollInfo();

/* 调试/自动化句柄：浏览器控制台可用 NS.S 查看状态 */
window.NS = { S, geo, layoutMatrix, drawMatrix, updateDetail };
