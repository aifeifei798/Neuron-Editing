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
  edits: {},        // {"12": {"4521": 0}}
  layer: 0,
  mode: "fit",      // fit | zoom
  probe: null,      // {"0": Uint8Array, ...}
  sel: new Set(),
  lastClick: -1,
  hover: -1,
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

const geo = { cols: 0, rows: 0, cell: 0, w: 0, h: 0 };

function layoutMatrix() {
  if (!S.interSize) return;
  const cv = $("matrix");
  const wrap = $("matrix-wrap");
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
  geo.cols = cols;
  geo.rows = rows;
  geo.cell = cell;
  geo.w = cols * cell;
  geo.h = rows * cell;
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
  const cDanger = cssColor("--color-danger");
  const cAccent = cssColor("--color-accent");
  const heat = [];
  for (let i = 0; i <= 8; i++) heat.push(cssColor(`--heat-${i}`));
  for (let n = 0; n < S.interSize; n++) {
    const cx = (n % geo.cols) * geo.cell;
    const cy = Math.floor(n / geo.cols) * geo.cell;
    let fill = idle;
    if (pd) fill = heat[Math.min(8, Math.round((pd[n] * 8) / 255))];
    ctx.fillStyle = fill;
    ctx.fillRect(cx, cy, size, size);
    const s = le[n];
    if (s !== undefined) {
      ctx.fillStyle = s === 0 ? cDanger : cAccent;
      ctx.fillRect(cx, cy, size, size);
    }
  }
  if (S.hover >= 0 && !S.sel.has(S.hover)) strokeCell(ctx, S.hover, 1, cssColor("--color-ink"));
  ctx.strokeStyle = cssColor("--color-accent");
  ctx.lineWidth = 2;
  for (const n of S.sel) strokeCell(ctx, n, 2, cssColor("--color-accent"));
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
  lines.push(s === undefined ? "未编辑" : s === 0 ? "已消融 ×0" : `缩放 ×${s}`);
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
    if (le[n] === undefined) applyEdits([n], 0);
    else removeEdits([n]);
  });
  cv.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      S.sel.clear();
      drawMatrix();
      updateDetail();
    }
  });
  new ResizeObserver(() => layoutMatrix()).observe($("matrix-wrap"));
}

/* ---------------- 编辑操作 ---------------- */

function afterEditsChanged() {
  $("edit-count").textContent = editCount();
  refreshRail();
  drawMatrix();
  updateDetail();
}

async function applyEdits(neurons, scale) {
  try {
    await api("/api/edits/batch", {
      method: "POST",
      body: JSON.stringify({
        layer: S.layer,
        items: neurons.map((n) => ({ neuron: n, scale })),
      }),
    });
    const m = S.edits[S.layer] || (S.edits[S.layer] = {});
    for (const n of neurons) m[n] = scale;
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
        items: neurons.map((n) => ({ neuron: n, scale: null })),
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

/* ---------------- 详情面板 ---------------- */

function updateDetail() {
  const body = $("detail-body");
  body.textContent = "";
  if (!S.interSize) {
    body.textContent = "模型加载完成后可在此编辑所选神经元。";
    return;
  }
  if (S.sel.size === 0) {
    body.textContent = "点击矩阵中的神经元开始编辑。Ctrl+点击多选，Shift+点击范围选择，双击快速消融/恢复，Esc 取消选择。";
    return;
  }

  const ids = [...S.sel].sort((a, b) => a - b);
  const le = S.edits[S.layer] || S.edits[String(S.layer)] || {};

  const head = el("p", "text-sm text-ink");
  const b = el("b", "font-mono");
  b.textContent = String(ids.length);
  head.append("已选 ", b, " 个神经元 · 第 ", document.createTextNode(String(S.layer)), " 层：");
  const code = el("span", "font-mono text-xs text-muted");
  const preview = ids.slice(0, 10).map((n) => `#${n}`).join(" ");
  code.textContent = preview + (ids.length > 10 ? ` … 共 ${ids.length} 个` : "");
  head.appendChild(code);
  body.appendChild(head);

  const current = ids.length === 1 && le[ids[0]] !== undefined ? le[ids[0]] : 1;
  const row = el("div", "mt-2 flex flex-wrap items-center gap-2");
  const lab = el("label", "text-xs text-muted");
  lab.setAttribute("for", "scale-slider");
  lab.textContent = "缩放系数";
  const slider = el("input");
  slider.id = "scale-slider";
  slider.type = "range";
  slider.min = "0";
  slider.max = "2";
  slider.step = "0.05";
  slider.value = String(current);
  slider.className = "w-52";
  const val = el("span", "font-mono text-sm");
  val.textContent = `×${Number(current).toFixed(2)}`;
  slider.addEventListener("input", () => {
    val.textContent = `×${Number(slider.value).toFixed(2)}`;
  });
  slider.addEventListener("change", () => {
    applyEdits(ids, Number(slider.value));
  });
  row.append(lab, slider, val);
  body.appendChild(row);

  const acts = el("div", "mt-2 flex flex-wrap items-center gap-2");
  const mk = (label, fn, cls = "btn btn-ghost") => {
    const btn = el("button", cls);
    btn.type = "button";
    btn.textContent = label;
    btn.addEventListener("click", fn);
    return btn;
  };
  acts.append(
    mk("置零消融", () => applyEdits(ids, 0)),
    mk("减半", () => applyEdits(ids, Number((current / 2).toFixed(2)))),
    mk("复原 ×1", () => applyEdits(ids, 1)),
    mk("加倍", () => applyEdits(ids, Number(Math.min(4, current * 2).toFixed(2)))),
    mk("清除编辑", () => removeEdits(ids), "btn btn-danger"),
  );
  body.appendChild(acts);

  const tools = el("div", "mt-3 flex flex-wrap items-center gap-2");
  const jump = el("input", "input-num w-32");
  jump.type = "number";
  jump.min = "0";
  jump.max = String(S.interSize - 1);
  jump.placeholder = "神经元编号";
  jump.setAttribute("aria-label", "按编号选中神经元");
  const jumpBtn = mk("选中", () => {
    const n = Number(jump.value);
    if (Number.isNaN(n) || n < 0 || n >= S.interSize) {
      toast(`编号超出范围：应在 0 到 ${S.interSize - 1} 之间`);
      return;
    }
    setLayer(S.layer);
    S.sel.add(n);
    S.lastClick = n;
    drawMatrix();
    updateDetail();
  });
  const clrLayer = mk(`清空第 ${S.layer} 层编辑`, () => clearLayer(S.layer), "btn btn-danger");
  tools.append(jump, jumpBtn, clrLayer);
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
  $("btn-clear-all").addEventListener("click", clearAll);
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
