"""Qwen3-4B-Thinking-2507 逐层神经元编辑工作台 — FastAPI 后端。

编辑机制
--------
对每层 MLP 的 ``down_proj`` 注册 forward-pre-hook。down_proj 的输入正是
SwiGLU 神经元激活 ``silu(gate_proj(x)) * up_proj(x)``，形状 ``[B, S, I]``。
把指定下标的激活乘以 scale 即完成编辑：0 = 消融，1 = 原样，>1 = 放大。

启动
----
    .venv/bin/python server.py            # 默认 127.0.0.1:8765
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

MODEL_PATH = os.environ.get("NEURON_MODEL_PATH", "Qwen/Qwen3-4B-Thinking-2507")
HOST = os.environ.get("NEURON_HOST", "127.0.0.1")
PORT = int(os.environ.get("NEURON_PORT", "8788"))
EDITS_PATH = Path(__file__).parent / "edits.json"
STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------

state_lock = threading.Lock()
edits: dict[int, dict[int, float]] = {}  # layer -> neuron -> scale
edits_enabled = True
ready = False
model = None
tokenizer = None
meta: dict = {}

INFERENCE_LOCK = threading.Lock()  # 串行化 probe 与 generate，避免显存争用

# probe 采集
probe_active = False
probe_acc: dict[int, torch.Tensor] = {}
probe_count = 0
probe_layer_idx: dict[int, int] = {}  # hook 闭包 -> layer id（写时确定，读时在锁内）


class AbortCriteria(StoppingCriteria):
    """每个 token 检查一次中止事件，客户端断开时尽快结束生成。"""

    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001
        return self.event.is_set()


# ---------------------------------------------------------------------------
# 编辑持久化
# ---------------------------------------------------------------------------


def _edits_to_json() -> dict:
    return {str(l): {str(n): s for n, s in neurons.items()} for l, neurons in sorted(edits.items())}


def load_edits() -> None:
    global edits
    if EDITS_PATH.exists():
        try:
            raw = json.loads(EDITS_PATH.read_text(encoding="utf-8"))
            edits = {int(l): {int(n): float(s) for n, s in neurons.items()} for l, neurons in raw.items()}
        except (ValueError, OSError) as exc:  # 损坏文件不阻断启动
            print(f"[edits] 读取失败，忽略: {exc}")


def save_edits() -> None:
    tmp = EDITS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_edits_to_json(), ensure_ascii=False), encoding="utf-8")
    tmp.replace(EDITS_PATH)


# ---------------------------------------------------------------------------
# Hook：编辑 + probe 采集（挂在 down_proj 上）
# ---------------------------------------------------------------------------


def _make_hook(layer_id: int):
    def down_proj_pre_hook(module, args):  # noqa: ANN001
        h = args[0]  # [B, S, I] SwiGLU 神经元激活
        global probe_count
        if probe_active:
            acc = probe_acc.get(layer_id)
            if acc is not None:
                # acc 是对 [B, S] 两个维度求和的结果，除数应为本层见过的 token 位置数
                probe_count = h.shape[0] * h.shape[1]
                acc += h.detach().abs().sum(dim=(0, 1), dtype=torch.float32)
        with state_lock:
            layer_edits = edits.get(layer_id)
            enabled = edits_enabled
        if layer_edits and enabled:
            idx = torch.tensor(sorted(layer_edits), device=h.device, dtype=torch.long)
            scales = torch.tensor(
                [layer_edits[int(i)] for i in idx.tolist()], device=h.device, dtype=h.dtype
            )
            h = h.index_copy(-1, idx, h.index_select(-1, idx) * scales)
            return (h,)
        return None

    return down_proj_pre_hook


def register_hooks() -> None:
    probe_acc.clear()
    for layer_id, block in enumerate(model.model.layers):
        probe_acc[layer_id] = torch.zeros(meta["intermediate_size"], dtype=torch.float32, device=model.device)
        block.mlp.down_proj.register_forward_pre_hook(_make_hook(layer_id))


# ---------------------------------------------------------------------------
# 模型加载（后台线程，接口立即可用）
# ---------------------------------------------------------------------------


def load_model() -> None:
    global model, tokenizer, meta, ready, probe_active
    try:
        print(f"[model] loading {MODEL_PATH} ...")
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        mdl.eval()
        with state_lock:
            model, tokenizer = mdl, tok
            cfg = mdl.config
            meta = {
                "num_layers": cfg.num_hidden_layers,
                "hidden_size": cfg.hidden_size,
                "intermediate_size": cfg.intermediate_size,
                "device": str(mdl.device),
                "dtype": str(next(mdl.parameters()).dtype),
            }
        with state_lock:
            probe_active = False
        register_hooks()
        with state_lock:
            ready = True
        print(f"[model] ready in {time.time() - t0:.1f}s on {meta['device']}")
    except Exception as exc:  # noqa: BLE001
        print(f"[model] 加载失败: {exc!r}")
        raise


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class EditItem(BaseModel):
    neuron: int = Field(ge=0)
    scale: float | None = Field(default=None, ge=0.0, le=4.0)


class EditBatch(BaseModel):
    layer: int = Field(ge=0)
    items: list[EditItem]


class EditSingle(BaseModel):
    layer: int = Field(ge=0)
    neuron: int = Field(ge=0)
    scale: float | None = Field(default=None, ge=0.0, le=4.0)


class ProbeRequest(BaseModel):
    text: str = Field(default="The quick brown fox jumps over the lazy dog.", max_length=4000)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    use_edits: bool = True
    max_new_tokens: int = Field(default=1024, ge=8, le=4096)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


def _require_ready() -> None:
    if not ready:
        raise HTTPException(status_code=503, detail="模型仍在加载中，请稍候")


def _validate_neuron(layer: int, neuron: int) -> None:
    if not ready:
        return  # 模型未就绪时允许先挂编辑，维度校验跳过
    if layer >= meta["num_layers"] or neuron >= meta["intermediate_size"]:
        raise HTTPException(status_code=422, detail="layer 或 neuron 超出模型维度")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_edits()
    threading.Thread(target=load_model, daemon=True).start()
    yield


app = FastAPI(title="NeuronScope", lifespan=lifespan)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.get("/api/info")
def api_info():
    with state_lock:
        return {
            "ready": ready,
            "model_path": MODEL_PATH,
            "edits_enabled": edits_enabled,
            "num_edits": sum(len(n) for n in edits.values()),
            "edits": _edits_to_json(),
            **meta,
        }


@app.put("/api/edits/enabled")
def api_edits_enabled(body: dict):
    global edits_enabled
    with state_lock:
        edits_enabled = bool(body.get("enabled", True))
        return {"edits_enabled": edits_enabled}


@app.get("/api/edits")
def api_get_edits():
    with state_lock:
        return _edits_to_json()


@app.put("/api/edits")
def api_set_edit(body: EditSingle):
    _validate_neuron(body.layer, body.neuron)
    with state_lock:
        if body.scale is None:
            edits.get(body.layer, {}).pop(body.neuron, None)
            if not edits.get(body.layer):
                edits.pop(body.layer, None)
        else:
            edits.setdefault(body.layer, {})[body.neuron] = body.scale
        save_edits()
        return {"ok": True, "num_edits": sum(len(n) for n in edits.values())}


@app.post("/api/edits/batch")
def api_batch_edits(body: EditBatch):
    for item in body.items:
        _validate_neuron(body.layer, item.neuron)
    with state_lock:
        layer_map = edits.setdefault(body.layer, {})
        for item in body.items:
            if item.scale is None:
                layer_map.pop(item.neuron, None)
            else:
                layer_map[item.neuron] = item.scale
        if not layer_map:
            edits.pop(body.layer, None)
        save_edits()
        return {"ok": True, "num_edits": sum(len(n) for n in edits.values())}


@app.delete("/api/edits/{layer}")
def api_clear_layer(layer: int):
    with state_lock:
        edits.pop(layer, None)
        save_edits()
        return {"ok": True, "num_edits": sum(len(n) for n in edits.values())}


@app.delete("/api/edits")
def api_clear_all():
    with state_lock:
        edits.clear()
        save_edits()
        return {"ok": True, "num_edits": 0}


@app.post("/api/probe")
def api_probe(body: ProbeRequest):
    """跑一次前向，返回每层每个神经元的平均 |激活|（按层分位数归一化到 0-255）。"""
    global probe_active, probe_count
    _require_ready()
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="探测文本为空，请输入一段话")
    with INFERENCE_LOCK:
        with state_lock:
            probe_active = True
            probe_count = 0
            for acc in probe_acc.values():
                acc.zero_()
        try:
            enc = tokenizer(body.text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**enc)
            with state_lock:
                probe_active = False
                count = max(probe_count, 1)
                data = {}
                for layer_id, acc in probe_acc.items():
                    v = acc / count
                    vmax = torch.quantile(v, 0.995).clamp_min(1e-8)
                    q = (v / vmax).clamp(0.0, 1.0)
                    data[str(layer_id)] = (q.sqrt() * 255.0).to(torch.uint8).tolist()
            return {"tokens": count, "data": data}
        finally:
            with state_lock:
                probe_active = False


def _generate_worker(req: ChatRequest, queue, loop, abort_event: threading.Event) -> None:
    """在工作线程里跑 generate，把增量推回 asyncio 队列；结束放 None 哨兵。"""
    def put(item: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, item)

    try:
        with INFERENCE_LOCK:
            with state_lock:
                global edits_enabled
                prev_flag = edits_enabled
                edits_enabled = req.use_edits
            try:
                t0 = time.time()
                enc = tokenizer.apply_chat_template(
                    [m.model_dump() for m in req.messages],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
                streamer = TextIteratorStreamer(
                    tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=90
                )
                kwargs = dict(
                    max_new_tokens=req.max_new_tokens,
                    do_sample=True,
                    temperature=req.temperature,
                    top_p=0.95,
                    top_k=20,
                    streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([AbortCriteria(abort_event)]),
                )
                if req.temperature is None:
                    kwargs.pop("temperature")
                thread = threading.Thread(target=lambda: model.generate(**enc, **kwargs))
                thread.start()
                n_chunks = 0
                for text in streamer:
                    if not text:
                        continue
                    n_chunks += 1
                    put({"t": "d", "x": text})
                    if abort_event.is_set():
                        break
                thread.join(timeout=150)
                sec = time.time() - t0
                put(
                    {
                        "t": "done",
                        "chunks": n_chunks,
                        "sec": round(sec, 1),
                        "tps": round(n_chunks / sec, 1) if sec > 0 else 0.0,
                        "aborted": abort_event.is_set(),
                    }
                )
            finally:
                with state_lock:
                    edits_enabled = prev_flag
    except Exception as exc:  # noqa: BLE001
        put({"t": "err", "msg": f"{type(exc).__name__}: {exc}"})
    finally:
        loop.call_soon_threadsafe(queue.put_nowait, None)


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    _require_ready()
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(status_code=422, detail="最后一条消息必须是 user")

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    abort_event = threading.Event()

    async def event_stream():
        worker = threading.Thread(
            target=_generate_worker, args=(req, queue, loop, abort_event), daemon=True
        )
        worker.start()
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            abort_event.set()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# 静态前端
# ---------------------------------------------------------------------------


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
