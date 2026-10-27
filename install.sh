#!/usr/bin/env bash
# Neuron-Editing 环境安装：Python 依赖 + 前端 CSS 构建（可重复执行，已装的会跳过）
set -euo pipefail
cd "$(dirname "$0")"

echo "==> [1/3] 检查 uv 与虚拟环境"
if ! command -v uv >/dev/null 2>&1; then
  echo "错误：未找到 uv，请先安装：curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
else
  echo "    .venv 已存在，跳过创建"
fi
PY=.venv/bin/python

echo "==> [2/3] 安装 Python 依赖"
uv pip install --python "$PY" fastapi "uvicorn[standard]" accelerate
if "$PY" -c "import torch" >/dev/null 2>&1; then
  echo "    torch 已安装，跳过"
else
  echo "    未检测到 torch，开始安装（体积较大，需几分钟）…"
  uv pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cu130
fi
if "$PY" -c "import transformers" >/dev/null 2>&1; then
  echo "    transformers 已安装，跳过"
else
  uv pip install --python "$PY" transformers
fi

echo "==> [3/3] 构建前端 CSS（Tailwind v4）"
if ! command -v npm >/dev/null 2>&1; then
  echo "错误：未找到 npm，请先安装 Node.js（含 npm）" >&2
  exit 1
fi
npm install --no-audit --no-fund
npm run build:css

"$PY" -c "import torch; print('    torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
"$PY" -c "import transformers; print('    transformers', transformers.__version__)"
echo "安装完成。运行 ./server.sh start 启动服务（默认 http://127.0.0.1:8788，./server.sh 不带参数看用法）"
