#!/usr/bin/env bash
# NeuronScope 服务控制脚本（统一入口）
set -euo pipefail
cd "$(dirname "$0")"

PID_FILE=server.pid
LOG_FILE=server.log
# 本地私有配置（不入 git）：可覆盖 NEURON_MODEL_PATH 等，测试环境用它指向本地模型
if [ -f .env.local ]; then
  set -a
  . ./.env.local
  set +a
fi
PORT="${NEURON_PORT:-8788}"   # 8765 被本机其他进程占用，勿用
PY=.venv/bin/python

usage() {
  cat <<'EOF'
用法: ./server.sh <命令>
  start     后台启动服务（nohup，日志追加到 server.log，PID 记录在 server.pid）
  stop      停止服务（先 SIGTERM，10 秒后仍存活则 SIGKILL）
  restart   重启服务
  status    查看运行状态与模型加载进度
  log [N]   查看最近 N 行日志（默认 50）
  logf      实时跟踪日志（Ctrl+C 退出，不影响服务）
  fg        前台启动（调试用，Ctrl+C 停止）

环境变量: NEURON_PORT=8788  NEURON_HOST=127.0.0.1  NEURON_MODEL_PATH=...
EOF
}

info() { echo "── $*"; }
err()  { echo "错误：$*" >&2; }

is_running() {
  [ -f "$PID_FILE" ] || return 1
  local pid
  pid=$(cat "$PID_FILE" 2>/dev/null) || return 1
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

# 兜底：找本项目目录下游离的 server.py 进程（PID 文件丢失/上次异常退出时）
stray_pids() {
  local pids="" here
  here=$(pwd)
  for pid in $(pgrep -f "server\.py" 2>/dev/null || true); do
    if [ "$(readlink -f "/proc/$pid/cwd" 2>/dev/null)" = "$here" ]; then
      pids="$pids $pid"
    fi
  done
  echo $pids
}

health() {
  local r
  for _ in $(seq 1 10); do
    r=$(curl -s -m 2 "http://127.0.0.1:$PORT/api/info" 2>/dev/null || true)
    if [ -n "$r" ]; then
      if printf '%s' "$r" | grep -q '"ready":true'; then
        info "健康检查：模型已就绪 → http://127.0.0.1:$PORT"
      else
        info "健康检查：HTTP 已通，模型加载中（几十秒后可用，./server.sh status 看进度）"
      fi
      return
    fi
    sleep 1
  done
  info "健康检查：端口 $PORT 暂未响应（首次加载较慢，可用 ./server.sh logf 看进度）"
}

cmd_start() {
  if is_running; then
    info "已在运行 (PID $(cat "$PID_FILE"))，无需重复启动"
    exit 0
  fi
  [ -x "$PY" ] || { err "未找到 .venv，请先运行 ./install.sh"; exit 1; }
  local strays
  strays=$(stray_pids)
  if [ -n "$strays" ]; then
    info "发现游离的旧进程:$strays，先停止"
    kill $strays 2>/dev/null || true
    sleep 1
  fi
  info "后台启动服务（端口 $PORT）…"
  nohup "$PY" server.py >>"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  disown || true
  sleep 1
  if ! is_running; then
    err "启动失败，最近日志："
    tail -n 20 "$LOG_FILE" >&2 || true
    exit 1
  fi
  info "已启动 (PID $(cat "$PID_FILE"))，日志：$LOG_FILE"
  health
}

cmd_stop() {
  local acted=""
  if is_running; then
    local pid
    pid=$(cat "$PID_FILE")
    info "停止服务 (PID $pid)…"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      is_running || break
      sleep 0.5
    done
    if is_running; then
      info "未响应 SIGTERM，强制结束…"
      kill -9 "$pid" 2>/dev/null || true
      sleep 1
    fi
    acted=1
  fi
  rm -f "$PID_FILE"
  local strays
  strays=$(stray_pids)
  if [ -n "$strays" ]; then
    info "清理游离进程:$strays"
    kill -9 $strays 2>/dev/null || true
    acted=1
  fi
  if [ -n "$acted" ]; then info "已停止"; else info "服务本来就没在运行"; fi
}

cmd_status() {
  if is_running; then
    info "服务运行中 (PID $(cat "$PID_FILE"))"
  else
    info "服务未运行"
    rm -f "$PID_FILE" 2>/dev/null || true
  fi
  local r
  r=$(curl -s -m 2 "http://127.0.0.1:$PORT/api/info" 2>/dev/null || true)
  if [ -n "$r" ]; then
    if [ -x "$PY" ]; then
      printf '%s' "$r" | "$PY" -c \
        "import json,sys; d=json.load(sys.stdin); print('   端口 $PORT ·', '模型就绪' if d['ready'] else '模型加载中', f\"· {d.get('num_layers','-')} 层 · 编辑 {d.get('num_edits',0)} 处\")"
    else
      printf '%s\n' "$r" | head -c 200
    fi
  else
    info "端口 $PORT 无响应"
  fi
}

case "${1:-}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_stop; echo; cmd_start ;;
  status)  cmd_status ;;
  log)
    [ -f "$LOG_FILE" ] || { err "暂无日志文件 $LOG_FILE"; exit 1; }
    exec tail -n "${2:-50}" "$LOG_FILE"
    ;;
  logf)
    [ -f "$LOG_FILE" ] || touch "$LOG_FILE"
    exec tail -n 20 -f "$LOG_FILE"
    ;;
  fg)
    [ -x "$PY" ] || { err "未找到 .venv，请先运行 ./install.sh"; exit 1; }
    info "前台运行（Ctrl+C 停止）…"
    echo $$ > "$PID_FILE"
    exec "$PY" server.py
    ;;
  *) usage; exit 1 ;;
esac
