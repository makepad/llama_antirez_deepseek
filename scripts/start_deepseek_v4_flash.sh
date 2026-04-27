#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/playe/llama.cpp-deepseek-v4-flash}"
BIN="${BIN:-${ROOT}/build/bin/llama-server}"
MODEL="${MODEL:-/home/playe/models/deepseek-v4-flash/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat.gguf}"
TEMPLATE="${TEMPLATE:-${ROOT}/models/templates/deepseek-ai-DeepSeek-V4.jinja}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
SLOTS="${SLOTS:-4}"
CTX_PER_SLOT="${CTX_PER_SLOT:-260096}"
CTX="${CTX:-$((SLOTS * CTX_PER_SLOT))}"
BATCH="${BATCH:-512}"
UBATCH="${UBATCH:-512}"
FLASH_ATTN="${FLASH_ATTN:-off}"
GRAPH_MAX_NODES="${GRAPH_MAX_NODES:-196608}"

LOG_DIR="${LOG_DIR:-/home/playe/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/llama-server-deepseek-v4-flash.log}"

mkdir -p "${LOG_DIR}"

if [[ ! -x "${BIN}" ]]; then
  echo "Missing llama-server binary: ${BIN}" >&2
  exit 1
fi

if [[ ! -f "${MODEL}" ]]; then
  echo "Missing model: ${MODEL}" >&2
  exit 1
fi

if [[ ! -f "${TEMPLATE}" ]]; then
  echo "Missing chat template: ${TEMPLATE}" >&2
  exit 1
fi

echo "Stopping existing llama processes before loading DeepSeek V4..."
pkill -x llama-server 2>/dev/null || true
pkill -x llama-cli 2>/dev/null || true
pkill -x llama-bench 2>/dev/null || true
pkill -x llama-completion 2>/dev/null || true
sleep 2

pids="$(pgrep -x llama-server || true)"
if [[ -n "${pids}" ]]; then
  kill -9 ${pids} 2>/dev/null || true
  sleep 1
fi

echo "Starting DeepSeek V4 Flash on ${HOST}:${PORT}"
echo "Slots=${SLOTS}, ctx_per_slot=${CTX_PER_SLOT}, ctx=${CTX}, flash_attn=${FLASH_ATTN}, log=${LOG_FILE}"

export GGML_CUDA_GRAPH_KEY_BY_NODE_COUNT="${GGML_CUDA_GRAPH_KEY_BY_NODE_COUNT:-1}"
export GGML_CUDA_GRAPH_UPDATE_ON_CHANGE="${GGML_CUDA_GRAPH_UPDATE_ON_CHANGE:-1}"
export LLAMA_DSV4_GRAPH_MAX_NODES="${LLAMA_DSV4_GRAPH_MAX_NODES:-${GRAPH_MAX_NODES}}"

exec "${BIN}" \
  -m "${MODEL}" \
  -ngl 999 \
  -c "${CTX}" \
  -np "${SLOTS}" \
  -b "${BATCH}" \
  -ub "${UBATCH}" \
  -fa "${FLASH_ATTN}" \
  -fit off \
  --cache-ram 0 \
  --jinja \
  --chat-template-file "${TEMPLATE}" \
  --reasoning-format deepseek \
  --host "${HOST}" \
  --port "${PORT}" \
  2>&1 | tee -a "${LOG_FILE}"
