#!/usr/bin/env bash
# AnBan: base-model eval -> LoRA train -> LoRA eval -> print both scorecards.
# Run after scripts/autodl_setup.sh, from anywhere:
#
#   bash scripts/autodl_train_eval.sh            # full loop
#   SKIP_BASE=1 bash scripts/autodl_train_eval.sh   # base scorecard already exists
#   SKIP_TRAIN=1 bash scripts/autodl_train_eval.sh  # re-evaluate an existing adapter
#
# Env overrides: MODEL_DIR, CONFIG, ADAPTER_DIR, RUN_TAG, PORT, WORKERS, LIMIT,
#                MAX_MODEL_LEN, MAX_LORA_RANK, READY_TIMEOUT
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

MODEL_DIR="${MODEL_DIR:-/path/to/Qwen3-8B}"
CONFIG="${CONFIG:-training/configs/qwen3_8b_lora_sft.yaml}"
ADAPTER_DIR="${ADAPTER_DIR:-$(sed -nE 's/^output_dir:[[:space:]]*([^[:space:]#]+).*/\1/p' "$CONFIG")}"
RUN_TAG="${RUN_TAG:-r1}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-16}"
LIMIT="${LIMIT:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"      # must be >= lora_rank in $CONFIG
READY_TIMEOUT="${READY_TIMEOUT:-900}"     # seconds to wait for vLLM to load
TEST_DATA="training/data/test.jsonl"
BASE_OUT="outputs/eval/qwen3_8b_base"
LORA_OUT="outputs/eval/qwen3_8b_lora_${RUN_TAG}"
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

log() { printf '\n\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '\n\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -d "$MODEL_DIR" ]] || die "base model not found at $MODEL_DIR (run scripts/autodl_setup.sh)"
[[ -s "$TEST_DATA" ]] || die "$TEST_DATA missing"
[[ -n "$ADAPTER_DIR" ]] || die "could not read output_dir from $CONFIG; set ADAPTER_DIR"
for v in .venv .venv-train .venv-vllm; do [[ -x "$v/bin/python" ]] || die "$v missing (run scripts/autodl_setup.sh)"; done

# ---------------------------------------------------------------- vLLM lifecycle
VLLM_PID=""
stop_vllm() {
  [[ -n "$VLLM_PID" ]] || return 0
  if kill -0 "$VLLM_PID" 2>/dev/null; then
    log "Stopping vLLM (pid $VLLM_PID)"
    kill -TERM -- "-$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$VLLM_PID" 2>/dev/null || kill -KILL "$VLLM_PID" 2>/dev/null || true
  fi
  wait "$VLLM_PID" 2>/dev/null || true
  VLLM_PID=""
  # Wait until the port is free so the next server can bind it.
  for _ in $(seq 1 30); do
    curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" || break
    sleep 1
  done
}
trap stop_vllm EXIT
trap 'exit 130' INT TERM

start_vllm() {  # start_vllm <logname> <vllm serve args...>
  local name="$1"; shift
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then
    die "port $PORT already serves something; stop it first (pkill -f 'vllm serve')"
  fi
  local logfile="$LOG_DIR/vllm_${name}.log"
  log "Starting vLLM ($name), log: $logfile"
  # setsid -> own process group, so stop_vllm also kills vLLM's worker processes.
  setsid .venv-vllm/bin/vllm serve "$MODEL_DIR" --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" "$@" >"$logfile" 2>&1 &
  VLLM_PID=$!
  local waited=0
  until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      tail -n 40 "$logfile" >&2
      die "vLLM exited during startup (see $logfile)"
    fi
    (( waited >= READY_TIMEOUT )) && { tail -n 40 "$logfile" >&2; die "vLLM not ready after ${READY_TIMEOUT}s"; }
    sleep 5; waited=$((waited + 5))
  done
  log "vLLM ready after ${waited}s: $(curl -s "http://127.0.0.1:$PORT/v1/models" | head -c 300)"
}

run_eval() {  # run_eval <served model name> <out dir>
  local extra=()
  [[ -n "$LIMIT" ]] && extra+=(--limit "$LIMIT")
  log "Evaluating '$1' -> $2"
  .venv/bin/python -m training.eval_planner --data "$TEST_DATA" --planner local \
    --base-url "http://127.0.0.1:$PORT/v1" --model "$1" --workers "$WORKERS" \
    --out "$2" ${extra[@]+"${extra[@]}"}
}

# ---------------------------------------------------------------- 1. base eval
if [[ "${SKIP_BASE:-0}" != "1" ]]; then
  start_vllm base --served-model-name base
  run_eval base "$BASE_OUT"
  stop_vllm
fi

# ---------------------------------------------------------------- 2. train
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  log "Training: $CONFIG -> $ADAPTER_DIR"
  .venv-train/bin/llamafactory-cli train "$CONFIG" 2>&1 | tee "$LOG_DIR/train_${RUN_TAG}.log"
fi
[[ -f "$ADAPTER_DIR/adapter_config.json" ]] || die "no adapter at $ADAPTER_DIR/adapter_config.json"

# ---------------------------------------------------------------- 3. LoRA eval
start_vllm "lora_${RUN_TAG}" --enable-lora --lora-modules "anban=$ADAPTER_DIR" \
  --max-lora-rank "$MAX_LORA_RANK"
run_eval anban "$LORA_OUT"
stop_vllm

# ---------------------------------------------------------------- 4. report
for out in "$BASE_OUT" "$LORA_OUT"; do
  if [[ -f "$out/scorecard.md" ]]; then
    printf '\n==================== %s ====================\n' "$out"
    cat "$out/scorecard.md"
  fi
done
log "Done. Adapter: $ADAPTER_DIR  Loss plot: $ADAPTER_DIR/training_loss.png"
