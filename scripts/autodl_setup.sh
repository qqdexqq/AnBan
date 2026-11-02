#!/usr/bin/env bash
# AnBan: one-shot, idempotent setup of a fresh AutoDL PyTorch VM (runbook steps 2-3 prep).
#
#   cd /root/autodl-tmp/anban && bash scripts/autodl_setup.sh
#
# Creates three venvs inside the repo (all on the large data disk):
#   .venv        app + tests + eval client   (pip install -e .[dev])
#   .venv-train  LLaMA-Factory (reuses the image's CUDA torch via --system-site-packages)
#   .venv-vllm   vLLM (pins its own torch; kept apart so it cannot break LLaMA-Factory)
# then downloads the base model via ModelScope, checks training/data, runs pytest
# and prints GPU info. Safe to re-run: every step skips work that is already done.
#
# Env overrides: USE_TURBO=0, PYTHON=/path/to/python3.12, MODEL_ID, MODEL_DIR,
#                LLAMAFACTORY_SPEC, VLLM_SPEC, SKIP_TESTS=1, SKIP_MODEL=1
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
MODEL_DIR="${MODEL_DIR:-$DATA_DISK/models/Qwen3-8B}"
# Versions verified by the local smoke test (docs/TRAINING.md, "Versions").
LLAMAFACTORY_SPEC="${LLAMAFACTORY_SPEC:-llamafactory[metrics]==0.9.5}"
VLLM_SPEC="${VLLM_SPEC:-vllm}"

log() { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[setup] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# Keep caches off the small (30 GB) system disk.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_DISK/.cache/pip}"
export HF_HOME="${HF_HOME:-$DATA_DISK/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$DATA_DISK/.cache/modelscope}"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$MODELSCOPE_CACHE"

# ---------------------------------------------------------------- 0. network
if [[ "${USE_TURBO:-1}" == "1" && -f /etc/network_turbo ]]; then
  log "Enabling AutoDL academic acceleration (/etc/network_turbo)"
  set +u; source /etc/network_turbo; set -u
fi

# ---------------------------------------------------------------- 1. python >= 3.12
py_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; }
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  for cand in python3.12 python3 python; do
    if command -v "$cand" >/dev/null && py_ok "$(command -v "$cand")"; then PY="$(command -v "$cand")"; break; fi
  done
fi
if [[ -z "$PY" ]]; then
  # pyproject.toml requires Python >= 3.12; older AutoDL images ship 3.10/3.11 + conda.
  CONDA_PY_DIR="$DATA_DISK/envs/py312"
  if [[ ! -x "$CONDA_PY_DIR/bin/python" ]]; then
    command -v conda >/dev/null || die "No Python >= 3.12 and no conda. Pick an image with Python 3.12 or set PYTHON=."
    log "Creating Python 3.12 via conda at $CONDA_PY_DIR"
    conda create -y -p "$CONDA_PY_DIR" python=3.12
  fi
  PY="$CONDA_PY_DIR/bin/python"
fi
py_ok "$PY" || die "$PY is older than 3.12"
log "Using $("$PY" --version) at $PY"

# Does the base interpreter already have a CUDA torch we can reuse for training?
BASE_HAS_TORCH=0
"$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null && BASE_HAS_TORCH=1

make_venv() {  # make_venv <dir> [--system-site-packages]
  local dir="$1"; shift
  if [[ ! -x "$dir/bin/python" ]]; then
    log "Creating $dir"
    "$PY" -m venv "$@" "$dir"
    "$dir/bin/python" -m pip install -q --upgrade pip
  fi
}

# ---------------------------------------------------------------- 2. app venv
make_venv .venv
log "Installing AnBan (.venv)"
.venv/bin/pip install -q -e ".[dev]"

# ---------------------------------------------------------------- 3. training venv
if [[ "$BASE_HAS_TORCH" == "1" ]]; then
  make_venv .venv-train --system-site-packages
else
  make_venv .venv-train
fi
if ! .venv-train/bin/python -c 'import llamafactory' 2>/dev/null; then
  log "Installing $LLAMAFACTORY_SPEC + modelscope (.venv-train)"
  extra=()
  if [[ "$BASE_HAS_TORCH" == "1" ]]; then
    # Pin the image's torch so pip cannot "upgrade" it to a wheel built for a newer CUDA/driver.
    .venv-train/bin/python -c 'import torch; print(f"torch=={torch.__version__}")' > .venv-train/torch-constraint.txt
    extra+=(-c .venv-train/torch-constraint.txt)
  else
    extra+=(torch)   # no reusable CUDA torch in the image
  fi
  .venv-train/bin/pip install -q "$LLAMAFACTORY_SPEC" modelscope "${extra[@]}"
fi
.venv-train/bin/python -c 'import torch; assert torch.cuda.is_available(), "torch has no CUDA in .venv-train"' \
  || die ".venv-train torch cannot see the GPU (install a CUDA torch wheel matching the driver)"

# ---------------------------------------------------------------- 4. serving venv
make_venv .venv-vllm
if ! .venv-vllm/bin/python -c 'import vllm' 2>/dev/null; then
  log "Installing $VLLM_SPEC (.venv-vllm)"
  .venv-vllm/bin/pip install -q "$VLLM_SPEC"
fi

# ---------------------------------------------------------------- 5. base model
if [[ "${SKIP_MODEL:-0}" != "1" ]]; then
  if [[ -f "$MODEL_DIR/.download_complete" ]]; then
    log "Model already present: $MODEL_DIR"
  else
    log "Downloading $MODEL_ID -> $MODEL_DIR (ModelScope, resumable)"
    mkdir -p "$MODEL_DIR"
    .venv-train/bin/modelscope download --model "$MODEL_ID" --local_dir "$MODEL_DIR"
    [[ -f "$MODEL_DIR/config.json" ]] || die "download finished but $MODEL_DIR/config.json is missing"
    touch "$MODEL_DIR/.download_complete"
  fi
fi

# ---------------------------------------------------------------- 6. dataset
log "Checking training/data"
for f in dataset_info.json train.jsonl dev.jsonl test.jsonl; do
  [[ -s "training/data/$f" ]] || die "training/data/$f missing. Upload training/data/ from the laptop (it is gitignored)."
done
.venv/bin/python - <<'PY'
import json
info = json.load(open("training/data/dataset_info.json", encoding="utf-8"))
for name in ("anban_train", "anban_dev"):
    assert name in info, f"dataset_info.json lacks {name}"
for split in ("train", "dev", "test"):
    n = sum(1 for _ in open(f"training/data/{split}.jsonl", encoding="utf-8"))
    print(f"  {split:5s} {n:6d} rows")
PY

# ---------------------------------------------------------------- 7. tests
if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
  log "Running pytest"
  .venv/bin/python -m pytest -q
fi

# ---------------------------------------------------------------- 8. GPU / disk info
log "GPU / disk"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
.venv-train/bin/python -c 'import torch, transformers, llamafactory; print("torch", torch.__version__, "cuda", torch.version.cuda, "bf16", torch.cuda.is_bf16_supported()); print("transformers", transformers.__version__, "llamafactory", llamafactory.__version__)'
.venv-vllm/bin/python -c 'import vllm; print("vllm", vllm.__version__)'
df -h "$DATA_DISK" / | sed 's/^/  /'
log "Done. Next: bash scripts/autodl_train_eval.sh"
