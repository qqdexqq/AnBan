#!/usr/bin/env bash
# Explicit training entry point. Never overwrites a run or downloads dependencies.
set -euo pipefail
STAGE="${1:-}"
CARDS="${2:-single}"
RUN_DIR="${3:-}"
[[ "$STAGE" == smoke || "$STAGE" == train ]] || { echo 'Usage: MODEL_DIR=/path bash scripts/train_r3.sh smoke|train single|six NEW_RUN_DIR'; exit 2; }
[[ -n "$RUN_DIR" ]] || { echo 'A new output directory is required'; exit 2; }
[[ -n "${MODEL_DIR:-}" && -d "$MODEL_DIR" ]] || { echo 'Set MODEL_DIR to the existing local Qwen3-8B checkpoint'; exit 2; }
DATA_DIR="${DATA_DIR:-training/data_v3}"
TRAIN_CLI="${TRAIN_CLI:-.venv-train/bin/llamafactory-cli}"
TRAIN_PY="${TRAIN_PY:-.venv-train/bin/python}"
CHECK_PY="${CHECK_PY:-$TRAIN_PY}"
READINESS_REPORT="${READINESS_REPORT:-$DATA_DIR/training_readiness.json}"
case "$CARDS" in
  single) CONFIG=training/configs/qwen3_8b_lora_sft_r3_4090.yaml ;;
  six) CONFIG=training/configs/qwen3_8b_lora_sft_r3_6gpu.yaml ;;
  *) echo 'Card selection must be single or six'; exit 2 ;;
esac
"$CHECK_PY" - "$DATA_DIR" "$RUN_DIR" "$READINESS_REPORT" "$CONFIG" "$MODEL_DIR" <<'PY'
import hashlib, json, sys
from pathlib import Path
import yaml
from agent.prompts import R3_SYSTEM_PROMPT
data, run, readiness_path, config_path, model = map(Path, sys.argv[1:])
model_config = json.loads((model / 'config.json').read_text(encoding='utf-8'))
if (model_config.get('model_type'), model_config.get('hidden_size'), model_config.get('num_hidden_layers')) != ('qwen3', 4096, 36):
    raise SystemExit('Expected the Qwen3-8B base checkpoint; this directory describes a different model')
if not ((model / 'model.safetensors').is_file() or (model / 'model.safetensors.index.json').is_file()):
    raise SystemExit('Base weights are missing. A tokenizer-only directory cannot train the model')
if run.exists():
    raise SystemExit('Refusing an existing run directory; all existing files were preserved')
stats = json.loads((data / 'stats.json').read_text(encoding='utf-8'))
if not readiness_path.is_file():
    raise SystemExit('Missing training_readiness.json. Run python -m training.check_v3 first; do not edit draft flags.')
ready = json.loads(readiness_path.read_text(encoding='utf-8'))
if ready.get('training_ready') is not True:
    raise SystemExit('Training data checks have not passed. Application release approval is a separate gate.')
config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
if ready.get('cutoff_len') != config['cutoff_len']:
    raise SystemExit('Measured sequence cutoff differs from the selected training configuration')
expected = hashlib.sha256(R3_SYSTEM_PROMPT.encode()).hexdigest()
if stats.get('prompt_sha256') != expected or ready.get('prompt_sha256') != expected:
    raise SystemExit('Training data and serving prompt differ; rebuild to a NEW data directory')
for name in ('train.jsonl', 'dev.jsonl', 'test.jsonl', 'dataset_info.json', 'stats.json'):
    if not (data / name).is_file():
        raise SystemExit(f'Missing dataset artifact: {name}')
    digest = hashlib.sha256((data / name).read_bytes()).hexdigest()
    if ready.get('artifact_sha256', {}).get(name) != digest or (name != 'stats.json' and stats.get('artifact_sha256', {}).get(name) != digest):
        raise SystemExit(f'Dataset checksum mismatch: {name}; preserve it and rebuild to a NEW directory')
print('Non-destructive input checks passed. Verify full-length VRAM smoke before a full run.')
PY
OVERRIDES=("model_name_or_path=$MODEL_DIR" "dataset_dir=$DATA_DIR" "output_dir=$RUN_DIR")
if [[ "$STAGE" == smoke ]]; then
  OVERRIDES+=(max_steps=40 save_steps=40 eval_steps=40 max_samples=64)
fi
if [[ "$CARDS" == six ]]; then
  export FORCE_TORCHRUN=1
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
  export NPROC_PER_NODE=6
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  unset FORCE_TORCHRUN NPROC_PER_NODE
fi
"$TRAIN_PY" - "$CARDS" <<'PY'
import sys, torch
expected = 6 if sys.argv[1] == 'six' else 1
observed = torch.cuda.device_count()
if observed != expected:
    raise SystemExit(f'Expected {expected} visible allocated CUDA GPUs, found {observed}. Set CUDA_VISIBLE_DEVICES within your allocation.')
PY
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"  # Atomic reservation: concurrent launches cannot share an output.
exec "$TRAIN_CLI" train "$CONFIG" "${OVERRIDES[@]}"
