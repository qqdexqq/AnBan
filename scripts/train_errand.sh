#!/usr/bin/env bash
# Train the errand-agent LoRA. Never overwrites a run, never touches r3 adapters.
# Usage: MODEL_DIR=/path/Qwen3-8B GPUS=0,1,2,3,4,5,6,7,8,9 bash scripts/train_errand.sh smoke|train DATA_DIR NEW_RUN_DIR
set -euo pipefail
STAGE="${1:-}"; DATA_DIR="${2:-}"; RUN_DIR="${3:-}"
[[ "$STAGE" == smoke || "$STAGE" == train ]] || { echo 'Usage: MODEL_DIR=... GPUS=0,1 bash scripts/train_errand.sh smoke|train DATA_DIR NEW_RUN_DIR'; exit 2; }
[[ -n "$DATA_DIR" && -d "$DATA_DIR" ]] || { echo 'DATA_DIR from training.build_errand_dataset is required'; exit 2; }
[[ -n "$RUN_DIR" ]] || { echo 'A new output directory is required'; exit 2; }
[[ -n "${MODEL_DIR:-}" && -d "$MODEL_DIR" ]] || { echo 'Set MODEL_DIR to the local Qwen3-8B base checkpoint'; exit 2; }
case "$RUN_DIR" in *r3*|*r4*) echo 'Refusing a run directory named like a planner adapter'; exit 2 ;; esac
[[ -e "$RUN_DIR" ]] && { echo 'Refusing an existing run directory; nothing was changed'; exit 2; }
TRAIN_CLI="${TRAIN_CLI:-.venv-train/bin/llamafactory-cli}"
TRAIN_PY="${TRAIN_PY:-.venv-train/bin/python}"
CONFIG=training/configs/qwen3_8b_lora_sft_errand.yaml
GPUS="${GPUS:-0}"

"$TRAIN_PY" - "$DATA_DIR" <<'PY'
import hashlib, json, sys
from pathlib import Path
from errand.prompts import ERRAND_SYSTEM_PROMPT
data = Path(sys.argv[1])
stats = json.loads((data / 'stats.json').read_text(encoding='utf-8'))
if stats.get('prompt_sha256') != hashlib.sha256(ERRAND_SYSTEM_PROMPT.encode()).hexdigest():
    raise SystemExit('Errand data was built with a different system prompt; rebuild to a NEW directory')
for name in ('train.jsonl', 'dev.jsonl', 'dataset_info.json'):
    digest = hashlib.sha256((data / name).read_bytes()).hexdigest()
    if stats.get('artifact_sha256', {}).get(name) != digest:
        raise SystemExit(f'Checksum mismatch: {name}')
print('Errand data checks passed:', stats.get('counts'))
PY

export CUDA_VISIBLE_DEVICES="$GPUS"
NGPU=$(awk -F, '{print NF}' <<<"$GPUS")
if (( NGPU > 1 )); then export FORCE_TORCHRUN=1 NPROC_PER_NODE="$NGPU"; else unset FORCE_TORCHRUN NPROC_PER_NODE; fi
OVERRIDES=("model_name_or_path=$MODEL_DIR" "dataset_dir=$DATA_DIR" "output_dir=$RUN_DIR")
[[ "$STAGE" == smoke ]] && OVERRIDES+=(max_steps=20 save_steps=20 eval_steps=20 max_samples=64)
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
exec "$TRAIN_CLI" train "$CONFIG" "${OVERRIDES[@]}"
