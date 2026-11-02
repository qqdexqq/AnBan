# Training the AnBan planner

Goal: LoRA-fine-tune Qwen3-8B to act as the AnBan planner, then show on the
held-out test set that it beats (or matches) the base model and the DeepSeek
prompt-only baseline.

The whole loop takes about 3 to 5 hours of GPU time.

Short version on a GPU host:

```bash
cd /path/to/anban
bash scripts/autodl_setup.sh        # venvs, model download, data check, pytest, GPU info
bash scripts/autodl_train_eval.sh   # base eval -> LoRA train -> LoRA eval -> both scorecards
```

The sections below describe what those scripts do, so you can also run the steps
by hand.

## What is not in the repository

Model weights, generated datasets and adapters are not in the repository.
`models/`, `outputs/` and `training/data/` are gitignored. To regenerate them:

- Base weights: run the ModelScope download command in "Set up the training host".
- Datasets: run the build command in "Build the data and check it". The build is deterministic for a given seed and `--today` value.
- Adapter: run the training command in "Baseline, train, evaluate". The adapter and its logs are written under `outputs/`.

## Versions that work

Verified with the smoke config `training/configs/smoke_qwen3_0_6b.yaml`:

| Package | Version |
|---|---|
| Python | 3.12 (AnBan needs >= 3.12; LLaMA-Factory needs >= 3.11) |
| llamafactory | 0.9.5 |
| transformers | 5.6.0 (LLaMA-Factory 0.9.5 caps it at <= 5.6.0) |
| peft / trl / datasets / accelerate | 0.18.1 / 0.24.0 / 4.0.0 / 1.11.0 |
| torch | 2.14.1+cu126 locally; on a GPU host reuse the image's CUDA torch (>= 2.4) |
| modelscope | 1.40.2 |
| vllm | latest; lives in its own venv (`.venv-vllm`) |

## Gotchas

1. **Template: `qwen3` + `enable_thinking: false`, not `qwen3_nothink`.** vLLM is called with
   `chat_template_kwargs.enable_thinking=False`, which renders the prompt ending in
   `<|im_start|>assistant\n<think>\n\n</think>\n\n`. LLaMA-Factory's `qwen3` template with
   `enable_thinking: false` puts exactly that (masked) empty think block in the training
   prompt (checked byte for byte against the Qwen3 tokenizer chat template).
   `qwen3_nothink` leaves it out, so the model would train on a prompt it never sees at
   serve time. For `llamafactory-cli export` use `--template qwen3`; the merged model keeps
   Qwen's own tokenizer chat template.
2. **`meta` must not be a nested object with varying keys in `train.jsonl` or `dev.jsonl`.**
   `datasets` (pyarrow) infers the JSON schema from the first block; `meta.token_roles`
   gains a new key (`date`) later in the file and loading crashes with
   `TypeError: Couldn't cast array of type struct<...>` before step 0. Fixed:
   `training/build_dataset.py` now writes `meta` as a JSON string (`eval_planner` decodes
   it). If you see that error, the data was built by an older version: rebuild it.
3. **Three venvs, not one.** vLLM pins its own torch and transformers and will break
   LLaMA-Factory's pins if installed into the same env. `.venv` is the app, tests and eval
   client; `.venv-train` is LLaMA-Factory (reuses the image torch via
   `--system-site-packages`, pinned with a constraint file); `.venv-vllm` is vLLM.
4. **Python 3.12 image.** `pyproject.toml` requires Python >= 3.12, so use an image with
   Python 3.12. On an older image, `scripts/autodl_setup.sh` creates one with conda.
5. **Caches on the data disk.** The system disk is about 30 GB. The setup script points
   `PIP_CACHE_DIR`, `HF_HOME` and `MODELSCOPE_CACHE` at a cache directory on the data disk
   (via `DATA_DISK`).
6. **Sequence lengths** (Qwen3 tokenizer, data seed 42): full samples 1365 to 1760 tokens,
   p99 about 1700, assistant answer <= 194 tokens. `cutoff_len: 3072` truncates nothing;
   `--max-model-len 4096` for vLLM is enough.
7. **Loss masking is correct**: the system prompt and user turn are in the input with label
   -100; only the assistant JSON plus `<|im_end|>\n` is trained (about 70 of about 1500 tokens).
8. **`--max-lora-rank`** for vLLM must be >= `lora_rank` in the config (32); the script uses 64.
9. A 4 GB laptop GPU cannot train even Qwen3-0.6B at full length (the 151k-vocab logits
   for 1.5k tokens need about 2 GB; 4-bit does not help). CPU is no option on 8 GB RAM (bf16 on
   CPU stalls, fp32 segfaults). The local smoke test therefore uses `cutoff_len=1024`, which
   truncates the prompt but still exercises loader, template, masking, LoRA and fp16.
10. `max_samples` takes the first N rows and `train.jsonl` is grouped by scenario (the first
    200 rows are one scenario with one answer), so a `max_samples` smoke run just memorises.
    Use a shuffled subset for a meaningful loss curve. The full run is fine (Trainer shuffles).
11. Windows only: set `PYTHONUTF8=1` for `llamafactory-cli`; `preprocessing_num_workers: 1`.

## Build the data and check it

```powershell
# rebuild the dataset if anything changed; teacher phrasings are cached, so reruns are free
.venv\Scripts\python.exe -m training.build_dataset --seed 42 --per-scenario 220 --out training/data --today 2026-10-17 --min-test-per-scenario 14 --teacher-seeds 24 --teacher-n 6
.venv\Scripts\python.exe -m pytest -q

# optional smoke test of the data format with LLaMA-Factory (separate venv, 4 GB GPU, about 5 to 25 min)
$env:PYTHONUTF8 = "1"
.venv-train\Scripts\llamafactory-cli.exe train training/configs/smoke_qwen3_0_6b.yaml cutoff_len=1024
```

Smoke setup (once): `python -m venv .venv-train`, then
`.venv-train\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126`,
`.venv-train\Scripts\pip install "llamafactory[metrics]==0.9.5" modelscope`,
`.venv-train\Scripts\modelscope download --model Qwen/Qwen3-0.6B --local_dir models/Qwen3-0.6B`.

Smoke result: full `training/data` loads (7,365 train rows). On a shuffled 400-row
subset (37 scenarios), loss went from about 1.8 (mean of steps 1 to 10) to about
0.28 (steps 31 to 40).

`training/data/` is gitignored. Copy it to the GPU host yourself.

## Set up the training host

Use a PyTorch 2.x + CUDA 12.x image with Python 3.12. The main config fits one
A800 80GB, H800 or RTX 4090 24GB. Put the repository and caches on the large data
disk. If the image provides `/etc/network_turbo`, the setup script sources it;
`USE_TURBO=0` skips it.

```bash
cd /path/to
git clone <repo-url> anban      # or copy the repository as a zip
cd anban
# copy training/data/ into ./training/data
bash scripts/autodl_setup.sh
```

By hand, the script does:

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e .[dev]
python3.12 -m venv --system-site-packages .venv-train
.venv-train/bin/pip install "llamafactory[metrics]==0.9.5" modelscope -c <torch==image version>
python3.12 -m venv .venv-vllm && .venv-vllm/bin/pip install vllm
.venv-train/bin/modelscope download --model Qwen/Qwen3-8B --local_dir /path/to/Qwen3-8B
.venv/bin/python -m pytest -q          # backend tests must pass on the host too
```

Note: check the model card for the current newest open-weight Qwen at 8B. If a newer one
(Apache-2.0) exists, download it and change `model_name_or_path` and `template` in the config.

## Baseline, train, evaluate

`bash scripts/autodl_train_eval.sh` (env: `SKIP_BASE=1`, `SKIP_TRAIN=1`, `RUN_TAG=r2`,
`LIMIT=50` for a quick check). It starts vLLM in its own process group, waits for
`/v1/models`, runs the eval, and always stops vLLM on exit. By hand:

```bash
# 3. base model, no fine-tuning
.venv-vllm/bin/vllm serve /path/to/Qwen3-8B --served-model-name base --max-model-len 4096 --port 8000 &
until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do sleep 5; done
.venv/bin/python -m training.eval_planner --data training/data/test.jsonl --planner local \
  --base-url http://127.0.0.1:8000/v1 --model base --workers 16 --out outputs/eval/qwen3_8b_base
kill %1; wait

# 4. train
.venv-train/bin/llamafactory-cli train training/configs/qwen3_8b_lora_sft.yaml

# 5. fine-tuned model
.venv-vllm/bin/vllm serve /path/to/Qwen3-8B --enable-lora \
  --lora-modules anban=outputs/qwen3-8b-anban-lora --max-lora-rank 64 \
  --max-model-len 4096 --port 8000 &
until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do sleep 5; done
.venv/bin/python -m training.eval_planner --data training/data/test.jsonl --planner local \
  --base-url http://127.0.0.1:8000/v1 --model anban --workers 16 --out outputs/eval/qwen3_8b_lora_r1
kill %1; wait
```

Watch `outputs/qwen3-8b-anban-lora/training_loss.png` and eval loss. On an A800, 2 epochs
takes about 1 hour.

Compare `outputs/eval/*/scorecard.md`. What to look at, in order:

1. `safety_pass` on X (attack/scam): must be 1.0 for the fine-tuned model.
2. `valid`: share of outputs our DAG check accepts (a rejected plan means the user sees
   "please rephrase").
3. `exact` and `tools_acc` per group, and the 10 worst scenarios; feed fixes into data v2.

## Results

Test set v2 (628 cases, `training/data/test.jsonl`):

| model | exact | valid | safety_pass |
|---|---|---|---|
| Qwen3-8B base | 0.159 | 0.505 | 0.907 |
| r1 (data v1, `outputs/qwen3-8b-anban-lora`) | 0.906 | 0.987 | 1.000 |
| **r2 (data v2, step 600, `outputs/anban-r2-final`)** | **0.995** | **0.998** | **1.000** |

r2 is the shipped adapter. Its run stopped at step 620/1048. checkpoint-600 had
the lowest dev loss (0.00079) and was kept instead of resuming. The 3 misses are safe: a
read-only lookup, a standing order without a recurrence word (code turns it into a clarify
question), and an invalid plan the DAG check rejects.

Gotchas from the r1 and r2 sessions:

- vLLM 0.32 needs `export VLLM_USE_FLASHINFER_SAMPLER=0` (no `ninja` on the image).
- If the host starts without a GPU, file work is fine, but vLLM fails with
  "Failed to infer device type".

## Export and serve the adapter

```bash
.venv-train/bin/llamafactory-cli export --model_name_or_path /path/to/Qwen3-8B \
  --adapter_name_or_path outputs/anban-r2-final --template qwen3 \
  --finetuning_type lora --export_dir /path/to/anban-planner-8b --export_size 4
```

Serve the merged model with vLLM and point the agent at it:
`ANBAN_PLANNER=local ANBAN_LOCAL_URL=http://<host>:8000/v1 ANBAN_LOCAL_MODEL=anban`.

Keep `outputs/eval/` and the adapter (`outputs/anban-r2-final`, about 350 MB). The adapter
is the artifact we submit.

## Between sessions

- Stop the host between sessions. Data on the data disk survives.
- Keep `outputs/eval/*/predictions.jsonl`: eval is resumable.
