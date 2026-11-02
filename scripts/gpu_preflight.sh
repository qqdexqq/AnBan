#!/usr/bin/env bash
# Read-only SSH preflight. Does not install, download, train, or change files.
set -u
printf 'AnBan read-only GPU preflight\n'
printf '\nGPU inventory\n'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
  nvidia-smi topo -m
else
  printf 'nvidia-smi unavailable; request a GPU allocation before training.\n'
fi
printf '\nCPU / RAM / working-directory disk\n'
uname -sm
getconf _NPROCESSORS_ONLN
free -h
df -h .
printf '\nPython and installed package versions\n'
if command -v python3 >/dev/null 2>&1; then
  python3 - <<'PY'
import importlib.metadata as metadata
import sys
print(sys.version)
for name in ('torch', 'transformers', 'llamafactory', 'peft', 'trl', 'accelerate', 'bitsandbytes', 'vllm'):
    try:
        print(f'{name}: {metadata.version(name)}')
    except metadata.PackageNotFoundError:
        print(f'{name}: not installed')
PY
else
  printf 'python3 unavailable.\n'
fi
printf '\nNo training started. Confirm scheduler allocation and GPU access before smoke testing.\n'
printf 'Use separate training/serving environments; measure full-sequence VRAM first.\n'
