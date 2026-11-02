"""Launch one NEW r4 contextual-candidate LoRA run without touching r3.

Inputs are checked read-only first; the fixed four-GPU mask, fresh output and
metadata/config/log are reserved before LLaMA-Factory is started from the
private training venv.  No torch import is needed or performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml

GPU_INDICES = (0, 3, 4, 5)
GPU_MASK = "0,3,4,5"
GPU_MEMORY_LIMIT_MIB = 200
WORLD_SIZE = 4
MASTER_ADDR = "127.0.0.1"
MASTER_PORTS = {"smoke": 29641, "train": 29642}
MODEL_REL = "models/Qwen3-8B"
R3_ADAPTER_REL = "outputs/anban-r3-3090-final-01"
R3_ADAPTER_SHA256 = (
    "95a18cce6b54db6a82dfcc90fb7df9fc56ae2345e67604d5624459a0a3830da0"
)
BASE_CONFIG_REL = "training/configs/qwen3_8b_lora_sft_r3_6gpu.yaml"
RUN_ID = "20261101-01"
RUN_PREFIX = "anban-r4-context"
STATE_DIR_REL = "overnight-20261101-01"
READINESS_CUTOFF = 4096
READINESS_FILES = ("train.jsonl", "dev.jsonl", "test.jsonl", "dataset_info.json")
TRAIN_CLI_REL = ".venv-train/bin/llamafactory-cli"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def output_paths(project_root: Path, stage: str) -> dict[str, Path]:
    """The NEW, fixed paths; no caller-supplied output or log path is accepted."""

    if stage not in MASTER_PORTS:
        raise ValueError(f"unknown stage: {stage!r}")
    root = Path(project_root)
    name = f"{RUN_PREFIX}-{stage}-{RUN_ID}"
    state = root / STATE_DIR_REL
    return {
        "output": root / "outputs" / name,
        "state_dir": state,
        "config": state / f"{name}.yaml",
        "log": state / f"{name}.log",
        "metadata": state / f"{name}.json",
    }


def build_environment(
    project_root: Path, stage: str, *, base_env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Inherit cache/C_INCLUDE_PATH values; add only the fixed launch settings."""

    if stage not in MASTER_PORTS:
        raise ValueError(f"unknown stage: {stage!r}")
    env = dict(os.environ if base_env is None else base_env)
    env.update({
        "FORCE_TORCHRUN": "1",
        "NPROC_PER_NODE": str(WORLD_SIZE),
        "CUDA_VISIBLE_DEVICES": GPU_MASK,
        "MASTER_ADDR": MASTER_ADDR,
        "MASTER_PORT": str(MASTER_PORTS[stage]),
    })
    env["PATH"] = (
        str(Path(project_root) / ".venv-train" / "bin")
        + os.pathsep
        + env.get("PATH", "")
    )
    return env


def build_config(
    project_root: Path,
    data: Path,
    stage: str,
    *,
    base_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the reviewed r4 overrides to the r3 YAML (or an injected mock)."""

    if stage not in MASTER_PORTS:
        raise ValueError(f"unknown stage: {stage!r}")
    root = Path(project_root)
    if base_config is None:
        base_config = yaml.safe_load(
            (root / BASE_CONFIG_REL).read_text(encoding="utf-8")
        )
    if not isinstance(base_config, Mapping):
        raise ValueError("base training configuration must be a mapping")
    config = dict(base_config)
    if config.get("cutoff_len") != READINESS_CUTOFF:
        raise ValueError("base configuration cutoff_len must be 4096")
    paths = output_paths(root, stage)
    config.update({
        "model_name_or_path": str(root / MODEL_REL),
        "adapter_name_or_path": str(root / R3_ADAPTER_REL),
        "create_new_adapter": False,
        "output_dir": str(paths["output"]),
        "dataset_dir": str(Path(data)),
        "learning_rate": 2.0e-5,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "warmup_ratio": 0.03,
        "enable_liger_kernel": True,
        "bf16": True,
        "trust_remote_code": False,
        "preprocessing_num_workers": 4,
        "logging_steps": 5,
        "save_steps": 100,
        "eval_strategy": "epoch",
        "report_to": "none",
        "overwrite_output_dir": False,
    })
    for key in (
        "save_total_limit", "resume_from_checkpoint",
        "max_steps", "max_samples", "eval_steps",
    ):
        config.pop(key, None)
    if stage == "smoke":
        config.update({
            "max_steps": 4,
            "max_samples": 64,
            "save_steps": 4,
            "logging_steps": 1,
            "eval_strategy": "steps",
            "eval_steps": 4,
        })
    return config


def validate_readiness(readiness_path: Path, data_dir: Path) -> dict[str, Any]:
    """Read-only check of training_ready, cutoff 4096, and four artifact hashes."""

    readiness_path, data_dir = Path(readiness_path), Path(data_dir)
    if not readiness_path.is_file():
        raise FileNotFoundError(f"readiness report not found: {readiness_path}")
    try:
        report = json.loads(readiness_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed readiness report: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError("readiness report must be a JSON object")
    if report.get("training_ready") is not True:
        raise ValueError("readiness report does not say training_ready=true")
    if report.get("cutoff_len") != READINESS_CUTOFF:
        raise ValueError(f"readiness cutoff_len must be {READINESS_CUTOFF}")
    recorded = report.get("artifact_sha256")
    if not isinstance(recorded, dict):
        raise ValueError("readiness report has no artifact_sha256 map")
    actual: dict[str, str] = {}
    for name in READINESS_FILES:
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing dataset artifact: {path}")
        actual[name] = _sha256(path)
        if recorded.get(name) != actual[name]:
            raise ValueError(f"readiness artifact_sha256 mismatch for {name}")
    return {
        "training_ready": True,
        "cutoff_len": READINESS_CUTOFF,
        "artifact_sha256": actual,
    }


def validate_adapter(
    project_root: Path, *, expected_sha256: str | None = None
) -> str:
    """Hash-check the preserved r3 adapter; never copy, write, or promote it."""

    adapter = Path(project_root) / R3_ADAPTER_REL
    found = next(
        (adapter / name for name in ("adapter_model.safetensors", "adapter_model.bin")
         if (adapter / name).is_file()),
        None,
    )
    if found is None:
        raise FileNotFoundError(f"r3 adapter weights not found under {adapter}")
    digest = _sha256(found)
    expected = R3_ADAPTER_SHA256 if expected_sha256 is None else expected_sha256
    if digest != expected:
        raise ValueError(f"r3 adapter SHA256 mismatch: {digest} != {expected}")
    return digest


def _validate_model(project_root: Path) -> Path:
    model = Path(project_root) / MODEL_REL
    config_path = model / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    observed = (
        config.get("model_type"),
        config.get("hidden_size"),
        config.get("num_hidden_layers"),
    )
    if observed != ("qwen3", 4096, 36):
        raise ValueError("models/Qwen3-8B does not describe the expected Qwen3-8B")
    if not ((model / "model.safetensors").is_file()
            or (model / "model.safetensors.index.json").is_file()):
        raise FileNotFoundError(f"base weights not found under {model}")
    return model


def parse_gpu_memory(
    text: str,
    *,
    expected_indices: tuple[int, ...] = GPU_INDICES,
    limit_mib: int = GPU_MEMORY_LIMIT_MIB,
) -> dict[int, int]:
    """Parse nvidia-smi CSV and reject wrong indices, duplicates, or busy cards."""

    observed: dict[int, int] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        try:
            index, used = int(fields[0]), int(fields[1])
        except ValueError as exc:
            raise ValueError(f"non-numeric nvidia-smi row: {line!r}") from exc
        if index in observed:
            raise ValueError(f"duplicate GPU index: {index}")
        observed[index] = used
    expected = set(expected_indices)
    if set(observed) != expected:
        raise ValueError(
            f"nvidia-smi indices {sorted(observed)} != allocated {sorted(expected)}"
        )
    for index in expected_indices:
        if observed[index] < 0 or observed[index] > limit_mib:
            raise RuntimeError(
                f"GPU {index} already uses {observed[index]} MiB "
                f"(limit {limit_mib} MiB)"
            )
    return observed


def _query_gpus() -> dict[int, int]:
    command = [
        "nvidia-smi",
        f"--id={GPU_MASK}",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"could not query allocated GPUs: {exc}") from exc
    return parse_gpu_memory(completed.stdout)


def _require_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((MASTER_ADDR, port))
        except OSError as exc:
            raise RuntimeError(f"MASTER_PORT {port} is already in use") from exc


def _reserve(
    paths: Mapping[str, Path], config: Mapping[str, Any]
) -> tuple[Any, Any]:
    """Atomically create the NEW output, config, log, and metadata paths."""

    for name in ("output", "config", "log", "metadata"):
        if _exists(paths[name]):
            raise FileExistsError(f"refusing existing {name}: {paths[name]}")
    output = paths["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    paths["state_dir"].mkdir(parents=True, exist_ok=True)
    output.mkdir()
    created: list[Path] = []
    log_handle = metadata_handle = None
    try:
        config_handle = paths["config"].open("x", encoding="utf-8", newline="\n")
        created.append(paths["config"])
        with config_handle:
            yaml.safe_dump(dict(config), config_handle, sort_keys=False, allow_unicode=True)
        log_handle = paths["log"].open("xb", buffering=0)
        created.append(paths["log"])
        metadata_handle = paths["metadata"].open("x", encoding="utf-8")
        created.append(paths["metadata"])
        return log_handle, metadata_handle
    except BaseException:
        for handle in (metadata_handle, log_handle):
            if handle is not None:
                handle.close()
        # Preserve even partial NEW artifacts for diagnosis; never delete data.
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(MASTER_PORTS), required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    data = args.data.expanduser().resolve()
    readiness = args.readiness.expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"project root not found: {project_root}")
    if not data.is_dir():
        raise FileNotFoundError(f"data directory not found: {data}")
    report = validate_readiness(readiness, data)
    _validate_model(project_root)
    adapter_sha = validate_adapter(project_root)
    paths = output_paths(project_root, args.stage)
    for name in ("output", "config", "log", "metadata"):
        if _exists(paths[name]):
            raise FileExistsError(f"refusing existing {name}: {paths[name]}")
    config = build_config(project_root, data, args.stage)
    cli = project_root / TRAIN_CLI_REL
    if not cli.is_file():
        raise FileNotFoundError(f"private training CLI not found: {cli}")
    gpu_memory = _query_gpus()
    port = MASTER_PORTS[args.stage]
    _require_port_free(port)
    env = build_environment(project_root, args.stage)

    log_handle, metadata_handle = _reserve(paths, config)
    command = [str(cli), "train", str(paths["config"])]
    metadata: dict[str, Any] = {
        "run_id": RUN_ID,
        "stage": args.stage,
        "status": "started",
        "pid": None,
        "project_root": str(project_root),
        "data": str(data),
        "readiness": str(readiness),
        "readiness_cutoff_len": report["cutoff_len"],
        "output": str(paths["output"]),
        "config": str(paths["config"]),
        "log": str(paths["log"]),
        "source_adapter": str(project_root / R3_ADAPTER_REL),
        "source_adapter_sha256": adapter_sha,
        "data_sha256": report["artifact_sha256"],
        "gpu_mask": GPU_MASK,
        "gpu_indices": list(GPU_INDICES),
        "gpu_memory_used_mib": gpu_memory,
        "world_size": WORLD_SIZE,
        "master_addr": MASTER_ADDR,
        "master_port": port,
        "command": command,
        "configuration": config,
        "environment": {
            key: env.get(key)
            for key in ("C_INCLUDE_PATH", "HF_HOME", "MODELSCOPE_CACHE", "PIP_CACHE_DIR")
            if env.get(key) is not None
        },
    }
    try:
        try:
            process = subprocess.Popen(
                command, cwd=str(project_root), env=env,
                stdin=subprocess.DEVNULL, stdout=log_handle,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        except OSError as exc:
            metadata.update({"status": "spawn_failed", "error": str(exc)})
            metadata_handle.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            raise RuntimeError(f"could not spawn training: {exc}") from exc
        metadata["pid"] = process.pid
        metadata_handle.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    finally:
        metadata_handle.close()
        log_handle.close()

    print(f"PID: {process.pid}")
    print(f"LOG: {paths['log']}")
    print(f"OUTPUT: {paths['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
