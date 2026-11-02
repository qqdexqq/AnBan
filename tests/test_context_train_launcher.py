"""Tests for the standalone r4 context-candidate training launcher.

No test imports torch, calls nvidia-smi, spawns training, or touches the real
r3 adapter.  Adapter hashes are injected into the internal validator.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts import train_context_candidate as launcher

BASE = {
    "model_name_or_path": "/old/model",
    "trust_remote_code": True,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_target": "all",
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "dataset_dir": "training/data_v3",
    "dataset": "anban_train",
    "eval_dataset": "anban_dev",
    "template": "qwen3",
    "enable_thinking": False,
    "cutoff_len": 4096,
    "overwrite_cache": False,
    "preprocessing_num_workers": 8,
    "output_dir": "outputs/anban-r3-6gpu",
    "logging_steps": 10,
    "save_steps": 200,
    "save_total_limit": 3,
    "plot_loss": True,
    "overwrite_output_dir": False,
    "report_to": "none",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 3,
    "learning_rate": 1.0e-4,
    "num_train_epochs": 2.0,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "bf16": True,
    "gradient_checkpointing": True,
    "per_device_eval_batch_size": 1,
    "eval_strategy": "steps",
    "eval_steps": 200,
    "resume_from_checkpoint": "outputs/old-checkpoint",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _data(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    for index, name in enumerate(launcher.READINESS_FILES):
        (data / name).write_bytes(f"artifact-{index}\n".encode())
    return data


def _readiness(tmp_path: Path, data: Path, **overrides: object) -> Path:
    report: dict[str, object] = {
        "training_ready": True,
        "cutoff_len": 4096,
        "artifact_sha256": {
            name: _sha(data / name) for name in launcher.READINESS_FILES
        },
    }
    report.update(overrides)
    path = tmp_path / "training_readiness.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _adapter(tmp_path: Path, payload: bytes = b"fake-r3") -> tuple[Path, str]:
    adapter = tmp_path / launcher.R3_ADAPTER_REL
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(payload)
    return adapter, hashlib.sha256(payload).hexdigest()


def test_build_config_applies_the_reviewed_overrides(tmp_path: Path) -> None:
    data = tmp_path / "candidate"
    config = launcher.build_config(
        tmp_path, data, "train", base_config=BASE
    )

    assert config["model_name_or_path"] == str(tmp_path / "models/Qwen3-8B")
    assert config["adapter_name_or_path"] == str(
        tmp_path / "outputs/anban-r3-3090-final-01"
    )
    assert config["output_dir"] == str(
        tmp_path / "outputs/anban-r4-context-train-20261101-01"
    )
    assert config["dataset_dir"] == str(data)
    assert config["create_new_adapter"] is False
    assert config["learning_rate"] == 2.0e-5
    assert config["num_train_epochs"] == 1
    assert config["per_device_train_batch_size"] == 1
    assert config["gradient_accumulation_steps"] == 4
    assert config["warmup_ratio"] == 0.03
    assert config["enable_liger_kernel"] is True
    assert config["bf16"] is True
    assert config["trust_remote_code"] is False
    assert config["preprocessing_num_workers"] == 4
    assert config["logging_steps"] == 5
    assert config["save_steps"] == 100
    assert config["eval_strategy"] == "epoch"
    assert config["report_to"] == "none"
    assert config["overwrite_output_dir"] is False
    assert "save_total_limit" not in config
    assert "resume_from_checkpoint" not in config


def test_smoke_config_is_short_but_uses_the_same_fresh_run(tmp_path: Path) -> None:
    config = launcher.build_config(
        tmp_path, tmp_path / "data", "smoke", base_config=BASE
    )

    assert config["max_steps"] == 4
    assert config["max_samples"] == 64
    assert config["save_steps"] == 4
    assert config["logging_steps"] == 1
    assert config["eval_strategy"] == "steps"
    assert config["eval_steps"] == 4
    assert config["output_dir"].endswith("anban-r4-context-smoke-20261101-01")
    assert "save_total_limit" not in config
    assert "resume_from_checkpoint" not in config


def test_build_config_does_not_mutate_the_source(tmp_path: Path) -> None:
    before = json.dumps(BASE, sort_keys=True)
    launcher.build_config(tmp_path, tmp_path / "data", "train", base_config=BASE)
    assert json.dumps(BASE, sort_keys=True) == before


def test_paths_are_fixed_and_existing_artifacts_are_refused(tmp_path: Path) -> None:
    paths = launcher.output_paths(tmp_path, "smoke")
    assert paths["output"] == (
        tmp_path / "outputs/anban-r4-context-smoke-20261101-01"
    )
    assert paths["config"].parent == tmp_path / "overnight-20261101-01"
    assert paths["log"].parent == paths["config"].parent
    assert paths["metadata"].parent == paths["config"].parent

    paths["log"].parent.mkdir()
    paths["log"].write_text("sentinel", encoding="utf-8")
    with pytest.raises(FileExistsError, match="existing log"):
        launcher._reserve(paths, BASE)
    assert paths["log"].read_text(encoding="utf-8") == "sentinel"
    assert not paths["output"].exists()


def test_refuses_existing_output_before_writing_config(tmp_path: Path) -> None:
    paths = launcher.output_paths(tmp_path, "train")
    paths["output"].mkdir(parents=True)
    with pytest.raises(FileExistsError, match="existing output"):
        launcher._reserve(paths, BASE)
    assert not paths["config"].exists()


def test_validate_readiness_accepts_matching_hashes(tmp_path: Path) -> None:
    data = _data(tmp_path)
    report = launcher.validate_readiness(_readiness(tmp_path, data), data)
    assert report["training_ready"] is True
    assert report["cutoff_len"] == 4096
    assert report["artifact_sha256"] == {
        name: _sha(data / name) for name in launcher.READINESS_FILES
    }


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"training_ready": False}, "training_ready"),
        ({"cutoff_len": 2048}, "cutoff_len"),
        ({"artifact_sha256": {}}, "mismatch"),
    ],
)
def test_validate_readiness_rejects_bad_flags(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    data = _data(tmp_path)
    path = _readiness(tmp_path, data, **overrides)
    with pytest.raises(ValueError, match=message):
        launcher.validate_readiness(path, data)


def test_validate_readiness_rejects_changed_artifact(tmp_path: Path) -> None:
    data = _data(tmp_path)
    path = _readiness(tmp_path, data)
    (data / "train.jsonl").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="train.jsonl"):
        launcher.validate_readiness(path, data)


def test_validate_readiness_is_read_only(tmp_path: Path) -> None:
    data = _data(tmp_path)
    before = {p.name: p.read_bytes() for p in data.iterdir()}
    path = _readiness(tmp_path, data)
    report_before = path.read_bytes()

    launcher.validate_readiness(path, data)

    assert {p.name: p.read_bytes() for p in data.iterdir()} == before
    assert path.read_bytes() == report_before


def test_adapter_hash_is_checked_with_injected_expected_value(tmp_path: Path) -> None:
    _adapter_path, digest = _adapter(tmp_path)
    assert launcher.validate_adapter(tmp_path, expected_sha256=digest) == digest
    with pytest.raises(ValueError, match="adapter SHA256 mismatch"):
        launcher.validate_adapter(tmp_path, expected_sha256="0" * 64)


def test_model_metadata_is_checked_read_only(tmp_path: Path) -> None:
    model = tmp_path / "models/Qwen3-8B"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36}
        ),
        encoding="utf-8",
    )
    (model / "model.safetensors.index.json").write_text("{}", encoding="utf-8")

    assert launcher._validate_model(tmp_path) == model
    (model / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen3", "hidden_size": 2048, "num_hidden_layers": 36}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Qwen3-8B"):
        launcher._validate_model(tmp_path)


def test_parse_gpu_memory_only_accepts_the_fixed_mask() -> None:
    text = "0, 0\n3, 12\n4, 200\n5, 4\n"
    assert launcher.parse_gpu_memory(text) == {0: 0, 3: 12, 4: 200, 5: 4}

    with pytest.raises(RuntimeError, match="GPU 3"):
        launcher.parse_gpu_memory("0, 0\n3, 201\n4, 0\n5, 0\n")
    with pytest.raises(ValueError, match="indices"):
        launcher.parse_gpu_memory("0, 0\n1, 0\n4, 0\n5, 0\n")
    with pytest.raises(ValueError, match="duplicate"):
        launcher.parse_gpu_memory("0, 0\n3, 0\n3, 0\n4, 0\n5, 0\n")
    with pytest.raises(ValueError, match="row"):
        launcher.parse_gpu_memory("0, 0, extra\n3, 0\n4, 0\n5, 0\n")


def test_environment_is_fixed_and_inherits_cache_and_include_paths(
    tmp_path: Path,
) -> None:
    base_env = {
        "PATH": "/usr/bin",
        "C_INCLUDE_PATH": "/private/include",
        "HF_HOME": "/cache/hf",
        "MODELSCOPE_CACHE": "/cache/modelscope",
        "PIP_CACHE_DIR": "/cache/pip",
    }
    env = launcher.build_environment(tmp_path, "smoke", base_env=base_env)

    assert env["FORCE_TORCHRUN"] == "1"
    assert env["NPROC_PER_NODE"] == "4"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,3,4,5"
    assert env["MASTER_ADDR"] == "127.0.0.1"
    assert env["MASTER_PORT"] == "29641"
    assert env["PATH"] == str(tmp_path / ".venv-train/bin") + os.pathsep + "/usr/bin"
    for key in ("C_INCLUDE_PATH", "HF_HOME", "MODELSCOPE_CACHE", "PIP_CACHE_DIR"):
        assert env[key] == base_env[key]


def test_fixed_ports_and_gpu_mask() -> None:
    assert launcher.GPU_MASK == "0,3,4,5"
    assert launcher.GPU_INDICES == (0, 3, 4, 5)
    assert launcher.WORLD_SIZE == 4
    assert launcher.MASTER_PORTS == {"smoke": 29641, "train": 29642}
    assert launcher.R3_ADAPTER_SHA256 == (
        "95a18cce6b54db6a82dfcc90fb7df9fc56ae2345e67604d5624459a0a3830da0"
    )
