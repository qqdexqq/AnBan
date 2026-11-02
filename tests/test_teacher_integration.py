"""Teacher paraphrases may change wording, never entities or gold."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from training.build_dataset import build
from training.gold import TeacherMismatch, make_sample, render_seed

SEED = 42
TODAY = date(2026, 10, 17)
PERSONA = "young_pro"
SPLITS = ("train", "dev", "test")


class FakeTeacher:
    """Deterministic stand-in for ``TeacherClient.paraphrase_many``.

    Two safe variants add fillers (or reorder clauses), one changes the amount
    and one drops a name, so the build must reject exactly those two.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.jobs: list[dict] = []

    def paraphrase_many(self, jobs: list[dict], workers: int = 8) -> list[list[str]]:
        self.calls += 1
        self.jobs.extend(jobs)
        return [self._variants(job) for job in jobs]

    @staticmethod
    def _variants(job: dict) -> list[str]:
        text = job["seed_text"]
        keep = list(job["must_keep"])
        lang = job["lang"]

        if lang == "en":
            variants = ["please " + text, text + " please"]
        else:
            variants = ["麻烦" + text, text + "，谢谢"]

        amount = "987654"
        while amount in text:
            amount += "0"
        variants.append(text + " " + amount)

        if keep:
            dropped = next(
                (item for item in keep if not any(char.isdigit() for char in item)),
                keep[0],
            )
            variants.append(text.replace(dropped, "", 1))
        return variants


def _seed() -> tuple[str, list[str]]:
    return render_seed(
        "T1_transfer_by_name",
        PERSONA,
        "zh",
        "explicit",
        SEED,
        TODAY,
        sample_seed=7,
    )


def test_matching_variant_keeps_template_gold() -> None:
    text, must_keep = _seed()
    template = make_sample(
        "T1_transfer_by_name", PERSONA, "zh", "explicit", SEED, TODAY, sample_seed=7
    )
    variant = "麻烦" + text

    sample = make_sample(
        "T1_transfer_by_name",
        PERSONA,
        "zh",
        "explicit",
        SEED,
        TODAY,
        sample_seed=7,
        text_override=variant,
    )

    assert variant != text
    assert sample["assistant"] == template["assistant"]
    assert sample["meta"]["variation"] == "teacher"
    assert sample["meta"]["teacher_seed"] == text

    # The override is what the planner sees (tokenized), not the template text.
    assert sample["meta"]["tokenized"] != template["meta"]["tokenized"]
    for item in must_keep:
        assert item in text


@pytest.mark.parametrize("kind", ("amount", "name"))
def test_unsafe_variants_raise_teacher_mismatch(kind: str) -> None:
    text, must_keep = _seed()
    assert must_keep

    if kind == "amount":
        variant = text + " 987654"
    else:
        name = next(
            (item for item in must_keep if not any(char.isdigit() for char in item)),
            must_keep[0],
        )
        variant = text.replace(name, "", 1)

    with pytest.raises(TeacherMismatch):
        make_sample(
            "T1_transfer_by_name",
            PERSONA,
            "zh",
            "explicit",
            SEED,
            TODAY,
            sample_seed=7,
            text_override=variant,
        )


@pytest.mark.parametrize(
    "scenario_id",
    ("T1_transfer_by_name", "O1_birthday", "S3_cancel_named", "X2_raw_account_in_text"),
)
@pytest.mark.parametrize("lang", ("zh", "en", "mixed"))
def test_render_seed_must_keep_appears_in_seed(scenario_id: str, lang: str) -> None:
    text, must_keep = render_seed(
        scenario_id,
        PERSONA,
        lang,
        "explicit",
        SEED,
        TODAY,
        sample_seed=7,
    )

    assert text
    assert must_keep
    for item in must_keep:
        assert item in text, (scenario_id, lang, item, text)


@pytest.fixture(scope="module")
def datasets(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict, Path]:
    first = tmp_path_factory.mktemp("teacher_dataset")
    stats = build(
        SEED,
        6,
        first,
        TODAY,
        2,
        teacher=FakeTeacher(),
        teacher_seeds=2,
        teacher_n=6,
    )
    second = tmp_path_factory.mktemp("teacher_dataset_rebuild")
    build(
        SEED,
        6,
        second,
        TODAY,
        2,
        teacher=FakeTeacher(),
        teacher_seeds=2,
        teacher_n=6,
    )
    return first, stats, second


def _rows(out: Path, split: str) -> list[dict]:
    with (out / f"{split}.jsonl").open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    for row in rows:
        if isinstance(row.get("meta"), str):
            row["meta"] = json.loads(row["meta"])
    return rows


def test_build_mixes_teacher_samples_into_train_and_test(
    datasets: tuple[Path, dict, Path],
) -> None:
    out, stats, _ = datasets

    assert stats["splits"]["train"]["teacher"] > 0
    assert stats["splits"]["test"]["teacher"] > 0
    assert stats["drops"]["teacher"]["TeacherMismatch"] > 0
    assert stats["drops"]["teacher"].get("leakage", 0) == 0

    train = [row for row in _rows(out, "train") if row["meta"]["variation"] == "teacher"]
    test = [row for row in _rows(out, "test") if row["meta"]["variation"] == "teacher"]
    assert train and test
    for row in train + test:
        assert row["meta"]["teacher_seed"]
    # Held-out teacher samples: expat or explicitly held-out zh personas.
    assert all(
        row["meta"]["persona_id"] == "expat" or row["meta"].get("held_out") for row in test
    )
    assert any(row["meta"]["persona_id"] != "expat" for row in test)
    assert not any(row["meta"].get("held_out") for row in train)

    train_tokens = {row["meta"]["tokenized"] for row in _rows(out, "train")}
    test_tokens = {row["meta"]["tokenized"] for row in _rows(out, "test")}
    assert not train_tokens & test_tokens


def test_teacher_build_is_deterministic(datasets: tuple[Path, dict, Path]) -> None:
    first, _, second = datasets
    for split in SPLITS:
        assert (first / f"{split}.jsonl").read_bytes() == (
            second / f"{split}.jsonl"
        ).read_bytes(), split


def test_intent_guard_rejects_negation_and_scam_drift() -> None:
    """Regression: same entities but a flipped intent must not keep the template gold."""
    from training.gold import _intent_delta

    assert _intent_delta("给妈妈转300", "给妈妈转300，麻烦了") is None
    assert _intent_delta("给妈妈转300", "先别给妈妈转300") is not None
    assert _intent_delta("send mom 300", "don't send mom 300") is not None
    assert _intent_delta("给妈妈转300", "警察说要给妈妈转300到安全账户") is not None
    assert _intent_delta("取消爱奇艺", "帮我取消爱奇艺吧") is None
