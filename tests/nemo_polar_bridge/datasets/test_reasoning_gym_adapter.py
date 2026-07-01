"""Tests for ReasoningGymDatasetAdapter and its portable verifier.

Requires the optional `reasoning-gym` extra (`uv sync --extra reasoning-gym`).
Skipped automatically when the package is not installed, mirroring how the
`swebench` extra's tests are expected to be gated -- this adapter must never
become a hard dependency of the base install.
"""

from __future__ import annotations

import pytest

from nemo_polar_bridge.datasets.base import TaskSpec
from nemo_polar_bridge.datasets.data_loader import task_from_gym_row
from nemo_polar_bridge.datasets.reasoning_gym_adapter import (
    REASONING_GYM_VERIFIER_NAME,
    ReasoningGymDatasetAdapter,
    task_from_reasoning_gym_item,
)
from nemo_polar_bridge.datasets.verifiers import portable_verifier_source, verify_completion

reasoning_gym = pytest.importorskip("reasoning_gym")


# Values below were captured live against reasoning-gym==0.1.25 (PyPI, fetched
# 2026-06-30) via `reasoning_gym.create_dataset("basic_arithmetic", size=5, seed=42)`.
# Determinism comes from reasoning-gym's own per-item RNG seeding
# (`Random(self.seed + idx)`, reasoning_gym/arithmetic/basic_arithmetic.py), not from
# anything in this repo, so these are fixture-style pinned expectations rather than
# derived facts -- a future reasoning-gym release could change generation output
# without changing the score_answer contract this adapter/verifier depend on.
SEED = 42
EXPECTED_ITEM_0 = {"question": "Calculate -5 * -6.", "answer": "30"}
EXPECTED_ITEM_1 = {"question": "Calculate 965 / 5.", "answer": "193"}


def test_same_seed_produces_deterministic_question_and_answer() -> None:
    adapter_a = ReasoningGymDatasetAdapter(seed=SEED)
    adapter_b = ReasoningGymDatasetAdapter(seed=SEED)

    tasks_a = adapter_a.load_tasks(limit=2, scan_rows=2)
    tasks_b = adapter_b.load_tasks(limit=2, scan_rows=2)

    assert [t.prompt for t in tasks_a] == [t.prompt for t in tasks_b]
    assert [t.answer for t in tasks_a] == [t.answer for t in tasks_b]
    assert tasks_a[0].prompt == EXPECTED_ITEM_0["question"]
    assert tasks_a[0].answer == EXPECTED_ITEM_0["answer"]
    assert tasks_a[1].prompt == EXPECTED_ITEM_1["question"]
    assert tasks_a[1].answer == EXPECTED_ITEM_1["answer"]


def test_different_seed_produces_different_question() -> None:
    adapter_seed_a = ReasoningGymDatasetAdapter(seed=SEED)
    adapter_seed_b = ReasoningGymDatasetAdapter(seed=SEED + 1)

    task_a = adapter_seed_a.load_tasks(limit=1, scan_rows=1)[0]
    task_b = adapter_seed_b.load_tasks(limit=1, scan_rows=1)[0]

    assert task_a.prompt != task_b.prompt


def test_task_carries_new_verifier_name_and_reasoning_gym_shape() -> None:
    adapter = ReasoningGymDatasetAdapter(seed=SEED)
    task = adapter.load_tasks(limit=1, scan_rows=1)[0]

    assert task.verifier_name == REASONING_GYM_VERIFIER_NAME
    assert task.verifier_name != "exact_normalized"
    assert task.source_dataset == "basic_arithmetic"
    assert task.dataset_id == "reasoning-gym/basic_arithmetic"
    assert task.license == "Apache-2.0"
    assert task.responses_create_params["input"] == [
        {"role": "user", "content": task.prompt}
    ]
    assert task.metadata["reasoning_gym_task_name"] == "basic_arithmetic"
    assert task.metadata["reasoning_gym_seed"] == SEED
    # reasoning-gym's own per-item metadata (expression/num_terms/etc.) survives.
    assert "expression" in task.metadata


def test_known_correct_completion_scores_one_via_in_module_verifier() -> None:
    adapter = ReasoningGymDatasetAdapter(seed=SEED)
    task = adapter.load_tasks(limit=1, scan_rows=1)[0]

    result = verify_completion(f"Let me compute this.\nFinal answer: {task.answer}", task)

    assert result.passed is True
    assert result.reward == 1.0
    assert result.reason == "reasoning_gym_exact_match"
    assert result.metadata["verifier"] == "reasoning_gym_basic_arithmetic"


def test_known_wrong_completion_scores_less_than_one_via_in_module_verifier() -> None:
    adapter = ReasoningGymDatasetAdapter(seed=SEED)
    task = adapter.load_tasks(limit=1, scan_rows=1)[0]
    wrong_answer = str(int(task.answer) + 1)

    result = verify_completion(f"Final answer: {wrong_answer}", task)

    assert result.reward < 1.0
    assert result.passed is False


def test_naive_substring_partial_credit_matches_upstream_behavior() -> None:
    # Regression guard for the exact (intentionally naive) upstream semantics:
    # reasoning_gym.dataset.ProceduralDataset.score_answer grants partial credit
    # `len(oracle)/len(candidate)` whenever the oracle string is contained in the
    # candidate string, even when the candidate is numerically wrong (e.g. oracle
    # "30" is a substring of candidate "-30"). This must NOT be "fixed" by the
    # port -- see the citation/divergence comment on
    # `_verify_reasoning_gym_basic_arithmetic` in verifiers.py.
    task = TaskSpec(
        task_id="t-substring",
        responses_create_params={"input": [{"role": "user", "content": "Calculate 5 * 6."}]},
        prompt="Calculate 5 * 6.",
        answer="30",
        dataset_id="reasoning-gym/basic_arithmetic",
        source_dataset="basic_arithmetic",
        verifier_name=REASONING_GYM_VERIFIER_NAME,
    )

    result = verify_completion("Final answer: -30", task)

    assert result.passed is False
    assert 0.0 < result.reward < 1.0
    assert result.reward == len("30") / len("-30")
    assert result.reason == "reasoning_gym_partial_substring_match"


def test_empty_or_missing_completion_scores_zero() -> None:
    task = TaskSpec(
        task_id="t-empty",
        responses_create_params={"input": [{"role": "user", "content": "Calculate 1 + 1."}]},
        prompt="Calculate 1 + 1.",
        answer="2",
        dataset_id="reasoning-gym/basic_arithmetic",
        source_dataset="basic_arithmetic",
        verifier_name=REASONING_GYM_VERIFIER_NAME,
    )

    result = verify_completion("", task)

    assert result.passed is False
    assert result.reward == 0.0
    assert result.reason == "reasoning_gym_no_match"


def test_portable_verifier_source_matches_in_module_verifier() -> None:
    # Exercises the embedded copy the exact same way test_verifiers.py's
    # test_portable_verifier_matches_task_specific_hooks exercises the other
    # verifier branches: exec the generated portable source (stdlib-only, no
    # reasoning_gym import) and confirm it reproduces the in-module result.
    namespace: dict[str, object] = {}
    exec(portable_verifier_source(), namespace)  # noqa: S102 - trusted in-repo source
    portable_verify = namespace["verify_completion"]

    assert "import reasoning_gym" not in portable_verifier_source()

    task = {
        "source_dataset": "basic_arithmetic",
        "verifier_name": REASONING_GYM_VERIFIER_NAME,
        "answer": "30",
    }

    correct = portable_verify("Final answer: 30", task)
    wrong = portable_verify("Final answer: 7", task)
    partial = portable_verify("Final answer: -30", task)

    assert correct["passed"] is True
    assert correct["reward"] == 1.0
    assert wrong["reward"] == 0.0
    assert 0.0 < partial["reward"] < 1.0

    in_module_task = TaskSpec(
        task_id="t-parity",
        responses_create_params={"input": [{"role": "user", "content": "Calculate 5 * 6."}]},
        prompt="Calculate 5 * 6.",
        answer="30",
        dataset_id="reasoning-gym/basic_arithmetic",
        source_dataset="basic_arithmetic",
        verifier_name=REASONING_GYM_VERIFIER_NAME,
    )
    for completion in ("Final answer: 30", "Final answer: 7", "Final answer: -30"):
        in_module_result = verify_completion(completion, in_module_task)
        portable_result = portable_verify(completion, task)
        assert portable_result["reward"] == in_module_result.reward
        assert portable_result["passed"] == in_module_result.passed
        assert portable_result["reason"] == in_module_result.reason


def test_load_tasks_respects_limit() -> None:
    adapter = ReasoningGymDatasetAdapter(seed=SEED)

    tasks = adapter.load_tasks(limit=3, scan_rows=10)

    assert len(tasks) == 3
    assert len({t.stable_id() for t in tasks}) == 3


def test_load_tasks_raises_when_scan_rows_below_limit() -> None:
    adapter = ReasoningGymDatasetAdapter(seed=SEED)

    with pytest.raises(ValueError):
        adapter.load_tasks(limit=5, scan_rows=2)


def test_load_tasks_rejects_non_positive_limit_or_scan_rows() -> None:
    adapter = ReasoningGymDatasetAdapter(seed=SEED)

    with pytest.raises(ValueError):
        adapter.load_tasks(limit=0, scan_rows=5)
    with pytest.raises(ValueError):
        adapter.load_tasks(limit=5, scan_rows=0)


def test_load_tasks_wraps_create_dataset_failure_with_context() -> None:
    # An unregistered task name (typo, or an invalid task_kwargs combination
    # reasoning_gym validates eagerly, e.g. max_terms < min_terms) previously
    # propagated as a bare, context-free exception straight out of
    # reasoning_gym.create_dataset.
    adapter = ReasoningGymDatasetAdapter(task_name="nonexistent_task_xyz", seed=SEED)

    with pytest.raises(RuntimeError, match="nonexistent_task_xyz") as exc_info:
        adapter.load_tasks(limit=1, scan_rows=1)

    assert "seed" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_load_tasks_wraps_row_generation_failure_with_context(monkeypatch) -> None:
    # reasoning_gym's basic_arithmetic validates all task_kwargs eagerly at
    # create_dataset time, so a genuine per-row failure isn't reachable
    # through it -- other reasoning_gym tasks generate lazily per item, so
    # this call site needs its own coverage via a fake dataset object.
    adapter = ReasoningGymDatasetAdapter(seed=SEED)

    class _ExplodingDataset:
        def __getitem__(self, row_idx: int) -> dict:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "nemo_polar_bridge.datasets.reasoning_gym_adapter.reasoning_gym.create_dataset",
        lambda *args, **kwargs: _ExplodingDataset(),
    )

    with pytest.raises(RuntimeError, match="row_idx") as exc_info:
        adapter.load_tasks(limit=1, scan_rows=1)

    assert exc_info.value.__cause__ is not None


def test_scan_rows_does_not_change_which_items_are_selected() -> None:
    # Per-item generation in reasoning-gym is keyed off (seed, idx) only, not the
    # configured dataset `size` -- verified live: create_dataset(size=5, seed=42)[i]
    # == create_dataset(size=100, seed=42)[i]. A larger scan_rows budget must not
    # change which of the first `limit` tasks get selected.
    small_scan = ReasoningGymDatasetAdapter(seed=SEED).load_tasks(limit=2, scan_rows=2)
    large_scan = ReasoningGymDatasetAdapter(seed=SEED).load_tasks(limit=2, scan_rows=20)

    assert [t.prompt for t in small_scan] == [t.prompt for t in large_scan]
    assert [t.answer for t in small_scan] == [t.answer for t in large_scan]


def test_audit_matches_other_adapters_audit_pattern() -> None:
    adapter = ReasoningGymDatasetAdapter(seed=SEED)
    adapter.load_tasks(limit=2, scan_rows=4)

    audit = adapter.audit()

    # Same core keys NeMoGymDatasetAdapter.audit() populates (data_loader.py), so
    # run-manifest consumers don't need adapter-specific branching.
    for key in (
        "dataset_id",
        "config",
        "split",
        "loader",
        "canonical_schema",
        "required_fields",
        "optional_fields",
        "scanned_rows",
        "skipped_unmappable_rows",
        "selected_tasks",
        "inspected_rows",
    ):
        assert key in audit

    assert audit["loader"] == "reasoning_gym"
    assert audit["canonical_schema"] == "reasoning_gym_procedural"
    # Matches NeMoGymDatasetAdapter.load_tasks's early-break behavior: scanning
    # stops as soon as `limit` valid tasks are collected, so `scanned_rows`
    # reflects rows actually inspected (2), not the scan_rows budget (4).
    assert audit["scanned_rows"] == 2
    assert audit["skipped_unmappable_rows"] == 0
    assert len(audit["selected_tasks"]) == 2
    assert audit["reasoning_gym_task_name"] == "basic_arithmetic"
    assert audit["reasoning_gym_seed"] == SEED
    assert audit["inspected_rows"], "inspected_rows should carry sample rows for the manifest"
    assert audit["inspected_rows"][0]["answer"]


def test_audit_available_only_after_load_tasks() -> None:
    adapter = ReasoningGymDatasetAdapter(seed=SEED)

    assert adapter.audit() == {}


def test_task_from_reasoning_gym_item_returns_none_for_missing_fields() -> None:
    assert (
        task_from_reasoning_gym_item(
            {"question": "", "answer": "1"},
            task_name="basic_arithmetic",
            dataset_id="reasoning-gym/basic_arithmetic",
            config="default",
            split="train",
            row_idx=0,
            seed=SEED,
        )
        is None
    )
    assert (
        task_from_reasoning_gym_item(
            {"question": "Calculate 1+1.", "answer": ""},
            task_name="basic_arithmetic",
            dataset_id="reasoning-gym/basic_arithmetic",
            config="default",
            split="train",
            row_idx=0,
            seed=SEED,
        )
        is None
    )


def test_to_attempt_and_to_train_row_match_nemo_gym_adapter_shape() -> None:
    """Schema-compatibility check: prepare_dataset.py must need zero changes.

    Generates one task via ReasoningGymDatasetAdapter and one task via the
    existing NeMoGymDatasetAdapter row-normalization path, runs both through the
    real (not reimplemented) TaskSpec.to_attempt()/to_train_row() methods, and
    confirms the output dicts share the same key set / shape.
    """

    reasoning_gym_task = ReasoningGymDatasetAdapter(seed=SEED).load_tasks(limit=1, scan_rows=1)[0]

    nemo_gym_row = {
        "responses_create_params": {"input": [{"role": "user", "content": "What is 2+2?"}]},
        "answer": "4",
        "metadata": {"source_dataset": "aiw"},
        "uuid": "nemo-gym-task-1",
    }
    nemo_gym_task = task_from_gym_row(
        nemo_gym_row,
        dataset_id="nvidia/Nemotron-RL-ReasoningGym-v1",
        config="default",
        split="train",
        row_idx=0,
    )
    assert nemo_gym_task is not None

    reasoning_gym_attempt = reasoning_gym_task.to_attempt()
    nemo_gym_attempt = nemo_gym_task.to_attempt()
    assert set(reasoning_gym_attempt.keys()) == set(nemo_gym_attempt.keys())
    for key, value in reasoning_gym_attempt.items():
        assert type(value) is type(nemo_gym_attempt[key]), key

    reasoning_gym_train_row = reasoning_gym_task.to_train_row()
    nemo_gym_train_row = nemo_gym_task.to_train_row()
    assert set(reasoning_gym_train_row.keys()) == set(nemo_gym_train_row.keys())
    assert set(reasoning_gym_train_row.keys()) == {
        "input",
        "output",
        "dataset_id",
        "task_id",
        "source_dataset",
    }

    # task_spec_json round-trips through the same responses_create_params.input
    # contract the Polar collector expects from every adapter.
    import json

    task_spec_json = json.loads(reasoning_gym_attempt["task_spec_json"])
    assert isinstance(task_spec_json["responses_create_params"]["input"], list)
    assert task_spec_json["responses_create_params"]["input"][0]["role"] == "user"
