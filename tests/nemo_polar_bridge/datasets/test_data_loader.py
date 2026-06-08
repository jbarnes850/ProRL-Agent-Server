from __future__ import annotations

from nemo_polar_bridge.datasets.data_loader import normalize_to_gym_row, task_from_gym_row


def test_task_from_nemo_gym_row_preserves_schema_fields() -> None:
    row = {
        "responses_create_params": {
            "input": [{"role": "user", "content": "What is 2+2?"}],
            "temperature": 0.3,
        },
        "answer": "4",
        "metadata": {"source_dataset": "aiw", "difficulty": {"n": 1}},
        "agent_ref": {"type": "responses_api_agents", "name": "reasoning_gym"},
        "uuid": "task-1",
        "license": "Creative Commons Attribution 4.0 International",
    }

    task = task_from_gym_row(
        row,
        dataset_id="nvidia/Nemotron-RL-ReasoningGym-v1",
        config="default",
        split="train",
        row_idx=7,
    )

    assert task is not None
    assert task.stable_id() == "task-1"
    assert task.prompt == "What is 2+2?"
    assert task.answer == "4"
    assert task.source_dataset == "aiw"
    assert task.agent_ref == {"type": "responses_api_agents", "name": "reasoning_gym"}
    assert task.responses_create_params["temperature"] == 0.3
    assert task.to_attempt()["task_spec_json"]


def test_normalize_to_gym_row_coerces_hf_reasoning_row() -> None:
    row = {
        "question": "Return the name.",
        "expected_answer": "Ada",
        "metadata": '{"source_dataset": "needle_haystack"}',
    }

    gym_row = normalize_to_gym_row(row)
    task = task_from_gym_row(
        gym_row,
        dataset_id="local/test",
        config="default",
        split="train",
        row_idx=0,
    )

    assert gym_row["responses_create_params"]["input"] == [
        {"role": "user", "content": "Return the name."}
    ]
    assert task is not None
    assert task.answer == "Ada"
    assert task.source_dataset == "needle_haystack"
