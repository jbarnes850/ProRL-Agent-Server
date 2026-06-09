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


def test_task_from_gym_row_preserves_agentic_metadata() -> None:
    row = {
        "responses_create_params": {
            "input": [{"role": "user", "content": "Schedule a meeting."}],
            "tools": [{"type": "function", "function": {"name": "calendar_search"}}],
        },
        "expected_answer": "scheduled",
        "metadata": {"source_dataset": "calendar"},
        "agent_ref": {"type": "responses_api_agents", "name": "calendar_simple_agent"},
        "resources_server": "calendar",
        "verifier_ref": {"name": "calendar_verify"},
        "max_turns": 3,
        "seed": 123,
    }

    task = task_from_gym_row(
        row,
        dataset_id="nvidia/Nemotron-RL-agent-calendar_scheduling",
        config="default",
        split="train",
        row_idx=0,
    )

    assert task is not None
    agentic = task.metadata["agentic"]
    assert agentic["resources_server"] == "calendar"
    assert agentic["verifier_ref"] == {"name": "calendar_verify"}
    assert agentic["max_turns"] == 3
    assert agentic["seed"] == 123
    assert agentic["agent_ref"]["name"] == "calendar_simple_agent"
    assert agentic["tools"][0]["function"]["name"] == "calendar_search"


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


def test_normalize_to_gym_row_accepts_agentic_ground_truth_fields() -> None:
    calendar_row = {
        "responses_create_params": {"input": [{"role": "user", "content": "Add lunch."}]},
        "exp_cal_state": [{"event_id": 0, "event_name": "lunch"}],
    }
    workplace_row = {
        "responses_create_params": {"input": [{"role": "user", "content": "Send email."}]},
        "ground_truth": {"sent": True},
    }

    calendar_task = task_from_gym_row(
        normalize_to_gym_row(calendar_row),
        dataset_id="calendar",
        config="default",
        split="train",
        row_idx=0,
    )
    workplace_task = task_from_gym_row(
        normalize_to_gym_row(workplace_row),
        dataset_id="workplace",
        config="default",
        split="train",
        row_idx=0,
    )

    assert calendar_task is not None
    assert calendar_task.answer == '[{"event_id": 0, "event_name": "lunch"}]'
    assert workplace_task is not None
    assert workplace_task.answer == '{"sent": true}'


def test_calendar_dataset_id_infers_calendar_source_and_verifier() -> None:
    row = {
        "responses_create_params": {"input": [{"role": "user", "content": "Add lunch."}]},
        "exp_cal_state": {
            "0": {
                "event_id": 0,
                "duration": 30,
                "constraint": None,
                "min_time": "10:00",
                "max_time": "16:00",
            },
            "1": None,
        },
    }

    task = task_from_gym_row(
        normalize_to_gym_row(row),
        dataset_id="nvidia/Nemotron-RL-agent-calendar_scheduling",
        config="default",
        split="train",
        row_idx=0,
    )

    assert task is not None
    assert task.source_dataset == "calendar"
    assert task.verifier_name == "calendar_gym"
