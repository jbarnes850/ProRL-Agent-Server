from __future__ import annotations

import pytest

from nemo_polar_bridge.collector import _result_trace
from polar.gateway.transform.openai_chat import OpenAIChatTransformer
from polar.trajectory.builder.compaction_segments import CompactionSegmentsBuilder
from polar.trajectory.models import CompletionRecord, CompletionSession, StrategySpec
from polar.trajectory.registry import default_builder_registry


def _completion(
    completion_id: str,
    timestamp: str,
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    text: str,
    marker: dict | None = None,
) -> CompletionRecord:
    original_request = {
        "model": "Qwen/Qwen3.5-9B",
        "messages": [{"role": "user", "content": "task"}],
    }
    if marker is not None:
        original_request["_polar_compaction"] = marker
    return CompletionRecord(
        completion_id=completion_id,
        timestamp=timestamp,
        original_request=original_request,
        request={"model": "Qwen/Qwen3.5-9B"},
        response={
            "choices": [
                {
                    "input_token_ids": prompt_ids,
                    "token_ids": response_ids,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {"token_id": token_id, "logprob": -0.1}
                            for token_id in response_ids
                        ]
                    },
                }
            ]
        },
    )


def _session(*, include_resume: bool = True) -> CompletionSession:
    summary_marker = {
        "event_id": "cmp-1",
        "phase": "summary",
        "context_limit": 16,
        "threshold_remaining_tokens": 12,
        "recent_turns_to_keep": 1,
        "policy_version": "7",
    }
    completions = [
        _completion(
            "exec-0",
            "2026-07-10T00:00:00Z",
            prompt_ids=[1, 2],
            response_ids=[10],
            text="working",
        ),
        _completion(
            "summary-0",
            "2026-07-10T00:00:01Z",
            prompt_ids=[1, 2, 10, 11],
            response_ids=[20, 21],
            text="goal, evidence, unresolved work",
            marker=summary_marker,
        ),
    ]
    if include_resume:
        completions.append(
            _completion(
                "resume-0",
                "2026-07-10T00:00:02Z",
                prompt_ids=[1, 2, 20, 21, 12],
                response_ids=[30],
                text="continued",
                marker={"event_id": "cmp-1", "phase": "resume"},
            )
        )
    return CompletionSession(
        session_id="session-1",
        task_id="task-1",
        metadata={"weight_version": 7},
        completions=completions,
    )


@pytest.mark.asyncio
async def test_compaction_builder_preserves_segments_and_lineage() -> None:
    trajectory = await CompactionSegmentsBuilder().build(_session())

    assert trajectory.status == "COMPLETED"
    assert [trace.segment_kind for trace in trajectory.traces] == [
        "execution",
        "summary",
        "execution",
    ]
    assert [trace.compaction_event_id for trace in trajectory.traces] == [
        None,
        "cmp-1",
        "cmp-1",
    ]
    assert trajectory.metadata["training_compatibility"] == (
        "segment_aware_trainer_required"
    )

    event = trajectory.compaction_events[0]
    assert event.source_completion_ids == ["exec-0"]
    assert event.summary_completion_id == "summary-0"
    assert event.resume_completion_id == "resume-0"
    assert event.summary_trace_index == 1
    assert event.resume_trace_index == 2
    assert event.pre_compaction_context_tokens == 4
    assert event.trigger_token_count == 4
    assert event.summary_tokens == 2
    assert event.resumed_context_tokens == 5
    assert event.summary_compression_ratio == 0.5
    assert event.summary_text == "goal, evidence, unresolved work"
    assert event.policy_version == "7"
    assert len(event.pre_compaction_token_hash) == 64
    assert len(event.summary_token_hash) == 64
    assert len(event.resumed_context_token_hash) == 64


@pytest.mark.asyncio
async def test_compaction_builder_rejects_missing_resume() -> None:
    with pytest.raises(ValueError, match="missing resume"):
        await CompactionSegmentsBuilder().build(_session(include_resume=False))


def test_compaction_marker_is_captured_but_not_forwarded() -> None:
    marker = {"event_id": "cmp-1", "phase": "summary"}
    request = {
        "model": "Qwen/Qwen3.5-9B",
        "messages": [{"role": "user", "content": "task"}],
        "_polar_compaction": marker,
    }

    transformed = OpenAIChatTransformer().transform_request(request)

    assert "_polar_compaction" not in transformed
    assert request["_polar_compaction"] == marker


def test_compaction_builder_is_registered() -> None:
    builder = default_builder_registry().create(
        StrategySpec(strategy="compaction_segments")
    )
    assert isinstance(builder, CompactionSegmentsBuilder)


def test_nemo_collector_rejects_compaction_flattening() -> None:
    with pytest.raises(ValueError, match="segment-aware trainer"):
        _result_trace(
            {
                "trajectory": {
                    "compaction_events": [{"event_id": "cmp-1"}],
                    "traces": [],
                }
            },
            tokenizer=None,
        )
