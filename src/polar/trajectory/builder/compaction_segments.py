"""Fail-closed builder for explicitly marked context-compaction rollouts."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from polar.trajectory.builder.base import BaseTrajectoryBuilder
from polar.trajectory.builder.prefix_merging import PrefixMergingBuilder
from polar.trajectory.models import (
    CompactionEvent,
    CompletionRecord,
    CompletionSession,
    Trace,
    Trajectory,
)

_MARKER_FIELD = "_polar_compaction"


def _token_hash(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _summary_text(trace: Trace) -> str:
    parts: list[str] = []
    for message in trace.response_messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif content is not None:
            parts.append(json.dumps(content, sort_keys=True))
    return "\n".join(parts)


def _marker(completion: CompletionRecord) -> dict[str, Any] | None:
    value = completion.original_request.get(_MARKER_FIELD)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{_MARKER_FIELD} must be an object")
    return dict(value)


class CompactionSegmentsBuilder(BaseTrajectoryBuilder):
    """Preserve execution and summary segments across prompt rewrites.

    Every compaction boundary must be explicitly marked in the original API
    request. Heuristic prefix-break classification is intentionally rejected:
    an unexplained rewrite may be truncation, harness drift, or corruption and
    must not enter training as a compaction event.
    """

    def __init__(self, *, end_of_turn_token_id: int | None = None) -> None:
        self._prefix_builder = PrefixMergingBuilder(
            end_of_turn_token_id=end_of_turn_token_id
        )

    async def build(self, session: CompletionSession) -> Trajectory:
        if not session.completions:
            return Trajectory(
                status="ERROR",
                metadata={
                    "builder": "compaction_segments",
                    "session_id": session.session_id,
                    "record_count": 0,
                },
                error="no completions",
            )

        blocks: list[tuple[Literal["execution", "summary"], str | None, list[CompletionRecord]]] = []
        execution: list[CompletionRecord] = []
        execution_event_id: str | None = None
        pending: dict[str, dict[str, Any]] = {}
        event_order: list[str] = []

        def flush_execution() -> None:
            nonlocal execution, execution_event_id
            if execution:
                blocks.append(("execution", execution_event_id, execution))
                execution = []
                execution_event_id = None

        for completion in session.completions:
            marker = _marker(completion)
            if marker is None:
                execution.append(completion)
                continue

            event_id = str(marker.get("event_id") or "").strip()
            phase = marker.get("phase")
            if not event_id or phase not in {"summary", "resume"}:
                raise ValueError(
                    f"{_MARKER_FIELD} requires nonempty event_id and phase summary|resume"
                )

            if phase == "summary":
                if event_id in pending:
                    raise ValueError(f"duplicate compaction summary event_id {event_id!r}")
                if not execution:
                    raise ValueError("compaction summary requires a preceding execution segment")
                source_ids = [item.completion_id for item in execution]
                flush_execution()
                required = (
                    "context_limit",
                    "threshold_remaining_tokens",
                    "recent_turns_to_keep",
                )
                missing = [key for key in required if key not in marker]
                if missing:
                    raise ValueError(
                        f"compaction summary marker missing fields: {', '.join(missing)}"
                    )
                context_limit = int(marker["context_limit"])
                threshold = int(marker["threshold_remaining_tokens"])
                if context_limit <= 0 or threshold <= 0:
                    raise ValueError("compaction token budgets must be positive")
                pending[event_id] = {
                    "marker": marker,
                    "source_completion_ids": source_ids,
                    "summary_completion": completion,
                }
                event_order.append(event_id)
                blocks.append(("summary", event_id, [completion]))
                continue

            if event_id not in pending:
                raise ValueError(f"resume marker has no summary for event_id {event_id!r}")
            if pending[event_id].get("resume_completion") is not None:
                raise ValueError(f"duplicate compaction resume event_id {event_id!r}")
            if execution:
                raise ValueError("resume marker must be the first request after a summary")
            pending[event_id]["resume_completion"] = completion
            execution_event_id = event_id
            execution.append(completion)

        flush_execution()
        incomplete = [
            event_id
            for event_id in event_order
            if pending[event_id].get("resume_completion") is None
        ]
        if incomplete:
            raise ValueError(f"compaction events missing resume: {', '.join(incomplete)}")

        traces: list[Trace] = []
        block_trace_index: dict[tuple[str, str | None], int] = {}
        resume_trace_index: dict[str, int] = {}
        for kind, event_id, completions in blocks:
            sub_session = CompletionSession(
                session_id=session.session_id,
                task_id=session.task_id,
                model_requested=session.model_requested,
                model_used=session.model_used,
                api_type=session.api_type,
                metadata=dict(session.metadata),
                completions=completions,
            )
            built = await self._prefix_builder.build(sub_session)
            stats = built.metadata.get("reconstruction_stats") or {}
            if len(built.traces) != 1 or stats.get("chains_reconstructed_truncated"):
                raise ValueError(
                    "unmarked prompt rewrite or multi-chain segment in compaction rollout"
                )
            trace = built.traces[0]
            trace.segment_index = len(traces)
            trace.segment_kind = kind
            trace.compaction_event_id = event_id
            trace.metadata = {
                **dict(trace.metadata),
                "compaction_segment_kind": kind,
                "compaction_event_id": event_id,
            }
            traces.append(trace)
            block_trace_index[(kind, event_id)] = trace.segment_index
            if kind == "execution" and completions:
                resume_marker = _marker(completions[0])
                if resume_marker is not None and resume_marker.get("phase") == "resume":
                    resume_trace_index[str(resume_marker["event_id"])] = trace.segment_index

        events: list[CompactionEvent] = []
        for event_id in event_order:
            info = pending[event_id]
            marker = info["marker"]
            summary_index = block_trace_index[("summary", event_id)]
            resume_index = resume_trace_index[event_id]
            summary_trace = traces[summary_index]
            resume_trace = traces[resume_index]
            pre_tokens = len(summary_trace.prompt_ids)
            summary_tokens = len(summary_trace.response_ids)
            if pre_tokens <= 0 or summary_tokens <= 0 or not resume_trace.prompt_ids:
                raise ValueError("compaction event has empty source, summary, or resumed context")
            context_limit = int(marker["context_limit"])
            threshold = int(marker["threshold_remaining_tokens"])
            if pre_tokens > context_limit:
                raise ValueError("actual compaction prompt exceeds context_limit")
            if context_limit - pre_tokens > threshold:
                raise ValueError("compaction fired before the declared threshold")
            declared_trigger = marker.get("trigger_token_count")
            if declared_trigger is not None and int(declared_trigger) != pre_tokens:
                raise ValueError(
                    "declared trigger_token_count does not match server tokenization"
                )
            events.append(
                CompactionEvent(
                    event_id=event_id,
                    summary_trace_index=summary_index,
                    resume_trace_index=resume_index,
                    source_completion_ids=info["source_completion_ids"],
                    summary_completion_id=info["summary_completion"].completion_id,
                    resume_completion_id=info["resume_completion"].completion_id,
                    context_limit=context_limit,
                    trigger_token_count=pre_tokens,
                    threshold_remaining_tokens=threshold,
                    recent_turns_to_keep=int(marker["recent_turns_to_keep"]),
                    pre_compaction_context_tokens=pre_tokens,
                    summary_tokens=summary_tokens,
                    resumed_context_tokens=len(resume_trace.prompt_ids),
                    summary_compression_ratio=summary_tokens / pre_tokens,
                    summary_text=_summary_text(summary_trace),
                    pre_compaction_token_hash=_token_hash(summary_trace.prompt_ids),
                    summary_token_hash=_token_hash(summary_trace.response_ids),
                    resumed_context_token_hash=_token_hash(resume_trace.prompt_ids),
                    policy_version=(
                        str(marker["policy_version"])
                        if marker.get("policy_version") is not None
                        else (
                            str(session.metadata["weight_version"])
                            if session.metadata.get("weight_version") is not None
                            else None
                        )
                    ),
                )
            )

        return Trajectory(
            status="COMPLETED",
            metadata={
                "builder": "compaction_segments",
                "session_id": session.session_id,
                "task_id": session.task_id,
                "record_count": len(session.completions),
                "trace_count": len(traces),
                "compaction_count": len(events),
                "training_compatibility": "segment_aware_trainer_required",
                "task_metadata": dict(session.metadata),
            },
            traces=traces,
            compaction_events=events,
        )
