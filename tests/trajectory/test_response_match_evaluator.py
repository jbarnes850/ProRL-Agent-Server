from __future__ import annotations

import asyncio

from polar.trajectory.evaluator.response_match import ResponseMatchEvaluator
from polar.trajectory.models import Trace, Trajectory


def test_response_match_evaluator_rewards_matching_trace_text() -> None:
    evaluator = ResponseMatchEvaluator(patterns=["final: 7"], case_sensitive=False)
    trajectory = Trajectory(
        status="COMPLETED",
        traces=[
            Trace(response_messages=[{"role": "assistant", "content": "Final: 7"}]),
            Trace(response_messages=[{"role": "assistant", "content": "Final: 3"}]),
        ],
    )

    result = asyncio.run(evaluator.evaluate(trajectory))

    assert result.trace_rewards == [1.0, 0.0]
    assert result.metadata["matched_traces"] == 1


def test_response_match_evaluator_supports_regex_and_custom_rewards() -> None:
    evaluator = ResponseMatchEvaluator(
        patterns=[r"\b(0|2|4|6|8)\b"],
        use_regex=True,
        reward=2.0,
        miss_reward=-1.0,
    )
    trajectory = Trajectory(
        status="COMPLETED",
        traces=[
            Trace(response_messages=[{"role": "assistant", "content": "6"}]),
            Trace(response_messages=[{"role": "assistant", "content": "5"}]),
        ],
    )

    result = asyncio.run(evaluator.evaluate(trajectory))

    assert result.trace_rewards == [2.0, -1.0]
