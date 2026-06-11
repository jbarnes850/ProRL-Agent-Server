from __future__ import annotations

import asyncio
from types import SimpleNamespace

from polar.trajectory.evaluator.verifier_result_file import VerifierResultFileEvaluator
from polar.trajectory.models import Trace, Trajectory


class FakeRuntime:
    async def exec(self, command: str, **kwargs):
        assert "verifier_result.json" in command
        return SimpleNamespace(
            return_code=0,
            stdout='{"passed": true, "reward": 0.225, "metadata": {"score_total": 18}}',
            stderr="",
        )


def test_verifier_result_file_evaluator_uses_numeric_reward() -> None:
    evaluator = VerifierResultFileEvaluator()
    trajectory = Trajectory(
        status="COMPLETED",
        traces=[Trace(response_messages=[{"role": "assistant", "content": "FINAL_JSON: {}"}])],
    )

    result = asyncio.run(evaluator.evaluate(trajectory, runtime=FakeRuntime()))

    assert result.outcome_reward == 0.225
    assert result.metadata["verifier_result"]["metadata"]["score_total"] == 18
