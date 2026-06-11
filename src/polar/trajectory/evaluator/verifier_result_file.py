"""Evaluator that rewards from a verifier_result.json file."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any

from polar.trajectory.evaluator.base import BaseTrajectoryEvaluator
from polar.trajectory.models import EvalResult, Trajectory


class VerifierResultFileEvaluator(BaseTrajectoryEvaluator):
    """Read a verifier result JSON file from the live runtime and use its reward."""

    def __init__(
        self,
        *,
        path: str = "/polar/session/workspace/verifier_result.json",
        reward_key: str = "reward",
    ) -> None:
        cleaned = str(path or "").strip()
        if not cleaned.startswith("/"):
            raise ValueError("verifier_result_file path must be absolute")
        self.path = cleaned
        self.reward_key = str(reward_key or "reward")

    async def evaluate(self, trajectory: Trajectory, **runtime: Any) -> EvalResult:
        del trajectory
        live_runtime = runtime.get("runtime")
        if live_runtime is None:
            raise RuntimeError("verifier_result_file requires live runtime")
        result = await live_runtime.exec(
            f"cat {self._shell_quote(self.path)}",
            timeout_sec=runtime.get("timeout_seconds"),
        )
        if result.return_code != 0:
            raise RuntimeError(
                "failed to read verifier result file "
                f"{self.path}: {(result.stderr or result.stdout or '').strip()}"
            )
        verifier_result = json.loads(result.stdout or "{}")
        reward = float(verifier_result.get(self.reward_key, 0.0))
        return EvalResult(
            outcome_reward=reward,
            metadata={
                "verifier_result_path": self.path,
                "verifier_result": verifier_result,
            },
        )

    @staticmethod
    def _shell_quote(path: str) -> str:
        # Restrict to a normalized absolute POSIX path before single-quoting.
        normalized = str(PurePosixPath(path))
        return "'" + normalized.replace("'", "'\"'\"'") + "'"
