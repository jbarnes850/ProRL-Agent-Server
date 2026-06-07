"""Evaluator that rewards response text matching configured patterns."""

from __future__ import annotations

import re
from typing import Any

from polar.trajectory.evaluator.base import BaseTrajectoryEvaluator
from polar.trajectory.models import EvalResult, Trace, Trajectory


class ResponseMatchEvaluator(BaseTrajectoryEvaluator):
    """Score each trace by matching assistant response text."""

    def __init__(
        self,
        *,
        patterns: str | list[str],
        use_regex: bool = False,
        case_sensitive: bool = False,
        require_all: bool = False,
        reward: float = 1.0,
        miss_reward: float = 0.0,
    ) -> None:
        raw_patterns = [patterns] if isinstance(patterns, str) else list(patterns)
        cleaned = [pattern for pattern in (p.strip() for p in raw_patterns) if pattern]
        if not cleaned:
            raise ValueError("response_match requires at least one non-empty pattern")
        self.patterns = cleaned
        self.use_regex = use_regex
        self.case_sensitive = case_sensitive
        self.require_all = require_all
        self.reward = float(reward)
        self.miss_reward = float(miss_reward)

    async def evaluate(self, trajectory: Trajectory, **runtime: Any) -> EvalResult:
        del runtime
        trace_rewards = [self._score_trace(trace) for trace in trajectory.traces]
        matched = sum(1 for value in trace_rewards if value == self.reward)
        return EvalResult(
            trace_rewards=trace_rewards,
            metadata={
                "matched_traces": matched,
                "total_traces": len(trace_rewards),
                "patterns": list(self.patterns),
                "use_regex": self.use_regex,
                "case_sensitive": self.case_sensitive,
                "require_all": self.require_all,
            },
        )

    def _score_trace(self, trace: Trace) -> float:
        text = _trace_response_text(trace)
        if not self.case_sensitive:
            text_cmp = text.lower()
            patterns = [pattern.lower() for pattern in self.patterns]
        else:
            text_cmp = text
            patterns = self.patterns

        if self.use_regex:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            matches = [re.search(pattern, text, flags=flags) is not None for pattern in self.patterns]
        else:
            matches = [pattern in text_cmp for pattern in patterns]
        passed = all(matches) if self.require_all else any(matches)
        return self.reward if passed else self.miss_reward


def _trace_response_text(trace: Trace) -> str:
    parts: list[str] = []
    for message in trace.response_messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
    return "\n".join(part for part in parts if part)


__all__ = ["ResponseMatchEvaluator"]
