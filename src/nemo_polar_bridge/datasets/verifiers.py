"""Deterministic completion verifiers for reasoning-style RL tasks."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from nemo_polar_bridge.datasets.base import TaskSpec, VerifierResult


_FINAL_PATTERNS = (
    re.compile(r"(?:final answer|answer|therefore|result)\s*[:=]\s*(.+)$", re.IGNORECASE),
    re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL),
)


def normalize_answer(value: str) -> str:
    """Normalize an answer for strict but format-tolerant exact matching."""

    value = strip_code_fence(str(value or "")).strip()
    value = value.replace("\u2212", "-")
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" \t\n\r`'\"")
    return value.casefold()


def extract_candidate_answer(completion: str) -> str:
    """Prefer explicit final-answer spans, falling back to the whole completion."""

    text = strip_code_fence(str(completion or "")).strip()
    for pattern in _FINAL_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            last = matches[-1]
            if isinstance(last, tuple):
                last = last[-1]
            return str(last).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


def strip_code_fence(value: str) -> str:
    value = value.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return value


def verify_completion(completion: str, task: TaskSpec) -> VerifierResult:
    """Score one model completion against a task.

    The default contract is deterministic: reward is 1 only when the normalized
    candidate answer equals the normalized ground truth or the completion
    contains the normalized ground truth as a standalone final answer span.
    Dataset-specific hooks can tighten this for tasks with structured metadata.
    """

    source = (task.source_dataset or "").casefold()
    if source == "cryptarithm":
        return _verify_cryptarithm(completion, task)
    if source == "manipulate_matrix":
        return _verify_matrix(completion, task)
    if source == "tower_of_hanoi":
        return _verify_tower_of_hanoi(completion, task)
    return _verify_exact(completion, task)


def _verify_exact(completion: str, task: TaskSpec) -> VerifierResult:
    candidate = normalize_answer(extract_candidate_answer(completion))
    expected = normalize_answer(task.answer)
    passed = bool(expected) and (candidate == expected or _contains_answer(candidate, expected))
    return VerifierResult(
        passed=passed,
        reward=1.0 if passed else 0.0,
        reason="exact_normalized_match" if passed else "normalized_answer_mismatch",
        normalized_completion=candidate,
        normalized_answer=expected,
        metadata={"verifier": "exact_normalized"},
    )


def _contains_answer(candidate: str, expected: str) -> bool:
    if not expected or len(expected) < 2:
        return False
    escaped = re.escape(expected)
    return re.search(rf"(^|[^\w.-]){escaped}($|[^\w.-])", candidate) is not None


def _verify_cryptarithm(completion: str, task: TaskSpec) -> VerifierResult:
    expected = _parse_assignment_map(task.answer)
    candidate = _parse_assignment_map(extract_candidate_answer(completion))
    if expected and candidate:
        passed = candidate == expected
        return VerifierResult(
            passed=passed,
            reward=1.0 if passed else 0.0,
            reason="cryptarithm_assignment_match" if passed else "cryptarithm_assignment_mismatch",
            normalized_completion=_format_assignment_map(candidate),
            normalized_answer=_format_assignment_map(expected),
            metadata={"verifier": "cryptarithm_assignment"},
        )
    return _verify_exact(completion, task)


def _verify_matrix(completion: str, task: TaskSpec) -> VerifierResult:
    expected_obj = _coerce_jsonish(task.answer)
    candidate_obj = _coerce_jsonish(extract_candidate_answer(completion))
    if expected_obj is not None and candidate_obj is not None:
        passed = candidate_obj == expected_obj
        return VerifierResult(
            passed=passed,
            reward=1.0 if passed else 0.0,
            reason="matrix_exact_match" if passed else "matrix_mismatch",
            normalized_completion=json.dumps(candidate_obj, sort_keys=True),
            normalized_answer=json.dumps(expected_obj, sort_keys=True),
            metadata={"verifier": "matrix_literal"},
        )
    expected_grid = _parse_numeric_grid(task.answer)
    candidate_grid = _parse_numeric_grid(extract_candidate_answer(completion))
    if expected_grid and candidate_grid != expected_grid:
        candidate_grid = _parse_numeric_grid(completion)
    if expected_grid and candidate_grid:
        passed = candidate_grid == expected_grid
        return VerifierResult(
            passed=passed,
            reward=1.0 if passed else 0.0,
            reason="matrix_grid_match" if passed else "matrix_grid_mismatch",
            normalized_completion=_format_numeric_grid(candidate_grid),
            normalized_answer=_format_numeric_grid(expected_grid),
            metadata={"verifier": "matrix_grid"},
        )
    return _verify_exact(completion, task)


def _verify_tower_of_hanoi(completion: str, task: TaskSpec) -> VerifierResult:
    # First iteration: require the expected move sequence exactly after normalization.
    # This is intentionally strict and deterministic; richer state simulation can
    # be added without changing the DatasetAdapter contract.
    return _verify_exact(completion, task)


def _coerce_jsonish(value: str) -> Any | None:
    text = str(value or "").strip()
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            continue
    return None


def _parse_assignment_map(value: str) -> dict[str, int]:
    pairs = re.findall(r"\b([A-Za-z])\s*=\s*(-?\d+)\b", str(value or ""))
    parsed: dict[str, int] = {}
    for key, raw_value in pairs:
        normalized_key = key.upper()
        if normalized_key in parsed:
            return {}
        parsed[normalized_key] = int(raw_value)
    return parsed


def _format_assignment_map(value: dict[str, int]) -> str:
    return ",".join(f"{key}={value[key]}" for key in sorted(value))


def _parse_numeric_grid(value: str) -> list[list[int]]:
    text = strip_code_fence(str(value or "")).strip()
    if not text:
        return []
    rows: list[list[int]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = line.strip("[]")
        cells = re.findall(r"-?\d+", line)
        if not cells:
            if re.search(r"(?:final answer|answer|therefore|result)\s*:?\s*$", line, re.I):
                continue
            return []
        rows.append([int(cell) for cell in cells])
    if not rows:
        return []
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        return []
    return rows


def _format_numeric_grid(value: list[list[int]]) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in value)


def portable_verifier_source() -> str:
    """Self-contained verifier code embedded into generated Polar task templates."""

    return r'''
import ast
import json
import re

FINAL_PATTERNS = (
    re.compile(r"(?:final answer|answer|therefore|result)\s*[:=]\s*(.+)$", re.IGNORECASE),
    re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL),
)

def strip_code_fence(value):
    value = str(value or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return value

def normalize_answer(value):
    value = strip_code_fence(str(value or "")).strip()
    value = value.replace("\u2212", "-")
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" \t\n\r`'\"")
    return value.casefold()

def extract_candidate_answer(completion):
    text = strip_code_fence(str(completion or "")).strip()
    for pattern in FINAL_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            last = matches[-1]
            if isinstance(last, tuple):
                last = last[-1]
            return str(last).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text

def contains_answer(candidate, expected):
    if not expected or len(expected) < 2:
        return False
    return re.search(r"(^|[^\w.-])" + re.escape(expected) + r"($|[^\w.-])", candidate) is not None

def coerce_jsonish(value):
    text = str(value or "").strip()
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            pass
    return None

def verify_completion(completion, task):
    source = str(task.get("source_dataset") or "").casefold()
    expected_text = str(task.get("answer") or "")
    candidate_raw = extract_candidate_answer(completion)
    if source == "cryptarithm":
        expected_map = parse_assignment_map(expected_text)
        candidate_map = parse_assignment_map(candidate_raw)
        if expected_map and candidate_map:
            passed = candidate_map == expected_map
            return {
                "passed": passed,
                "reward": 1.0 if passed else 0.0,
                "reason": "cryptarithm_assignment_match" if passed else "cryptarithm_assignment_mismatch",
                "normalized_completion": format_assignment_map(candidate_map),
                "normalized_answer": format_assignment_map(expected_map),
                "metadata": {"verifier": "cryptarithm_assignment"},
            }
    if source == "manipulate_matrix":
        expected_obj = coerce_jsonish(expected_text)
        candidate_obj = coerce_jsonish(candidate_raw)
        if expected_obj is not None and candidate_obj is not None:
            passed = candidate_obj == expected_obj
            return {
                "passed": passed,
                "reward": 1.0 if passed else 0.0,
                "reason": "matrix_exact_match" if passed else "matrix_mismatch",
                "normalized_completion": json.dumps(candidate_obj, sort_keys=True),
                "normalized_answer": json.dumps(expected_obj, sort_keys=True),
                "metadata": {"verifier": "matrix_literal"},
            }
        expected_grid = parse_numeric_grid(expected_text)
        candidate_grid = parse_numeric_grid(candidate_raw)
        if expected_grid and candidate_grid != expected_grid:
            candidate_grid = parse_numeric_grid(completion)
        if expected_grid and candidate_grid:
            passed = candidate_grid == expected_grid
            return {
                "passed": passed,
                "reward": 1.0 if passed else 0.0,
                "reason": "matrix_grid_match" if passed else "matrix_grid_mismatch",
                "normalized_completion": format_numeric_grid(candidate_grid),
                "normalized_answer": format_numeric_grid(expected_grid),
                "metadata": {"verifier": "matrix_grid"},
            }
    candidate = normalize_answer(candidate_raw)
    expected = normalize_answer(expected_text)
    passed = bool(expected) and (candidate == expected or contains_answer(candidate, expected))
    return {
        "passed": passed,
        "reward": 1.0 if passed else 0.0,
        "reason": "exact_normalized_match" if passed else "normalized_answer_mismatch",
        "normalized_completion": candidate,
        "normalized_answer": expected,
        "metadata": {"verifier": "exact_normalized"},
    }

def parse_assignment_map(value):
    pairs = re.findall(r"\b([A-Za-z])\s*=\s*(-?\d+)\b", str(value or ""))
    parsed = {}
    for key, raw_value in pairs:
        normalized_key = key.upper()
        if normalized_key in parsed:
            return {}
        parsed[normalized_key] = int(raw_value)
    return parsed

def format_assignment_map(value):
    return ",".join(f"{key}={value[key]}" for key in sorted(value))

def parse_numeric_grid(value):
    text = strip_code_fence(str(value or "")).strip()
    if not text:
        return []
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = line.strip("[]")
        cells = re.findall(r"-?\d+", line)
        if not cells:
            if re.search(r"(?:final answer|answer|therefore|result)\s*:?\s*$", line, re.I):
                continue
            return []
        rows.append([int(cell) for cell in cells])
    if not rows:
        return []
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        return []
    return rows

def format_numeric_grid(value):
    return "\n".join(" ".join(str(cell) for cell in row) for row in value)
'''
