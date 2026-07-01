"""Deterministic completion verifiers for reasoning-style RL tasks."""

from __future__ import annotations

import ast
import json
import re
import subprocess
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


def extract_shell_command(completion: str) -> str:
    """Pull a single shell command out of a model completion.

    Execution-based analogue of :func:`extract_candidate_answer`: for shell
    tasks the model's answer is a command to *run*, not text to compare. Prefer
    the contents of the first fenced code block (```` ```bash ... ``` ````);
    otherwise fall back to the fence-stripped body. Comment-only lines are
    dropped so a stray ``# explanation`` does not become the command.
    """

    text = str(completion or "")
    match = re.search(r"```[A-Za-z0-9_-]*\n(.*?)```", text, re.DOTALL)
    block = match.group(1).strip() if match else strip_code_fence(text).strip()
    lines = [
        line
        for line in block.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return "\n".join(lines).strip() if lines else block.strip()


def grade_shell_command_execution(
    completion: str,
    expected_stdout: str,
    *,
    cwd: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Execute the model's proposed shell command and score by its stdout.

    Reward is 1.0 iff the extracted command exits 0 and its stdout, stripped,
    equals the expected stdout, stripped. This scores by *running* the command
    in a sandbox, not by string-matching the command text -- the whole point of
    the shell-harness task type. Returns the portable dict shape the Polar
    ``verifier_result_file`` evaluator consumes (``reward``/``passed``/...).
    """

    command = extract_shell_command(completion)
    expected = str(expected_stdout or "").strip()
    result: dict[str, Any] = {
        "passed": False,
        "reward": 0.0,
        "reason": "shell_exec_no_command",
        "normalized_completion": command,
        "normalized_answer": expected,
        "metadata": {"verifier": "shell_command_exec"},
    }
    if not command:
        return result
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        result["reason"] = "shell_exec_timeout"
        result["metadata"]["timeout_seconds"] = timeout
        return result
    except Exception as exc:  # pragma: no cover - defensive
        result["reason"] = "shell_exec_error"
        result["metadata"]["error"] = str(exc)
        return result
    actual = str(proc.stdout or "").strip()
    result["metadata"].update(
        {
            "returncode": proc.returncode,
            "command": command[:2000],
            "actual_stdout": actual[:2000],
            "stderr": str(proc.stderr or "")[:1000],
        }
    )
    if proc.returncode == 0 and actual == expected:
        result.update({"passed": True, "reward": 1.0, "reason": "shell_exec_match"})
    elif proc.returncode != 0:
        result["reason"] = "shell_exec_nonzero_exit"
    else:
        result["reason"] = "shell_exec_stdout_mismatch"
    return result


def verify_completion(completion: str, task: TaskSpec) -> VerifierResult:
    """Score one model completion against a task.

    The default contract is deterministic: reward is 1 only when the normalized
    candidate answer equals the normalized ground truth or the completion
    contains the normalized ground truth as a standalone final answer span.
    Dataset-specific hooks can tighten this for tasks with structured metadata.
    """

    source = (task.source_dataset or "").casefold()
    verifier_name = (task.verifier_name or "").casefold()
    if verifier_name == "materials_tensile_numeric":
        return _verify_materials_tensile_numeric(completion, task)
    if verifier_name == "reasoning_gym_basic_arithmetic":
        return _verify_reasoning_gym_basic_arithmetic(completion, task)
    if source == "calendar" or verifier_name == "calendar_gym":
        return _verify_calendar_gym(completion, task)
    if source == "cryptarithm":
        return _verify_cryptarithm(completion, task)
    if source == "manipulate_matrix":
        return _verify_matrix(completion, task)
    if source == "tower_of_hanoi":
        return _verify_tower_of_hanoi(completion, task)
    return _verify_exact(completion, task)


def _verify_materials_tensile_numeric(completion: str, task: TaskSpec) -> VerifierResult:
    expected = _coerce_jsonish(task.answer)
    if not isinstance(expected, dict):
        return VerifierResult(
            passed=False,
            reward=0.0,
            reason="materials_missing_expected_answer",
            normalized_completion="",
            normalized_answer="",
            metadata={"verifier": "materials_tensile_numeric"},
        )
    parsed = _extract_materials_prediction(completion)
    if not isinstance(parsed, dict):
        return VerifierResult(
            passed=False,
            reward=0.0,
            reason="materials_prediction_parse_failed",
            normalized_completion=str(completion or "")[:1000],
            normalized_answer=json.dumps(_redact_materials_answer(expected), sort_keys=True),
            metadata={"verifier": "materials_tensile_numeric"},
        )

    answer_values = expected.get("answer_values")
    if not isinstance(answer_values, dict):
        answer_values = expected
    schedule = expected.get("scoring_schedule")
    properties = expected.get("properties") or [
        "yield_strength_mpa",
        "elastic_modulus_gpa",
        "ultimate_tensile_strength_mpa",
        "strain_at_uts_mm_per_mm",
    ]
    prediction = parsed.get("prediction") if isinstance(parsed.get("prediction"), dict) else parsed
    property_scores: dict[str, dict[str, Any]] = {}
    total_score = 0
    parse_errors: list[str] = []
    for field in properties:
        answer = answer_values.get(field)
        if not isinstance(answer, dict):
            parse_errors.append(f"missing_answer_field:{field}")
            continue
        try:
            value = float(prediction[field])
            mean = float(answer["mean"])
            std = float(answer["std"])
        except Exception:
            parse_errors.append(f"non_numeric_field:{field}")
            property_scores[field] = {"score": 0, "z": None}
            continue
        z = abs(value - mean) / std if std > 0 else float("inf")
        score = _materials_score_from_z(z, schedule)
        property_scores[field] = {
            "prediction": value,
            "mean": mean,
            "std": std,
            "z": z,
            "score": score,
        }
        total_score += score

    max_score = int(expected.get("max_score") or (20 * len(properties)))
    reward = float(total_score / max_score) if max_score > 0 else 0.0
    passed = total_score > 0 and not parse_errors
    reason = "materials_numeric_score" if not parse_errors else "materials_numeric_parse_error"
    return VerifierResult(
        passed=passed,
        reward=reward,
        reason=reason,
        normalized_completion=json.dumps(parsed, sort_keys=True),
        normalized_answer=json.dumps(_redact_materials_answer(expected), sort_keys=True),
        metadata={
            "verifier": "materials_tensile_numeric",
            "score_total": total_score,
            "score_max": max_score,
            "property_scores": property_scores,
            "parse_errors": parse_errors,
            "boundary_policy": expected.get("boundary_policy")
            or "lower_exclusive_upper_inclusive_interpolated_bins",
            "integrity_policy_id": expected.get("integrity_policy_id"),
            "dataset_hashes": expected.get("dataset_hashes") or {},
        },
    )


def _materials_score_from_z(z: float, schedule: Any) -> int:
    epsilon = 1e-9
    if isinstance(schedule, list):
        for row in schedule:
            if not isinstance(row, dict):
                continue
            lower = float(row.get("lower_z", row.get("lower", 0)))
            upper_raw = row.get("upper_z", row.get("upper"))
            upper = float("inf") if upper_raw in (None, "inf", "infinity") else float(upper_raw)
            lower_inclusive = bool(row.get("lower_inclusive", lower == 0))
            upper_inclusive = bool(row.get("upper_inclusive", True))
            above_lower = z >= lower - epsilon if lower_inclusive else z > lower + epsilon
            below_upper = z <= upper + epsilon if upper_inclusive else z < upper - epsilon
            if above_lower and below_upper:
                return int(row.get("points", row.get("score", 0)))
    if z <= 1 + epsilon:
        return 20
    if z > 20 + epsilon:
        return 0
    return max(0, 21 - int(z if z == int(z) else int(z) + 1))


def _extract_materials_prediction(completion: str) -> dict[str, Any] | None:
    text = strip_code_fence(str(completion or "")).strip()
    tagged = re.findall(r"FINAL_JSON\s*:\s*(\{.*\})", text, flags=re.IGNORECASE | re.DOTALL)
    candidates = tagged or re.findall(r"\{.*\}", text, flags=re.DOTALL)
    for candidate in reversed(candidates):
        parsed = _coerce_jsonish(candidate)
        if isinstance(parsed, dict):
            return parsed
    return None


def _redact_materials_answer(answer: dict[str, Any]) -> dict[str, Any]:
    return {
        "properties": answer.get("properties"),
        "max_score": answer.get("max_score"),
        "boundary_policy": answer.get("boundary_policy"),
        "integrity_policy_id": answer.get("integrity_policy_id"),
        "dataset_hashes": answer.get("dataset_hashes") or {},
    }


def _verify_reasoning_gym_basic_arithmetic(completion: str, task: TaskSpec) -> VerifierResult:
    """Faithful port of reasoning-gym's default ``score_answer``.

    Upstream source (verified live against reasoning-gym==0.1.25, PyPI, fetched
    2026-06-30; also matches github.com/open-thought/reasoning-gym commit
    49b07130b3fcd12f2d064bba7c43869543a0e7e7,
    reasoning_gym/dataset.py lines 63-72):

        def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
            \"\"\"Overwrite this method in derived classes if a single oracle answer
            is not available.\"\"\"
            oracle_answer = entry["answer"]
            reward = 0.0
            if isinstance(answer, str) and len(answer) > 0:
                if answer == oracle_answer:
                    reward = 1.0
                elif oracle_answer in answer:
                    reward = len(oracle_answer) / len(answer)
            return reward

    `reasoning_gym.arithmetic.basic_arithmetic.BasicArithmeticDataset` does not
    override `score_answer` (confirmed live: `"score_answer" not in
    BasicArithmeticDataset.__dict__`), so it inherits this exact
    `ProceduralDataset.score_answer` implementation unmodified. The comparison
    logic below (exact match -> reward 1.0; oracle substring-contained in the
    candidate -> partial credit `len(oracle)/len(candidate)`; anything else,
    including a None/empty candidate -> reward 0.0) is reproduced verbatim,
    including the "somewhat naive" substring-containment behavior (e.g. an
    answer of "-30" against oracle "30" earns partial credit because "30" is a
    substring of "-30", even though it is numerically wrong).

    Divergence from upstream (for portability/correctness, not reinvention):
    upstream's own training/eval harnesses do not call `score_answer` on a raw,
    unprocessed model completion -- reasoning_gym/utils.py ships
    `extract_answer(completion, tag_name="answer")` plus an `<answer>...</answer>`
    system-prompt convention (SYSTEM_PROMPTS["default"]/["simple"]) specifically
    so a short candidate string is extracted from the completion before scoring.
    This port reuses this module's own `extract_candidate_answer()` (already used
    by every other verifier in this file, and already handling `<answer>...
    </answer>` plus "final answer:"-style spans) as that pre-extraction step,
    instead of importing reasoning_gym.utils.extract_answer, because this function
    must stay stdlib-only to run inside Polar's isolated task sandbox (no
    reasoning-gym install there). Scoring the full raw completion directly against
    `oracle_answer in answer` would make the substring-containment fallback branch
    fire on almost every completion that ever states the correct number in prose,
    which is not what upstream's own scoring convention intends when used with a
    chain-of-thought policy.
    """

    oracle_answer = str(task.answer or "")
    candidate = extract_candidate_answer(completion)
    reward = 0.0
    if isinstance(candidate, str) and len(candidate) > 0:
        if candidate == oracle_answer:
            reward = 1.0
        elif oracle_answer and oracle_answer in candidate:
            reward = len(oracle_answer) / len(candidate)
    passed = reward >= 1.0
    if passed:
        reason = "reasoning_gym_exact_match"
    elif reward > 0.0:
        reason = "reasoning_gym_partial_substring_match"
    else:
        reason = "reasoning_gym_no_match"
    return VerifierResult(
        passed=passed,
        reward=reward,
        reason=reason,
        normalized_completion=candidate,
        normalized_answer=oracle_answer,
        metadata={
            "verifier": "reasoning_gym_basic_arithmetic",
            "ported_from": "reasoning_gym.dataset.ProceduralDataset.score_answer",
        },
    )


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


def _verify_calendar_gym(completion: str, task: TaskSpec) -> VerifierResult:
    expected = _coerce_jsonish(task.answer)
    exp_cal_state = _normalize_calendar_state(expected)
    reward, reason = _grade_calendar_response(completion, exp_cal_state)
    passed = reward > 0
    return VerifierResult(
        passed=passed,
        reward=float(reward),
        reason=f"calendar_gym_{reason}",
        normalized_completion=json.dumps(_extract_calendar_json_list(completion), sort_keys=True),
        normalized_answer=json.dumps(exp_cal_state, sort_keys=True),
        metadata={
            "verifier": "calendar_gym",
            "resource_server": "resources_servers/calendar",
            "compatible_with": "NVIDIA-NeMo/Gym/resources_servers/calendar/utils.py",
            "expected_event_count": len(exp_cal_state),
        },
    )


def _normalize_calendar_state(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for event_id, event in value.items():
        if not isinstance(event, dict):
            continue
        normalized[str(event_id)] = event
    return normalized


def _grade_calendar_response(
    assistant_response: str,
    exp_cal_state: dict[str, dict[str, Any]],
) -> tuple[int, str]:
    # Mirrors NVIDIA-NeMo/Gym resources_servers/calendar/utils.py.
    if "<think>" in assistant_response:
        return 0, "think_found"
    if len(exp_cal_state) == 0:
        return 1, "pass"
    try:
        cal_state = _extract_calendar_json_list(assistant_response)
        if cal_state is None or len(cal_state) == 0:
            return 0, "no_json_list"

        events_dict: dict[str, dict[str, Any]] = {}
        for event in cal_state:
            if not isinstance(event, dict) or "event_id" not in event:
                return 0, "error_in_grading"
            events_dict[str(event["event_id"])] = event

        if len(events_dict) != len(exp_cal_state):
            return 0, "different_number_of_events"

        for event in cal_state:
            if _is_calendar_event_conflicting(cal_state, event, exclude_event=event):
                return 0, "conflicting_events"

        for event_id, expected_event in exp_cal_state.items():
            if event_id not in events_dict:
                return 0, "different_number_of_events"
            if not _is_calendar_constraint_satisfied(events_dict[event_id], expected_event):
                return 0, "constraint_violated"
    except Exception:
        return 0, "error_in_grading"
    return 1, "pass"


def _extract_calendar_json_list(text: str) -> list[Any] | None:
    pattern = r"\[(?:[^\[\]]|\{[^}]*\})*\{(?:[^\[\]]|\{[^}]*\})*\}(?:[^\[\]]|\{[^}]*\})*\]"
    match = re.search(pattern, str(text or ""), re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def _calendar_time_to_minutes(time_str: str) -> int:
    time_str = str(time_str).strip().lower()
    if "am" in time_str or "pm" in time_str:
        if ":" not in time_str:
            if "am" in time_str:
                hour = int(time_str.replace("am", ""))
                return hour * 60 if hour != 12 else 0
            if "pm" in time_str:
                hour = int(time_str.replace("pm", ""))
                return (hour * 60 if hour != 12 else 0) + 12 * 60
        if "am" in time_str:
            hour, minute = map(int, time_str.replace("am", "").split(":"))
            return (hour * 60 if hour != 12 else 0) + minute
        if "pm" in time_str:
            hour, minute = map(int, time_str.replace("pm", "").split(":"))
            return ((hour * 60 if hour != 12 else 0) + 12 * 60) + minute
        raise ValueError(f"Unable to parse time: {time_str}")
    hours, minutes = map(int, time_str.split(":"))
    return hours * 60 + minutes


def _is_calendar_event_conflicting(
    events: list[dict[str, Any]],
    check_event: dict[str, Any],
    *,
    exclude_event: dict[str, Any] | None = None,
) -> bool:
    event_start = _calendar_time_to_minutes(str(check_event["start_time"]))
    event_end = event_start + int(check_event["duration"])
    for existing in events:
        if exclude_event and existing == exclude_event:
            continue
        existing_start = _calendar_time_to_minutes(str(existing["start_time"]))
        existing_end = existing_start + int(existing["duration"])
        if not (event_end <= existing_start or event_start >= existing_end):
            return True
    return False


def _is_calendar_constraint_satisfied(
    event: dict[str, Any],
    expected_event: dict[str, Any],
) -> bool:
    if int(event["duration"]) != int(expected_event["duration"]):
        return False

    min_time = _calendar_time_to_minutes(str(expected_event["min_time"]))
    max_time = _calendar_time_to_minutes(str(expected_event["max_time"]))
    event_start = _calendar_time_to_minutes(str(event["start_time"]))
    event_end = event_start + int(event["duration"])
    if event_start < min_time or event_end > max_time:
        return False

    constraint = expected_event.get("constraint")
    if constraint is None:
        return True
    constraint = str(constraint)
    if constraint.startswith("before "):
        constraint_time = _calendar_time_to_minutes(constraint.replace("before ", ""))
        return event_end <= constraint_time
    if constraint.startswith("after "):
        constraint_time = _calendar_time_to_minutes(constraint.replace("after ", ""))
        return event_start >= constraint_time
    if constraint.startswith("between "):
        parts = constraint.replace("between ", "").split(" and ")
        if len(parts) != 2:
            raise ValueError(f"Invalid 'between' constraint format: {constraint}")
        time_x = _calendar_time_to_minutes(parts[0])
        time_y = _calendar_time_to_minutes(parts[1])
        return event_start >= time_x and event_end <= time_y
    if constraint.startswith("at "):
        constraint_time = _calendar_time_to_minutes(constraint.replace("at ", ""))
        return event_start == constraint_time
    return True


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

def materials_score_from_z(z, schedule):
    epsilon = 1e-9
    if isinstance(schedule, list):
        for row in schedule:
            if not isinstance(row, dict):
                continue
            lower = float(row.get("lower_z", row.get("lower", 0)))
            upper_raw = row.get("upper_z", row.get("upper"))
            upper = float("inf") if upper_raw in (None, "inf", "infinity") else float(upper_raw)
            lower_inclusive = bool(row.get("lower_inclusive", lower == 0))
            upper_inclusive = bool(row.get("upper_inclusive", True))
            above_lower = z >= lower - epsilon if lower_inclusive else z > lower + epsilon
            below_upper = z <= upper + epsilon if upper_inclusive else z < upper - epsilon
            if above_lower and below_upper:
                return int(row.get("points", row.get("score", 0)))
    if z <= 1 + epsilon:
        return 20
    if z > 20 + epsilon:
        return 0
    return max(0, 21 - int(z if z == int(z) else int(z) + 1))

def extract_materials_prediction(completion):
    text = strip_code_fence(str(completion or "")).strip()
    tagged = re.findall(r"FINAL_JSON\s*:\s*(\{.*\})", text, flags=re.IGNORECASE | re.DOTALL)
    candidates = tagged or re.findall(r"\{.*\}", text, flags=re.DOTALL)
    for candidate in reversed(candidates):
        parsed = coerce_jsonish(candidate)
        if isinstance(parsed, dict):
            return parsed
    return None

def redact_materials_answer(answer):
    return {
        "properties": answer.get("properties"),
        "max_score": answer.get("max_score"),
        "boundary_policy": answer.get("boundary_policy"),
        "integrity_policy_id": answer.get("integrity_policy_id"),
        "dataset_hashes": answer.get("dataset_hashes") or {},
    }

def verify_materials_tensile_numeric(completion, task):
    expected = coerce_jsonish(str(task.get("answer") or ""))
    if not isinstance(expected, dict):
        return {
            "passed": False,
            "reward": 0.0,
            "reason": "materials_missing_expected_answer",
            "normalized_completion": "",
            "normalized_answer": "",
            "metadata": {"verifier": "materials_tensile_numeric"},
        }
    parsed = extract_materials_prediction(completion)
    if not isinstance(parsed, dict):
        return {
            "passed": False,
            "reward": 0.0,
            "reason": "materials_prediction_parse_failed",
            "normalized_completion": str(completion or "")[:1000],
            "normalized_answer": json.dumps(redact_materials_answer(expected), sort_keys=True),
            "metadata": {"verifier": "materials_tensile_numeric"},
        }

    answer_values = expected.get("answer_values")
    if not isinstance(answer_values, dict):
        answer_values = expected
    schedule = expected.get("scoring_schedule")
    properties = expected.get("properties") or [
        "yield_strength_mpa",
        "elastic_modulus_gpa",
        "ultimate_tensile_strength_mpa",
        "strain_at_uts_mm_per_mm",
    ]
    prediction = parsed.get("prediction") if isinstance(parsed.get("prediction"), dict) else parsed
    property_scores = {}
    parse_errors = []
    total_score = 0
    for field in properties:
        answer = answer_values.get(field)
        if not isinstance(answer, dict):
            parse_errors.append("missing_answer_field:" + field)
            continue
        try:
            value = float(prediction[field])
            mean = float(answer["mean"])
            std = float(answer["std"])
        except Exception:
            parse_errors.append("non_numeric_field:" + field)
            property_scores[field] = {"score": 0, "z": None}
            continue
        z = abs(value - mean) / std if std > 0 else float("inf")
        score = materials_score_from_z(z, schedule)
        property_scores[field] = {
            "prediction": value,
            "mean": mean,
            "std": std,
            "z": z,
            "score": score,
        }
        total_score += score

    max_score = int(expected.get("max_score") or (20 * len(properties)))
    reward = float(total_score / max_score) if max_score > 0 else 0.0
    return {
        "passed": bool(total_score > 0 and not parse_errors),
        "reward": reward,
        "reason": "materials_numeric_score" if not parse_errors else "materials_numeric_parse_error",
        "normalized_completion": json.dumps(parsed, sort_keys=True),
        "normalized_answer": json.dumps(redact_materials_answer(expected), sort_keys=True),
        "metadata": {
            "verifier": "materials_tensile_numeric",
            "score_total": total_score,
            "score_max": max_score,
            "property_scores": property_scores,
            "parse_errors": parse_errors,
            "boundary_policy": expected.get("boundary_policy")
            or "lower_exclusive_upper_inclusive_interpolated_bins",
            "integrity_policy_id": expected.get("integrity_policy_id"),
            "dataset_hashes": expected.get("dataset_hashes") or {},
        },
    }

def verify_reasoning_gym_basic_arithmetic(completion, task):
    # Ported verbatim from reasoning_gym.dataset.ProceduralDataset.score_answer
    # (reasoning-gym==0.1.25 / github.com/open-thought/reasoning-gym commit
    # 49b07130b3fcd12f2d064bba7c43869543a0e7e7, reasoning_gym/dataset.py:63-72),
    # which BasicArithmeticDataset inherits unmodified. See the in-module
    # docstring on _verify_reasoning_gym_basic_arithmetic in verifiers.py for the
    # full citation and the documented divergence (extract_candidate_answer() as
    # a stdlib-only stand-in for reasoning_gym.utils.extract_answer()).
    oracle_answer = str(task.get("answer") or "")
    candidate = extract_candidate_answer(completion)
    reward = 0.0
    if isinstance(candidate, str) and len(candidate) > 0:
        if candidate == oracle_answer:
            reward = 1.0
        elif oracle_answer and oracle_answer in candidate:
            reward = len(oracle_answer) / len(candidate)
    passed = reward >= 1.0
    if passed:
        reason = "reasoning_gym_exact_match"
    elif reward > 0.0:
        reason = "reasoning_gym_partial_substring_match"
    else:
        reason = "reasoning_gym_no_match"
    return {
        "passed": passed,
        "reward": reward,
        "reason": reason,
        "normalized_completion": candidate,
        "normalized_answer": oracle_answer,
        "metadata": {
            "verifier": "reasoning_gym_basic_arithmetic",
            "ported_from": "reasoning_gym.dataset.ProceduralDataset.score_answer",
        },
    }

def verify_completion(completion, task):
    source = str(task.get("source_dataset") or "").casefold()
    verifier_name = str(task.get("verifier_name") or "").casefold()
    expected_text = str(task.get("answer") or "")
    candidate_raw = extract_candidate_answer(completion)
    if verifier_name == "materials_tensile_numeric":
        return verify_materials_tensile_numeric(completion, task)
    if verifier_name == "reasoning_gym_basic_arithmetic":
        return verify_reasoning_gym_basic_arithmetic(completion, task)
    if source == "calendar" or verifier_name == "calendar_gym":
        exp_cal_state = normalize_calendar_state(coerce_jsonish(expected_text))
        reward, reason = grade_calendar_response(completion, exp_cal_state)
        passed = reward > 0
        return {
            "passed": passed,
            "reward": float(reward),
            "reason": "calendar_gym_" + reason,
            "normalized_completion": json.dumps(extract_calendar_json_list(completion), sort_keys=True),
            "normalized_answer": json.dumps(exp_cal_state, sort_keys=True),
            "metadata": {
                "verifier": "calendar_gym",
                "resource_server": "resources_servers/calendar",
                "compatible_with": "NVIDIA-NeMo/Gym/resources_servers/calendar/utils.py",
                "expected_event_count": len(exp_cal_state),
            },
        }
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

def normalize_calendar_state(value):
    if not isinstance(value, dict):
        return {}
    normalized = {}
    for event_id, event in value.items():
        if isinstance(event, dict):
            normalized[str(event_id)] = event
    return normalized

def grade_calendar_response(assistant_response, exp_cal_state):
    if "<think>" in str(assistant_response):
        return 0, "think_found"
    if len(exp_cal_state) == 0:
        return 1, "pass"
    try:
        cal_state = extract_calendar_json_list(assistant_response)
        if cal_state is None or len(cal_state) == 0:
            return 0, "no_json_list"
        events_dict = {}
        for event in cal_state:
            if not isinstance(event, dict) or "event_id" not in event:
                return 0, "error_in_grading"
            events_dict[str(event["event_id"])] = event
        if len(events_dict) != len(exp_cal_state):
            return 0, "different_number_of_events"
        for event in cal_state:
            if is_calendar_event_conflicting(cal_state, event, exclude_event=event):
                return 0, "conflicting_events"
        for event_id, expected_event in exp_cal_state.items():
            if event_id not in events_dict:
                return 0, "different_number_of_events"
            if not is_calendar_constraint_satisfied(events_dict[event_id], expected_event):
                return 0, "constraint_violated"
    except Exception:
        return 0, "error_in_grading"
    return 1, "pass"

def extract_calendar_json_list(text):
    pattern = r"\[(?:[^\[\]]|\{[^}]*\})*\{(?:[^\[\]]|\{[^}]*\})*\}(?:[^\[\]]|\{[^}]*\})*\]"
    match = re.search(pattern, str(text or ""), re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None

def calendar_time_to_minutes(time_str):
    time_str = str(time_str).strip().lower()
    if "am" in time_str or "pm" in time_str:
        if ":" not in time_str:
            if "am" in time_str:
                hour = int(time_str.replace("am", ""))
                return hour * 60 if hour != 12 else 0
            if "pm" in time_str:
                hour = int(time_str.replace("pm", ""))
                return (hour * 60 if hour != 12 else 0) + 12 * 60
        if "am" in time_str:
            hour, minute = map(int, time_str.replace("am", "").split(":"))
            return (hour * 60 if hour != 12 else 0) + minute
        if "pm" in time_str:
            hour, minute = map(int, time_str.replace("pm", "").split(":"))
            return ((hour * 60 if hour != 12 else 0) + 12 * 60) + minute
        raise ValueError("Unable to parse time: " + str(time_str))
    hours, minutes = map(int, time_str.split(":"))
    return hours * 60 + minutes

def is_calendar_event_conflicting(events, check_event, exclude_event=None):
    event_start = calendar_time_to_minutes(check_event["start_time"])
    event_end = event_start + int(check_event["duration"])
    for existing in events:
        if exclude_event and existing == exclude_event:
            continue
        existing_start = calendar_time_to_minutes(existing["start_time"])
        existing_end = existing_start + int(existing["duration"])
        if not (event_end <= existing_start or event_start >= existing_end):
            return True
    return False

def is_calendar_constraint_satisfied(event, expected_event):
    if int(event["duration"]) != int(expected_event["duration"]):
        return False
    min_time = calendar_time_to_minutes(expected_event["min_time"])
    max_time = calendar_time_to_minutes(expected_event["max_time"])
    event_start = calendar_time_to_minutes(event["start_time"])
    event_end = event_start + int(event["duration"])
    if event_start < min_time or event_end > max_time:
        return False
    constraint = expected_event.get("constraint")
    if constraint is None:
        return True
    constraint = str(constraint)
    if constraint.startswith("before "):
        return event_end <= calendar_time_to_minutes(constraint.replace("before ", ""))
    if constraint.startswith("after "):
        return event_start >= calendar_time_to_minutes(constraint.replace("after ", ""))
    if constraint.startswith("between "):
        parts = constraint.replace("between ", "").split(" and ")
        if len(parts) != 2:
            raise ValueError("Invalid between constraint: " + constraint)
        return event_start >= calendar_time_to_minutes(parts[0]) and event_end <= calendar_time_to_minutes(parts[1])
    if constraint.startswith("at "):
        return event_start == calendar_time_to_minutes(constraint.replace("at ", ""))
    return True
'''
