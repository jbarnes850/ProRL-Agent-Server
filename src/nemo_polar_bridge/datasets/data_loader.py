"""Load NeMo Gym-shaped rows and normalize them into Polar task specs."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import quote, urlencode
from urllib.request import urlopen

from nemo_polar_bridge.datasets.base import DatasetAdapter, TaskSpec


DEFAULT_NEMO_GYM_DATASET_ID = "nvidia/Nemotron-RL-ReasoningGym-v1"
DEFAULT_REASONING_GYM_DATASET_ID = DEFAULT_NEMO_GYM_DATASET_ID
DATASETS_SERVER = "https://datasets-server.huggingface.co"
AGENTIC_METADATA_KEYS = (
    "environment_ref",
    "resource_ref",
    "resources_ref",
    "resources_server",
    "runtime_ref",
    "verifier_ref",
    "max_turns",
    "max_steps",
    "max_rollout_turns",
    "tool_schema",
    "tools",
    "state",
    "initial_state",
    "seed",
    "session",
)


class NeMoGymDatasetAdapter(DatasetAdapter):
    """Load NeMo Gym JSONL/HF rows.

    The stable schema is NeMo Gym's row shape:
    ``responses_create_params.input``, ``answer``/``expected_answer``,
    ``metadata``, ``agent_ref``, and ``license``. Direct Hugging Face loading is
    only an acquisition path; rows are normalized through this Gym contract
    before Polar artifacts are written.
    """

    name = "nemo_gym"

    def __init__(
        self,
        *,
        dataset_id: str = DEFAULT_REASONING_GYM_DATASET_ID,
        config: str = "default",
        split: str = "train",
        local_jsonl: str | None = None,
        source_datasets: Iterable[str] | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.config = config
        self.split = split
        self.local_jsonl = local_jsonl
        self.source_datasets = {value for value in (source_datasets or []) if value}
        self._audit: dict[str, Any] = {}

    def load_tasks(self, *, limit: int, scan_rows: int) -> list[TaskSpec]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if scan_rows <= 0:
            raise ValueError("scan_rows must be positive")

        inspected_rows: list[dict[str, Any]] = []
        tasks: list[TaskSpec] = []
        skipped = 0
        for row_idx, source_row in self._iter_source_rows(scan_rows=scan_rows):
            inspected_rows.append({"row_idx": row_idx, "row": source_row})
            gym_row = normalize_to_gym_row(source_row)
            task = task_from_gym_row(
                gym_row,
                dataset_id=self.dataset_id,
                config=self.config,
                split=self.split,
                row_idx=row_idx,
            )
            if task is None:
                skipped += 1
                continue
            if self.source_datasets and task.source_dataset not in self.source_datasets:
                continue
            tasks.append(task)
            if len(tasks) >= limit:
                break

        self._audit = {
            "dataset_id": self.dataset_id,
            "config": self.config,
            "split": self.split,
            "loader": self.name,
            "canonical_schema": "nemo_gym_jsonl",
            "required_fields": ["responses_create_params.input", "answer|expected_answer"],
            "optional_fields": ["metadata", "agent_ref", "license", "question", "uuid"],
            "local_jsonl": self.local_jsonl,
            "source_dataset_filter": sorted(self.source_datasets),
            "scanned_rows": len(inspected_rows),
            "skipped_unmappable_rows": skipped,
            "selected_tasks": [task.stable_id() for task in tasks],
            "inspected_rows": [_summarize_row(wrapped) for wrapped in inspected_rows[:10]],
        }
        if len(tasks) < limit:
            raise ValueError(
                f"only found {len(tasks)} usable NeMo Gym tasks for filters "
                f"{sorted(self.source_datasets)} after scanning {len(inspected_rows)} rows"
            )
        return tasks

    def audit(self) -> dict[str, Any]:
        return dict(self._audit)

    def _iter_source_rows(self, *, scan_rows: int) -> Iterable[tuple[int, dict[str, Any]]]:
        if self.local_jsonl:
            yield from _iter_local_jsonl(Path(self.local_jsonl), limit=scan_rows)
            return
        try:
            yield from _iter_datasets_server_rows(
                dataset_id=self.dataset_id,
                config=self.config,
                split=self.split,
                length=scan_rows,
            )
        except (URLError, TimeoutError, OSError, ValueError):
            yield from _iter_hf_jsonl_rows(
                dataset_id=self.dataset_id,
                split=self.split,
                limit=scan_rows,
            )


def normalize_to_gym_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a Gym-shaped row, coercing common HF reasoning rows when needed."""

    normalized = dict(row)
    responses_create_params = normalized.get("responses_create_params")
    if not isinstance(responses_create_params, dict):
        prompt = _first_present(
            normalized,
            ("question", "prompt", "input", "instruction", "problem"),
        )
        if prompt is not None:
            normalized["responses_create_params"] = {
                "input": [{"role": "user", "content": str(prompt)}]
            }
    if "expected_answer" not in normalized and "answer" not in normalized:
        answer = _first_present(
            normalized,
            ("target", "output", "label", "ground_truth", "exp_cal_state"),
        )
        if answer is not None:
            normalized["answer"] = answer
    return normalized


def task_from_gym_row(
    row: dict[str, Any],
    *,
    dataset_id: str,
    config: str,
    split: str,
    row_idx: int | None,
) -> TaskSpec | None:
    responses_create_params = row.get("responses_create_params")
    if not isinstance(responses_create_params, dict):
        return None
    messages = responses_create_params.get("input")
    prompt = _messages_to_prompt(messages)
    answer = row.get("answer", row.get("expected_answer"))
    if not prompt or answer in (None, ""):
        return None
    answer_text = answer if isinstance(answer, str) else json.dumps(answer, sort_keys=True)

    metadata = _parse_metadata(row.get("metadata"))
    source_dataset = _string_or_none(metadata.get("source_dataset") or row.get("source_dataset"))
    if source_dataset is None:
        source_dataset = _infer_source_dataset(dataset_id)
    task_id = _string_or_none(
        row.get("uuid")
        or row.get("id")
        or row.get("task_id")
        or metadata.get("task_id")
    )
    if task_id is None:
        row_part = "unknown" if row_idx is None else str(row_idx)
        source_part = source_dataset or "default"
        task_id = f"{dataset_id}:{split}:{source_part}:{row_part}"

    agent_ref = row.get("agent_ref")
    if agent_ref is not None and not isinstance(agent_ref, dict):
        agent_ref = {"raw": agent_ref}
    metadata = _with_agentic_metadata(
        metadata,
        row=row,
        responses_create_params=responses_create_params,
        agent_ref=agent_ref,
    )

    return TaskSpec(
        task_id=task_id,
        responses_create_params=responses_create_params,
        prompt=prompt,
        answer=str(answer_text),
        dataset_id=dataset_id,
        split=split,
        config=config,
        row_idx=row_idx,
        source_dataset=source_dataset,
        cohort=source_dataset or _agent_ref_name(agent_ref),
        metadata=metadata,
        agent_ref=agent_ref,
        verifier_name=str(
            row.get("verifier")
            or metadata.get("verifier")
            or _infer_verifier_name(dataset_id)
            or "exact_normalized"
        ),
        license=_string_or_none(row.get("license")),
    )


def _iter_datasets_server_rows(
    *,
    dataset_id: str,
    config: str,
    split: str,
    length: int,
) -> Iterable[tuple[int, dict[str, Any]]]:
    payload = _fetch_json(
        "/rows",
        {
            "dataset": dataset_id,
            "config": config,
            "split": split,
            "offset": 0,
            "length": length,
        },
    )
    for wrapped in payload.get("rows") or []:
        row = wrapped.get("row")
        if isinstance(row, dict):
            yield int(wrapped.get("row_idx") or 0), row


def _iter_hf_jsonl_rows(
    *,
    dataset_id: str,
    split: str,
    limit: int,
) -> Iterable[tuple[int, dict[str, Any]]]:
    safe_dataset = quote(dataset_id, safe="/")
    url = f"https://huggingface.co/datasets/{safe_dataset}/resolve/main/data/{split}.jsonl"
    with urlopen(url, timeout=60) as response:
        for row_idx, raw_line in enumerate(response):
            if row_idx >= limit:
                break
            line = raw_line.decode("utf-8").strip()
            if line:
                yield row_idx, json.loads(line)


def _iter_local_jsonl(path: Path, *, limit: int) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open() as handle:
        for row_idx, line in enumerate(handle):
            if row_idx >= limit:
                break
            line = line.strip()
            if line:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"local JSONL row {row_idx} is not an object")
                yield row_idx, row


def _fetch_json(path: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{DATASETS_SERVER}{path}?{urlencode(params)}"
    with urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _messages_to_prompt(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for message in value:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif content is not None:
            parts.append(json.dumps(content, sort_keys=True))
    return "\n".join(part for part in parts if part).strip()


def _first_present(row: dict[str, Any], keys: Iterable[str]) -> Any | None:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw_metadata": value}
        return parsed if isinstance(parsed, dict) else {"metadata": parsed}
    return {}


def _with_agentic_metadata(
    metadata: dict[str, Any],
    *,
    row: dict[str, Any],
    responses_create_params: dict[str, Any],
    agent_ref: dict[str, Any] | None,
) -> dict[str, Any]:
    agentic = dict(metadata.get("agentic") or {})
    for key in AGENTIC_METADATA_KEYS:
        if key in row and row[key] not in (None, ""):
            agentic.setdefault(key, row[key])
    if "tools" in responses_create_params and responses_create_params["tools"]:
        agentic.setdefault("tools", responses_create_params["tools"])
    if "tool_choice" in responses_create_params and responses_create_params["tool_choice"]:
        agentic.setdefault("tool_choice", responses_create_params["tool_choice"])
    if agent_ref:
        agentic.setdefault("agent_ref", agent_ref)
    if not agentic:
        return metadata
    return {**metadata, "agentic": agentic}


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _infer_source_dataset(dataset_id: str) -> str | None:
    normalized = dataset_id.casefold()
    if "calendar" in normalized:
        return "calendar"
    if "workplace" in normalized:
        return "workplace"
    return None


def _infer_verifier_name(dataset_id: str) -> str | None:
    normalized = dataset_id.casefold()
    if "calendar" in normalized:
        return "calendar_gym"
    return None


def _agent_ref_name(agent_ref: Any) -> str | None:
    if isinstance(agent_ref, dict):
        return _string_or_none(agent_ref.get("name"))
    return None


def _summarize_row(wrapped: dict[str, Any]) -> dict[str, Any]:
    row = normalize_to_gym_row(wrapped["row"])
    metadata = _parse_metadata(row.get("metadata"))
    task = task_from_gym_row(
        row,
        dataset_id="audit",
        config="default",
        split="train",
        row_idx=wrapped["row_idx"],
    )
    return {
        "row_idx": wrapped["row_idx"],
        "keys": sorted(row.keys()),
        "has_responses_create_params": isinstance(row.get("responses_create_params"), dict),
        "source_dataset": metadata.get("source_dataset") or row.get("source_dataset"),
        "agent_ref": row.get("agent_ref"),
        "prompt_head": "" if task is None else task.prompt[:240],
        "answer": row.get("answer", row.get("expected_answer")),
        "metadata_head": {key: metadata[key] for key in sorted(metadata)[:8]},
    }
