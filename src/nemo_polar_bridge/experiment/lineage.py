"""Content-addressed lineage store for experiment-as-code runs.

Each registered experiment becomes a JSON asset record under a store directory,
keyed by spec id. A record carries two digests -- a *spec digest* (canonical
identity of the declared experiment) and a *compiled digest* (sha256 of the NeMo
base config plus the sorted overrides, i.e. what actually ran). Records link via
``parent``, forming a DAG the store traverses in both directions (Laguna
"end-to-end lineage", tech report S 2.1.1). Run artifacts bind forward by content
hash.

No GPU and no NeMo dependency: this is pure spec + compiler bookkeeping.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from .compile import CompiledExperiment, compile_spec
from .spec import ExperimentSpec


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_within(run_dir: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``run_dir``, rejecting paths that escape it."""
    base = run_dir.resolve()
    target = (run_dir / rel).resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"artifact path {rel!r} escapes run_dir {run_dir}")
    if not target.is_file():
        raise FileNotFoundError(f"artifact not found: {target}")
    return target


def spec_digest(spec: ExperimentSpec) -> str:
    """Canonical, order-independent sha256 of the declared spec."""
    canonical = json.dumps(
        spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return _sha256_text(canonical)


def compiled_digest(spec_or_compiled: Union[ExperimentSpec, CompiledExperiment]) -> str:
    """sha256 of ``base_config`` + sorted override args: what actually ran.

    The payload is a structured JSON array (``[base_config, [sorted args]]``) so
    the encoding is injective: JSON string-escaping prevents a newline embedded
    in a free-form override value (e.g. a model path) from colliding with an
    override boundary, which a flat newline-join would allow.
    """
    compiled = (
        spec_or_compiled
        if isinstance(spec_or_compiled, CompiledExperiment)
        else compile_spec(spec_or_compiled)
    )
    payload = json.dumps(
        [compiled.base_config, sorted(compiled.nemo_override_args())],
        separators=(",", ":"),
    )
    return _sha256_text(payload)


@dataclass
class LineageRecord:
    spec_id: str
    parent: Optional[str]
    spec_digest: str
    compiled_digest: str
    base_config: str
    matrix: dict
    algorithm: dict
    n_overrides: int
    artifacts: list = field(default_factory=list)

    def to_json_dict(self) -> dict:
        return {
            "spec_id": self.spec_id,
            "parent": self.parent,
            "spec_digest": self.spec_digest,
            "compiled_digest": self.compiled_digest,
            "base_config": self.base_config,
            "matrix": self.matrix,
            "algorithm": self.algorithm,
            "n_overrides": self.n_overrides,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "LineageRecord":
        return cls(
            spec_id=data["spec_id"],
            parent=data.get("parent"),
            spec_digest=data["spec_digest"],
            compiled_digest=data["compiled_digest"],
            base_config=data["base_config"],
            matrix=data.get("matrix", {}),
            algorithm=data.get("algorithm", {}),
            n_overrides=data.get("n_overrides", 0),
            artifacts=data.get("artifacts", []),
        )


def build_record(
    spec: ExperimentSpec, compiled: Optional[CompiledExperiment] = None
) -> LineageRecord:
    compiled = compiled or compile_spec(spec)
    return LineageRecord(
        spec_id=spec.id,
        parent=spec.parent,
        spec_digest=spec_digest(spec),
        compiled_digest=compiled_digest(compiled),
        base_config=compiled.base_config,
        matrix={
            "dataset_family": spec.dataset.family,
            "verifier_type": spec.verifier.type,
            "execution_type": spec.verifier.execution,
        },
        algorithm={"name": spec.algorithm.name, "advantage": spec.algorithm.advantage},
        n_overrides=len(compiled.nemo_overrides),
    )


def register_experiment(
    spec: ExperimentSpec,
    store_dir,
    *,
    compiled: Optional[CompiledExperiment] = None,
    run_dir=None,
    artifacts: Optional[list[str]] = None,
) -> LineageRecord:
    """Compile, build, and persist a lineage record under ``store_dir``.

    ``artifacts`` are run-relative paths hashed against ``run_dir`` to bind the
    spec forward to what it produced.
    """
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    record = build_record(spec, compiled)
    if artifacts:
        if run_dir is None:
            raise ValueError("run_dir is required when binding artifacts")
        run_dir = Path(run_dir)
        record.artifacts = [
            {"path": rel, "sha256": _sha256_file(_resolve_within(run_dir, rel))}
            for rel in artifacts
        ]
    (store_dir / f"{spec.id}.json").write_text(
        json.dumps(record.to_json_dict(), indent=2, sort_keys=True) + "\n"
    )
    return record


class LineageStore:
    """Read-side view over a directory of lineage records, with DAG traversal."""

    def __init__(self, store_dir):
        self.store_dir = Path(store_dir)
        self._records: dict[str, LineageRecord] = {}
        for path in sorted(self.store_dir.glob("*.json")):
            record = LineageRecord.from_json_dict(json.loads(path.read_text()))
            self._records[record.spec_id] = record

    def __contains__(self, spec_id: str) -> bool:
        return spec_id in self._records

    def get(self, spec_id: str) -> LineageRecord:
        return self._records[spec_id]

    def ancestors(self, spec_id: str) -> list[str]:
        """Parent chain, nearest first; cycle-safe.

        May include an unresolved terminal id when a spec declares a ``parent``
        that is not (yet) registered: the dangling id is surfaced rather than
        hidden, so callers must not assume every returned id is in the store
        (guard ``store.get`` with ``in`` for traversal-derived ids).
        """
        out: list[str] = []
        seen: set[str] = set()
        cursor = self._records[spec_id].parent
        while cursor is not None and cursor not in seen:
            seen.add(cursor)
            out.append(cursor)
            record = self._records.get(cursor)
            cursor = record.parent if record else None
        return out

    def children(self, spec_id: str) -> list[str]:
        return sorted(
            r.spec_id for r in self._records.values() if r.parent == spec_id
        )

    def descendants(self, spec_id: str) -> list[str]:
        out: set[str] = set()
        stack = self.children(spec_id)
        while stack:
            child = stack.pop()
            if child in out:
                continue
            out.add(child)
            stack.extend(self.children(child))
        return sorted(out)
