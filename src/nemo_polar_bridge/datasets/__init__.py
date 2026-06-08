"""Dataset adapters for NeMo + Polar external rollout experiments."""

from nemo_polar_bridge.datasets.base import DatasetAdapter, TaskSpec, VerifierResult
from nemo_polar_bridge.datasets.data_loader import (
    DEFAULT_NEMO_GYM_DATASET_ID,
    DEFAULT_REASONING_GYM_DATASET_ID,
    NeMoGymDatasetAdapter,
)
from nemo_polar_bridge.datasets.run_matrix import RunMatrixCell, build_run_matrix_cell
from nemo_polar_bridge.datasets.verifiers import verify_completion

__all__ = [
    "DEFAULT_NEMO_GYM_DATASET_ID",
    "DEFAULT_REASONING_GYM_DATASET_ID",
    "DatasetAdapter",
    "NeMoGymDatasetAdapter",
    "RunMatrixCell",
    "TaskSpec",
    "VerifierResult",
    "build_run_matrix_cell",
    "verify_completion",
]
