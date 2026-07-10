"""Weight-name adapters for disaggregated NeMo-to-vLLM refits."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

TensorT = TypeVar("TensorT")


def adapt_qwen35_language_only_weights(
    weights: Iterable[tuple[str, TensorT]],
    *,
    target_module_names: Iterable[str],
) -> tuple[list[tuple[str, TensorT]], int, int]:
    """Map HF Qwen3.5 checkpoint names onto vLLM's internal module names.

    NeMo AutoModel exports the HF composite namespace. vLLM text-only mode may
    expose either its causal-LM namespace or the conditional-generation Python
    module namespace (``language_model.model``). Without this mapping, NeMo's
    FP8 classifier misses every text linear and the refit leaves unusable
    weights in the dummy-initialized rollout model.

    Returns the adapted weights plus ``(rewritten_count, dropped_count)``.
    The adapter is a no-op when the target exposes the composite namespace.
    """

    materialized = list(weights)
    module_names = set(target_module_names)
    has_vllm_composite_text = any(
        name.startswith("language_model.model.layers") for name in module_names
    )
    has_hf_composite_text = any(
        name.startswith("model.language_model.layers") for name in module_names
    )
    has_causal_text = any(name.startswith("model.layers") for name in module_names)
    if has_hf_composite_text:
        return materialized, 0, 0
    if not has_vllm_composite_text and not has_causal_text:
        return materialized, 0, 0

    adapted: list[tuple[str, TensorT]] = []
    rewritten = 0
    dropped = 0
    for name, tensor in materialized:
        if name.startswith("model.visual."):
            dropped += 1
            continue
        if name.startswith("model.language_model."):
            suffix = name.removeprefix("model.language_model.")
            if has_vllm_composite_text:
                name = "language_model.model." + suffix
            else:
                name = "model." + suffix
            rewritten += 1
        elif name.startswith("lm_head.") and has_vllm_composite_text:
            name = "language_model." + name
            rewritten += 1
        adapted.append((name, tensor))
    return adapted, rewritten, dropped
