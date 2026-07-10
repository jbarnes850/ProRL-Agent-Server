"""Exact replay of rollout-time truncated sampling support for NeMo DTensor.

The rollout records the complete kept-token set for each sampled token. During
training, current-policy logits are normalized only over that recorded set.
This avoids reconstructing a potentially different nucleus from BF16 trainer
logits after generation used FP8 rollout weights.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import torch


SUPPORT_FIELDS = (
    "sampling_support_token_ids",
    "sampling_support_lengths",
    "sampling_support_mass",
)
_GRPO_PATCH_MARKER = "# nemo-polar: preserve rollout sampling support"
_GRPO_TRAIN_DATA_ANCHOR = """                            "sample_mask": repeated_batch["loss_multiplier"],
                        }
                    )"""
_GRPO_TRAIN_DATA_REPLACEMENT = _GRPO_TRAIN_DATA_ANCHOR + """
                    # nemo-polar: preserve rollout sampling support
                    for support_key in (
                        "sampling_support_token_ids",
                        "sampling_support_lengths",
                        "sampling_support_mass",
                    ):
                        if support_key not in flat_messages:
                            raise RuntimeError(
                                f"required rollout sampling field {support_key!r} is missing"
                            )
                        train_data[support_key] = flat_messages[support_key]"""
_GRPO_SYNC_LOG_ANCHOR = """                log_data["prev_logprobs"] = train_data["prev_logprobs"].tolist()

                logger.log_batched_dict_as_jsonl("""
_GRPO_ASYNC_LOG_ANCHOR = """            log_data["prev_logprobs"] = train_data["prev_logprobs"].tolist()
            logger.log_batched_dict_as_jsonl("""
_GRPO_SYNC_LOG_REPLACEMENT = """                log_data["prev_logprobs"] = train_data["prev_logprobs"].tolist()
                for support_key in (
                    "sampling_support_token_ids",
                    "sampling_support_lengths",
                    "sampling_support_mass",
                ):
                    log_data[support_key] = train_data[support_key].tolist()

                logger.log_batched_dict_as_jsonl("""
_GRPO_ASYNC_LOG_REPLACEMENT = """            log_data["prev_logprobs"] = train_data["prev_logprobs"].tolist()
            for support_key in (
                "sampling_support_token_ids",
                "sampling_support_lengths",
                "sampling_support_mass",
            ):
                log_data[support_key] = train_data[support_key].tolist()
            logger.log_batched_dict_as_jsonl("""


def sampling_support_replay_enabled() -> bool:
    return os.environ.get("NEMO_POLAR_REQUIRE_SAMPLING_SUPPORT_REPLAY") == "1"


def patch_c236_grpo_for_sampling_support(path: str | Path) -> bool:
    """Patch the selected c236 GRPO driver before it is imported.

    c236 constructs train data independently in synchronous and asynchronous
    GRPO. Both exact anchors must be present; otherwise this refuses to mutate
    the installed source.
    """

    target = Path(path)
    source = target.read_text()
    if _GRPO_PATCH_MARKER in source:
        return False
    anchor_count = source.count(_GRPO_TRAIN_DATA_ANCHOR)
    if anchor_count != 2:
        raise RuntimeError(
            "refusing sampling-support patch: expected two c236 GRPO train-data "
            f"anchors, found {anchor_count}"
        )
    if source.count(_GRPO_SYNC_LOG_ANCHOR) != 1:
        raise RuntimeError("refusing sampling-support patch: sync logging anchor changed")
    if source.count(_GRPO_ASYNC_LOG_ANCHOR) != 1:
        raise RuntimeError("refusing sampling-support patch: async logging anchor changed")
    patched = source.replace(
        _GRPO_TRAIN_DATA_ANCHOR, _GRPO_TRAIN_DATA_REPLACEMENT
    ).replace(_GRPO_SYNC_LOG_ANCHOR, _GRPO_SYNC_LOG_REPLACEMENT)
    patched = patched.replace(_GRPO_ASYNC_LOG_ANCHOR, _GRPO_ASYNC_LOG_REPLACEMENT)
    target.write_text(patched)
    return True


def install_c236_grpo_sampling_support_patch(
    path: str | Path = "/opt/nemo-rl/nemo_rl/algorithms/grpo.py",
) -> bool:
    """Install the driver patch before NeMo imports ``algorithms.grpo``."""

    if not sampling_support_replay_enabled():
        return False
    if "nemo_rl.algorithms.grpo" in sys.modules:
        raise RuntimeError(
            "sampling-support replay must be installed before importing nemo_rl.algorithms.grpo"
        )
    return patch_c236_grpo_for_sampling_support(path)


def _local_logits(logits: Any) -> torch.Tensor:
    """Return full local logits, rejecting multi-rank DTensor layouts."""

    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover - supported torch builds provide it
        DTensor = ()  # type: ignore[assignment,misc]
    if isinstance(logits, DTensor):
        mesh_size = int(logits.device_mesh.size())
        if mesh_size != 1:
            raise NotImplementedError(
                "recorded sampling-support replay currently requires tensor_parallel_size=1"
            )
        return logits.to_local()
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"expected tensor logits, got {type(logits)!r}")
    return logits


def restricted_support_next_token_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    support_token_ids: torch.Tensor,
    support_lengths: torch.Tensor,
    active_token_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute selected-token logprobs over recorded rollout support.

    Support at token position ``t`` describes the distribution that sampled
    ``input_ids[t]``. Consequently logits position ``t-1`` is paired with
    support position ``t``. Only active response tokens are gathered, avoiding
    a ``sequence_length x support_width`` allocation for long prompts.
    """

    logits = _local_logits(logits)
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B,S,V], got {tuple(logits.shape)}")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B,S]")
    if support_token_ids.ndim != 3:
        raise ValueError("sampling_support_token_ids must have shape [B,S,K]")
    if support_lengths.ndim != 2:
        raise ValueError("sampling_support_lengths must have shape [B,S]")
    batch_size, sequence_length, vocab_size = logits.shape
    expected_2d = (batch_size, sequence_length)
    if tuple(input_ids.shape) != expected_2d:
        raise ValueError("input_ids shape must match logits batch and sequence dimensions")
    if tuple(support_token_ids.shape[:2]) != expected_2d:
        raise ValueError("sampling support shape must align with input_ids")
    if tuple(support_lengths.shape) != expected_2d:
        raise ValueError("sampling support lengths must align with input_ids")
    if tuple(active_token_mask.shape) == expected_2d:
        active_next = active_token_mask[:, 1:]
    elif tuple(active_token_mask.shape) == (batch_size, sequence_length - 1):
        active_next = active_token_mask
    else:
        raise ValueError("active_token_mask must have shape [B,S] or [B,S-1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    output = torch.zeros(
        (batch_size, sequence_length - 1),
        dtype=torch.float32,
        device=logits.device,
    )
    active_rows = torch.nonzero(active_next > 0, as_tuple=False)
    if active_rows.numel() == 0:
        return output

    batch_index = active_rows[:, 0]
    predictor_index = active_rows[:, 1]
    token_index = predictor_index + 1
    lengths = support_lengths[batch_index, token_index].to(torch.long)
    support_width = support_token_ids.shape[-1]
    if bool(((lengths <= 0) | (lengths > support_width)).any().item()):
        raise ValueError("every active token must have a nonempty in-bounds support")

    support_ids = support_token_ids[batch_index, token_index].to(torch.long)
    support_positions = torch.arange(support_width, device=logits.device)
    valid_support = support_positions.unsqueeze(0) < lengths.unsqueeze(1)
    valid_ids = support_ids[valid_support]
    if bool(((valid_ids < 0) | (valid_ids >= vocab_size)).any().item()):
        raise ValueError("sampling support contains an out-of-vocabulary token ID")

    sampled_ids = input_ids[batch_index, token_index].to(torch.long)
    sampled_in_support = ((support_ids == sampled_ids.unsqueeze(1)) & valid_support).any(
        dim=1
    )
    if not bool(sampled_in_support.all().item()):
        raise ValueError("sampled token is absent from its rollout sampling support")

    predictor_logits = logits[batch_index, predictor_index].to(torch.float32)
    if temperature != 1.0:
        predictor_logits = predictor_logits / temperature
    gathered = predictor_logits.gather(1, support_ids.clamp(min=0))
    gathered = gathered.masked_fill(~valid_support, -torch.inf)
    normalizer = torch.logsumexp(gathered, dim=1)
    selected = predictor_logits.gather(1, sampled_ids.unsqueeze(1)).squeeze(1)
    values = selected - normalizer
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("restricted-support logprobs must be finite")
    return output.index_put((batch_index, predictor_index), values)


def _require_support_fields(data: Any) -> None:
    missing = [field for field in SUPPORT_FIELDS[:2] if field not in data]
    if missing:
        raise RuntimeError(f"recorded sampling-support replay is missing {missing}")


def install_nemo_worker_sampling_support_patch() -> type[Any] | None:
    """Patch loss preparation and return a support-aware logprob processor.

    The selected Qwen3.5 path is TP=1, CP=1, and sequence packing disabled.
    Other layouts fail closed rather than silently reverting to nucleus
    reconstruction.
    """

    if not sampling_support_replay_enabled():
        return None

    from nemo_rl.algorithms.logits_sampling_utils import (
        need_top_k_or_top_p_filtering,
    )
    from nemo_rl.algorithms.loss.interfaces import LossInputType
    import nemo_rl.models.automodel.train as automodel_train

    original_prepare_loss_input = automodel_train.prepare_loss_input
    base_logprobs_post_processor = automodel_train.LogprobsPostProcessor

    def prepare_loss_input_with_recorded_support(
        logits: torch.Tensor,
        data: Any,
        loss_fn: Any,
        vocab_parallel_rank: int | None = None,
        vocab_parallel_group: Any = None,
        context_parallel_group: Any = None,
        sampling_params: Any = None,
        d2t: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], Any]:
        if not need_top_k_or_top_p_filtering(sampling_params):
            return original_prepare_loss_input(
                logits,
                data,
                loss_fn,
                vocab_parallel_rank=vocab_parallel_rank,
                vocab_parallel_group=vocab_parallel_group,
                context_parallel_group=context_parallel_group,
                sampling_params=sampling_params,
                d2t=d2t,
            )
        _require_support_fields(data)
        if loss_fn.input_type != LossInputType.LOGPROB:
            raise NotImplementedError(
                "recorded sampling-support replay requires a logprob-input loss"
            )
        if vocab_parallel_group is not None or context_parallel_group is not None:
            raise NotImplementedError(
                "recorded sampling-support replay currently requires TP=1 and CP=1"
            )
        if getattr(loss_fn, "use_linear_ce_fusion", False):
            raise NotImplementedError(
                "recorded sampling-support replay is incompatible with fused linear CE"
            )
        active_mask = data["token_mask"] * data["sample_mask"].unsqueeze(-1)
        restricted = restricted_support_next_token_logprobs(
            logits,
            data["input_ids"],
            data["sampling_support_token_ids"],
            data["sampling_support_lengths"],
            active_mask,
            temperature=float(sampling_params.temperature),
        )
        if getattr(loss_fn, "reference_policy_kl_penalty", 0) != 0:
            unfiltered, data = original_prepare_loss_input(
                logits,
                data,
                loss_fn,
                vocab_parallel_rank=vocab_parallel_rank,
                vocab_parallel_group=vocab_parallel_group,
                context_parallel_group=context_parallel_group,
                sampling_params=None,
                d2t=d2t,
            )
            data["curr_logprobs_unfiltered"] = unfiltered["next_token_logprobs"]
        return {"next_token_logprobs": restricted}, data

    class RecordedSupportLogprobsPostProcessor(base_logprobs_post_processor):
        def __call__(
            self,
            logits: torch.Tensor,
            data_dict: Any,
            processed_inputs: Any,
            original_batch_size: int,
            original_seq_len: int,
            sequence_dim: int = 1,
        ) -> torch.Tensor:
            if not need_top_k_or_top_p_filtering(self.sampling_params):
                return super().__call__(
                    logits,
                    data_dict,
                    processed_inputs,
                    original_batch_size,
                    original_seq_len,
                    sequence_dim,
                )
            _require_support_fields(data_dict)
            if self.cp_size != 1 or self.enable_seq_packing:
                raise NotImplementedError(
                    "recorded sampling-support replay requires CP=1 and sequence packing disabled"
                )
            active_mask = data_dict["token_mask"] * data_dict[
                "sample_mask"
            ].unsqueeze(-1)
            next_logprobs = restricted_support_next_token_logprobs(
                logits,
                processed_inputs.input_ids,
                data_dict["sampling_support_token_ids"],
                data_dict["sampling_support_lengths"],
                active_mask,
                temperature=float(self.sampling_params.temperature),
            )
            token_logprobs = torch.cat(
                [torch.zeros_like(next_logprobs[:, :1]), next_logprobs], dim=1
            )
            post_attention_mask = torch.zeros_like(token_logprobs, dtype=torch.bool)
            for row, length in enumerate(data_dict["input_lengths"]):
                post_attention_mask[row, : int(length.item())] = True
            return token_logprobs * post_attention_mask

    automodel_train.prepare_loss_input = prepare_loss_input_with_recorded_support
    return RecordedSupportLogprobsPostProcessor
