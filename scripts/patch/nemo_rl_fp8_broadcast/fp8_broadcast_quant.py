# Copyright (c) 2026. Apache License, Version 2.0.
"""Producer-side fp8 block-quant for weight-sync lever 2 (torch-only).

Bind-mounted into the pinned NeMo RL image at
``/opt/nemo-rl/nemo_rl/utils/fp8_broadcast_quant.py`` so the DTensor policy
worker can quantize the weight-sync broadcast payload to fp8 instead of shipping
bf16 and requantizing post-receipt in vLLM.

``cast_tensor_to_fp8_blockwise_producer`` is a verbatim, self-contained copy of
``nemo_rl.models.generation.vllm.quantization.fp8.cast_tensor_to_fp8_blockwise``
(no vLLM import). Fed the same input the consumer feeds it today
(``weight.to(bf16).to(float32)``) it returns bit-identical fp8 bytes and scale,
so the rollout policy is unchanged. ``test_fp8_equivalence.py`` proves that
identity against the image's own reference function, run-agnostically.
"""

from __future__ import annotations

import os
import re

import torch

FP8_BLOCK_SIZE = (128, 128)

# Qwen3 dense fp8-eligible weights: the seven linear projections per layer.
# vLLM's default block-fp8 quantizes exactly these; embeddings, norms, and
# lm_head stay bf16. A predicate mismatch surfaces as a shape error at load or a
# failed equivalence check, never a silent corruption.
_FP8_BROADCAST_NAME_RE = re.compile(
    r"\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
)


def fp8_broadcast_enabled() -> bool:
    return os.getenv("NRL_FP8_BROADCAST") == "1"


def fp8_broadcast_should_quantize(name: str, tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and bool(_FP8_BROADCAST_NAME_RE.search(name))


def fp8_broadcast_scale_shape(shape) -> torch.Size:
    b0, b1 = FP8_BLOCK_SIZE
    return torch.Size(
        [(int(shape[0]) + b0 - 1) // b0, (int(shape[1]) + b1 - 1) // b1]
    )


def cast_tensor_to_fp8_blockwise_producer(
    data_hp, weight_block_size, use_weight_pow2_scale: bool = False
):
    """Verbatim block-fp8 cast, self-contained (no vLLM import).

    Bit-identical to nemo_rl...quantization.fp8.cast_tensor_to_fp8_blockwise on
    the same input, so producer-side quantization exactly replaces the vLLM
    post-receipt requant it removes from the wire.
    """
    assert len(data_hp.shape) == 2, "Only 2d input tensor is supported"
    block_size1 = weight_block_size[1]
    block_size0 = weight_block_size[0]
    shape_before_padding = data_hp.shape
    if data_hp.shape[1] % block_size1 != 0 or data_hp.shape[0] % block_size0 != 0:
        pad1 = (
            0
            if data_hp.shape[1] % block_size1 == 0
            else block_size1 - data_hp.shape[1] % block_size1
        )
        pad0 = (
            0
            if data_hp.shape[0] % block_size0 == 0
            else block_size0 - data_hp.shape[0] % block_size0
        )
        data_hp = torch.nn.functional.pad(
            data_hp, (0, pad1, 0, pad0), mode="constant", value=data_hp[-1, -1]
        )
    max_dtype = torch.finfo(torch.float8_e4m3fn).max
    original_shape = data_hp.shape
    blk_m, blk_n = data_hp.shape[0] // block_size0, data_hp.shape[1] // block_size1
    assert block_size1 == block_size0
    data_hp = data_hp.reshape(blk_m, block_size0, blk_n, block_size1)
    data_hp = data_hp.permute(0, 2, 1, 3)
    data_hp = data_hp.to(torch.float32).contiguous().flatten(start_dim=2)
    max_abs = torch.amax(torch.abs(data_hp), dim=-1, keepdim=True)
    descale = max_abs / max_dtype
    if use_weight_pow2_scale:
        exponent = torch.ceil(torch.log2(descale))
        exponent = torch.clamp(exponent, min=-127, max=127) + 127
        exponent = exponent.to(torch.uint8)
        scale_fp = torch.where(
            exponent == 0, 1.0, torch.exp2(127 - exponent.to(torch.float32))
        )
        descale_fp = torch.reciprocal(scale_fp)
    else:
        scale_fp = max_dtype / max_abs
        scale_fp = torch.where(max_abs == 0, 1.0, scale_fp)
        scale_fp = torch.where(max_abs == torch.inf, 1.0, scale_fp)
        descale_fp = torch.reciprocal(scale_fp)
    data_lp = torch.clamp(data_hp * scale_fp, min=-1 * max_dtype, max=max_dtype)
    fp_data = data_lp.to(torch.float8_e4m3fn)
    fp_data = (
        fp_data.reshape(blk_m, blk_n, block_size0, block_size1)
        .permute(0, 2, 1, 3)
        .reshape(original_shape)
    )
    if data_hp.shape != shape_before_padding:
        fp_data = fp_data[: shape_before_padding[0], : shape_before_padding[1]]
    return fp_data, descale_fp
