#!/usr/bin/env python3
"""Equivalence gate for weight-sync lever 2 (run inside the pinned image).

Proves the producer-side block-fp8 cast used by the fp8 broadcast path is
bit-identical to the image's own post-receipt requant
(nemo_rl.models.generation.vllm.quantization.fp8.cast_tensor_to_fp8_blockwise),
on the same input both paths feed it. Because the function is deterministic and
the input is identical (weight.to(bf16).to(float32)), identity here proves the
rollout policy is unchanged regardless of which GPU runs the cast -- run-agnostic,
no confounding from async sampling nondeterminism.

    docker run --rm --gpus all -v <repo>:/work/repo:ro <image> \
      python /work/repo/scripts/patch/nemo_rl_fp8_broadcast/test_fp8_equivalence.py
"""

from __future__ import annotations

import ast
import os
import sys
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from fp8_broadcast_quant import (  # noqa: E402
    FP8_BLOCK_SIZE,
    cast_tensor_to_fp8_blockwise_producer,
    fp8_broadcast_scale_shape,
    fp8_broadcast_should_quantize,
)

DEVICE = os.getenv("TEST_DEVICE", "cpu")


def _load_reference_cast():
    """Extract the image's own cast_tensor_to_fp8_blockwise from the pristine
    fp8.py copy (byte-identical to /opt/nemo-rl/.../quantization/fp8.py, pulled
    from the pinned image) and exec its exact source in a torch-only namespace.

    The function is pure torch; only the module's top-level `import vllm` blocks a
    normal import. Exec'ing the exact source runs the real reference logic without
    that dependency, so the comparison is against the image's function, not a
    hand-retyped copy.
    """
    ref_path = os.path.join(_HERE, "_ref_fp8_quant.py")
    source = open(ref_path).read()
    tree = ast.parse(source)
    fn_src = None
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "cast_tensor_to_fp8_blockwise"
        ):
            fn_src = ast.get_source_segment(source, node)
            break
    if fn_src is None:
        raise RuntimeError("cast_tensor_to_fp8_blockwise not found in _ref_fp8_quant.py")
    # Our runs use plain fp8 (no pow2 weight scaling); the function reads
    # global_fp8_config.use_weight_pow2_scale.
    ns = {
        "torch": torch,
        "global_fp8_config": types.SimpleNamespace(use_weight_pow2_scale=False),
    }
    exec(compile(fn_src, ref_path, "exec"), ns)
    return ns["cast_tensor_to_fp8_blockwise"]


_ref_cast = _load_reference_cast()

# Real Qwen3-4B-Instruct-2507 linear-weight shapes (all divisible by 128) plus
# deliberate non-128-multiple shapes to exercise the padding path.
SHAPES = [
    (4096, 2560),   # q_proj
    (1024, 2560),   # k_proj / v_proj
    (2560, 4096),   # o_proj
    (9728, 2560),   # gate_proj / up_proj
    (2560, 9728),   # down_proj
    (256, 384),     # small, divisible
    (130, 300),     # NOT divisible by 128 -> padding path both dims
    (128, 129),     # off-by-one column
    (127, 128),     # off-by-one row
]


def _make_input(shape, kind, seed):
    torch.manual_seed(seed)
    if kind == "normal":
        x = torch.randn(shape, dtype=torch.float32)
    elif kind == "large":
        x = torch.randn(shape, dtype=torch.float32) * 500.0  # force fp8 saturation
    elif kind == "with_zeros":
        x = torch.randn(shape, dtype=torch.float32)
        x[: shape[0] // 2] = 0.0  # zero blocks -> max_abs==0 branch
    else:
        raise ValueError(kind)
    # Mirror the real data path: the wire carries bf16, both casts see bf16->f32.
    return x.to(torch.bfloat16).to(torch.float32).to(DEVICE)


def main() -> int:
    failures = 0
    checks = 0
    for shape in SHAPES:
        for kind in ("normal", "large", "with_zeros"):
            x = _make_input(shape, kind, seed=1234 + shape[0] + shape[1])
            mine_fp8, mine_scale = cast_tensor_to_fp8_blockwise_producer(
                x.clone(), FP8_BLOCK_SIZE
            )
            ref_fp8, ref_scale = _ref_cast(
                x.clone(), weight_block_size=list(FP8_BLOCK_SIZE)
            )
            checks += 1

            data_ok = torch.equal(
                mine_fp8.view(torch.uint8), ref_fp8.view(torch.uint8)
            )
            scale_ok = torch.equal(mine_scale, ref_scale)
            # scale-shape helper must match the reference's squeezed scale.
            declared = fp8_broadcast_scale_shape(torch.Size(shape))
            shape_ok = tuple(declared) == tuple(torch.squeeze(ref_scale, dim=-1).shape)

            if not (data_ok and scale_ok and shape_ok):
                failures += 1
                print(
                    f"FAIL {shape} {kind}: "
                    f"data={data_ok} scale={scale_ok} scale_shape={shape_ok} "
                    f"(mine_scale{tuple(mine_scale.shape)} ref_scale{tuple(ref_scale.shape)})"
                )

    # Predicate must match the intended Qwen3 dense fp8 set exactly.
    w = torch.zeros(4, 4)
    predicate_cases = {
        "model.layers.0.self_attn.q_proj.weight": True,
        "model.layers.5.self_attn.k_proj.weight": True,
        "model.layers.5.self_attn.v_proj.weight": True,
        "model.layers.5.self_attn.o_proj.weight": True,
        "model.layers.9.mlp.gate_proj.weight": True,
        "model.layers.9.mlp.up_proj.weight": True,
        "model.layers.9.mlp.down_proj.weight": True,
        "model.embed_tokens.weight": False,
        "lm_head.weight": False,
        "model.layers.0.input_layernorm.weight": False,
        "model.layers.0.self_attn.q_norm.weight": False,
        "model.norm.weight": False,
        "model.layers.0.self_attn.q_proj.bias": False,
    }
    for name, expected in predicate_cases.items():
        got = fp8_broadcast_should_quantize(name, w)
        checks += 1
        if got != expected:
            failures += 1
            print(f"FAIL predicate {name}: got {got}, expected {expected}")
    # 1D tensor must never be quantized even if the name matches.
    if fp8_broadcast_should_quantize(
        "model.layers.0.self_attn.q_proj.weight", torch.zeros(4)
    ):
        failures += 1
        print("FAIL predicate: 1D tensor was marked fp8")
    checks += 1

    print(f"\n{checks - failures}/{checks} checks passed on device={DEVICE}")
    if failures:
        print(f"EQUIVALENCE FAILED: {failures} mismatch(es)")
        return 1
    print("EQUIVALENCE PASSED: producer fp8 cast is bit-identical to vLLM requant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
