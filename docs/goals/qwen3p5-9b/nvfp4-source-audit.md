# NVFP4 Source Audit For Qwen3.5-9B

Status: source-complete; live serving unproven
Date: 2026-07-10
Decision: keep FP8 rollout as the Stage 3 baseline; admit NVFP4 only as a
separate Stage 6 serving and refit ablation.

## Verdict

The proposed tool boundary is wrong. NeMo AutoModel supports dense Qwen3.5-9B
training, but it is not the current NVIDIA NVFP4 compressor. The relevant stack
is ModelOpt for PTQ/export, vLLM for a static ModelOpt checkpoint, and the
Megatron-only NeMo RL QARL path if packed NVFP4 weights must be regenerated and
refitted after every policy update.

The article does not establish near-zero loss for dense Qwen3.5-9B. It shows a
close average reward result for Qwen3-30B-A3B MoE at 8K, but its final NVFP4
recipe still has approximately 3.26 times the BF16 train/rollout absolute
logprob difference in the embedded data. Its online SGLang implementation is
explicitly MoE-only and does not quantize dense linears.

Confidence:

- AutoModel is the wrong compression component: high.
- A static ModelOpt W4A16 Qwen3.5-9B checkpoint can be attempted on GB10: high.
- That checkpoint improves 65K-prefill throughput over the proven FP8 server:
  unknown until measured.
- W4A4 NVFP4 can replace FP8 in the live BF16-DTensor RL path with near-zero
  policy mismatch: unsupported by current evidence.

## What The Article Actually Tests

The [humans& NVFP4 RL article](https://humansand.ai/blog/nvfp4-rl?v=3)
uses this experimental contract unless a figure says otherwise:

- Qwen3-30B-A3B, a mixture-of-experts model.
- DAPO-math-17k at 8,192 tokens.
- NVFP4 forward on MoE experts only; non-MoE components remain BF16.
- Per-token activation global scaling, block-16 FP8 E4M3 scales, and E2M1 FP4
  values.
- BF16 backward using dequantized copies of the exact quantized forward
  operands, for both weights and activations.
- Adaptive four-over-six selection for weights and activations.
- Approximately the final 15% of layers and shared experts retained in BF16.
- A TransformerEngine + FlashInfer + SGLang + Megatron-LM + Miles runtime.

Each qualifier matters. The experiment is neither dense Qwen3.5-9B nor 131K,
and the article's training stabilization is not ordinary post-training
quantization.

### Embedded Figure Data

The article ships its plot data as JavaScript rather than only raster figures.
The relevant values are:

| Metric | BF16 reference | Final NVFP4 recipe | Reading |
|---|---:|---:|---|
| Mean raw reward | 0.59879 | 0.59594 | 0.48% lower for this MoE/math run |
| Mean rollout time | 109.23 s | 83.69 s | 23.4% lower |
| Illustrated model memory | 61.1 GB | 26.3 GB | 57.0% lower |
| Train/rollout absolute logprob difference | 0.00959 | 0.03123 | 3.26x BF16 |
| GLM-5.1 judge mean | FP8 54.3 +/- 3.4 | FP4 54.5 +/- 3.5 | overlapping means |
| Per-episode FP8/FP4 judge correlation | - | 0.924 | correlation, not equality |

The article's reward claim is credible within its experiment. It is not a
losslessness claim for token probabilities, another architecture, long-context
scientific tasks, or policy-gradient importance ratios.

### The Three Stabilizers

1. **Dequantized backward.** A BF16 backward over the original unquantized
   weights differentiates a different function from the NVFP4 forward. The
   article instead uses BF16-dequantized values of the exact forward tensors.
   This improves gradient alignment but can increase variance, which Adam
   controls better than SGD. The implementation is in
   [TransformerEngine PR 2644](https://github.com/NVIDIA/TransformerEngine/pull/2644)
   with memory work in
   [PR 2865](https://github.com/NVIDIA/TransformerEngine/pull/2865).
2. **Four-over-six.** Each block selects whether the representable maximum is 4
   or 6 based on reconstruction error. The optimized FP16 selection contract
   matches the reference choice more than 99.97% of the time and is reported
   about 2.8x faster than the scalar FP32 reference. Trainer and sampler must be
   bit-exact. See
   [TransformerEngine PR 2972](https://github.com/NVIDIA/TransformerEngine/pull/2972),
   [PR 3068](https://github.com/NVIDIA/TransformerEngine/pull/3068), and
   [FlashInfer PR 3264](https://github.com/flashinfer-ai/flashinfer/pull/3264).
3. **Selective high precision.** The final roughly 15% of layers and shared
   experts remain BF16. The article reports that partial combinations still
   exhibit gradient spikes; all three are used in the final five-run result.

ModelOpt main now contains a four-over-six PTQ preset, but it is weight-only.
Its tests explicitly reject `mtq.compress` for this state and require quantize
plus export. It therefore does not reproduce the article's weight-and-activation
four-over-six trainer/sampler contract. See
[ModelOpt commit e2c4d083](https://github.com/NVIDIA/Model-Optimizer/commit/e2c4d083d40976c38bf7efd9a05628ef5eed80a5).

## Component Audit

### NeMo AutoModel

Current AutoModel main was inspected at
[`acc8df3e`](https://github.com/NVIDIA-NeMo/Automodel/commit/acc8df3ed9f325d27b122ccbdc59ef3e08e8848b).

- It has a native dense Qwen3.5 implementation and a Qwen3.5-9B VLM fine-tune
  recipe.
- Its checked-in QAT surface uses TorchAO INT8-activation/INT4-weight and
  INT4-weight-only fake quantization for supervised fine-tuning.
- AutoModel main has no merged NVFP4 compressor/export path.
- An unmerged `hemil/nvfp4-moe` branch adds an exploratory
  `NVFP4BlockScaling` TransformerEngine recipe for Qwen3 MoE experts on GB200.
  It is neither dense Qwen3.5 support nor a serving checkpoint converter.

Conclusion: AutoModel is relevant to model support and training, not to the
requested NVFP4 deployment artifact.

### NVIDIA ModelOpt

Current ModelOpt main was inspected at
[`d69d5aab`](https://github.com/NVIDIA/Model-Optimizer/commit/d69d5aab8bcc7f905d39f96953621286bc2533be).

This repository contains the actual PTQ and unified-Hugging-Face export path.
It now ships a dense `qwen3_5` W4A16 recipe:

- NVFP4 static MSE-calibrated weights for MLP projections and `lm_head`.
- FP8 weights and activations for softmax-attention projections and the large
  Gated DeltaNet projections.
- BF16 for excluded/sensitive linears, convolution, visual tower, and MTP.
- FP8 KV-cache cast in the stock recipe.

The exact source is the
[Qwen3.5 W4A16-MSE recipe](https://github.com/NVIDIA/Model-Optimizer/blob/d69d5aab8bcc7f905d39f96953621286bc2533be/modelopt_recipes/huggingface/qwen3_5/ptq/w4a16_nvfp4_mse-fp8_attn-kv_fp8_cast.yaml)
and its
[quantizer rules](https://github.com/NVIDIA/Model-Optimizer/blob/d69d5aab8bcc7f905d39f96953621286bc2533be/modelopt_recipes/huggingface/qwen3_5/ptq/w4a16_nvfp4_mse-fp8_attn-kv_fp8_cast.quant_cfg.yaml).

The stock recipe violates this program's KV-cache contract because it embeds
FP8 KV-cache metadata. The admitted experiment must derive an otherwise
identical recipe with the KV-cache unit removed and keep vLLM
`kv_cache_dtype=auto`.

The local checkpoint header contains 9.653B parameters:

| Component | Parameters | BF16 storage |
|---|---:|---:|
| Text MLP | 4.983B | 9.28 GiB |
| Text attention and GDN | 2.146B | 4.00 GiB |
| Embeddings and LM head | 2.034B | 3.79 GiB |
| Vision tower | 0.456B | 0.85 GiB |

This explains the likely tradeoff. MLP-only NVFP4 saves substantial MLP memory,
but a selective W4A16 artifact can still be similar to, or larger than, an
all-linear FP8 rollout artifact. It must win on measured prefill rather than on
the four-bit label.

### vLLM 0.24 On GB10

The retained `vllm/vllm-openai:v0.24.0-aarch64` image was probed live on
`spark-f7e2`:

- vLLM 0.24.0, PyTorch 2.11.0+cu130, CUDA 13.0.
- Device capability `(12, 1)`.
- `modelopt_fp4`, `modelopt_mixed`, and W4A16 ModelOpt classes import.
- W4A4 kernel selection returns `FlashInferCutlassNvFp4LinearKernel`.
- W4A16 kernel selection returns `MarlinNvFp4LinearKernel`.

The source confirms that a serialized ModelOpt `NVFP4` checkpoint uses W4A4
with activation quantization, while `W4A16_NVFP4` pins the Marlin weight-only
path. See the
[vLLM 0.24 ModelOpt implementation](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/layers/quantization/modelopt.py).

Kernel selection is not an end-to-end serving result. W4A16 Marlin reduces
weight bandwidth but does not exercise the same native W4A4 compute path as the
article. The real 65K-prefill workload may improve or regress.

### SGLang Online NVFP4

[SGLang PR 26083](https://github.com/sgl-project/sglang/pull/26083) merged on
2026-06-10. Its stated initial scope is Blackwell-only, MoE-only, no dense
linear quantization, and FlashInfer TRT-LLM MoE backends only. It quantizes BF16,
FP16, or FP8 expert weights during load and applies runtime per-token activation
scaling.

Conclusion: `--quantization nvfp4_online` implements the article's MoE serving
path, but it cannot compress dense Qwen3.5-9B.

### NeMo RL QARL

Latest NeMo RL main was audited through
[`6ab16882`](https://github.com/NVIDIA-NeMo/RL/commit/6ab16882addfcfc778f6f6bee21454edda9487ff).
The commit after the Stage 3 image pin adds MFU reporting only; it does not
invalidate the `718ee57f` Stage 3 build.

NeMo RL now has two QARL rollout modes:

- fake-quant rollout using folded full-precision weights;
- real W4A16 rollout that streams packed ModelOpt NVFP4 tensors and scales into
  vLLM on every refit.

The [QARL guide](https://github.com/NVIDIA-NeMo/RL/blob/6ab16882addfcfc778f6f6bee21454edda9487ff/docs/guides/quantization-aware-rl.md)
marks W4A4 GRPO as a known convergence problem and W4A16 real-quant rollout as
the validated GRPO route. That validation is Qwen3-8B on two nodes by eight
GPUs, roughly 30K context, not Qwen3.5-9B on TP1 GB10 at 131K.

The implementation also rejects DTensor quantization. Real-quant weight refit
requires `MegatronQuantPolicyWorker`. Training keeps full-precision master
weights/backward through an STE, but the policy forward is quantization-aware.
That is not the current ordinary BF16-DTensor Stage 3 topology.

Consequences:

- A static ModelOpt checkpoint proves frozen serving only.
- After the first BF16 policy update, a dense NVFP4 rollout model needs a
  quantize/export/transfer/repack/refit mechanism.
- Current NeMo provides that mechanism through Megatron QARL, not through the
  selected BF16 DTensor trainer.
- Moving to it is a separately gated Stage 6 architecture change, not a launch
  flag optimization.

## Admitted Experiment Ladder

Stage 3 remains unchanged: BF16 DTensor training, FP8 vLLM rollout weights,
`kv_cache_dtype=auto`, 131K context, text-only serving, and exact logprobs.

### Stage 6A: Frozen Serving

1. Derive the ModelOpt dense-Qwen3.5 W4A16-MSE recipe without FP8 KV cache.
2. Quantize only the language path; retain the outer checkpoint metadata needed
   by Qwen3.5 but exclude the vision tower from serving.
3. Export a unified HF checkpoint and record base hash, recipe hash, ModelOpt
   hash, tokenizer hash, and quantization metadata.
4. Load it in vLLM 0.24 with `--language-model-only`,
   `--quantization modelopt`, 131,072 context, and KV cache `auto`.
5. Run the exact FP8 comparison workload: 65K prompt, k=8 identical prefixes,
   1,536-token cap, cold plus warm cache, token IDs, selected-token logprobs,
   memory, latency, and clean teardown.

### Stage 6B: Probability And Task Parity

Reject before training unless all artifacts are present:

- teacher-forced BF16/FP8/W4A16 token-logprob comparison;
- first-token and selected-token KL by prompt-length bucket;
- generated-token disagreement rate under matched sampling seeds where
  deterministic execution permits it;
- rollout-logprob versus BF16 trainer-logprob absolute difference;
- importance-ratio median, tails, clipping rate, NaN/Inf rate;
- scientific verifier pass rate and reward distribution;
- prefix-cache hit rate and cache invalidation after a weight-version change.

No universal `near-zero` threshold is assumed. The go/no-go report must set the
allowed verifier delta and importance-ratio tail from the FP8 control before
reading the W4A16 result.

### Stage 6C: Live Refit

Only after 6A and 6B pass:

1. Port Qwen3.5-9B into the Megatron QARL W4A16 real-quant path.
2. Keep BF16 master weights and backward, but label the policy forward as
   quantization-aware rather than ordinary BF16.
3. Measure quantize/export, RoCE transfer, vLLM repack, drain/resume, and cache
   invalidation time on every weight sync.
4. Require the same `weight_version` and checkpoint/recipe hashes on every
   trajectory.
5. Run one reward-bearing step before any throughput or convergence claim.

Native W4A4 plus article-faithful activation four-over-six and dequantized
backward is deferred. It currently requires a dense-model port across trainer,
export/refit, and sampler kernels and has no Qwen3.5-9B/GB10/131K proof.

## Go/No-Go

Current decision: **go for a frozen W4A16 ModelOpt serving ablation after Stage
3; no-go for replacing FP8 rollouts or changing the trainer today.**

W4A16 advances only if it:

- starts at 131K with `kv_cache_dtype=auto` and text-only allocation;
- returns complete token IDs and logprobs;
- improves the real 65K x k=8 prefill-dominated workload or provides material
  headroom without throughput regression;
- stays inside the FP8 control's predeclared probability and verifier bounds;
- survives repeated requests and clean teardown.

The live RL path advances only if packed refit is proven on Qwen3.5-9B and the
Megatron QARL change is explicitly accepted as the Stage 6 trainer topology.
