# Stage 2 Context And Grouped-Rollout Optimization

Status: FP8 production profile passed and selected
Host: `spark-f7e2`
Other Spark: `spark-cfd0` remained idle and untouched

## Contract Fit

The BF16, text-only, `kv_cache_dtype=auto`, 0.35-memory-utilization server
started at every safety-gate context from 4,096 through 131,072 tokens. At the
hard 131,072-token contract point, a raw completion request containing exactly
65,000 prompt token IDs returned HTTP 200, 16 generated token IDs, and 16 aligned
selected-token logprobs in 20.389147 seconds. vLLM allocated 670,963 KV-cache
tokens and reported 5.12x theoretical full-context concurrency. The container
stopped with exit 0 and `OOMKilled=false`.

## Prefix Cache Proof

Configuration held fixed at 131,072 context, one sequence, 4,096 batched tokens,
BF16 weights, auto KV dtype, and aligned hybrid prefix caching.

- A deterministic 2,048-token cold/warm pair produced identical generated token
  IDs and maximum absolute selected-logprob delta 0.
- A sequential k=8 group over an identical 65,000-token prompt completed in
  30.101513 seconds versus 20.483630 seconds for the cold member alone.
- Seven warm requests had p50 1.373424 seconds and p95 1.375048 seconds.
- vLLM recorded 456,192 prefix hits over 524,096 queries in the combined parity
  and group test. The isolated group tests below consistently recorded 454,608
  hits over 520,000 queries, or 87.4%: within 392 tokens of the theoretical 7/8
  ceiling for eight equal 65K prompts.
- The measured sequential amortization was 5.444x relative to eight cold
  requests. This is an end-to-end wallclock ratio, not a claim of exact 8x cache
  speedup.

## Concurrency Ladder

All rows use 131,072 context, 4,096 batched tokens, a 65K-token identical prefix,
256 generated tokens per rollout, prefix caching in `align` mode, and one cold
request followed by the remaining requests at the candidate concurrency.

| Max sequences | Cold seconds | k=8 wall seconds | End-to-end generated tok/s | Decision |
|---:|---:|---:|---:|---|
| 1 | 20.483630 | 30.101513 | 4.252 | safe, too little decode concurrency |
| 2 | 57.735758 | 145.357229 | 14.089 | safe |
| 4 | 56.437370 | 102.167061 | 20.046 | safe |
| 8 | 56.397342 | 80.998668 | 25.284 | keep |

The one-sequence row used a shorter parity/group run shape and is not directly
comparable to the 256-token concurrent rows. Sequence concurrency 8 is the
retained grouped-rollout shape: it improved end-to-end generated-token rate 26%
over sequence concurrency 4 and 79% over sequence concurrency 2.

## Chunked-Prefill Token Budget

At fixed sequence concurrency 8:

| Max batched tokens | KV-cache tokens | Cold seconds | k=8 wall seconds | End-to-end generated tok/s | Decision |
|---:|---:|---:|---:|---:|---|
| 4,096 | 670,963 | 56.397342 | 80.998668 | 25.284 | candidate |
| 8,192 | 667,696 | 57.004330 | 81.654254 | 25.081 | safe; no gain |
| 16,384 | 657,416 | 56.654064 | 80.825753 | 25.338 | candidate; difference under 1% |
| 32,768 | 591,623 | 56.487171 | 81.100202 | 25.253 | safe; no gain and less KV headroom |
| 65,536 | 459,009 | 55.528174 | 80.057643 | 25.582 | provisional throughput leader; full-decode gate open |

The first five candidates differ by at most 2%. The 65,536-token point reduced
cold time by 1.9% and group wallclock by 1.0% relative to 16,384, while reducing
KV capacity by 30%. That is enough capacity for the shared-prefix workload and a
full 131K request.

The full decode contract then passed at 65,536 batched tokens: all eight 65K
prompt requests returned exactly 1,536 generated token IDs and aligned logprobs.
The uncached request took 162.205278 seconds; the seven concurrent cached
requests took at most 144.619040 seconds; total group wallclock was 306.874395
seconds for 12,288 generated tokens, or 40.042 generated tokens per end-to-end
group second. vLLM's live decode rate held at approximately 74.2-74.9 tokens/s.
The cache hit rate remained 87.4%, logs contained no error pattern, and the
container stopped cleanly with `OOMKilled=false`.

## Resolved Runtime

The 16,384-token launch log confirms:

- asynchronous scheduling enabled in both API and engine processes;
- vLLM compile mode with Inductor;
- `FULL_AND_PIECEWISE` CUDA graphs, capture sizes through 16, and 0.67 GiB
  estimated graph memory;
- Triton/FLA GDN prefill kernel;
- FlashAttention full-attention backend;
- aligned Mamba/attention pages of 528 tokens;
- chunked prefill and experimental aligned hybrid prefix caching enabled;
- no speculative configuration.

The first long request triggered several Triton JIT compilations. A production
profile should warm the retained long-prefill/decode shapes before accepting
traffic; cold-start latency is reported separately from steady grouped rollout
throughput.

## Pre-NeMo Research Check-In

The background official-source review returned a conditional GO for the exact
NeMo/Polar collector-surface smoke, but a NO-GO for reward-bearing training until:

1. the exact collector endpoint proves generated token IDs and logprobs;
2. the real weight-sync lifecycle drains trajectories, invalidates prefix/KV
   cache, applies weights, and resumes without a mixed-policy trajectory;
3. the 1,536-token decode contract passes; and
4. three distinct 65K-prefix records each show one miss plus seven hits without
   preemption, OOM, or cross-record contamination.

The review judged primer-versus-simultaneous submission deferrable because the
live 87.4% cache-hit result is already effectively the theoretical ceiling.

A direct follow-up inspection corrected the revision-risk scope. vLLM PR #47384
fixed only `/v1/chat/completions/batch` in commit `586fc702`; Polar's
`Pipeline.run_batch` fans out independent ordinary Chat Completions sessions.
That bug is therefore not a blocker for the pinned image. The ordinary-chat
surface still requires live end-to-end token-ID/logprob validation through
Polar, but a current-main upgrade is not required if it passes.

## Multi-Record And Ordinary-Chat Gate

Run directory:
`/home/jarrodbarnes/qwen35-serving/qwen35-text-131k-auto-035-prefix-seq8-bt65536-multirecord-r2-20260710-1645z`

The exact vLLM ordinary Chat Completions request shape used by Polar returned
HTTP 200. Its 16 prompt token IDs exactly matched Qwen3.5's local chat-template
tokenization, and its 32 output token IDs aligned one-for-one with 32 selected
token logprobs.

Three distinct 65,000-token prefix records were then submitted as k=8 groups in
one server lifetime. Each record independently recorded 520,000 queried prefix
tokens, 454,608 hits, an 87.4246% hit rate, eight HTTP 200 responses, and aligned
token-ID/logprob lengths. Cold times were 19.178559, 18.600135, and 18.563164
seconds; warm maxima were 2.159466, 1.815289, and 1.795685 seconds. The final
preemption counter was zero. No OOM/error pattern appeared and cleanup exited 0
with the port clear.

## Decision

Keep the 65,536-batched-token profile. Its throughput advantage over 16,384 is
small, but it is the measured leader for the 65K prefill workload, passes the
full 1,536-token decode contract, preserves 459K KV-token capacity, and survives
three distinct cache populations without preemption. Stage 3 may now begin with
the exact Polar ordinary-chat surface as its first live gate. Reward-bearing
training remains blocked until actual NeMo weight sync proves cache invalidation
and no trajectory crosses a policy version.

## FP8 Weight Gate

The 131K throughput matrix above used BF16 model weights. It must not be cited
as the final FP8 inference profile.

Standalone vLLM 0.24 online per-tensor FP8 passed the conservative 4K gate:
10/10 HTTP 200 responses, coherent text, 16 generated token IDs aligned with 16
finite selected-token logprobs per response, 627,060 KV tokens, no OOM, and
clean memory recovery from 72 GiB available under load to 116 GiB after stop.
Run directory:
`/home/jarrodbarnes/qwen35-serving/qwen35-fp8-4k-20260710-1900z-r3`.

At that checkpoint, FP8 at 131K, k=8, and the 65K-prefix/1,536-decode workload
was still unproven. The ladder below closes that gate and replaces BF16.

## FP8 Production Result

The missing ladder now passes. Online per-tensor FP8 at 131,072 context returned
aligned finite selected-token logprobs for a raw 65,000-token request and
allocated 874,333 KV tokens at the one-sequence fit gate.

At one sequence with aligned prefix caching, the cold 65K request took
33.333429 seconds and the identical warm request took 0.896092 seconds. Output
token IDs were identical and maximum absolute selected-logprob delta was zero.

The k=8, 256-token comparison selected the 4,096-token prefill chunk over
16,384: 32.061 versus 29.365 generated tokens/s end-to-end. The actual workload
gate then passed at 65,536 batched tokens:

| Metric | FP8 | BF16 comparator |
|---|---:|---:|
| Context | 131,072 | 131,072 |
| Shared prompt | 65,000 x 8 | 65,000 x 8 |
| Generated tokens | 1,536 x 8 | 1,536 x 8 |
| Cold request | 121.900856s | 162.205278s |
| Warm max | 102.527895s | 144.619040s |
| Group wall | 224.445773s | 306.874395s |
| End-to-end generated tok/s | 54.748 | 40.042 |
| Prefix-hit rate | 87.4246% | 87.4246% |

FP8 is 36.7% faster on the production-shaped serving workload. It retains
691,340 KV tokens, reports 5.27x theoretical full-context concurrency, returns
12,288 aligned token IDs/logprobs, and stops without OOM or leaked resources.
This replaces the BF16 command as the serving selection.
