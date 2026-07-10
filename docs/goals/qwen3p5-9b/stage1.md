# Stage 1 Minimal Serving Baseline

Status: passed
Host: `spark-f7e2`
Run directory:
`/home/jarrodbarnes/qwen35-serving/qwen35-text-4k-auto-035-20260710-1530z`

## Configuration

| Field | Value |
|---|---|
| Image | `vllm/vllm-openai:v0.24.0-aarch64` |
| Model | `/home/jarrodbarnes/models/Qwen/Qwen3.5-9B` |
| Context | 4,096 |
| Max sequences | 1 |
| Max batched tokens | 4,096 |
| Memory utilization | 0.35 |
| KV cache | `auto` |
| Weights | BF16, unquantized |
| Execution | compiled/CUDA graphs; not eager |
| Modality | `--language-model-only` |

## Runtime Evidence

- vLLM logged `running in text-only mode` in both API and engine processes.
- Model residency was 16.8 GiB, down from 17.66 GiB in the historical
  multimodal launch.
- No multimodal encoder-cache allocation or multimodal warmup appeared.
- Available KV cache was 20.71 GiB / 478,487 tokens, with a reported theoretical
  116.82x maximum concurrency at 4,096 tokens.
- The first post-start request returned HTTP 200 with 32 generated token IDs,
  32 aligned token logprobs, and top-5 alternatives. Its 18.52 second latency
  included first-request compilation/JIT effects.
- The warm repeat loop passed 10/10 HTTP requests. Every response carried 16
  token IDs and 16 logprob entries. Warm p50 was 1.187398 seconds, p95 was
  1.190621 seconds, and end-to-end completion throughput was 13.466 tokens/s.
- An image-bearing request was rejected with HTTP 400: `At most 0 image(s) may
  be provided in one prompt.`
- Container memory remained 17.97-17.98 GiB after load, generation, and idle.
  Host available memory was stable at roughly 74.56 GiB while live.
- Docker reported `OOMKilled=false`; logs contained no OOM, CUDA error,
  traceback, or engine-failure pattern.
- Graceful stop exited 0, cleared port 8001, and restored available host memory
  to roughly 117.17 GiB.

## Decision

Keep. Stage 1 is a valid recovery baseline. Advance to the context ladder with
the same image, model, text-only control, BF16 weights, `kv_cache_dtype=auto`,
memory utilization 0.35, one sequence, and one context change per run.
