# Stage 0 Recovery And Inventory

Observed: 2026-07-10 15:23-15:27 UTC
Status: passed

## Host Safety

| Host | Memory available | Disk available | GPU | Active containers | Lab ports | OOM journal |
|---|---:|---:|---|---|---|---|
| `spark-f7e2` | 117 GiB | 823 GiB | GB10, 0% | none | clear | none in current or previous boot |
| `spark-cfd0` | 117 GiB | 2.5 TiB | GB10, 0% | none | clear | none in current or previous boot |

Both hosts were reachable through strict existing SSH trust. Containerized CUDA
was available on both. Direct-fabric carrier, addresses, peer ping, and
bidirectional authenticated SSH passed. No unrelated workload required
interruption.

## Model Inventory

Exact path on both hosts:

```text
/home/jarrodbarnes/models/Qwen/Qwen3.5-9B
```

Both copies report `19,329,395,397` bytes and identical identity hashes:

```text
config.json                  d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05
tokenizer_config.json        316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8
model.safetensors.index.json 26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5
```

The checkpoint has four safetensors shards; vLLM reports a checkpoint size of
17.98 GiB and BF16 model-load memory of 17.66 GiB.

## Image Selection

Selected standalone serving image on both hosts:

```text
vllm/vllm-openai:v0.24.0-aarch64
sha256:730a973ed3917e4eb96cb5c3a195272fe2712d291d86001ceba2f91053f41d4e
```

This exact image previously resolved `Qwen3_5ForConditionalGeneration`, served
the model, and exposes the source-backed `--language-model-only` control needed
for the new text-only baseline.

The pinned NeMo image also exists on both hosts:

```text
local/nemo-rl-main-cu132:c236061b
sha256:b0f8dd966dec1efd0d15b9d6732c611ad4f1f1499a6ec29e37b1eb86259e871d
```

## Prior Failure Correction

The retained Docker evidence contradicts the premise that both previous 9B
servers OOM-killed:

| Host | Prior config | Docker OOMKilled | Successful requests | Error patterns | Exit |
|---|---|---:|---:|---|---|
| `spark-f7e2` | 64K, FP8 KV, util 0.70, seqs 32 | false | 1,068 | none | 255 at host reboot |
| `spark-cfd0` | 64K, FP8 KV, util 0.88, seqs 64 | false | 258 | none | 255 at host reboot |

These rows are historical serving evidence, not accepted RL baselines: they used
FP8 KV cache, high memory utilization, and no text-only flag. Exit 255 followed
the host reboot window and is not an OOM classification. The program still
restarts conservatively at 4K/one sequence because the required `kv_cache_dtype=auto`,
text-only, token-ID, logprob, and clean-stop contract has not yet been proven.

## Exit Criteria

- Both Sparks reachable: passed.
- No accidental job disruption risk: passed.
- Exact identical model path on both hosts: passed.
- Candidate vLLM image selected: passed.
- Disk, memory, containers, ports, and OOM evidence inventoried: passed.

## Post-Run Recovery Audit

A later direct kernel-journal audit found NVIDIA driver allocation failures
that the original status-script pattern missed. Consequently, the Stage 0
`OOM journal: none` cells above are retracted:

| Host | Time | Correlated run | Decision |
|---|---|---|---|
| `spark-cfd0` | 14:36:39-14:36:43 | initial latest-main wiring probe | already discarded |
| `spark-cfd0` | 15:41:48-15:41:50 | latest-main `lora32-text-refit-r4` | already discarded |
| `spark-cfd0` | 16:26:02 | fork-stable FP8 integration `infra-r4` | discard despite exit 0 |
| `spark-f7e2` | 16:28:29 | fork-stable FP8 integration `infra-r4` | discard despite exit 0 |

The signature was
`NVRM: ... Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal`.
Docker did not report `OOMKilled=true`, and neither host went offline. The
health checker now matches this driver-level signature.

The selected optimized FP8 run `r6` ran later, from 17:05:36 through
17:11:02, with no new NVIDIA memory event. Final recovery showed no containers,
no lab listeners, and approximately 115 GiB available on each host. Historical
current-boot events remain visible and must not be described as absent.
