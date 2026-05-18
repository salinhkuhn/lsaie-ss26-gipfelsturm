# Wall-Clock MoE Scaling Benchmark

## Benchmark Goal

This benchmark compares dense, quantized dense, MoE, and quantized MoE training under strict post-startup wall-clock budgets on 8 Clariden nodes. The goal is not to make MoE look good; it is to answer whether any MoE recipe reaches lower final validation loss than strong dense baselines when all runs get the same hardware and the same benchmark clock.

The active matrix is 9 configs x 3 scales = 27 jobs. The scheduler can also add the optional shared-expert MoE config for 30 jobs.

## Fairness Principle

The main fairness criterion is:

> Same 8-node hardware, same dataset/tokenizer/eval, same post-startup wall-clock budget, and a guaranteed final eval.

A MoE model has total parameters and active parameters. A dense model uses all parameters per token, while a MoE model only activates a subset. Therefore, same total-parameter comparison is not the main fairness criterion for this benchmark. The main criterion is best eval loss under the same hardware and post-startup wall-clock budget.

Every job uses 8 nodes, 4 GPUs per node, 32 GPUs total. Dense and MoE sizes are allowed to differ because MoE capacity is sparse per token.

## Scales

| Scale | Benchmark Clock | Final Buffer | Effective Training | SLURM Limit |
|---|---:|---:|---:|---:|
| `small` | 30 min | 5 min | 25 min | 45 min |
| `medium` | 60 min | 10 min | 50 min | 85 min |
| `large` | 120 min | 10 min | 110 min | 160 min |

The benchmark clock starts immediately before the first real training step, after startup, distributed init, model construction, and dataloader setup. Training stops at `benchmark_wallclock_seconds - final_buffer_seconds`, then the job saves a final checkpoint and runs final validation.

## Configs

| Config | Architecture | Precision | Parallelism | Purpose |
|---|---|---|---|---|
| `dense_bf16_fsdp` | Dense | BF16 | DP plus Megatron distributed optimizer, TP for larger scales | Strong non-quantized dense baseline. |
| `dense_fp8_fsdp` | Dense | FP8 | Same dense size as BF16 baseline | Tests whether FP8 improves wall-clock efficiency at the same dense size. |
| `dense_fp8_maxfit` | Dense | FP8 | Larger dense preset where possible | Tests whether quantization is best spent on fitting a larger dense model. |
| `moe_local_bf16` | MoE | BF16 | Local experts, no dedicated EP | Tests the simplest low-communication MoE. |
| `moe_ep_bf16` | MoE | BF16 | Expert parallelism plus TP/DP | Main scalable BF16 MoE baseline. |
| `moe_ep_fp8` | MoE | FP8 experts | Expert parallelism plus TP/DP | Tests whether FP8 makes scalable MoE practical. Router stays FP32. |
| `moe_local_fp8` | MoE | FP8 experts | Local experts | Separates quantization benefit from expert-parallel benefit. |
| `dense_bestsize_bf16` | Dense | BF16 | Best expected dense size per clock | Challenges MoE against the dense size most likely to win by seeing enough tokens. |
| `moe_bestcapacity_bf16` | MoE | BF16 | Best expected MoE capacity per clock | Strongest non-quantized MoE-vs-best-dense comparison. |
| `moe_sharedexpert_bf16` | MoE | BF16 | Optional EP + shared expert | Optional 10th config to test a stabilizing shared expert path. |

The exact per-scale sizes and flags are generated in `configs/generated/*.yaml`. Those files are the source of truth for each run.

## Files And HF Locations

The Hugging Face project/repo name used in run metadata is:

```text
wallclock-moe-scaling-benchmark
```

Each run records its intended HF checkpoint as the exact run name in the catalog below. For example, the run

```text
large-120m-moe-fp8-ep-32e-top2-88btotal-8bactive-run001
```

maps to the intended HF checkpoint/artifact name:

```text
wallclock-moe-scaling-benchmark/large-120m-moe-fp8-ep-32e-top2-88btotal-8bactive-run001
```

Local files are organized as follows:

| File type | Location |
|---|---|
| Generated YAML configs | `wallclock-moe-scaling-benchmark/configs/generated/<run_name>.yaml` |
| Result JSON files | `wallclock-moe-scaling-benchmark/results/<run_name>.json` |
| Capacity add-on result JSON files | `wallclock-moe-scaling-benchmark/results/addons/<run_name>.json` |
| SLURM scripts and logs | `logs/gipfel-*.sbatch` and `logs/gipfel-*.log` |
| W&B project | `wallclock-moe-scaling-benchmark` |
| Checkpoint directory pattern | `/iopsstor/scratch/cscs/$USER/gipfelsturm/wallclock-moe-scaling-benchmark/<suite>/checkpoints/<run_name>` |

The Megatron patch writes the HF project and checkpoint name into the result JSON. If HF upload is enabled outside this repo, those names are the stable object names to use.

## Model Presets

Dense and MoE runs share the same transformer backbone presets. MoE replaces or augments dense FFNs with routed expert FFNs according to the run-specific MoE settings.

| Preset | Layers | Hidden | Dense FFN | Attention heads | Query groups | Default MBS |
|---|---:|---:|---:|---:|---:|---:|
| `350m` | 24 | 1024 | 2816 | 16 | 4 | 8 |
| `760m` | 24 | 1536 | 4096 | 16 | 4 | 4 |
| `1.5b` | 48 | 1600 | 4352 | 20 | 4 | 4 |
| `3b` | 32 | 3072 | 8192 | 24 | 8 | 4 |
| `8b` | 32 | 4096 | 14336 | 32 | 8 | 2 |

All runs use GPT-style decoder-only architecture, RoPE positional embeddings, RMSNorm, SwiGLU, untied input/output embeddings, sequence length 4096, BF16 base precision, Transformer Engine, and FlashAttention.

## MoE Implementation Details

For MoE runs, each MoE layer has a router projection from hidden size to `num_experts`. The router scores are computed in FP32, use softmax routing, use pre-softmax routing where configured, and include auxiliary load-balancing loss. The scheduled MoE configs use:

| Router setting | Value |
|---|---|
| Router precision | FP32 |
| Score function | softmax |
| Load balancing | auxiliary loss |
| Aux loss coeff | `1e-2` |
| Router z-loss coeff | `1e-3` |
| Input jitter | `1e-2` |
| Token dispatcher | all-to-all |
| Expert tensor parallelism | `ETP1` |
| Grouped GEMM | enabled |

Expert size is controlled by:

```text
moe_expert_ffn_hidden_size = dense_ffn_hidden_size / MOE_FFN_DIVISOR
```

Top-1 runs usually use `MOE_FFN_DIVISOR=1`, so each selected expert has the full dense FFN width. Top-2 runs usually use `MOE_FFN_DIVISOR=2`, so two half-width experts are active per token and the active FFN compute stays closer to the dense baseline.

Expert placement:

- `EP1` local MoE: every data-parallel/model-parallel replica owns all experts locally. There is no cross-rank expert sharding, so this avoids the main expert-parallel all-to-all cost.
- `EP4`: experts are sharded across 4 expert-parallel ranks. Each EP rank owns `num_experts / 4` experts.
- `EP8`: experts are sharded across 8 expert-parallel ranks. Each EP rank owns `num_experts / 8` experts.

The data-parallel degree is:

```text
DP = 32 / (TP * PP * EP)
```

## Run Catalog

This section names every run in the 27-job matrix. Each run uses the same dataset, tokenizer, validation setup, 8 nodes, 4 GPUs per node, FlashAttention/TE attention, Megatron distributed optimizer, W&B project `wallclock-moe-scaling-benchmark`, and the final checkpoint/final eval patch.

### Small Runs

Small runs use a 30-minute benchmark clock with a 5-minute final buffer, so they train for about 25 minutes after startup before final checkpoint/eval.

| Run name | Config | What it trains | Why it exists |
|---|---|---|---|
| `small-30m-dense-bf16-fsdp-760m-run001` | `dense_bf16_fsdp` | Dense 760M BF16 GPT, `TP1 + DP32`, active params equal total params. | Main small-scale dense baseline. Every small MoE should be compared against this first. |
| `small-30m-dense-fp8-fsdp-760m-run001` | `dense_fp8_fsdp` | Dense 760M GPT with FP8 training/parameter gather where supported, `TP1 + DP32`. | Tests whether quantization improves throughput enough to beat the same-size BF16 dense run. |
| `small-30m-dense-fp8-fsdp-1p5b-run001` | `dense_fp8_maxfit` | Dense 1.5B FP8 GPT, `TP1 + DP32`. | Tests the “use quantization to fit a larger dense model” baseline. MoE should beat this too, not only the 760M baseline. |
| `small-30m-moe-bf16-local-4e-top1-1.1btotal-350mactive-run001` | `moe_local_bf16` | Local BF16 MoE on a 350M-size backbone, 4 experts, top-1 routing, no expert parallelism, about 1.1B total and 350M active params. | Low-communication MoE sanity check. It asks whether MoE helps at small scale without all-to-all routing overhead. |
| `small-30m-moe-bf16-ep-8e-top2-2.8btotal-760mactive-run001` | `moe_ep_bf16` | Expert-parallel BF16 MoE on a 760M-active backbone, 8 experts, top-2 routing, `EP8 + DP4`, about 2.8B total params. | Main small expert-parallel MoE baseline. It tests whether EP overhead is already worth it at short wall-clock. |
| `small-30m-moe-fp8-ep-8e-top2-5.5btotal-1.5bactive-run001` | `moe_ep_fp8` | Expert-parallel FP8-expert MoE on a 1.5B-active backbone, 8 experts, top-2 routing, `EP8 + DP4`, about 5.5B total params. | Aggressive small quantized MoE. It tests whether FP8 lets MoE use more active capacity within the same clock. |
| `small-30m-moe-fp8-local-4e-top1-1.8btotal-760mactive-run001` | `moe_local_fp8` | Local FP8-expert MoE on a 760M-active backbone, 4 experts, top-1 routing, about 1.8B total params. | Separates the value of FP8 experts from the cost/benefit of expert parallelism. |
| `small-30m-dense-bf16-bestfsdp-350m-run001` | `dense_bestsize_bf16` | Dense 350M BF16 GPT, `TP1 + DP32`. | Best-size dense candidate for short runs. A smaller dense model may win by seeing more tokens in 25 minutes. |
| `small-30m-moe-bf16-epbest-4e-top1-2.3btotal-760mactive-run001` | `moe_bestcapacity_bf16` | Best-capacity small BF16 MoE, 760M active params, 4 experts, top-1 routing, `EP4 + DP8`, about 2.3B total params. | The strongest small BF16 MoE guess: lower communication than top-2 EP8 while still adding sparse capacity. |

### Medium Runs

Medium runs use a 60-minute benchmark clock with a 10-minute final buffer, so they train for about 50 minutes after startup before final checkpoint/eval.

| Run name | Config | What it trains | Why it exists |
|---|---|---|---|
| `medium-60m-dense-bf16-tpfsdp-3b-run001` | `dense_bf16_fsdp` | Dense 3B BF16 GPT, `TP2 + DP16`, active params equal total params. | Main medium-scale dense BF16 baseline. |
| `medium-60m-dense-fp8-tpfsdp-3b-run001` | `dense_fp8_fsdp` | Dense 3B FP8 GPT, `TP2 + DP16`. | Same-size quantized dense baseline for medium scale. |
| `medium-60m-dense-fp8-tpfsdp-8b-run001` | `dense_fp8_maxfit` | Dense 8B FP8 GPT, `TP4 + DP8`. | Quantized max-fit dense baseline. It tests whether a larger dense model is a better use of FP8 than MoE. |
| `medium-60m-moe-bf16-local-4e-top1-3.6btotal-1.5bactive-run001` | `moe_local_bf16` | Local BF16 MoE on a 1.5B-active backbone, 4 experts, top-1 routing, MoE every 2 layers, about 3.6B total params. | Medium local-MoE baseline with modest sparse capacity and low communication overhead. |
| `medium-60m-moe-bf16-ep-16e-top2-17btotal-3bactive-run001` | `moe_ep_bf16` | Expert-parallel BF16 MoE, 3B active params, 16 experts, top-2 routing, `TP2 + EP8 + DP2`, about 17B total params. | Main medium expert-parallel BF16 MoE. This is where EP may start becoming worthwhile. |
| `medium-60m-moe-fp8-ep-32e-top2-33btotal-3bactive-run001` | `moe_ep_fp8` | Expert-parallel FP8-expert MoE, 3B active params, 32 experts, top-2 routing, `TP2 + EP8 + DP2`, about 33B total params. | Quantized scalable MoE: keeps active compute near the 3B dense baseline while greatly increasing total capacity. |
| `medium-60m-moe-fp8-local-4e-top1-7btotal-3bactive-run001` | `moe_local_fp8` | Local FP8-expert MoE, 3B active params, 4 experts, top-1 routing, `TP2 + DP16`, about 7B total params. | Tests whether a local quantized MoE can beat EP by avoiding all-to-all overhead. |
| `medium-60m-dense-bf16-bestfsdp-1p5b-run001` | `dense_bestsize_bf16` | Dense 1.5B BF16 GPT, `TP1 + DP32`. | Best-size dense candidate for medium runs. It may beat 3B/8B if throughput dominates. |
| `medium-60m-moe-bf16-epbest-32e-top2-33btotal-3bactive-run001` | `moe_bestcapacity_bf16` | Best-capacity medium BF16 MoE, 3B active params, 32 experts, top-2 routing, `TP2 + EP8 + DP2`, about 33B total params. | Strongest medium BF16 sparse-capacity test. It asks whether more experts beat the simpler 16-expert BF16 EP baseline. |

### Large Runs

Large runs use a 120-minute benchmark clock with a 10-minute final buffer, so they train for about 110 minutes after startup before final checkpoint/eval.

| Run name | Config | What it trains | Why it exists |
|---|---|---|---|
| `large-120m-dense-bf16-tpfsdp-8b-run001` | `dense_bf16_fsdp` | Dense 8B BF16 GPT, `TP4 + DP8`, active params equal total params. | Main large dense BF16 baseline. This is the key baseline for the 8B-active MoE comparisons. |
| `large-120m-dense-fp8-tpfsdp-8b-run001` | `dense_fp8_fsdp` | Dense 8B FP8 GPT, `TP4 + DP8`. | Same-size quantized dense large baseline. |
| `large-120m-dense-fp8-tp2maxfit-8b-run001` | `dense_fp8_maxfit` | Dense 8B FP8 GPT using `TP2 + DP16` to test a different throughput/memory tradeoff. | Dense FP8 max-fit/parallelism variant. It checks whether the best dense FP8 layout is not the same as the standard TP4 layout. |
| `large-120m-moe-bf16-local-4e-top1-7btotal-3bactive-run001` | `moe_local_bf16` | Local BF16 MoE on a 3B-active backbone, 4 experts, top-1 routing, `TP2 + DP16`, about 7B total params. | Large local-MoE baseline. It checks whether avoiding EP still wins even at larger scale. |
| `large-120m-moe-bf16-ep-16e-top2-45btotal-8bactive-run001` | `moe_ep_bf16` | Expert-parallel BF16 MoE, 8B active params, 16 experts, top-2 routing, `TP4 + EP8 + DP1`, about 45B total params. | Main inference-matched MoE-vs-8B-dense run. Active compute is about 8B, but total capacity is much larger than dense. |
| `large-120m-moe-fp8-ep-32e-top2-88btotal-8bactive-run001` | `moe_ep_fp8` | Expert-parallel FP8-expert MoE, 8B active params, 32 experts, top-2 routing, `TP4 + EP8 + DP1`, about 88B total params. | Strong quantized MoE capacity run. It tests the main MoE advantage: much larger total capacity at roughly 8B-active inference compute. |
| `large-120m-moe-fp8-local-4e-top1-7btotal-3bactive-run001` | `moe_local_fp8` | Local FP8-expert MoE, 3B active params, 4 experts, top-1 routing, `TP2 + DP16`, about 7B total params. | Quantized local-MoE control for large scale. It separates sparse capacity from expert-parallel communication. |
| `large-120m-dense-bf16-besttpfsdp-8b-run001` | `dense_bestsize_bf16` | Dense 8B BF16 GPT, `TP4 + DP8`, marked as the best expected dense size for the 2-hour budget. | Strongest dense BF16 comparison point for large scale. This may duplicate size with the standard dense baseline, but it is explicitly the best-size dense candidate. |
| `large-120m-moe-bf16-epbest-32e-top2-88btotal-8bactive-run001` | `moe_bestcapacity_bf16` | Best-capacity large BF16 MoE, 8B active params, 32 experts, top-2 routing, `TP4 + EP8 + DP1`, about 88B total params. | Strongest BF16 MoE-vs-best-dense comparison. It tests whether sparse capacity helps even without FP8 experts. |

### Capacity Add-On Runs

These runs are outside the main 27-job matrix. They are extra capacity-frontier probes to make the “MoE scales total capacity beyond dense at similar active compute” argument more explicit.

| Run name | What it trains | Why it exists |
|---|---|---|
| `large-120mtrain-dense-bf16-tp8pp2fsdp-32b-capacity001` | Approximate dense 32B BF16 GPT, `TP8 + PP2`, activation recompute, reduced global batch size. | Stress-test whether a much larger dense model can fit and produce a useful 64-GPU-hour result. This is not the primary fair baseline because it uses much more active compute per token than 8B dense. |
| `large-120mtrain-dense-fp8-tp8pp2fsdp-32b-capacity001` | Approximate dense 32B FP8 GPT, `TP8 + PP2`, activation recompute. | Tests whether FP8 makes a dense 32B capacity baseline feasible under the large clock. |
| `large-120mtrain-moe-bf16-ep4-12e-top2-34btotal-8bactive-capacity001` | BF16 MoE with about 34B total params and 8B active params, 12 experts, top-2 routing, `TP4 + EP4 + DP2`. | A direct “32B-ish total capacity but 8B-active inference compute” MoE comparison. |
| `large-120mtrain-moe-fp8-ep8-64e-top2-176btotal-8bactive-capacity001` | FP8-expert MoE with about 176B total params and 8B active params, 64 experts, top-2 routing, `TP4 + EP8 + DP1`. | Extreme sparse-capacity probe. It keeps active compute near the 8B dense baseline while scaling total expert capacity far beyond dense. |
| `large-120mtrain-dense-bf16-dp32-3b-parallelism001` | Dense 3B BF16 GPT with pure `DP32`, no tensor parallelism, no pipeline parallelism. | Dense parallelism ablation. It tests whether avoiding TP communication improves wall-clock loss for a 3B dense model. |
| `large-120mtrain-dense-bf16-dp32-8b-parallelism001` | Dense 8B BF16 GPT with pure `DP32`, no tensor parallelism, no pipeline parallelism, reduced microbatch if needed. | Dense parallelism ablation. It tests whether the 8B model can train as replicated DP32 and whether that is faster/slower than `TP4 + DP8`. |

## MoE Geometry By Run

This table expands every MoE run into concrete backbone, router, expert, and placement details.

| Run name | Backbone | Router | Experts | Placement | Active/total interpretation |
|---|---|---|---|---|---|
| `small-30m-moe-bf16-local-4e-top1-1.1btotal-350mactive-run001` | `350m`: L24, H1024, dense FFN 2816 | FP32 router `1024 -> 4`, softmax, top-1 | 4 BF16 experts, expert FFN 2816, MoE every layer | `TP1 + EP1 + DP32`; all 4 experts local and replicated in each DP replica | One full-width expert active per token, about 350M active / 1.1B total params |
| `small-30m-moe-bf16-ep-8e-top2-2.8btotal-760mactive-run001` | `760m`: L24, H1536, dense FFN 4096 | FP32 router `1536 -> 8`, softmax, top-2 | 8 BF16 experts, expert FFN 2048, MoE every layer | `TP1 + EP8 + DP4`; 1 expert per EP rank | Two half-width experts active per token, about 760M active / 2.8B total params |
| `small-30m-moe-fp8-ep-8e-top2-5.5btotal-1.5bactive-run001` | `1.5b`: L48, H1600, dense FFN 4352 | FP32 router `1600 -> 8`, softmax, top-2 | 8 FP8 experts, expert FFN 2176, MoE every layer | `TP1 + EP8 + DP4`; 1 expert per EP rank | Two half-width FP8 experts active per token, about 1.5B active / 5.5B total params |
| `small-30m-moe-fp8-local-4e-top1-1.8btotal-760mactive-run001` | `760m`: L24, H1536, dense FFN 4096 | FP32 router `1536 -> 4`, softmax, top-1 | 4 FP8 experts, expert FFN 4096, MoE every 2 layers | `TP1 + EP1 + DP32`; all experts local and replicated | One full-width FP8 expert active per token, about 760M active / 1.8B total params |
| `small-30m-moe-bf16-epbest-4e-top1-2.3btotal-760mactive-run001` | `760m`: L24, H1536, dense FFN 4096 | FP32 router `1536 -> 4`, softmax, top-1 | 4 BF16 experts, expert FFN 4096, MoE every layer | `TP1 + EP4 + DP8`; 1 expert per EP rank | One full-width expert active per token, about 760M active / 2.3B total params |
| `medium-60m-moe-bf16-local-4e-top1-3.6btotal-1.5bactive-run001` | `1.5b`: L48, H1600, dense FFN 4352 | FP32 router `1600 -> 4`, softmax, top-1 | 4 BF16 experts, expert FFN 4352, MoE every 2 layers | `TP1 + EP1 + DP32`; all experts local and replicated | One full-width expert active per token, about 1.5B active / 3.6B total params |
| `medium-60m-moe-bf16-ep-16e-top2-17btotal-3bactive-run001` | `3b`: L32, H3072, dense FFN 8192 | FP32 router `3072 -> 16`, softmax, top-2 | 16 BF16 experts, expert FFN 4096, MoE every layer | `TP2 + EP8 + DP2`; 2 experts per EP rank, expert weights tensor-parallel with shared model TP | Two half-width experts active per token, about 3B active / 17B total params |
| `medium-60m-moe-fp8-ep-32e-top2-33btotal-3bactive-run001` | `3b`: L32, H3072, dense FFN 8192 | FP32 router `3072 -> 32`, softmax, top-2 | 32 FP8 experts, expert FFN 4096, MoE every layer | `TP2 + EP8 + DP2`; 4 experts per EP rank | Two half-width FP8 experts active per token, about 3B active / 33B total params |
| `medium-60m-moe-fp8-local-4e-top1-7btotal-3bactive-run001` | `3b`: L32, H3072, dense FFN 8192 | FP32 router `3072 -> 4`, softmax, top-1 | 4 FP8 experts, expert FFN 8192, MoE every 2 layers | `TP2 + EP1 + DP16`; all experts local within each TP/DP replica | One full-width FP8 expert active per token, about 3B active / 7B total params |
| `medium-60m-moe-bf16-epbest-32e-top2-33btotal-3bactive-run001` | `3b`: L32, H3072, dense FFN 8192 | FP32 router `3072 -> 32`, softmax, top-2 | 32 BF16 experts, expert FFN 4096, MoE every layer | `TP2 + EP8 + DP2`; 4 experts per EP rank | Two half-width experts active per token, about 3B active / 33B total params |
| `large-120m-moe-bf16-local-4e-top1-7btotal-3bactive-run001` | `3b`: L32, H3072, dense FFN 8192 | FP32 router `3072 -> 4`, softmax, top-1 | 4 BF16 experts, expert FFN 8192, MoE every 2 layers | `TP2 + EP1 + DP16`; all experts local within each TP/DP replica | One full-width expert active per token, about 3B active / 7B total params |
| `large-120m-moe-bf16-ep-16e-top2-45btotal-8bactive-run001` | `8b`: L32, H4096, dense FFN 14336 | FP32 router `4096 -> 16`, softmax, top-2 | 16 BF16 experts, expert FFN 7168, MoE every layer | `TP4 + EP8 + DP1`; 2 experts per EP rank | Two half-width experts active per token, about 8B active / 45B total params |
| `large-120m-moe-fp8-ep-32e-top2-88btotal-8bactive-run001` | `8b`: L32, H4096, dense FFN 14336 | FP32 router `4096 -> 32`, softmax, top-2 | 32 FP8 experts, expert FFN 7168, MoE every layer | `TP4 + EP8 + DP1`; 4 experts per EP rank | Two half-width FP8 experts active per token, about 8B active / 88B total params |
| `large-120m-moe-fp8-local-4e-top1-7btotal-3bactive-run001` | `3b`: L32, H3072, dense FFN 8192 | FP32 router `3072 -> 4`, softmax, top-1 | 4 FP8 experts, expert FFN 8192, MoE every 2 layers | `TP2 + EP1 + DP16`; all experts local within each TP/DP replica | One full-width FP8 expert active per token, about 3B active / 7B total params |
| `large-120m-moe-bf16-epbest-32e-top2-88btotal-8bactive-run001` | `8b`: L32, H4096, dense FFN 14336 | FP32 router `4096 -> 32`, softmax, top-2 | 32 BF16 experts, expert FFN 7168, MoE every layer | `TP4 + EP8 + DP1`; 4 experts per EP rank | Two half-width experts active per token, about 8B active / 88B total params |
| `large-120mtrain-moe-bf16-ep4-12e-top2-34btotal-8bactive-capacity001` | `8b`: L32, H4096, dense FFN 14336 | FP32 router `4096 -> 12`, softmax, top-2 | 12 BF16 experts, expert FFN 7168, MoE every layer | `TP4 + EP4 + DP2`; 3 experts per EP rank | Two half-width experts active per token, about 8B active / 34B total params |
| `large-120mtrain-moe-fp8-ep8-64e-top2-176btotal-8bactive-capacity001` | `8b`: L32, H4096, dense FFN 14336 | FP32 router `4096 -> 64`, softmax, top-2 | 64 FP8 experts, expert FFN 7168, MoE every layer | `TP4 + EP8 + DP1`; 8 experts per EP rank | Two half-width FP8 experts active per token, about 8B active / 176B total params |

## Parallelism

Dense runs use data parallelism plus Megatron distributed optimizer, which is ZeRO/FSDP-like sharding of optimizer state and gradients. Larger dense runs add tensor parallelism:

- small dense: usually `TP1 + DP32`
- medium dense: usually `TP1/TP2 + DP32/DP16`
- large dense: usually `TP4 + DP8`

MoE runs use:

- `moe_local_*`: `EP1`, all experts local to the rank group; this avoids expert all-to-all overhead.
- `moe_ep_*`: `EP4` or `EP8`, with all-to-all token dispatch and grouped GEMM.
- large EP MoE: `TP4 + EP8 + DP1`, using the full 32 GPUs for one model replica.

All configs use FlashAttention/TE attention, BF16 base precision, constant LR, the same dataset, same tokenizer, same validation split, same final eval logic, and the same per-scale eval interval.

## Metrics

Report and compare:

- Final eval loss: primary score.
- Perplexity: `exp(final_eval_loss)`.
- Tokens/sec and tokens/sec/GPU.
- Total training tokens.
- Total parameters and active parameters.
- GPU-hours: 16, 32, and 64 GPU-hours for small/medium/large benchmark clocks at 32 GPUs.
- Peak memory.
- Final step.
- Eval time and checkpoint time.
- MoE router stats: expert usage, load balance loss, z-loss, dropped-token rate if available.

W&B metrics to inspect: `lm loss validation`, `tokens-per-sec-per-gpu`, `throughput`, train `lm loss`, and any MoE auxiliary/router metrics emitted by Megatron.

## Checkpoint Naming

Checkpoint names follow:

```text
{scale}-{minutes}m-{arch}-{precision}-{parallelism}-{size_or_totalparams}-{moe_details}-{runid}
```

Example:

```text
medium-60m-moe-fp8-ep-32e-top2-33btotal-3bactive-run001
```

This means medium-scale, 60-minute benchmark clock, FP8 expert-parallel MoE, 32 experts total, top-2 routing, about 33B total parameters, about 3B active parameters per token, first run.

Dense names omit MoE details:

```text
small-30m-dense-bf16-fsdp-760m-run001
```

## Success Criteria

A job is successful only if it has:

1. A completed final eval score.
2. A final checkpoint, unless checkpointing was explicitly disabled.
3. A metrics JSON in `results/`.
4. No NaN/Inf loss.
5. For MoE, non-collapsed expert usage.

## Failure Criteria

A job fails if it is killed before final eval, has no eval score, gets NaN/Inf loss, collapses routing completely, exceeds the documented benchmark clock, or diverges from the generated YAML without documentation.

## Capacity Add-Ons

The main 27-job sweep already includes large 8B-active MoEs with 16 and 32 experts. The 32-expert runs are the fair inference-matched comparison to an 8B dense model: they keep active compute around 8B while increasing total sparse capacity to roughly 88B.

This schedules four extra large-scale jobs:

- `dense32_bf16_capacity`: approximate 32B dense BF16, `TP8+PP2`, activation recompute.
- `dense32_fp8_capacity`: approximate 32B dense FP8, `TP8+PP2`, activation recompute.
- `moe32btotal_8bactive_bf16`: 8B-active MoE, 12 experts, roughly 34B total parameters.
- `moe176btotal_8bactive_fp8`: 8B-active MoE, 64 experts, roughly 176B total parameters with FP8 experts.

These add-ons use 120 minutes of effective post-startup training plus a 10-minute final checkpoint/eval buffer. They are exploratory add-ons, not part of the main 27-job winner selection.

## Evaluation

Use `results/summary_template.csv` as the reporting sheet. After jobs finish, each run should also have a `results/<run_name>.json` file with final eval loss, final ppl, final step, train tokens, checkpoint time, and status.

In W&B, group by `RUN_GROUP`, then compare within each scale. The winner for each scale is the lowest final eval loss among successful jobs. A MoE method only wins if it beats `dense_bf16_fsdp`, `dense_fp8_fsdp`, `dense_fp8_maxfit`, and `dense_bestsize_bf16`.

Report separately:

- Best dense
- Best quantized dense
- Best MoE
- Best quantized MoE
- Best overall

Then explain whether MoE wins because of more total capacity, higher throughput, better sample efficiency, better large-scale behavior, or whether the gain disappears after accounting for fewer trained tokens.
