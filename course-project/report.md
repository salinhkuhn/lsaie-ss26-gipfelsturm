---
Gipfelsturm: From Megatron Defaults to a Throughput–Loss Frontier on Alps
---

## §1 — Abstract 

> Training large language models at scale on distributed systems requires informed engineering decisions. The Gipfelsturm course project tasked us with exactly this: either minimizing evaluation loss under fixed wall-clock budgets or maximizing training throughput at given model scales on the **Alps Clariden** supercomputer. However, analogous to training an LLM for a realistic deployment setting, we addressed both challenges via three coordinated investigations using Megatron-LM and the Nemotron-ClimbMix dataset. First, we measure single-node 8B dense training throughput across a curated set of configurations, identifying bottlenecks in the canonical Megatron launcher through profiling traces and reporting per-flag throughput changes for FP8 hybrid precision, larger micro-batches with selective activation recomputation, tied embeddings, and FlashAttention via Transformer Engine. Arguing that throughput is not the only measure to caraterize a preferable system, we run a wall-clock-budgeted benchmark comparing dense and Mixture-of-Experts architectures at 30-, 60-, and 120-minute budgets, measuring final validation loss at each operating point. Third, we conduct an empirical scaling sweep characterizing MoE all-to-all communication overhead on Slingshot-11 as a function of `(nodes × expert parallelism × top-k × num_experts)`. Combining all three studies, we plot the Pareto frontier on the (throughput, eval-loss) plane at our target scale. Our results expose which Megatron-LM flags actually move the throughput–loss frontier on Alps Clariden and give concrete operating-point recommendations for loss and/or throughput optimzation.
---

## §2 — Introduction 

> Training large language models on modern GPU clusters is an act of balance along multiple axes. A training framework must balance numerical precision (BF16, FP8) against numerical stability, micro-batch size against activation memory, dense parameter density against the routing cost of sparse Mixture-of-Experts architectures, and intra-node NVLink bandwidth against inter-node Slingshot bandwidth. Each individual axis has received attention in prior work: FlashAttention reduces attention's memory traffic [Dao et al. 2022, Shah et al. 2024], Transformer Engine exposes FP8 matmul on Hopper, ZeRO shards optimizer state across data-parallel ranks [Rajbhandari et al. 2020], and MoE architectures decouple model capacity from per-token compute [Shazeer et al. 2017, Fedus et al. 2021, Jiang et al. 2024] but looking at the config files within the Megatron LLM repo we see: The axes interact, and the operating point that maximizes throughput or minimizes training loss for a model on a specifc system depends on their combination. The system we use in this project is the **Alps Clariden** (Grace-Hopper GH200 compute nodes) and is recent that such efficiency numbers for combing Megatron knobs for this its hardware remain scarce.
>
> Thus the Gipfelsturm course project defined two challenges. 1. ) Minimize evaluation loss within fixed wall-clock budgets (30 min / 1 h / 2 h) on up to 8 nodes and 2.) the highest tok/s/GPU at specified model scales. We engage with both and structure our work as three coordinated investigations that share methodology (tuning flags), dataset (Nemotron-ClimbMix), tokenizer (GPT-2 BPE), and reproducibility (launch patterns around the canonical Megatron-LM launcher).
>
> First, we address Challenge 2 at the single-node 8B scale. Starting from the canonical Megatron launcher (BF16, MBS=2), we profile the baseline with `torch.profiler` and then add (in separate runs) FP8 hybrid precision through Transformer Engine, micro-batch size 4 with selective activation recomputation, tied input/output embeddings, and `--no-create-attention-mask-in-dataloader` (relying on FlashAttention's implicit causal-mask handling). We isolate the effect of micro-batch scaling from precision via an additional BF16+MBS=4 control run.
>
> Next, observing that throughput alone is incomplete, for example because a configuration that achieves higher tok/s but worse loss per training token may not be preferable, we address Challenge 1 with a wall-clock-budgeted benchmark comparing 9 configurations spanning BF16/FP8 precision, local-expert vs expert-parallel MoE, and dense vs sparse architectures at three budgets (30/60/120 min). The cell matrix is run on 8 nodes throughout to fix the parallelism budget.
>
> Third, we conduct an empirical scaling runs characterizing MoE all-to-all-related throughput on Slingshot-11. The runs span a subset of `(nodes ∈ {1,2,4,8}) × (EP ∈ {1,4,8}) × (top-k ∈ {1,2,4}) × (num_experts ∈ {8,16,32,64})` plus one FP8-expert control point. We use W&B-logged per-step throughput as the primary measurement and derive a scaling efficiency curve.

> Combining all three, we produce a **Pareto frontier on the (throughput, eval-loss) plane** characterizing  how the standard Megatron-LM knobs combine and identify which operating point dominates which course-defined budget and present an actionable map for picking among them under each of the course's two challenges.

---

## §3 — Background 

We provide background on the systems and theirs scales and dependencies relevant to this report.

> **3.1 — Hardware: Grace-Hopper GH200 and Slingshot-11.**  Each Alps Clariden node hosts four GH200 superchips connected by NVLink. Each superchip pairs a Grace ARM CPU with a Hopper H200 GPU via NVLink, giving the GPU near-DRAM access to the CPU's. Inter-node communication uses Slingshot-11 and large NCCL bus bandwidth intra-node than inter-node.
>
> **3.2 — Numerical precision on Hopper.** Hopper's tensor cores expose vrious precision formats, for exmple FP8 in two format that run at a higher peak FLOP/s than BF16. The different precicons can be used in a hybrid recipe using a more precise format in the forward pass and a format with larger range in the backward pass. In Megatron LLM we can use FP8 runs with this hybrid recipe.
>
> **3.3 — Attention: FlashAttention.** Standard attention materializes a `[batch × heads × seq × seq]` matrix while FlashAttention restructures the computation as tiled, IO-aware matmul keeping the intermediate results in SRAM [Dao 2022, Dao 2023]. FlashAttention-3 [Shah et al. 2024] adds Hopper-specific optimziations. We can access these optimizations via Transformer Engine that selects an attention backend automatically based on shapes and dtypes. In our work we identify which backend was active from runs rather than choosing one manually.
>
> **3.4 — Distributed training paradigms.** We have seen common sharding strategies such as DDP and ZeRO-1/2/3 [Rajbhandari et al. 2020] (progressive sharding of optimizer state, gradients, parameters). Megatron-LM's `--use-distributed-optimizer` lets us access  ZeRO-1 and we identified that the flags `--overlap-grad-reduce` / `--overlap-param-gather` allow us to overlap communication with backward compute. All our runs use this Megatron path. Another form of distributed paradigm is expert parallelism described bellow.
>
> **3.5 — Mixture of Experts.** In MoE a standard transformer's feed-forward block is replaced with `N` parallel "experts" plus a router that selects `top-k` of them per token [Shazeer et al. 2017, Lepikhin et al. 2020, Fedus et al. 2021]. Expert parallelism (EP) shards experts across ranks, requiring an all-to-all token dispatch each layer. EP scales the model's total parameter count without proportionally scaling per-token compute but costs inter-rank communication.
>

---

## §4 — Methodology 
**TODO**. 

- **4.1 — Hardware and container.** 8 nodes of Alps Clariden, GH200, container image `ngc-pytorch:26.01-py3-alps3` configured via EDF `~/.edf/alps3.toml`.
- **4.2 — Software stack.** Megatron-LM
- **4.3 — Dataset and tokenizer.** Nemotron-ClimbMix subset `climbmix_small_megatron`, pre-tokenized with GPT-2 BPE (50,257-vocab). Same dataset/tokenizer for all experiments to enable comparison.
- **4.4 — Reproducibility tooling.** Sibling-launcher pattern: `launch.sh` (canonical) + `launch_profile.sh`, `launch_bf16_mbs4.sh`, `launch_stacked.sh`, `launch_stacked_3b.sh`, `launch_moe_sweep.sh`, `moe_sweep_driver.sh`. Each launcher writes a self-contained `.sbatch` file in `logs/`. 
- **4.5 — Measurement protocol.** `tokens-per-sec-per-gpu` averaged over W&B-logged steps 20–49 (post-warmup steady state). Eval loss is the last logged `lm loss validation` for completed runs.


## §5 — Single-node dense throughput

> **5.1 — Baseline and bottleneck identification.**
>
> We begin from the unmodified `launch.sh throughput 8b 50 1` configuration (BF16 mixed precision, micro-batch 2, global batch 256, sequence length 4096, grouped-query attention with 8 KV heads, distributed optimizer with `--overlap-grad-reduce` and `--overlap-param-gather`, FlashAttention as selected automatically by Transformer Engine, single node, 4 GH200 GPUs). This launcher is our project's canonical baseline and deviations from it are tracked as "sibling" launcher scripts in our repository. (sarah did that ask mischa how he did his)
>
> A 50-step profiling run (`launch_profile.sh`) captures `torch.profiler` traces at steps 10–12 on rank 0, after an initial warmup. Figure F1 shows the resulting trace decomposition for a representative step : to do insert.

<!-- : `[X1]`% matmul kernels, `[X2]`% attention kernels (the specific FlashAttention version selected by Transformer Engine at runtime will be verified from the trace and reported), `[X3]`% NCCL gradient reduction, `[X4]`% time outside CUDA kernels. We do not assert the bottleneck identity in advance: the diagnosis (compute-bound vs memory-bandwidth-bound vs kernel-launch-bound) is read off the trace once the profile job completes, with the supporting evidence cited (kernel occupancy, gaps between launches, NCCL-overlap behavior). -->
>
> **Headline baseline throughput**: `[X tok/s/GPU]` averaged over steps 20–49 (W&B run `[ID]`).
>
> **5.2 — Single-knob ablations (Mischa).**
>
> We conducted a series of single-flag ablations on an 8B FP8 hybrid baseline at micro-batch 1 (`gipfelsturm` and `single_node` W&B projects von mischa). The full table (to do where do we put it ) Of the 4 nominal replicates of the same baseline configuration that ran to completion, the W&B `tokens-per-sec-per-gpu` summary values were 15,346, 14,296, 13,809, and 13,621 (mean 14,268, range 13,621–15,346, ~12% spread). One additional run (`throughput-8b-1n-2279409`, 2026-05-17) of the same nominal configuration recorded 9,72 which looks like an outlier to us, possibly explained by cluster-side I/O contention or the W&B summary capturing issues. (TO DO:MAYBE RERUN ???)
>
> We obserbed that the tested single-knob variants returned no clear effect: CUDA-graph capture (`--cuda-graph-impl local`), and the FP8 tensorwise vs delayed scaling recipe each produced throughput values within the spread which could be noise only. TP=2 produced 13,150 tok/s/GPU, slightly below the baseline mean which we interpret this as either within-noise or even modestly negative, but not large enough to claim a robust effect from a single run. The single-knob ablation set therefore yields  for us a noise-range result for each individual flag and we cannot identify a positive "winning knob".
>
> Whether the bottlenecking constraint at this configuration is kernel scheduling, memory bandwidth, or something else cannot be concluded from these data alone.
>
> **5.3 — Stacked optimizations (Sarah) **
>
> We ran a stacked experiment (`launch_stacked.sh`) combining five known flags: FP8 hybrid precision, micro-batch size 4 (vs 2), selective activation recomputation, tied input/output embeddings, and the `--no-create-attention-mask-in-dataloader` flag (which removes a redundant CPU-side mask construction). Compared to the unmodified baseline:
>
> | Configuration | tok/s/GPU | Speedup vs baseline |
> |---|---:|---:|
> | `launch.sh` baseline (BF16, MBS=2) | `[X]` | 1.0× (reference) |
> | Mischa's FP8 ablation mean (MBS=1) | 14,162 | `[X_FP8]`× |
> | `launch_bf16_mbs4.sh` (BF16, MBS=4, recompute) | `[X_MBS]` | `[X_MBS]`× |
> | `launch_stacked.sh` (FP8 + MBS=4 + selective recompute + tied + no-mask) | `[X_STACK]` | `[X_STACK]`× |
> | `launch_stacked_3b.sh` (same stack on 3B backbone, MBS=8) | `[X_3B]` | (different model) |
>
> Multiplicative decomposition: the FP8-only effect (`[A]`×), the MBS-only effect (`[B]`×), and the stacked effect (`[C]`×) give a composition bonus `C / (A × B)` = `[D]`. A value greater than 1 indicates that the knobs reinforce each other (FP8 frees memory headroom for larger MBS, larger MBS gives FP8 matmuls more arithmetic intensity), a value less than 1 indicates interference (e.g. memory contention or precision-scaling instability at high batch).
>
<!-- > **5.4 — Roofline placement.**
>
> Figure F3 places all single-node configurations on the roofline plane (arithmetic intensity vs achieved TFLOP/s) with the GH200's published BF16 (990 TFLOP/s dense) and FP8 (1980 TFLOP/s dense) compute ceilings and 4 TB/s HBM3e bandwidth slope drawn for reference. Achieved TFLOP/s is estimated as `6 × N_active_params × tok_per_sec_per_gpu`, the standard `6N` approximation for transformer training (2N forward, 4N backward); this excludes attention-block FLOPs scaling with `seq²` which are non-negligible at our sequence length and FlashAttention's actual arithmetic, so the resulting TFLOP/s should be treated as a lower bound on the true achieved compute. Arithmetic intensity per run is estimated via the dominant linear-layer matmul shapes; ideally we would also extract it directly from the profile trace via cumulative kernel FLOPs and HBM traffic, and we note this as future work. The `stacked_3b` variant achieves the highest absolute tok/s/GPU number in the study, but at a different point on the loss frontier (§7) — comparing it to the 8B configurations is a Pareto comparison, not a strictly-better one. -->

---

### §6 — Multi-node MoE scaling

> **6.1 — Why all-to-all matters.**
>
> Mixture-of-Experts (MoE) layers replace the dense feed-forward block with `N` parallel "experts" plus a router that sends each token to `top-k` of them. When experts are sharded across GPU ranks ("expert parallelism", EP > 1), each transformer layer dispatches tokens to their assigned experts via an all-to-all permutation, computes the experts, then gathers the results back via a second all-to-all. The bytes dispatched per layer per step scale roughly with `MBS × seq_len × hidden × top_k × bytes_per_elem`. On Alps Clariden the our `test-infra.sbatch` benchmark reports ~340 GB/s NCCL bus bandwidth intra-node (NVLink) and ~93 GB/s inter-node (Slingshot-11); thus once expert parallelism spans nodes, dispatch crosses an interconnect with roughly 3.6× lower effective bandwidth than the intra-node. We are not aware of a public characterization of MoE all-to-all overhead for this specific GH200 + Slingshot-11 combination and thus we decided to implement it.
>
> **6.2 — Sweep design.**
>
> We ran a 14-cell sweep with a 760M-active backbone, MBS 4, BF16 base precision, and Megatron-LM's standard MoE recipe (softmax routing in FP32, auxiliary load-balance loss with coefficient 1e-2, router z-loss 1e-3, grouped GEMM, all-to-all dispatcher). Three axes vary:
>
> - **Nodes × EP** (8 cells): `nodes ∈ {1,2,4,8}` × `EP ∈ {1,4,8}` (subset). Fixed `top-k=2`, `num_experts=8`. Tests the scaling efficiency frontier.
> - **Top-k** (2 cells): `top-k ∈ {1, 4}` at `nodes=8, EP=8, num_experts=8`. Companion to cell 8's top-k=2. Tests linear scaling of all-to-all traffic with k.
> - **Expert count** (3 cells): `num_experts ∈ {16, 32, 64}` at `nodes=8, EP=8, top-k=2`. Companion to cell 8's `num_experts=8`. Note that since `--moe-ffn-hidden-size` defaults to the dense FFN width when unset (Megatron `arguments.py:1016`), all experts in our sweep are full-width; increasing `num_experts` therefore increases *total* parameters (and per-EP-rank parameters) but keeps active compute per token constant. This is a different setup than DeepSeekMoE's recipe, which shrinks expert width when increasing expert count to keep active *and* total parameters comparable. We do *not* directly test the DeepSeekMoE claim with this slice; we only characterize how all-to-all overhead and grouped-GEMM throughput change with expert count at fixed top-k.
> - **Precision control** (1 cell): cell 8 setup with FP8 experts. The matmul inside each expert uses FP8 (1 byte per element vs 2 for BF16). Whether the dispatched-token-bytes are correspondingly halved depends on which precision the dispatch operates in inside Megatron-LM's `alltoall` token dispatcher; we will verify this from the trace rather than assume it.
>
> Each cell is one SLURM submission via `launch_moe_sweep.sh` and the full sweep is driven by `moe_sweep_driver.sh` whcih is a driver script.
>
> **6.3 — Scaling efficiency.**
>

>
> **6.4 — All-to-all overhead vs traffic.**
>

>
> **6.5 — Top-k and expert-count results.**
>


---

### §7 — Throughput–loss frontier

> **7.1 — Combining throughput and eval-loss data.**
>

> **7.2 — CONCLUSION FIHURE MAYBE ???.**
>

>
> **7.3 — Our guidance/ interpretation given our results.**
>


---

