---
Gipfelsturm: From Megatron Defaults to a Throughput–Loss Frontier on Alps
---

## §1 — Abstract 

> Training large language models at scale on distributed systems requires informed engineering decisions. The Gipfelsturm course project tasked us with exactly this: either minimizing evaluation loss under fixed wall-clock budgets or maximizing training throughput at given model scales on the **Alps Clariden** supercomputer. However, analogous to training an LLM for a realistic deployment setting, we addressed both challenges via three coordinated investigations using Megatron-LM and the Nemotron-ClimbMix dataset. First, we measure single-node 8B dense training throughput across a curated set of configurations, identifying bottlenecks in the canonical Megatron launcher through profiling traces and reporting per-flag throughput changes for FP8 hybrid precision, larger micro-batches with selective activation recomputation, tied embeddings, and FlashAttention via Transformer Engine. Arguing that throughput is not the only measure to caraterize a preferable system, we run a wall-clock-budgeted benchmark comparing dense and Mixture-of-Experts architectures at 30-, 60-, and 120-minute budgets, measuring final validation loss at each operating point. Third, we conduct an empirical scaling sweep characterizing MoE all-to-all communication overhead on Slingshot-11 as a function of `(nodes × expert parallelism × top-k × num_experts)`. Combining all three studies, we plot the Pareto frontier on the (throughput, eval-loss) plane at our target scale. Our results expose which Megatron-LM flags actually move the throughput–loss frontier on Alps Clariden and give concrete operating-point recommendations for loss and/or throughput optimzation.
---

## §2 — Introduction (draft)

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
