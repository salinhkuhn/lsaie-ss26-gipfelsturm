#!/bin/bash
#
# moe_sweep_driver.sh — Submits the 14-cell MoE all-to-all scaling sweep.
#
# Each line below corresponds to one cell. Args: <nodes> <ep> <topk> <num_experts> <precision>
#
# Usage:
#   1) SMOKE TEST FIRST: ./launch_moe_sweep.sh 1 1 2 8 bf16
#      Wait for it to reach iteration 50/50 cleanly in W&B before running the
#      full sweep. Catches bugs in the launcher itself before wasting 14 slots.
#
#   2) Full sweep: ./moe_sweep_driver.sh
#
# Note:   SLURM serializes jobs by allocation, not submission time, so no
#         submission-stagger is needed. The `flock` in launch.sh protects the
#         shared Megatron-LM submodule's patch-apply step.

set -euo pipefail
cd "$(dirname "$0")"

echo "=== Submitting 15-cell MoE sweep to gipfelsturm-moe-sweep W&B project ==="

# --- Main grid (8 cells): nodes x EP, fixed top-k=2, num_experts=8 ---
./launch_moe_sweep.sh 1 1 2 8 bf16   # cell  1: local single-node anchor
./launch_moe_sweep.sh 1 4 2 8 bf16   # cell  2: EP=4 intra-node only
./launch_moe_sweep.sh 2 1 2 8 bf16   # cell  3: local 2 nodes
./launch_moe_sweep.sh 2 4 2 8 bf16   # cell  4: EP=4 across 2 nodes
./launch_moe_sweep.sh 4 1 2 8 bf16   # cell  5: local 4 nodes
./launch_moe_sweep.sh 4 4 2 8 bf16   # cell  6: EP=4 across 4 nodes (first inter-node a2a)
./launch_moe_sweep.sh 8 4 2 8 bf16   # cell  7: EP=4 across 8 nodes
./launch_moe_sweep.sh 8 8 2 8 bf16   # cell  8: EP=8 across 8 nodes (worst case)

# --- Top-k slice (2 cells): nodes=8, EP=8, num_experts=8, vary top-k ---
./launch_moe_sweep.sh 8 8 1 8 bf16   # cell  9: top-1 (Switch-style)
./launch_moe_sweep.sh 8 8 4 8 bf16   # cell 10: top-4 (DeepSeekMoE-style)

# --- Expert count slice (3 cells): nodes=8, EP=8, top-k=2, vary num_experts ---
# NOTE: Megatron requires num_experts % EP == 0. With EP=8, num_experts < 8 is
# invalid; cell with num_experts=4 dropped. Slice starts at 16.
./launch_moe_sweep.sh 8 8 2 16 bf16  # cell 11 (was 12): 16 experts
./launch_moe_sweep.sh 8 8 2 32 bf16  # cell 12 (was 13): 32 experts (DeepSeekMoE-style)
./launch_moe_sweep.sh 8 8 2 64 bf16  # cell 13 (was 14): 64 experts (very fine)

# --- Precision control (1 cell): cell 8 setup with FP8 experts ---
./launch_moe_sweep.sh 8 8 2 8 fp8    # cell 14 (was 15): FP8 control

echo "=== All 13 cells submitted. Monitor with: squeue --me --start ==="
echo "Note: cell 2 (EP=4, DP=1) runs grad_accumulation=64, expect ~50 min instead of ~15."
