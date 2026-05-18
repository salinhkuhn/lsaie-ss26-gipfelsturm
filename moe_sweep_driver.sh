#!/bin/bash
#
# moe_sweep_driver.sh — Submits the 15-cell MoE all-to-all scaling sweep.
# See course-project/moe-sweep-plan.md for the rationale and plot-TODOs.
#
# Each line below corresponds to one cell. Args: <nodes> <ep> <topk> <num_experts> <precision>
#
# Usage:  ./moe_sweep_driver.sh
# Note:   each call submits a SLURM job; all 15 enter the queue immediately and
#         run as resources free. No sleep between submissions — SLURM handles ordering.

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

# --- Expert count slice (4 cells): nodes=8, EP=8, top-k=2, vary num_experts ---
./launch_moe_sweep.sh 8 8 2 4 bf16   # cell 11: 4 experts (coarse)
./launch_moe_sweep.sh 8 8 2 16 bf16  # cell 12: 16 experts
./launch_moe_sweep.sh 8 8 2 32 bf16  # cell 13: 32 experts (DeepSeekMoE)
./launch_moe_sweep.sh 8 8 2 64 bf16  # cell 14: 64 experts (very fine)

# --- Precision control (1 cell): cell 8 setup with FP8 experts ---
./launch_moe_sweep.sh 8 8 2 8 fp8    # cell 15: FP8 control

echo "=== All 15 cells submitted. Monitor with: squeue --me --start ==="
