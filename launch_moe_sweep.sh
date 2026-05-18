#!/bin/bash
#
# launch_moe_sweep.sh — Parameterized launcher for the MoE all-to-all scaling sweep.
#
# Sibling of launch.sh. Used by moe_sweep_driver.sh to submit 15 cells of a
# (nodes x EP x top-k x num_experts) sweep. Canonical launch.sh untouched.
#
# Usage:
#   ./launch_moe_sweep.sh <nodes> <ep> <topk> <num_experts> [precision]
#
#   precision = "bf16" (default) or "fp8"
#
# Example:
#   ./launch_moe_sweep.sh 8 8 2 8 bf16     # cell #8: 8 nodes, EP=8, top-2, 8 experts
#   ./launch_moe_sweep.sh 8 8 2 8 fp8      # cell #15: same but FP8 experts
#
# Fixed across the sweep:
#   - Backbone:    760m (L24, H1536, FFN4096, 16 heads, KV=4)
#   - MBS:         4
#   - Steps:       50
#   - Seq length:  4096
#   - Recompute:   selective
#   - Tied embeddings
#   - --no-create-attention-mask-in-dataloader
#   - Router:      softmax, FP32, aux_loss=1e-2, z-loss=1e-3
#   - Grouped GEMM, alltoall dispatcher
#   - PROJECT_NAME = gipfelsturm-moe-sweep

set -euo pipefail

source "$(dirname "$0")/config.sh"

NODES=${1:?Usage: ./launch_moe_sweep.sh <nodes> <ep> <topk> <num_experts> [precision]}
EP=${2:?Usage: ./launch_moe_sweep.sh <nodes> <ep> <topk> <num_experts> [precision]}
TOPK=${3:?Usage: ./launch_moe_sweep.sh <nodes> <ep> <topk> <num_experts> [precision]}
NUM_EXPERTS=${4:?Usage: ./launch_moe_sweep.sh <nodes> <ep> <topk> <num_experts> [precision]}
PRECISION=${5:-bf16}

if [[ "$PRECISION" != "bf16" && "$PRECISION" != "fp8" ]]; then
    echo "precision must be 'bf16' or 'fp8'; got '$PRECISION'"
    exit 1
fi

################ Fixed config ################
TRAINING_STEPS=50
TIME=00:45:00
EVAL_INTERVAL=$TRAINING_STEPS
EVAL_ITERS=0
LR_WARMUP_ITERS=10
WANDB=true

# 760m backbone
NUM_LAYERS=24
HIDDEN=1536
FFN=4096
HEADS=16
KV_HEADS=4
MBS=4
GBS=256
SEQ_LEN=4096

JOB_NAME="gipfel-moe-n${NODES}-ep${EP}-k${TOPK}-e${NUM_EXPERTS}-${PRECISION}-${TRAINING_STEPS}s"

################ W&B block ################
WANDB_BLOCK='
if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB enabled."
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOG_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
fi'

################ FP8 block (only if precision=fp8) ################
if [ "$PRECISION" = "fp8" ]; then
    FP8_ARGS='
    --fp8-format hybrid
    --fp8-amax-history-len 1024
    --fp8-amax-compute-algo max'
else
    FP8_ARGS=''
fi

################ Generate sbatch script ################
mkdir -p logs

SCRIPT="logs/${JOB_NAME}.sbatch"

cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
HEADER

cat >> "$SCRIPT" << SBATCH_DIRECTIVES
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue
SBATCH_DIRECTIVES

cat >> "$SCRIPT" << 'BODY_HEAD'

echo "START TIME: \$(date)"

################ Configs ################
BODY_HEAD

cat >> "$SCRIPT" << BODY_WORKDIR
WORKDIR=${WORKDIR}
MEGATRON_LM_DIR=\$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/cache
BODY_WORKDIR

cat >> "$SCRIPT" << CONFIGS

# Training config
MBS=${MBS}
GBS=${GBS}
SEQ_LEN=${SEQ_LEN}
TRAINING_STEPS=${TRAINING_STEPS}

# Logging
PROJECT_NAME=gipfelsturm-moe-sweep
EXP_NAME=moe-n${NODES}-ep${EP}-k${TOPK}-e${NUM_EXPERTS}-${PRECISION}
LOG_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/\$PROJECT_NAME/\$EXP_NAME
TENSORBOARD_DIR=\$LOG_DIR/tensorboard
CONFIGS

cat >> "$SCRIPT" << 'SETUP'

#########################################

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR

cd $MEGATRON_LM_DIR
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)

SETUP

cat >> "$SCRIPT" << MODEL
NETWORK_SIZE_ARGS=(
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN}
    --ffn-hidden-size ${FFN}
    --num-attention-heads ${HEADS}
    --group-query-attention
    --num-query-groups ${KV_HEADS}
    --max-position-embeddings \$SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --seq-length \$SEQ_LEN
)
MODEL

cat >> "$SCRIPT" << TRAINING

TRAINING_ARGS=(
    --micro-batch-size \$MBS
    --global-batch-size \$GBS
    --train-iters \$TRAINING_STEPS
    --log-interval 1
    --eval-interval ${EVAL_INTERVAL}
    --eval-iters ${EVAL_ITERS}
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer adam
    --dataloader-type single
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 50
    --recompute-granularity selective
    --no-create-attention-mask-in-dataloader
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
    --lr 3e-4
    --lr-decay-style constant
    --lr-warmup-iters ${LR_WARMUP_ITERS}
)
TRAINING

cat >> "$SCRIPT" << MOE

MOE_ARGS=(
    --num-experts ${NUM_EXPERTS}
    --moe-router-topk ${TOPK}
    --moe-router-pre-softmax
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-z-loss-coeff 1e-3
    --moe-input-jitter-eps 1e-2
    --expert-model-parallel-size ${EP}
)
MOE

cat >> "$SCRIPT" << 'REST'

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

MIXED_PRECISION_ARGS=(
    --bf16
REST

cat >> "$SCRIPT" << FP8_INSERT
${FP8_ARGS}
)
FP8_INSERT

cat >> "$SCRIPT" << 'REST2'

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
)

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/data/gpt2-vocab.json
    --merge-file $WORKDIR/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $DATA_PREFIX
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 1
)

TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

TRAINING_CMD="torchrun ${TORCHRUN_ARGS[@]} $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

REST2

cat >> "$SCRIPT" << WANDB_INSERT
${WANDB_BLOCK}
WANDB_INSERT

cat >> "$SCRIPT" << 'FOOTER'

echo "CMD: $TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "numactl --membind=0-3 $TRAINING_CMD"

echo "END TIME: $(date)"
FOOTER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT"
sbatch "$SCRIPT"
