#!/bin/bash
# Training script for MLA + LRU on Causal Loop Task
#
# Usage:
#   ./scripts/train_lru.sh [experiment_group]
#
# Experiment groups:
#   baseline     - Qwen-0.5B without MLA/LRU
#   mla_only     - MLA attention without LRU
#   mla_lru_fixed    - MLA + LRU with fixed iterations
#   mla_lru_adaptive - MLA + LRU with ACT halting (default)

set -e

# Default configuration
EXPERIMENT_GROUP=${1:-mla_lru_adaptive}
DATA_DIR=${DATA_DIR:-~/data/causal_loop}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-0.5B}
N_GPUS=${N_GPUS:-1}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}

# Experiment-specific settings
case $EXPERIMENT_GROUP in
  "baseline")
    EXPERIMENT_NAME="causal_loop_baseline"
    USE_MLA=false
    USE_LRU=false
    ;;
  "mla_only")
    EXPERIMENT_NAME="causal_loop_mla_only"
    USE_MLA=true
    USE_LRU=false
    ;;
  "mla_lru_fixed")
    EXPERIMENT_NAME="causal_loop_mla_lru_fixed"
    USE_MLA=true
    USE_LRU=true
    HALT_THRESHOLD=1.0
    ;;
  "mla_lru_adaptive")
    EXPERIMENT_NAME="causal_loop_mla_lru_adaptive"
    USE_MLA=true
    USE_LRU=true
    HALT_THRESHOLD=0.99
    ;;
  *)
    echo "Unknown experiment group: $EXPERIMENT_GROUP"
    echo "Available: baseline, mla_only, mla_lru_fixed, mla_lru_adaptive"
    exit 1
    ;;
esac

echo "============================================"
echo "ZeroModel LRU Training"
echo "============================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Data: $DATA_DIR"
echo "Model: $BASE_MODEL"
echo "GPUs: $N_GPUS"
echo "MLA: $USE_MLA"
echo "LRU: $USE_LRU"
echo "============================================"

# Step 1: Generate data if not exists
if [ ! -f "$DATA_DIR/train.parquet" ]; then
  echo "Generating causal loop dataset..."
  python examples/data_preprocess/causal_loop.py \
    --local_dir $DATA_DIR \
    --train_size 50000 \
    --test_size 1000 \
    --template_type qwen-instruct \
    --seed 42
fi

# Step 2: Run training
python3 -m verl.trainer.main_ppo \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet \
  data.train_batch_size=256 \
  data.val_batch_size=64 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path=$BASE_MODEL \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
  critic.optim.lr=1e-5 \
  critic.model.path=$BASE_MODEL \
  critic.ppo_micro_batch_size=8 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger=['wandb'] \
  +trainer.val_before_train=False \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=50 \
  trainer.project_name=ZeroModel \
  trainer.experiment_name=$EXPERIMENT_NAME \
  trainer.total_epochs=15 \
  +reward.style=rule \
  +reward.reward_module=verl.utils.reward_score.causal_loop \
  2>&1 | tee logs/${EXPERIMENT_NAME}.log
