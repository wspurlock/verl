#!/usr/bin/env bash
# VAPO | text-only, single-turn | v1 sync | FSDP2 | vLLM
# Runnable defaults are intentionally smaller than the paper-scale recipe.

set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:-$MODEL_PATH}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/gsm8k/test.parquet}
N_GPUS=${N_GPUS:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
ROLLOUT_N=${ROLLOUT_N:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/vapo}

python3 -m verl.trainer.main_ppo \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync \
    algorithm.adv_estimator=gae \
    algorithm.gamma=1.0 \
    algorithm.use_kl_in_reward=false \
    algorithm.gae.decoupled=true \
    algorithm.gae.critic_lambda=1.0 \
    algorithm.gae.length_adaptive=true \
    algorithm.gae.length_alpha=0.05 \
    algorithm.gae.policy_lambda_min=0.0 \
    algorithm.gae.policy_lambda_max=1.0 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.clip_ratio_low=0.20 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.positive_lm.enabled=true \
    actor_rollout_ref.actor.positive_lm.coef=0.1 \
    actor_rollout_ref.actor.positive_lm.correctness_key=rm_scores \
    actor_rollout_ref.actor.positive_lm.correctness_threshold=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    critic.model.path="$CRITIC_MODEL_PATH" \
    critic.strategy=fsdp2 \
    critic.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    critic.use_dynamic_bsz=true \
    critic.optim.lr=2e-6 \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    trainer.value_warmup_steps=50 \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    "$@"
