#!/usr/bin/env bash
# Run one VAPO ablation. Keep generated trajectories approximately fixed by
# setting TRAIN_BATCH_SIZE * ROLLOUT_N to TOTAL_TRAJECTORIES.

set -euo pipefail

ABLATION=${1:?usage: run_ablation.sh NAME [hydra overrides...]}
shift
TOTAL_TRAJECTORIES=${TOTAL_TRAJECTORIES:-512}
ROLLOUT_N=${ROLLOUT_N:-16}

if [[ "$ABLATION" != "full_n16" ]]; then
    ROLLOUT_N=1
fi
if (( TOTAL_TRAJECTORIES % ROLLOUT_N != 0 )); then
    echo "TOTAL_TRAJECTORIES must be divisible by ROLLOUT_N" >&2
    exit 2
fi
export ROLLOUT_N
export TRAIN_BATCH_SIZE=$((TOTAL_TRAJECTORIES / ROLLOUT_N))
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-vapo_${ABLATION}}

COMMON=()
case "$ABLATION" in
    vanilla_ppo)
        COMMON+=(
            algorithm.gae.decoupled=false
            algorithm.gae.length_adaptive=false
            actor_rollout_ref.actor.clip_ratio_high=0.20
            actor_rollout_ref.actor.positive_lm.enabled=false
            trainer.value_warmup_steps=0
        )
        ;;
    token_mean_clip_higher)
        COMMON+=(
            algorithm.gae.decoupled=false
            algorithm.gae.length_adaptive=false
            actor_rollout_ref.actor.positive_lm.enabled=false
            trainer.value_warmup_steps=0
        )
        ;;
    decoupled_gae)
        COMMON+=(
            algorithm.gae.length_adaptive=false
            actor_rollout_ref.actor.positive_lm.enabled=false
            trainer.value_warmup_steps=0
        )
        ;;
    length_adaptive_gae)
        COMMON+=(
            actor_rollout_ref.actor.positive_lm.enabled=false
            trainer.value_warmup_steps=0
        )
        ;;
    value_pretraining)
        COMMON+=(actor_rollout_ref.actor.positive_lm.enabled=false)
        ;;
    positive_lm)
        ;;
    full_n16 | full_n1)
        ;;
    *)
        echo "unknown ablation: $ABLATION" >&2
        exit 2
        ;;
esac

exec bash "$(dirname "$0")/run_qwen3_8b_fsdp2.sh" "${COMMON[@]}" "$@"
