#!/usr/bin/env bash
# 学習を開始する。runs/<名前> にチェックポイントがあれば、そこから再開する(Linux / Docker / WSL)。
#
#   bash scripts/start_training.sh [--device cuda] [--n-envs 16] [--total-steps 50000000]
#                                  [--run-name solo_main] [--vec-env dummy|subproc]
#                                  [--checkpoint-every 500000] [--video-every 4]
#                                  [--set key=value ...]
set -euo pipefail

RUN_NAME=solo_main
TOTAL_STEPS=50000000
N_ENVS=16
VEC_ENV=dummy
DEVICE=auto
CHECKPOINT_EVERY=500000
VIDEO_EVERY=4
EXTRA=()

usage() { sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --run-name)         RUN_NAME=$2; shift 2 ;;
        --total-steps)      TOTAL_STEPS=$2; shift 2 ;;
        --n-envs)           N_ENVS=$2; shift 2 ;;
        --vec-env)          VEC_ENV=$2; shift 2 ;;
        --device)           DEVICE=$2; shift 2 ;;
        --checkpoint-every) CHECKPOINT_EVERY=$2; shift 2 ;;
        --video-every)      VIDEO_EVERY=$2; shift 2 ;;
        --set)              EXTRA+=(--set "$2"); shift 2 ;;
        -h|--help)          usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 2 ;;
    esac
done

cd "$(dirname "$0")/.."
run="runs/$RUN_NAME"
args=(scripts/train.py)
if compgen -G "$run/checkpoints/step_*.zip" > /dev/null; then
    echo "Resuming $run (target $TOTAL_STEPS steps)"
    args+=(--resume "$run")
else
    echo "Starting new run $run (target $TOTAL_STEPS steps)"
    args+=(--run-dir "$run")
fi
args+=(
    --set "ppo.total_timesteps=$TOTAL_STEPS"
    --set "env.n_envs=$N_ENVS"
    --set "env.vec_env=$VEC_ENV"
    --set "device=$DEVICE"
    --set "checkpoint.every_steps=$CHECKPOINT_EVERY"
    --set "video.every_checkpoints=$VIDEO_EVERY"
)
echo "Stop with Ctrl+C (or docker stop); the current weights are saved. Run again to resume."
exec python "${args[@]}" "${EXTRA[@]}"
