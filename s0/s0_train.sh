#!/usr/bin/env bash
# S0 FlashSAC training (teammate-hand parquet, S0 observation).   s0_train.sh <episode> <gpu> <run_name> [hydra args...]
set -uo pipefail
EP=$1; GPU=$2; NAME=$3; shift 3
ROOT=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req
SEQ=$(printf 'episode_%06d' "$EP"); RUN=$ROOT/training_s0/$NAME; mkdir -p "$RUN"
exec docker run --rm --runtime=nvidia --gpus "device=$GPU" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --entrypoint bash --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYGLET_HEADLESS=1 -e PYTHONPATH=/fc/src:/env/s0 \
  -e S0_EMA=${S0_EMA:-} -e S0_VOC_GATE=${S0_VOC_GATE:-} -e FLASH_CHORD_CACHE_DIR=/cache -e WARP_CACHE_PATH=/cache/warp \
  -v /usr1/e2s/retarget_env/flashchord_cache:/cache -v /usr1/e2s/retarget_env:/env:ro \
  -v /usr1/e2s/video_to_data/robotic_grounding/flash_chord:/fc:ro -v "$ROOT/assets":/v2d/assets:ro \
  -v "${FEAT_DIR:-$ROOT/s0_features}":/feat:ro ${PKG_DIR:+-v $PKG_DIR:/pkg:ro} -v "$RUN":/run_out \
  flashchord-cuda12-575-test:193382a1-20261018 -c "cd /tmp && python /env/train_logged.py experiment=sharpa_flash_sac \
    \"task.parquet='${PARQUET:-/v2d/assets/human_motion_data/ego_recon/processed/sequence_id=${SEQ}${SEQ_SUFFIX-_teammate}/robot_name=sharpa_wave}'\" \
    hydra.searchpath=[file:///env/cfg] observation=policy_s0 observation.extras_path=/feat/${SEQ}.npz \
    training.replay.observation_storage_dtype=float16 \
    logging.mode=online logging.upload_checkpoints=false hydra.run.dir=/run_out $*"
