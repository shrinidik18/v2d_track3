#!/usr/bin/env bash
# Evaluate and/or render one S0 checkpoint.   s0_eval.sh <run_dir> <env_steps> <gpu> [world_count]
#   env: RENDER=1 (default 1) renders a zoomed video; RENDER_ONLY=1; FINAL=1 writes official-style eval_final outputs
set -uo pipefail
RUN=$1; STEP=$2; GPU=$3; WC=${4:-64}; ROOT=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req
mkdir -p "$RUN/eval"
D="docker run --rm --runtime=nvidia --gpus device=$GPU -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics --entrypoint bash \
  --user $(id -u):$(id -g) -e HOME=/tmp -e PYGLET_HEADLESS=1 -e PYTHONPATH=/fc/src:/env/s0 -e FLASH_CHORD_CACHE_DIR=/cache -e WARP_CACHE_PATH=/cache/warp \
  -v /usr1/e2s/retarget_env/flashchord_cache:/cache -v /usr1/e2s/retarget_env:/env:ro -v /usr1/e2s/video_to_data/robotic_grounding/flash_chord:/fc:ro \
  -v $ROOT/assets:/v2d/assets:ro -v ${FEAT_DIR:-$ROOT/s0_features}:/feat:ro ${PKG_DIR:+-v $PKG_DIR:/pkg:ro} -v $RUN:/run_out flashchord-cuda12-575-test:193382a1-20261018"
CK=/run_out/policy_${STEP}.safetensors
if [[ "${RENDER_ONLY:-0}" != 1 ]]; then
  if [[ "${FINAL:-0}" == 1 ]]; then mkdir -p "$RUN/eval_final"; MO=/run_out/eval_final/metrics_${STEP}_w${WC}.json; OUT="evaluation.object_trajectories_output=/run_out/eval_final/objects_${STEP}_w${WC}.parquet"
  else MO=/run_out/eval/metrics_${STEP}.json; OUT=""; fi
  $D -c "cd /tmp && python /fc/scripts/evaluate_policy.py evaluation.checkpoint=$CK evaluation.world_count=$WC evaluation.metrics_output=$MO $OUT hydra.run.dir=/tmp/hydra" > "$RUN/eval/eval_${STEP}_w${WC}.log" 2>&1
fi
if [[ "${RENDER:-1}" == 1 ]]; then
  EP=$(echo "$RUN" | grep -oE "/ep[0-9]+_" | tail -1 | tr -dc 0-9)
  if [[ -n "${PKG_DIR:-}" ]]; then          # evaluation episode: camera and length from Rian's package
    read -r CP CT < <(/usr1/e2s/video_to_data/reconstruction/.venv/bin/python /usr1/e2s/retarget_env/sub/cam_for_pkg.py "$EP" 0.5 0.25)
    N=$(( $(awk -F, -v e=$EP '$1==e{print $3}' /home/rborah/v2d/share/ego2sim_eval_recon_v1/quality.csv) * 5 / 2 ))
  else
    read -r CP CT < <(/usr1/e2s/video_to_data/reconstruction/.venv/bin/python /usr1/e2s/retarget_env/cam_for.py "$EP" 0.45 0.22)
    N=$(/usr1/e2s/video_to_data/reconstruction/.venv/bin/python -c "import numpy as np;print(int(np.load('$ROOT/episodes/episode_%06d/hand_trajectories.npz'%$EP)['time_s'].shape[0])*5//2)")
  fi
  $D -c "cd /tmp && python /fc/scripts/view_policy.py evaluation.checkpoint=$CK viewer.backend=mp4 viewer.output_path=/run_out/eval/policy_${STEP}.mp4 \
    viewer.num_frames=$N viewer.video_fps=50 viewer.width=1280 viewer.height=720 viewer.headless=true viewer.realtime=false \
    'viewer.camera_position=[$CP]' 'viewer.camera_target=[$CT]' markers.keypoints=false markers.contacts=false hydra.run.dir=/tmp/hydra" > "$RUN/eval/render_${STEP}.log" 2>&1
fi
