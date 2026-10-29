#!/usr/bin/env bash
# settle_all.sh <gpu> <episodes...>: frame-0 spawn settle on the stage-1 cleaned references
C=/usr1/e2s/v2d_challenge_data/outputs/rian_eval_compare/_cleaned; GPU=$1; shift
for EP in "$@"; do
  docker run --rm --runtime=nvidia --gpus device=$GPU --entrypoint bash --user $(id -u):$(id -g) -e HOME=/tmp -e PYTHONPATH=/fc/src \
    -e FLASH_CHORD_CACHE_DIR=/cache -e WARP_CACHE_PATH=/cache/warp -v /usr1/e2s/retarget_env/flashchord_cache:/cache \
    -v /usr1/e2s/retarget_env:/env:ro -v /usr1/e2s/video_to_data/robotic_grounding/flash_chord:/fc:ro -v $C/package:/pkg:ro -v $C:/out \
    flashchord-cuda12-575-test:193382a1-20261018 -c "cd /tmp && python /env/rian_eval/settle_spawn.py /pkg/sim_reference/processed/sequence_id=ep$EP/robot_name=sharpa_wave /out/ep${EP}_settle.json replay.voc_scale=0.0 2>&1 | grep -E 'settle:|Error|Traceback'" | sed "s/^/ep$EP /"
done
