#!/usr/bin/env bash
# Run a python command in the retargeting environment (no new Docker image).
#   run_in_env.sh [--gpu N] <python args...>
# Example:
#   run_in_env.sh -m v2d.task_library_loader.lib.run_loader --dataset ego_recon --help
#   run_in_env.sh /rgscripts/retarget/ego_recon_to_sharpa.py --help
# The container is the existing v2d_hamer image (torch, manotorch, trimesh, ...). Extra packages
# (pinocchio, pink, daqp, viser, pyarrow) live in /usr1/e2s/retarget_env/venv (system site-packages
# visible, numpy stays 1.26.4). Repo trees are mounted read-only; only /data (the challenge outputs
# folder) and /env are writable. Nothing is written to /usr0 or to the host system.
set -euo pipefail
GPU_ARGS=()
if [[ "${1:-}" == "--gpu" ]]; then GPU_ARGS=(--runtime=nvidia --gpus "device=$2"); shift 2; fi
RG=/usr1/e2s/video_to_data/robotic_grounding
exec docker run --rm "${GPU_ARGS[@]}" --entrypoint /env/venv/bin/python \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONPATH=/rg \
  -e HUMAN_MOTION_DATA_DIR=/data/outputs/rl_train_req/_work/human_motion_data \
  -e ROBOTIC_GROUNDING_INTERMEDIATE_DIR=/data/outputs/rl_train_req/_work/intermediate \
  -v /usr1/e2s/retarget_env:/env \
  -v /usr1/e2s/video_to_data/reconstruction/modules:/workspace:ro \
  -v "$RG/source/robotic_grounding":/rg:ro \
  -v "$RG/scripts":/rgscripts:ro \
  -v /usr1/e2s/video_to_data/reconstruction/data/weights:/weights:ro \
  -v /usr1/e2s/v2d_submission_kit:/kit:ro \
  -v /usr1/e2s/v2d_challenge_data:/data \
  v2d_hamer "$@"
