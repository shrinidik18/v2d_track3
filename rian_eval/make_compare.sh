#!/usr/bin/env bash
# Per evaluation episode: ego video (cam_a), Rian's reconstructed reference trajectory rendered in the simulator (kinematic, no
# physics), his best policy rollout (official-style explicit reset at frame 0, physics, no assistance), and a side-by-side.
#   make_compare.sh <gpu> <episodes...>
set -uo pipefail
GPU=$1; shift
O=/usr1/e2s/v2d_challenge_data/outputs/rian_eval_compare; RUNS=/home/rborah/v2d/runs; VID=/home/rborah/v2d/v2d_data/track_3/evaluation/videos/chunk-000
P=/home/rborah/v2d/vtd_pinned/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed
PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python
for EP in "$@"; do
  D=$O/ep$EP; mkdir -p $D
  # best checkpoint = lowest MPPE vs his reference across his eval runs
  read -r RUN STEP MPPE AUC < <($PY - <<PY
import glob
best=None
for r in sorted(glob.glob("$RUNS/e2s_ep${EP}_40M_eval*")):
    for l in open(r+"/eval_table.txt").read().splitlines()[1:]:
        f=l.split()
        if len(f)>=3 and (best is None or float(f[1])<best[2]): best=(r,int(f[0]),float(f[1]),float(f[2]))
print(*best)
PY
)
  T=$(awk -F, -v e=$EP '$1==e{print $3}' /home/rborah/v2d/share/ego2sim_eval_recon_v1/quality.csv); N=$((T*5/2))
  PARQ=$(dirname $(ls /home/rborah/v2d/ego2sim/train/ep$EP/processed/sequence_id=ep$EP/robot_name=sharpa_wave/*.parquet | head -1))
  read -r CP CT < <($PY /usr1/e2s/retarget_env/sub/cam_for_pkg.py $EP 0.5 0.25)
  printf "episode %s\nRian best checkpoint: %s/policy_%s.safetensors\nits MPPE / ADD-AUC vs his own reference: %s cm / %s\nreference parquet: %s\nframes: %s\n" $EP $RUN $STEP $MPPE $AUC $PARQ $T > $D/info.txt
  cp $VID/observation.images.ego_cam_a/episode_$(printf %06d $EP).mp4 $D/ego_video_cam_a.mp4
  DK="docker run --rm --runtime=nvidia --gpus device=$GPU -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics --entrypoint bash --user $(id -u):$(id -g) -e HOME=/tmp -e PYGLET_HEADLESS=1 -e PYTHONPATH=/fc/src -e FLASH_CHORD_CACHE_DIR=/cache -e WARP_CACHE_PATH=/cache/warp -v /usr1/e2s/retarget_env/flashchord_cache:/cache -v /usr1/e2s/retarget_env:/env:ro -v /usr1/e2s/video_to_data/robotic_grounding/flash_chord:/fc:ro -v /home/rborah/v2d/ego2sim:/home/rborah/v2d/ego2sim:ro -v $P:$P:ro -v $RUN:$RUN:ro -v $D:/out flashchord-cuda12-575-test:193382a1-20261018"
  VIEW="viewer.backend=mp4 viewer.num_frames=$N viewer.video_fps=50 viewer.width=1280 viewer.height=720 viewer.headless=true viewer.realtime=false 'viewer.camera_position=[$CP]' 'viewer.camera_target=[$CT]' markers.keypoints=false markers.contacts=false hydra.run.dir=/tmp/hydra"
  $DK -c "cd /tmp && python /env/replay_reference_video.py \"task.parquet='$PARQ'\" task.motion_speed=1.0 replay=kinematic action=residual_hand_pose_stiff collision=kinematic embodiment=sharpa_hands_stiff viewer.output_path=/out/rian_reference_trajectory.mp4 $VIEW" > $O/_logs/ref_ep$EP.log 2>&1
  $DK -c "cd /tmp && python /fc/scripts/view_policy.py evaluation.checkpoint=$RUN/policy_$STEP.safetensors viewer.output_path=/out/rian_policy_rollout.mp4 $VIEW" > $O/_logs/policy_ep$EP.log 2>&1
  # side-by-side, all retimed to the episode length (T frames at 20 Hz), 480 px high
  DUR=$($PY -c "print($T/20.0)")
  ffmpeg -loglevel error -y -i $D/ego_video_cam_a.mp4 -i $D/rian_reference_trajectory.mp4 -i $D/rian_policy_rollout.mp4 -filter_complex \
   "[0:v]scale=-2:480,fps=20,drawtext=text='ego video (cam_a)':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5[a];\
[1:v]scale=-2:480,fps=20,drawtext=text='Rian reference trajectory (kinematic)':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5[b];\
[2:v]scale=-2:480,fps=20,drawtext=text='Rian policy rollout (best ckpt $((STEP/1000000))M)':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5[c];[a][b][c]hstack=inputs=3" \
   -t $DUR -c:v libx264 -pix_fmt yuv420p $D/side_by_side.mp4 2>> $O/_logs/ffmpeg_ep$EP.log
  echo "ep$EP: $(ls $D | tr '\n' ' ')" >> $O/_logs/done.txt
done
