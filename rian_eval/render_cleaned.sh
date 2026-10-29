#!/usr/bin/env bash
# render_cleaned.sh <gpu> <episodes...>: kinematic render of the cleaned reference (same camera as the old one) and a
# side-by-side of ego video | old reference | cleaned reference into rian_eval_compare/ep<N>/
O=/usr1/e2s/v2d_challenge_data/outputs/rian_eval_compare; C=$O/_cleaned; GPU=$1; shift
PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python
for EP in "$@"; do
  D=$O/ep$EP; T=$(grep frames: $D/info.txt | awk '{print $2}'); N=$((T*5/2))
  read -r CP CT < <($PY /usr1/e2s/retarget_env/sub/cam_for_pkg.py $EP 0.5 0.25)
  docker run --rm --runtime=nvidia --gpus device=$GPU -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics --entrypoint bash --user $(id -u):$(id -g) \
    -e HOME=/tmp -e PYGLET_HEADLESS=1 -e PYTHONPATH=/fc/src -e FLASH_CHORD_CACHE_DIR=/cache -e WARP_CACHE_PATH=/cache/warp -v /usr1/e2s/retarget_env/flashchord_cache:/cache \
    -v /usr1/e2s/retarget_env:/env:ro -v /usr1/e2s/video_to_data/robotic_grounding/flash_chord:/fc:ro -v $C/package:/pkg:ro -v $D:/out flashchord-cuda12-575-test:193382a1-20261018 \
    -c "cd /tmp && python /env/replay_reference_video.py \"task.parquet='/pkg/sim_reference/processed/sequence_id=ep$EP/robot_name=sharpa_wave'\" task.motion_speed=1.0 replay=kinematic \
        action=residual_hand_pose_stiff collision=kinematic embodiment=sharpa_hands_stiff viewer.backend=mp4 viewer.output_path=/out/cleaned_reference_trajectory.mp4 viewer.num_frames=$N \
        viewer.video_fps=50 viewer.width=1280 viewer.height=720 viewer.headless=true viewer.realtime=false 'viewer.camera_position=[$CP]' 'viewer.camera_target=[$CT]' \
        markers.keypoints=false markers.contacts=false hydra.run.dir=/tmp/hydra" > $O/_logs/cleaned_ep$EP.log 2>&1
  DUR=$($PY -c "print($T/20.0)"); L="fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5:x=10:y=10"
  ffmpeg -loglevel error -y -i $D/ego_video_cam_a.mp4 -i $D/rian_reference_trajectory.mp4 -i $D/cleaned_reference_trajectory.mp4 -filter_complex \
    "[0:v]scale=-2:480,fps=20,drawtext=text='ego video (cam_a)':$L[a];[1:v]scale=-2:480,fps=20,drawtext=text='Rian reference (original)':$L[b];\
[2:v]scale=-2:480,fps=20,drawtext=text='cleaned reference (spin, flips, spawn fixed)':$L[c];[a][b][c]hstack=inputs=3" \
    -t $DUR -c:v libx264 -pix_fmt yuv420p $D/side_by_side_cleaned.mp4 2>> $O/_logs/ffmpeg_cleaned_ep$EP.log
  cp $C/ep${EP}_report.json $D/cleaning_report.json
  echo "ep$EP $(ls $D/*cleaned* | wc -l)" >> $O/_logs/cleaned_done.txt
done
