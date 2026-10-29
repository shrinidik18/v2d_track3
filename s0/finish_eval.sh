#!/usr/bin/env bash
# After all eval trainings: (1) checkpoints.json with each episode's best checkpoint (64-world AUC against the cleaned reference),
# (2) the kit's official evaluator -> results/episode_XXXXXX.parquet, (3) meshes/episode_XXXXXX/<object>.glb from Rian's
# reconstructed meshes (same body frame as the poses), (4) ego | cleaned reference | best policy side-by-sides.
# Packing and upload need the producing code's GitHub commit URL and are left to tools/pack_submission.py / submit.
set -uo pipefail
T=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0/eval; SUB=/usr1/e2s/v2d_challenge_data/outputs/submission_track3; mkdir -p $SUB/meshes $SUB/results
O=/usr1/e2s/v2d_challenge_data/outputs/rian_eval_compare; PKG=$O/_cleaned/package; PY=/usr1/e2s/video_to_data/reconstruction/.venv/bin/python
for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 60; done; done
$PY - <<PY
import json, glob, re, os
T="$T"; SUB="$SUB"; ck={}; table=[]
for d in sorted(glob.glob(T+"/ep*_v1_*"), key=lambda x:int(re.search(r"/ep(\d+)_",x).group(1))):
    ep=int(re.search(r"/ep(\d+)_",d).group(1)); pts={}
    for f in glob.glob(d+"/p1_96M/eval/metrics_*.json"):
        pts[int(re.findall(r"metrics_(\d+)",f)[0])]=json.load(open(f))["metrics"]["add_auc"]
    if not pts: table.append((ep,None,None)); continue
    best=max(pts,key=pts.get); ck[str(ep)]=f"/runs/{os.path.basename(d)}/p1_96M/policy_{best}.safetensors"; table.append((ep,best,pts[best]))
json.dump(ck, open(SUB+"/checkpoints_container_paths.json","w"), indent=1)
open(SUB+"/checkpoint_selection.txt","w").write("episode, best checkpoint (env steps), 64-world ADD-AUC vs the cleaned reference\n"+"\n".join(f"{e}, {b}, {a}" for e,b,a in table)+"\n")
print(len(ck),"episodes with checkpoints")
PY
docker run --rm --runtime=nvidia --gpus device=0 --entrypoint bash --user $(id -u):$(id -g) -e HOME=/tmp -e PYTHONPATH=/fc/src:/env/s0 -e FLASH_CHORD_CACHE_DIR=/cache -e WARP_CACHE_PATH=/cache/warp \
  -v /usr1/e2s/retarget_env/flashchord_cache:/cache -v /usr1/e2s/retarget_env:/env:ro -v /usr1/e2s/video_to_data/robotic_grounding/flash_chord:/fc:ro -v /usr1/e2s/v2d_submission_kit:/kit:ro \
  -v $PKG:/pkg:ro -v /usr1/e2s/v2d_challenge_data/outputs/rl_train_req/s0_features_eval:/feat:ro -v $T:/runs:ro -v $SUB:/sub flashchord-cuda12-575-test:193382a1-20261018 \
  -c "cd /tmp && python /kit/examples/run_policy_evaluation.py --checkpoints /sub/checkpoints_container_paths.json --results /sub/results --runtime-root /fc" > $SUB/kit_evaluation.log 2>&1
$PY - <<PY
import json, shutil
from pathlib import Path
roster=json.load(open("/usr1/e2s/v2d_submission_kit/data/track_3_evaluation_objects.json"))["episodes"]
for ep, objs in roster.items():
    d=Path("$SUB/meshes")/f"episode_{int(ep):06d}"; d.mkdir(parents=True, exist_ok=True)
    for o in objs: shutil.copy2(f"$PKG/meshes/{o}.glb", d/f"{o}.glb")
print("meshes written")
PY
for f in $SUB/results/episode_*.parquet; do
  EP=$((10#$(basename $f .parquet | tr -dc 0-9))); D=$O/ep$EP; EXP=$(ls -d $T/ep${EP}_v1_* | head -1); STEP=$(grep -o "policy_[0-9]*" $SUB/checkpoints_container_paths.json | head -0; $PY -c "import json;print(json.load(open('$SUB/checkpoints_container_paths.json'))['$EP'].split('policy_')[1].split('.')[0])")
  cp $EXP/p1_96M/eval/policy_$STEP.mp4 $D/our_policy_rollout_best.mp4
  T2=$(grep frames: $D/info.txt | awk '{print $2}'); DUR=$($PY -c "print($T2/20.0)"); L="fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5:x=10:y=10"
  ffmpeg -loglevel error -y -i $D/ego_video_cam_a.mp4 -i $D/cleaned_reference_trajectory.mp4 -i $D/our_policy_rollout_best.mp4 -filter_complex \
    "[0:v]scale=-2:480,fps=20,drawtext=text='ego video (cam_a)':$L[a];[1:v]scale=-2:480,fps=20,drawtext=text='cleaned reference':$L[b];\
[2:v]scale=-2:480,fps=20,drawtext=text='our policy (best ckpt $((STEP/1000000))M)':$L[c];[a][b][c]hstack=inputs=3" -t $DUR -c:v libx264 -pix_fmt yuv420p $D/side_by_side_our_policy.mp4
done
echo "finished $(date)" > $SUB/FINISHED
