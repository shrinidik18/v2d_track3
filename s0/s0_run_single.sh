#!/usr/bin/env bash
# Single-phase S0 experiment: training_s0/<exp>/p1_96M/ (+ p1_96M.log, driver.log, overrides.txt), then official evaluation
# of the final and the best checkpoint and the per-episode comparison.
#   s0_run_single.sh <episode> <gpu> <exp>
#   env: SEQ_SUFFIX, WORLDS, OVERRIDES, CURRICULUM, TOTAL (default 96776192), CKPT (default 4194304), S0_VOC_GATE, S0_EMA
set -uo pipefail
EP=$1; GPU=$2; EXP=$3; cd /usr1/e2s/retarget_env/s0
T=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0; mkdir -p $T/$EXP; echo "${OVERRIDES:-}" > $T/$EXP/overrides.txt
[[ -n "${PKG_DIR:-}" && -z "${PARQUET:-}" ]] && export PARQUET=/pkg/sim_reference/processed/sequence_id=ep$EP/robot_name=sharpa_wave
TOTAL=${TOTAL:-96776192}; PH=p1_$((TOTAL/1000000))M
./s0_train.sh $EP $GPU $EXP/$PH scene.world_count=${WORLDS:-1024} task.motion_speed=1.0 ${OVERRIDES:-} training.replay.capacity=2097152 \
  training.replay.minimum_size=204800 training.checkpoint.save_interval_environment_steps=${CKPT:-4194304} \
  training.total_environment_steps=$TOTAL curriculum=$CURRICULUM > $T/$EXP/$PH.log 2>&1 || { echo "training failed" >> $T/$EXP/$PH.log; exit 1; }
for i in $(seq 90); do [[ -f $T/$EXP/$PH/eval/policy_$TOTAL.mp4 ]] && break; sleep 20; done
BEST=$(/usr1/e2s/video_to_data/reconstruction/.venv/bin/python -c "
import json,glob,re; p=[(json.load(open(f))['metrics']['add_auc'],int(re.findall(r'metrics_(\d+)',f)[0])) for f in glob.glob('$T/$EXP/$PH/eval/metrics_*.json')]; print(max(p)[1])")
for s in $TOTAL $BEST; do for wc in 1 64; do RENDER=0 FINAL=1 ./s0_eval.sh $T/$EXP/$PH $s $GPU $wc; done; done
echo "best 64-world checkpoint: $BEST" > $T/$EXP/best_checkpoint.txt
[[ -n "${NO_COMPARE:-}" ]] || /usr1/e2s/video_to_data/reconstruction/.venv/bin/python s0_compare_episode.py $EP
