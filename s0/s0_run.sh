#!/usr/bin/env bash
# One S0 experiment = one folder training_s0/<exp>/ holding both phases:
#   p1_80M/              initial 80M-step run (from scratch, or weights from <init_state>)
#   p2_ft_objonly_16M/   16.8M-step fine-tune, object-tracking reward only
#   *.log, driver.log, overrides.txt
# Naming: ep<n>_v<k>_<tag>  (k counts experiments on that episode in order).
#   s0_run.sh <episode> <gpu> <exp> [init_state]
#   env: SEQ_SUFFIX ('' our hands, '_teammate'), WORLDS, OVERRIDES (hydra args), P1_CURRICULUM, P2_CURRICULUM, S0_VOC_GATE
set -uo pipefail
EP=$1; GPU=$2; EXP=$3; INIT=${4:-}; cd /usr1/e2s/retarget_env/s0
T=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0; mkdir -p $T/$EXP; echo "${OVERRIDES:-}" > $T/$EXP/overrides.txt
COMMON="scene.world_count=${WORLDS:-1024} task.motion_speed=1.0 ${OVERRIDES:-} training.replay.capacity=2097152 training.replay.minimum_size=204800 training.checkpoint.save_interval_environment_steps=8388608"
P1="training.total_environment_steps=79998976 curriculum=${P1_CURRICULUM}"
[[ -n "$INIT" ]] && P1="$P1 training.resume=true training.checkpoint_path=/run_out/init/$(basename $INIT) training.checkpoint.load_optimizer=false"
P2="training.total_environment_steps=96776192 curriculum=${P2_CURRICULUM} training.resume=true training.checkpoint_path=/run_out/init/state_79998976.safetensors training.checkpoint.load_optimizer=false training.optimization.learning_rate_initial=0.00015 training.optimization.learning_rate_peak=0.00015 training.optimization.learning_rate_end=0.00003 training.optimization.learning_rate_schedule_updates=65536 training.exploration.target_sigma=0.12"
./s0_train.sh $EP $GPU $EXP/p1_80M $COMMON $P1 > $T/$EXP/p1_80M.log 2>&1 || { echo "phase 1 failed" >> $T/$EXP/p1_80M.log; exit 1; }
mkdir -p $T/$EXP/p2_ft_objonly_16M/init; cp $T/$EXP/p1_80M/state_79998976.safetensors $T/$EXP/p2_ft_objonly_16M/init/
./s0_train.sh $EP $GPU $EXP/p2_ft_objonly_16M $COMMON $P2 > $T/$EXP/p2_ft_objonly_16M.log 2>&1 || { echo "phase 2 failed" >> $T/$EXP/p2_ft_objonly_16M.log; exit 1; }
for i in $(seq 60); do [[ -f $T/$EXP/p2_ft_objonly_16M/eval/policy_96776192.mp4 ]] && break; sleep 20; done
for wc in 1 64; do RENDER=0 FINAL=1 ./s0_eval.sh $T/$EXP/p2_ft_objonly_16M 96776192 $GPU $wc; done
/usr1/e2s/video_to_data/reconstruction/.venv/bin/python s0_compare_episode.py $EP
