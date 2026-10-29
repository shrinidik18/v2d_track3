#!/usr/bin/env bash
# S0 two-phase recipe, configurable.   s0_two_phase_v2.sh <episode> <gpu> <run_name> [init_state]
#   env: SEQ_SUFFIX (hand source; '' = our hands), WORLDS (default 1024), SIM (extra sim args), P1_CURRICULUM
#   phase 1: 80M steps from scratch, or from <init_state> (weights only: fresh optimizer, step counter ~0)
#   phase 2: +16.8M object-tracking only; then official-style evaluation of the final checkpoint
set -uo pipefail
EP=$1; GPU=$2; NAME=$3; INIT=${4:-}; cd /usr1/e2s/retarget_env/s0
T=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0
COMMON="scene.world_count=${WORLDS:-1024} task.motion_speed=1.0 ${SIM:-} training.replay.capacity=2097152 training.replay.minimum_size=204800 training.checkpoint.save_interval_environment_steps=8388608"
P1="training.total_environment_steps=79998976 curriculum=${P1_CURRICULUM:-tracking_80m_handoff55}"
[[ -n "$INIT" ]] && P1="$P1 training.resume=true training.checkpoint_path=/run_out/init/$(basename $INIT) training.checkpoint.load_optimizer=false"
P2="training.total_environment_steps=96776192 curriculum=${P2_CURRICULUM:-object_only_16m} training.resume=true training.checkpoint_path=/run_out/init/state_79998976.safetensors training.checkpoint.load_optimizer=false training.optimization.learning_rate_initial=0.00015 training.optimization.learning_rate_peak=0.00015 training.optimization.learning_rate_end=0.00003 training.optimization.learning_rate_schedule_updates=65536 training.exploration.target_sigma=0.12"
./s0_train.sh $EP $GPU ${NAME}_p1_80M $COMMON $P1 > $T/${NAME}_p1_80M.log 2>&1 || { echo "phase 1 failed" >> $T/${NAME}_p1_80M.log; exit 1; }
mkdir -p $T/${NAME}_p2_ft_objonly/init; cp $T/${NAME}_p1_80M/state_79998976.safetensors $T/${NAME}_p2_ft_objonly/init/
./s0_train.sh $EP $GPU ${NAME}_p2_ft_objonly $COMMON $P2 > $T/${NAME}_p2_ft_objonly.log 2>&1 || { echo "phase 2 failed" >> $T/${NAME}_p2_ft_objonly.log; exit 1; }
for i in $(seq 60); do [[ -f $T/${NAME}_p2_ft_objonly/eval/policy_96776192.mp4 ]] && break; sleep 20; done
for wc in 1 64; do RENDER=0 FINAL=1 ./s0_eval.sh $T/${NAME}_p2_ft_objonly 96776192 $GPU $wc; done
/usr1/e2s/video_to_data/reconstruction/.venv/bin/python s0_compare_ep18.py
