#!/usr/bin/env bash
T=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0; cd /usr1/e2s/retarget_env/s0
export SEQ_SUFFIX="" WORLDS=1024 SIM="sim.njmax=1024 sim.nconmax=512 termination.wrist_position.threshold=0.4 action.wrist_pos_clip=0.4"
setsid nohup ./s0_two_phase_v2.sh 18 0 ep18_s0 > $T/ep18_s0_driver.log 2>&1 < /dev/null &
setsid nohup ./s0_two_phase_v2.sh 18 1 ep18_s0.5_from_ep40_25M $T/ep18_s0.5_from_ep40_25M_p1_80M/init/state_2048.safetensors > $T/ep18_s0.5_driver.log 2>&1 < /dev/null &
setsid nohup /usr1/e2s/video_to_data/reconstruction/.venv/bin/python s0_monitor.py ep18_s0_p1_80M:0 ep18_s0_p2_ft_objonly:0 ep18_s0.5_from_ep40_25M_p1_80M:1 ep18_s0.5_from_ep40_25M_p2_ft_objonly:1 > $T/monitor_ep18.log 2>&1 < /dev/null &
