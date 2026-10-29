#!/usr/bin/env bash
T=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0; cd /usr1/e2s/retarget_env/s0
setsid nohup ./s0_two_phase.sh 40 0 > $T/ep40_driver.log 2>&1 < /dev/null &
setsid nohup ./s0_two_phase.sh 41 1 > $T/ep41_driver.log 2>&1 < /dev/null &
setsid nohup /usr1/e2s/video_to_data/reconstruction/.venv/bin/python s0_monitor.py ep40_s0_p1_80M:0 ep40_s0_p2_ft_objonly:0 ep41_s0_p1_80M:1 ep41_s0_p2_ft_objonly:1 > $T/monitor.log 2>&1 < /dev/null &
