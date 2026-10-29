#!/usr/bin/env bash
# ep18 retrain with the fixes: reference-free relative action, no reconstructed-hand reward terms, video hand-ray guidance
# until 55M, relaxed object termination (25 cm / 1.5 rad), wrist termination 40 cm. S0 from scratch and S0.5 from ep40.
T=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0; cd /usr1/e2s/retarget_env/s0
export SEQ_SUFFIX="" WORLDS=1024 P1_CURRICULUM=tracking_80m_s0fix P2_CURRICULUM=object_only_16m_s0fix SIM="$(cat $T/s0fix_overrides.txt)"
setsid nohup ./s0_two_phase_v2.sh 18 0 ep18_s0fix > $T/ep18_s0fix_driver.log 2>&1 < /dev/null &
setsid nohup ./s0_two_phase_v2.sh 18 1 ep18_s0.5fix_from_ep40_25M $T/ep18_s0.5fix_from_ep40_25M_p1_80M/init/state_2048.safetensors > $T/ep18_s0.5fix_driver.log 2>&1 < /dev/null &
setsid nohup /usr1/e2s/video_to_data/reconstruction/.venv/bin/python s0_monitor.py ep18_s0fix_p1_80M:0 ep18_s0fix_p2_ft_objonly:0 ep18_s0.5fix_from_ep40_25M_p1_80M:1 ep18_s0.5fix_from_ep40_25M_p2_ft_objonly:1 > $T/monitor_ep18fix.log 2>&1 < /dev/null &
