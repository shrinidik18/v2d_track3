#!/usr/bin/env bash
# ep18 retrain #3: S0-fix + object-trajectory contact reward (object_contact, weight 10 in phase 1) + contact-gated VOC.
T=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0; cd /usr1/e2s/retarget_env/s0
export S0_VOC_GATE=1 SEQ_SUFFIX="" WORLDS=1024 P1_CURRICULUM=tracking_80m_s0contact P2_CURRICULUM=object_only_16m_s0contact SIM="$(cat $T/s0contact_overrides.txt)"
setsid nohup ./s0_two_phase_v2.sh 18 0 ep18_s0contact > $T/ep18_s0contact_driver.log 2>&1 < /dev/null &
setsid nohup ./s0_two_phase_v2.sh 18 1 ep18_s0.5contact_from_ep40_25M $T/ep18_s0.5contact_from_ep40_25M_p1_80M/init/state_2048.safetensors > $T/ep18_s0.5contact_driver.log 2>&1 < /dev/null &
setsid nohup /usr1/e2s/video_to_data/reconstruction/.venv/bin/python s0_monitor.py ep18_s0contact_p1_80M:0 ep18_s0contact_p2_ft_objonly:0 ep18_s0.5contact_from_ep40_25M_p1_80M:1 ep18_s0.5contact_from_ep40_25M_p2_ft_objonly:1 > $T/monitor_ep18contact.log 2>&1 < /dev/null &
