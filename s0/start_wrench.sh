#!/usr/bin/env bash
# ep18_v7_wrenchsupport (GPU 0, our hands) and ep41_v2_wrenchsupport (GPU 1, teammate hands), from scratch:
# S0 inputs, reference-free relative action, no reconstructed-hand rewards, video hand rays until 55M,
# required-wrench support reward (10, phase 1) + VOC gated on it (no weight cancellation), relaxed terminations;
# phase 2 = 16.8M object-tracking-only fine-tune.
T=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0; cd /usr1/e2s/retarget_env/s0
export S0_VOC_GATE=wrench P1_CURRICULUM=tracking_80m_wrench P2_CURRICULUM=object_only_16m_wrench
SEQ_SUFFIX="" WORLDS=1024 OVERRIDES="$(cat $T/_scratch/overrides_ep18_wrench.txt)" setsid nohup ./s0_run.sh 18 0 ep18_v7_wrenchsupport > $T/_scratch/driver_ep18_v7.log 2>&1 < /dev/null &
SEQ_SUFFIX="_teammate" WORLDS=2048 OVERRIDES="$(cat $T/_scratch/overrides_ep41_wrench.txt)" setsid nohup ./s0_run.sh 41 1 ep41_v2_wrenchsupport > $T/_scratch/driver_ep41_v2.log 2>&1 < /dev/null &
setsid nohup /usr1/e2s/video_to_data/reconstruction/.venv/bin/python s0_monitor.py ep18_v7_wrenchsupport/p1_80M:0 ep18_v7_wrenchsupport/p2_ft_objonly_16M:0 ep41_v2_wrenchsupport/p1_80M:1 ep41_v2_wrenchsupport/p2_ft_objonly_16M:1 > $T/_summaries/monitor_wrench.log 2>&1 < /dev/null &
