#!/usr/bin/env bash
# Train the latest recipe on all 20 Track 3 evaluation episodes (cleaned references from Rian's reconstruction).
T=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0; S=$T/_scratch/eval; cd /usr1/e2s/retarget_env/s0
export PKG_DIR=/usr1/e2s/v2d_challenge_data/outputs/rian_eval_compare/_cleaned/package FEAT_DIR=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/s0_features_eval NO_COMPARE=1
setsid nohup ./s0_queue.sh 0 $S/jobs_gpu0.txt > $T/eval/queue_gpu0.out 2>&1 < /dev/null &
setsid nohup ./s0_queue.sh 1 $S/jobs_gpu1.txt > $T/eval/queue_gpu1.out 2>&1 < /dev/null &
