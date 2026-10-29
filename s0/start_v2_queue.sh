#!/usr/bin/env bash
# Train the latest recipe (as start_eval_queue.sh) on ep1 + ep15 with the Ego2Sim v2 reconstruction (meshes + trajectories, 2026-10-27).
# Features first (s0_features_v2.py), then one episode per GPU.
T=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0; S=$T/_scratch/v2; cd /usr1/e2s/retarget_env/s0
export PKG_DIR=/usr1/e2s/v2d_challenge_data/outputs/ego2sim_recon_v2_pkg FEAT_DIR=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/s0_features_v2 NO_COMPARE=1
[[ -f $FEAT_DIR/episode_000001.npz && -f $FEAT_DIR/episode_000015.npz ]] || \
  ../run_in_env.sh --gpu 0 /env/s0/s0_features_v2.py 1 15 > $S/features.log 2>&1 || { echo "features failed" >> $S/features.log; exit 1; }
setsid nohup ./s0_queue.sh 0 $S/jobs_gpu0.txt > $T/eval/queue_v2_gpu0.out 2>&1 < /dev/null &
setsid nohup ./s0_queue.sh 1 $S/jobs_gpu1.txt > $T/eval/queue_v2_gpu1.out 2>&1 < /dev/null &
