#!/usr/bin/env bash
# As start_v2_queue.sh, for the other 18 Track 3 evaluation episodes (Ego2Sim v2 meshes + trajectories), split over both GPUs.
# Run as: setsid nohup sg e2sshare -c ./start_v2_all_queue.sh   (the containers need e2sshare as the primary group)
T=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0; S=$T/_scratch/v2; cd /usr1/e2s/retarget_env/s0
export PKG_DIR=/usr1/e2s/v2d_challenge_data/outputs/ego2sim_recon_v2_pkg FEAT_DIR=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/s0_features_v2 NO_COMPARE=1
EPS="3 5 6 8 10 14 16 19 20 25 28 30 32 33 35 36 37 38"
../run_in_env.sh --gpu 0 /env/s0/s0_features_v2.py $EPS > $S/features_all.log 2>&1 || { echo "features failed" >> $S/features_all.log; exit 1; }
for ep in $EPS; do [[ -f $FEAT_DIR/$(printf 'episode_%06d' $ep).npz ]] || { echo "missing features for ep$ep" >> $S/features_all.log; exit 1; }; done
setsid nohup ./s0_queue.sh 0 $S/jobs_all_gpu0.txt > $T/eval/queue_v2all_gpu0.out 2>&1 < /dev/null &
setsid nohup ./s0_queue.sh 1 $S/jobs_all_gpu1.txt > $T/eval/queue_v2all_gpu1.out 2>&1 < /dev/null &
