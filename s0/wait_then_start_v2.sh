#!/usr/bin/env bash
# Start start_v2_queue.sh 60 s after no Blender process is left on either GPU (other users' renders), checking every 15 s.
# Run as: setsid nohup sg e2sshare -c ./wait_then_start_v2.sh   (the containers need e2sshare as the primary group)
S=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0/_scratch/v2; cd /usr1/e2s/retarget_env/s0
echo "[$(date '+%m-%d %H:%M')] waiting for blender to leave the GPUs" >> $S/wait.log
while nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -qi blender; do sleep 15; done
echo "[$(date '+%m-%d %H:%M:%S')] GPUs free of blender, waiting 60 s" >> $S/wait.log
sleep 60
echo "[$(date '+%m-%d %H:%M:%S')] starting start_v2_queue.sh" >> $S/wait.log
exec ./start_v2_queue.sh
