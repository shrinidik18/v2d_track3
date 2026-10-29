#!/usr/bin/env bash
# Run jobs back to back on one GPU, each with its own monitor.   s0_queue.sh <gpu> <jobfile>
# jobfile lines: <single|two> <episode> <exp> <seq_suffix|-> <worlds> <overrides_file> <curriculum(s)> <voc_gate> <ema|->
GPU=$1; JOBS=$2; T=/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0; cd /usr1/e2s/retarget_env/s0
while read -r MODE EP EXP SUF W OVF CUR GATE EMA; do
  [[ -z "$MODE" || "$MODE" == \#* ]] && continue
  [[ "$SUF" == "-" ]] && SUF=""; [[ "$EMA" == "-" ]] && EMA=""
  mkdir -p $T/$EXP; echo "[$(date '+%m-%d %H:%M')] GPU$GPU start $EXP" >> $T/_summaries/queue.log
  if [[ "$MODE" == single ]]; then
    /usr1/e2s/video_to_data/reconstruction/.venv/bin/python s0_monitor_exp.py $EXP $GPU 1 > $T/$EXP/monitor.log 2>&1 &
    SEQ_SUFFIX="$SUF" WORLDS=$W OVERRIDES="$(cat $OVF)" CURRICULUM=$CUR S0_VOC_GATE=$GATE S0_EMA=$EMA ./s0_run_single.sh $EP $GPU $EXP > $T/$EXP/driver.log 2>&1
  else
    /usr1/e2s/video_to_data/reconstruction/.venv/bin/python s0_monitor_exp.py $EXP $GPU 2 > $T/$EXP/monitor.log 2>&1 &
    SEQ_SUFFIX="$SUF" WORLDS=$W OVERRIDES="$(cat $OVF)" P1_CURRICULUM=${CUR%%,*} P2_CURRICULUM=${CUR##*,} S0_VOC_GATE=$GATE S0_EMA=$EMA ./s0_run.sh $EP $GPU $EXP > $T/$EXP/driver.log 2>&1
  fi
  RC=$?; MON=$!
  if [[ $RC -ne 0 ]]; then pkill -f "s0_monitor_exp.py $EXP " ; fi       # a failed run never completes: stop its monitor
  wait; echo "[$(date '+%m-%d %H:%M')] GPU$GPU done $EXP (driver exit $RC)" >> $T/_summaries/queue.log
done < $JOBS
