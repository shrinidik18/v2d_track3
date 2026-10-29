#!/usr/bin/env bash
# after_queue.sh <wait_pid> <gpu> <jobfile>: start a queue once the previous queue on that GPU has exited
while kill -0 $1 2>/dev/null; do sleep 30; done
exec ./s0_queue.sh $2 $3
