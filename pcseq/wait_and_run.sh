#!/bin/bash
# Starts the BiGRU fold once the cnxt_sev fold loop (PID $1) has exited, if the CPU smoke run finished cleanly.
#   PY=<python> RAPTOR_INPUT=<ds dir> nohup ./wait_and_run.sh <fold loop pid> > wait.log 2>&1 &
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
if ! grep -q "wrote SWA OOF" train_cnxt_seq_f1_smoke.log || grep -q Traceback train_cnxt_seq_f1_smoke.log; then
  echo "$(date +%H:%M) smoke run not clean: not starting"; exit 1
fi
echo "$(date +%H:%M) fold loop done; starting cnxt_seq_f1"
exec systemd-inhibit --what=sleep:idle --why="cnxt seq fold" ./run_cnxt_seq.sh 1
