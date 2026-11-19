#!/bin/bash
# Scheduling only (no method change): idle cores after a NASDAQ/NYSE queue finishes take over the tail of
# rp_q6 (NYSE industry s4 + NYSE perms 78-98), in DESCENDING order so they meet rp_q6 (ascending) with
# at most one duplicated model. Same frozen commands; existing npz are skipped.
cd "/Users/hilmibukhori/Documents/PCGV - Graph Paper" || exit 1
RL=outputs/rsr_replica_v1/logs
R="$HOME/.venvs/rsr_arm/bin/python -u scripts/run_rsr_replica.py --mode confirm --market NYSE"
desc(){ c=""; for ((k=$1; k>=$2; k--)); do c="$c $R --graph perm --k $k:$((k+1)) --seeds 0;"; done; echo "$c"; }
h(){ screen -dmS "$1" bash -c "while screen -ls | grep -q '\.$2[[:space:]]'; do sleep 60; done; caffeinate -i bash -c '$3' >> $RL/$1.log 2>&1"; }
h rp_q7 rp_q1 "$(desc 98 90)"   # industry s4 left to rp_q6 (would have duplicated)
h rp_q8 rp_q2 "$(desc 89 84)"
h rp_q9 rp_q3 "$(desc 83 78)"
echo "$(date '+%F %T') helpers rp_q7/8/9 queued (NYSE tail of rp_q6, descending)" >> outputs/rtgcn_v1/logs/launcher.log
