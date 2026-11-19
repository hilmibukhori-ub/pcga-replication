#!/bin/bash
# Robust orchestration (PREREG_RTGCN.md §11; DEVIATIONS RT1):
#  1. wait until the RSR replication queues (screens rp_q*) are gone
#  2. RSR sweep: re-run the frozen RSR queue commands. Finished models are skipped, so this only fills
#     models that failed or are missing. Then wait.
#  3. start the 6 RT-GCN queues. Each runs its command list twice; the second pass fills any model
#     that errored. Per-epoch checkpoints make interrupted models resume bit-identically.
#  4. write a completeness report.
cd "/Users/hilmibukhori/Documents/PCGV - Graph Paper" || exit 1
L=outputs/rtgcn_v1/logs; mkdir -p "$L"
log(){ echo "$(date '+%F %T') $*" >> "$L/launcher.log"; }
wait_for(){ while screen -ls | grep -q "\.$1"; do sleep 60; done; }
q(){ screen -dmS "$1" bash -c "cd '$PWD' && caffeinate -i bash -c '$2' >> $3/$1.log 2>&1"; }

log "launcher started; waiting for RSR queues"
wait_for rp_q
log "RSR queues finished"

# ---- 2. RSR sweep (same commands as the original launch; existing npz are skipped)
RL=outputs/rsr_replica_v1/logs; mkdir -p "$RL"
R="$HOME/.venvs/rsr_arm/bin/python -u scripts/run_rsr_replica.py --mode confirm"
q rp_sweep1 "$R --market NASDAQ --graph real --seeds 0:20; $R --market NASDAQ --graph perm --k 0:50 --seeds 0" $RL
q rp_sweep2 "$R --market NASDAQ --graph none --seeds 0:20; $R --market NASDAQ --relation sector_industry --graph real --seeds 0:5; $R --market NASDAQ --graph perm --k 50:99 --seeds 0" $RL
q rp_sweep3 "$R --market NYSE --graph real --seeds 0:10; $R --market NYSE --graph perm --k 0:25 --seeds 0" $RL
q rp_sweep4 "$R --market NYSE --graph real --seeds 10:20; $R --market NYSE --graph perm --k 25:50 --seeds 0" $RL
q rp_sweep5 "$R --market NYSE --graph none --seeds 0:10; $R --market NYSE --graph perm --k 50:75 --seeds 0" $RL
q rp_sweep6 "$R --market NYSE --graph none --seeds 10:20; $R --market NYSE --relation sector_industry --graph real --seeds 0:5; $R --market NYSE --graph perm --k 75:99 --seeds 0" $RL
wait_for rp_sweep
O=outputs/rsr_replica_v1
for m in NASDAQ NYSE; do
  log "RSR $m: real $(ls $O/${m}_wikidata_real_s*.npz 2>/dev/null | wc -l)/20 none $(ls $O/${m}_wikidata_none_s*.npz 2>/dev/null | wc -l)/20 ind $(ls $O/${m}_sector_industry_real_s*.npz 2>/dev/null | wc -l)/5 perm $(ls $O/${m}_wikidata_perm*_s0.npz 2>/dev/null | wc -l)/99"
done

# ---- 3. RT-GCN queues (each command list twice)
T="$HOME/.venvs/rsr_arm/bin/python -u scripts/run_rtgcn_replica.py"
c1="$T --mode val --cfg paper --graph real --seeds 0:10; $T --mode val --cfg paper --graph perm --k 0:6 --seeds 0"
c2="$T --mode val --cfg paper --graph real --seeds 10:20; $T --mode val --cfg paper --graph perm --k 6:12 --seeds 0"
c3="$T --mode val --cfg paper --graph none --seeds 0:10; $T --mode val --cfg paper --graph perm --k 12:19 --seeds 0"
c4="$T --mode val --cfg paper --graph none --seeds 10:20; $T --mode val --cfg paper --graph ind --seeds 0:5"
c5="$T --mode pub --cfg paper --graph real --seeds 0:10; $T --mode val --cfg released --graph none --seeds 5:10"
c6="$T --mode val --cfg released --graph real --seeds 0:10; $T --mode val --cfg released --graph none --seeds 0:5"
q rt_q1 "$c1; $c1" $L; q rt_q2 "$c2; $c2" $L; q rt_q3 "$c3; $c3" $L
q rt_q4 "$c4; $c4" $L; q rt_q5 "$c5; $c5" $L; q rt_q6 "$c6; $c6" $L
log "RT-GCN queues launched"
wait_for rt_q

# ---- 4. completeness report
V=outputs/rtgcn_v1
n(){ ls $V/NASDAQ_$1_s*.npz 2>/dev/null | wc -l | tr -d ' '; }
log "RT-GCN done: val/paper real $(n val_paper_real)/20 none $(n val_paper_none)/20 ind $(n val_paper_ind)/5 perm $(ls $V/NASDAQ_val_paper_perm*_s0.npz 2>/dev/null | wc -l | tr -d ' ')/19 | pub/paper real $(n pub_paper_real)/10 | val/released real $(n val_released_real)/10 none $(n val_released_none)/10 | leftover ckpts $(ls $V/*.ckpt.pt 2>/dev/null | wc -l | tr -d ' ') | errors $(grep -c '^ERROR' $L/rt_q*.log | awk -F: '{s+=$2} END {print s}')"
