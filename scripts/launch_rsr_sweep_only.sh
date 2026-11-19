#!/bin/bash
# Replaces the laptop rt_launcher (DEVIATIONS RT2): steps 1-2 of scripts/launch_rtgcn_after_rsr.sh unchanged
# (wait for RSR queues incl. helpers, then the resumable RSR sweep + completeness log). The RT-GCN confirm
# queues (step 3) now run on hyperion via scripts/hyperion/rtgcn_confirm.sbatch.
cd "/Users/hilmibukhori/Documents/PCGV - Graph Paper" || exit 1
L=outputs/rtgcn_v1/logs; mkdir -p "$L"
log(){ echo "$(date '+%F %T') $*" >> "$L/launcher.log"; }
wait_for(){ while screen -ls | grep -q "\.$1"; do sleep 60; done; }
q(){ screen -dmS "$1" bash -c "cd '$PWD' && caffeinate -i bash -c '$2' >> $3/$1.log 2>&1"; }
log "rsr_sweep_only launcher started (RT2); waiting for RSR queues"
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
log "RSR sweep done; laptop has no RT-GCN work (RT2)"
