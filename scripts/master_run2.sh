#!/bin/bash
# Overnight sweep 2 — remaining steps after A1. Priority order; independent
# steps fail into <step>_FAILED.txt and the chain CONTINUES (per Kavin's note).
# A1 (heterogeneity) and B3 (peak freq) are run separately before this.
cd ~/jne-revision/submitted || exit 1
source ~/work/bgnet-env/bin/activate
LOG=results/sensitivity2/logs
mkdir -p "$LOG"

# run_step <name> <logfile> <cmd...>
run_step () {
  local name="$1"; local logf="$2"; shift 2
  echo ">>>>> [$(date +%H:%M:%S)] START $name"
  ( "$@" ) > "$logf" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    echo ">>>>> [$(date +%H:%M:%S)] OK $name"
  else
    echo ">>>>> [$(date +%H:%M:%S)] FAILED $name (exit $rc) -> ${name}_FAILED.txt"
    { echo "STEP $name exited $rc at $(date)"; echo "---- log tail ----"; tail -40 "$logf"; } > "$LOG/${name}_FAILED.txt"
  fi
  return 0
}

# ---- Step 3: B1 scaling re-measurement ----
run_step B1_scaling "$LOG/B1_scaling.log" \
  python3 scripts/run_scaling_study.py
python3 - <<'PY' > "$LOG/B1_scaling_summary.txt" 2>&1 || true
import pickle
d = pickle.load(open('results/validation/scaling_study.pkl','rb'))
for label, r in d['results'].items():
    if 'error' not in r:
        print(f"{label:15s}  H={r['healthy']['run_time_s']:6.2f}s  PD={r['pd']['run_time_s']:6.2f}s  PD_beta={r['pd']['beta']['stn']*100:5.1f}%")
PY

# ---- Step 4: A2 connectivity density (3 conditions) ----
run_step A2_K_default "$LOG/A2_density_K_default.log" \
  python3 scripts/sweep_driver_density.py --condition K_default \
    --output results/sensitivity2/density/K_default.pkl --n-trials 1000
run_step A2_K_low "$LOG/A2_density_K_low.log" \
  python3 scripts/sweep_driver_density.py --condition K_low \
    --k-stn-gpe 7 --k-gpe-stn 7 --k-stn-gpi 15 --k-gpe-gpi 5 \
    --output results/sensitivity2/density/K_low.pkl --n-trials 1000
run_step A2_K_high "$LOG/A2_density_K_high.log" \
  python3 scripts/sweep_driver_density.py --condition K_high \
    --k-stn-gpe 22 --k-gpe-stn 21 --k-stn-gpi 45 --k-gpe-gpi 15 \
    --output results/sensitivity2/density/K_high.pkl --n-trials 1000

# ---- Step 5: C1 healthy seed sweep (5 seeds) ----
for s in 1 7 42 123 2024; do
  run_step C1_healthy_seed${s} "$LOG/C1_healthy_seed${s}.log" \
    python3 scripts/run_healthy_optimization.py --seed $s \
      --output results/sensitivity2/healthy_seeds/healthy_seed${s}.pkl \
      --n-trials 500
done

# ---- Step 6: B4 healthy STN miss analysis (no sim) ----
run_step B4_healthy_miss "$LOG/B4_healthy_miss.log" \
  python3 scripts/analyze_healthy_miss.py

# ---- Step 7: C2 OU tau sensitivity (4 conditions, nice-to-have) ----
for tau in 1 5 10 20; do
  run_step C2_tau${tau} "$LOG/C2_tau${tau}.log" \
    python3 scripts/sweep_driver_outau.py --condition tau${tau} --ou-tau-ms $tau \
      --output results/sensitivity2/ou_tau/tau${tau}.pkl --n-trials 1000
done

# ---- Step 8: B2 statistical validation (Welch edit already applied) ----
run_step B2_statval "$LOG/B2_statval.log" \
  python3 scripts/run_statistical_validation.py

# ---- Step 10: regenerate figures (edits already applied) ----
run_step figs_regen "$LOG/figs_regen.log" \
  python3 scripts/generate_figures.py
run_step figs_dbs "$LOG/figs_dbs.log" \
  python3 scripts/run_dbs_simulation.py

# Copy all final figures to clean dir
mkdir -p results/figures_revised
cp results/figures/*.png results/figures/*.pdf results/figures_revised/ 2>/dev/null || true
ls results/figures_revised/ > "$LOG/figures_revised_listing.txt" 2>&1 || true

# ---- Step 11: master summary (always run last) ----
python3 scripts/summarize_all.py > results/sensitivity2/MASTER_SUMMARY.txt 2>&1 || true

echo ">>>>> [$(date +%H:%M:%S)] MASTER RUN COMPLETE"
