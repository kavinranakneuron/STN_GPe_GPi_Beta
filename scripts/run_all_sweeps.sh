#!/bin/bash
set -e

cd ~/jne-revision/submitted

source ~/work/bgnet-env/bin/activate

LOGDIR=results/sensitivity/logs
mkdir -p $LOGDIR

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOGDIR/master.log
}

run_sweep() {
    local sweep=$1
    local condition=$2
    local seed=$3
    local extra_args=$4

    local out=results/sensitivity/${sweep}/${condition}.pkl
    local log=$LOGDIR/${sweep}_${condition}.log

    log "START: ${sweep}/${condition} (seed=${seed})"
    python scripts/sweep_driver.py \
        --condition ${sweep}_${condition} \
        --output $out \
        --seed $seed \
        --n-trials 1000 \
        $extra_args \
        > $log 2>&1
    log "DONE: ${sweep}/${condition} -> $out"
}

log "==== Sensitivity sweep run started ===="

log "--- Bounds sweep ---"
run_sweep bounds paper 42 ""
run_sweep bounds symmetric 42 "--bounds-override scripts/configs/bounds/symmetric.json"

log "--- Weights sweep ---"
run_sweep weights paper 42 ""
run_sweep weights low_beta 42 "--weights-override scripts/configs/weights/low_beta.json"
run_sweep weights high_beta 42 "--weights-override scripts/configs/weights/high_beta.json"
run_sweep weights equal_weights 42 "--weights-override scripts/configs/weights/equal_weights.json"
run_sweep weights no_cv 42 "--weights-override scripts/configs/weights/no_cv.json"

log "--- Seeds sweep ---"
run_sweep seeds seed42 42 ""
run_sweep seeds seed1 1 ""
run_sweep seeds seed7 7 ""
run_sweep seeds seed123 123 ""
run_sweep seeds seed2024 2024 ""

log "==== All sweeps complete ===="

log "Generating summary..."
python scripts/summarize_sweeps.py 2>&1 | tee $LOGDIR/SUMMARY.txt
log "==== Done. Summary at $LOGDIR/SUMMARY.txt ===="
