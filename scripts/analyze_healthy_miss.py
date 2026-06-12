"""Why did the healthy fit miss STN 20 Hz? Loss-landscape analysis.

Reads a healthy optimization pickle and asks: across all trials, how low can the
loss go while keeping STN firing near 20 Hz, and were such points feasible on the
other constraints? Prefers a pickle that contains per-trial data; the submitted
healthy_study.pkl saved only the best point, so we prefer the seed42 healthy
sweep (results/sensitivity2/healthy_seeds/) when present.
"""
import sys, os, pickle
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CANDIDATES = [
    'results/sensitivity2/healthy_seeds/healthy_seed42.pkl',
    'results/optimization/healthy_study.pkl',
]

src = next((p for p in CANDIDATES if os.path.exists(p)), None)
if src is None:
    print("No healthy pickle found in", CANDIDATES)
    sys.exit(0)
print(f"Source: {src}")

d = pickle.load(open(src, 'rb'))

# Report the achieved best regardless of trial availability.
bm = d.get('best_metrics', {})
bv = d.get('best_value', None)
tgt = d.get('targets', {})
print(f"Best loss: {bv}")
print(f"Best metrics: STN={bm.get('r_stn','?')} GPe={bm.get('r_gpe','?')} "
      f"GPi={bm.get('r_gpi','?')} betaGPe={bm.get('beta_gpe','?')}")
print(f"Targets: {tgt}")

# Locate per-trial records.
trials = d.get('all_trials') or d.get('trials')
if trials is None and hasattr(d, 'trials'):
    trials = [{'params': t.params, 'value': t.value,
               'user_attrs': dict(t.user_attrs)} for t in d.trials]

records = []
if trials:
    for t in trials:
        if t.get('value') is None or t['value'] >= 1e6:
            continue
        a = t.get('user_attrs', {})
        if 'r_stn' not in a:
            continue
        records.append((a['r_stn'], a['r_gpe'], a['r_gpi'],
                        a.get('beta_gpe', 0), t['value']))

if not records:
    note = (f"No per-trial metric records available in {src}. "
            f"The submitted healthy_study.pkl stored only the best point. "
            f"Best fit achieved STN={bm.get('r_stn','?')} Hz against a "
            f"{tgt.get('rate_stn','?')} Hz target (loss={bv}). "
            f"Loss-landscape scatter requires a sweep pickle with all_trials "
            f"(produced by the patched run_healthy_optimization.py / C1 sweep).")
    print("\n" + note)
    # Still emit a placeholder figure documenting the situation.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, note, ha='center', va='center', wrap=True,
            transform=ax.transAxes, fontsize=10)
    ax.axis('off')
    plt.savefig('results/sensitivity2/healthy_miss_analysis.png', dpi=200,
                bbox_inches='tight')
    print("Saved: results/sensitivity2/healthy_miss_analysis.png")
    sys.exit(0)

r = np.array(records)
target_stn = float(tgt.get('rate_stn', 20.0))
fit_stn = float(bm.get('r_stn', r[np.argmin(r[:, 4]), 0]))
print(f"\nn trials with metrics: {len(r)}")
near = (np.abs(r[:, 0] - target_stn) <= 2)
print(f"Trials with r_stn in [{target_stn-2:.0f},{target_stn+2:.0f}]: {near.sum()}")
if near.sum() > 0:
    print(f"  best loss among those: {r[near, 4].min():.4f}")
    print(f"  headline best loss:    {r[:, 4].min():.4f}")
    tgt_gpe = float(tgt.get('rate_gpe', 70.0))
    tgt_gpi = float(tgt.get('rate_gpi', 80.0))
    ok = near & (np.abs(r[:, 1] - tgt_gpe) <= 10) & (np.abs(r[:, 2] - tgt_gpi) <= 10) & (r[:, 3] <= 0.05)
    print(f"  feasible (rates near targets, beta_gpe<0.05): {ok.sum()}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(r[:, 0], r[:, 4], s=4, alpha=0.3)
ax.axvline(target_stn, color='r', linestyle='--', label=f'target {target_stn:.0f} Hz')
ax.axvline(fit_stn, color='g', linestyle='--', label=f'fit {fit_stn:.1f} Hz')
ax.set_xlabel('STN firing rate (Hz)')
ax.set_ylabel('Loss')
ax.set_yscale('log')
ax.legend()
ax.set_title('Healthy study: loss vs STN rate across trials')
plt.savefig('results/sensitivity2/healthy_miss_analysis.png', dpi=200, bbox_inches='tight')
print("Saved: results/sensitivity2/healthy_miss_analysis.png")
