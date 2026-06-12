import pickle, json, glob, os, numpy as np, statistics as st

print("="*78)
print("JNE-110355 OVERNIGHT SWEEP 2 — MASTER SUMMARY")
print("="*78)

# B3 peak frequency
try:
    pf = json.load(open('results/sensitivity2/peak_frequencies.json'))
    print(f"\n[B3] Peak STN PSD frequency (PD):")
    print(f"  Figure pipeline (500-sub, ~20k nperseg, {pf['pd']['pipeline_figure']['resolution_hz']:.1f} Hz res): {pf['pd']['pipeline_figure']['peak_hz']:.1f} Hz")
    print(f"  Metric pipeline (full pop, 8k nperseg,  {pf['pd']['pipeline_metric']['resolution_hz']:.1f} Hz res): {pf['pd']['pipeline_metric']['peak_hz']:.1f} Hz")
    print(f"  (Healthy: figure {pf['healthy']['pipeline_figure']['peak_hz']:.1f} Hz, metric {pf['healthy']['pipeline_metric']['peak_hz']:.1f} Hz)")
except Exception as e:
    print(f"[B3] FAILED: {e}")

# A1 heterogeneity
print(f"\n[A1] Heterogeneity (10%):")
try:
    d = pickle.load(open('results/sensitivity2/heterogeneity/het010.pkl','rb'))
    p, m = d['best_params'], d['best_metrics']
    print(f"  loss={d['best_value']:.3f} STN={m['r_stn']:.1f} GPe={m['r_gpe']:.1f} GPi={m['r_gpi']:.1f} "
          f"betaGPe={m['beta_gpe']*100:.1f}% | STN->GPe={p['g_stn_gpe_mult']:.2f} GPe->STN={p['g_gpe_stn_mult']:.2f}")
except Exception as e:
    print(f"  SKIPPED/FAILED: {e}")

# B1 scaling re-measurement
print(f"\n[B1] Scaling re-measurement:")
try:
    sc = pickle.load(open('results/validation/scaling_study.pkl','rb'))
    for label, r in sc['results'].items():
        if 'error' not in r:
            print(f"  {label:12s}  H={r['healthy']['run_time_s']:5.2f}s  PD={r['pd']['run_time_s']:5.2f}s  "
                  f"PD STN beta={r['pd']['beta']['stn']*100:.1f}%")
except Exception as e:
    print(f"  FAILED: {e}")

# A2 density
print(f"\n[A2] Connectivity density:")
for cond in ['K_default','K_low','K_high']:
    try:
        d = pickle.load(open(f'results/sensitivity2/density/{cond}.pkl','rb'))
        p, m = d['best_params'], d['best_metrics']
        print(f"  {cond:10s} loss={d['best_value']:.3f} STN={m['r_stn']:.1f} GPe={m['r_gpe']:.1f} "
              f"betaGPe={m['beta_gpe']*100:.1f}% | SG={p['g_stn_gpe_mult']:.2f} GS={p['g_gpe_stn_mult']:.2f}")
    except Exception as e:
        print(f"  {cond}: SKIPPED/FAILED")

# C1 healthy seeds
print(f"\n[C1] Healthy seeds (n=5):")
try:
    hfiles = sorted(glob.glob('results/sensitivity2/healthy_seeds/healthy_seed*.pkl'))
    for f in hfiles:
        d = pickle.load(open(f,'rb'))
        if 'best_params' in d:
            p, m = d['best_params'], d.get('best_metrics', {})
            print(f"  {os.path.basename(f)}: loss={d.get('best_value','?'):.3f} "
                  f"STN={m.get('r_stn',float('nan')):.1f} GPe={m.get('r_gpe',float('nan')):.1f} "
                  f"GPi={m.get('r_gpi',float('nan')):.1f} betaGPe={m.get('beta_gpe',float('nan'))*100:.1f}% | "
                  f"SG={p['g_stn_gpe_mult']:.2f} GS={p['g_gpe_stn_mult']:.2f} "
                  f"SGi={p['g_stn_gpi_mult']:.2f} GGi={p['g_gpe_gpi_mult']:.2f}")
        else:
            print(f"  {os.path.basename(f)}: structure={list(d.keys())[:8]}")
    # multiplier median (range) across seeds
    if hfiles:
        acc = {k: [] for k in ['g_stn_gpe_mult','g_gpe_stn_mult','g_stn_gpi_mult','g_gpe_gpi_mult']}
        for f in hfiles:
            p = pickle.load(open(f,'rb'))['best_params']
            for k in acc: acc[k].append(p[k])
        print("  --- multiplier median (min-max) across seeds ---")
        for k,v in acc.items():
            print(f"    {k:16s}: {np.median(v):.2f} ({min(v):.2f}-{max(v):.2f})")
except Exception as e:
    print(f"  FAILED: {e}")

# C2 OU tau
print(f"\n[C2] OU tau sensitivity:")
for tau in [1, 5, 10, 20]:
    try:
        d = pickle.load(open(f'results/sensitivity2/ou_tau/tau{tau}.pkl','rb'))
        p, m = d['best_params'], d['best_metrics']
        print(f"  tau={tau:3d}ms  loss={d['best_value']:.3f} STN={m['r_stn']:.1f} GPe={m['r_gpe']:.1f} "
              f"betaGPe={m['beta_gpe']*100:.1f}% | SG={p['g_stn_gpe_mult']:.2f} GS={p['g_gpe_stn_mult']:.2f}")
    except Exception as e:
        print(f"  tau={tau}: SKIPPED/FAILED")

# B2 stat validation already prints its own summary to log
print(f"\n[B2] Statistical validation (Welch's t-test): see results/sensitivity2/logs/B2_statval.log")
print(f"     and results/figures/fig6_statistical_validation.{{png,pdf}}")

# B4 healthy miss
print(f"\n[B4] Healthy STN 20 Hz miss analysis: see results/sensitivity2/logs/B4_healthy_miss.log")
print(f"     and results/sensitivity2/healthy_miss_analysis.png")

print(f"\n[FIGS] Revised figures in results/figures_revised/")
print("="*78)
