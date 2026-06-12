"""Generate a morning summary of all sensitivity sweep results."""
import pickle
from pathlib import Path

BASE = Path('results/sensitivity')


def load(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return {'_error': str(e)}


def fmt_row(name, r):
    if '_error' in r:
        return f"  {name:20s}  ERROR: {r['_error']}"
    m = r['best_metrics']
    p = r['best_params']
    return (f"  {name:20s}  loss={r['best_value']:.3f}  "
            f"STN={m['r_stn']:5.1f}Hz  GPe={m['r_gpe']:5.1f}Hz  GPi={m['r_gpi']:5.1f}Hz  "
            f"betaGPe={m['beta_gpe']*100:4.1f}%  betaSTN={m['beta_stn']*100:4.1f}%  "
            f"| STN->GPe={p['g_stn_gpe_mult']:.2f}  GPe->STN={p['g_gpe_stn_mult']:.2f}  "
            f"STN->GPi={p['g_stn_gpi_mult']:.2f}  GPe->GPi={p['g_gpe_gpi_mult']:.2f}")


print("=" * 120)
print("SENSITIVITY SWEEP SUMMARY")
print("=" * 120)

for sweep_name in ['bounds', 'weights', 'seeds']:
    sweep_dir = BASE / sweep_name
    if not sweep_dir.exists():
        print(f"\n[{sweep_name}] directory missing")
        continue
    print(f"\n--- {sweep_name.upper()} ---")
    for pkl in sorted(sweep_dir.glob('*.pkl')):
        r = load(pkl)
        print(fmt_row(pkl.stem, r))

print()
print("=" * 120)
print("HEADLINE COMPARISONS")
print("=" * 120)

try:
    paper = load(BASE / 'bounds' / 'paper.pkl')
    print(f"\nPaper baseline synaptic multipliers (from bounds/paper.pkl):")
    p = paper['best_params']
    print(f"  STN->GPe: {p['g_stn_gpe_mult']:.3f}  (paper reports 4.13 / 1.87 = 2.21)")
    print(f"  GPe->STN: {p['g_gpe_stn_mult']:.3f}  (paper reports 0.11 / 0.99 = 0.11)")
    print(f"  STN->GPi: {p['g_stn_gpi_mult']:.3f}")
    print(f"  GPe->GPi: {p['g_gpe_gpi_mult']:.3f}")
except Exception as e:
    print(f"Could not load paper baseline: {e}")

try:
    paper = load(BASE / 'bounds' / 'paper.pkl')
    sym = load(BASE / 'bounds' / 'symmetric.pkl')
    print(f"\nBOUNDS sweep -- does asymmetric pattern survive symmetric bounds?")
    print(f"  Paper:     STN->GPe={paper['best_params']['g_stn_gpe_mult']:.2f}  "
          f"GPe->STN={paper['best_params']['g_gpe_stn_mult']:.2f}")
    print(f"  Symmetric: STN->GPe={sym['best_params']['g_stn_gpe_mult']:.2f}  "
          f"GPe->STN={sym['best_params']['g_gpe_stn_mult']:.2f}")
    print(f"  ==> If STN->GPe > 1 and GPe->STN < 1 under symmetric bounds,")
    print(f"      the pattern is data-driven, not bound-driven.")
except Exception as e:
    print(f"Could not run bounds comparison: {e}")

try:
    seed_files = sorted((BASE / 'seeds').glob('*.pkl'))
    print(f"\nSEEDS sweep -- variance across {len(seed_files)} CMA-ES seeds:")
    import statistics
    for key in ['g_stn_gpe_mult', 'g_gpe_stn_mult', 'g_stn_gpi_mult', 'g_gpe_gpi_mult']:
        vals = [load(p)['best_params'][key] for p in seed_files]
        print(f"  {key:20s}  mean={statistics.mean(vals):.3f}  "
              f"sd={statistics.stdev(vals):.3f}  range=[{min(vals):.3f}, {max(vals):.3f}]")
    losses = [load(p)['best_value'] for p in seed_files]
    print(f"  {'final_loss':20s}  mean={statistics.mean(losses):.3f}  "
          f"sd={statistics.stdev(losses):.3f}  range=[{min(losses):.3f}, {max(losses):.3f}]")
except Exception as e:
    print(f"Could not compute seed variance: {e}")

print("\n" + "=" * 120)
print("Summary complete. Review results/sensitivity/logs/master.log for run-level timing.")
print("=" * 120)
