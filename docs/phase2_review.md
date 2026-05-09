# Phase 2 review notes

End-of-phase pause per CLAUDE.md and AGENTS.md §3 Phase 2 (decision points
list says: "After the first real-sized smoke test (~50 trials), report
whether the constraint formulation appears to be working or whether
fallback to threshold-with-bonus is needed. Surface to user.").

## Status

All eight Phase 2 steps complete and committed. Test suite: **73 passed,
0 skipped**. Phase 2 added 48 unit tests across observables, objective,
config, results, and optimize helpers; the Phase 1 placeholder
`test_observables.py` is now real.

Phase 2 exit conditions (AGENTS.md):
- [x] All Phase 1 tests still pass
- [x] Smoke runs of scripts 01 and 02 produce valid results directories
- [x] Constraint handling works (after auto-fallback, see below)
- [x] A run can be reproduced from saved config + git SHA + hardware
      (every run dir contains config.yaml, optuna_study.db, results.pkl,
      and metadata.json with bgnet_version, git_sha, git_dirty,
      jax_devices, host info, wall-clock, and a structured log.txt)

## Constraint formulation: surfacing per the AGENTS.md decision point

**Result: automatic fallback to penalty mode.**

AGENTS.md §3 Phase 2 step 3 specified `CmaEsSampler(constraints_func=...)`
as the primary constraint approach. **Optuna 4.8.0's `CmaEsSampler` does
not accept a `constraints_func` argument** — only the multi-objective
samplers (`TPESampler`, `NSGAIISampler`) do. This was discovered on the
first smoke run.

The driver detects the missing kwarg via TypeError, logs a clear warning
citing the AGENTS.md §3 Phase 2 step 3 fallback authority, and switches
to the penalty formulation (`loss_with_beta_penalty`) automatically.
Both `constraint_mode_requested` and `constraint_mode_effective` are
recorded in `results.pkl["constraint_mode_*"]` so headline runs document
exactly what they used.

Per-trial `c_beta` and `feasible` user-attributes are still set on every
trial regardless of mode, so feasibility filtering at result-summary time
works identically across modes.

### Smoke run feasibility breakdown

100 trials × 200 ms simulation each, 450 neurons, dt = 0.025 ms:

| Smoke run | Constraint | n_complete | n_feasible | feasibility rate | wall-clock |
|---|---|---:|---:|---:|---:|
| `healthy_smoke` (β<0.05) | trivially satisfied | 100 | 100 | 100% | 57 s |
| `pd_smoke` (β>0.15) | hard | 100 | 0 | 0% | 57 s |

Per-trial cost ~0.5 s after JIT warm-up — well within the 1500-trial
headline-run budget (~13 min projected wall-clock + ~30 s compile).

### Why all PD-smoke trials are infeasible — flagging for review

Every PD-smoke trial reports `beta_stn = 0.000`, so the constraint
`c_beta = 0.15 - 0 = +0.15` is violated by every trial regardless of
parameters. Two factors contribute:

1. **Short analysis window.** PD smoke runs 200 ms simulation – 50 ms
   burn-in = 150 ms of analysis at 1 ms bins → 150 samples. Welch with
   50% overlap gives ~2 segments. Frequency resolution is 1000/150 ≈
   6.7 Hz, which is too coarse for the [8, 15] Hz band (band width 7 Hz).
   The PSD essentially cannot resolve beta at this scale.
2. **Network not yet oscillating in beta.** Even at the headline 400 ms
   simulation, 300 ms of analysis → 3.3 Hz frequency resolution — still
   coarse. And the linear-`I_Ca` STN we adopted in Phase 1.5 paces via
   sub-threshold I_Ca rather than T-current rebound, which may make
   STN-GPe loop oscillation harder to elicit.

This isn't a Phase 2 deliverable problem (the smoke tests do their job:
they exercise the wiring end-to-end). But it surfaces a question for
Phase 3:

**Decision point:** If the headline 1500-trial PD optimization at 400 ms
also finds 0% feasibility, the constraint isn't reachable inside the
optimizer's bounded search space — same class of problem as Phase 1's
GW-STN headroom issue, but for the network-level beta dynamic instead of
the cell-level f-I curve. Mitigations to consider before Phase 3:
- Lengthen the PD optimization simulation duration (e.g. 600-800 ms) to
  improve the [8, 15] Hz frequency resolution.
- Verify on a hand-tuned strong-beta config (large g_STN→GPe, small
  g_GPe→STN, etc.) that the network can produce STN beta > 0.15 in
  principle. If yes, the optimizer just needs more trials / longer sims.
  If no, the model can't exhibit the target dynamic.

I would not run the headline PD optimization without this check.

## Other Phase 2 decisions taken

### Bound order in CmaEsSampler

Optuna's CmaEsSampler accepts `x0` (initial mean dict) and a single
`sigma0` (in normalized [0, 1] space). AGENTS.md §4.8 asked for "Initial
sigma: 1/4 of bound range for each parameter" — in normalized space this
is `sigma0 = 0.25` (the same scalar applies to every parameter post-
normalization). Recorded in the optimize.py docstring.

### Optuna verbosity

Optuna emits an INFO-level log line per trial by default which made the
log.txt unreadable. Driver sets `optuna.logging.set_verbosity(WARNING)`
and emits its own structured progress lines every 25 trials with the
relevant metrics (rates, beta, feasibility, running best feasible).

### Storage format

Each run uses an isolated SQLite study (`optuna_study.db` inside the
run dir). This is the simplest path to reproducibility — the file is a
self-contained study and can be reopened with
`optuna.load_study(storage="sqlite:///<path>")` without any external state.

## What is in good shape entering Phase 3

- All 13 search parameters wire end-to-end from YAML → CMA-ES sampling
  → NetworkConfig → simulation → metrics → loss → optimizer feedback.
- `metrics_from_sim` is a single function consuming the simulator's
  output dict; nothing in the pipeline reads raw spike arrays except
  this one entry point.
- Auto-fallback for the constraint API means future Optuna versions
  that add `CmaEsSampler(constraints_func=...)` will be picked up
  automatically with no code change beyond removing the try/except.
- Run-dir layout is the unit of provenance: every figure or table that
  ever cites a number can point to one such directory.
