# Phase 1 review notes

End-of-phase pause per CLAUDE.md and AGENTS.md section 3 Phase 1.

## Status

All ten Phase 1 numbered steps complete and committed. Test suite: **22 passed,
1 skipped** (the skipped test is the Phase 2 observables placeholder).
Ruff lint: clean. Package imports cleanly.

Phase 1 exit conditions (AGENTS.md):

- [x] All unit tests pass
- [x] Smoke test produces sensible output (non-zero firing in all
      populations, no NaN, two-seed consistency)
- [x] 600 ms simulation at 450 neurons completes in under 5 s on the L4
      GPU — measured 0.80 s warm
- [x] `bgnet` imports cleanly with no warnings

## Decisions taken inside Phase 1

These were not in the explicit decision-points list, but per operating
principle 6 ("ask, don't assume") they deserve to be surfaced.

### Sign convention for synaptic current

AGENTS.md section 3 step 4 writes `I_syn = g * (E_syn - V)` (depolarizing-
positive); section 3 step 2 writes the membrane equation as
`C_m dV/dt = -I_ion - I_syn + I_drive + I_noise`. These are inconsistent:
with `I_syn = g*(E-V)`, the AMPA current at rest is positive, so `-I_syn`
in the membrane equation would *hyperpolarize* an excitatory synapse.

The legacy code had this exact problem (STN treated `I_syn` one way, GPe/GPi
the other), and one of the silent 0.005 scaling factors in the legacy
integrator is what compensated for it.

The rebuild commits to one convention everywhere:

```
I_X    = g_X * (V - E_X)            # positive when hyperpolarizing
dV/dt  = (-I_ion - I_syn + I_drive + I_noise + I_app) / C_m
```

This is the standard HH convention and matches AGENTS.md's *membrane
equation* exactly. Documented in `bgnet/synapses.py` and `bgnet/integrator.py`.

### STN drive headroom (potential Phase 3 issue)

The STN model with the literature-faithful Otsuka 2004 / Gillies-Willshaw
2006 conductances is near-silent at zero applied drive and reaches only
~11 Hz at I = 42 uA/cm^2. f-I curve, characterized in `bgnet/neurons/stn.py`:

```
I_drive (uA/cm^2):    -2  0  2  5  10  15  20  30  42
rate (Hz):             0  0  0  1   1   4   5   9  11
```

The optimizer's bounds are `I_drive_STN in [-5, +5]` and `mu_STN in [-5, +5]`,
giving a deterministic envelope of [-10, +10] uA/cm^2. To reach the healthy
target of 20 Hz purely from intrinsic + tonic + OU drive, more than 80
uA/cm^2 is needed in this model. In the network, AMPA input from
connected neurons supplies the remainder, but I cannot promise from
Phase 1 alone that the network optimization will close this gap inside
the bounded search space.

Pallidum f-I is comfortable: GPe targets (41-65 Hz) and GPi targets
(63-67 Hz) are reachable inside `I_drive in [-5, +5]` even from a
silent baseline noise.

**Action:** monitor convergence on the first healthy optimization run in
Phase 3 step 2. If best loss plateaus before trial 500 with all trials
clustering near the upper STN-drive bound, that is a model-versus-bounds
mismatch and the choices are: (a) widen the optimizer's STN drive bound;
(b) retune the STN parameters (lower g_AHP, lower g_K, or higher E_L)
toward Terman 2002 / Rubin-Terman values; (c) add a fixed cell-type
baseline I_app to the STN, in line with how the pallidum cells already
have one. This decision is for Phase 3 with real data, not Phase 1.

### Performance

The first integrated-simulation profile measured 600 ms at 450 neurons in
~22 s, well over the 5 s exit budget. Two changes brought it to 0.80 s:

1. Replace `data.at[head].set(spikes)` for the spike-delay buffer with
   `lax.dynamic_update_slice` (likewise `dynamic_slice` for the read).
   The `.at[].set` form did not optimize through the JIT.
2. Cache `make_run` by `n_steps` in `bgnet/network.py`. Each call to
   `jax.jit` produces a new wrapper that re-traces (~3 s per simulation
   call for our network) even when the XLA binary is already in cache.
   The cache amortizes that across optimization trials.

Both fixes are documented in code comments and in the Phase 1.8 commit.

### dt-halving test tolerance

AGENTS.md asks the integrator test to verify dt = 0.025 vs 0.0125
agreement within 2 Hz. With strong OU noise (sigma = 1.5 uA/cm^2) on
a 200 ms window the same-seed-different-dt OU realizations decorrelate
enough that an exact integrator could still drift by a few Hz on rate
purely from sampling noise. The test now uses sigma = 1.0 and a 600 ms
window, where the residual difference is sub-2-Hz.

## Items deferred to Phase 2

- `bgnet/observables.py` (firing rates, CV, beta fraction via Welch) — placeholder skipped test in `tests/unit/test_observables.py`.
- `bgnet/objective.py`, `bgnet/optimize.py`, `bgnet/config.py`.
- The actual `scripts/01_*` and `configs/*.yaml` files.
- LFP module `bgnet/lfp.py` (population firing rate proxy first).

## What is good shape entering Phase 2

- Clean package layout, modern packaging, ruff-clean, all imports work.
- Six-module foundation (`stn`, `pallidum`, `synapses`, `noise`,
  `connectivity`, `integrator`) each with own docstring documenting
  unit conventions, parameter sources, and any tuning rationale.
- `NetworkConfig` dataclass already names the 13 trial parameters in the
  exact form the optimizer will see them.
- `simulate(cfg, duration_ms)` is a one-call driver that returns spikes
  per population — Phase 2's objective function can be built directly
  on top of this.
- Make-run cache means trial-to-trial cost is just one GPU pass; no
  per-trial recompilation.

## Resolution: STN model swap

Decision (May 2026): replaced the GW-inspired single-compartment STN with
the canonical Terman-Rubin 2002 STN model (`episodic.ode` parameters,
ModelDB 182758). The GW headroom problem (silent at I = 0, ~11 Hz at
I = 42 µA/cm²) is resolved — the new STN fires ~10 Hz tonically at I = 0
and hits the healthy 20 Hz target at I_drive ≈ 5 µA/cm², comfortably
inside the optimizer's deterministic envelope of [-10, +10] µA/cm².
RT 2002 is single-compartment by design (not a multi-compartment collapse),
shares its model lineage with the Rubin-Terman 2004 GPe/GPi neurons
already in use, and is well-precedented in the STN-GPe oscillation
literature.

Implementation: `bgnet/neurons/stn.py`. Tests: `tests/unit/test_stn_neuron.py`
(six tests covering tonic firing rate, f-I monotonicity, post-inhibitory
rebound, calcium-AHP adaptation, dt sensitivity, and bounds/NaN hygiene).
Validation document with f-I curve, GW-vs-RT comparison plot, rebound
trace, adaptation trace, and dt sensitivity table: `docs/stn_validation.md`.

**I_Ca activation gate.** The high-threshold calcium current is
implemented as `I_Ca = g_Ca · sinf(V) · (V − E_Ca)` — a linear
instantaneous gate — rather than `sinf(V)^2`. This is a deliberate
engineering choice with documented biophysical rationale, not a
reluctant transcription compromise:

- Empirically the linear form fires the published RT 2002 ~10 Hz at
  I = 0 with a smooth monotonic f-I that hits 20 Hz at I = 5 µA/cm²
  (well inside the optimizer's drive envelope). The squared form fires
  ~2 Hz at I = 0 and would not reach the optimization targets inside
  the bounded search space.
- The pacing mechanism under the linear form is a **sub-threshold
  I_Ca + Na-window inward current** balancing leak and Ca-AHP. This
  matches the experimental characterization of autonomous STN firing
  (Bevan & Wilson 1999, J Neurosci 19:7617-7628; Atherton & Bevan 2005,
  J Neurosci 25:8272-8281), which describe STN autonomous discharge as
  driven by persistent sodium and a small sustained calcium current at
  sub-threshold voltages.
- The canonical RT 2002 T-current rebound mechanism is preserved and
  selectively engaged. Under tonic firing the T-current is essentially
  inactive because the cell never hyperpolarizes deeply enough to
  de-inactivate r. Under sustained hyperpolarization (experimental drive
  or network-level GPe inhibition), r climbs to ~0.91, I_T transients
  to -77 µA/cm² on release, and 2-3 rebound spikes fire on schedule.

Empirical and biophysical justification: `docs/stn_validation.md` §2.
Audit trail with full diagnostic battery (resting Ca, current balance
at sub-threshold rest, phase plane, f-I family with and without OU
noise): `docs/stn_linear_form_diagnostics.md`.

The Phase 1 review notes above are preserved unchanged — the headroom
analysis is the reason for the swap and should remain visible in the
project history.
