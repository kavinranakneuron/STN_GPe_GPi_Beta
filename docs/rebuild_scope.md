# Rebuild Scope: JNE Revision of GPU-Accelerated Basal Ganglia Framework

**Document type:** Internal planning document
**Author/PI:** Kavineshvar Ranak Nakkeeran (Functional Neurosurgery Lab, Johns Hopkins)
**Manuscript:** JNE-110355, *GPU-Accelerated Optimization Investigates Synaptic Reorganization Underlying Pathological Beta Oscillations in a Basal Ganglia Network Model*
**Decision letter:** Major revisions, due 18-Jun-2026
**Status:** Pre-build planning complete; awaiting implementation

---

## 1. Where we are

### 1.1 The submitted manuscript and its critique

The paper as submitted presents a JAX/GPU-based spiking network model of the STN-GPe-GPi circuit, optimized via CMA-ES within Optuna to reproduce primate basal ganglia firing statistics in healthy and Parkinsonian states. The headline results are: (i) a 490× speedup over a CPU reference implementation; (ii) discovery of an asymmetric synaptic reorganization pattern (STN→GPe excitation up 2.21×, GPe→STN inhibition collapsed to 0.11×) that produces robust beta oscillations; (iii) scale invariance across 450–45,000 neurons; and (iv) qualitative DBS suppression of pathological beta.

Reviewers raised five categories of concern:

- **Reviewer 1:** the reported beta peak is at ~30 Hz, not 20 Hz as repeatedly stated; the I_drive term is undefined until late in the methods; the OU mean parameter is described as optimized but never appears in the parameter list; "known directions" of dopamine-dependent plasticity overstates the literature; robustness to bound choices is not characterized; CV is missing from the objective-function box in Figure 1.
- **Reviewer 2:** the relative contributions of platform development, optimization methodology, and applied results are unclear; CMA-ES initialization, stopping criteria, and stochasticity handling are underspecified; loss-function weights (5.0, 0.2, 15.0) lack justification or sensitivity analysis; fixed-indegree connectivity without anatomical specificity is a strong simplification that needs more discussion; the OU process choice (τ=5 ms) is not justified.
- **Editor-in-Chief:** the simplified model (no cortex or striatum) limits the biological conclusions one can draw about beta generation. References should expand to full author lists for ≤10 authors; figure and table captions need to define all abbreviations.

### 1.2 What an audit of the codebase revealed

A complete audit of the submitted codebase identified discrepancies that go beyond the reviewers' comments:

- **Saved optimization results do not match the manuscript.** `healthy_study.pkl` contains only 6 parameters, with no synaptic multipliers; the headline healthy multipliers (1.87×, 1.00×, 1.83×, 0.69×) cannot be recovered from any saved artifact. `parkinsonian_study.pkl` has g_GPe→STN multiplier 0.76, while the paper and figures use 0.11 — qualitatively different network behavior.
- **Statistical-validation pickles disagree with the paper's Table 5.** The saved pickle shows healthy STN beta ≈ 2.7%, PD STN beta ≈ 1.3%; the paper reports 0.38% and 49.33%. The figure was generated from a different run than the pickle.
- **Timestep sensitivity is hidden.** The saved sensitivity pickle uses a different parameter set than the headline run, and shows STN beta drifting from 2% → 10% → 25% as dt decreases — a result the paper does not report.
- **The "490× speedup" is JAX-jit-on-GPU vs. JAX-eager-in-Python-loop, not vs. the `numpy_baseline/` package.** That package uses entirely different models (AdEx vs. Rubin-Terman, single-exponential vs. double-exponential synapses, fixed-probability vs. fixed-indegree connectivity) and has never been used to produce any number in the paper.
- **The `integrator.py` contains undocumented scaling factors** — synaptic currents into GPe/GPi are silently scaled by 0.005, noise currents by 0.01. These compensate for an effective unit mismatch between Gillies-Willshaw STN and Rubin-Terman GPe/GPi conventions but are not mentioned in the manuscript.
- **The optimization objective targets GPe beta** but the paper reports STN beta as the headline result.
- **The OU mean is documented as optimized but is in fact fixed** at build defaults (1.8 µA/cm² for STN, 0 elsewhere); only the σ is searched.
- **Healthy LFP traces show ~22 mV monotonic drift over 300 ms** in the membrane-voltage proxy, inflating broadband power and depressing reported beta fraction.

These issues are individually fixable, but their pattern — saved artifacts inconsistent with reported numbers, inherited fudge factors, hidden sensitivity results, optimization target mismatched with reported target — indicates that point-fixes will not produce a defensible revision. The framework needs to be rebuilt cleanly.

### 1.3 Time and resource constraints

Six weeks to revision deadline. Single L4 GPU on Google Cloud (g2-standard-8 instance, ~$0.84/hour). Parallelization possible within memory limits. PI is also writing a downstream Parkinson's biomarker / patient-fitting paper that will build on this framework; robustness here is consequential beyond the revision itself.

---

## 2. What we are doing — and how it differs from the original

### 2.1 Reframing the contribution

The original paper presented its biological finding (asymmetric STN-GPe disruption produces beta) as a primary scientific contribution alongside the methodological one. The EIC's cortex-and-striatum critique landed because the paper claimed mechanistic insight into a phenomenon (beta generation) that plausibly involves circuits the model does not include.

The revised paper reframes the contribution as **methodological**. The framework itself — GPU-accelerated spiking network simulation in JAX, two-stage CMA-ES optimization within Optuna, fixed-indegree connectivity for scale invariance, and a configurable objective combining electrophysiological targets — is the deliverable. The STN-GPe-GPi work is a *worked demonstration* of the framework operating on one biological hypothesis among many, not a novel biological claim about the cause of beta. Under this framing, the EIC's concern is acknowledged as a scope statement: the framework is agnostic to which hypothesis it tests, and the worked example is constrained to STN-GPe-GPi by deliberate methodological choice rather than by claim of biological completeness.

This framing fits the established genre of methods-first papers in computational neuroscience — CBGTPy (Vich et al., 2023), PymoNNto/PymoNNtorch (Vieth et al., 2024), and the automated parameter tuning framework of Carlson et al. (2014) are all conceptual relatives. JNE's stated scope explicitly includes "theoretical and computational neuroscience" and "neural circuits: artificial & biological," and there is no hard length limit; framework papers in this space routinely run 25-35 pages with extensive supplementary material.

### 2.2 Decisions that have been made, and why

The pre-build planning surfaced and resolved a number of design choices. They are documented here so the rebuild has a clear specification.

#### 2.2.1 Targets: all primate, all MPTP

Throughout the rebuild, both healthy and Parkinsonian targets are anchored to non-human primate (NHP) MPTP electrophysiology rather than to human PD intraoperative recordings. This decision was made deliberately, after considering a hybrid (NHP healthy, human PD) and an all-human alternative.

The case for all-primate:

- **Internal consistency.** A single species, a single experimental paradigm, a single recording context. Healthy and PD targets come from the same data lineage with before/after recordings in the same animals.
- **Convention.** Every major STN-GPe-GPi computational model — Rubin and Terman (2004), Hahn and McIntyre (2010), Kumaravelu et al. (2016), Ebert et al. (2014) — anchors to primate MPTP. The rebuild follows this convention.
- **Beta-band cleanliness.** Primate MPTP literature consistently reports increased oscillatory power in the 8–15 Hz range (Stein and Bar-Gad, 2013; Devergnas et al., 2014; Tachibana et al., 2014). Anchoring to primate lets us use the primate-native band rather than awkwardly mapping primate 8-15 Hz onto human 13-30 Hz.
- **The translational gap is honestly disclosed.** "We use primate MPTP data because that is where the clean before/after data exists. Translation to human PD is the subject of separate work." This is defensible and matches how the field operates.

The downstream patient-fitting paper will anchor to human PD data and will explicitly bridge the species gap — but that is the next paper's problem, not this paper's problem.

#### 2.2.2 Firing rate and CV targets

Targets are taken from Tachibana et al. (2014), which presents the most carefully measured contemporary primate MPTP rate data and is the source of the "GPi unchanged" finding that contradicts the classical rate model.

| Population | Healthy | PD | Source |
|---|---|---|---|
| STN | 20 Hz | 27 Hz | Tachibana et al., 2014 (consistent with Bergman et al., 1994) |
| GPe | 65 Hz | 41 Hz | Tachibana et al., 2014 |
| GPi | 67 Hz | 63 Hz | Tachibana et al., 2014 |
| STN CV | 0.4 | 0.6 | approximate, soft constraint |
| GPe CV | 0.35 | 0.40 | approximate, soft constraint |
| GPi CV | 0.20 | 0.30 | approximate, soft constraint |

The PD GPi target of 63 Hz is the most consequential change relative to the original paper, which targeted 82.5 Hz. The "GPi unchanged in PD" finding is an active controversy: older work (Filion and Tremblay, 1991; Boraud et al., 2002) and the classical rate model (Albin et al., 1989; DeLong, 1990) predict GPi rate increases, while more recent careful work (Tachibana et al., 2014; Wichmann et al., 2011; Raz et al., 2000; Rivlin-Etzion et al., 2008) finds little or no significant change. We use Tachibana as the primary target and discuss the controversy in the manuscript. We do not run a separate sensitivity analysis on the alternative target — the discussion suffices.

CV targets are approximate. Quantitative CV values are rarely tabulated in the primate literature, which more commonly reports burst proportions (Bergman et al., 1994; Soares et al., 2004; Wichmann and Soares, 2006). The targets above are order-of-magnitude estimates capturing the qualitative pattern of increased ISI variability in PD. CV is included as a soft constraint (low weight in the objective) so the optimizer cannot satisfy rate targets via tonic-locked firing without irregularity.

#### 2.2.3 Beta band: 8–15 Hz, treated as a constraint

The current paper uses 13–30 Hz throughout while citing primate firing-rate data. This is an internal inconsistency: primate MPTP studies consistently identify the relevant oscillation band as 8–15 Hz, not 13–30 Hz. Tachibana et al. (2014) explicitly state: "the mean power of the 8–15 Hz (low-β) oscillations was increased in the GPi/GPe and STN, whereas there were no consistent changes in the 3–8 Hz and 15–30 Hz (high-β) oscillations." Stein and Bar-Gad (2013) review this discrepancy and propose primate 8–15 Hz as the homolog of human 13–30 Hz beta. Devergnas et al. (2014) and Connolly et al. (2015) confirm the species difference.

The rebuild uses **8–15 Hz** as the beta band. This is the cleanest primate-anchored choice. It overlaps slightly with primate alpha but captures the strongest signal in MPTP literature. Citation: Stein and Bar-Gad (2013), Devergnas et al. (2014), Tachibana et al. (2014).

The original paper framed beta as a value target with bounds: penalty if outside [0.20, 0.40] of total spectral power. With this formulation and a high penalty weight, the optimizer over-prioritized beta at the expense of firing rate accuracy. The rebuild instead treats beta as a **constraint**, not an objective term. Optuna's CMA-ES sampler supports trial constraints; we will compute STN beta fraction and either accept (beta condition satisfied) or reject (penalty applied) the trial accordingly.

Specifically:
- Healthy condition: STN beta fraction (8–15 Hz / 1–100 Hz) < 0.05
- Parkinsonian condition: STN beta fraction > 0.15

This directly matches the biological story — PD is *defined by* elevated beta; given that, what configuration also matches firing rates? — and avoids the weight-tuning problem of weighted-sum scalarization.

If the constrained formulation proves intractable in practice (for instance, if the optimizer spends excessive trials in the infeasible region), we fall back to a one-sided threshold-with-bonus formulation: penalty proportional to `max(0, threshold - beta)`, no penalty for exceeding threshold. This will be evaluated empirically in the first PD optimization run; the choice does not need to be pre-committed.

#### 2.2.4 Synaptic conductance bounds

The current code parameterizes synaptic strength as multipliers on arbitrary baseline values (g_baseline = 0.2, 0.9, 0.2, 0.3 mS/cm² for STN→GPe, GPe→STN, STN→GPi, GPe→GPi respectively). Combined with the silent 0.005 scaling factor in `integrator.py`, the effective per-synapse conductance lands near canonical Rubin-Terman values; but the formulation makes the multipliers uninterpretable without recovering the implicit scaling.

The rebuild searches **absolute synaptic conductances** in mS/cm², dropping all scaling factors. Bounds are 0.005–0.5 mS/cm² per synapse, derived from the Rubin-Terman lineage (Rubin and Terman, 2004; Ebert et al., 2014, Tables 1–2). Tonic drives are bounded ±5 µA/cm². OU mean is bounded ±5 µA/cm². OU sigma is bounded 0–5 µA/cm². All units consistent throughout (current density convention, matching the inherited models).

Healthy uses symmetric bounds (same range for all four synaptic conductances). PD runs both **asymmetric** (literature-motivated narrower bounds: g_STN→GPe biased high, g_GPe→STN biased low, etc.) and **symmetric** (same bounds as healthy) variants. Both are reported. This directly answers Reviewer 1's concern about robustness to bound choices: if the asymmetric and symmetric runs produce qualitatively similar configurations, the framework's discovery is genuine; if not, the dependence on prior knowledge is itself a finding.

#### 2.2.5 Parameters under optimization

Thirteen total:

- Four synaptic peak conductances: g_STN→GPe, g_GPe→STN, g_STN→GPi, g_GPe→GPi (mS/cm²)
- Three tonic drives: I_STN, I_GPe, I_GPi (µA/cm²)
- Three OU process means: μ_STN, μ_GPe, μ_GPi (µA/cm²)
- Three OU process sigmas: σ_STN, σ_GPe, σ_GPi (µA/cm²)

The original code searches only σ, fixing μ at build-time defaults; the manuscript inaccurately states both are optimized. The rebuild optimizes both, making the manuscript description accurate and giving the optimizer the freedom to set tonic afferent drive (a stand-in for unmodeled cortical/striatal/thalamic input) consistently with the sought firing statistics.

#### 2.2.6 Stochasticity handling

Three random-number streams must be managed: (i) network seed, controlling connectivity sampling and initial neuron states; (ii) OU noise seed, controlling stochastic input; (iii) CMA-ES seed, controlling optimizer sampling.

Policy:
- During optimization search: network seed fixed (one value), OU seed fresh per trial, CMA-ES seed fixed for the headline run. CMA-ES is robust to objective noise (Hansen, 2016) and the alternative — averaging k seeds per trial — costs k× compute for sqrt(k) noise reduction, which is unfavorable in our regime.
- After optimization: the top configurations from each run are validated across **10 seeds** (varying both network and OU seeds), and the most robust configuration is reported as the result. This gives a clean methodological story: "we ran the optimization, then validated the top configurations across 10 seeds and selected the most robust."
- For convergence characterization (Validation B3b): the CMA-ES seed is varied across N independent optimizations to assess whether the optimizer reliably converges to the same region of parameter space.

#### 2.2.7 Network size and trial counts

Optimization runs at **450 neurons** (100 STN / 200 GPe / 150 GPi), as in the original. Final validation runs at **45,000 neurons** (10,000 STN / 20,000 GPe / 15,000 GPi). The fixed-indegree connectivity scheme (Gerstner et al., 2014) is what makes parameters optimized at small networks transfer to large networks without retuning, and this transfer is itself one of the validations.

Trial counts: **1000 per optimization run as standard, 1500 for headline runs.** With 13 parameters, the textbook CMA-ES recommendation is roughly 100×n_dim ≈ 1300 trials; 1500 is comfortably above that. The 1000 standard is sufficient for sensitivity-sweep runs where individual convergence quality is less critical than the comparison across runs.

#### 2.2.8 Models

- **STN:** single-compartment Hodgkin-Huxley neuron from Terman, Rubin, Yew, and Wilson (2002). The canonical model from this lineage is single-compartment by design (not a collapse from multi-compartment), and is the foundational model for STN–GPe oscillation studies; Rubin and Terman (2004) — used unchanged for the GPe and GPi cells in this rebuild — builds directly on it. Parameters from the `episodic.ode` source distributed with the paper (ModelDB accession 182758) are used as the healthy-state ground truth; full parameter table and validation in `docs/stn_validation.md`. The Gillies–Willshaw (2006) model was initially considered for greater biophysical detail (h-current, dendritic T-current) but, when collapsed to a single compartment, produces a near-silent f-I curve at the drive levels accessible to the optimizer's bounded search space (~11 Hz at I_drive = 42 µA/cm²; see `docs/phase1_review.md`). Terman–Rubin gives a comfortable f-I regime (~10 Hz tonic at I = 0, ~20 Hz at I = 5 µA/cm²) appropriate for a methods-paper demonstration. The implementation uses a linear I_Ca activation gate (`I_Ca = g_Ca · sinf(V) · (V − E_Ca)`), which provides sub-threshold pacemaking consistent with the experimental characterization of autonomous STN firing (Bevan and Wilson, 1999); the canonical T-current rebound mechanism is preserved and selectively engaged on hyperpolarization. See `docs/stn_validation.md` §2 for the empirical and biophysical justification.
- **GPe and GPi:** Rubin-Terman single-compartment formalism (Rubin and Terman, 2004), unchanged in structure. These models were designed as point neurons.
- **Numerical integration:** forward Euler, dt = 0.025 ms. Validated against dt = 0.0125 ms.
- **Connectivity:** fixed-indegree, K_STN→GPe = 15, K_GPe→STN = 14, K_STN→GPi = 30, K_GPe→GPi = 10. Justification: Gerstner et al. (2014) recommend this scheme for networks intended to generalize across sizes, since fixed-probability connectivity loses fluctuation-driven dynamics in large networks (variance scales as 1/√N).
- **Synapses:** double-exponential conductance-based model. AMPA for excitatory (E_syn = 0 mV), GABA_A for inhibitory (E_syn = −70 mV).
- **Background input:** Ornstein-Uhlenbeck process per neuron, τ = 5 ms. Both μ and σ optimized.
- **Synaptic delays:** 5 ms (STN→GPe, STN→GPi, GPe→GPi), 8 ms (GPe→STN). Consistent with prior model lineage.

#### 2.2.9 LFP proxy

Mean membrane voltage, the original paper's choice, is a poor LFP proxy in single-compartment networks: real LFP is dominated by transmembrane currents through extended dendritic structure, weighted by distance to the recording electrode, and point neurons have neither dendrites nor spatial structure. The current paper's healthy LFP traces show 22 mV monotonic drift over 300 ms, inflating broadband power and depressing the reported beta fraction.

The rebuild uses **population firing rate** (binned spike count) as the primary LFP proxy. This is standard for spiking-network LFP analysis (Mallet et al., 2008), reflects spiking synchrony, and has no DC drift problem. Mean Vm and summed synaptic currents are computed as **secondary cross-checks**; consistency across the three proxies is itself a methodological validation. The primary headline numbers come from the firing-rate proxy.

This change will shift the reported beta fractions. The qualitative result (PD has more beta synchrony than healthy) almost certainly holds; the specific percentages will differ from the current paper.

#### 2.2.10 Engineering

- Package name: `bgnet/`. Importable as a real Python package.
- One YAML config per study (e.g. `configs/optimization_healthy.yaml`, `configs/optimization_pd_asymmetric.yaml`). No base-and-overrides.
- Structured logging via Python's `logging` module, not print statements.
- Each run produces a directory `results/<study>/<timestamp>/{config.yaml, log.txt, results.pkl, figures/}`. This solves the "which run produced this number" problem the current codebase has.
- Tests: unit tests for neuron step functions, synaptic step, integration; integration tests for short simulations against expected firing rates; smoke tests for each script with tiny networks and few trials. Not full TDD, but enough to catch regressions.

### 2.3 Validations and sensitivity analyses

#### 2.3.1 Carry-over from original paper (B1)

These remain in scope:
- Scaling validation across 450–45,000 neurons
- Statistical validation across 10 seeds at 45,000
- Timestep sensitivity (dt = 0.025, 0.0125 ms)
- Sampler comparison (CMA-ES vs. TPE vs. Random)
- Performance benchmarks (wall-clock vs. network size, GPU vs. CPU)
- DBS qualitative sanity check, framed as such

#### 2.3.2 Reviewer-driven (B2)

- **Loss-weight sensitivity sweep.** Grid over weight combinations in the rate/CV portion of the objective. Directly answers Reviewer 2 #3.
- **Bound sensitivity (asymmetric vs. symmetric).** Directly answers Reviewer 1 #4.
- **Stochasticity characterization.** Variance of metrics across seeds at the converged configurations. Directly answers Reviewer 2 #2.
- **LFP proxy comparison.** Beta fraction reported under firing-rate, mean-Vm, and synaptic-current proxies; consistency demonstrated.

#### 2.3.3 Above-and-beyond (B3)

- **B3a: Parameter recovery test.** Take the discovered configuration, generate synthetic ground-truth data from it, run the optimization again targeting the synthetic data. Does it recover the original parameters? This is a standard inverse-problem validation that the original paper never performed and that bears directly on the framework's well-posedness for downstream patient fitting.
- **B3b: Optimization convergence characterization.** Run ~10 independent CMA-ES optimizations with different sampler seeds; characterize the distribution of discovered configurations. If they cluster tightly, the framework is reliable; if they scatter, that itself is an important finding for downstream work.

B3c (manual-tuning-baseline comparison) and B3d (extended time complexity analysis) are not in scope.

### 2.4 Manuscript structure

**Main paper:**
- Framework architecture and design
- Worked example: STN-GPe-GPi healthy and PD optimization (both bound conditions)
- Scaling validation, statistical validation across seeds
- DBS sanity check
- Performance benchmarks
- Parameter recovery test (B3a)
- Convergence characterization (B3b)
- Sampler comparison

**Supplementary:**
- Loss-weight sensitivity sweeps
- Bound sensitivity details
- LFP proxy comparison
- Timestep sensitivity
- Implementation notes

If the bound-sensitivity result is qualitatively interesting (asymmetric and symmetric runs differ meaningfully), it moves to main. If they agree, the asymmetric run is the headline and the symmetric run lives in supplementary as confirmation.

---

## 3. How we are going to do it

### 3.1 Build sequencing

Roughly six weeks. Suggested phasing:

1. **Week 1: Foundation rebuild.** New `bgnet/` package. Neuron modules with explicit unit conventions (no scaling factors, current density throughout). Synapse module without fudge factors. Noise module. Integrator. Tests at each step.
2. **Week 2: Network builder + optimization scaffolding.** YAML config loading. Structured logging. Results directory layout. CMA-ES wrapper with constraint handling. Smoke tests for each script.
3. **Week 2-3: Healthy optimization.** First real test that the rebuild works. Likely some debugging here. Validate that the discovered healthy configuration produces reasonable network behavior.
4. **Week 3: PD optimization.** Both bound conditions. Compare configurations. Validate top configurations across 10 seeds.
5. **Week 3-4: Validations and sensitivity sweeps.** B1 + B2 + B3a + B3b. Parallelize where memory permits.
6. **Week 4-5: Figure generation and tables.** All figures regenerated from rebuild data. Consistent styling. Captions with all abbreviations defined.
7. **Week 5-6: Manuscript revision.** Reframe to methodological contribution. Update Section 1, Section 4. Address every reviewer point in response document. Update references to full author lists (≤10 authors) per editor instruction.
8. **Week 6: Buffer.** For things that will go wrong.

### 3.2 Compute strategy

Single L4 GPU on GCP. Each optimization run at 450 neurons takes ~30 minutes (1000 trials × ~2s). Parallelization opportunities:

- Sensitivity sweeps over loss weights: independent runs, run sequentially or batched.
- Bound sensitivity (asymmetric vs symmetric for PD): two independent runs, sequential.
- Convergence characterization (B3b): N independent runs, sequential or batched depending on memory.
- Parameter recovery (B3a): one extra optimization run.

Total optimization compute budget: roughly 15-25 runs × 30 minutes = 8-12 hours of GPU time, distributed across the build period. At $0.84/hour, ≈ $10 in cloud compute for the entire revision.

Validation runs at 45,000 neurons take ~3 seconds each but generate large data; the binding constraint is memory and disk, not GPU time.

### 3.3 What success looks like

The revised paper has:

- A clear methodological framing: framework first, demonstration second.
- A clean codebase that any reader can clone, configure, and run, with results that match the manuscript.
- Healthy and PD configurations discovered by the optimizer in absolute biophysical units, with the asymmetric/symmetric bound comparison demonstrating robustness.
- A parameter recovery test confirming the framework's well-posedness.
- A convergence characterization quantifying the reliability of the optimization.
- A loss-weight sensitivity sweep addressing Reviewer 2's concern.
- LFP proxy comparison demonstrating that the headline beta result is not an artifact of proxy choice.
- Honest reporting of what the framework actually produced, including any disagreements with the original paper's numbers.
- All references in correct style (full author lists ≤10).
- A response document addressing every reviewer point.

### 3.4 What could go wrong

- **The discovered PD configuration in the rebuild may not match the original paper's headline ratios.** This is acceptable — the original paper's numbers are not reproducible from the saved code, so reproducing them is not the goal. The rebuild reports what the rebuild produces.
- **The constraint formulation for beta may prove tricky.** Fallback is one-sided threshold with bonus, evaluated empirically.
- **Symmetric-bound PD may converge to a different configuration than asymmetric-bound.** This is itself a finding and goes in the manuscript.
- **The LFP-proxy switch may produce qualitatively different beta numbers.** Likely the qualitative pattern holds; if it does not, that is a real finding worth reporting honestly.
- **Six weeks is tight.** The Week 6 buffer absorbs slippage; if it doesn't, we can request a deadline extension via the editor.

---

## References

Albin, R. L., Young, A. B., and Penney, J. B. (1989). The functional anatomy of basal ganglia disorders. *Trends in Neurosciences*, 12(10), 366–375.

Atherton, J. F., and Bevan, M. D. (2005). Ionic mechanisms underlying autonomous action potential generation in the somata and dendrites of GABAergic substantia nigra pars reticulata neurons in vitro. *Journal of Neuroscience*, 25(36), 8272–8281.

Bergman, H., Wichmann, T., Karmon, B., and DeLong, M. R. (1994). The primate subthalamic nucleus. II. Neuronal activity in the MPTP model of parkinsonism. *Journal of Neurophysiology*, 72(2), 507–520.

Bevan, M. D., and Wilson, C. J. (1999). Mechanisms underlying spontaneous oscillation and rhythmic firing in rat subthalamic neurons. *Journal of Neuroscience*, 19(17), 7617–7628.

Boraud, T., Bezard, E., Bioulac, B., and Gross, C. E. (2002). From single extracellular unit recording in experimental and human Parkinsonism to the development of a functional concept of the role played by the basal ganglia in motor control. *Progress in Neurobiology*, 66(4), 265–283.

Carlson, K. D., Nageswaran, J. M., Dutt, N., and Krichmar, J. L. (2014). An efficient automated parameter tuning framework for spiking neural networks. *Frontiers in Neuroscience*, 8, 10.

Connolly, A. T., Jensen, A. L., Bello, E. M., Netoff, T. I., Baker, K. B., Johnson, M. D., and Vitek, J. L. (2015). Modulations in oscillatory frequency and coupling in globus pallidus with increasing parkinsonian severity. *Journal of Neuroscience*, 35(15), 6231–6240.

DeLong, M. R. (1990). Primate models of movement disorders of basal ganglia origin. *Trends in Neurosciences*, 13(7), 281–285.

Devergnas, A., Pittard, D., Bliwise, D., and Wichmann, T. (2014). Relationship between oscillatory activity in the cortico-basal ganglia network and parkinsonism in MPTP-treated monkeys. *Neurobiology of Disease*, 68, 156–166.

Ebert, M., Hauptmann, C., and Tass, P. A. (2014). Coordinated reset stimulation in a large-scale model of the STN-GPe circuit. *Frontiers in Computational Neuroscience*, 8, 154.

Filion, M., and Tremblay, L. (1991). Abnormal spontaneous activity of globus pallidus neurons in monkeys with MPTP-induced parkinsonism. *Brain Research*, 547(1), 142–151.

Gerstner, W., Kistler, W. M., Naud, R., and Paninski, L. (2014). *Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition*. Cambridge University Press.

Gillies, A., and Willshaw, D. (2006). Membrane channel interactions underlying rat subthalamic projection neuron rhythmic and bursting activity. *Journal of Neurophysiology*, 95(4), 2352–2365.

Hahn, P. J., and McIntyre, C. C. (2010). Modeling shifts in the rate and pattern of subthalamopallidal network activity during deep brain stimulation. *Journal of Computational Neuroscience*, 28(3), 425–441.

Hansen, N. (2016). The CMA evolution strategy: a tutorial. *arXiv:1604.00772*.

Hutchison, W. D., Lozano, A. M., Tasker, R. R., Lang, A. E., and Dostrovsky, J. O. (1998). Identification and characterization of neurons with tremor-frequency activity in human globus pallidus. *Experimental Brain Research*, 113(3), 557–563.

Kumaravelu, K., Brocker, D. T., and Grill, W. M. (2016). A biophysical model of the cortex-basal ganglia-thalamic network in the 6-OHDA lesioned rat model of Parkinson's disease. *Journal of Computational Neuroscience*, 40(2), 207–229.

Mallet, N., Pogosyan, A., Sharott, A., Csicsvari, J., Bolam, J. P., Brown, P., and Magill, P. J. (2008). Disrupted dopamine transmission and the emergence of exaggerated beta oscillations in subthalamic nucleus and cerebral cortex. *Journal of Neuroscience*, 28(18), 4795–4806.

Raz, A., Vaadia, E., and Bergman, H. (2000). Firing patterns and correlations of spontaneous discharge of pallidal neurons in the normal and the tremulous 1-methyl-4-phenyl-1,2,3,6-tetrahydropyridine vervet model of parkinsonism. *Journal of Neuroscience*, 20(22), 8559–8571.

Rivlin-Etzion, M., Marmor, O., Saban, G., Rosin, B., Haber, S. N., Vaadia, E., Prut, Y., and Bergman, H. (2008). Low-pass filter properties of basal ganglia cortical muscle loops in the normal and MPTP primate model of parkinsonism. *Journal of Neuroscience*, 28(3), 633–649.

Rubin, J. E., and Terman, D. (2004). High frequency stimulation of the subthalamic nucleus eliminates pathological thalamic rhythmicity in a computational model. *Journal of Computational Neuroscience*, 16(3), 211–235.

Soares, J., Kliem, M. A., Betarbet, R., Greenamyre, J. T., Yamamoto, B., and Wichmann, T. (2004). Role of external pallidal segment in primate parkinsonism: comparison of the effects of 1-methyl-4-phenyl-1,2,3,6-tetrahydropyridine-induced parkinsonism and lesions of the external pallidal segment. *Journal of Neuroscience*, 24(29), 6417–6426.

Stein, E., and Bar-Gad, I. (2013). β oscillations in the cortico-basal ganglia loop during parkinsonism. *Experimental Neurology*, 245, 52–59.

Tachibana, Y., Iwamuro, H., Kita, H., Takada, M., and Nambu, A. (2014). Mechanism of parkinsonian neuronal oscillations in the primate basal ganglia: some considerations based on our recent work. *Frontiers in Systems Neuroscience*, 8, 74.

Terman, D., Rubin, J. E., Yew, A. C., and Wilson, C. J. (2002). Activity patterns in a model for the subthalamopallidal network of the basal ganglia. *Journal of Neuroscience*, 22(7), 2963–2976.

Vich, C. et al. (2023). CBGTPy: An extensible cortico-basal ganglia-thalamic framework for modeling biological decision making. *bioRxiv* 2023.09.05.556301.

Vieth, M., Rahimi, A., Mohammadi, A. G., Triesch, J., and Ganjtabesh, M. (2024). Accelerating spiking neural network simulations with PymoNNto and PymoNNtorch. *Frontiers in Neuroinformatics*, 18, 1331220.

Wichmann, T., Bergman, H., Starr, P. A., Subramanian, T., Watts, R. L., and DeLong, M. R. (2011). Comparison of MPTP-induced changes in spontaneous neuronal discharge in the internal pallidal segment and in the substantia nigra pars reticulata in primates. *Experimental Brain Research*, 125, 397–409.

Wichmann, T., and Soares, J. (2006). Neuronal firing before and after burst discharges in the monkey basal ganglia is predictably patterned in the normal state and altered in parkinsonism. *Journal of Neurophysiology*, 95(4), 2120–2133.
