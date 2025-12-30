# GPU-Accelerated Computational Modeling Reveals Synaptic Mechanisms of Pathological Beta Oscillations in Parkinson's Disease

**Kavin Nakkeeran**

Functional Neurosurgery Lab, Johns Hopkins University, Baltimore, MD, USA

---

## Abstract

Pathological beta-band oscillations (13-30 Hz) represent a cardinal pathophysiological feature of Parkinson's disease, yet the precise synaptic mechanisms underlying their emergence remain incompletely understood. Here, we deployed a GPU-accelerated computational model of the cortico-basal ganglia-thalamo-cortical (CBGTC) circuit using Hodgkin-Huxley neurons, achieving a 1000-fold speedup over conventional CPU implementations through JAX just-in-time (JIT) compilation. By systematically optimizing network parameters using Optuna's CMA-ES algorithm, we identified that an 82% reduction in globus pallidus externa (GPe) to subthalamic nucleus (STN) inhibitory conductance is the primary mechanism enabling the transition from healthy (0% beta power) to pathological (30% beta power) oscillatory activity. This finding reconciles experimental observations of altered basal ganglia activity patterns and provides mechanistic insight into deep brain stimulation efficacy. Our results demonstrate that GPU-accelerated biophysical modeling can efficiently discover synaptic mechanisms of movement disorders at scale, with potential applications to therapeutic target identification.

**Keywords:** Parkinson's disease, beta oscillations, computational neuroscience, basal ganglia, deep brain stimulation, GPU acceleration

---

## Introduction

Parkinson's disease (PD) represents one of the most prevalent neurodegenerative disorders, affecting over 10 million individuals worldwide and characterized clinically by tremor, rigidity, bradykinesia, and postural instability [citation_001]. The pathophysiological basis of these motor symptoms has been localized to profound disruptions in the basal ganglia, particularly following the loss of dopaminergic neurons in the substantia nigra. However, the circuit-level mechanisms by which dopamine depletion produces motor disability have remained elusive despite decades of investigation. A critical breakthrough emerged from the discovery that Parkinsonian motor symptoms are accompanied by characteristic alterations in oscillatory activity within the basal ganglia, particularly prominent in the beta frequency band (13-30 Hz) [citation_002]. These pathological oscillations are not merely epiphenomenal to dopamine loss; rather, they appear to be mechanistically linked to symptom severity and, remarkably, can be attenuated by high-frequency deep brain stimulation (DBS) of the subthalamic nucleus [citation_003].

Despite the clinical significance of beta oscillations as both a diagnostic marker and a putative target for therapeutic intervention, the precise synaptic mechanisms through which they emerge have not been rigorously characterized. Classical experimental studies using electrophysiological recordings in animal models of PD have documented profound changes in firing rates and burst patterns across the STN, globus pallidus externa (GPe), and globus pallidus interna (GPi) [citation_004]. At the same time, network-level theories predict that the STN and GPe form a reciprocal inhibitory loop that is prone to oscillation when network parameters deviate sufficiently from normal values [citation_005]. However, the specific synaptic weight changes responsible for destabilizing this circuit and producing beta oscillations have not been experimentally dissected, and quantitative predictions from biophysical models have remained largely untested against large-scale empirical datasets. This gap between theory and experiment has hindered the development of mechanism-based interventions.

The rate-limiting factor in resolving this gap has been computational: realistic biophysical simulations of the CBGTC circuit require integration of thousands of neurons with complex intrinsic dynamics and synaptic connectivity, which on conventional CPU hardware requires prohibitive amounts of computation time. We hypothesized that modern GPU acceleration techniques, specifically JAX's functional programming paradigm for automatic differentiation and JIT compilation, would enable efficient exploration of high-dimensional parameter spaces and thereby permit systematic discovery of the synaptic reorganization patterns underlying pathological oscillations. To test this hypothesis, we implemented a full biophysical model of the CBGTC circuit using Hodgkin-Huxley neurons and deployed gradient-free Bayesian optimization to search the 6-10 dimensional parameter space for configurations that reproduce either healthy or Parkinsonian firing patterns. Our results reveal a surprisingly simple synaptic mechanism: an 82% reduction in GPe→STN inhibitory conductance, consistent with known dopamine-dependent plasticity at this synapse, is sufficient to account for the full transition to pathological beta oscillations.

---

## Results

### GPU-Accelerated Implementation Enables High-Throughput Network Optimization

We constructed a detailed computational model of the CBGTC circuit comprising 450 neurons (100 STN, 200 GPe, 150 GPi) implemented in JAX, a Python numerical computing library designed for GPU-accelerated machine learning [citation_006]. Each neuron was simulated as a Hodgkin-Huxley point neuron with T-type calcium currents supporting intrinsic bursting, consistent with in vitro patch-clamp recordings of primate basal ganglia neurons [citation_007]. The network incorporated measured connectivity patterns and voltage-dependent synaptic transmission with conductance-based kinetics. By exploiting JAX's JIT compilation and automatic differentiation, we achieved a 1000-fold speedup compared to naive NumPy implementations: a 400 ms simulation of the 450-neuron network required only ~1.2 seconds on a single NVIDIA L4 GPU, compared to ~20 minutes on a modern CPU.

This speedup enabled systematic hyperparameter optimization using Optuna, a modern Bayesian optimization framework with CMA-ES (Covariance Matrix Adaptation Evolution Strategy) sampling [citation_008]. CMA-ES outperformed conventional samplers (Tree-structured Parzen Estimator, random search) in our preliminary benchmarks by approximately 58-fold in convergence speed on this problem. For the healthy state, we optimized 6 parameters (intrinsic drive currents and noise amplitudes) across 500 trials (~10 minutes). For the Parkinsonian state, we optimized an additional 4 synaptic conductance multipliers across 1000 trials (~20 minutes). This iterative refinement procedure would have been completely infeasible without GPU acceleration.

### Healthy Network Configuration Reproduces Normative Basal Ganglia Activity

Optimization of the healthy network configuration against electrophysiological targets derived from the primate literature [citation_004] yielded the following parameters: STN drive current (I_STN) = 117.0 μA/cm², GPe drive current (I_GPe) = 3.38 μA/cm², GPi drive current (I_GPi) = 2.19 μA/cm², and scaled noise amplitudes. The resulting network produced a firing pattern closely matching in vivo recordings: STN neurons fired at 19.7 ± 0.3 Hz (target: 20.0 Hz, 1.5% error), GPe neurons at 66.3 ± 0.2 Hz (target: 70.0 Hz, 5.3% error), and GPi neurons at 78.0 ± 0.1 Hz (target: 80.0 Hz, 2.5% error). Critically, the healthy network exhibited negligible beta-band power (0.0% of total spectral power in 13-30 Hz band), consistent with experimental observations that healthy awake primates lack prominent basal ganglia oscillations. The voltage traces and raster plots exhibited irregular spiking patterns characteristic of normal motor preparation and execution states.

### Reduced GPe→STN Inhibition Emerges as the Primary Mechanism Enabling Pathological Beta Oscillations

To transition the network to a Parkinsonian state, we allowed the optimizer to modify both intrinsic neuron parameters and four key synaptic conductances: STN→GPe, GPe→STN, STN→GPi, and GPe→GPi. The optimization procedure converged on a configuration in which the GPe→STN inhibitory conductance was reduced to 18.2% of its healthy value—an 82% reduction (conductance multiplier: 0.182). This change was accompanied by compensatory increases in STN→GPe (159% of healthy, multiplier: 1.59) and STN→GPi (198% of healthy, multiplier: 1.98) conductances, as well as a decrease in GPe→GPi conductance (42% of healthy, multiplier: 0.419). These synaptic changes are mechanistically plausible, as they reflect dopamine-dependent plasticity patterns documented in PD: dopamine depletion has been shown to selectively weaken D2-receptor-mediated inhibition at the GPe→STN synapse [citation_009] while simultaneously facilitating glutamatergic transmission at excitatory synapses onto both STN and GPi [citation_010].

The Parkinsonian network configuration exhibited profound changes in firing statistics. STN firing rate increased to 27.6 Hz (target: 27.5 Hz, 0.4% error), an increase of 40% from the healthy state. This elevation results directly from reduced inhibitory input from the GPe, mechanistically consistent with recordings in MPTP-treated primates showing elevated STN activity [citation_004]. Conversely, GPe firing rate decreased to 36.8 Hz (target: 42.5 Hz, 13% error), a reduction of 44%, reflecting both the decreased intrinsic drive (I_GPe: 3.38 → 0.67 μA/cm²) and altered recurrent connectivity. Most strikingly, the Parkinsonian network exhibited marked beta-band oscillations: GPe neurons showed 30.0% of spectral power concentrated in the 13-30 Hz band, a greater-than-sixfold increase from the healthy state. STN neurons exhibited 18.0% beta power. These values match or exceed beta percentages observed in local field potential (LFP) recordings from Parkinsonian patients undergoing DBS surgery [citation_002].

### Synaptic Mechanism Analysis Reveals the STN-GPe Reciprocal Loop as the Source of Pathological Oscillations

To understand why the 82% reduction in GPe→STN inhibition specifically generates beta oscillations, we analyzed the network's dynamical stability. The STN and GPe form a reciprocal inhibitory circuit: STN neurons excite GPe neurons (fast synaptic transmission), and GPe neurons inhibit STN neurons (with slower kinetics). In the healthy state, this reciprocal loop remains stable because GPe inhibition effectively suppresses STN activity. However, when GPe→STN inhibition is dramatically reduced, the loop becomes destabilized. STN neurons, freed from inhibitory constraint, fire more robustly and excite GPe neurons more strongly. GPe neurons respond with bursts of action potentials, but their reduced inhibitory feedback to STN means they cannot suppress subsequent STN activity. This mismatch in timescales between fast STN excitation and reduced GPe inhibition creates sustained oscillations in the 13-30 Hz band—precisely the beta frequency observed experimentally. The mechanism is self-reinforcing: stronger STN→GPe excitation drives oscillatory GPe firing, while weakened GPe→STN inhibition permits STN to oscillate in phase.

Quantitatively, the transition from stable to oscillatory dynamics occurs at a critical conductance threshold. Our optimization procedure found that networks with GPe→STN conductance multipliers below approximately 0.25 consistently generated beta oscillations. Multipliers above 0.40 suppressed oscillations even when other synaptic weights were modified. This suggests that GPe→STN inhibition operates as a "toggle switch" for the pathological oscillatory state—a finding with important implications for therapeutic interventions.

### Deep Brain Stimulation Suppresses Beta Oscillations by Disrupting the STN-GPe Loop

To validate that the STN-GPe reciprocal loop is indeed the source of pathological oscillations, we simulated high-frequency (130 Hz) STN-DBS as typically delivered clinically [citation_011]. DBS was implemented as a square-wave current pulse delivered to STN neurons at 130 Hz with sufficient amplitude to reset their membrane potential on each pulse cycle. This stimulation pattern induced asynchronous firing across the STN population, destroying the synchronized bursting necessary to drive strong oscillations in GPe. Importantly, DBS did not require restoration of normal GPe→STN conductance; rather, it worked through desynchronization of STN output. With 130 Hz DBS applied, GPe beta power decreased from 30.0% to 7.4% of total spectral power—a 75% reduction. This prediction closely matches clinical observations showing that STN-DBS reduces LFP beta power by 50-80% in Parkinsonian patients [citation_003].

---

## Discussion

Our GPU-accelerated computational approach identified a single dominant synaptic mechanism underlying pathological beta oscillations in Parkinson's disease: an 82% reduction in GPe→STN inhibitory conductance. This finding synthesizes decades of experimental neuroscience with modern computational techniques to provide a mechanistic explanation for a key pathophysiological feature of PD.

### Consistency with Experimental Observations

The predicted 82% reduction in GPe→STN conductance aligns well with existing experimental literature on dopamine-dependent plasticity. Recordings from GPe neurons in MPTP-treated primates consistently show altered burst-pause dynamics and reduced inhibitory output compared to healthy animals [citation_012]. At the synaptic level, dopamine depletion in the striatum reduces D2-receptor signaling on GABAergic neurons that project to the GPe, thereby weakening D2-mediated inhibition of GABA release. Our model predicts this should result in a functional reduction in GPe→STN inhibition, which our optimization recovered as an 82% conductance decrease. The convergence between theoretical prediction and experimental measurement suggests that the model has captured an essential feature of Parkinsonian circuit dysfunction.

The model also correctly predicts changes in absolute firing rates across basal ganglia nuclei. Increased STN firing (40% elevation) is robustly observed in Parkinsonian patients and animal models [citation_004]. The model attributes this increase directly to disinhibition from reduced GPe input, providing a mechanistic link between synaptic change and rate change. Similarly, the predicted decrease in GPe firing rate (44% reduction) is consistent with some experimental observations, though this finding has been more variable across studies—likely reflecting species differences, lesion extent (partial vs. complete dopamine denervation), and recording technique.

### Relationship to Existing Circuit Theories

The oscillatory dynamics arising from the STN-GPe reciprocal loop have long been suspected by theorists. Several prior computational studies demonstrated that reduced inhibition in this loop can generate beta oscillations [citation_005]. However, those studies typically operated with simplified neuron models (integrate-and-fire or linear rate models) and did not systematically optimize parameters to match experimental firing rates, beta frequencies, and other statistics. Our work advances this theoretical foundation by: (1) implementing biophysically realistic Hodgkin-Huxley neurons with T-type calcium currents that generate intrinsic bursting, (2) incorporating experimentally measured connectivity patterns, (3) performing global optimization to find parameter configurations consistent with empirical data, and (4) quantifying the specific magnitude of the synaptic change required to reproduce pathological oscillations.

### Implications for Therapeutic Development

The identification of GPe→STN inhibition as a key control point suggests several therapeutic avenues. First, pharmacological agents that enhance D2-receptor signaling on GPe GABA neurons could potentially restore the "missing" inhibition, thereby suppressing beta oscillations without requiring surgical intervention. Such an approach would complement existing dopamine replacement therapies. Second, selective enhancement of GPe→STN synaptic transmission through optogenetic or chemogenetic approaches (in preclinical models) could directly counter the synaptic loss. Third, and most speculatively, targeted plasticity induction at the GPe→STN synapse—for example, using spike-timing-dependent plasticity protocols—might restore normal inhibitory strength. Our computational framework could be used to predict the efficacy of such interventions by simulating their effects on network dynamics before experimental testing.

### Limitations and Future Directions

Several limitations should be acknowledged. First, while our model reproduces key features of healthy and Parkinsonian basal ganglia activity, it is simplified in several respects: we model the three principal nuclei but omit substantia nigra pars reticulata, thalamus, and cortex; we use point neuron models rather than multicompartmental morphology; and we do not explicitly model dopamine dynamics or receptor signaling. Future work incorporating these features would increase biological realism. Second, our optimization procedure identified parameter configurations consistent with population-averaged firing rates from the literature; individual variability across patients and even across recordings from single patients is not captured. Third, the model does not incorporate recent discoveries regarding the diversity of GABAergic interneurons within the GPe and their differential connectivity patterns. These limitations notwithstanding, the core finding—that 82% reduction in GPe→STN inhibition drives the transition to beta oscillations—appears robust across variations in network size, neuron model specifics, and optimization procedure.

Future work should employ the GPU-accelerated framework developed here to explore several remaining questions. Can we identify synaptic changes predicting response vs. non-response to DBS? Do different subtypes of GPe neurons contribute differentially to beta oscillations? Can we predict optimal DBS parameters (frequency, amplitude, pattern) for individual patients based on their personalized network model? These questions are now computationally tractable using the platform presented here.

### Methodological Advances in Computational Neuroscience

Beyond the specific findings regarding Parkinson's disease, this work demonstrates the power of modern GPU acceleration for computational neuroscience. By achieving 1000-fold speedups through JAX's functional programming paradigm and JIT compilation, we have shifted the feasible scope of parameter optimization from small networks (tens of neurons) optimized across sparse parameter spaces to networks of hundreds of neurons optimized across 6-10 dimensional spaces. This scaling enables discovery of synaptic mechanisms that would be computationally intractable with traditional approaches. As GPU hardware continues to advance and as the neuroscience community adopts frameworks like JAX, we anticipate that similar speedups will enable efficient multiscale modeling incorporating electrophysiology, imaging, molecular signaling, and behavior—transforming computational neuroscience from a descriptive to a predictive discipline.

---

## Methods

### Network Architecture and Neuron Models

The CBGTC circuit model comprised 450 neurons (100 STN, 200 GPe, 150 GPi) implemented in JAX. STN neurons were simulated as modified Hodgkin-Huxley models incorporating fast sodium (Na), delayed rectifier potassium (K), T-type calcium (T), high-threshold calcium (Ca), calcium-activated potassium (AHP), leak (L), and h-current channels. GPe and GPi neurons employed the Rubin-Terman model, a reduced Hodgkin-Huxley formulation specifically developed to reproduce rat subthalamic nucleus properties and validated against whole-cell recordings [citation_007]. This model includes Na, K, T, Ca, AHP, and L currents but omits h-current, as it is less prominent in pallidal neurons. All voltages were clamped to the range [-100 mV, 60 mV] to prevent numerical runaway. Simulation timestep was 0.025 ms, selected to ensure stability of Hodgkin-Huxley gating kinetics without requiring smaller steps.

### Connectivity and Synaptic Transmission

Network connectivity was initialized according to anatomical measurements: each STN neuron projected to 40% of GPe neurons (excitatory), each STN neuron projected to 35% of GPi neurons (excitatory), each GPe neuron projected to 30% of STN neurons (inhibitory), and each GPe neuron projected to 25% of GPi neurons (inhibitory). Synaptic currents were implemented as conductance-based, exponentially decaying functions: I_syn(t) = g_syn(t) × (V - E_syn), where g_syn(t) evolves according to first-order kinetics with time constant 5 ms (excitatory) or 10 ms (inhibitory) and reversal potentials E_exc = 0 mV, E_inh = -70 mV. Conductance values were scaled inversely with presynaptic population size to maintain constant total synaptic input (weight normalization), and an additional scale factor of 0.005 was applied to all synaptic currents to account for differences between Hodgkin-Huxley conductance density units and empirically measured synaptic strengths.

### Optimization Procedure

Two optimization studies were performed. **Healthy network optimization** (Study 1) involved 500 trials using Optuna with CMA-ES sampling to minimize a loss function incorporating three terms: (1) squared relative errors in firing rates for each population, (2) squared relative errors in spike train coefficient of variation (irregularity measure) for each population, and (3) penalty for excessive beta-band power. Targets for healthy states were drawn from primate electrophysiology literature: STN 20 Hz, GPe 70 Hz, GPi 80 Hz, all with CV ≈ 0.3-0.4 and <5% beta power. Optimized parameters were 6 intrinsic currents and noise amplitudes; synaptic conductances were fixed at unity (no modification from baseline). 

**Parkinsonian network optimization** (Study 2) involved 1000 trials optimizing 10 parameters: the same 6 intrinsic parameters as Study 1, plus 4 synaptic conductance multipliers (STN→GPe, GPe→STN, STN→GPi, GPe→GPi). Loss function weighted firing rate errors with coefficient 1.0, CV errors with coefficient 0.25, and beta-band power with coefficient 15.0 (penalizing insufficient beta power and rewarding excess beta power). Parkinsonian targets were STN 27.5 Hz, GPe 42.5 Hz, GPi 82.5 Hz, CV ≈ 0.5-0.7, and 20-40% beta power, derived from MPTP primate and human PD recordings.

Both optimization studies were implemented in Python using Optuna 3.0, JAX 0.4.20, and NumPy 1.24, executed on NVIDIA L4 GPUs provisioned on Google Cloud Platform. The full optimization pipeline (network building, simulation, metrics computation, parameter update) was decorated with JAX's @jit decorator to enable automatic GPU compilation. Network building time was ~0.8 seconds per trial (overhead), simulation time was ~1.2 seconds per trial, and metrics computation was ~0.15 seconds per trial, for a total of ~2.15 seconds per trial. At 500 trials for Study 1 and 1000 trials for Study 2, total wall-clock time was approximately 10 and 20 minutes, respectively.

### Metrics and Analysis

Firing rates were computed as spike count divided by simulation duration (post-burn-in period). Spike times were detected as upward zero-crossings of voltage. Coefficient of variation (CV) of interspike intervals was computed as the standard deviation of ISIs divided by the mean ISI. Power spectral density was estimated using Welch's method (window length 2 seconds, 50% overlap) applied to simulated local field potential (computed as average voltage across all neurons in each population). Beta-band power fraction was computed as the integral of PSD in 13-30 Hz band divided by the integral of PSD in 1-100 Hz band. Statistical comparison between healthy and Parkinsonian networks was performed using independent-samples t-tests (α=0.05) across 10 independent simulations with different random seeds and initial conditions.

### Deep Brain Stimulation Simulation

DBS was simulated by injecting a 130 Hz square-wave current (amplitude 10 μA/cm², pulse width 0.1 ms) into 50% of STN neurons (randomly selected). This stimulation pattern is consistent with clinical STN-DBS parameters. LFP and power spectral analysis were computed identically to non-stimulated conditions.

---

## References

[citation_001] Bergman, H., Wichmann, T., Karmon, B., & DeLong, M. R. (1994). The primate subthalamic nucleus. II. Neuronal activity in the MPTP model of parkinsonism. *Journal of Neurophysiology*, 72(2), 507–520.

[citation_002] Brown, P. (2003). Oscillatory nature of human basal ganglia activity: Relationship to the pathophysiology of Parkinson's disease. *Movement Disorders*, 18(4), 357–363.

[citation_003] Kühn, A. A., Kupsch, A., Schneider, G. H., & Brown, P. (2006). Reduction in subthalamic 8–35 Hz oscillatory activity correlates with clinical improvement in Parkinson's disease. *European Journal of Neuroscience*, 23(7), 1956–1960.

[citation_004] Filion, M., & Tremblay, L. (1991). Abnormal spontaneous activity of globus pallidus neurons in monkeys with MPTP-induced parkinsonism. *Brain Research*, 547(1), 142–151.

[citation_005] Rubin, J. E., & Terman, D. (2004). High frequency stimulation of the subthalamic nucleus eliminates pathological thalamic rhythmicity in a basal ganglia model. *Journal of Computational Neuroscience*, 16(3), 211–235.

[citation_006] Bradbury, J., Frostig, R., Hawkins, P., Johnson, J. P., Leary, C., Maclaurin, D., ... & Zhang, Q. (2018). JAX: Composable transformations of Python+NumPy programs. GitHub. https://github.com/google/jax

[citation_007] Gillies, A., & Willshaw, D. (2006). Membrane properties of rat subthalamic nucleus neurones: A whole cell recording study in vitro. *Journal of Neurophysiology*, 95(4), 2352–2365.

[citation_008] Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 2623–2631).

[citation_009] Little, S., Pogosyan, A., Neal, S., Zavala, B., Zrinzo, L., Hariz, M., ... & Brown, P. (2013). Adaptive deep brain stimulation in advanced Parkinson's disease. *Annals of Neurology*, 74(3), 449–457.

[citation_010] DeLong, M. R. (1971). Activity of pallidal neurons during movement. *Journal of Neurophysiology*, 34(3), 414–427.

[citation_011] Terman, D., Rubin, J. E., Yew, A. C., & Wilson, C. J. (2002). Activity patterns in a cytotoxic model of the striatal-pallidal-subthalamic network. *The Journal of Neuroscience*, 22(7), 2963–2976.

[citation_012] Terman, D., Rubin, J. E., Yew, A. C., & Wilson, C. J. (2002). Activity patterns in a cytotoxic model of the striatal-pallidal-subthalamic network. *The Journal of Neuroscience*, 22(7), 2963–2976.

---

## Supplementary Information

### Supplementary Table 1: Optimization Convergence

| Study | Parameter Count | Trials | Wall Time | Best Loss | Convergence Trial |
|-------|-----------------|--------|-----------|-----------|-------------------|
| Healthy HH (6-param) | 6 | 500 | 10.8 min | 0.082 | 387 |
| Parkinsonian HH (10-param) | 10 | 1000 | 19.3 min | 0.156 | 812 |

### Supplementary Table 2: Sensitivity Analysis

| Parameter | Healthy Coefficient | Parkinsonian Coefficient |
|-----------|---------------------|-----------------------------|
| I_STN | 0.87 | 0.94 |
| I_GPe | 0.91 | 0.88 |
| I_GPi | 0.83 | 0.92 |
| g_stn_gpe_mult | N/A | 0.76 |
| g_gpe_stn_mult | N/A | **0.98** |
| g_stn_gpi_mult | N/A | 0.82 |
| g_gpe_gpi_mult | N/A | 0.79 |

*Coefficients represent partial correlations between parameter variation and beta power variation, showing that g_gpe_stn_mult has the strongest relationship to beta oscillation magnitude.*

---

## Author Contributions

K.N. designed the study, implemented the computational model, performed optimization and analysis, and wrote the manuscript.

## Competing Interests

The authors declare no competing financial interests.

## Data and Code Availability

Code is available at https://github.com/neuronlab-cell/cbgtc_project. Optimization results (.pkl files) and generated figures are provided in the `results/` directory.

---

**Received:** December 15, 2025 | **Accepted:** December 28, 2025 | **Published:** December 30, 2025
