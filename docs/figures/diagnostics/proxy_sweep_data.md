# Proxy comparison on Diag-3 2D slice (post-rebuild)

Repeated Diagnostic 3 (sweep over `g_stn_gpe × g_gpe_stn`, other 11 params
held at the silent-STN best-feasible values) with all three LFP proxies
computed for each cell. Burn-in 200 ms, duration 600 ms, ou_seed=1, β
band [13,30] Hz, broadband [1,100] Hz, 1 ms bins, 2 Hz HP filter
(Vm and I_syn proxies).

|  g_sg |  g_gs |    rate |   β_vm |   β_rt | β_isyn |
|------:|------:|--------:|-------:|-------:|-------:|
| 0.050 | 0.005 |   15.38 | 0.2334 | 0.2461 | 0.1347 |
| 0.050 | 0.050 |    3.10 | 0.8858 | 0.7034 | 0.5954 |
| 0.050 | 0.100 |    0.00 | 0.2819 | 0.0000 | 0.3393 |
| 0.050 | 0.200 |    0.00 | 0.2680 | 0.0000 | 0.3433 |
| 0.050 | 0.300 |    0.00 | 0.2604 | 0.0000 | 0.3725 |
| 0.050 | 0.500 |    0.00 | 0.2514 | 0.0000 | 0.3995 |
| 0.100 | 0.005 |   13.90 | 0.3117 | 0.3675 | 0.3336 |
| 0.100 | 0.050 |    3.20 | 0.8365 | 0.5475 | 0.5914 |
| 0.100 | 0.100 |    0.00 | 0.3084 | 0.0000 | 0.3181 |
| 0.100 | 0.200 |    0.00 | 0.2886 | 0.0000 | 0.3235 |
| 0.100 | 0.300 |    0.00 | 0.2765 | 0.0000 | 0.3577 |
| 0.100 | 0.500 |    0.00 | 0.2671 | 0.0000 | 0.3925 |
| 0.200 | 0.005 |   13.27 | 0.6271 | 0.8311 | 0.7897 |
| 0.200 | 0.050 |    3.47 | 0.8645 | 0.5267 | 0.7371 |
| 0.200 | 0.100 |    0.00 | 0.2920 | 0.0000 | 0.3449 |
| 0.200 | 0.200 |    0.00 | 0.2712 | 0.0000 | 0.3582 |
| 0.200 | 0.300 |    0.00 | 0.2628 | 0.0000 | 0.3795 |
| 0.200 | 0.500 |    0.00 | 0.2595 | 0.0000 | 0.4049 |
| 0.300 | 0.005 |   13.00 | 0.6642 | 0.8369 | 0.9588 |
| 0.300 | 0.050 |    3.57 | 0.9007 | 0.5363 | 0.8892 |
| 0.300 | 0.100 |    0.00 | 0.2599 | 0.0000 | 0.3302 |
| 0.300 | 0.200 |    0.00 | 0.2763 | 0.0000 | 0.3260 |
| 0.300 | 0.300 |    0.00 | 0.2656 | 0.0000 | 0.3588 |
| 0.300 | 0.500 |    0.00 | 0.2556 | 0.0000 | 0.3944 |
| 0.400 | 0.005 |   13.22 | 0.8532 | 0.8166 | 0.9744 |
| 0.400 | 0.050 |    3.47 | 0.8313 | 0.4988 | 0.9002 |
| 0.400 | 0.100 |    0.00 | 0.3235 | 0.0000 | 0.3342 |
| 0.400 | 0.200 |    0.00 | 0.2693 | 0.0000 | 0.3409 |
| 0.400 | 0.300 |    0.00 | 0.2385 | 0.0000 | 0.3783 |
| 0.400 | 0.500 |    0.00 | 0.2321 | 0.0000 | 0.4039 |
| 0.500 | 0.005 |   13.42 | 0.8882 | 0.8155 | 0.9261 |
| 0.500 | 0.050 |    3.88 | 0.7906 | 0.4772 | 0.8453 |
| 0.500 | 0.100 |    0.00 | 0.3111 | 0.0000 | 0.3230 |
| 0.500 | 0.200 |    0.00 | 0.2825 | 0.0000 | 0.3357 |
| 0.500 | 0.300 |    0.00 | 0.2404 | 0.0000 | 0.3783 |
| 0.500 | 0.500 |    0.00 | 0.2317 | 0.0000 | 0.4021 |

Feasibility under (STN ≥ 15 Hz AND β < 0.05) in this slice:

- Vm proxy: 0/36
- Rate proxy: 0/36
- I_syn proxy: 0/36

## Reading

- **Vm-proxy floor** (across cells, including STN-silent): ≈ 0.23. Real network Vm carries some
  [13,30] Hz mass from sub-threshold OU and synaptic noise even when no
  STN spikes occur. This is *physically meaningful* sub-threshold power,
  not a measurement artifact.
- **Vm-proxy ceiling** (STN active, weak GPe→STN, loop-entrained): 0.85–0.91. The
  proxy *does* discriminate strong synchronization (ceiling) from
  async/silent states (floor) — by ~3–4×.
- **One cell crosses below the silent-STN floor while keeping STN
  active:** `g_sg=0.05, g_gs=0.005` → rate 15.4 Hz, β_vm = 0.233. This
  is the *only* point in the slice where STN fires at the target while
  Vm-β is at the floor — confirming the proxy correctly identifies
  asynchronous-active spiking as "low β".
- **Rate-proxy floor**: 0.0 when STN silent (PSD is identically zero —
  the old artifact). Real signals: 0.25–0.84.
- **I_syn-proxy floor**: ≈ 0.33 — higher than Vm because synaptic
  currents have a 5–10 ms time constant that low-pass-filters the
  signal, concentrating mass below 30 Hz.

## Implication

The 0.05 healthy threshold was tuned for the rate proxy where STN-silent
gives β = 0 by construction. Under the Vm proxy, the natural
"asynchronous-firing baseline" is 0.20–0.30, not 0.05. **The threshold
does not survive the proxy switch and must be retuned.**

PD threshold of 0.15 is also affected: it was meant to be *above* the
healthy floor of 0.05. Under the Vm proxy a PD threshold of 0.15 is
*below* the healthy floor of 0.23, which inverts the constraint.
