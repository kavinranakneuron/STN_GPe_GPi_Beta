"""Objective function and constraint formulation for the optimizer.

Loss combines two terms (per AGENTS.md §3 Phase 2 step 2):

    loss = w_rate * sum_{pop in {STN, GPe, GPi}} ((rate_pop - target_rate_pop)
                                                  / target_rate_pop)**2
         + w_cv   * sum_{pop in {STN, GPe, GPi}} (cv_pop - target_cv_pop)**2

Defaults: w_rate = 1.0, w_cv = 0.2. CV is intentionally a soft term
(low weight) per the rebuild scope decision: CV targets are
order-of-magnitude estimates, not measured numbers.

Beta band [13, 30] Hz fraction is treated as a **constraint**, not a loss
term, to avoid the "bounded-value target" failure mode of the original
paper (over-prioritization of a flat metric at the expense of rates).
Optuna treats constraint values <= 0 as feasible, > 0 as infeasible.
Sign conventions:

    healthy: c_beta = actual_beta - 0.05    (feasible iff actual_beta < 0.05)
    pd     : c_beta = 0.15 - actual_beta    (feasible iff actual_beta > 0.15)

Fallback if the constraint formulation proves intractable in early
testing (excessive infeasible trials, no convergence): a one-sided
threshold-with-bonus penalty is added to the loss instead. See
``loss_with_beta_penalty`` and ``run_study(..., constraint_mode="penalty")``
in ``bgnet/optimize.py``. The PHASE 2 smoke test surfaces which mode is
in use.

Inputs to the loss function are the dict returned by
``bgnet.network.simulate``. The objective module imports nothing from
Optuna; the optimization driver wires the constraint output into Optuna's
``constraints_func`` API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from bgnet.observables import (
    beta_fraction,
    cv_isi,
    firing_rate,
    population_rate_proxy,
    synaptic_current_lfp_proxy,
    vm_lfp_proxy,
)

ConditionType = Literal["healthy", "pd"]
LFPProxyKind = Literal["vm", "population_rate", "synaptic_current"]


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Targets:
    """Per-population firing-rate and CV targets and the beta-fraction
    constraint threshold for a given condition (healthy or PD).

    Defaults reflect AGENTS.md §4.1, §4.2, §4.3 (Tachibana et al. 2014
    primate MPTP for rates; Stein & Bar-Gad 2013 for the beta band; CV
    targets are order-of-magnitude per the rebuild scope decision).
    """
    rate_stn_Hz: float
    rate_gpe_Hz: float
    rate_gpi_Hz: float
    cv_stn: float
    cv_gpe: float
    cv_gpi: float
    beta_threshold: float       # the cutoff value (0.05 for healthy, 0.15 for PD)
    condition: ConditionType


def healthy_targets() -> Targets:
    return Targets(
        rate_stn_Hz=20.0, rate_gpe_Hz=65.0, rate_gpi_Hz=67.0,
        cv_stn=0.4, cv_gpe=0.35, cv_gpi=0.20,
        beta_threshold=0.05,
        condition="healthy",
    )


def pd_targets() -> Targets:
    return Targets(
        rate_stn_Hz=27.0, rate_gpe_Hz=41.0, rate_gpi_Hz=63.0,
        cv_stn=0.6, cv_gpe=0.40, cv_gpi=0.30,
        beta_threshold=0.15,
        condition="pd",
    )


# ---------------------------------------------------------------------------
# Per-trial metric extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrialMetrics:
    """Everything we measure from one simulation run, for both the
    objective and downstream logging."""
    rate_stn: float
    rate_gpe: float
    rate_gpi: float
    cv_stn: float
    cv_gpe: float
    cv_gpi: float
    beta_stn: float       # beta fraction of the STN population-rate proxy

    def as_dict(self) -> dict:
        return {
            "rate_stn": self.rate_stn, "rate_gpe": self.rate_gpe, "rate_gpi": self.rate_gpi,
            "cv_stn": self.cv_stn, "cv_gpe": self.cv_gpe, "cv_gpi": self.cv_gpi,
            "beta_stn": self.beta_stn,
        }


def _stn_proxy_trace(sim_output: dict, proxy: LFPProxyKind,
                     burn_in_ms: float, bin_ms: float,
                     hp_cutoff_hz: float, hp_order: int
                     ) -> tuple[np.ndarray, float]:
    """Pick the STN LFP-proxy trace + sample dt based on ``proxy``."""
    dt_ms = sim_output["dt_ms"]
    if proxy == "vm":
        vmean = np.asarray(sim_output["vmean_stn"])
        return vm_lfp_proxy(vmean, dt_ms, burn_in_ms, bin_ms,
                            hp_cutoff_hz, hp_order)
    if proxy == "population_rate":
        sp = np.asarray(sim_output["spikes_stn"])
        return population_rate_proxy(sp, dt_ms, bin_ms, burn_in_ms)
    if proxy == "synaptic_current":
        isyn = np.asarray(sim_output["isyn_mean_stn"])
        return synaptic_current_lfp_proxy(isyn, dt_ms, burn_in_ms, bin_ms,
                                          hp_cutoff_hz, hp_order)
    raise ValueError(
        f"unknown lfp_proxy {proxy!r}; expected 'vm', 'population_rate', "
        f"or 'synaptic_current'")


def metrics_from_sim(sim_output: dict, burn_in_ms: float = 100.0,
                     proxy_bin_ms: float = 1.0,
                     beta_band: tuple[float, float] = (13.0, 30.0),
                     broadband: tuple[float, float] = (1.0, 100.0),
                     lfp_proxy: LFPProxyKind = "vm",
                     hp_cutoff_hz: float = 2.0, hp_order: int = 4,
                     ) -> TrialMetrics:
    """Compute per-population rate, CV, and STN β fraction from one
    ``bgnet.network.simulate`` output.

    The β fraction is computed from the LFP proxy selected by
    ``lfp_proxy`` (default ``"vm"`` — high-pass-filtered mean Vm). The
    ``"population_rate"`` alternate reproduces the pre-rebuild proxy and
    is kept for the LFP-proxy-comparison validation. The
    ``"synaptic_current"`` alternate is also available.

    Default ``beta_band`` matches AGENTS.md §4.3 (13-30 Hz).
    """
    dt_ms = sim_output["dt_ms"]
    sp_stn = np.asarray(sim_output["spikes_stn"])
    sp_gpe = np.asarray(sim_output["spikes_gpe"])
    sp_gpi = np.asarray(sim_output["spikes_gpi"])
    proxy_stn, bin_dt = _stn_proxy_trace(
        sim_output, lfp_proxy, burn_in_ms, proxy_bin_ms,
        hp_cutoff_hz, hp_order,
    )
    beta_stn, _, _ = beta_fraction(proxy_stn, bin_dt,
                                   beta_band=beta_band, broadband=broadband)
    return TrialMetrics(
        rate_stn=firing_rate(sp_stn, dt_ms, burn_in_ms),
        rate_gpe=firing_rate(sp_gpe, dt_ms, burn_in_ms),
        rate_gpi=firing_rate(sp_gpi, dt_ms, burn_in_ms),
        cv_stn=cv_isi(sp_stn, dt_ms, burn_in_ms),
        cv_gpe=cv_isi(sp_gpe, dt_ms, burn_in_ms),
        cv_gpi=cv_isi(sp_gpi, dt_ms, burn_in_ms),
        beta_stn=beta_stn,
    )


# ---------------------------------------------------------------------------
# Loss and constraint
# ---------------------------------------------------------------------------

def rate_loss_term(m: TrialMetrics, t: Targets) -> float:
    """Sum of squared relative errors on the three rates."""
    rs = ((m.rate_stn - t.rate_stn_Hz) / t.rate_stn_Hz) ** 2
    rg = ((m.rate_gpe - t.rate_gpe_Hz) / t.rate_gpe_Hz) ** 2
    ri = ((m.rate_gpi - t.rate_gpi_Hz) / t.rate_gpi_Hz) ** 2
    return float(rs + rg + ri)


def cv_loss_term(m: TrialMetrics, t: Targets) -> float:
    """Sum of squared absolute errors on the three CVs."""
    cs = (m.cv_stn - t.cv_stn) ** 2
    cg = (m.cv_gpe - t.cv_gpe) ** 2
    ci = (m.cv_gpi - t.cv_gpi) ** 2
    return float(cs + cg + ci)


def loss(m: TrialMetrics, t: Targets, w_rate: float = 1.0,
         w_cv: float = 0.2) -> float:
    """Weighted-sum scalar loss. Beta is *not* in this expression — see
    ``beta_constraint`` for the constraint."""
    return w_rate * rate_loss_term(m, t) + w_cv * cv_loss_term(m, t)


def beta_constraint(m: TrialMetrics, t: Targets) -> float:
    """Beta-band constraint, sign-converted for Optuna's "<= 0 is
    feasible" convention.

    healthy: c_beta = actual - 0.05  (feasible iff actual < 0.05)
    pd     : c_beta = 0.15 - actual  (feasible iff actual > 0.15)
    """
    if t.condition == "healthy":
        return float(m.beta_stn - t.beta_threshold)
    if t.condition == "pd":
        return float(t.beta_threshold - m.beta_stn)
    raise ValueError(f"unknown condition {t.condition!r}")


def is_feasible(m: TrialMetrics, t: Targets) -> bool:
    return beta_constraint(m, t) <= 0.0


# ---------------------------------------------------------------------------
# Penalty fallback (used iff the proper constraint formulation proves
# intractable; controlled by the optimizer driver, not this module).
# ---------------------------------------------------------------------------

def loss_with_beta_penalty(m: TrialMetrics, t: Targets,
                           w_rate: float = 1.0, w_cv: float = 0.2,
                           beta_penalty_weight: float = 5.0) -> float:
    """Fallback formulation: add a one-sided penalty proportional to
    constraint violation onto the standard loss. No penalty when feasible.

    AGENTS.md §3 Phase 2 step 3 explicitly authorizes this fallback if
    the constraint formulation produces excessive infeasible trials. The
    smoke test (Phase 2 step 8) decides which mode the headline runs use.
    """
    base = loss(m, t, w_rate, w_cv)
    c = beta_constraint(m, t)
    return float(base + beta_penalty_weight * max(0.0, c))
