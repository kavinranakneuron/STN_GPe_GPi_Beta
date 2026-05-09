"""Unit tests for bgnet.objective.

Verify the loss and constraint formulations on hand-picked TrialMetrics
where the answer is computable in closed form.
"""
from __future__ import annotations

import math

from bgnet.objective import (
    Targets,
    TrialMetrics,
    beta_constraint,
    cv_loss_term,
    healthy_targets,
    is_feasible,
    loss,
    loss_with_beta_penalty,
    pd_targets,
    rate_loss_term,
)


def _exact(targets, beta_stn=0.01) -> TrialMetrics:
    """Metrics that exactly hit the targets — should give zero loss."""
    return TrialMetrics(
        rate_stn=targets.rate_stn_Hz,
        rate_gpe=targets.rate_gpe_Hz,
        rate_gpi=targets.rate_gpi_Hz,
        cv_stn=targets.cv_stn,
        cv_gpe=targets.cv_gpe,
        cv_gpi=targets.cv_gpi,
        beta_stn=beta_stn,
    )


# ---------------------------------------------------------------------------
# Targets defaults match AGENTS.md §4
# ---------------------------------------------------------------------------

def test_healthy_target_values():
    t = healthy_targets()
    assert t.rate_stn_Hz == 20.0
    assert t.rate_gpe_Hz == 65.0
    assert t.rate_gpi_Hz == 67.0
    assert t.cv_stn == 0.4
    assert t.beta_threshold == 0.05
    assert t.condition == "healthy"


def test_pd_target_values():
    t = pd_targets()
    assert t.rate_stn_Hz == 27.0
    assert t.rate_gpe_Hz == 41.0
    assert t.rate_gpi_Hz == 63.0
    assert t.cv_stn == 0.6
    assert t.beta_threshold == 0.15
    assert t.condition == "pd"


# ---------------------------------------------------------------------------
# Loss is zero at the targets, positive elsewhere
# ---------------------------------------------------------------------------

def test_loss_zero_at_exact_targets():
    t = healthy_targets()
    m = _exact(t)
    assert rate_loss_term(m, t) == 0.0
    assert cv_loss_term(m, t) == 0.0
    assert loss(m, t) == 0.0


def test_loss_increases_when_rates_off():
    t = healthy_targets()
    m_off = TrialMetrics(
        rate_stn=t.rate_stn_Hz * 1.5,
        rate_gpe=t.rate_gpe_Hz, rate_gpi=t.rate_gpi_Hz,
        cv_stn=t.cv_stn, cv_gpe=t.cv_gpe, cv_gpi=t.cv_gpi,
        beta_stn=0.01,
    )
    # 50% off on STN rate -> rate term = 0.5**2 = 0.25
    assert math.isclose(rate_loss_term(m_off, t), 0.25)
    # CV unchanged -> cv term = 0
    assert cv_loss_term(m_off, t) == 0.0
    # Total weighted loss = 1.0 * 0.25 + 0.2 * 0 = 0.25
    assert math.isclose(loss(m_off, t), 0.25)


def test_cv_term_uses_absolute_squared():
    t = healthy_targets()
    m = TrialMetrics(
        rate_stn=t.rate_stn_Hz, rate_gpe=t.rate_gpe_Hz, rate_gpi=t.rate_gpi_Hz,
        cv_stn=t.cv_stn + 0.1,  # +0.1 absolute
        cv_gpe=t.cv_gpe, cv_gpi=t.cv_gpi,
        beta_stn=0.01,
    )
    assert math.isclose(cv_loss_term(m, t), 0.01)


def test_default_weights_match_spec():
    """w_cv = 0.2 (per AGENTS.md §3 Phase 2 step 2 docstring)."""
    t = healthy_targets()
    # Single-population CV deviation by 1.0 unit
    m = TrialMetrics(
        rate_stn=t.rate_stn_Hz, rate_gpe=t.rate_gpe_Hz, rate_gpi=t.rate_gpi_Hz,
        cv_stn=t.cv_stn + 1.0,
        cv_gpe=t.cv_gpe, cv_gpi=t.cv_gpi,
        beta_stn=0.01,
    )
    # cv_term = 1.0; loss = w_rate*0 + 0.2*1.0 = 0.2
    assert math.isclose(loss(m, t), 0.2)


# ---------------------------------------------------------------------------
# Beta constraint sign convention
# ---------------------------------------------------------------------------

def test_healthy_constraint_feasible_when_low():
    t = healthy_targets()
    m = _exact(t, beta_stn=0.02)   # below 0.05 → feasible
    assert beta_constraint(m, t) < 0
    assert is_feasible(m, t)


def test_healthy_constraint_infeasible_when_high():
    t = healthy_targets()
    m = _exact(t, beta_stn=0.10)   # above 0.05 → infeasible
    assert beta_constraint(m, t) > 0
    assert not is_feasible(m, t)


def test_pd_constraint_feasible_when_high():
    t = pd_targets()
    m = _exact(t, beta_stn=0.20)   # above 0.15 → feasible
    assert beta_constraint(m, t) < 0
    assert is_feasible(m, t)


def test_pd_constraint_infeasible_when_low():
    t = pd_targets()
    m = _exact(t, beta_stn=0.05)   # below 0.15 → infeasible
    assert beta_constraint(m, t) > 0
    assert not is_feasible(m, t)


# ---------------------------------------------------------------------------
# Penalty fallback
# ---------------------------------------------------------------------------

def test_penalty_fallback_no_penalty_when_feasible():
    t = healthy_targets()
    m = _exact(t, beta_stn=0.02)   # feasible
    assert math.isclose(loss_with_beta_penalty(m, t), 0.0)


def test_penalty_fallback_adds_violation_when_infeasible():
    t = healthy_targets()
    m = _exact(t, beta_stn=0.10)   # infeasible by 0.05
    base = loss(m, t)
    pen = loss_with_beta_penalty(m, t, beta_penalty_weight=5.0)
    assert math.isclose(pen - base, 5.0 * 0.05)
