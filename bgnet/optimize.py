"""CMA-ES optimization driver via Optuna.

The 13-parameter search space (4 synaptic conductances, 3 tonic drives,
3 OU means, 3 OU sigmas) is sampled by Optuna's ``CmaEsSampler``. Each
trial: sample params -> build NetworkConfig -> run one simulation ->
compute population observables -> compute loss and beta constraint ->
report to Optuna.

Constraint handling. The default mode (``constraint_mode = "constraint"``
in StudyConfig) uses Optuna's native constraint API: each trial calls
``trial.set_user_attr("constraint", (c_beta,))`` and the sampler is
constructed with ``constraints_func=_read_constraint`` so CmaEsSampler's
constraint-aware variant is engaged. The fallback
(``constraint_mode = "penalty"``) folds the constraint into the loss
via ``bgnet.objective.loss_with_beta_penalty``.

The smoke-test driver in scripts/00_smoke_optimize.py reports on which
mode behaves sensibly; the headline runs in scripts/01, /02 read that
choice from their config files.

Per AGENTS.md §4.8:
    Initial mean: midpoint of bounds for each parameter.
    Initial sigma: 1/4 of bound range for each parameter
        (in CmaEsSampler's normalized [0, 1] internal space, this is sigma0=0.25).
    Population size: Optuna default (~11 for n_dim=13).
"""
from __future__ import annotations

import logging
import pickle
import time
from dataclasses import asdict
from typing import Any, Sequence

import numpy as np
import optuna
from optuna.samplers import CmaEsSampler
from optuna.trial import FrozenTrial, TrialState

from bgnet.config import Bound, StudyConfig
from bgnet.network import NetworkConfig, simulate
from bgnet.objective import (
    Targets,
    beta_constraint,
    is_feasible,
    loss,
    loss_with_beta_penalty,
    metrics_from_sim,
)
from bgnet.results import RunDir


# Suppress Optuna's per-trial INFO chatter; we log structured progress ourselves.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _targets_from_cfg(cfg: StudyConfig) -> Targets:
    return Targets(
        rate_stn_Hz=cfg.target_rate_stn_Hz,
        rate_gpe_Hz=cfg.target_rate_gpe_Hz,
        rate_gpi_Hz=cfg.target_rate_gpi_Hz,
        cv_stn=cfg.target_cv_stn,
        cv_gpe=cfg.target_cv_gpe,
        cv_gpi=cfg.target_cv_gpi,
        beta_threshold=cfg.beta_threshold,
        condition=cfg.condition,
    )


def _build_network_cfg(cfg: StudyConfig, params: dict, ou_seed: int) -> NetworkConfig:
    """Translate a sampled parameter dict + study config to a NetworkConfig
    consumable by ``bgnet.network.simulate``."""
    return NetworkConfig(
        n_stn=cfg.n_stn, n_gpe=cfg.n_gpe, n_gpi=cfg.n_gpi,
        K_stn_gpe=cfg.K_stn_gpe, K_gpe_stn=cfg.K_gpe_stn,
        K_stn_gpi=cfg.K_stn_gpi, K_gpe_gpi=cfg.K_gpe_gpi,
        delay_stn_gpe_ms=cfg.delay_stn_gpe_ms,
        delay_stn_gpi_ms=cfg.delay_stn_gpi_ms,
        delay_gpe_stn_ms=cfg.delay_gpe_stn_ms,
        delay_gpe_gpi_ms=cfg.delay_gpe_gpi_ms,
        dt_ms=cfg.dt_ms,
        g_stn_gpe=params["g_stn_gpe"], g_stn_gpi=params["g_stn_gpi"],
        g_gpe_stn=params["g_gpe_stn"], g_gpe_gpi=params["g_gpe_gpi"],
        I_drive_stn=params["I_drive_stn"], I_drive_gpe=params["I_drive_gpe"],
        I_drive_gpi=params["I_drive_gpi"],
        mu_stn=params["mu_stn"], mu_gpe=params["mu_gpe"], mu_gpi=params["mu_gpi"],
        sigma_stn=params["sigma_stn"], sigma_gpe=params["sigma_gpe"],
        sigma_gpi=params["sigma_gpi"],
        network_seed=cfg.network_seed,
        ou_seed=ou_seed,
        init_seed=cfg.init_seed,
    )


def _suggest_params(trial: optuna.Trial, cfg: StudyConfig) -> dict:
    """Sample one parameter vector from the configured bounds."""
    params = {}
    for name in cfg.bounds.names():
        b: Bound = cfg.bounds.get(name)
        params[name] = trial.suggest_float(name, b.low, b.high)
    return params


def _read_constraint(trial: FrozenTrial) -> Sequence[float]:
    """Return the per-trial constraint vector for CmaEsSampler.

    We store ``c_beta`` on each trial as user attr ``constraint``; this
    function reads that back. Unset (e.g. on a failed trial) is treated
    as a hard violation so the trial doesn't silently look feasible.
    """
    val = trial.user_attrs.get("constraint")
    if val is None:
        return (1e6,)
    return tuple(val)


def _initial_mean(cfg: StudyConfig) -> dict[str, float]:
    return {name: cfg.bounds.get(name).midpoint() for name in cfg.bounds.names()}


# ---------------------------------------------------------------------------
# Run a study
# ---------------------------------------------------------------------------

def run_study(cfg: StudyConfig, run_dir: RunDir,
              progress_log_every: int = 25) -> dict[str, Any]:
    """Run a CMA-ES optimization with the given config, writing all
    artefacts into ``run_dir``. Returns a summary dict (best params,
    best loss, feasibility breakdown, wall-clock).

    The function is the single entry point used by both
    ``scripts/01_run_healthy_optimization.py`` and
    ``scripts/02_run_pd_optimization.py`` (the latter just passes a
    different config).

    Constraint handling. AGENTS.md asked for ``CmaEsSampler(constraints_func=...)``
    but Optuna 4.x's ``CmaEsSampler`` does not accept ``constraints_func``
    (only multi-objective samplers like NSGA-II / TPE do). When the config
    requests ``constraint_mode="constraint"`` we attempt the constraint API
    and fall back to the penalty formulation (AGENTS.md §3 Phase 2 step 3
    explicit fallback) if that signature is not available. The summary
    records both the *requested* and *effective* mode.
    """
    logger = logging.getLogger("bgnet.optimize")
    targets = _targets_from_cfg(cfg)
    requested_mode = cfg.constraint_mode

    sampler_kwargs: dict[str, Any] = {
        "seed": cfg.cma_seed,
        "consider_pruned_trials": False,
        "x0": _initial_mean(cfg),
        "sigma0": 0.25,                # 1/4 of normalized [0,1] range
    }

    effective_mode = requested_mode
    if requested_mode == "constraint":
        try:
            sampler = CmaEsSampler(**sampler_kwargs,
                                   constraints_func=_read_constraint)
            logger.info("CmaEsSampler engaged with constraints_func; "
                        "constraint mode active.")
        except TypeError:
            logger.warning(
                "Optuna %s CmaEsSampler does not accept constraints_func; "
                "falling back to penalty formulation (AGENTS.md §3 Phase 2 "
                "step 3 explicit fallback). Trials still record c_beta on "
                "user_attrs so feasibility is filtered post-hoc.",
                optuna.__version__,
            )
            sampler = CmaEsSampler(**sampler_kwargs)
            effective_mode = "penalty"
    else:
        sampler = CmaEsSampler(**sampler_kwargs)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=run_dir.storage_url(),
        study_name=cfg.name,
        load_if_exists=False,
    )

    n_complete = 0
    n_feasible = 0
    n_infeasible = 0
    n_failed = 0

    def objective(trial: optuna.Trial) -> float:
        nonlocal n_complete, n_feasible, n_infeasible, n_failed
        params = _suggest_params(trial, cfg)
        net_cfg = _build_network_cfg(cfg, params,
                                     ou_seed=cfg.ou_seed_base + trial.number)
        try:
            sim = simulate(net_cfg, cfg.duration_ms)
            metrics = metrics_from_sim(
                sim, burn_in_ms=cfg.burn_in_ms,
                beta_band=cfg.beta_band, broadband=cfg.broadband,
                lfp_proxy=cfg.lfp_proxy,
                hp_cutoff_hz=cfg.lfp_hp_cutoff_hz,
                hp_order=cfg.lfp_hp_order,
            )
        except Exception as e:
            n_failed += 1
            logger.warning("trial %d simulation failed: %r", trial.number, e)
            raise

        c = beta_constraint(metrics, targets)
        feasible = c <= 0.0
        n_complete += 1
        if feasible:
            n_feasible += 1
        else:
            n_infeasible += 1

        # Persist per-trial metrics for later analysis
        for k, v in metrics.as_dict().items():
            trial.set_user_attr(k, float(v))
        trial.set_user_attr("constraint", (float(c),))
        trial.set_user_attr("feasible", bool(feasible))

        if effective_mode == "constraint":
            l = loss(metrics, targets, cfg.w_rate, cfg.w_cv)
        else:
            l = loss_with_beta_penalty(metrics, targets,
                                       cfg.w_rate, cfg.w_cv,
                                       cfg.beta_penalty_weight)

        if (trial.number + 1) % progress_log_every == 0:
            best = best_feasible_loss(study)
            logger.info(
                "trial %4d/%d  loss=%.4f  c_beta=%+.4f  feasible=%s  "
                "best_feasible=%.4f  rates=(%.1f, %.1f, %.1f) Hz  beta=%.3f",
                trial.number + 1, cfg.n_trials, l, c, feasible,
                best if best is not None else float("nan"),
                metrics.rate_stn, metrics.rate_gpe, metrics.rate_gpi,
                metrics.beta_stn,
            )
        return l

    t0 = time.time()
    study.optimize(objective, n_trials=cfg.n_trials,
                   show_progress_bar=False, gc_after_trial=True)
    elapsed = time.time() - t0

    summary = _summarize(study, cfg, n_complete, n_feasible, n_infeasible,
                         n_failed, elapsed, effective_mode, requested_mode)
    run_dir.results_pkl.write_bytes(pickle.dumps(summary))
    return summary


# ---------------------------------------------------------------------------
# Result post-processing
# ---------------------------------------------------------------------------

def best_feasible_loss(study: optuna.Study) -> float | None:
    """Return the best loss among feasible completed trials, or None.

    Reads the per-trial ``feasible`` user-attr written by the objective
    closure in ``run_study``.
    """
    best = None
    for t in study.trials:
        if t.state != TrialState.COMPLETE:
            continue
        if not t.user_attrs.get("feasible", False):
            continue
        if best is None or t.value < best:
            best = t.value
    return best


def _summarize(study: optuna.Study, cfg: StudyConfig, n_complete: int,
               n_feasible: int, n_infeasible: int, n_failed: int,
               elapsed: float, effective_mode: str,
               requested_mode: str) -> dict[str, Any]:
    """Build the dict that is pickled to results.pkl."""
    feasible_trials = [t for t in study.trials
                       if t.state == TrialState.COMPLETE
                       and t.user_attrs.get("feasible", False)]
    infeasible_trials = [t for t in study.trials
                         if t.state == TrialState.COMPLETE
                         and not t.user_attrs.get("feasible", False)]

    if feasible_trials:
        best = min(feasible_trials, key=lambda t: t.value)
    elif study.trials:
        # No feasible trials; best-infeasible is reported but flagged
        best = min((t for t in study.trials if t.state == TrialState.COMPLETE),
                   key=lambda t: t.value, default=None)
    else:
        best = None

    summary = {
        "study_name": cfg.name,
        "condition": cfg.condition,
        "n_trials_requested": cfg.n_trials,
        "n_complete": n_complete,
        "n_feasible": n_feasible,
        "n_infeasible": n_infeasible,
        "n_failed": n_failed,
        "feasibility_rate": (n_feasible / n_complete) if n_complete else 0.0,
        "wall_clock_s": elapsed,
        "constraint_mode_requested": requested_mode,
        "constraint_mode_effective": effective_mode,
        "best_loss_feasible": (
            best.value if best and best.user_attrs.get("feasible", False)
            else None
        ),
        "best_loss_any": best.value if best else None,
        "best_params": dict(best.params) if best else None,
        "best_metrics": (
            {k: best.user_attrs.get(k) for k in
             ("rate_stn", "rate_gpe", "rate_gpi",
              "cv_stn", "cv_gpe", "cv_gpi", "beta_stn")}
            if best else None
        ),
        "best_feasible_flag": bool(best.user_attrs.get("feasible", False))
                               if best else False,
        "config": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else None,
    }
    # Per-trial table (lightweight) for downstream analysis
    summary["trials"] = [
        {
            "number": t.number,
            "value": t.value,
            "state": str(t.state),
            "params": dict(t.params),
            **{k: t.user_attrs.get(k) for k in
               ("rate_stn", "rate_gpe", "rate_gpi",
                "cv_stn", "cv_gpe", "cv_gpi", "beta_stn",
                "feasible", "constraint")},
        }
        for t in study.trials
    ]
    return summary
