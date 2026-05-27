"""YAML configuration loader for optimization studies.

Each study (healthy, pd_asymmetric, pd_symmetric, sweeps, etc.) is fully
described by a single YAML file. The loader produces a typed
``StudyConfig`` dataclass and validates ranges, required fields, and
internal consistency at load time.

Why dataclasses, not free dicts: typos like ``mu_sgn`` instead of
``mu_stn`` should fail loudly at config load, not silently as zero
during a 30-minute optimization. The ``from_yaml`` constructor only
accepts known keys; unknown keys raise.

Each study config records the package version
(``bgnet.__version__``) at load time so the saved metadata can pin
exact provenance.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

import bgnet


ConditionType = Literal["healthy", "pd"]
ConstraintMode = Literal["constraint", "penalty"]
LFPProxyKind = Literal["vm", "population_rate", "synaptic_current"]


@dataclass(frozen=True)
class Bound:
    """Inclusive lower/upper bound for one optimization parameter."""
    low: float
    high: float

    def __post_init__(self):
        if self.low > self.high:
            raise ValueError(f"bound low ({self.low}) > high ({self.high})")

    def midpoint(self) -> float:
        return 0.5 * (self.low + self.high)

    def quarter_range(self) -> float:
        return 0.25 * (self.high - self.low)


@dataclass(frozen=True)
class SearchBounds:
    """The 13 search bounds in absolute biophysical units (mS/cm² for
    conductances, µA/cm² for drives/means/sigmas)."""
    g_stn_gpe: Bound
    g_stn_gpi: Bound
    g_gpe_stn: Bound
    g_gpe_gpi: Bound
    I_drive_stn: Bound
    I_drive_gpe: Bound
    I_drive_gpi: Bound
    mu_stn: Bound
    mu_gpe: Bound
    mu_gpi: Bound
    sigma_stn: Bound
    sigma_gpe: Bound
    sigma_gpi: Bound

    def names(self) -> list[str]:
        return [f.name for f in dataclasses.fields(self)]

    def get(self, name: str) -> Bound:
        return getattr(self, name)


@dataclass(frozen=True)
class StudyConfig:
    """One end-to-end optimization study.

    Defaults reflect AGENTS.md §3 Phase 3 step 1 (healthy headline) and
    §4 Reference values. PD configs override ``condition``,
    ``targets_*``, and the bounds; healthy uses these defaults.
    """
    # Identity
    name: str
    condition: ConditionType

    # Targets (defaults track AGENTS.md §4.1 healthy; PD overrides them)
    target_rate_stn_Hz: float = 20.0
    target_rate_gpe_Hz: float = 65.0
    target_rate_gpi_Hz: float = 67.0
    target_cv_stn: float = 0.4
    target_cv_gpe: float = 0.35
    target_cv_gpi: float = 0.20
    beta_threshold: float = 0.05

    # Network sizes (AGENTS.md §4.5 optimization defaults)
    n_stn: int = 100
    n_gpe: int = 200
    n_gpi: int = 150

    # Connectivity (AGENTS.md §4.5)
    K_stn_gpe: int = 15
    K_gpe_stn: int = 14
    K_stn_gpi: int = 30
    K_gpe_gpi: int = 10

    # Synaptic delays (AGENTS.md §4.6)
    delay_stn_gpe_ms: float = 5.0
    delay_stn_gpi_ms: float = 5.0
    delay_gpe_stn_ms: float = 8.0
    delay_gpe_gpi_ms: float = 5.0

    # Numerical (AGENTS.md §4.7)
    dt_ms: float = 0.025
    duration_ms: float = 400.0
    burn_in_ms: float = 100.0

    # Search bounds — explicit per study; default below corresponds to
    # the symmetric healthy bounds in AGENTS.md §4.4. PD asymmetric
    # studies override the four conductance bounds.
    bounds: SearchBounds = field(default_factory=lambda: SearchBounds(
        g_stn_gpe=Bound(0.005, 0.5), g_stn_gpi=Bound(0.005, 0.5),
        g_gpe_stn=Bound(0.005, 0.5), g_gpe_gpi=Bound(0.005, 0.5),
        I_drive_stn=Bound(-5.0, 5.0), I_drive_gpe=Bound(-5.0, 5.0),
        I_drive_gpi=Bound(-5.0, 5.0),
        mu_stn=Bound(-5.0, 5.0), mu_gpe=Bound(-5.0, 5.0),
        mu_gpi=Bound(-5.0, 5.0),
        sigma_stn=Bound(0.0, 5.0), sigma_gpe=Bound(0.0, 5.0),
        sigma_gpi=Bound(0.0, 5.0),
    ))

    # Loss
    w_rate: float = 1.0
    w_cv: float = 0.2
    constraint_mode: ConstraintMode = "constraint"
    beta_penalty_weight: float = 5.0   # only used if constraint_mode == "penalty"

    # Spectral analysis (AGENTS.md §4.3, default 13-30 Hz per the
    # pre-Phase-3 sanity check; see docs/network_beta_sanity_check.md).
    beta_band: tuple[float, float] = (13.0, 30.0)
    broadband: tuple[float, float] = (1.0, 100.0)

    # LFP proxy feeding the β fraction (and constraint). Default ``"vm"``
    # = high-pass-filtered mean Vm; see ``docs/silent_stn_diagnostics.md``
    # for why the population-rate proxy was retired as primary.
    # Alternates kept for proxy-comparison validation runs.
    lfp_proxy: LFPProxyKind = "vm"
    lfp_hp_cutoff_hz: float = 2.0
    lfp_hp_order: int = 4

    # Optimizer
    n_trials: int = 1000
    cma_seed: int = 42                  # CMA-ES sampler seed
    network_seed: int = 42              # connectivity seed; fixed during opt
    init_seed: int = 7                  # neuron initial-state seed
    ou_seed_base: int = 1               # OU seed; fresh per trial as base + i

    # Provenance — populated at load time
    bgnet_version: str = bgnet.__version__

    def __post_init__(self):
        if self.condition not in ("healthy", "pd"):
            raise ValueError(f"condition must be 'healthy' or 'pd', got {self.condition!r}")
        if self.constraint_mode not in ("constraint", "penalty"):
            raise ValueError(f"constraint_mode must be 'constraint' or 'penalty'")
        if self.lfp_proxy not in ("vm", "population_rate", "synaptic_current"):
            raise ValueError(
                f"lfp_proxy must be 'vm', 'population_rate', or "
                f"'synaptic_current'; got {self.lfp_proxy!r}")
        if self.lfp_hp_cutoff_hz <= 0:
            raise ValueError("lfp_hp_cutoff_hz must be > 0")
        if self.lfp_hp_order < 1:
            raise ValueError("lfp_hp_order must be >= 1")
        if self.n_stn <= 0 or self.n_gpe <= 0 or self.n_gpi <= 0:
            raise ValueError("network sizes must be positive")
        if self.duration_ms <= self.burn_in_ms:
            raise ValueError(
                f"duration_ms ({self.duration_ms}) must exceed burn_in_ms "
                f"({self.burn_in_ms})"
            )
        if self.dt_ms <= 0:
            raise ValueError("dt_ms must be positive")
        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive")
        for label, band in (("beta_band", self.beta_band),
                            ("broadband", self.broadband)):
            if (not isinstance(band, tuple)) or len(band) != 2:
                raise ValueError(
                    f"{label} must be a (low, high) tuple, got {band!r}"
                )
            if band[0] >= band[1]:
                raise ValueError(
                    f"{label} low ({band[0]}) must be < high ({band[1]})"
                )


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

_BOUND_FIELDS = {f.name for f in dataclasses.fields(SearchBounds)}


def _parse_bounds(raw: dict) -> SearchBounds:
    if not isinstance(raw, dict):
        raise ValueError("`bounds` must be a mapping")
    unknown = set(raw) - _BOUND_FIELDS
    if unknown:
        raise ValueError(f"unknown keys in bounds: {sorted(unknown)}")
    missing = _BOUND_FIELDS - set(raw)
    if missing:
        raise ValueError(f"missing keys in bounds: {sorted(missing)}")
    bound_kwargs = {}
    for name in _BOUND_FIELDS:
        v = raw[name]
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            raise ValueError(
                f"bounds.{name} must be a [low, high] pair, got {v!r}"
            )
        bound_kwargs[name] = Bound(low=float(v[0]), high=float(v[1]))
    return SearchBounds(**bound_kwargs)


_STUDY_FIELDS = {f.name for f in dataclasses.fields(StudyConfig)}


def from_dict(d: dict) -> StudyConfig:
    """Build a StudyConfig from a parsed mapping (validates keys).

    ``bgnet_version`` is accepted on input (so a saved-and-reloaded config
    round-trips) but always overwritten with the current package version,
    so a config loaded under a newer bgnet records the new version, not
    the stale one written by the previous run.
    """
    if not isinstance(d, dict):
        raise ValueError("config root must be a mapping")
    unknown = set(d) - _STUDY_FIELDS
    if unknown:
        raise ValueError(f"unknown keys in config: {sorted(unknown)}")
    kwargs: dict[str, Any] = dict(d)
    kwargs.pop("bgnet_version", None)
    if "bounds" in kwargs:
        kwargs["bounds"] = _parse_bounds(kwargs["bounds"])
    # YAML parses tuples-of-floats as lists; normalize for the dataclass.
    for k in ("beta_band", "broadband"):
        if k in kwargs and isinstance(kwargs[k], list):
            kwargs[k] = tuple(kwargs[k])
    return StudyConfig(**kwargs)


def from_yaml(path: str | Path) -> StudyConfig:
    """Load a study config from a YAML file."""
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"empty or invalid YAML: {path}")
    return from_dict(raw)


# ---------------------------------------------------------------------------
# Persistence — copy verbatim into results dir
# ---------------------------------------------------------------------------

def to_yaml_dict(cfg: StudyConfig) -> dict:
    """Render a StudyConfig back to a YAML-friendly mapping (Bound -> [lo, hi],
    band tuples -> [lo, hi] lists)."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        if isinstance(v, SearchBounds):
            out[f.name] = {
                bf.name: [getattr(v, bf.name).low, getattr(v, bf.name).high]
                for bf in dataclasses.fields(SearchBounds)
            }
        elif isinstance(v, tuple):
            out[f.name] = list(v)
        else:
            out[f.name] = v
    return out


def write_yaml(cfg: StudyConfig, path: str | Path) -> None:
    """Write a StudyConfig back to YAML for the results manifest."""
    Path(path).write_text(yaml.safe_dump(to_yaml_dict(cfg), sort_keys=False))
