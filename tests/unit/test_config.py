"""Unit tests for bgnet.config (YAML loader, validation, round-trip)."""
from __future__ import annotations

from pathlib import Path

import pytest

from bgnet.config import (
    Bound,
    SearchBounds,
    StudyConfig,
    from_dict,
    from_yaml,
    to_yaml_dict,
    write_yaml,
)


def _minimal_dict(name="healthy", condition="healthy", **kw):
    return {"name": name, "condition": condition, **kw}


def test_minimal_config_loads_with_defaults():
    cfg = from_dict(_minimal_dict())
    assert cfg.name == "healthy"
    assert cfg.condition == "healthy"
    assert cfg.target_rate_stn_Hz == 20.0
    assert cfg.bounds.g_stn_gpe.low == 0.005
    assert cfg.bounds.g_stn_gpe.high == 0.5


def test_unknown_top_level_key_raises():
    with pytest.raises(ValueError, match="unknown keys"):
        from_dict(_minimal_dict(typo_field=42))


def test_invalid_condition_raises():
    with pytest.raises(ValueError, match="condition must be"):
        from_dict(_minimal_dict(condition="weird"))


def test_invalid_constraint_mode_raises():
    with pytest.raises(ValueError, match="constraint_mode must be"):
        from_dict(_minimal_dict(constraint_mode="weird"))


def test_burnin_must_be_less_than_duration():
    with pytest.raises(ValueError, match="duration_ms"):
        from_dict(_minimal_dict(duration_ms=50.0, burn_in_ms=100.0))


def test_bound_low_gt_high_raises():
    with pytest.raises(ValueError, match="bound low"):
        Bound(low=1.0, high=0.5)


def test_bound_midpoint_and_quarter_range():
    b = Bound(low=-2.0, high=6.0)
    assert b.midpoint() == 2.0
    assert b.quarter_range() == 2.0


def test_bounds_subset_in_yaml_raises():
    with pytest.raises(ValueError, match="missing keys in bounds"):
        from_dict(_minimal_dict(bounds={"g_stn_gpe": [0.005, 0.5]}))


def test_bounds_unknown_key_raises():
    full_bounds = {f: [0.0, 1.0] for f in SearchBounds.__dataclass_fields__}
    full_bounds["banana"] = [0.0, 1.0]
    with pytest.raises(ValueError, match="unknown keys in bounds"):
        from_dict(_minimal_dict(bounds=full_bounds))


def test_bounds_pair_must_have_two_elements():
    full_bounds = {f: [0.0, 1.0] for f in SearchBounds.__dataclass_fields__}
    full_bounds["g_stn_gpe"] = [0.0]   # bad
    with pytest.raises(ValueError, match=r"bounds\.g_stn_gpe"):
        from_dict(_minimal_dict(bounds=full_bounds))


def test_to_yaml_dict_round_trips_through_from_dict(tmp_path: Path):
    full_bounds = {f: [0.005, 0.1] for f in SearchBounds.__dataclass_fields__}
    cfg_in = from_dict(_minimal_dict(bounds=full_bounds, n_trials=42))
    yaml_path = tmp_path / "study.yaml"
    write_yaml(cfg_in, yaml_path)
    cfg_out = from_yaml(yaml_path)
    assert cfg_out.n_trials == 42
    assert cfg_out.bounds.g_stn_gpe == Bound(0.005, 0.1)


def test_pd_targets_via_config():
    cfg = from_dict(_minimal_dict(
        condition="pd", target_rate_stn_Hz=27.0, target_rate_gpe_Hz=41.0,
        target_rate_gpi_Hz=63.0, beta_threshold=0.15,
        target_cv_stn=0.6, target_cv_gpe=0.40, target_cv_gpi=0.30,
    ))
    assert cfg.condition == "pd"
    assert cfg.target_rate_stn_Hz == 27.0
    assert cfg.beta_threshold == 0.15


def test_bgnet_version_recorded():
    cfg = from_dict(_minimal_dict())
    assert isinstance(cfg.bgnet_version, str) and len(cfg.bgnet_version) > 0


def test_bound_names_count_is_thirteen():
    """The 13-parameter search space (4 conductances + 3 drives + 3 mus + 3 sigmas)."""
    bounds = StudyConfig(name="x", condition="healthy").bounds
    assert len(bounds.names()) == 13


# ---------------------------------------------------------------------------
# beta_band wiring
# ---------------------------------------------------------------------------

def test_beta_band_default_is_13_30():
    """Default band picked in §2.2.3 of the rebuild scope."""
    cfg = from_dict(_minimal_dict())
    assert cfg.beta_band == (13.0, 30.0)
    assert cfg.broadband == (1.0, 100.0)


def test_beta_band_yaml_list_roundtrips_as_tuple(tmp_path: Path):
    cfg = from_dict(_minimal_dict(beta_band=[8.0, 15.0]))
    assert cfg.beta_band == (8.0, 15.0)
    yaml_path = tmp_path / "study.yaml"
    write_yaml(cfg, yaml_path)
    cfg_back = from_yaml(yaml_path)
    assert cfg_back.beta_band == (8.0, 15.0)


def test_beta_band_invalid_raises():
    with pytest.raises(ValueError, match=r"beta_band low"):
        from_dict(_minimal_dict(beta_band=[15.0, 8.0]))


def test_broadband_invalid_raises():
    with pytest.raises(ValueError, match=r"broadband low"):
        from_dict(_minimal_dict(broadband=[100.0, 1.0]))
