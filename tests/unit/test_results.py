"""Unit tests for bgnet.results (RunDir layout, metadata, logging)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from bgnet.results import RunDir, collect_metadata


def test_create_lays_out_files(tmp_path: Path):
    rd = RunDir.create("smoke", results_root=tmp_path, timestamp="20260509_010203")
    assert rd.root == tmp_path / "smoke" / "20260509_010203"
    assert rd.root.is_dir()
    assert rd.figures_dir.is_dir()
    # The files do not exist yet (only the directory tree)
    assert not rd.config_yaml.exists()
    assert not rd.log_txt.exists()
    assert not rd.results_pkl.exists()
    assert not rd.metadata_json.exists()


def test_create_refuses_existing_timestamp(tmp_path: Path):
    RunDir.create("smoke", results_root=tmp_path, timestamp="20260509_010203")
    with pytest.raises(FileExistsError):
        RunDir.create("smoke", results_root=tmp_path, timestamp="20260509_010203")


def test_collect_metadata_has_required_fields():
    md = collect_metadata()
    for key in ("bgnet_version", "python", "platform", "hostname",
                "started_at_utc"):
        assert key in md
    # gpu/git fields are optional; just confirm presence
    assert "gpu" in md and "git_sha" in md and "git_dirty" in md


def test_logging_writes_to_log_txt(tmp_path: Path):
    rd = RunDir.create("smoke", results_root=tmp_path, timestamp="20260509_120000")
    logger = rd.configure_logging(name="bgnet.test", level=logging.INFO)
    logger.info("hello world")
    logger.warning("careful")
    # Detach so the file handle is flushed/closed
    for h in list(logger.handlers):
        h.flush()
    text = rd.log_txt.read_text()
    assert "hello world" in text and "careful" in text


def test_metadata_finalize_writes_status_and_wallclock(tmp_path: Path):
    rd = RunDir.create("smoke", results_root=tmp_path, timestamp="20260509_120100")
    rd.write_metadata({"n_trials": 7})
    rd.finalize_metadata(status="completed")
    md = json.loads(rd.metadata_json.read_text())
    assert md["status"] == "completed"
    assert md["n_trials"] == 7
    assert "wall_clock_s" in md
    assert "finished_at_utc" in md


def test_storage_url_is_sqlite_path(tmp_path: Path):
    rd = RunDir.create("smoke", results_root=tmp_path, timestamp="20260509_120200")
    url = rd.storage_url()
    assert url.startswith("sqlite:///")
    assert url.endswith("optuna_study.db")
