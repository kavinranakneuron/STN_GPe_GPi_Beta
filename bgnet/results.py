"""Results manager: every run writes a self-contained directory.

Each invocation of an optimization script produces

    results/<study_name>/<YYYYMMDD_HHMMSS>/
        config.yaml         verbatim copy of the input config (post-validation)
        log.txt             structured logging output for the run
        optuna_study.db     full Optuna SQLite study
        results.pkl         best params + complete trial metadata
        metadata.json       git SHA, package version, host info, wall-clock
        figures/            generated figures for this run

The contract this module exposes is small on purpose: open a ``RunDir``
once at the top of a script, log into it, save artefacts into it, finish.
There is no mutable global state.

This is the artefact that solves the "which run produced this number"
problem in the original codebase. Every figure and table in the paper
points back to one of these directories.
"""
from __future__ import annotations

import json
import logging
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bgnet


# ---------------------------------------------------------------------------
# Provenance gathering
# ---------------------------------------------------------------------------

def _git_sha() -> str | None:
    """Return the current git HEAD SHA if available, else None.

    We do not raise on absence: a run started outside a git checkout
    (e.g. from an installed wheel) is a legitimate state to record.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=2.0, check=False,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=2.0, check=False,
        )
        if out.returncode != 0:
            return None
        return bool(out.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _gpu_info() -> dict | None:
    """Best-effort capture of the JAX device the run will execute on."""
    try:
        import jax
        devices = [str(d) for d in jax.devices()]
        return {"jax_devices": devices, "default_backend": jax.default_backend()}
    except Exception:
        return None


def collect_metadata(extra: dict | None = None) -> dict:
    """Snapshot of provenance fields written into ``metadata.json``."""
    md = {
        "bgnet_version": bgnet.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "gpu": _gpu_info(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        md.update(extra)
    return md


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------

@dataclass
class RunDir:
    """Handle to one results directory.

    Attribute names match files on disk so callers can treat them as
    paths directly.
    """
    root: Path
    config_yaml: Path
    log_txt: Path
    optuna_db: Path
    results_pkl: Path
    metadata_json: Path
    figures_dir: Path

    started: float = field(default_factory=time.time)

    @classmethod
    def create(cls, study_name: str, results_root: str | Path = "results",
               timestamp: str | None = None) -> "RunDir":
        """Create the directory tree for a new run and return a handle.

        ``timestamp`` is the YYYYMMDD_HHMMSS suffix; defaults to "now".
        """
        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(results_root) / study_name / ts
        root.mkdir(parents=True, exist_ok=False)
        figures = root / "figures"
        figures.mkdir()
        return cls(
            root=root,
            config_yaml=root / "config.yaml",
            log_txt=root / "log.txt",
            optuna_db=root / "optuna_study.db",
            results_pkl=root / "results.pkl",
            metadata_json=root / "metadata.json",
            figures_dir=figures,
        )

    def configure_logging(self, level: int = logging.INFO,
                          name: str = "bgnet") -> logging.Logger:
        """Attach a file-and-stream logger to this run dir.

        Format: ``%(asctime)s %(levelname)-7s %(name)s — %(message)s`` so
        log.txt grep lines like "infeasible" / "best loss" pop out.
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        for h in list(logger.handlers):
            logger.removeHandler(h)
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(self.log_txt, mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        return logger

    def write_metadata(self, extra: dict | None = None) -> None:
        md = collect_metadata(extra)
        self.metadata_json.write_text(json.dumps(md, indent=2))

    def finalize_metadata(self, status: str = "completed",
                          extra: dict | None = None) -> None:
        """Append wall-clock and status to metadata.json after the run."""
        md = json.loads(self.metadata_json.read_text()) if self.metadata_json.exists() \
             else collect_metadata()
        md["status"] = status
        md["wall_clock_s"] = time.time() - self.started
        md["finished_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if extra:
            md.update(extra)
        self.metadata_json.write_text(json.dumps(md, indent=2))

    def storage_url(self) -> str:
        """SQLite URL for Optuna's ``storage=`` argument."""
        return f"sqlite:///{self.optuna_db.resolve()}"
