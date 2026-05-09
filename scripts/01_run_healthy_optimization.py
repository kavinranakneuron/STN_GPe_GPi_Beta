"""Run the healthy CMA-ES optimization study.

Thin wrapper around bgnet.optimize.run_study. Default config is
``configs/healthy.yaml``; override with --config <path> for smoke or
sweep runs.

Outputs land in ``results/<study_name>/<YYYYMMDD_HHMMSS>/`` with the
verbatim config, log.txt, optuna_study.db, results.pkl, metadata.json,
and a figures/ subdirectory. The directory is the unit of provenance —
every figure or table downstream points back to one of these.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/01_run_healthy_optimization.py` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bgnet.config import from_yaml, write_yaml
from bgnet.optimize import run_study
from bgnet.results import RunDir


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/healthy.yaml",
                   type=Path, help="Path to study YAML")
    p.add_argument("--results-root", default="results", type=Path,
                   help="Root directory for run outputs")
    args = p.parse_args()

    cfg = from_yaml(args.config)
    if cfg.condition != "healthy":
        raise SystemExit(
            f"Config {args.config} has condition={cfg.condition!r}; "
            f"this script is only for the healthy study. Use script 02 for PD."
        )

    run_dir = RunDir.create(cfg.name, results_root=args.results_root)
    write_yaml(cfg, run_dir.config_yaml)
    run_dir.write_metadata({"script": Path(__file__).name,
                            "config_path": str(args.config)})
    logger = run_dir.configure_logging(name="bgnet")
    logger.info("Starting healthy optimization: study=%s, n_trials=%d, "
                "duration=%.1f ms, dt=%.4f ms, run_dir=%s",
                cfg.name, cfg.n_trials, cfg.duration_ms, cfg.dt_ms, run_dir.root)
    logger.info("Constraint mode: %s", cfg.constraint_mode)

    summary = run_study(cfg, run_dir)
    run_dir.finalize_metadata(status="completed",
                              extra={"summary": {k: summary.get(k) for k in
                                                 ("n_complete", "n_feasible",
                                                  "n_infeasible", "n_failed",
                                                  "feasibility_rate",
                                                  "best_loss_feasible",
                                                  "best_loss_any",
                                                  "wall_clock_s")}})

    logger.info("Done: %d/%d feasible (rate=%.2f), best_feasible=%.4f, "
                "wall=%.1fs", summary["n_feasible"], summary["n_complete"],
                summary["feasibility_rate"],
                (summary["best_loss_feasible"] if summary["best_loss_feasible"]
                 is not None else float("nan")),
                summary["wall_clock_s"])
    print(f"Results written to: {run_dir.root}")


if __name__ == "__main__":
    main()
