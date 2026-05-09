"""Run a PD CMA-ES optimization study.

Takes the config path as a CLI argument so the same script handles
the asymmetric and symmetric variants:

    python scripts/02_run_pd_optimization.py --config configs/pd_asymmetric.yaml
    python scripts/02_run_pd_optimization.py --config configs/pd_symmetric.yaml

Outputs land in ``results/<study_name>/<YYYYMMDD_HHMMSS>/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bgnet.config import from_yaml, write_yaml
from bgnet.optimize import run_study
from bgnet.results import RunDir


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, type=Path,
                   help="Path to PD study YAML (asymmetric or symmetric)")
    p.add_argument("--results-root", default="results", type=Path,
                   help="Root directory for run outputs")
    args = p.parse_args()

    cfg = from_yaml(args.config)
    if cfg.condition != "pd":
        raise SystemExit(
            f"Config {args.config} has condition={cfg.condition!r}; "
            f"this script is only for PD studies. Use script 01 for healthy."
        )

    run_dir = RunDir.create(cfg.name, results_root=args.results_root)
    write_yaml(cfg, run_dir.config_yaml)
    run_dir.write_metadata({"script": Path(__file__).name,
                            "config_path": str(args.config)})
    logger = run_dir.configure_logging(name="bgnet")
    logger.info("Starting PD optimization: study=%s, n_trials=%d, "
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
