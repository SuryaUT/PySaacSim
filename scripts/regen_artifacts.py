"""Regenerate calibration artifacts (overlays + dynamics.json + summary.txt)
from a *known* dynamics fit vector, without redoing CMA-ES.

Use case: the upstream `calibrate_from_log.py` crashed mid-write (e.g. UTF-8
encoding bug) but the optimizer already converged and `proposed.yaml` landed.
This script lets us recover the visual artifacts without burning another
30 minutes on CMA-ES.

The fit result must be supplied either from the monitor log of the original
run (--x ...) or by pointing at a previous artifact's dynamics.json
(--from-json ...).

All file writes use UTF-8 explicitly so the σ unicode in summary lines does
not crash on Windows cp1252.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


sys.path.insert(0, str(_repo_root().parent))


from PySaacSim.sim.calibration import SensorCalibration  # noqa: E402
from PySaacSim.calib.log_io import load_log, load_logs  # noqa: E402
from PySaacSim.calib.ir_xlsx import fit_xlsx as fit_ir_xlsx  # noqa: E402
from PySaacSim.calib.imu_bias import estimate_bias  # noqa: E402
from PySaacSim.calib.noise_fit import fit_noise  # noqa: E402
from PySaacSim.calib.latency import estimate as estimate_latency  # noqa: E402
from PySaacSim.calib.dynamics_fit import (  # noqa: E402
    FitResult, PARAM_NAMES, _log_loss,
)
from PySaacSim.calib import report as report_mod  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ir-xlsx", type=Path, required=True)
    ap.add_argument("--steady-csv", type=Path, required=True)
    ap.add_argument("--drive-csvs", type=Path, nargs="+", required=True)
    ap.add_argument("--proposed-yaml", type=Path, required=True,
                    help="proposed.yaml from the crashed run")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="will be created; overlays/, dynamics.json, summary.txt land here")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--x", type=float, nargs=6,
                     help="6-element fit vector (order: " + ", ".join(PARAM_NAMES) + ")")
    grp.add_argument("--from-json", type=Path,
                     help="path to a prior dynamics.json to lift x from")
    ap.add_argument("--loss", type=float, default=None,
                    help="known training loss for the summary header (cosmetic)")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_json:
        data = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        x = np.array(data["x"], dtype=float)
        loss = data.get("loss_train", args.loss or 0.0)
    else:
        x = np.array(args.x, dtype=float)
        loss = args.loss if args.loss is not None else 0.0

    print(f"Using dynamics x = {x.tolist()}  (loss={loss:.2f})")

    # Re-derive the rest of Phase 1-3 outputs (they're cheap relative to CMA-ES).
    print("[1/4] Refitting IR + IMU + noise (fast)...")
    ir_fit = fit_ir_xlsx(args.ir_xlsx)
    steady = load_log(args.steady_csv)
    proposed = SensorCalibration.from_yaml(args.proposed_yaml)
    bias = estimate_bias(steady)
    noise = fit_noise(steady, proposed.ir)

    print("[2/4] Latency...")
    drive_logs = load_logs(list(args.drive_csvs))
    latency = estimate_latency(drive_logs)

    print("[3/4] Per-log loss + holdout (replays at fitted params)...")
    sigma = (
        max(bias.gyro_std_lsb, 1.0),
        max(bias.accel_x_std_lsb, 1.0),
        max(bias.accel_y_std_lsb, 1.0),
    )
    bias_tuple = (
        bias.gyro_bias[2],
        bias.accel_bias[0],
        bias.accel_bias[1],
    )
    train = drive_logs[:-1]
    holdout = drive_logs[-1]
    per_log = [
        _log_loss(L, x, proposed.imu, sigma, *bias_tuple)
        for L in train
    ]
    holdout_loss = _log_loss(holdout, x, proposed.imu, sigma, *bias_tuple)

    fit = FitResult(
        x=x, loss=float(loss),
        n_evals=0, per_log_loss=per_log, holdout_loss=holdout_loss, cov=None,
    )

    print("[4/4] Writing artifacts (overlays + dynamics.json + summary.txt)...")
    overlays_dir = args.out_dir / "overlays"
    written = report_mod.plot_overlays(
        drive_logs, fit, proposed.imu,
        bias=bias_tuple, out_dir=overlays_dir,
    )
    print(f"  wrote {len(written)} overlay(s) to {overlays_dir}")

    dyn_json = {
        "param_names": list(PARAM_NAMES),
        "x": x.tolist(),
        "loss_train": float(loss),
        "per_log_loss": per_log,
        "loss_holdout": holdout_loss,
        "n_evals": 0,
    }
    (args.out_dir / "dynamics.json").write_text(
        json.dumps(dyn_json, indent=2), encoding="utf-8"
    )

    summary_lines = report_mod.summary_lines(ir_fit, noise, bias, latency, fit)
    (args.out_dir / "summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print("\n=== summary ===")
    for line in summary_lines:
        print(line)
    print(f"\nholdout / train ratio: {holdout_loss / max(loss, 1e-9):.3f}")
    print(f"\nArtifacts in: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
