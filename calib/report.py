"""Summary + overlay artifacts for the calibration pass.

Outputs:
  - per-channel overlay PNGs (sim vs real) for each driving log
  - a YAML diff between current and proposed calibration
  - a short text summary printed to stdout
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import yaml

import numpy as np

from ..sim.calibration import (
    IRCalibration, IRSideCalibration, IMUCalibration, LidarCalibration,
    SensorCalibration,
)
from .ir_xlsx import IRXlsxFit
from .tfluna_xlsx import TFLunaXlsxFit
from .log_io import Log
from .noise_fit import SensorNoiseFit
from .imu_bias import IMUBiasEstimate
from .latency import LatencyResult
from .dynamics_fit import FitResult, PARAM_NAMES
from .replay import replay


def plot_overlays(logs: list[Log], fit: FitResult, imu_cal: IMUCalibration,
                  bias: tuple[float, float, float], out_dir: Path) -> list[Path]:
    """Write one overlay PNG per log: sim_gz/ax/ay vs real."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = {k: float(v) for k, v in zip(PARAM_NAMES, fit.x)}
    out: list[Path] = []
    bias_gz, bias_ax, bias_ay = bias
    for log in logs:
        trace = replay(log, imu_cal=imu_cal, overrides=overrides)
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        t = log.t_s - log.t_s[0]
        axes[0].plot(t, log.gyro_z_lsb  - bias_gz, label="real", alpha=0.7)
        axes[0].plot(t, trace.gyro_z_lsb,          label="sim",  alpha=0.8)
        axes[0].set_ylabel("gyro_z (LSB)")
        axes[0].legend()
        axes[1].plot(t, log.accel_x_lsb - bias_ax, alpha=0.7)
        axes[1].plot(t, trace.accel_x_lsb,          alpha=0.8)
        axes[1].set_ylabel("accel_x (LSB)")
        axes[2].plot(t, log.accel_y_lsb - bias_ay, alpha=0.7)
        axes[2].plot(t, trace.accel_y_lsb,          alpha=0.8)
        axes[2].set_ylabel("accel_y (LSB)")
        axes[2].set_xlabel("t (s)")
        fig.suptitle(f"{log.path.name} — fit loss = {fit.loss:.3f}")
        fig.tight_layout()
        path = out_dir / f"{log.path.stem}_overlay.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        out.append(path)
    return out


def yaml_diff(current: SensorCalibration, proposed: SensorCalibration) -> str:
    cur = asdict(current)
    pro = asdict(proposed)
    lines: list[str] = []

    def walk(prefix: str, a, b) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            keys = sorted(set(a) | set(b))
            for k in keys:
                walk(f"{prefix}.{k}" if prefix else k, a.get(k), b.get(k))
        elif isinstance(a, list) and isinstance(b, list):
            if a != b:
                lines.append(f"  {prefix}: {a}  ->  {b}")
        else:
            if a != b:
                lines.append(f"  {prefix}: {a}  ->  {b}")

    walk("", cur, pro)
    return "\n".join(lines) if lines else "  (no changes)"


def summary_lines(
    ir_fit: Optional[IRXlsxFit],
    noise: Optional[SensorNoiseFit],
    imu_bias: Optional[IMUBiasEstimate],
    latency: Optional[LatencyResult],
    dynamics: Optional[FitResult],
) -> list[str]:
    lines: list[str] = []
    if ir_fit is not None:
        l = ir_fit.left; r = ir_fit.right
        lines.append(
            f"IR fit: left  a={l.a:.1f}  b={l.b:.1f}  c={l.c:.1f}  "
            f"thr={l.adc_threshold}  RMSE={l.rmse_mm:.1f} mm  (n={l.n_points})"
        )
        lines.append(
            f"IR fit: right a={r.a:.1f}  b={r.b:.1f}  c={r.c:.1f}  "
            f"thr={r.adc_threshold}  RMSE={r.rmse_mm:.1f} mm  (n={r.n_points})"
        )
    if noise is not None:
        lines.append(
            f"Noise: IR L/R mm std = {noise.ir_left_mm_std:.2f} / {noise.ir_right_mm_std:.2f};"
            f" TF L/F/R mm std = {noise.tf_left_mm_std:.2f} / {noise.tf_front_mm_std:.2f}"
            f" / {noise.tf_right_mm_std:.2f}"
        )
    if imu_bias is not None:
        lines.append(
            f"IMU bias (LSB): gyro {imu_bias.gyro_bias}  accel {imu_bias.accel_bias}"
            f"  (n={imu_bias.n_samples})"
        )
        lines.append(
            f"IMU noise (LSB): gyro_z σ={imu_bias.gyro_std_lsb:.1f}"
            f"  accel_x σ={imu_bias.accel_x_std_lsb:.1f}"
            f"  accel_y σ={imu_bias.accel_y_std_lsb:.1f}"
        )
        if imu_bias.notes:
            lines.append(f"  {imu_bias.notes}")
    if latency is not None:
        lines.append(
            f"Latency: steer→(-gyro_z) {latency.steer_to_gyro_ms:.0f} ms"
            f" (r={latency.steer_peak_corr:.2f});"
            f" throttle→accel_y {latency.throttle_to_accel_ms:.0f} ms"
            f" (r={latency.throttle_peak_corr:.2f})"
        )
    if dynamics is not None:
        lines.append(f"Dynamics fit (n={dynamics.n_evals}):")
        for name, val in zip(PARAM_NAMES, dynamics.x):
            lines.append(f"  {name:<22s} {val:.4f}")
        lines.append(f"  loss (train): {dynamics.loss:.4f}")
        if dynamics.holdout_loss is not None:
            lines.append(f"  loss (holdout): {dynamics.holdout_loss:.4f}")
    return lines


def apply_to_calibration(
    current: SensorCalibration,
    ir_fit: Optional[IRXlsxFit] = None,
    noise: Optional[SensorNoiseFit] = None,
    imu_bias: Optional[IMUBiasEstimate] = None,
    tfluna_fit: Optional[TFLunaXlsxFit] = None,
) -> SensorCalibration:
    """Return a new SensorCalibration with proposed updates merged in."""
    out = current.copy()
    if ir_fit is not None:
        out.ir.left = IRSideCalibration(
            a=float(ir_fit.left.a), b=float(ir_fit.left.b),
            c=float(ir_fit.left.c), adc_threshold=int(ir_fit.left.adc_threshold),
        )
        out.ir.right = IRSideCalibration(
            a=float(ir_fit.right.a), b=float(ir_fit.right.b),
            c=float(ir_fit.right.c), adc_threshold=int(ir_fit.right.adc_threshold),
        )
    if noise is not None:
        # Average the per-side ADC std (both sides use the same front-end).
        adc_stds = [s for s in (noise.ir_left_adc_std, noise.ir_right_adc_std) if s > 0]
        if adc_stds:
            out.ir.adc_noise_std = float(np.mean(adc_stds))
        # TFLuna uses the plan's "one std for all" — average the three.
        tf_stds = [s for s in (noise.tf_left_mm_std, noise.tf_front_mm_std,
                               noise.tf_right_mm_std) if s > 0]
        if tf_stds:
            out.lidar.noise_std_cm = float(np.mean(tf_stds)) / 10.0
    if imu_bias is not None:
        out.imu.gyro_bias = list(imu_bias.gyro_bias)
        out.imu.accel_bias = list(imu_bias.accel_bias)
        out.imu.gyro_noise_lsb = float(imu_bias.gyro_std_lsb)
        out.imu.accel_noise_lsb = float(
            0.5 * (imu_bias.accel_x_std_lsb + imu_bias.accel_y_std_lsb)
        )
    if tfluna_fit is not None and tfluna_fit.per_lidar:
        # Single shared LidarCalibration for now — average per-lidar fits.
        # If per-unit divergence later proves to matter for sim-to-real, the
        # schema can be promoted to per-lidar fields without breaking older
        # YAMLs.
        scales = [s.scale for s in tfluna_fit.per_lidar.values()]
        biases_cm = [s.bias_cm for s in tfluna_fit.per_lidar.values()]
        noises_cm = [s.noise_std_cm for s in tfluna_fit.per_lidar.values()]
        out.lidar.distance_scale = float(np.mean(scales))
        out.lidar.distance_bias_cm = float(np.mean(biases_cm))
        # Use the larger of (xlsx residual std, in-log noise std) so the
        # simulated noise is at least as wide as the real-world noise floor.
        if noises_cm:
            out.lidar.noise_std_cm = max(out.lidar.noise_std_cm, float(np.mean(noises_cm)))
    return out
