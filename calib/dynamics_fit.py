"""CMA-ES over the 6-param dynamics vector against the 4 driving CSVs.

Parameter vector (log-spaced sigmas where positive-only):
    MOTOR_MAX_FORCE_N    now 5.0,  bounds (1.0, 20.0)
    MOTOR_LAG_TAU_S      now 0.100, bounds (0.02, 0.5)
    LINEAR_DRAG          now 0.1,  bounds (0.01, 2.0)
    ROLLING_RESIST       now 0.02, bounds (0.0, 0.2)
    SERVO_RAD_PER_SEC    now 6.16, bounds (1.0, 20.0)
    MU_KINETIC           now 0.6,  bounds (0.1, 1.5)

Held fixed: MAX_SPEED_CMS (measured), MASS_KG (measured), INERTIA_KG_M2,
chassis geometry, IMU LSB scales.

Loss (whitened MSE):
    L = w_gz * MSE(sim_gz, real_gz) / σ_gz^2
      + w_ay * MSE(sim_ay, real_ay) / σ_ay^2
      + w_ax * MSE(sim_ax, real_ax) / σ_ax^2
with σ from Phase 2 noise fits and weights (1, 1, 0.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..sim.calibration import IMUCalibration
from .log_io import Log
from .replay import replay


PARAM_NAMES = (
    "MOTOR_MAX_FORCE_N",
    "MOTOR_LAG_TAU_S",
    "LINEAR_DRAG",
    "ROLLING_RESIST",
    "SERVO_RAD_PER_SEC",
    "MU_KINETIC",
    "STEER_TRANSPORT_LAG_S",
    "THROTTLE_TRANSPORT_LAG_S",
)

# Defaults seeded from the v1 (6-param) fit's best point + 30 ms transport lags.
DEFAULT_X0 = np.array(
    [14.22, 0.10, 0.78, 0.0, 5.0, 0.22, 0.030, 0.030], dtype=float
)
BOUNDS = np.array([
    [1.0,  20.0 ],   # MOTOR_MAX_FORCE_N
    [0.02, 0.5  ],   # MOTOR_LAG_TAU_S
    [0.01, 10.0 ],   # LINEAR_DRAG (relaxed upper from 2.0 → 10.0; terminal
                     # velocity from drag balance needs higher drag now that
                     # the hard speed governor is removed)
    [0.0,  0.2  ],   # ROLLING_RESIST
    [4.0,  20.0 ],   # SERVO_RAD_PER_SEC (lower tightened 1.0 → 4.0; below 4
                     # rad/s means we're hiding latency in slew rather than
                     # the new transport lag — datasheet says 6.16 unloaded)
    [0.1,  1.5  ],   # MU_KINETIC
    [0.0,  0.150],   # STEER_TRANSPORT_LAG_S (CAN+PWM+servo-deadband chain)
    [0.0,  0.150],   # THROTTLE_TRANSPORT_LAG_S
], dtype=float)
WEIGHTS = np.array([1.0, 0.5, 1.0], dtype=float)  # (gz, ax, ay)


@dataclass
class FitResult:
    x: np.ndarray                         # fitted parameter vector
    loss: float
    n_evals: int
    per_log_loss: list[float]             # on the *training* logs
    holdout_loss: Optional[float] = None  # on the held-out log (if supplied)
    cov: Optional[np.ndarray] = None      # CMA-ES posterior covariance (for DR)


def _overrides(x: np.ndarray) -> dict:
    return {name: float(val) for name, val in zip(PARAM_NAMES, x)}


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, BOUNDS[:, 0], BOUNDS[:, 1])


def _mse_whitened(
    sim_gz: np.ndarray, sim_ax: np.ndarray, sim_ay: np.ndarray,
    real_gz: np.ndarray, real_ax: np.ndarray, real_ay: np.ndarray,
    sigma: tuple[float, float, float],
) -> float:
    s_gz, s_ax, s_ay = (max(s, 1e-3) for s in sigma)
    err_gz = np.mean((sim_gz - real_gz) ** 2) / (s_gz * s_gz)
    err_ax = np.mean((sim_ax - real_ax) ** 2) / (s_ax * s_ax)
    err_ay = np.mean((sim_ay - real_ay) ** 2) / (s_ay * s_ay)
    return float(WEIGHTS[0] * err_gz + WEIGHTS[1] * err_ax + WEIGHTS[2] * err_ay)


def _log_loss(
    log: Log,
    x: np.ndarray,
    imu_cal: IMUCalibration,
    sigma: tuple[float, float, float],
    bias_gz: float, bias_ax: float, bias_ay: float,
) -> float:
    trace = replay(log, imu_cal=imu_cal, overrides=_overrides(x))
    real_gz = log.gyro_z_lsb - bias_gz
    real_ax = log.accel_x_lsb - bias_ax
    real_ay = log.accel_y_lsb - bias_ay
    return _mse_whitened(
        trace.gyro_z_lsb, trace.accel_x_lsb, trace.accel_y_lsb,
        real_gz, real_ax, real_ay, sigma,
    )


def fit_dynamics(
    train_logs: list[Log],
    imu_cal: IMUCalibration,
    sigma: tuple[float, float, float],
    bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
    x0: Optional[np.ndarray] = None,
    holdout_log: Optional[Log] = None,
    max_evals: int = 250,
    sigma0: float = 0.3,
    seed: Optional[int] = 0,
    verbose: bool = True,
) -> FitResult:
    """Run CMA-ES over the parameter vector. Returns best-found + covariance."""
    try:
        import cma
    except ImportError as e:
        raise RuntimeError(
            "CMA-ES not available. `pip install cma` in the venv."
        ) from e

    if x0 is None:
        x0 = DEFAULT_X0.copy()
    x0 = _clip(np.asarray(x0, dtype=float))
    bias_gz, bias_ax, bias_ay = bias

    # Scale each parameter into ~O(1) for CMA-ES by working in log space for
    # positive-only params. Simpler: use a diagonal sigma in native space,
    # with CMA-ES bounded.
    opts = {
        "bounds": [BOUNDS[:, 0].tolist(), BOUNDS[:, 1].tolist()],
        "maxfevals": max_evals,
        "CMA_stds": (BOUNDS[:, 1] - BOUNDS[:, 0]) * 0.15,  # 15 % of range
        "verbose": -9 if not verbose else 0,
        "seed": 0 if seed is None else int(seed),
    }
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)
    best_x = x0.copy()
    best_loss = float("inf")
    evals = 0
    while not es.stop() and evals < max_evals:
        xs = es.ask()  # always evaluate the full population — cma.tell requires it
        losses: list[float] = []
        for xi in xs:
            xi_arr = np.asarray(xi, dtype=float)
            parts = [
                _log_loss(L, xi_arr, imu_cal, sigma, bias_gz, bias_ax, bias_ay)
                for L in train_logs
            ]
            total = float(np.mean(parts))
            losses.append(total)
            evals += 1
            if total < best_loss:
                best_loss = total
                best_x = xi_arr.copy()
        es.tell(xs, losses)
        if verbose:
            print(f"  eval={evals:4d}  best_loss={best_loss:.4f}  x={np.round(best_x, 4)}")

    per_log = [
        _log_loss(L, best_x, imu_cal, sigma, bias_gz, bias_ax, bias_ay)
        for L in train_logs
    ]
    holdout = None
    if holdout_log is not None:
        holdout = _log_loss(holdout_log, best_x, imu_cal, sigma, bias_gz, bias_ax, bias_ay)

    # CMA-ES internal covariance at convergence (for DR ranges).
    try:
        cov = np.asarray(es.C, dtype=float)
    except Exception:
        cov = None

    return FitResult(
        x=best_x, loss=best_loss, n_evals=evals,
        per_log_loss=per_log, holdout_loss=holdout, cov=cov,
    )
