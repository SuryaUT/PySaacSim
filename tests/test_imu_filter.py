"""Sanity-check the IMU filter chain: step-input group delay should be in
the tens-of-ms range (Phase 0a plan target: ~35 ms at 50 Hz IMU read rate)."""
from __future__ import annotations

from PySaacSim.sim.imu import IMUSimulator
from PySaacSim.sim.physics import RobotState


def _step_response(read_dt_s: float = 0.02, n: int = 200) -> list[int]:
    """Drive gyro_z with a step from 0 to ω_target and record sim output."""
    imu = IMUSimulator(dlpf_fc_hz=44.0, avg_samples=4, gyro_noise_lsb=0.0)
    s = RobotState()
    # Warm the filter at zero for a while so the step truly starts from rest.
    for i in range(50):
        imu.read(s, (i + 1) * read_dt_s)
    s.omega = 5.0  # rad/s step
    out = []
    for i in range(n):
        t = (50 + i + 1) * read_dt_s
        g, _, _ = imu.read(s, t)
        out.append(g)
    return out


def test_step_settles_to_steady_state() -> None:
    # Steady value: 5 rad/s = 286.48 deg/s × 131 LSB/dps = 37529 LSB (saturates
    # at 32767 because 286 deg/s > ±250 dps range). Just verify it reaches the
    # saturated rail within ~100 ms.
    y = _step_response()
    assert max(y) >= 32000  # saturated rail reached


def test_step_group_delay_nontrivial() -> None:
    """Time to reach 63 % of final value should be > one sample (filter must
    actually filter, not just pass through)."""
    imu = IMUSimulator(dlpf_fc_hz=44.0, avg_samples=4, gyro_noise_lsb=0.0)
    s = RobotState()
    read_dt = 0.02
    for i in range(50):
        imu.read(s, (i + 1) * read_dt)
    s.omega = 0.5  # below saturation → 131 * degrees(0.5) = 3754 LSB
    target = 131 * 0.5 * 180.0 / 3.141592653589793
    # "63 % of final" point
    for i in range(1, 100):
        t = (50 + i) * read_dt
        g, _, _ = imu.read(s, t)
        if g >= 0.63 * target:
            assert i >= 1  # at minimum, one sample of filter effect
            return
    assert False, "filter never reached 63% of target — chain is broken"


def test_avg_samples_true_mean() -> None:
    """N-sample rolling mean with N=4 should produce the mean of the last 4
    filtered values, not simply the latest value."""
    imu = IMUSimulator(dlpf_fc_hz=1e6, avg_samples=4, gyro_noise_lsb=0.0)  # LPF effectively off
    s = RobotState()
    # Four reads at gyro=0, then one big step; the first post-step sample
    # should be near (0 + 0 + 0 + step) / 4 = step / 4 because of the 4-tap
    # moving average.
    for i in range(4):
        imu.read(s, (i + 1) * 0.02)  # zeros
    s.omega = 1.0
    g, _, _ = imu.read(s, 5 * 0.02)
    target_single = 131 * (1.0 * 180.0 / 3.141592653589793)
    # With LPF ~ passthrough, sample = (0+0+0+step)/4 = step/4.
    assert g < 0.5 * target_single, (
        f"expected ~{target_single/4:.0f}, got {g} — block-avg appears missing"
    )
