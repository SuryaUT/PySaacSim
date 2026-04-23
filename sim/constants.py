"""Physical constants — chassis geometry, dynamics, motor/servo limits.
Sensor calibration lives in calibration.py (YAML-driven). Constants here
describe the unchanging physics of the Lab 7 robot and are only edited if the
hardware itself is modified."""
from __future__ import annotations

import math


# -- Geometry (cm) ----------------------------------------------------------
# Measured on the physical robot.
CHASSIS_WIDTH_CM = 19.0
CHASSIS_LENGTH_CM = 29.5
CHASSIS_HEIGHT_CM = 13.97

FRONT_WHEEL_DIAM_CM = 7.62
REAR_WHEEL_DIAM_CM = 6.985

# Axle-from-edge values picked to match the measured wheelbase below.
# Rendering only; physics uses WHEELBASE_CM / REAR_TRACK_CM directly.
FRONT_AXLE_FROM_FRONT_CM = 9.85
REAR_AXLE_FROM_REAR_CM = 4.85
WHEELBASE_CM = 14.8            # measured
REAR_TRACK_CM = 18.0           # measured


# -- Steering (MG996R) ------------------------------------------------------
# Firmware servo formula (RTOS_MotorBoard/bump.c): count = 3120 + angle*1200/53.
# Endpoints derived from that formula (angle in deg, positive = right turn):
#   1920 = full right (+53°, firmware sign) -> -53° in sim's CCW-positive frame
#   3120 = center
#   4320 = full left  (-53°, firmware sign) -> +53° in sim's CCW-positive frame
SERVO_TIMER_HZ = 2_000_000
SERVO_CENTER_COUNT = 3120
SERVO_COUNTS_PER_53DEG = 1200
SERVO_MIN_COUNT = SERVO_CENTER_COUNT - SERVO_COUNTS_PER_53DEG  # 1920
SERVO_MAX_COUNT = SERVO_CENTER_COUNT + SERVO_COUNTS_PER_53DEG  # 4320
STEER_LIMIT_RAD = math.radians(53)
SERVO_RAD_PER_SEC = math.radians(60) / 0.17


def servo_count_to_steer_angle(count: int) -> float:
    """Servo PWM count -> wheel angle (rad). Firmware uses
    count = 3120 + angle_deg*1200/53, positive = right turn; sim's yaw frame is
    CCW-positive, so right turn -> negative wheel angle."""
    c = max(SERVO_MIN_COUNT, min(SERVO_MAX_COUNT, count))
    t = (c - SERVO_CENTER_COUNT) / SERVO_COUNTS_PER_53DEG
    return -t * STEER_LIMIT_RAD


# -- Motors -----------------------------------------------------------------
MOTOR_PWM_MAX_COUNT = 10000
MOTOR_PWM_HZ = 200
MOTOR_MAX_FORCE_N = 5.0
MOTOR_LAG_TAU_S = 0.100

# Top speed: measured 1.219 m/s at full charge on the real robot
# (2026-04-22, smooth lab tile). Supersedes the earlier 0.47 m/s figure.
# NOTE: This value is no longer enforced as a hard clamp by physics.py —
# terminal velocity emerges from MOTOR_MAX_FORCE_N vs LINEAR_DRAG balance.
# Kept here for reference and as a sanity-check ceiling in tests / fit bounds.
MAX_SPEED_CMS = 122

# Actuator transport delay between firmware command (sent over CAN) and the
# motor/servo physically responding. Decomposition: CAN serialization (~5-10
# ms), motor-board ISR + PWM register write (~1-5 ms), 50 Hz servo PWM cycle
# wait (0-20 ms), servo internal deadband (5-15 ms). Both fitted by CMA-ES.
STEER_TRANSPORT_LAG_S = 0.030
THROTTLE_TRANSPORT_LAG_S = 0.030


def duty_count_to_pwm01(count: int) -> float:
    if count <= 0:
        return 0.0
    if count >= MOTOR_PWM_MAX_COUNT:
        return 1.0
    return count / MOTOR_PWM_MAX_COUNT


# -- Dynamics ---------------------------------------------------------------
MASS_KG = 1.453                # measured 1453 g
_w_m = CHASSIS_WIDTH_CM / 100.0
_l_m = CHASSIS_LENGTH_CM / 100.0
INERTIA_KG_M2 = MASS_KG * (_w_m * _w_m + _l_m * _l_m) / 12.0

MU_STATIC = 0.8
MU_KINETIC = 0.6
ROLLING_RESIST = 0.02
LINEAR_DRAG = 0.1
G = 9.81


# -- Timing -----------------------------------------------------------------
PHYSICS_DT_S = 0.001
RENDER_RATE_HZ = 60
IR_SAMPLE_RATE_HZ = 1000
