# PySaacSim sim-to-real calibration pass

## Context

**Why this work exists.** PySaacSim is the training environment for an RL
policy that will be flashed to the real Lab 7 car. The sim's value is bounded
by how faithfully it predicts the real car's response — every gap between sim
and hardware is a gap the trained policy will hit on first hardware run. **No
sim-trained policy has been deployed yet**, so this is the first chance to
close the gap before the first hardware run, not a debug pass on a known
miss.

**Available data artifacts** (all in `C:/RTOS Labs/Lab7/lab-7-tweinstein-1/`):
- `IR_Calib.xlsx` — IR per-side ADC ↔ distance table **and** TFLuna
  calibration table (one workbook, multiple sheets). Columns: inches, mm,
  IR_Left_ADC, IR_Right_ADC.
- `IMU_steady_state.csv` — 1717 rows, 80 ms cadence. Despite the name the
  car is **not motors-off**: throttle stays at 9999 throughout, accel_y sits
  near −1100 LSB suggesting either ~4° nose-up pitch or a real bias. Useful
  for **noise characterization with motor vibration present**, plus a coarse
  bias estimate. **A true motors-off, flat-on-ground capture is still
  needed** to disambiguate bias from pitch.
- `robot_clockwise.csv`, `robot_counter_clockwise.csv`, `more_CW.csv`,
  `more_CCW.csv` — 1660–1665 rows each, ~130 s of wall-following per file.
  Two directions × two takes. CSV columns:
  `time_ms, ir_r, ir_l, tf_r, tf_l, tf_front, throttle_l, throttle_r,
   steering, gyro_z, accel_x, accel_y`.

**Confirmed log conventions** (from firmware code, agent investigation 2026-04-22):
- `time_ms`: ms timestamp, ~80 ms cadence (no jitter).
- `ir_r`, `ir_l`: **firmware-converted distance in mm**, not raw ADC. Each
  side uses `d_mm = a/(adc + b) + c` with **per-side constants**. Saturates
  at 305 mm (→ "out of range"). Source:
  `lab-7-tweinstein-1/RTOS_SensorBoard/IRDistance.c:78-97`.
  - Right: `a = 52850`, `b = -1239`, `c = 69`. Sat when `adc < 1476`.
  - Left:  `a = 137932`, `b = -859`,  `c = 32`. Sat when `adc < 1376`.
- `tf_*`: raw TFLuna mm, averaged over 8 consecutive samples in firmware
  (USE_MEDIAN_FILTER=0 path). Caps near 1000 mm in normal config.
- `throttle_l`, `throttle_r`: PWM duty 0–9999 **after** PD baseline,
  differential-steering reduction (|steer|≥15° → inside motor −2000), **and**
  model residual application — i.e. the values actually sent over CAN to the
  motor board. `9999` is literal full duty (10000 disables the waveform).
  Throttle is never negative in these logs (forward-only operation).
- `steering`: degrees, range ±35° per `Model.h` `CAP_STEERING`.
  **Sign: positive = right turn** (per `IMU_behavior.md:34` and user
  confirmation 2026-04-22).
- `gyro_z`: raw int16 LSB at 131 LSB/(deg/s). **Positive = left turn (CCW)**
  — opposite of steering sign convention. The firmware negates this before
  feeding the model.
- `accel_x`: raw int16 LSB at 16384 LSB/g. **Positive = chassis accel toward
  the right.**
- `accel_y`: raw int16 LSB at 16384 LSB/g. **Positive = chassis accel
  forward.**

**Confirmed operating conditions for the driving logs.**
- Surface: smooth lab floor (tile/vinyl). Fitted friction values are expected
  to transfer to other lab tests on the same floor.
- Battery: fully charged. **Top speed = 1.219 m/s (121.9 cm/s).** This
  supersedes `MAX_SPEED_CMS = 47` in `sim/constants.py:55` and the stale
  0.47 m/s note in `memory/project_robot_measurements.md`. Memory update
  required after plan mode exits.
- IMU mounting unchanged from `IMU_behavior.md`; sign conventions there are
  authoritative.

## Confirmed firmware facts (used to align sim faithfully)

These are the runtime characteristics of the real car that the sim must
match. Cited from agent investigation 2026-04-22. Plan changes follow.

**Sensor board (`RTOS_SensorBoard/`)**
- `Robot()` task runs at **~12.5 Hz** (waits on TFLuna semaphores; one tick
  ≈ one CSV row). Foreground thread, priority 1, 128-word stack.
- DAS task runs IR ADC at **1 kHz**, applies a 60 Hz IIR notch filter
  (`RTOS_SensorBoard.c:393`), then converts ADC → mm via `IRDistance.c`.
- TFLuna readings arrive at **~100 Hz per lidar via UART ISR**, are
  averaged 8 samples deep before logging.
- IMU (`IMU.c`) sampled at **50 Hz**, 4-sample firmware block average per
  read, MPU-6050 DLPF set to **0x03 = 44 Hz BW** (`IMU.h:18`,
  `IMU.c:34`). **NOT 10 Hz / 16 samples** as the v1 plan assumed.
- PD wall-follower (in `Robot()`) gains: `kp_d = 1`, `kd_d = 2` (distance);
  `kp_a = 5`, `kd_a = 2` (angle). Reference angle 5°, reference distance
  0 mm. Front collision avoidance kicks in below 800 mm.
- Differential steering: |steer| ≥ 15° → inside motor throttle reduced by
  2000. Applied **before** model residual.
- Logging: SD card via `eFile_WriteSDec` (`RTOS_SensorBoard.c:361-384`),
  not UART. CSV columns are exactly the order above.

**Motor board (`RTOS_MotorBoard/`)**
- Motor PWM: **200 Hz**, period 10 000 counts. 0 = stop, 9999 ≈ full
  forward. Direction set by which PWM pin is driven (forward vs. backward).
- Servo PWM: 200 Hz (5 ms period? no — period 40 000 counts at 2 MHz =
  **20 ms / 50 Hz**). Center = **3120 counts (1.5 ms)**, formula
  `count = center + (angle * 1200) / 53` for ±53° → counts 1920–4320.
  **Sim has 3200 in `constants.py:34` — wrong by 80 counts.**
- CAN protocol `CMD_MOTOR`: 3 fields (left duty 0–10000, right duty
  0–10000, steering deg int16 ±53°). Pure pass-through to PWM.
- Safety: WiFi watchdog + 180 s no-CAN timeout brake the motors. Bump
  switches trigger `PWMA0/1_Break()`. None of this matters for sim, but
  worth noting if the real car ever stops mid-test for "no reason".

**Common (`RTOS_Labs_common/`)**
- RTOS tick = 2 ms; SysTick preempts foreground; PendSV at priority 3.
- `OS_MsTime()` is the source of `time_ms` in CSVs.
- Model fixed-point: `typedef int32_t fixed_t`. CAP constants in `Model.h`:
  `CAP_IR = 305`, `CAP_TFLUNA = 1000`, `CAP_THROTTLE = 9999`,
  **`CAP_STEERING = 35`**. Sim has `CAP_STEERING = 30` in
  `PySaacSim/sim/model.py:30` — **off by 5°.**

## Sim ↔ firmware discrepancies discovered

These have to be fixed before any meaningful comparison. They were the source
of all the v1 plan's wrong assumptions.

| # | Sim location | Sim has | Firmware truth | Fix |
|---|---|---|---|---|
| D1 | `sim/constants.py:55` `MAX_SPEED_CMS` | 47 | 121.9 cm/s @ full charge | → 122 |
| D2 | `sim/constants.py:34` `SERVO_CENTER_COUNT` | 3200 | 3120 | → 3120 |
| D3 | `sim/constants.py:32-35` servo min/max | 2000 / 4400 | 1920 / 4320 | re-derive from `count = 3120 + (angle*1200)/53` |
| D4 | `sim/model.py:30` `CAP_STEERING` | 30 | 35 | → 35 |
| D5 | `sim/imu.py` instantaneous output | none | DLPF 44 Hz BW + 4-sample avg | add filter chain |
| D6 | `sim/sensors.py` IR pipeline | true distance + Gaussian noise | true distance → voltage curve → ADC quantization → firmware `d=a/(adc+b)+c` round-trip with per-side constants | rewrite IR sensor to round-trip through firmware formula |
| D7 | `sim/sensors.py` IR per-side | shared `IRCalibration` | per-side `(a, b, c)` constants | promote `IRCalibration` to `IRSideCalibration` × 2 |
| D8 | `sim/sensors.py` TFLuna 8-sample firmware avg | none | 8-sample avg in firmware before log | add 8-sample avg to lidar output for replay-time comparison only (training-time sim can be instantaneous + same DR-matched noise) |
| D9 | `sim/calibration.py` no IMU bias block | n/a | static IMU bias not removed in firmware | add `imu.gyro_bias`, `imu.accel_bias` |
| D10 | Robot loop rate | 50 Hz assumed in v1 plan | **12.5 Hz** (one CSV row = one Robot tick) | replay at 12.5 Hz, not 50 Hz |

## What gets fit, by data source

| Artifact | Constrains | Cannot constrain |
|---|---|---|
| `IR_Calib.xlsx` IR sheet | per-side IR `(a, b, c)` constants in firmware form; IR ADC noise std | dynamics, lidar |
| `IR_Calib.xlsx` TFLuna sheet | per-lidar `distance_scale`, `distance_bias_cm`, `noise_std_cm` | dynamics, IR |
| `IMU_steady_state.csv` (motors-on, stationary) | gyro/accel **noise std with motor vibration**; **coarse** bias estimate | clean static bias (needs motors-off recording) |
| Future motors-off capture | clean IMU bias offsets | noise (handled by IMU_steady_state.csv) |
| 4× wall-follower CSVs | dynamics: `MOTOR_LAG_TAU_S`, `MOTOR_MAX_FORCE_N`, `LINEAR_DRAG`, `ROLLING_RESIST`, `MU_KINETIC`, `SERVO_RAD_PER_SEC`; latency | sensor bias/scale (no ground-truth pose in log) |

`MAX_SPEED_CMS` is held fixed at 122 (measured). `MASS_KG` (1.453, also
measured) and chassis geometry are fixed. The fit only nudges the dynamics
parameters above.

## Implementation phases

### Phase 0 — Sim correctness fixes (gating, before any fitting)

Without these, comparison loss has nothing to do with physics.

- **0a. IMU filter chain.** `sim/imu.py` currently returns instantaneous
  values. Add a per-channel 1st-order Butterworth LPF at fc = 44 Hz (matches
  `IMU_DLPF_CFG = 0x03`) followed by a rolling 4-sample mean (matches
  `IMU_AVG_SAMPLES = 4`). Both knobs configurable via the new `imu:` block in
  `calibration.yaml` so future firmware tweaks don't require code edits.
  Group delay at 44 Hz LPF ≈ 4–5 ms; 4-sample mean at 50 Hz ≈ 30 ms group
  delay. Combined ≈ 35 ms — non-negligible at 12.5 Hz Robot rate.
- **0b. IMU bias subtraction.** New `imu.gyro_bias`, `imu.accel_bias`
  3-vectors in `calibration.yaml` (raw int16 LSB, default zero). Calibration
  pipeline subtracts them from the **real** log before comparing to sim;
  sim emits bias-free signals. Bootstrap with `IMU_steady_state.csv` means
  but **flag the result as "biased by ~4° pitch"** until a true motors-off
  flat capture is recorded.
- **0c. Servo + steering caps.** Patch `sim/constants.py`
  (D2, D3) and `sim/model.py` (D4). Update
  `servo_count_to_steer_angle` to `(c - 3120) / (4320 - 3120) * STEER_LIMIT_RAD`
  with sign per existing convention.
- **0d. IR sensor pipeline rewrite.** Refactor `sim/sensors.py:ir_reading` to:
    1. true distance d_true (from raycast, cm).
    2. Convert to voltage via current `ir_distance_to_volts(d_true)` (kept,
       this is the real analog response).
    3. Add voltage noise.
    4. Convert volts → ADC count: `adc = round(v / Vref * 4095) + adc_noise`.
    5. Apply firmware formula: `d_mm = a/(adc+b) + c` with per-side `(a, b, c)`.
    6. Saturate at 305 mm if `adc < adc_threshold`.
    7. Output mm (matching what the CSV logs, what the model receives).
  Promote `IRCalibration` → `IRSideCalibration` (per-side `a, b, c, adc_threshold`).
- **0e. TFLuna 8-sample averaging in replay path.** Sim's `lidar_reading`
  stays instantaneous for training (faster, no fidelity cost since training
  doesn't compare to a logged trace). Replay-mode sim wraps the lidar with
  an 8-sample moving average before output, matching what the CSV logs see.

After 0a–0e, sim output for any (state, command) should be byte-comparable
to what the real car would log, modulo physics fitting in later phases.

### Phase 1 — Calibration ingestion package (`PySaacSim/calib/`)

```
calib/
├── __init__.py
├── log_io.py             # CSV → typed Log dataclass
├── ir_xlsx.py            # IR_Calib.xlsx → per-side fit of d=a/(adc+b)+c
├── tfluna_xlsx.py        # TFLuna sheet → per-lidar scale/bias/noise
├── windows.py            # quiescent-window detector (low |gyro|, near-stop throttle)
├── imu_bias.py           # mean of stationary window → bias offsets
├── noise_fit.py          # per-channel sensor noise std on quiescent windows
├── filters.py            # MPU-6050 DLPF + block-average emulation (used by sim AND by replay)
├── replay.py             # injects logged throttle/steering into physics at 12.5 Hz; returns sim sensor + IMU trace
├── latency.py            # command↔response cross-correlation
├── dynamics_fit.py       # CMA-ES over Phase 4 parameter vector
└── report.py             # diagnostic PNGs + YAML diff
scripts/
└── calibrate_from_log.py # one-shot CLI: xlsx + steady + driving CSVs → proposed YAML/constants diff
tests/
├── test_replay_identity.py        # current sim, current params: replay one log, basic sanity
├── test_ir_fit_repro.py           # synth ADC samples → fit → recover within 1%
├── test_tfluna_fit_repro.py       # same
├── test_imu_filter.py             # step input → expect 30+ ms group delay
└── test_dynamics_fit_synthetic.py # sim-generated trace, CMA-ES recovers params within 5%
```

### Phase 2 — Sensor refits (cheap, do first)

- **IR per-side fit.** Read `IR_Calib.xlsx`, columns inches/mm/IR_Left_ADC/
  IR_Right_ADC. Convert inches if needed (the mm column is canonical).
  Fit `d_mm = a/(adc + b) + c` per side via `scipy.optimize.curve_fit`.
  Cross-check against firmware constants
  (`IRDistance.c:78-97`); the xlsx is the authoritative new value, the
  firmware constants are the previous fit. If new constants drift > 5 %
  from firmware, also propose a firmware-side update for the user to flash.
- **TFLuna per-lidar fit.** Same workbook, separate sheet. Fit
  `d_reported = scale·d_true + bias` per lidar via least squares; per-lidar
  noise std as residual.
- **IR ADC noise + IR distance noise** from `IMU_steady_state.csv`: the IR
  columns are stable at 305/162 in that recording (motors on but car
  stationary). Per-side std of the IR mm column → distance noise std.
  Inverse-map through the fitted formula to recover ADC noise std.
- **TFLuna noise from log**: same approach on `tf_*` columns of stationary
  segments.
- **IMU noise std from `IMU_steady_state.csv`**: per-channel std of
  gyro_z/accel_x/accel_y after subtracting per-channel mean. This includes
  motor vibration noise — that's the realistic noise level for in-motion
  comparisons.
- **IMU bias from `IMU_steady_state.csv`**: per-channel mean. Flag with a
  warning that accel_y carries a likely ~−1100 LSB pitch contribution and
  should be re-measured against a motors-off flat capture before being
  trusted as final.

### Phase 3 — Latency identification

Cross-correlate (steering → −gyro_z) and (throttle_avg → accel_y) at lags
0–500 ms across all 4 driving logs combined. Peak lag = effective
actuator+filter latency. Compare against `MOTOR_LAG_TAU_S = 0.100` and
`SERVO_RAD_PER_SEC = math.radians(60)/0.17 ≈ 6.16` after Phase 0a (IMU
filter) is in place — without 0a the latency estimate gets contaminated by
filter delay and we'd over-attribute to actuator lag.

### Phase 4 — Dynamics replay-fit (the main course)

- **Replay rate**: 12.5 Hz (one CSV row = one Robot tick). For each row,
  call `physics.apply_command(state, throttle_l, throttle_r, servo_count)`
  where `servo_count = 3120 + (steer_deg * 1200) / 53`, then advance the
  1 ms physics sim by 80 ms.
- **Sign handling**: real CSV `gyro_z` and sim's IMUSimulator output share
  CCW-positive convention. Sim's existing `imu.py:11` says so explicitly.
  No flip needed.
- **Parameter vector** (CMA-ES, log-spaced sigma where positive-only):
  - `MOTOR_MAX_FORCE_N` (now 5.0; bound 1.0–20.0)
  - `MOTOR_LAG_TAU_S` (now 0.100; bound 0.02–0.5)
  - `LINEAR_DRAG` (now 0.1; bound 0.01–2.0)
  - `ROLLING_RESIST` (now 0.02; bound 0.0–0.2)
  - `SERVO_RAD_PER_SEC` (now 6.16; bound 1.0–20.0)
  - `MU_KINETIC` (now 0.6; bound 0.1–1.5)
- **Held fixed**: `MAX_SPEED_CMS = 122`, `MASS_KG = 1.453`,
  `INERTIA_KG_M2`, geometry, IMU LSB/g and LSB/dps.
- **Loss**: per-channel whitened MSE,
  `L = w_gz·MSE(sim_gz, real_gz)/σ_gz² + w_ay·MSE(sim_ay, real_ay)/σ_ay²
       + w_ax·MSE(sim_ax, real_ax)/σ_ax²`,
  with σ from Phase 2 noise fits and weights `(1, 1, 0.5)`.
- **Pre-optimizer ablation**:
  1. Replay one CW + one CCW log with current sim params (after Phase 0
     fixes, before Phase 4 fit). Visualize sim vs real IMU for each
     channel. Confirm signs and rough magnitudes. **If gyro signs disagree
     between CW vs CCW logs, that's the sign-convention bug — fix before
     fitting.**
  2. Hold-out split: fit on 3 of the 4 driving logs, validate on the 4th.
     Holdout MSE > 2× train MSE → battery drift / temperature / surface
     change between captures; report and stop.
- **Optimizer budget**: ~500 evals. Each eval is one full-log replay
  (~1 s at 1 ms physics tick × 130 s of log = 130 k iterations × 4 logs).
  Total wall-clock: tens of minutes on a single core. Parallelize across
  logs via `multiprocessing.Pool` if it's painful.

### Phase 5 — Apply, diff, commit

- `report.py` writes a side-by-side YAML diff between current
  `config/calibration.yaml` and proposed values, plus per-channel overlay
  PNGs for each driving log (before/after fit).
- `scripts/calibrate_from_log.py --apply` writes updates only after explicit
  user confirmation (per `memory/project_sim_to_real_goal.md`).
- `sim/constants.py` edits (D1–D4) go through the same diff-and-confirm gate.

### Phase 6 — Hand off to RL training (out of scope here, the destination)

Use the CMA-ES posterior covariance at convergence to set the **DR ranges**
for `training/finetune.py` (per-parameter ±2σ), instead of the hand-picked
ranges in `docs/implementation_plan.md` §5.3. This ties domain randomization
to actual measured uncertainty.

## Critical files

Sim (write):
- `PySaacSim/sim/constants.py` — D1, D2, D3 fixes
- `PySaacSim/sim/model.py` — D4 fix
- `PySaacSim/sim/imu.py` — D5 (filter chain)
- `PySaacSim/sim/sensors.py` — D6, D7, D8
- `PySaacSim/sim/calibration.py` — D7, D9 (per-side IR; IMU bias block)
- `PySaacSim/config/calibration.yaml` — re-fit values, IMU block

Sim (new):
- `PySaacSim/calib/` package per Phase 1 layout
- `PySaacSim/scripts/calibrate_from_log.py`
- `PySaacSim/tests/test_*` per Phase 1

Read-only inputs (firmware / data):
- `lab-7-tweinstein-1/IR_Calib.xlsx` — IR + TFLuna calibration tables
- `lab-7-tweinstein-1/IMU_steady_state.csv` — IMU noise + coarse bias
- `lab-7-tweinstein-1/{robot,more}_{CW,CCW,clockwise,counter_clockwise}.csv`
  — 4 wall-follower logs
- `lab-7-tweinstein-1/RTOS_SensorBoard/IRDistance.c` — IR formula constants
- `lab-7-tweinstein-1/RTOS_SensorBoard/IMU.c`, `IMU.h` — IMU filter config
- `lab-7-tweinstein-1/RTOS_SensorBoard/Model.c`, `Model.h` — input/output cap and ordering
- `lab-7-tweinstein-1/RTOS_MotorBoard/PWMA0.c`, `PWMA1.c`, `PWMG6.c` — PWM details
- `lab-7-tweinstein-1/RTOS_MotorBoard/bump.c` — servo formula
- `lab-7-tweinstein-1/RTOS_SensorBoard/RTOS_SensorBoard.c` — `Robot()` task,
  PD baseline, differential steering, CSV column order

## Verification

End-to-end checks before declaring done:

1. **Phase 0a regression** (`tests/test_imu_filter.py`): step input through
   the filter chain → measured group delay within 5 % of theoretical
   (~5 ms LPF + ~30 ms 4-sample mean = ~35 ms total at 50 Hz).
2. **Phase 0d round-trip** (`tests/test_ir_pipeline.py`): for a sequence of
   true distances 5–80 cm, run sim's IR pipeline and assert output mm
   matches `IR_Calib.xlsx` measured mm to within 1 σ of the fit residual.
3. **Phase 2 fit reproducibility** (`test_ir_fit_repro.py`,
   `test_tfluna_fit_repro.py`): generate samples from a known fit + Gaussian
   noise, recover within 1 % of true params.
4. **Phase 4 synthetic recovery** (`test_dynamics_fit_synthetic.py`):
   sim-generated 130 s trace from known params, perturb starting guess by
   ±50 %, CMA-ES recovers within 5 %.
5. **End-to-end on real logs**: after Phase 5 apply, replay each driving log
   against the updated sim. Pass criterion: gyro_z RMSE < 1 σ_gz, accel_y
   RMSE < 1 σ_ay on the holdout log. Report per-channel overlay PNGs.
6. **No PPO regression**: `examples/train_ppo.py` for 100 k steps with the
   new calibration; reward curve trends upward (catches sign errors that
   slip past the synthetic test).

## Resolved questions

- Floor: lab tile.
- Battery: full charge; top speed 1.219 m/s → `MAX_SPEED_CMS = 122`.
- IMU mounting: unchanged from doc.
- IR_Calib.xlsx is current; covers both IR and TFLuna; per-side IR.
- IR conversion form is firmware-side `d = a/(adc+b) + c`.
- Logged steering: positive = right turn.
- Logged throttle: never negative; `9999` is literal full duty.
- CSV cadence: clean 80 ms (no jitter).
- 5 CSVs available; 4 wall-following + 1 motors-on-stationary.
- No prior sim-trained policy on hardware.

## Outstanding requests

- A short **motors-off, flat-on-ground** IMU recording (≥3 s) so accel_y
  bias can be cleanly separated from a possible pitch contribution in
  `IMU_steady_state.csv`. Optional but improves Phase 0b confidence.
- Confirmation whether D4 (`CAP_STEERING = 30 → 35`) is intended to match
  firmware exactly, or whether the sim's tighter cap was deliberate (e.g.
  to prevent the trained policy from requesting angles the firmware would
  also reject). Default: match firmware.

## RESUME-HERE checkpoint (2026-04-22 late, hand-off to next session)

**Phase 7 implementation is mid-flight.** Below is the exact state so the
next Claude can continue without re-investigating.

### Done (committed to disk, all tests passing — 15 passed, 2 skipped)

- **7a** `PySaacSim/sim/physics.py` — `RobotState` now has `sim_t_s`,
  `steer_cmd_queue`, `motor_cmd_queue` fields. `apply_command` writes to
  the queues. `_integrate_dynamics` advances `sim_t_s += dt`, drains
  queues whose release-time has passed, then runs the existing servo
  slew + motor lag + longitudinal/yaw integration. **The hard speed
  clamp at the old line 94 is removed** — terminal v emerges from
  drag/force balance.
- **7b** `PySaacSim/sim/constants.py` — added
  `STEER_TRANSPORT_LAG_S = 0.030`, `THROTTLE_TRANSPORT_LAG_S = 0.030`,
  and updated the `MAX_SPEED_CMS` comment to note it's no longer
  enforced as a hard clamp.
- **7c** `PySaacSim/calib/dynamics_fit.py` — `PARAM_NAMES` is now 8
  entries (added `STEER_TRANSPORT_LAG_S`, `THROTTLE_TRANSPORT_LAG_S`).
  `DEFAULT_X0` and `BOUNDS` updated: `LINEAR_DRAG` upper relaxed to
  10.0; `SERVO_RAD_PER_SEC` lower tightened to 4.0; new lag bounds
  (0.0, 0.150).
- **7d** `PySaacSim/calib/replay.py` — **no edits needed**. The
  override mechanism (line 84-88) generically applies any attribute
  present on `_phys_mod`. Both new lag constants are imported into
  `sim/physics.py` so `setattr(_phys_mod, "STEER_TRANSPORT_LAG_S", v)`
  works.
- **7e** `PySaacSim/calib/tfluna_xlsx.py` — fully rewritten for both
  the new `Sensor_Calib.xlsx` `LiDARs` sheet (cols: `inches, mm, Left
  LiDAR, Center LiDAR, Right LiDAR`) and the legacy column convention
  (`true_mm, tf_left, tf_middle, tf_right`). Sanity-tested against the
  real xlsx — Left scale=1.019 / bias=-34mm, Center scale=1.023 /
  bias=-20mm, Right scale=1.024 / bias=-26mm. All three lidars read
  ~2-3 cm short. Per-unit divergence is small (0.5% scale spread,
  14 mm bias range) — **kept single shared `LidarCalibration` for now;
  per-lidar promotion deferred** as a future enhancement.
- **7e (wiring)** `PySaacSim/scripts/calibrate_from_log.py` +
  `PySaacSim/calib/report.py` — added `fit_tfluna_xlsx` import,
  pipeline now calls it during Phase 1 and passes the result to both
  the `interim` (pre-CMA-ES) and final `proposed` calibrations via the
  new `tfluna_fit=` kwarg on `apply_to_calibration`. Averages per-lidar
  scale/bias into the single `LidarCalibration`. Larger of (xlsx
  residual std, in-log noise std) is taken as `noise_std_cm`.
- **7f** `PySaacSim/calib/imu_bias.py` — dropped the pitch-warning
  block at the old lines 53-58. Module docstring rewritten to reflect
  that `IMU_steady_state.csv` is genuinely motors-off (motor board was
  unpowered during capture; throttle column is a stale register value
  to ignore).
- **7g** `PySaacSim/scripts/calibrate_from_log.py` — `fit_noise` is
  no longer called on the steady CSV. Instead, after `drive_logs` is
  loaded for latency, the script tries `fit_noise(L, current.ir)` on
  each driving log and uses the first quiescent window it finds. If
  none, `noise = None` and the existing YAML defaults stay (downstream
  `apply_to_calibration` and `summary_lines` already null-guard).
- **7i** `PySaacSim/tests/test_transport_delay.py` — three new tests:
  steer lag holds setpoint, motor lag holds setpoint, and "no speed
  governor" sanity (5 sim-seconds at full throttle, state.v stays
  finite and positive — no clamp at MAX_SPEED_CMS, no runaway).
  Full pytest pass: **15 passed, 2 skipped** (the 2 skips are
  `test_log_io` cases that skip when their fixture CSVs aren't on the
  expected path — pre-existing).

### NOT YET DONE — pick up here

1. **7h — full 500-eval CMA-ES run** with the new physics + new
   xlsx. Command (Windows / git-bash, run from
   `C:/RTOS Labs/Lab7/PySaacSim`):
   ```bash
   PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python scripts/calibrate_from_log.py \
     --ir-xlsx ../lab-7-tweinstein-1/RTOS_SensorBoard/Sensor_Calib.xlsx \
     --steady-csv ../lab-7-tweinstein-1/IMU_steady_state.csv \
     --drive-csvs ../lab-7-tweinstein-1/robot_clockwise.csv \
                  ../lab-7-tweinstein-1/robot_counter_clockwise.csv \
                  ../lab-7-tweinstein-1/more_CW.csv \
                  ../lab-7-tweinstein-1/more_CCW.csv \
     --max-evals 500 \
     > artifacts/logs/calib_phase7_$(date +%Y%m%d_%H%M%S).log 2>&1 &
   ```
   Wait ~30 minutes. Use `Monitor` with the log file + a grep for
   `eval=|Traceback|Error|loss \(train\)|holdout` so progress events
   stream in. Inspect the latest `artifacts/calib_<ts>/` afterwards.

5. **7h — what to look for after the run finishes:**
   - Loss < 58.7K (the v1 result). Goal: meaningfully under that.
   - `SERVO_RAD_PER_SEC` > 4.0 (above the new tightened floor — this
     is the headline check).
   - `MOTOR_LAG_TAU_S` rises above 0.02 (free of old floor).
   - Both transport-lag params nonzero, < 100 ms.
   - **`accel_y` overlay PNGs now show oscillation in sim** — that's
     the headline visual proof the governor removal worked.
   - `holdout / train` ratio still < 2.0.

6. **7j — apply.**
   - `--apply` to write the YAML (will prompt y/N).
   - **Manually port** the 8 fitted dynamics values into
     `PySaacSim/sim/constants.py` (`MOTOR_MAX_FORCE_N`,
     `MOTOR_LAG_TAU_S`, `LINEAR_DRAG`, `ROLLING_RESIST`,
     `SERVO_RAD_PER_SEC`, `MU_KINETIC`, `STEER_TRANSPORT_LAG_S`,
     `THROTTLE_TRANSPORT_LAG_S`) — the CLI does **not** edit
     `constants.py`. Get user sign-off on the diff first.

### Recovery script if the run crashes

`PySaacSim/scripts/regen_artifacts.py` already exists from this
session — pass `--x` (8 values now) + `--proposed-yaml` from a
crashed run's artifact dir + `--out-dir` to regenerate
`overlays/`, `dynamics.json`, `summary.txt` without redoing CMA-ES.
**Note:** the script's `--x` accepts exactly 6 values currently and
will need a small update to accept 8 (or use `--from-json` once the
CLI drops a partial `dynamics.json`).

### Plan file

Updated and committed at:
`C:\Users\surya\.claude\plans\you-said-we-need-giggly-valley.md`

The plan continues below from the original "Phase 7 — Refinement"
section, which has the full architectural rationale.

---

## Phase 7 — Refinement after 500-eval results (2026-04-22 evening)

### Why we need this phase

The 500-eval CMA-ES converged to loss = 58.7K (down from 218K smoke-test
baseline) with holdout/train ratio 1.39 (well under the 2.0 fail threshold).
Sign consistency verified between CW and CCW logs — gyro_z goes negative on
CW turns, positive on CCW turns, both in sim and real. **However, three of
six dynamics parameters pinned at their lower bounds** and **sim accel_y is
flat at zero throughout while real oscillates ±5000 LSB**. Both symptoms
trace to two missing pieces of the sim model rather than fit-budget issues.

Final v1 fit:
| Param | Default | Fitted | Note |
|---|---|---|---|
| MOTOR_MAX_FORCE_N | 5.0 | 14.22 | within bounds |
| MOTOR_LAG_TAU_S | 0.100 | **0.020** | at lower bound |
| LINEAR_DRAG | 0.1 | 0.78 | within |
| ROLLING_RESIST | 0.02 | **0.0002** | at lower bound (probably real on smooth tile) |
| SERVO_RAD_PER_SEC | 6.16 | **1.02** | at lower bound (datasheet says 6.16) |
| MU_KINETIC | 0.6 | 0.22 | within |

### Root causes identified

1. **No actuator transport delay.** The sim's `apply_command()`
   (`sim/physics.py:52-63`) writes setpoints into state and the next 1 ms
   physics tick consumes them. Real chain has CAN + motor-board ISR + PWM
   cycle wait + servo deadband ≈ 30 ms before the actuator starts to move.
   CMA-ES has only `SERVO_RAD_PER_SEC` and `MOTOR_LAG_TAU_S` to "delay" the
   response, so it pins both at their floors.

2. **Hard top-speed governor.** `physics.py:93-94`:
   `v_new = max(-v_max, min(v_max, v_new))` clamps at 122 cm/s. With
   fitted force/drag, sim natural terminal velocity is ~18 m/s, so it
   slams into the clamp in <1 ms and stays there → `dv/dt = 0` forever.
   Real car has natural terminal velocity from force-vs-drag balance and
   slows in turns.

3. **IMU_steady_state bias interpretation was wrong** (now corrected by
   user 2026-04-22): the **motor board was unpowered** during this
   capture — only the sensor board was running. The 9999 throttle column
   is a stale firmware register value, not actual motor command. The
   IMU readings are genuinely **motors-off** (no motor vibration mixed
   in), so the bias estimate is even cleaner than initially thought,
   and the noise std is pure sensor noise — not the realistic in-motion
   noise envelope we'd want for whitening the dynamics-fit MSE. Drop the
   "pitch contribution" warning at `calib/imu_bias.py:53-58`. **Only
   the columns `time_ms, gyro_z, accel_x, accel_y` are valid in this
   CSV** — ignore IR / lidar / throttle / steering columns from this
   file specifically. (Other CSVs are full-fidelity.)

4. **Wrong xlsx file in use.** Friend's pipeline reads `IR_Calib.xlsx`;
   user has fresher `Sensor_Calib.xlsx` with both `IRs` and `LiDARs`
   sheets. The LiDAR sheet finally unlocks the per-lidar scale/bias fit
   that `calib/tfluna_xlsx.py` was a no-op for.

### Concrete changes (edit / add only what's listed)

**Sim physics — `PySaacSim/sim/physics.py`**
- Add command queues to `RobotState`:
  `motor_cmd_queue: list[tuple[float, float]]`,
  `steer_cmd_queue: list[float]`. Each holds `(t_seconds, value)` tuples
  pushed by `apply_command()`.
- Modify `apply_command()` to push to queues instead of writing
  `state.motor_cmd_*` / `state.steer_cmd` directly.
- Modify `_integrate_dynamics()` to read the most-recent queue entry
  whose timestamp is `<= now - LAG`, where `LAG` comes from new module
  globals `STEER_TRANSPORT_LAG_S`, `THROTTLE_TRANSPORT_LAG_S`. Drop
  consumed entries to keep queues short.
- **Remove the hard speed clamp at line 94.** Terminal velocity will be
  set naturally by force/drag balance.
- Keep MAX_SPEED_CMS in `constants.py` for reference / fallback only.

**Sim constants — `PySaacSim/sim/constants.py`**
- Add `STEER_TRANSPORT_LAG_S = 0.030` (~30 ms typical chain).
- Add `THROTTLE_TRANSPORT_LAG_S = 0.030`.

**Calib package — `PySaacSim/calib/`**
- `replay.py`: pass new lag params via the `overrides` dict the same way
  existing dynamics params are.
- `dynamics_fit.py`:
  - Extend `PARAM_NAMES` to 8: append `STEER_TRANSPORT_LAG_S`,
    `THROTTLE_TRANSPORT_LAG_S`.
  - Extend `DEFAULT_X0` to `[14.22, 0.10, 0.78, 0.0, 5.0, 0.22, 0.030, 0.030]`
    (start from prior fit, free the pinned ones).
  - Extend `BOUNDS`:
    - SERVO_RAD_PER_SEC: tighten lower from 1.0 → **4.0** (force realism;
      below 4 means we're hiding latency in slew).
    - LINEAR_DRAG: relax upper from 2.0 → **10.0** (terminal velocity
      from drag balance needs higher drag than 2.0 with measured forces).
    - MOTOR_LAG_TAU_S: keep (0.02, 0.5) — should naturally rise once
      transport lag absorbs that role.
    - STEER_TRANSPORT_LAG_S: bounds (0.0, 0.150).
    - THROTTLE_TRANSPORT_LAG_S: bounds (0.0, 0.150).
- `ir_xlsx.py`: switch default xlsx path or just pass new path via CLI
  (the file is `lab-7-tweinstein-1/RTOS_SensorBoard/Sensor_Calib.xlsx`,
  IR data in sheet `IRs`, columns `inches, mm, IR_Left_ADC, IR_Right_ADC`
  — same column names, so the fitter should work as-is, just point at
  the new file/sheet).
- `tfluna_xlsx.py`: implement the per-lidar fit. Sheet `LiDARs`,
  columns `inches, mm, Left LiDAR, Center LiDAR, Right LiDAR`. Fit
  `reported_mm = scale * true_mm + bias_mm` per side via least squares.
  Promote `LidarCalibration` to per-lidar scale/bias if the three
  diverge >5%.
- `imu_bias.py`: drop the
  `WARNING: accel_y bias is much larger…` block at lines 53-58.
  The user confirmed the steady CSV is genuinely motors-off-still
  (any earlier "pitch" reading was incidental).
- `noise_fit.py`: **stop using `IMU_steady_state.csv` for IR / TFLuna
  noise** — the user clarified those columns are unreliable in that
  CSV (motor board unpowered, sensors logged but not pointed at
  calibration targets). Source IR / TFLuna noise std from a stationary
  *prefix* of the driving logs instead (find rows where `time_ms` <
  some threshold AND `gyro_z` ≈ 0 AND consecutive sensor readings
  match within a few units). If no such window exists, leave the
  existing calibration noise values in place and flag in summary.txt.
  IMU noise from the steady CSV is still valid (and now even cleaner —
  pure sensor noise without motor vibration).

**Sim calibration schema — `PySaacSim/sim/calibration.py`,
`PySaacSim/config/calibration.yaml`**
- If per-lidar fit shows divergence: add per-lidar `scale_*`/`bias_*`
  fields. Otherwise leave `LidarCalibration` shared and just write the
  better averaged values to the YAML.

**CLI — `PySaacSim/scripts/calibrate_from_log.py`**
- Already patched to use `encoding="utf-8"` for write_text. No further
  edits needed beyond pointing `--ir-xlsx` at `Sensor_Calib.xlsx`.

**Tests**
- Update `tests/test_replay_identity.py` (or add a new one) to cover
  the transport-delay path: queue an `apply_command`, advance physics
  by less than `LAG`, assert state setpoint unchanged; advance past
  `LAG`, assert it changed.
- Existing 12 passing tests should continue to pass; the IR pipeline
  test in particular needs no change because the API surface is the
  same.

### Verification sequence

1. Run `pytest tests/` — all green (existing 12 + 1 new transport test).
2. Re-run `scripts/calibrate_from_log.py` with `--ir-xlsx Sensor_Calib.xlsx`
   and `--max-evals 500`. Expect:
   - Loss < 58K (the v1 result).
   - SERVO_RAD_PER_SEC > 4.0 (above tightened floor).
   - MOTOR_LAG_TAU_S > 0.02 (free of floor).
   - Both transport-lag params nonzero and < 100 ms.
   - **accel_y in sim now oscillates** in overlay PNGs, tracking real.
3. Holdout/train ratio still < 2.0.
4. Use `scripts/regen_artifacts.py` (already in repo from this session)
   if the run crashes at the artifact-write step again.
5. Inspect overlays for both CW and CCW logs at fitted params before
   `--apply`. Confirm: signs match, gyro_z amplitudes match within
   ~10%, accel_y now non-flat.
6. `--apply` writes the YAML; **manually port** the 8 fitted dynamics
   params into `sim/constants.py` (the CLI doesn't do this). Diff the
   change and commit.
7. Smoke-test PPO with the new calibration for 100 k steps; reward
   trends upward.

### Critical files for Phase 7

Edit:
- `PySaacSim/sim/physics.py` — remove governor (line 94), add command queues, add lag-honoring logic in `_integrate_dynamics`.
- `PySaacSim/sim/constants.py` — add `STEER_TRANSPORT_LAG_S`, `THROTTLE_TRANSPORT_LAG_S`.
- `PySaacSim/calib/dynamics_fit.py` — extend param vector + bounds.
- `PySaacSim/calib/replay.py` — pass new overrides through.
- `PySaacSim/calib/tfluna_xlsx.py` — implement (currently a no-op).
- `PySaacSim/calib/imu_bias.py` — drop pitch warning.
- `PySaacSim/sim/calibration.py` — possibly add per-lidar fields.
- `PySaacSim/config/calibration.yaml` — refit values.

Read-only inputs:
- `lab-7-tweinstein-1/RTOS_SensorBoard/Sensor_Calib.xlsx` — primary data source for IR + LiDAR fits (replaces older IR_Calib.xlsx).
- `lab-7-tweinstein-1/IMU_steady_state.csv` — pure bias (user-confirmed).
- `lab-7-tweinstein-1/{robot_clockwise, robot_counter_clockwise, more_CW, more_CCW}.csv` — driving logs for dynamics fit.

### Notes on data conditions (user clarification 2026-04-22)

- `robot_clockwise.csv` / `robot_counter_clockwise.csv`: **NOT** full
  speed/charge — slower laps. Useful for low-speed dynamics validation.
- `more_CW.csv` / `more_CCW.csv`: **near-full speed**. Primary fit data
  for high-speed dynamics + governor-removal test (sim should reach
  near-cap velocity here without the explicit clamp).
- `IMU_steady_state.csv`: pure bias capture; trust means.

### What to flag back to user after the run

- Final 8-param fit values + which (if any) still pin at bounds.
- Per-channel RMS error sim-vs-real on the 4 driving logs (esp. accel_y
  since it's the headline mismatch we're trying to fix).
- Whether per-lidar fit shows enough divergence to justify per-lidar
  fields, or whether one shared `LidarCalibration` still fits all three.

## Post-exit-plan-mode housekeeping (memory updates)

These cannot be written in plan mode but should be committed once it exits:
- `memory/project_robot_measurements.md`: top speed 0.47 → 1.219 m/s (full
  charge, 2026-04-22). The 0.47 figure is stale and was wrong.
- `memory/project_pending_measurements.md`: remove top-speed TBD; add
  motors-off IMU capture as the new pending item.
- New memory `memory/project_firmware_runtime.md` (or extend
  `reference_firmware.md`): record the runtime facts in the
  "Confirmed firmware facts" section above so we don't re-investigate
  every session — Robot at 12.5 Hz, IMU at 50 Hz with DLPF=44 Hz + AVG=4,
  per-side IR formula constants, servo center 3120 (not 3200), CAP_STEERING
  = 35, motor PWM 200 Hz / 10000-count period, CAN payload format.
- New memory or update to `feedback_legacy_naming.md`: the v1 plan got the
  IMU filter wrong (assumed 10 Hz / 16 samples) by trusting prose summaries
  in IMU_behavior.md instead of `IMU.h` constants. Lesson: always verify
  filter / rate constants from the .h file, not the doc.
