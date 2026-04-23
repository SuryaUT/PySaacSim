# PySaacSim sim-to-real calibration pass

## Implementation status (2026-04-22, Elijah_Branch)

**Done, landed, tests green:** Phases 0a–0e, Phase 1 (calib package
scaffold), Phase 2 (IR xlsx per-side + noise fits), Phase 3 (latency xcorr),
CMA-ES dynamics replay infrastructure, report/diff/overlay, CLI, and 14
pytest tests.

**Partially done:** Phase 4 — the fit *runs* end-to-end but only a
27-eval smoke-test has been executed (train loss 218k → holdout 239k on
that tiny budget; numbers only meaningful at the planned ~500 evals).

**Not yet done:** Phase 5 `--apply` (needs a longer CMA-ES run first and
user sign-off on the yaml diff); Phase 6 DR handoff.

### Where to pick up

1. **Run the full 500-eval CMA-ES fit** (expected wall-clock ~30 min):
   ```
   cd /home/elijah/Development/ECE_445M
   PySaacSim/.venv/bin/python PySaacSim/scripts/calibrate_from_log.py \
     --ir-xlsx MSPM0LabProjects/RTOS_SensorBoard/IR_Calib.xlsx \
     --steady-csv MSPM0LabProjects/IMU_steady_state.csv \
     --drive-csvs MSPM0LabProjects/robot_clockwise.csv \
                  MSPM0LabProjects/robot_counter_clockwise.csv \
                  MSPM0LabProjects/more_CW.csv \
                  MSPM0LabProjects/more_CCW.csv \
     --max-evals 500
   ```
   Artifacts land in `PySaacSim/artifacts/calib_<ts>/`: `summary.txt`,
   `diff.txt`, `proposed.yaml`, `dynamics.json`, `overlays/*.png`.

2. **Look at the overlay PNGs first**. If gyro_z sign disagrees between CW
   and CCW logs, stop and fix the sign-convention bug in `calib/replay.py`
   (see Phase 4 pre-optimizer ablation, below). The 27-eval smoke test
   already produced plausible overlays but didn't verify sign consistency
   end-to-end.

3. **Holdout separation.** If `loss_holdout > 2 × loss_train` per the
   plan's pass criterion, stop and investigate (battery / surface drift,
   not a code bug). The 27-eval smoke showed holdout/train ≈ 1.1 which is
   fine; a real 500-eval fit should converge much tighter.

4. **Apply** (with user confirmation):
   ```
   PySaacSim/.venv/bin/python PySaacSim/scripts/calibrate_from_log.py \
     ... --apply
   ```

5. **Memory updates** (per the plan's "Post-exit-plan-mode housekeeping"
   section at the bottom) — not done yet; the `memory/` dir exists but no
   files written.

### Key findings from sensor-only pass (already committed)

- **IR per-side fits drift a lot from firmware** — new Left constants
  `a=83326, b=-498, c=18` vs. firmware `a=137932, b=-859, c=32`. Old
  firmware constants don't fit today's xlsx data (predict 150 mm at
  adc=2023 where xlsx says 76 mm). Recommend flashing new constants to the
  MSPM0 after we confirm a physical re-measurement isn't needed.
- **IR_Calib.xlsx has no TFLuna sheet** (only inch / IR_Left_ADC /
  IR_Right_ADC); `calib/tfluna_xlsx.py` is a graceful no-op. TFLuna
  calibration stays at defaults (`scale=1, bias=0`) until a calibration
  capture is added. TFLuna noise std fit from the steady capture
  (`1.46 mm` on tf_right, the only stable channel).
- **IMU bias** from steady capture: gyro_z=46 LSB (0.35 °/s), accel_x=264,
  accel_y=−1074. The accel_y value is flagged as likely containing a
  ~4° pitch contribution; `calib/imu_bias.py` emits a WARNING. The
  "motors-off flat" capture is still outstanding.
- **Latency**: steer → (−gyro_z) peaks at 160 ms, r=0.95 across all 4
  drive logs. Throttle → accel_y peaks at 80 ms but weakly (r=0.21); log
  conditions don't include many forward-accel bursts after cold start.

### Files touched

Sim:
- `sim/constants.py` — D1/D2/D3 (servo 3120-centered, 1920..4320; max
  speed 122 cm/s), new `SERVO_COUNTS_PER_53DEG`.
- `sim/model.py` — D4 (`CAP_STEERING = 35`).
- `sim/imu.py` — 44 Hz LPF + 4-sample rolling mean, bias-free.
- `sim/sensors.py` — IR pipeline goes volts → ADC (+noise) → firmware
  formula `a/(adc+b)+c` → saturate at 305 mm.
- `sim/calibration.py` — new `IRSideCalibration`, `IMUCalibration`;
  `ir_side(placement_id)`, `ir_firmware_convert`.
- `config/calibration.yaml` — schema updated (per-side IR; `imu:` block).
- `control/pd_baseline.py`, `control/base.py` — comment/constants refresh.
- `gui/pages/robot_builder.py` — IR spinners updated for new fields.

Calib (new package):
- `calib/log_io.py`, `ir_xlsx.py`, `tfluna_xlsx.py`, `windows.py`,
  `imu_bias.py`, `noise_fit.py`, `filters.py`, `replay.py`, `latency.py`,
  `dynamics_fit.py`, `report.py`.
- `scripts/calibrate_from_log.py` — one-shot CLI with `--apply` gate.

Tests (new, 14 pass):
- `tests/test_imu_filter.py` (3), `test_ir_fit_repro.py` (2),
  `test_ir_pipeline.py` (7), `test_log_io.py` (2).

### Caveats for the next session

- **Dependencies:** the fit needs `openpyxl`, `scipy`, `cma` in the venv
  (`PySaacSim/.venv/bin/pip install openpyxl scipy cma pytest`). The
  repo's `requirements.txt` wasn't updated on purpose — those are
  calib-only dependencies, not runtime ones.
- **`Track_Of_Doom.csv`** in `MSPM0LabProjects/` is **out of scope** per
  user instruction — do not pass it to the CLI.
- **`calib/replay.py` mutates `sim.physics` module globals** via
  `setattr` for the CMA-ES overrides. Safe because replay runs on the
  main thread and restores on exit, but don't parallelize replay across
  threads in-process without refactoring this.
- **Matplotlib lazy-imports** — if the venv doesn't have it,
  `plot_overlays` silently returns `[]`. `requirements.txt` already
  includes it for the GUI.

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
