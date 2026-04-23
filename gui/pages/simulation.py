"""Simulation / Training page: canvas + controls + live training."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter, QTextEdit,
    QVBoxLayout, QWidget,
)

from ...control.pd_baseline import PDBaseline, compute_geometry, ir_correction
from ...env.robot_env import (
    RobotEnv, build_observation, sensors_to_mm, steer_deg_to_servo_count,
)
from ...sim.constants import (
    MAX_SPEED_CMS, MOTOR_PWM_MAX_COUNT, PHYSICS_DT_S, SERVO_CENTER_COUNT,
    SERVO_MAX_COUNT, SERVO_MIN_COUNT, STEER_LIMIT_RAD,
)
from ...sim.geometry import chassis_segments
from ...sim.imu import IMUSimulator
from ...sim.model import action_to_delta, apply_residual
from ...sim.physics import RobotState, apply_command, initial_robot, step_physics
from ...sim.sensors import sample_sensors
from ..app_state import AppState, RobotSpec
from ..canvas import TrackCanvas
from ..training.linear_policy import (
    ACTION_NAMES, FEATURE_NAMES, policy_forward,
)
from ..training.worker import TrainingWorker
from ..widgets.mpl_canvas import RewardCurve, WeightHeatmap


CTRL_PERIOD_MS = 80.0
PHYSICS_HZ = 60  # how often the GUI timer fires

class SimulationPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._selected_robot: Optional[int] = None
        self._worker: Optional[TrainingWorker] = None
        self._W: Optional[np.ndarray] = None
        self._b: Optional[np.ndarray] = None
        self._steps_since_ctrl: dict[int, float] = {}
        # Per-robot policy state for the residual RL controller (see
        # _compute_command "rl" branch). Lazy-init in _ensure_rl_state.
        self._rl_state: dict[int, dict] = {}

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- left: canvas -------------------------------------------------
        self.canvas = TrackCanvas(state, mode="sim")
        self.canvas.empty_placed.connect(self._on_canvas_empty_placed)
        self.canvas.robot_clicked.connect(self._on_robot_selected)
        self.canvas.robot_moved.connect(self._on_robot_moved)
        self.canvas.rotate_selected_requested.connect(self._rotate_selected)
        self.canvas.robot_theta_changed.connect(self._on_ring_angle)
        splitter.addWidget(self.canvas)

        # --- right: controls (scrollable) --------------------------------
        right_inner = QWidget()
        v = QVBoxLayout(right_inner)

        v.addWidget(self._build_robots_group())
        v.addWidget(self._build_selected_robot_group())
        v.addWidget(self._build_playback_group())
        v.addWidget(self._build_training_group())
        v.addStretch(1)

        right_inner.setMinimumWidth(480)
        right = QScrollArea()
        right.setWidgetResizable(True)
        right.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        right.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        right.setMinimumWidth(500)
        right.setWidget(right_inner)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, 520])

        # Page-level keyboard shortcuts (also work when canvas lacks focus).
        QShortcut(QKeySequence("R"), self,
                  activated=lambda: self._rotate_selected(-1))
        QShortcut(QKeySequence("Shift+R"), self,
                  activated=lambda: self._rotate_selected(+1))

        root = QHBoxLayout(self)
        root.addWidget(splitter)

        # state sync
        state.robots_changed.connect(self._refresh_robot_list)
        state.controllers_changed.connect(self._refresh_controller_choices)
        self._refresh_robot_list()

        # physics timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._running = False
        self._time_scale = 1.0

    # ---- UI builders ------------------------------------------------------

    def _build_robots_group(self) -> QGroupBox:
        g = QGroupBox("Robots")
        lay = QVBoxLayout(g)
        hint = QLabel("Click empty track to add · drag to move · right-click to "
                      "rotate/delete")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        lay.addWidget(hint)

        self.robot_list = QListWidget()
        self.robot_list.itemSelectionChanged.connect(self._on_list_selection)
        lay.addWidget(self.robot_list)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Controller:"))
        self.ctrl_combo = QComboBox()
        self.ctrl_combo.currentTextChanged.connect(self._on_ctrl_chosen)
        ctrl_row.addWidget(self.ctrl_combo, 1)
        lay.addLayout(ctrl_row)

        btn_row = QHBoxLayout()
        btn_clear = QPushButton("Clear all")
        btn_clear.clicked.connect(self.state.clear_robots)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_clear)
        lay.addLayout(btn_row)
        self._refresh_controller_choices()
        return g

    def _build_selected_robot_group(self) -> QGroupBox:
        g = QGroupBox("Selected robot — position & heading")
        lay = QVBoxLayout(g)
        hint = QLabel("Drag on empty track to place with heading · "
                      "press R to rotate clockwise, Shift+R counter-clockwise")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        lay.addWidget(hint)

        row = QHBoxLayout()
        self.sel_x = QDoubleSpinBox(); self.sel_x.setRange(-1e4, 1e4); self.sel_x.setDecimals(2)
        self.sel_y = QDoubleSpinBox(); self.sel_y.setRange(-1e4, 1e4); self.sel_y.setDecimals(2)
        self.sel_theta = QDoubleSpinBox(); self.sel_theta.setRange(-360, 360); self.sel_theta.setDecimals(1)
        self.sel_theta.setSuffix("°")
        for lbl, w in (("x", self.sel_x), ("y", self.sel_y), ("θ", self.sel_theta)):
            col = QVBoxLayout()
            col.addWidget(QLabel(lbl))
            col.addWidget(w)
            row.addLayout(col)
        lay.addLayout(row)
        self.sel_x.valueChanged.connect(self._on_sel_edited)
        self.sel_y.valueChanged.connect(self._on_sel_edited)
        self.sel_theta.valueChanged.connect(self._on_sel_edited)
        return g

    def _build_playback_group(self) -> QGroupBox:
        g = QGroupBox("Playback")
        lay = QVBoxLayout(g)

        row = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._reset_runtimes)
        row.addWidget(self.btn_play)
        row.addWidget(self.btn_reset)
        lay.addLayout(row)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed:"))
        self.speed = QSlider(Qt.Orientation.Horizontal)
        self.speed.setMinimum(10)   # 0.1x
        self.speed.setMaximum(400)  # 4.0x
        self.speed.setValue(100)
        self.speed.valueChanged.connect(lambda v: setattr(self, "_time_scale", v / 100.0))
        self.speed_lbl = QLabel("1.0x")
        self.speed.valueChanged.connect(lambda v: self.speed_lbl.setText(f"{v/100:.1f}x"))
        speed_row.addWidget(self.speed, 1)
        speed_row.addWidget(self.speed_lbl)
        lay.addLayout(speed_row)

        # Ghost mode: when ON, cars pass through each other and don't appear
        # in each other's sensors. When OFF (default), cars are solid — they
        # collide and each appears as an obstacle in the others' lidar/IR.
        # Turn ghost ON for pure wall-following preview / faster training
        # visualization; turn OFF to evaluate car-to-car avoidance behavior.
        ghost_row = QHBoxLayout()
        self.chk_cars_interact = QCheckBox("Cars see & collide with each other")
        self.chk_cars_interact.setChecked(True)
        self.chk_cars_interact.setToolTip(
            "ON  — cars are solid: they sense each other's chassis as "
            "obstacles and physics stops them on contact (realistic multi-car).\n"
            "OFF — ghost mode: cars ignore each other entirely (fast preview, "
            "isolated wall-following behavior)."
        )
        ghost_row.addWidget(self.chk_cars_interact)
        ghost_row.addStretch(1)
        lay.addLayout(ghost_row)

        lay.addWidget(QLabel("Live sensor readings (selected robot):"))
        self.telemetry = QTextEdit()
        self.telemetry.setReadOnly(True)
        self.telemetry.setFixedHeight(170)
        self.telemetry.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;"
                                     " background: #111; color: #ddd;")
        self.telemetry.setPlaceholderText(
            "Select a robot and press Play to see live pose + sensor values here."
        )
        lay.addWidget(self.telemetry)
        return g

    def _build_training_group(self) -> QGroupBox:
        g = QGroupBox("RL Training — Linear Policy (W·x + b)")
        lay = QVBoxLayout(g)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        self.total_steps = QSpinBox()
        self.total_steps.setRange(1_000, 5_000_000)
        self.total_steps.setSingleStep(10_000)
        self.total_steps.setValue(200_000)
        self.total_steps.setMinimumWidth(110)
        self.total_steps.setGroupSeparatorShown(True)

        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(5)
        self.lr.setSingleStep(1e-4)
        self.lr.setRange(1e-6, 1e-1)
        self.lr.setValue(3e-4)
        self.lr.setMinimumWidth(110)

        self.device = QComboBox()
        self.device.addItems(["cpu", "cuda"])
        self.device.setMinimumWidth(110)
        self.device.setToolTip(
            "For this 8-input / 2-output linear policy CPU is typically "
            "faster — PCIe transfer dominates GPU compute."
        )

        self.n_envs = QSpinBox()
        self.n_envs.setRange(1, 32)
        self.n_envs.setValue(4)
        self.n_envs.setMinimumWidth(110)
        self.n_envs.setToolTip(
            "Number of parallel envs feeding one PPO policy.\n"
            "  • Same-scene OFF: N independent track copies (1 car each).\n"
            "  • Same-scene ON : N cars sharing ONE track, seeing each other."
        )

        # 2-column grid — labels left, widgets right.
        grid.addWidget(QLabel("Total steps:"), 0, 0)
        grid.addWidget(self.total_steps,       0, 1)
        grid.addWidget(QLabel("LR:"),          0, 2)
        grid.addWidget(self.lr,                0, 3)
        grid.addWidget(QLabel("Device:"),      1, 0)
        grid.addWidget(self.device,            1, 1)
        grid.addWidget(QLabel("Parallel envs:"), 1, 2)
        grid.addWidget(self.n_envs,            1, 3)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        lay.addLayout(grid)

        row2 = QHBoxLayout()
        self.chk_respawn = QCheckBox("Auto-respawn on crash")
        self.chk_respawn.setChecked(True)
        self.chk_respawn.setToolTip(
            "When a robot collides, reset it to its spawn pose so it keeps "
            "running. Applies to all controllers (RL, PD, C, manual)."
        )
        row2.addWidget(self.chk_respawn)
        row2.addStretch(1)
        lay.addLayout(row2)

        row3 = QHBoxLayout()
        self.chk_same_scene = QCheckBox("Same-scene multi-agent (cars see each other)")
        self.chk_same_scene.setChecked(False)
        self.chk_same_scene.setToolTip(
            "When on, all N parallel envs share ONE track and cars collide "
            "with / sense each other. Policy learns car-to-car avoidance in "
            "addition to wall-following. Requires Parallel envs > 1."
        )
        row3.addWidget(self.chk_same_scene)
        row3.addStretch(1)
        lay.addLayout(row3)

        # ---- checkpoint / resume controls --------------------------------
        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Save path:"))
        self.save_path = QLineEdit("checkpoints/wall_follower_ppo")
        self.save_path.setToolTip(
            "Base path (no extension). SB3 appends .zip. Checkpoints and "
            "the final model both overwrite this file."
        )
        save_row.addWidget(self.save_path, 1)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._choose_save_path)
        save_row.addWidget(btn_browse)
        lay.addLayout(save_row)

        save_row2 = QHBoxLayout()
        save_row2.addWidget(QLabel("Save every:"))
        self.save_every = QSpinBox()
        self.save_every.setRange(0, 10_000_000)
        self.save_every.setSingleStep(10_000)
        self.save_every.setValue(50_000)
        self.save_every.setGroupSeparatorShown(True)
        self.save_every.setToolTip("Checkpoint interval in timesteps. 0 = only save at end.")
        save_row2.addWidget(self.save_every)
        save_row2.addWidget(QLabel("steps"))
        save_row2.addStretch(1)
        self.chk_resume = QCheckBox("Continue from existing model")
        self.chk_resume.setToolTip(
            "ON  — load the .zip at Save path (if it exists) and continue training.\n"
            "OFF — ignore any existing file and start a fresh model (will overwrite)."
        )
        save_row2.addWidget(self.chk_resume)
        lay.addLayout(save_row2)

        btn_row = QHBoxLayout()
        self.btn_train = QPushButton("Start Training")
        self.btn_train.clicked.connect(self._toggle_training)
        self.btn_apply = QPushButton("Use current weights on RL robots")
        self.btn_apply.clicked.connect(self._apply_weights_note)
        btn_row.addWidget(self.btn_train)
        btn_row.addWidget(self.btn_apply)
        lay.addLayout(btn_row)

        self.train_status = QLabel("idle")
        self.train_status.setStyleSheet("color: #666;")
        self.train_status.setWordWrap(True)
        lay.addWidget(self.train_status)

        lay.addWidget(QLabel("Training log:"))
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setFixedHeight(120)
        self.train_log.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        self.train_log.setPlaceholderText("Training output (errors, progress) appears here.")
        lay.addWidget(self.train_log)

        self.reward_curve = RewardCurve(self)
        self.reward_curve.setFixedHeight(180)
        lay.addWidget(self.reward_curve)

        self.weight_heatmap = WeightHeatmap(FEATURE_NAMES, ACTION_NAMES, self)
        self.weight_heatmap.setFixedHeight(180)
        lay.addWidget(self.weight_heatmap)
        return g

    # ---- robot list sync -------------------------------------------------

    def _refresh_robot_list(self) -> None:
        prev = self._selected_robot
        self.robot_list.blockSignals(True)
        self.robot_list.clear()
        for r in self.state.robots:
            text = f"#{r.id}  ({r.x:.1f}, {r.y:.1f}, {math.degrees(r.theta):+.1f}°) — {r.controller_id}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, r.id)
            self.robot_list.addItem(item)
        if prev is not None:
            for i in range(self.robot_list.count()):
                if self.robot_list.item(i).data(Qt.ItemDataRole.UserRole) == prev:
                    self.robot_list.setCurrentRow(i)
                    break
        self.robot_list.blockSignals(False)
        self.engine.ensure_runtimes(self.state.robots)

    def _refresh_controller_choices(self) -> None:
        current = self.ctrl_combo.currentText() or "manual-drive"
        self.ctrl_combo.blockSignals(True)
        self.ctrl_combo.clear()
        self.ctrl_combo.addItem("manual-drive")   # constant forward throttle
        self.ctrl_combo.addItem("manual-stop")
        self.ctrl_combo.addItem("rl")
        for name in self.state.controllers.keys():
            self.ctrl_combo.addItem(name)
        idx = self.ctrl_combo.findText(current)
        if idx >= 0:
            self.ctrl_combo.setCurrentIndex(idx)
        self.ctrl_combo.blockSignals(False)

    def _on_list_selection(self) -> None:
        items = self.robot_list.selectedItems()
        if not items:
            self._selected_robot = None
        else:
            self._selected_robot = items[0].data(Qt.ItemDataRole.UserRole)
        self._sync_selected_spinboxes()
        self.canvas.highlight_robot(self._selected_robot)
        self._update_telemetry()

    def _sync_selected_spinboxes(self) -> None:
        spec = next((r for r in self.state.robots
                     if r.id == self._selected_robot), None)
        if spec is None:
            return
        idx = self.ctrl_combo.findText(spec.controller_id)
        if idx >= 0:
            self.ctrl_combo.blockSignals(True)
            self.ctrl_combo.setCurrentIndex(idx)
            self.ctrl_combo.blockSignals(False)
        for w, v in ((self.sel_x, spec.x), (self.sel_y, spec.y),
                     (self.sel_theta, math.degrees(spec.theta))):
            w.blockSignals(True); w.setValue(v); w.blockSignals(False)

    def _on_robot_selected(self, robot_id: int) -> None:
        for i in range(self.robot_list.count()):
            if self.robot_list.item(i).data(Qt.ItemDataRole.UserRole) == robot_id:
                self.robot_list.setCurrentRow(i)
                break

    def _on_ctrl_chosen(self, text: str) -> None:
        if self._selected_robot is None or not text:
            return
        self.state.update_robot(self._selected_robot, controller_id=text)

    def _on_canvas_empty_placed(self, x: float, y: float, theta: float) -> None:
        ctrl = self.ctrl_combo.currentText() or "manual-drive"
        spec = self.state.add_robot(x, y, theta=theta, controller_id=ctrl)
        self._selected_robot = spec.id
        self._sync_selected_spinboxes()
        self.canvas.highlight_robot(spec.id)

    def _on_robot_moved(self, rid: int, x: float, y: float, theta: float) -> None:
        self.state.update_robot(rid, x=x, y=y, theta=theta)
        if rid == self._selected_robot:
            self._sync_selected_spinboxes()
        # Running state stays in sync: snap live runtime to the new spawn.
<<<<<<< HEAD
        self.engine.force_robot_pose(rid, x, y, theta)
=======
        if rid in self._runtimes:
            self._runtimes[rid] = initial_robot(x, y, theta)
            self._steps_since_ctrl[rid] = 0.0
            self._reset_rl_state(rid)
>>>>>>> origin/main

    def _on_sel_edited(self) -> None:
        if self._selected_robot is None:
            return
        self.state.update_robot(
            self._selected_robot,
            x=self.sel_x.value(),
            y=self.sel_y.value(),
            theta=math.radians(self.sel_theta.value()),
        )
        # keep canvas item + runtime in sync
        rid = self._selected_robot
        item = self.canvas._robot_items.get(rid)
        if item is not None:
            item.setPos(self.sel_x.value(), self.sel_y.value())
            item.setRotation(self.sel_theta.value())
<<<<<<< HEAD
        self.engine.force_robot_pose(rid, self.sel_x.value(), self.sel_y.value(), math.radians(self.sel_theta.value()))
=======
        if rid in self._runtimes:
            self._runtimes[rid] = initial_robot(
                self.sel_x.value(), self.sel_y.value(),
                math.radians(self.sel_theta.value()))
            self._steps_since_ctrl[rid] = 0.0
            self._reset_rl_state(rid)
>>>>>>> origin/main

    def _on_ring_angle(self, rid: int, theta: float) -> None:
        self.state.update_robot(rid, theta=theta)
        self._sync_selected_spinboxes()
        item = self.canvas._robot_items.get(rid)
        if item is not None:
            item.setRotation(math.degrees(theta))
        if rid in self.engine.runtimes:
            spec = next((r for r in self.state.robots if r.id == rid), None)
            if spec is not None:
<<<<<<< HEAD
                self.engine.force_robot_pose(rid, spec.x, spec.y, theta)
=======
                self._runtimes[rid] = initial_robot(spec.x, spec.y, theta)
                self._steps_since_ctrl[rid] = 0.0
                self._reset_rl_state(rid)
>>>>>>> origin/main
        self.canvas.sync_ring_to_selected()

    def _rotate_selected(self, direction: int) -> None:
        """direction: +1 = ccw, -1 = cw, in 15° steps."""
        if self._selected_robot is None and self.state.robots:
            self._selected_robot = self.state.robots[-1].id
        if self._selected_robot is None:
            return
        spec = next((r for r in self.state.robots
                     if r.id == self._selected_robot), None)
        if spec is None:
            return
        new_theta = spec.theta + direction * math.radians(15)
        self.state.update_robot(spec.id, theta=new_theta)
        self._sync_selected_spinboxes()
        item = self.canvas._robot_items.get(spec.id)
        if item is not None:
            item.setRotation(math.degrees(new_theta))
<<<<<<< HEAD
        if spec.id in self.engine.runtimes:
            self.engine.force_robot_pose(spec.id, spec.x, spec.y, new_theta)
=======
        if spec.id in self._runtimes:
            self._runtimes[spec.id] = initial_robot(spec.x, spec.y, new_theta)
            self._steps_since_ctrl[spec.id] = 0.0
            self._reset_rl_state(spec.id)
>>>>>>> origin/main

    # ---- sim loop --------------------------------------------------------

    def _toggle_play(self) -> None:
        if self._running:
            self._timer.stop()
            self._running = False
            self.btn_play.setText("▶ Play")
            return
        self.engine.ensure_runtimes(self.state.robots)
        self._timer.start(int(1000 / PHYSICS_HZ))
        self._running = True
        self.btn_play.setText("⏸ Pause")

    def _reset_runtimes(self) -> None:
<<<<<<< HEAD
        self.engine.reset_runtimes(self.state.robots)
        self.canvas._rebuild_robots()  # snap visuals back
        self._update_telemetry()

=======
        self._runtimes = {}
        self._steps_since_ctrl = {}
        self._rl_state = {}
        self._ensure_runtimes()
        self.canvas._rebuild_robots()  # snap visuals back
        self._update_telemetry()

    def _ensure_runtimes(self) -> None:
        for r in self.state.robots:
            if r.id not in self._runtimes:
                self._runtimes[r.id] = initial_robot(r.x, r.y, r.theta)
                self._steps_since_ctrl[r.id] = 0.0

    def _prune_runtimes(self) -> None:
        live_ids = {r.id for r in self.state.robots}
        for rid in list(self._runtimes.keys()):
            if rid not in live_ids:
                self._runtimes.pop(rid, None)
                self._steps_since_ctrl.pop(rid, None)
                self._rl_state.pop(rid, None)

>>>>>>> origin/main
    def _tick(self) -> None:
        self.engine.time_scale = self._time_scale
        self.engine.cars_interact = self.chk_cars_interact.isChecked()
        self.engine.auto_respawn = self.chk_respawn.isChecked()
        self.engine.controllers = self.state.controllers
        
        # Pass linear policy evaluation if W and b are set
        if self._W is not None and self._b is not None:
            self.engine.rl_policy = lambda obs: policy_forward(self._W, self._b, obs)
        else:
            self.engine.rl_policy = None

        self.engine.tick_hz(PHYSICS_HZ, self.state.robots, self.state.walls, 
                            self.state.dims, self.state.calibration)

        # Push to canvas visuals.
        for r in self.state.robots:
<<<<<<< HEAD
            state = self.engine.runtimes.get(r.id)
            if state is None: continue
=======
            state = self._runtimes[r.id]
            # When cars_interact: other cars' chassis edges act as dynamic wall
            # segments so this robot both SEES them (lidar/IR rays) and COLLIDES
            # with them (inside step_physics). Otherwise ghost mode — each car
            # is alone in its own perception of the track.
            if cars_interact:
                other_walls = self._other_car_segments(r.id, dims)
                effective_walls = walls + other_walls
            else:
                effective_walls = walls
            for _ in range(n_phys):
                # Apply control at CTRL_PERIOD boundaries.
                self._steps_since_ctrl[r.id] += dt * 1000.0
                if self._steps_since_ctrl[r.id] >= CTRL_PERIOD_MS:
                    self._steps_since_ctrl[r.id] = 0.0
                    sensors = sample_sensors(effective_walls, state.pose, cal)
                    duty_l, duty_r, servo = self._compute_command(r, state, sensors)
                    apply_command(state, duty_l, duty_r, servo)
                step_physics(state, effective_walls, dims.chassis_length_cm,
                             dims.chassis_width_cm, dt)
                if state.collided:
                    break
            # Auto-respawn on crash so the robot keeps running.
            if state.collided and self.chk_respawn.isChecked():
                self._runtimes[r.id] = initial_robot(r.x, r.y, r.theta)
                self._steps_since_ctrl[r.id] = 0.0
                if r.controller_id == "rl":
                    self._reset_rl_state(r.id)
                state = self._runtimes[r.id]
            # Push to canvas visuals.
>>>>>>> origin/main
            item = self.canvas._robot_items.get(r.id)
            if item is not None:
                item.setPos(state.pose.x, state.pose.y)
                item.setRotation(math.degrees(state.pose.theta))
        self.canvas.sync_ring_to_selected()
        self._update_telemetry()

<<<<<<< HEAD
=======
    def _other_car_segments(self, me_id: int, dims) -> list:
        """Chassis segments of every other live robot, in world frame.
        These become obstacles for this robot's sensors and physics."""
        out = []
        for other in self.state.robots:
            if other.id == me_id:
                continue
            rt = self._runtimes.get(other.id)
            if rt is None:
                continue
            pose = (rt.pose.x, rt.pose.y, rt.pose.theta)
            out.extend(chassis_segments(pose, dims.chassis_length_cm,
                                        dims.chassis_width_cm))
        return out

    def _ensure_rl_state(self, rid: int) -> dict:
        s = self._rl_state.get(rid)
        if s is None:
            s = {
                "pd": PDBaseline(),
                "imu": IMUSimulator(),
                "prev_thr_l": 0.0,
                "prev_thr_r": 0.0,
                "prev_steer": 0.0,
                "sim_t_s": 0.0,
            }
            self._rl_state[rid] = s
        return s

    def _reset_rl_state(self, rid: int) -> None:
        s = self._ensure_rl_state(rid)
        s["pd"].reset()
        s["imu"].reset(v_cms=0.0, t_s=0.0)
        s["prev_thr_l"] = 0.0
        s["prev_thr_r"] = 0.0
        s["prev_steer"] = 0.0
        s["sim_t_s"] = 0.0

    def _compute_command(self, spec: RobotSpec, state: RobotState, sensors: dict):
        """Returns (duty_l, duty_r, servo_count) — differential drive."""
        cid = spec.controller_id
        if cid == "manual-stop":
            return 0, 0, SERVO_CENTER_COUNT
        if cid == "manual-drive":
            return 9000, 9000, SERVO_CENTER_COUNT
        if cid == "rl":
            # Residual-on-PD: policy outputs a delta added to the PD baseline.
            # Untrained policy (zero W, zero b) ⇒ delta=0 ⇒ pure PD behavior,
            # so the car drives reasonably even before training kicks in.
            rs = self._ensure_rl_state(spec.id)
            d_ir, ld_ir, d2, ld2, front = sensors_to_mm(sensors)
            d_ir, ld_ir = ir_correction(d_ir, ld_ir, d2, ld2)
            geom = compute_geometry(d_ir, ld_ir, d2, ld2, front)
            pd = rs["pd"].tick(geom)
            if self._W is None or self._b is None:
                # No model loaded — fall back to pure PD (safe default).
                thr_l, thr_r, steer_deg = pd.throttle_l, pd.throttle_r, pd.steering
            else:
                rs["sim_t_s"] += CTRL_PERIOD_MS / 1000.0
                gz, ax, ay = rs["imu"].read(state, rs["sim_t_s"])
                obs = build_observation(
                    sensors, gz, ax, ay,
                    rs["prev_thr_l"], rs["prev_thr_r"], rs["prev_steer"],
                )
                action = policy_forward(self._W, self._b, obs)
                d_thr_l, d_thr_r, d_steer = action_to_delta(action)
                thr_l, thr_r, steer_deg = apply_residual(
                    pd.throttle_l, pd.throttle_r, pd.steering,
                    d_thr_l, d_thr_r, d_steer,
                )
            rs["prev_thr_l"] = thr_l
            rs["prev_thr_r"] = thr_r
            rs["prev_steer"] = steer_deg
            return int(thr_l), int(thr_r), steer_deg_to_servo_count(steer_deg)
        # Named C/Python controller (e.g. pd_controller.c) — already differential.
        ctrl = self.state.controllers.get(cid)
        if ctrl is None:
            return 0, 0, SERVO_CENTER_COUNT
        cmd = ctrl.tick(sensors, 0.0)
        return cmd.duty_l, cmd.duty_r, cmd.servo

>>>>>>> origin/main
    def _update_telemetry(self) -> None:
        if self._selected_robot is None:
            self.telemetry.setPlainText("(no robot selected — click on one in the list or track)")
            return
        spec = next((r for r in self.state.robots if r.id == self._selected_robot), None)
        if spec is None:
            self.telemetry.setPlainText("(selection invalid)")
            return
        state = self.engine.runtimes.get(self._selected_robot)
        from ...sim.physics import Pose
        pose = state.pose if state is not None else Pose(spec.x, spec.y, spec.theta)
        v    = state.v if state is not None else 0.0
        om   = state.omega if state is not None else 0.0
        steer = state.steer_angle if state is not None else 0.0
        collided = state.collided if state is not None else False
        try:
            sensors = self.engine.last_sensors.get(spec.id, {})
            if not sensors:
                return # No telemetry yet
        except Exception as e:
            self.telemetry.setPlainText(f"(sensor sample failed: {e})")
            return
        lines = [
            f"robot #{spec.id}  ctrl={spec.controller_id}  "
            f"{'RUNNING' if self._running else 'stopped'}",
            f"pose   x={pose.x:6.2f} cm  y={pose.y:6.2f} cm  "
            f"θ={math.degrees(pose.theta):+7.2f}°",
            f"v={v:6.2f} cm/s  ω={om:+6.3f} rad/s  "
            f"steer={math.degrees(steer):+5.1f}°  collided={collided}",
            "",
            "Lidar (cm):",
            f"  center = {sensors['lidar']['center']['distance_cm']:6.1f}"
            f"   valid={sensors['lidar']['center']['valid']}",
            f"  left   = {sensors['lidar']['left']['distance_cm']:6.1f}"
            f"   valid={sensors['lidar']['left']['valid']}",
            f"  right  = {sensors['lidar']['right']['distance_cm']:6.1f}"
            f"   valid={sensors['lidar']['right']['valid']}",
            "",
            "IR (cm):",
            f"  left   = {sensors['ir']['left']['distance_cm']:5.2f}  "
            f"adc={sensors['ir']['left']['adc']:4d}  valid={sensors['ir']['left']['valid']}",
            f"  right  = {sensors['ir']['right']['distance_cm']:5.2f}  "
            f"adc={sensors['ir']['right']['adc']:4d}  valid={sensors['ir']['right']['valid']}",
        ]
        self.telemetry.setPlainText("\n".join(lines))

    # ---- training --------------------------------------------------------

    def _toggle_training(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self.btn_train.setText("Start Training")
            return
        self.reward_curve.clear()
        save_path = self.save_path.text().strip() or None
        self._worker = TrainingWorker(
            walls=list(self.state.walls),
            calibration=self.state.calibration,
            total_timesteps=self.total_steps.value(),
            learning_rate=self.lr.value(),
            device=self.device.currentText(),
            n_envs=self.n_envs.value(),
            same_scene=self.chk_same_scene.isChecked(),
            save_path=save_path,
            save_every=self.save_every.value(),
            resume=self.chk_resume.isChecked(),
        )
        self._worker.iteration.connect(self._on_train_iter)
        self._worker.weights.connect(self._on_train_weights)
        self._worker.state_changed.connect(self._on_train_state)
        self._worker.log.connect(self._append_train_log)
        self._worker.finished.connect(lambda: self.btn_train.setText("Start Training"))
        self.train_log.clear()
        self._append_train_log("Starting training worker…")
        self._worker.start()
        self.btn_train.setText("Stop Training")

    def _on_train_iter(self, steps: int, mean_r: float) -> None:
        self.reward_curve.add_point(steps, mean_r)
        self.train_status.setText(f"step={steps:,}  mean_ep_reward={mean_r:.2f}")

    def _on_train_weights(self, W, b) -> None:
        self._W, self._b = W, b
        self.weight_heatmap.update_weights(W, b)

    def _on_train_state(self, s: str) -> None:
        self._append_train_log(f"[state] {s}")
        if s == "done":
            self.train_status.setText("training complete — weights applied")
        elif s == "stopped":
            self.train_status.setText("training stopped (see log below)")
        elif s == "running":
            self.train_status.setText("training running…")
        elif s == "paused":
            self.train_status.setText("training paused")

    def _append_train_log(self, msg: str) -> None:
        self.train_log.append(msg.rstrip())
        sb = self.train_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _apply_weights_note(self) -> None:
        if self._W is None:
            self.train_status.setText("No weights yet — start training first.")
            return
        self.train_status.setText("Weights are already live on RL robots.")

    def _choose_save_path(self) -> None:
        """Pick a .zip checkpoint path. We strip the extension so SB3 can
        re-append it consistently when saving and loading."""
        current = self.save_path.text().strip() or "checkpoints/wall_follower_ppo"
        start = str(Path(current).with_suffix(".zip"))
        path, _ = QFileDialog.getSaveFileName(
            self, "Model save path", start,
            "PPO checkpoint (*.zip);;All files (*)"
        )
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() == ".zip":
            p = p.with_suffix("")
        self.save_path.setText(str(p))
