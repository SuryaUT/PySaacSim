"""Robot customization: close-up canvas + all-measurements form.

Drag sensor handles on the canvas to reposition. Angles and fine details are
edited in the form on the right. Every edit live-updates both sides.
"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QLineF, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QFormLayout, QGraphicsEllipseItem,
    QGraphicsItem, QGraphicsLineItem, QGraphicsScene, QGraphicsView,
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea,
    QSplitter, QVBoxLayout, QWidget,
)

from ...sim.calibration import SensorCalibration, SensorPlacement
from ..app_state import AppState, RobotDims
from ..persistence import load_dims, save_dims


class _SensorHandle(QGraphicsEllipseItem):
    """Draggable circle representing one sensor's body-frame (x, y).

    While dragging, calls `on_move(id, x, y)` which updates state+visuals
    in place without rebuilding the scene. On release, calls `on_release(id)`
    so the rest of the app (sim canvas, forms) can re-sync once."""
    def __init__(self, sensor_id: str, x: float, y: float, color: str,
                 on_move: Callable[[str, float, float], None],
                 on_release: Callable[[str], None]):
        r = 1.6
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.sensor_id = sensor_id
        self._on_move = on_move
        self._on_release = on_release
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor("#111"), 0.3))
        self.setZValue(20)
        self.setPos(x, y)
        # Enable move signals *after* setting initial position.
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._on_move(self.sensor_id, self.pos().x(), self.pos().y())
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        self._on_release(self.sensor_id)


class _RobotCloseup(QGraphicsView):
    """Draws one robot big + sensor handles, for visual editing."""
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.scale(1.0, -1.0)
        self.setBackgroundBrush(QBrush(QColor("#fafafa")))
        self._handles: dict[str, _SensorHandle] = {}
        self._rays: dict[str, QGraphicsLineItem] = {}
        self._ray_lens: dict[str, float] = {}
        self._dragging: bool = False
        state.calibration_changed.connect(self._on_cal_signal)
        state.dims_changed.connect(self._rebuild)
        self._rebuild()

    def _on_cal_signal(self) -> None:
        # Ignore external calibration updates while the user is dragging a
        # handle — rebuilding would destroy the item Qt is tracking.
        if self._dragging:
            return
        self._rebuild()

    def _rebuild(self) -> None:
        self._scene.clear()
        self._handles = {}
        self._rays = {}
        self._ray_lens = {}
        dims = self.state.dims
        cal = self.state.calibration
        L, W = dims.chassis_length_cm, dims.chassis_width_cm

        # Chassis.
        pen = QPen(QColor("#111"), 0.4)
        self._scene.addRect(QRectF(-L / 2, -W / 2, L, W), pen,
                            QBrush(QColor(200, 220, 240, 200)))
        # Centerline + heading arrow.
        arrow_pen = QPen(QColor("#333"), 0.4)
        self._scene.addLine(0, 0, L / 2, 0, arrow_pen)

        # Sensor placements.
        for p in cal.lidar_placements:
            self._add_sensor(p, "#4fc3f7", "lidar", cal)
        for p in cal.ir_placements:
            self._add_sensor(p, "#ff9800", "ir", cal)

        self._scene.setSceneRect(-L - 5, -W - 5, 2 * (L + 5), 2 * (W + 5))
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _add_sensor(self, p: SensorPlacement, color: str, kind: str,
                    cal: SensorCalibration) -> None:
        # Ray pointing in the placement's direction.
        length = cal.lidar.max_cm * 0.06 if kind == "lidar" else cal.ir.max_cm * 0.4
        self._ray_lens[p.id] = length
        ex = p.x + length * math.cos(p.theta)
        ey = p.y + length * math.sin(p.theta)
        pen = QPen(QColor(color), 0.5)
        pen.setStyle(Qt.PenStyle.DashLine)
        ray = self._scene.addLine(p.x, p.y, ex, ey, pen)
        self._rays[p.id] = ray

        handle = _SensorHandle(p.id, p.x, p.y, color,
                               self._on_sensor_moved, self._on_sensor_release)
        handle.setToolTip(p.id)
        self._scene.addItem(handle)
        self._handles[p.id] = handle

    def _on_sensor_moved(self, sensor_id: str, x: float, y: float) -> None:
        """Called on every pixel of drag. Update state data + ray line in place,
        WITHOUT emitting calibration_changed (which would tear down the handle
        Qt is still tracking mid-drag and crash)."""
        self._dragging = True
        cal = self.state.calibration
        theta = 0.0
        for plist in (cal.lidar_placements, cal.ir_placements):
            for i, p in enumerate(plist):
                if p.id == sensor_id:
                    theta = p.theta
                    plist[i] = SensorPlacement(id=p.id, x=x, y=y, theta=theta)
                    break
        ray = self._rays.get(sensor_id)
        L = self._ray_lens.get(sensor_id, 10.0)
        if ray is not None:
            ray.setLine(x, y, x + L * math.cos(theta), y + L * math.sin(theta))

    def _on_sensor_release(self, sensor_id: str) -> None:
        """Drag finished: now it's safe to emit so the rest of the app (form
        spinboxes, sim page canvas) re-syncs."""
        self._dragging = False
        self.state.calibration_changed.emit()


class RobotBuilderPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: menu (scrollable).
        menu_scroll = QScrollArea()
        menu_scroll.setWidgetResizable(True)
        menu = QWidget()
        v = QVBoxLayout(menu)
        v.addWidget(self._build_dims_group())
        v.addWidget(self._build_placements_group())
        v.addWidget(self._build_ir_group())
        v.addWidget(self._build_lidar_group())
        v.addWidget(self._build_file_group())
        v.addStretch(1)
        menu_scroll.setWidget(menu)
        splitter.addWidget(menu_scroll)

        # Right: close-up canvas.
        self.canvas = _RobotCloseup(state)
        splitter.addWidget(self.canvas)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 900])

        root = QHBoxLayout(self)
        root.addWidget(splitter)

        state.calibration_changed.connect(self._reload_forms)
        state.dims_changed.connect(self._reload_forms)

    # ---- groups -----------------------------------------------------------

    def _spin(self, val: float, lo: float, hi: float, step: float,
             on_change: Callable[[float], None], decimals: int = 3) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(decimals)
        s.setValue(val)
        s.valueChanged.connect(on_change)
        return s

    def _build_dims_group(self) -> QGroupBox:
        g = QGroupBox("Chassis & Wheels (cm)")
        lay = QFormLayout(g)
        d = self.state.dims
        self._dim_spins = {}
        for attr, label in [
            ("chassis_length_cm", "Chassis length"),
            ("chassis_width_cm", "Chassis width"),
            ("chassis_height_cm", "Chassis height"),
            ("wheelbase_cm", "Wheelbase"),
            ("rear_track_cm", "Rear track"),
            ("front_wheel_diam_cm", "Front wheel Ø"),
            ("rear_wheel_diam_cm", "Rear wheel Ø"),
            ("max_speed_cms", "Max speed (cm/s)"),
            ("mass_kg", "Mass (kg)"),
        ]:
            s = self._spin(getattr(d, attr), 0.01, 1000.0, 0.1,
                           lambda v, a=attr: self._on_dim_changed(a, v))
            self._dim_spins[attr] = s
            lay.addRow(label, s)
        return g

    def _build_placements_group(self) -> QGroupBox:
        g = QGroupBox("Sensor Placements (body frame)")
        lay = QVBoxLayout(g)
        self._placement_spins: dict[str, tuple[QDoubleSpinBox, QDoubleSpinBox, QDoubleSpinBox]] = {}
        cal = self.state.calibration
        for p in list(cal.lidar_placements) + list(cal.ir_placements):
            sub = QGroupBox(p.id)
            sf = QFormLayout(sub)
            sx = self._spin(p.x, -100, 100, 0.1,
                            lambda v, pid=p.id: self._on_placement_changed(pid, x=v))
            sy = self._spin(p.y, -100, 100, 0.1,
                            lambda v, pid=p.id: self._on_placement_changed(pid, y=v))
            st = self._spin(math.degrees(p.theta), -180, 180, 1,
                            lambda v, pid=p.id: self._on_placement_changed(pid, theta=math.radians(v)),
                            decimals=1)
            sf.addRow("x (cm)", sx)
            sf.addRow("y (cm)", sy)
            sf.addRow("θ (deg)", st)
            lay.addWidget(sub)
            self._placement_spins[p.id] = (sx, sy, st)
        return g

    def _build_ir_group(self) -> QGroupBox:
        g = QGroupBox("IR Calibration")
        lay = QFormLayout(g)
        ir = self.state.calibration.ir
        self._ir_spins = {}
        for attr, label in [
            ("curve_k", "curve_k"),
            ("curve_c", "curve_c"),
            ("peak_cm", "peak_cm"),
            ("v_clamp", "v_clamp"),
            ("min_cm", "min_cm"),
            ("max_cm", "max_cm"),
            ("adc_noise_std", "noise (ADC counts)"),
            ("voltage_noise_std", "noise (V)"),
        ]:
            s = self._spin(getattr(ir, attr), -100, 10000, 0.01,
                           lambda v, a=attr: self._on_ir_changed(a, v))
            self._ir_spins[attr] = s
            lay.addRow(label, s)
        return g

    def _build_lidar_group(self) -> QGroupBox:
        g = QGroupBox("Lidar Calibration")
        lay = QFormLayout(g)
        lc = self.state.calibration.lidar
        self._lidar_spins = {}
        for attr, label in [
            ("max_cm", "max (cm)"),
            ("noise_std_cm", "noise std (cm)"),
            ("distance_scale", "distance_scale"),
            ("distance_bias_cm", "distance_bias (cm)"),
        ]:
            s = self._spin(getattr(lc, attr), -1e4, 1e4, 0.05,
                           lambda v, a=attr: self._on_lidar_changed(a, v))
            self._lidar_spins[attr] = s
            lay.addRow(label, s)
        return g

    def _build_file_group(self) -> QGroupBox:
        g = QGroupBox("Save / Load")
        lay = QVBoxLayout(g)
        row1 = QHBoxLayout()
        btn_save_cal = QPushButton("Save Calibration…")
        btn_load_cal = QPushButton("Load Calibration…")
        btn_save_cal.clicked.connect(self._save_calibration)
        btn_load_cal.clicked.connect(self._load_calibration)
        row1.addWidget(btn_save_cal)
        row1.addWidget(btn_load_cal)
        lay.addLayout(row1)
        row2 = QHBoxLayout()
        btn_save_dims = QPushButton("Save Dims…")
        btn_load_dims = QPushButton("Load Dims…")
        btn_save_dims.clicked.connect(self._save_dims)
        btn_load_dims.clicked.connect(self._load_dims)
        row2.addWidget(btn_save_dims)
        row2.addWidget(btn_load_dims)
        lay.addLayout(row2)
        btn_reset = QPushButton("Reset Robot to Defaults")
        btn_reset.clicked.connect(self._reset)
        lay.addWidget(btn_reset)
        return g

    # ---- change handlers -------------------------------------------------

    def _on_dim_changed(self, attr: str, val: float) -> None:
        setattr(self.state.dims, attr, val)
        self.state.dims_changed.emit()

    def _on_placement_changed(self, pid: str, **changes) -> None:
        cal = self.state.calibration
        for plist in (cal.lidar_placements, cal.ir_placements):
            for i, p in enumerate(plist):
                if p.id == pid:
                    new = SensorPlacement(
                        id=p.id,
                        x=changes.get("x", p.x),
                        y=changes.get("y", p.y),
                        theta=changes.get("theta", p.theta),
                    )
                    plist[i] = new
                    self.state.calibration_changed.emit()
                    return

    def _on_ir_changed(self, attr: str, val: float) -> None:
        setattr(self.state.calibration.ir, attr, val)
        self.state.calibration_changed.emit()

    def _on_lidar_changed(self, attr: str, val: float) -> None:
        setattr(self.state.calibration.lidar, attr, val)
        self.state.calibration_changed.emit()

    # ---- reload forms when state changes externally ----------------------

    def _reload_forms(self) -> None:
        d = self.state.dims
        for attr, spin in self._dim_spins.items():
            spin.blockSignals(True)
            spin.setValue(getattr(d, attr))
            spin.blockSignals(False)
        cal = self.state.calibration
        for p in list(cal.lidar_placements) + list(cal.ir_placements):
            if p.id in self._placement_spins:
                sx, sy, st = self._placement_spins[p.id]
                for s, v in ((sx, p.x), (sy, p.y), (st, math.degrees(p.theta))):
                    s.blockSignals(True); s.setValue(v); s.blockSignals(False)
        for attr, spin in self._ir_spins.items():
            spin.blockSignals(True)
            spin.setValue(getattr(cal.ir, attr))
            spin.blockSignals(False)
        for attr, spin in self._lidar_spins.items():
            spin.blockSignals(True)
            spin.setValue(getattr(cal.lidar, attr))
            spin.blockSignals(False)

    # ---- file I/O --------------------------------------------------------

    def _save_calibration(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Calibration",
                                              "calibration.yaml", "YAML (*.yaml *.yml)")
        if path:
            self.state.calibration.to_yaml(path)

    def _load_calibration(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Calibration",
                                              "", "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            cal = SensorCalibration.from_yaml(path)
        except Exception as e:
            QMessageBox.warning(self, "Load failed", str(e))
            return
        self.state.set_calibration(cal)

    def _save_dims(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Dims",
                                              "robot_dims.yaml", "YAML (*.yaml *.yml)")
        if path:
            save_dims(self.state.dims, path)

    def _load_dims(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Dims",
                                              "", "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            dims = load_dims(path)
        except Exception as e:
            QMessageBox.warning(self, "Load failed", str(e))
            return
        self.state.set_dims(dims)

    def _reset(self) -> None:
        self.state.set_calibration(SensorCalibration.default())
        self.state.set_dims(RobotDims.default())
