# PySaacSim Web UI Port Implementation Plan

Moving a ~3,000-line PyQt desktop application to the web requires a multi-phase approach to ensure the original GUI continues functioning exactly as before, while completely rebuilding the visual and interactive layers in HTML5, CSS, and JS (alongside a new set of API endpoints for the web). 

This plan ensures total backward compatibility. The desktop GUI will continue reading the standard `sim/world` and `gui/app_state` Python files natively, while the FastAPI server exposes that exact same underlying data to the web app via REST and WebSockets.

---

## Phase 1: API Foundation & State Management (FastAPI)
The existing desktop GUI uses a shared `app_state.py` containing PyQT `QObjects` and `pyqtSignals`. We cannot use PyQT signals natively on the web.
- **Task 1.1**: Create `server/routers/gui_state.py` to hold CRUD operations for Track layouts, Robot configurations, and Controller source code.
- **Task 1.2**: Implement endpoints (e.g., `GET /api/track`, `PUT /api/track`) that read and write JSON representations of the track walls, splines, and sensor calibrations.
- **Task 1.3**: Sync this data with the existing `.yaml` or JSON persistence formats used by the desktop app so both clients edit the same saved files.

## Phase 2: Controller Editor 
The simplest graphical feature to port is the text editor and compilation loop.
- **Task 2.1**: Update `server/static/index.html` to include a new "Controller" tab.
- **Task 2.2**: Integrate a lightweight web code editor (like Monaco Editor or CodeMirror) inside `server/static/controller.js`.
- **Task 2.3**: Hook up the editor to load the current Python control logic (`GET /api/controller`) and save modifications securely (`PUT /api/controller`).
- **Task 2.4**: Add a web-based "Compile/Test" button returning parsing errors to the UI (using the existing AbstractController validation logic).

## Phase 3: Track Builder 
This involves porting the `gui/canvas.py` and `gui/pages/track_builder.py` 2D manipulation logic to an HTML5 `<canvas>`.
- **Task 3.1**: Create `server/static/track_builder.js` managing an interactive HTML5 Canvas.
- **Task 3.2**: Implement line drawing, point selection, wall snapping, and curve calculation in JavaScript to mirror the PyQT `QGraphicsScene` math.
- **Task 3.3**: Sync drawn tracks to the backend. Modify the Track schema to broadcast WS events across the app when tracks are saved.

## Phase 4: Robot Builder
Porting `gui/pages/robot_builder.py`, allowing users to construct and visualize the robot chassis and sensor rays.
- **Task 4.1**: Create `server/static/robot_builder.js` to render the `RobotDims` and `SensorCalibration` configuration visually.
- **Task 4.2**: Add HTML forms for adjusting chassis length, width, max speed, mass, etc.
- **Task 4.3**: Dynamically re-render the 2D bounding box and LiDAR raycasts on an HTML canvas preview as input values change.

## Execution

Because this is a large migration, the best way to execute it safely without breaking existing logic is incrementally. I am ready to implement the **Phase 1 backend APIs** and the **Phase 2 Controller Editor** right now if you approve.