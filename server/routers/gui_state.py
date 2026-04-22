from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..auth import User, current_user
from ..schemas import ControllerCompileBody, ControllerCompileResponse

router = APIRouter(prefix="/gui", tags=["gui"])

@router.post("/compile_controller", response_model=ControllerCompileResponse)
async def compile_controller(
    body: ControllerCompileBody,
    request: Request,
    user: User = Depends(current_user),
) -> ControllerCompileResponse:
    """Takes C source code, saves it to a temp file, runs CBridge compiler,
    and loads it into the global sim_runner."""
    
    sim_runner = request.app.state.sim_runner
    
    # We call the new method we just injected into sim_runner.
    result = await sim_runner.load_c_controller(body.name, body.code)
    
    if result == "ok":
        return ControllerCompileResponse(status="ok")
    else:
        return ControllerCompileResponse(status="error", error=result)

@router.post("/stop_controller", response_model=dict)
async def stop_controller(
    request: Request,
    user: User = Depends(current_user),
) -> dict:
    """Stops the active C Controller."""
    sim_runner = request.app.state.sim_runner
    await sim_runner.unload_c_controller()
    return {"status": "ok"}
