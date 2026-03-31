"""
FastAPI server for chunk diffusion WebUI.

Provides endpoints for:
- Listing/loading region files
- Extracting and viewing chunks
- Running step-by-step diffusion generation
"""

import asyncio
from pathlib import Path
from typing import Optional
import json

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anvil

from .palette import (
    CHUNK_X,
    CHUNK_Y,
    CHUNK_Z,
    KERNEL_SIZE,
    NUM_BLOCK_TYPES,
    BLOCK_NAMES,
    BLOCK_COLORS,
)
from .chunk_extractor import extract_chunk_fast
from .dataset import STATE_REAL, STATE_MASK, STATE_TARGET, make_state_volume
from .unet3d import ChunkUNet3D
from .diffusion import CategoricalDiffusion

app = FastAPI(title="Chunk Diffusion API", version="0.1.0")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global state
class AppState:
    model: Optional[ChunkUNet3D] = None
    diffusion: Optional[CategoricalDiffusion] = None
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_world_path: Optional[Path] = None
    checkpoint_path: Optional[Path] = None

    # Current diffusion session
    session_active: bool = False
    session_grid: Optional[torch.Tensor] = None  # (1, Y, 3Z, 3X) integer
    session_state_map: Optional[np.ndarray] = None  # (3, 3)
    session_current_t: int = 0
    session_x_t: Optional[torch.Tensor] = None  # Current noised chunk
    session_total_steps: int = 200


state = AppState()


# --- Pydantic models ---


class ConfigRequest(BaseModel):
    training_world_path: str
    checkpoint_path: str
    num_timesteps: int = 200


class RegionInfo(BaseModel):
    filename: str
    region_x: int
    region_z: int
    path: str


class ChunkCoord(BaseModel):
    cx: int
    cz: int


class KernelRequest(BaseModel):
    center_cx: int
    center_cz: int


class NeighborState(BaseModel):
    """3x3 grid of neighbor states. 0=REAL, 1=MASK, 2=TARGET (center only)"""

    states: list[list[int]]  # 3x3 grid, row-major (Z then X)


class StartDiffusionRequest(BaseModel):
    center_cx: int
    center_cz: int
    neighbor_states: list[list[int]]  # 3x3 grid
    num_steps: int = 200
    temperature: float = 0.8


class StepRequest(BaseModel):
    num_steps: int = 1  # How many steps to advance


# --- Helper functions ---


def get_region_dir() -> Path:
    if state.training_world_path is None:
        raise HTTPException(status_code=400, detail="Training world not configured")
    region_dir = state.training_world_path / "region"
    if not region_dir.exists():
        raise HTTPException(status_code=404, detail="Region directory not found")
    return region_dir


def chunk_to_json(chunk: np.ndarray) -> dict:
    """Convert a chunk array to JSON-serializable format.

    Returns dict with:
        shape: [Y, Z, X]
        data: flat list of block indices (row-major: Y, then Z, then X)
    """
    return {
        "shape": list(chunk.shape),
        "data": chunk.flatten().tolist(),
    }


def load_region_chunk(
    region_path: Path, local_x: int, local_z: int
) -> Optional[np.ndarray]:
    """Load a single chunk from a region file."""
    try:
        region = anvil.Region.from_file(str(region_path))
        return extract_chunk_fast(region, local_x, local_z)
    except Exception:
        return None


# --- Endpoints ---


@app.get("/")
async def root():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "training_world": str(state.training_world_path)
        if state.training_world_path
        else None,
        "session_active": state.session_active,
    }


@app.post("/config")
async def configure(req: ConfigRequest):
    """Configure the API with paths to training world and model checkpoint."""
    state.training_world_path = Path(req.training_world_path)
    state.checkpoint_path = Path(req.checkpoint_path)

    if not state.training_world_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Training world not found: {req.training_world_path}",
        )
    if not state.checkpoint_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Checkpoint not found: {req.checkpoint_path}"
        )

    # Load model
    print(f"Loading checkpoint: {state.checkpoint_path}")
    ckpt = torch.load(
        state.checkpoint_path, map_location=state.device, weights_only=False
    )
    train_args = ckpt.get("args", {})

    state.model = ChunkUNet3D(
        base_channels=train_args.get("base_channels", 48),
        block_embed_dim=train_args.get("block_embed_dim", 16),
    ).to(state.device)
    state.model.load_state_dict(ckpt["model"])
    state.model.eval()

    state.diffusion = CategoricalDiffusion(
        num_timesteps=req.num_timesteps,
        noise_schedule="cosine",
        device=state.device,
    )
    state.session_total_steps = req.num_timesteps

    return {
        "status": "ok",
        "epoch": ckpt.get("epoch", -1) + 1,
        "loss": ckpt.get("loss", None),
        "device": str(state.device),
    }


@app.get("/palette")
async def get_palette():
    """Get the block palette info for rendering."""
    return {
        "num_types": NUM_BLOCK_TYPES,
        "names": BLOCK_NAMES,
        "colors": [list(c) for c in BLOCK_COLORS],  # RGBA 0-1
    }


@app.get("/regions")
async def list_regions():
    """List all available region files."""
    region_dir = get_region_dir()
    regions = []

    for mca_path in sorted(region_dir.glob("*.mca")):
        parts = mca_path.stem.split(".")
        if len(parts) == 3 and parts[0] == "r":
            try:
                rx, rz = int(parts[1]), int(parts[2])
                regions.append(
                    RegionInfo(
                        filename=mca_path.name,
                        region_x=rx,
                        region_z=rz,
                        path=str(mca_path),
                    )
                )
            except ValueError:
                continue

    return {"regions": regions}


@app.get("/region/{region_x}/{region_z}")
async def get_region_chunks(region_x: int, region_z: int):
    """Get all chunks in a region, mapped to our palette.

    Returns a dict mapping "cx,cz" to chunk data for all non-empty chunks.
    """
    region_dir = get_region_dir()
    region_path = region_dir / f"r.{region_x}.{region_z}.mca"

    if not region_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Region not found: r.{region_x}.{region_z}.mca"
        )

    try:
        region = anvil.Region.from_file(str(region_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load region: {e}")

    chunks = {}
    for local_z in range(32):
        for local_x in range(32):
            chunk = extract_chunk_fast(region, local_x, local_z)
            if chunk is not None and chunk.max() > 0:
                global_cx = region_x * 32 + local_x
                global_cz = region_z * 32 + local_z
                chunks[f"{global_cx},{global_cz}"] = chunk_to_json(chunk)

    return {
        "region_x": region_x,
        "region_z": region_z,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


@app.get("/region/{region_x}/{region_z}/summary")
async def get_region_summary(region_x: int, region_z: int):
    """Get just the list of available chunks in a region (without voxel data)."""
    region_dir = get_region_dir()
    region_path = region_dir / f"r.{region_x}.{region_z}.mca"

    if not region_path.exists():
        raise HTTPException(status_code=404, detail=f"Region not found")

    try:
        region = anvil.Region.from_file(str(region_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load region: {e}")

    available = []
    for local_z in range(32):
        for local_x in range(32):
            try:
                chunk = region.get_chunk(local_x, local_z)
                if chunk is not None:
                    global_cx = region_x * 32 + local_x
                    global_cz = region_z * 32 + local_z
                    available.append({"cx": global_cx, "cz": global_cz})
            except Exception:
                continue

    return {
        "region_x": region_x,
        "region_z": region_z,
        "available_chunks": available,
    }


@app.get("/chunk/{cx}/{cz}")
async def get_chunk(cx: int, cz: int):
    """Get a single chunk by global coordinates."""
    region_x = cx // 32
    region_z = cz // 32
    local_x = cx % 32
    local_z = cz % 32

    region_dir = get_region_dir()
    region_path = region_dir / f"r.{region_x}.{region_z}.mca"

    if not region_path.exists():
        raise HTTPException(status_code=404, detail="Region not found")

    chunk = load_region_chunk(region_path, local_x, local_z)
    if chunk is None:
        raise HTTPException(status_code=404, detail="Chunk not found or empty")

    return {
        "cx": cx,
        "cz": cz,
        "chunk": chunk_to_json(chunk),
    }


@app.post("/kernel")
async def get_kernel(req: KernelRequest):
    """Get the 3x3 kernel of chunks around a center position.

    Returns chunks for all 9 positions in the kernel, with null for missing chunks.
    """
    region_dir = get_region_dir()

    kernel = {}
    for dz in range(-1, 2):
        for dx in range(-1, 2):
            cx = req.center_cx + dx
            cz = req.center_cz + dz

            region_x = cx // 32
            region_z = cz // 32
            local_x = cx % 32
            local_z = cz % 32

            region_path = region_dir / f"r.{region_x}.{region_z}.mca"

            chunk = None
            if region_path.exists():
                chunk = load_region_chunk(region_path, local_x, local_z)

            key = f"{dx},{dz}"  # Relative position
            if chunk is not None and chunk.max() > 0:
                kernel[key] = {
                    "cx": cx,
                    "cz": cz,
                    "chunk": chunk_to_json(chunk),
                }
            else:
                kernel[key] = None

    return {
        "center_cx": req.center_cx,
        "center_cz": req.center_cz,
        "kernel": kernel,
    }


@app.post("/diffusion/start")
async def start_diffusion(req: StartDiffusionRequest):
    """Start a new diffusion session.

    Initializes the grid with the specified chunks and neighbor states,
    and prepares for step-by-step generation.
    """
    if state.model is None:
        raise HTTPException(
            status_code=400, detail="Model not loaded. Call /config first."
        )

    region_dir = get_region_dir()

    # Build the 3x3 grid
    grid = np.zeros(
        (CHUNK_Y, KERNEL_SIZE * CHUNK_Z, KERNEL_SIZE * CHUNK_X), dtype=np.int64
    )
    neighbor_states = np.array(req.neighbor_states, dtype=np.float32)

    if neighbor_states.shape != (3, 3):
        raise HTTPException(status_code=400, detail="neighbor_states must be 3x3")

    # Force center to be TARGET
    neighbor_states[1, 1] = STATE_TARGET

    # Load chunks for REAL neighbors
    for gz in range(3):
        for gx in range(3):
            dz = gz - 1
            dx = gx - 1
            cx = req.center_cx + dx
            cz = req.center_cz + dz

            z_start = gz * CHUNK_Z
            x_start = gx * CHUNK_X

            if neighbor_states[gz, gx] == STATE_REAL:
                # Load the real chunk
                region_x = cx // 32
                region_z = cz // 32
                local_x = cx % 32
                local_z = cz % 32

                region_path = region_dir / f"r.{region_x}.{region_z}.mca"
                chunk = None
                if region_path.exists():
                    chunk = load_region_chunk(region_path, local_x, local_z)

                if chunk is not None:
                    grid[
                        :, z_start : z_start + CHUNK_Z, x_start : x_start + CHUNK_X
                    ] = chunk
                else:
                    # Chunk doesn't exist, treat as MASK
                    neighbor_states[gz, gx] = STATE_MASK

    # Initialize session state
    state.session_grid = torch.from_numpy(grid).unsqueeze(0).to(state.device)
    state.session_state_map = neighbor_states
    state.session_total_steps = req.num_steps
    state.session_current_t = req.num_steps - 1  # Start from t=T-1, go to t=0

    # Initialize x_t with pure noise
    state.session_x_t = torch.randint(
        0,
        NUM_BLOCK_TYPES,
        (1, CHUNK_Y, CHUNK_Z, CHUNK_X),
        device=state.device,
    )

    # Update diffusion if steps changed
    if state.diffusion.num_timesteps != req.num_steps:
        state.diffusion = CategoricalDiffusion(
            num_timesteps=req.num_steps,
            noise_schedule="cosine",
            device=state.device,
        )

    state.session_active = True

    return {
        "status": "started",
        "total_steps": req.num_steps,
        "current_t": state.session_current_t,
        "neighbor_states": neighbor_states.tolist(),
        "initial_chunk": chunk_to_json(
            state.session_x_t[0].cpu().numpy().astype(np.uint8)
        ),
    }


@app.post("/diffusion/step")
async def step_diffusion(req: StepRequest):
    """Advance the diffusion by N steps.

    Returns the current state of the generated chunk after stepping.
    """
    if not state.session_active:
        raise HTTPException(status_code=400, detail="No active diffusion session")

    if state.session_current_t < 0:
        return {
            "status": "complete",
            "current_t": 0,
            "steps_remaining": 0,
            "chunk": chunk_to_json(state.session_x_t[0].cpu().numpy().astype(np.uint8)),
        }

    # Build state volume
    state_map_t = (
        torch.from_numpy(state.session_state_map).unsqueeze(0).to(state.device)
    )
    state_vol = make_state_volume(state_map_t)

    steps_taken = 0
    with torch.no_grad():
        for _ in range(req.num_steps):
            if state.session_current_t < 0:
                break

            state.session_x_t = state.diffusion.p_sample_step(
                state.model,
                state.session_x_t,
                state.session_current_t,
                state.session_grid,
                state_vol,
                temperature=0.8,
            )
            state.session_current_t -= 1
            steps_taken += 1

    is_complete = state.session_current_t < 0

    return {
        "status": "complete" if is_complete else "in_progress",
        "current_t": max(0, state.session_current_t),
        "steps_taken": steps_taken,
        "steps_remaining": max(0, state.session_current_t + 1),
        "chunk": chunk_to_json(state.session_x_t[0].cpu().numpy().astype(np.uint8)),
    }


@app.get("/diffusion/state")
async def get_diffusion_state():
    """Get the current state of the diffusion session without stepping."""
    if not state.session_active:
        raise HTTPException(status_code=400, detail="No active diffusion session")

    return {
        "active": state.session_active,
        "current_t": state.session_current_t,
        "total_steps": state.session_total_steps,
        "steps_remaining": max(0, state.session_current_t + 1),
        "neighbor_states": state.session_state_map.tolist()
        if state.session_state_map is not None
        else None,
        "chunk": chunk_to_json(state.session_x_t[0].cpu().numpy().astype(np.uint8))
        if state.session_x_t is not None
        else None,
    }


@app.post("/diffusion/stop")
async def stop_diffusion():
    """Stop the current diffusion session."""
    state.session_active = False
    state.session_grid = None
    state.session_state_map = None
    state.session_x_t = None
    state.session_current_t = 0

    return {"status": "stopped"}


# --- Run server ---


def run_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
