"""Quick test: generate a single chunk from the trained model."""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")

from src.inefficient_worldgen.unet3d import ChunkUNet3D
from src.inefficient_worldgen.diffusion import CategoricalDiffusion
from src.inefficient_worldgen.generate import make_seed_chunks
from src.inefficient_worldgen.dataset import STATE_REAL, STATE_TARGET, make_state_volume
from src.inefficient_worldgen.palette import (
    CHUNK_X,
    CHUNK_Y,
    CHUNK_Z,
    KERNEL_SIZE,
    NUM_BLOCK_TYPES,
)
from src.inefficient_worldgen.visualize import plot_chunk_overview

device = torch.device("cuda")

# Load model
print("Loading checkpoint...")
ckpt = torch.load(
    "./checkpoints/epoch_0025.pt", map_location=device, weights_only=False
)
train_args = ckpt.get("args", {})

model = ChunkUNet3D(
    base_channels=train_args.get("base_channels", 48),
    block_embed_dim=train_args.get("block_embed_dim", 16),
).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Loaded epoch {ckpt['epoch'] + 1}, loss was {ckpt['loss']:.4f}")

# Set up diffusion with fewer steps for speed
diffusion = CategoricalDiffusion(
    num_timesteps=50,  # Fewer steps for quick test
    noise_schedule="cosine",
    device=device,
)

# Get seed chunks and build context for generating chunk at (2, 0)
# This is just outside the 3x3 seed, so it has 3 real neighbors
seed = make_seed_chunks()
target_cx, target_cz = 2, 0

print(f"\nBuilding context for chunk ({target_cx}, {target_cz})...")

# Build 3x3 grid centered on target
grid = np.zeros((CHUNK_Y, KERNEL_SIZE * CHUNK_Z, KERNEL_SIZE * CHUNK_X), dtype=np.int64)
state_map = np.full((KERNEL_SIZE, KERNEL_SIZE), STATE_REAL, dtype=np.float32)

for gz in range(KERNEL_SIZE):
    for gx in range(KERNEL_SIZE):
        dz = gz - 1
        dx = gx - 1
        cx, cz = target_cx + dx, target_cz + dz
        z_start = gz * CHUNK_Z
        x_start = gx * CHUNK_X

        if dx == 0 and dz == 0:
            state_map[gz, gx] = STATE_TARGET
        elif (cx, cz) in seed:
            grid[:, z_start : z_start + CHUNK_Z, x_start : x_start + CHUNK_X] = seed[
                (cx, cz)
            ]
        else:
            state_map[gz, gx] = 1  # MASK for missing neighbors

# Convert to tensors
grid_t = torch.from_numpy(grid).unsqueeze(0).to(device)  # (1, Y, 3Z, 3X) integer
state_map_t = torch.from_numpy(state_map).unsqueeze(0).to(device)
state_vol = make_state_volume(state_map_t)

print(f"Grid shape: {grid_t.shape}")
print(f"State vol shape: {state_vol.shape}")

# Generate!
print("\nGenerating chunk (50 diffusion steps)...")
with torch.no_grad():
    generated = diffusion.sample(
        model,
        grid_t,
        state_vol,
        temperature=0.85,
    )

chunk = generated[0].cpu().numpy().astype(np.uint8)
print(f"Generated chunk shape: {chunk.shape}")
print(f"Block distribution:")
for i in range(8):
    pct = (chunk == i).sum() / chunk.size * 100
    if pct > 0.1:
        print(f"  {i}: {pct:.1f}%")

# Visualize
plot_chunk_overview(
    chunk,
    title=f"Generated Chunk ({target_cx}, {target_cz}) - Epoch 25",
    save_path="./generated_chunk_test.png",
)
print("\nSaved to generated_chunk_test.png")
