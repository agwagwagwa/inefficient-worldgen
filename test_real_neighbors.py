"""Test generation with REAL chunk neighbors from the training data."""

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")

from src.inefficient_worldgen.unet3d import ChunkUNet3D
from src.inefficient_worldgen.diffusion import CategoricalDiffusion
from src.inefficient_worldgen.dataset import (
    ChunkKernelDataset,
    STATE_REAL,
    STATE_TARGET,
    make_state_volume,
)
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
print(f"Loaded epoch {ckpt['epoch'] + 1}")

# Load dataset to get real chunks
print("Loading real chunks...")
dataset = ChunkKernelDataset("./extracted_chunks", preload=False)
print(f"Found {len(dataset)} valid 3x3 kernels")

# Pick a sample from the dataset - this gives us real neighbors
sample_idx = len(dataset) // 2  # Pick one from the middle
sample = dataset[sample_idx]

grid = sample["grid"].numpy()  # (Y, 3Z, 3X)
target_gt = sample["target"].numpy()  # (Y, Z, X) - ground truth
state_map = sample["state_map"].numpy()  # (3, 3)
cx, cz = sample["center_cx"], sample["center_cz"]

print(f"\nUsing real chunk ({cx}, {cz}) as test case")
print(
    f"Ground truth surface height: {np.where(target_gt[:, 8, 8] != 0)[0].max() if target_gt[:, 8, 8].max() > 0 else 'N/A'}"
)

# Zero out the center (we'll generate it)
grid_input = grid.copy()
grid_input[:, CHUNK_Z : 2 * CHUNK_Z, CHUNK_X : 2 * CHUNK_X] = 0

# Convert to tensor
grid_t = torch.from_numpy(grid_input).unsqueeze(0).to(device)
state_map_t = torch.from_numpy(state_map).unsqueeze(0).to(device)
state_vol = make_state_volume(state_map_t)

# Generate with fewer steps for speed
diffusion = CategoricalDiffusion(
    num_timesteps=100,  # Faster than 200
    noise_schedule="cosine",
    device=device,
)

print("\nGenerating chunk (100 diffusion steps)...")
with torch.no_grad():
    generated = diffusion.sample(
        model,
        grid_t,
        state_vol,
        temperature=0.8,
    )

chunk = generated[0].cpu().numpy().astype(np.uint8)

# Compare heights
gen_surface = np.where(chunk[:, 8, 8] != 0)[0].max() if chunk[:, 8, 8].max() > 0 else 0
gt_surface = (
    np.where(target_gt[:, 8, 8] != 0)[0].max() if target_gt[:, 8, 8].max() > 0 else 0
)
print(f"Generated surface height at center: {gen_surface}")
print(f"Ground truth surface height at center: {gt_surface}")

# Save visualizations
plot_chunk_overview(
    chunk,
    title=f"Generated with Real Neighbors ({cx}, {cz})",
    save_path="./generated_real_neighbors.png",
)
plot_chunk_overview(
    target_gt, title=f"Ground Truth ({cx}, {cz})", save_path="./ground_truth_chunk.png"
)

print("\nSaved generated_real_neighbors.png and ground_truth_chunk.png")
