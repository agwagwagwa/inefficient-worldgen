"""Generate a small grid of chunks and visualize."""

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.inefficient_worldgen.unet3d import ChunkUNet3D
from src.inefficient_worldgen.diffusion import CategoricalDiffusion
from src.inefficient_worldgen.generate import (
    generate_world,
    save_world,
    make_seed_chunks,
)
from src.inefficient_worldgen.visualize import plot_world_heightmap

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

# Use full 200 timesteps for better quality
diffusion = CategoricalDiffusion(
    num_timesteps=200,
    noise_schedule="cosine",
    device=device,
)

# Generate world with radius 3 (will have 7x7 = 49 chunks total, 9 seed + 40 generated)
print("\nGenerating world (radius=3, 200 steps)...")
print("This will take a few minutes...")

world = generate_world(
    model,
    diffusion,
    radius=3,
    temperature=0.8,
    device=device,
)

print(f"\nGenerated {len(world)} total chunks")

# Save
save_world(world, "./generated_world_r3")

# Visualize full world
plot_world_heightmap(
    world,
    title="Generated World (Epoch 25, r=3)",
    save_path="./generated_world_r3_heightmap.png",
)

print("\nDone! Check generated_world_r3_heightmap.png")
