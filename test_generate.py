"""Consolidated test script for chunk generation.

Subcommands:
  single          Generate a single chunk at a given position using seed neighbors.
  real-neighbors  Generate a chunk using real training data as context, compare to ground truth.
  grid            Generate a full world grid via spiral expansion.

Examples:
  python test_generate.py single
  python test_generate.py single --position 2 0 --timesteps 80 --temperature 0.9
  python test_generate.py real-neighbors --sample-index 500
  python test_generate.py grid --radius 4 --timesteps 200
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from src.inefficient_worldgen.unet3d import ChunkUNet3D
from src.inefficient_worldgen.diffusion import CategoricalDiffusion
from src.inefficient_worldgen.palette import CHUNK_X, CHUNK_Y, CHUNK_Z, NUM_BLOCK_TYPES
from src.inefficient_worldgen.visualize import plot_chunk_overview, plot_world_heightmap


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def load_model(
    checkpoint: str,
    device: torch.device,
) -> tuple[ChunkUNet3D, dict]:
    """Load a trained ChunkUNet3D from a checkpoint file."""
    print(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})

    model = ChunkUNet3D(
        base_channels=train_args.get("base_channels", 48),
        block_embed_dim=train_args.get("block_embed_dim", 16),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("loss")
    loss_str = f", loss={loss:.4f}" if loss is not None else ""
    print(f"  Loaded epoch {epoch}{loss_str}")
    return model, ckpt


def make_diffusion(timesteps: int, device: torch.device) -> CategoricalDiffusion:
    return CategoricalDiffusion(
        num_timesteps=timesteps,
        noise_schedule="cosine",
        device=device,
    )


def print_block_distribution(chunk: np.ndarray) -> None:
    """Print the block-type percentages for a chunk."""
    print("Block distribution:")
    for i in range(NUM_BLOCK_TYPES):
        pct = (chunk == i).sum() / chunk.size * 100
        if pct > 0.1:
            print(f"  {i}: {pct:.1f}%")


# ---------------------------------------------------------------------------
# Subcommand: single
# ---------------------------------------------------------------------------


def cmd_single(args: argparse.Namespace) -> None:
    """Generate a single chunk using procedural seed neighbours."""
    from src.inefficient_worldgen.generate import make_seed_chunks, build_context_grid

    device = torch.device(args.device)
    model, ckpt = load_model(args.checkpoint, device)
    diffusion = make_diffusion(args.timesteps, device)

    seed = make_seed_chunks()
    target_cx, target_cz = args.position

    print(
        f"\nGenerating chunk ({target_cx}, {target_cz}) "
        f"with {args.timesteps} steps, temp={args.temperature} ..."
    )

    # build_context_grid works on a world dict — the seed dict is exactly that
    grid_t, state_vol = build_context_grid(seed, target_cx, target_cz, device)

    with torch.no_grad():
        generated = diffusion.sample(
            model,
            grid_t,
            state_vol,
            temperature=args.temperature,
        )

    chunk = generated[0].cpu().numpy().astype(np.uint8)
    print(f"Generated chunk shape: {chunk.shape}")
    print_block_distribution(chunk)

    out = args.output or f"./generated_chunk_{target_cx}_{target_cz}.png"
    epoch = ckpt.get("epoch", "?")
    plot_chunk_overview(
        chunk,
        title=f"Generated Chunk ({target_cx}, {target_cz}) - Epoch {epoch}",
        save_path=out,
    )
    print(f"\nSaved to {out}")


# ---------------------------------------------------------------------------
# Subcommand: real-neighbors
# ---------------------------------------------------------------------------


def cmd_real_neighbors(args: argparse.Namespace) -> None:
    """Generate a chunk using real training-data neighbours and compare."""
    from src.inefficient_worldgen.dataset import make_state_volume, ChunkKernelDataset

    device = torch.device(args.device)
    model, ckpt = load_model(args.checkpoint, device)
    diffusion = make_diffusion(args.timesteps, device)

    print("Loading real chunks...")
    dataset = ChunkKernelDataset(args.chunk_dir, preload=False)
    print(f"Found {len(dataset)} valid 3x3 kernels")

    idx = args.sample_index if args.sample_index is not None else len(dataset) // 2
    sample = dataset[idx]

    grid = sample["grid"].numpy()
    target_gt = sample["target"].numpy()
    state_map = sample["state_map"].numpy()
    cx, cz = sample["center_cx"], sample["center_cz"]

    gt_col = target_gt[:, 8, 8]
    gt_surface = int(np.where(gt_col != 0)[0].max()) if gt_col.max() > 0 else "N/A"
    print(f"\nUsing real chunk ({cx}, {cz}) [sample {idx}]")
    print(f"Ground truth surface height at center column: {gt_surface}")

    # Zero out center chunk in the input grid
    grid_input = grid.copy()
    grid_input[:, CHUNK_Z : 2 * CHUNK_Z, CHUNK_X : 2 * CHUNK_X] = 0

    grid_t = torch.from_numpy(grid_input).unsqueeze(0).to(device)
    state_map_t = torch.from_numpy(state_map).unsqueeze(0).to(device)
    state_vol = make_state_volume(state_map_t)

    print(
        f"\nGenerating chunk ({args.timesteps} diffusion steps, temp={args.temperature}) ..."
    )
    with torch.no_grad():
        generated = diffusion.sample(
            model,
            grid_t,
            state_vol,
            temperature=args.temperature,
        )

    chunk = generated[0].cpu().numpy().astype(np.uint8)

    gen_col = chunk[:, 8, 8]
    gen_surface = int(np.where(gen_col != 0)[0].max()) if gen_col.max() > 0 else 0
    print(f"Generated surface height at center column: {gen_surface}")
    print(f"Ground truth surface height at center column: {gt_surface}")

    out_gen = args.output or "./generated_real_neighbors.png"
    out_gt = args.output_gt or "./ground_truth_chunk.png"
    epoch = ckpt.get("epoch", "?")

    plot_chunk_overview(
        chunk,
        title=f"Generated with Real Neighbors ({cx}, {cz}) - Epoch {epoch}",
        save_path=out_gen,
    )
    plot_chunk_overview(
        target_gt,
        title=f"Ground Truth ({cx}, {cz})",
        save_path=out_gt,
    )
    print(f"\nSaved {out_gen} and {out_gt}")


# ---------------------------------------------------------------------------
# Subcommand: grid
# ---------------------------------------------------------------------------


def cmd_grid(args: argparse.Namespace) -> None:
    """Generate a full world via spiral expansion and save/visualize."""
    from src.inefficient_worldgen.generate import generate_world, save_world

    device = torch.device(args.device)
    model, ckpt = load_model(args.checkpoint, device)
    diffusion = make_diffusion(args.timesteps, device)

    r = args.radius
    total = (2 * r + 1) ** 2
    print(
        f"\nGenerating world (radius={r}, {total} chunks, {args.timesteps} steps) ..."
    )
    print("This may take a while...")

    world = generate_world(
        model,
        diffusion,
        radius=r,
        temperature=args.temperature,
        device=device,
    )
    print(f"\nGenerated {len(world)} total chunks")

    output_dir = args.output_dir or f"./generated_world_r{r}"
    save_world(world, output_dir)

    heightmap_path = args.output or f"{output_dir}/heightmap.png"
    epoch = ckpt.get("epoch", "?")
    plot_world_heightmap(
        world,
        title=f"Generated World (Epoch {epoch}, r={r})",
        save_path=heightmap_path,
    )
    print(f"\nDone! Heightmap at {heightmap_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test generation from a trained chunk diffusion model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Global options
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/latest.pt",
        help="Path to model checkpoint (default: ./checkpoints/latest.pt)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100,
        help="Number of diffusion timesteps (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (default: cuda)",
    )

    subs = parser.add_subparsers(dest="command", required=True)

    # -- single --
    p_single = subs.add_parser(
        "single",
        help="Generate a single chunk using seed neighbours",
    )
    p_single.add_argument(
        "--position",
        type=int,
        nargs=2,
        default=[2, 0],
        metavar=("CX", "CZ"),
        help="Target chunk coordinate (default: 2 0)",
    )
    p_single.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path (default: ./generated_chunk_CX_CZ.png)",
    )

    # -- real-neighbors --
    p_real = subs.add_parser(
        "real-neighbors",
        help="Generate a chunk using real training data as context",
    )
    p_real.add_argument(
        "--chunk-dir",
        type=str,
        default="./extracted_chunks",
        help="Directory of extracted .npy chunks (default: ./extracted_chunks)",
    )
    p_real.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Dataset sample index (default: middle of dataset)",
    )
    p_real.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG for generated chunk (default: ./generated_real_neighbors.png)",
    )
    p_real.add_argument(
        "--output-gt",
        type=str,
        default=None,
        help="Output PNG for ground truth (default: ./ground_truth_chunk.png)",
    )

    # -- grid --
    p_grid = subs.add_parser(
        "grid",
        help="Generate a full world grid via spiral expansion",
    )
    p_grid.add_argument(
        "--radius",
        type=int,
        default=3,
        help="World radius in chunks (default: 3)",
    )
    p_grid.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save world (default: ./generated_world_rN)",
    )
    p_grid.add_argument(
        "--output",
        type=str,
        default=None,
        help="Heightmap PNG path (default: <output-dir>/heightmap.png)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "single": cmd_single,
        "real-neighbors": cmd_real_neighbors,
        "grid": cmd_grid,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
