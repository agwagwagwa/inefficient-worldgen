"""
CLI for extracting chunks from a Minecraft world.

Usage:
    uv run python -m inefficient_worldgen.extract \
        --world ./training-world \
        --output ./extracted_chunks
"""

import argparse
from pathlib import Path

from .chunk_extractor import extract_world


def main():
    parser = argparse.ArgumentParser(description="Extract chunks from Minecraft world")
    parser.add_argument(
        "--world",
        type=str,
        required=True,
        help="Path to Minecraft world directory (contains level.dat)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./extracted_chunks",
        help="Output directory for .npy chunk files",
    )
    parser.add_argument(
        "--dimension",
        type=str,
        default="overworld",
        choices=["overworld", "nether", "end"],
        help="Which dimension to extract",
    )
    args = parser.parse_args()

    world_path = Path(args.world)
    if not world_path.exists():
        raise FileNotFoundError(f"World directory not found: {world_path}")
    if not (world_path / "level.dat").exists() and not (world_path / "region").exists():
        raise FileNotFoundError(
            f"Doesn't look like a Minecraft world: {world_path}\n"
            f"Expected to find level.dat or region/ directory."
        )

    chunks = extract_world(world_path, args.output, dimension=args.dimension)
    print(f"\nDone! Extracted {len(chunks)} chunks to {args.output}")
    print("Next step: run training with:")
    print(f"  uv run python -m inefficient_worldgen.train --chunk-dir {args.output}")


if __name__ == "__main__":
    main()
