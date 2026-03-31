"""
Extract chunks from Minecraft region files (.mca) and convert to palette indices.

Reads from a Minecraft world's region/ directory, extracts each chunk,
maps all blocks to our 8-type palette, and crops to Y=0..127.

Output: numpy arrays of shape (128, 16, 16) with dtype uint8, values 0-7.

This uses direct NBT access and bitwise unpacking for speed, avoiding
the slow per-block object creation of anvil-parser's high-level API.
"""

from pathlib import Path
import math

import anvil
import numpy as np
from tqdm import tqdm

from .palette import (
    CHUNK_X,
    CHUNK_Y,
    CHUNK_Z,
    block_name_to_palette,
)


def _unpack_blockstates(
    blockstates: list[int], bits_per_block: int, size: int = 4096
) -> np.ndarray:
    """Unpack a BlockStates long array into block palette indices.

    Minecraft 1.16 uses a packed format where each long (64 bits) contains
    multiple block indices. Blocks don't span across longs.

    Args:
        blockstates: List of signed 64-bit integers from BlockStates tag.
        bits_per_block: Bits used per block (determined by palette size).
        size: Number of blocks to unpack (4096 for a 16x16x16 section).

    Returns:
        1D numpy array of palette indices in YZX order.
    """
    if bits_per_block < 4:
        bits_per_block = 4  # Minimum is 4 bits

    blocks_per_long = 64 // bits_per_block
    mask = (1 << bits_per_block) - 1

    result = np.zeros(size, dtype=np.uint16)
    block_idx = 0

    for long_val in blockstates:
        # Convert signed to unsigned
        if long_val < 0:
            long_val += 1 << 64

        for _ in range(blocks_per_long):
            if block_idx >= size:
                break
            result[block_idx] = long_val & mask
            long_val >>= bits_per_block
            block_idx += 1

    return result


def extract_chunk_fast(
    region: anvil.Region, chunk_x: int, chunk_z: int
) -> np.ndarray | None:
    """Extract a single chunk using direct NBT access for speed.

    Args:
        region: An open anvil.Region object.
        chunk_x: Chunk X coordinate within the region (0-31).
        chunk_z: Chunk Z coordinate within the region (0-31).

    Returns:
        numpy array of shape (CHUNK_Y, CHUNK_Z, CHUNK_X) with dtype uint8,
        or None if the chunk doesn't exist / is empty.
    """
    try:
        chunk = region.get_chunk(chunk_x, chunk_z)
    except Exception:
        return None

    arr = np.zeros((CHUNK_Y, CHUNK_Z, CHUNK_X), dtype=np.uint8)
    num_sections = CHUNK_Y // 16

    for section_y in range(num_sections):
        try:
            section = chunk.get_section(section_y)
            if section is None:
                continue

            # Get palette and build mapping to our 8-type palette
            if "Palette" not in section or "BlockStates" not in section:
                continue

            palette_tag = section["Palette"]
            palette_mapping = []
            for block_tag in palette_tag:
                block_name = str(block_tag["Name"].value)
                palette_mapping.append(block_name_to_palette(block_name))
            palette_mapping = np.array(palette_mapping, dtype=np.uint8)

            # Calculate bits per block from palette size
            palette_size = len(palette_mapping)
            bits_per_block = (
                max(4, math.ceil(math.log2(palette_size))) if palette_size > 1 else 4
            )

            # Unpack blockstates
            blockstates = list(section["BlockStates"].value)
            indices = _unpack_blockstates(blockstates, bits_per_block)

            # Map to our palette
            mapped = palette_mapping[indices]

            # Reshape: indices are in YZX order within the section
            section_blocks = mapped.reshape(16, 16, 16)  # (Y, Z, X)

            # Place into the full chunk array
            y_start = section_y * 16
            y_end = y_start + 16
            arr[y_start:y_end, :, :] = section_blocks

        except Exception as e:
            # Section error -> leave as air
            continue

    return arr


def extract_chunk(
    region: anvil.Region, chunk_x: int, chunk_z: int
) -> np.ndarray | None:
    """Extract a single chunk - wrapper that uses fast method."""
    return extract_chunk_fast(region, chunk_x, chunk_z)


def extract_world(
    world_path: str | Path,
    output_dir: str | Path,
    dimension: str = "overworld",
) -> dict[tuple[int, int], Path]:
    """Extract all chunks from a Minecraft world's region files.

    Args:
        world_path: Path to the Minecraft world directory (contains level.dat).
        output_dir: Where to save extracted chunk .npy files.
        dimension: 'overworld', 'nether', or 'end'.

    Returns:
        Dict mapping (global_chunk_x, global_chunk_z) to saved .npy file paths.
    """
    world_path = Path(world_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find the region directory
    if dimension == "overworld":
        region_dir = world_path / "region"
    elif dimension == "nether":
        region_dir = world_path / "DIM-1" / "region"
    elif dimension == "end":
        region_dir = world_path / "DIM1" / "region"
    else:
        raise ValueError(f"Unknown dimension: {dimension}")

    if not region_dir.exists():
        raise FileNotFoundError(f"Region directory not found: {region_dir}")

    mca_files = sorted(region_dir.glob("*.mca"))
    if not mca_files:
        raise FileNotFoundError(f"No .mca files found in {region_dir}")

    print(f"Found {len(mca_files)} region files in {region_dir}")

    chunk_paths: dict[tuple[int, int], Path] = {}

    for mca_path in tqdm(mca_files, desc="Reading regions"):
        # Parse region coordinates from filename: r.X.Z.mca
        parts = mca_path.stem.split(".")
        if len(parts) != 3 or parts[0] != "r":
            continue
        try:
            region_x = int(parts[1])
            region_z = int(parts[2])
        except ValueError:
            continue

        try:
            region = anvil.Region.from_file(str(mca_path))
        except Exception as e:
            print(f"  Skipping {mca_path.name}: {e}")
            continue

        for local_z in range(32):
            for local_x in range(32):
                chunk_arr = extract_chunk(region, local_x, local_z)
                if chunk_arr is None:
                    continue

                # Skip chunks that are entirely air (ungenerated)
                if chunk_arr.max() == 0:
                    continue

                global_cx = region_x * 32 + local_x
                global_cz = region_z * 32 + local_z

                out_path = output_dir / f"chunk_{global_cx}_{global_cz}.npy"
                np.save(out_path, chunk_arr)
                chunk_paths[(global_cx, global_cz)] = out_path

    print(f"Extracted {len(chunk_paths)} non-empty chunks")
    return chunk_paths


def load_chunk(path: str | Path) -> np.ndarray:
    """Load a previously extracted chunk from a .npy file."""
    return np.load(path)
