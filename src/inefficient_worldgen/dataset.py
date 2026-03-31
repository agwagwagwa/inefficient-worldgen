"""
Dataset for 3x3 chunk kernel training.

Each sample is a 3x3 grid of chunks where:
- The center chunk is the TARGET (what the model learns to generate)
- The 8 surrounding chunks are REAL (conditioning context)
- Missing neighbors are MASK (filled with a learned or fixed mask token)

The model input is a tensor of shape:
  (9, CHUNK_Y, CHUNK_Z, CHUNK_X) -- 9 = one-hot encoded block types per chunk position

Actually, let's think about this more carefully. The model needs:
1. The 3x3 grid as spatial context (48x48 in XZ, 128 in Y)
2. A state mask indicating which chunks are REAL vs MASK vs TARGET
3. The target chunk ground truth for the loss

We concatenate the 3x3 grid into a single volume of shape (CHUNK_Y, 3*CHUNK_Z, 3*CHUNK_X)
and provide a per-chunk state channel.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .palette import CHUNK_X, CHUNK_Y, CHUNK_Z, NUM_BLOCK_TYPES, KERNEL_SIZE


# Chunk states
STATE_REAL = 0
STATE_MASK = 1
STATE_TARGET = 2


class ChunkKernelDataset(Dataset):
    """Dataset of 3x3 chunk kernels extracted from a world.

    For every chunk that has all 8 neighbors present, we create a training
    sample where the center is TARGET and neighbors are REAL.
    """

    def __init__(self, chunk_dir: str | Path, preload: bool = True):
        """
        Args:
            chunk_dir: Directory containing chunk_X_Z.npy files.
            preload: If True, load all chunks into memory upfront.
        """
        self.chunk_dir = Path(chunk_dir)
        self.preload = preload

        # Discover all available chunks
        self._discover_chunks()

        # Find all valid 3x3 kernels (center + all 8 neighbors exist)
        self._find_valid_kernels()

        if preload:
            print(f"Preloading {len(self.chunk_index)} chunks into memory...")
            self.chunk_cache: dict[tuple[int, int], np.ndarray] = {}
            for coord, path in self.chunk_index.items():
                self.chunk_cache[coord] = np.load(path)
            print("Done.")

    def _discover_chunks(self):
        """Find all chunk files and index them by (cx, cz) coordinate."""
        self.chunk_index: dict[tuple[int, int], Path] = {}
        for f in self.chunk_dir.glob("chunk_*.npy"):
            parts = f.stem.split("_")
            if len(parts) == 3:
                try:
                    cx, cz = int(parts[1]), int(parts[2])
                    self.chunk_index[(cx, cz)] = f
                except ValueError:
                    continue
        print(f"Found {len(self.chunk_index)} chunks on disk")

    def _find_valid_kernels(self):
        """Find all chunk positions where all 8 neighbors exist."""
        self.valid_centers: list[tuple[int, int]] = []
        coords = set(self.chunk_index.keys())

        for cx, cz in sorted(coords):
            neighbors_ok = True
            for dz in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dz == 0:
                        continue
                    if (cx + dx, cz + dz) not in coords:
                        neighbors_ok = False
                        break
                if not neighbors_ok:
                    break
            if neighbors_ok:
                self.valid_centers.append((cx, cz))

        print(f"Found {len(self.valid_centers)} valid 3x3 kernels")

    def _load_chunk(self, cx: int, cz: int) -> np.ndarray:
        """Load a chunk by coordinate."""
        if self.preload:
            return self.chunk_cache[(cx, cz)]
        return np.load(self.chunk_index[(cx, cz)])

    def __len__(self) -> int:
        return len(self.valid_centers)

    def __getitem__(self, idx: int) -> dict:
        """Get a training sample.

        Returns a dict with:
            'grid': LongTensor of shape (CHUNK_Y, 3*CHUNK_Z, 3*CHUNK_X)
                The full 3x3 grid with block type indices (0-7).
            'state_map': FloatTensor of shape (3, 3)
                Per-chunk state: 0=REAL, 1=MASK, 2=TARGET
            'target': LongTensor of shape (CHUNK_Y, CHUNK_Z, CHUNK_X)
                The ground truth center chunk.
            'center_coords': tuple (cx, cz) for reference.
        """
        cx, cz = self.valid_centers[idx]

        # Build the 3x3 grid (uint8 is sufficient for palette indices 0-7;
        # cast to long only right before F.one_hot in the training loop)
        grid = np.zeros(
            (CHUNK_Y, KERNEL_SIZE * CHUNK_Z, KERNEL_SIZE * CHUNK_X),
            dtype=np.uint8,
        )
        state_map = np.full((KERNEL_SIZE, KERNEL_SIZE), STATE_REAL, dtype=np.float32)

        for gz in range(KERNEL_SIZE):
            for gx in range(KERNEL_SIZE):
                dz = gz - 1  # -1, 0, 1
                dx = gx - 1

                chunk = self._load_chunk(cx + dx, cz + dz)

                y_start = 0
                z_start = gz * CHUNK_Z
                x_start = gx * CHUNK_X

                grid[
                    y_start : y_start + CHUNK_Y,
                    z_start : z_start + CHUNK_Z,
                    x_start : x_start + CHUNK_X,
                ] = chunk

                if dx == 0 and dz == 0:
                    state_map[gz, gx] = STATE_TARGET

        # Extract ground truth target before we zero it in the grid
        target = grid[:, CHUNK_Z : 2 * CHUNK_Z, CHUNK_X : 2 * CHUNK_X].copy()

        return {
            "grid": torch.from_numpy(grid),  # uint8 tensor — saves ~8x memory
            "state_map": torch.from_numpy(state_map),
            "target": torch.from_numpy(target.astype(np.int64)),
            "center_cx": cx,
            "center_cz": cz,
        }


def make_state_volume(state_map: torch.Tensor) -> torch.Tensor:
    """Expand a (B, 3, 3) state map into a volume matching the grid.

    Returns: (B, 1, CHUNK_Y, 3*CHUNK_Z, 3*CHUNK_X) float tensor
    where each voxel has the state value of its parent chunk.
    """
    # state_map: (B, 3, 3) -> tile each cell to fill its chunk region
    # repeat_interleave along Z and X axes to expand 3->48 in each dimension
    vol_2d = state_map.repeat_interleave(CHUNK_Z, dim=1).repeat_interleave(
        CHUNK_X, dim=2
    )  # (B, 3*CHUNK_Z, 3*CHUNK_X)
    # Broadcast along the Y and channel dimensions (expand returns a view --
    # no extra memory allocated; Conv3d handles non-contiguous inputs fine)
    return vol_2d[:, None, None, :, :].expand(-1, 1, CHUNK_Y, -1, -1)
