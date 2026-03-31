"""
Visualization utilities for chunk data.

Provides matplotlib-based rendering for sanity checking:
- Cross-section slices (top-down, side views)
- Heightmap rendering
- Block distribution histograms
- Multi-chunk grid views
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Patch

from .palette import (
    BLOCK_COLORS,
    BLOCK_NAMES,
    CHUNK_X,
    CHUNK_Y,
    CHUNK_Z,
    NUM_BLOCK_TYPES,
)


def _make_colormap():
    """Create a discrete colormap from our block palette."""
    # Use RGB only (drop alpha for the colormap)
    colors = [c[:3] for c in BLOCK_COLORS]
    # Replace air with white for visibility
    colors[0] = (0.95, 0.95, 0.95)
    cmap = mcolors.ListedColormap(colors, N=NUM_BLOCK_TYPES)
    norm = mcolors.BoundaryNorm(range(NUM_BLOCK_TYPES + 1), NUM_BLOCK_TYPES)
    return cmap, norm


CMAP, NORM = _make_colormap()


def _legend_handles():
    """Create legend handles for block types."""
    handles = []
    colors = [c[:3] for c in BLOCK_COLORS]
    colors[0] = (0.95, 0.95, 0.95)
    for i, (name, color) in enumerate(zip(BLOCK_NAMES, colors)):
        handles.append(Patch(facecolor=color, edgecolor="black", label=f"{i}: {name}"))
    return handles


def plot_slice_y(
    chunk: np.ndarray, y: int, title: str = "", ax=None, show: bool = True
):
    """Plot a horizontal slice at a given Y level (top-down view).

    Args:
        chunk: (Y, Z, X) array of block indices.
        y: Y level to slice.
        title: Plot title.
        ax: Matplotlib axis (creates new figure if None).
        show: Whether to call plt.show().
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(
        chunk[y, :, :], cmap=CMAP, norm=NORM, origin="lower", interpolation="nearest"
    )
    ax.set_title(f"{title} (Y={y})" if title else f"Y={y}")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    if show:
        ax.legend(handles=_legend_handles(), loc="upper right", fontsize=6)
        plt.tight_layout()
        plt.show()


def plot_slice_x(
    chunk: np.ndarray, x: int, title: str = "", ax=None, show: bool = True
):
    """Plot a vertical cross-section at a given X (side view, looking east).

    Args:
        chunk: (Y, Z, X) array.
        x: X coordinate to slice.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 10))
    # Show Y on vertical axis, Z on horizontal
    section = chunk[:, :, x]  # (Y, Z)
    ax.imshow(
        section,
        cmap=CMAP,
        norm=NORM,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title(f"{title} (X={x})" if title else f"X={x}")
    ax.set_xlabel("Z")
    ax.set_ylabel("Y")
    if show:
        ax.legend(handles=_legend_handles(), loc="upper right", fontsize=6)
        plt.tight_layout()
        plt.show()


def plot_slice_z(
    chunk: np.ndarray, z: int, title: str = "", ax=None, show: bool = True
):
    """Plot a vertical cross-section at a given Z (side view, looking north).

    Args:
        chunk: (Y, Z, X) array.
        z: Z coordinate to slice.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 10))
    section = chunk[:, z, :]  # (Y, X)
    ax.imshow(
        section,
        cmap=CMAP,
        norm=NORM,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title(f"{title} (Z={z})" if title else f"Z={z}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if show:
        ax.legend(handles=_legend_handles(), loc="upper right", fontsize=6)
        plt.tight_layout()
        plt.show()


def plot_heightmap(chunk: np.ndarray, title: str = "", ax=None, show: bool = True):
    """Plot the heightmap of a chunk (highest non-air block at each XZ position).

    Args:
        chunk: (Y, Z, X) array.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    heightmap = np.zeros((CHUNK_Z, CHUNK_X), dtype=np.int32)
    for z in range(chunk.shape[1]):
        for x in range(chunk.shape[2]):
            for y in range(chunk.shape[0] - 1, -1, -1):
                if chunk[y, z, x] != 0:  # not air
                    heightmap[z, x] = y
                    break

    im = ax.imshow(heightmap, origin="lower", interpolation="nearest", cmap="terrain")
    ax.set_title(title or "Heightmap")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    plt.colorbar(im, ax=ax, label="Height (Y)")
    if show:
        plt.tight_layout()
        plt.show()


def plot_block_distribution(chunk: np.ndarray, title: str = "", show: bool = True):
    """Plot histogram of block type distribution."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    counts = np.bincount(chunk.flatten(), minlength=NUM_BLOCK_TYPES)
    total = counts.sum()
    pcts = counts / total * 100

    colors = [c[:3] for c in BLOCK_COLORS]
    colors[0] = (0.95, 0.95, 0.95)
    bars = ax.bar(range(NUM_BLOCK_TYPES), pcts, color=colors, edgecolor="black")
    ax.set_xticks(range(NUM_BLOCK_TYPES))
    ax.set_xticklabels(BLOCK_NAMES, rotation=45, ha="right")
    ax.set_ylabel("% of voxels")
    ax.set_title(title or "Block Distribution")

    for bar, pct in zip(bars, pcts):
        if pct > 0.5:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{pct:.1f}%",
                ha="center",
                fontsize=8,
            )

    plt.tight_layout()
    if show:
        plt.show()


def plot_chunk_overview(
    chunk: np.ndarray, title: str = "", save_path: str | None = None
):
    """Plot a comprehensive overview of a single chunk: multiple Y slices + side views."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))

    # Top row: Y slices at different heights
    y_levels = [1, 30, 55, 63]
    for i, y in enumerate(y_levels):
        if y < chunk.shape[0]:
            plot_slice_y(chunk, y, title=f"Y={y}", ax=axes[0, i], show=False)

    # Bottom row: X and Z cross-sections + heightmap + distribution
    plot_slice_x(
        chunk, chunk.shape[2] // 2, title="Side (X=8)", ax=axes[1, 0], show=False
    )
    plot_slice_z(
        chunk, chunk.shape[1] // 2, title="Side (Z=8)", ax=axes[1, 1], show=False
    )
    plot_heightmap(chunk, title="Heightmap", ax=axes[1, 2], show=False)

    # Block distribution in last subplot
    counts = np.bincount(chunk.flatten(), minlength=NUM_BLOCK_TYPES)
    total = counts.sum()
    pcts = counts / total * 100
    colors = [c[:3] for c in BLOCK_COLORS]
    colors[0] = (0.95, 0.95, 0.95)
    axes[1, 3].bar(range(NUM_BLOCK_TYPES), pcts, color=colors, edgecolor="black")
    axes[1, 3].set_xticks(range(NUM_BLOCK_TYPES))
    axes[1, 3].set_xticklabels(BLOCK_NAMES, rotation=45, ha="right", fontsize=7)
    axes[1, 3].set_title("Distribution")

    fig.suptitle(title or "Chunk Overview", fontsize=14, fontweight="bold")

    # Add legend
    fig.legend(handles=_legend_handles(), loc="upper right", fontsize=7, ncol=1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def plot_world_heightmap(
    world: dict[tuple[int, int], np.ndarray],
    title: str = "",
    save_path: str | None = None,
):
    """Plot a top-down heightmap of the entire world grid."""
    if not world:
        print("Empty world!")
        return

    # Find bounds
    min_cx = min(cx for cx, cz in world.keys())
    max_cx = max(cx for cx, cz in world.keys())
    min_cz = min(cz for cx, cz in world.keys())
    max_cz = max(cz for cx, cz in world.keys())

    w_chunks = max_cx - min_cx + 1
    h_chunks = max_cz - min_cz + 1

    heightmap = np.full((h_chunks * CHUNK_Z, w_chunks * CHUNK_X), -1, dtype=np.float32)
    surface_blocks = np.zeros((h_chunks * CHUNK_Z, w_chunks * CHUNK_X), dtype=np.int32)

    for (cx, cz), chunk in world.items():
        gx = cx - min_cx
        gz = cz - min_cz
        for z in range(CHUNK_Z):
            for x in range(CHUNK_X):
                for y in range(chunk.shape[0] - 1, -1, -1):
                    if chunk[y, z, x] != 0:
                        heightmap[gz * CHUNK_Z + z, gx * CHUNK_X + x] = y
                        surface_blocks[gz * CHUNK_Z + z, gx * CHUNK_X + x] = chunk[
                            y, z, x
                        ]
                        break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax1.imshow(heightmap, origin="lower", cmap="terrain", interpolation="nearest")
    ax1.set_title("Heightmap")
    plt.colorbar(im1, ax=ax1, label="Y level")

    ax2.imshow(
        surface_blocks, cmap=CMAP, norm=NORM, origin="lower", interpolation="nearest"
    )
    ax2.set_title("Surface Block Types")
    ax2.legend(handles=_legend_handles(), loc="upper right", fontsize=6)

    # Draw chunk grid lines
    for ax in (ax1, ax2):
        for i in range(w_chunks + 1):
            ax.axvline(i * CHUNK_X - 0.5, color="black", linewidth=0.3, alpha=0.3)
        for j in range(h_chunks + 1):
            ax.axhline(j * CHUNK_Z - 0.5, color="black", linewidth=0.3, alpha=0.3)
        ax.set_xlabel("Block X")
        ax.set_ylabel("Block Z")

    fig.suptitle(title or "World Overview", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
