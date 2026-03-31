"""
3D U-Net for chunk diffusion.

The model takes:
- A noised version of the target chunk (one-hot encoded, so NUM_BLOCK_TYPES channels)
- The surrounding context (one-hot encoded border chunks)
- A diffusion timestep embedding
- A state volume indicating REAL/MASK/TARGET per voxel

And predicts the noise (or directly predicts x0, depending on the formulation).

Since we're doing discrete/categorical diffusion, we'll predict logits over the
NUM_BLOCK_TYPES classes for each voxel in the target region.

Architecture:
- Input is the full 3x3 grid: (B, C_in, 128, 48, 48)
- Encoder path downsamples spatially
- Decoder path upsamples back
- Skip connections
- Output: (B, NUM_BLOCK_TYPES, 128, 16, 16) -- only the center chunk

We use the full grid as input so the model sees neighbor context,
but we only predict/loss on the center chunk region.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .palette import CHUNK_X, CHUNK_Y, CHUNK_Z, KERNEL_SIZE, NUM_BLOCK_TYPES


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) tensor of timesteps (integer or float).
        Returns:
            (B, dim) embedding.
        """
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb
        )
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ConvBlock3D(nn.Module):
    """Two 3D convolutions with GroupNorm and SiLU, plus timestep conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        # Ensure groups divides both in_ch and out_ch
        groups_1 = min(groups, in_ch)
        while in_ch % groups_1 != 0:
            groups_1 -= 1
        groups_2 = min(groups, out_ch)
        while out_ch % groups_2 != 0:
            groups_2 -= 1

        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups_1, in_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups_2, out_ch)
        self.act = nn.SiLU()

        # Timestep projection
        self.time_proj = nn.Linear(time_dim, out_ch)

        # Residual connection if channel count changes
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
            t_emb: (B, time_dim)
        """
        residual = self.skip(x)

        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Add timestep embedding
        t = self.act(self.time_proj(t_emb))
        h = h + t[:, :, None, None, None]

        h = self.act(self.norm2(h))
        h = self.conv2(h)

        return h + residual


class DownBlock(nn.Module):
    """Downsample: ConvBlock then strided conv to halve spatial dims."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv_block = ConvBlock3D(in_ch, out_ch, time_dim)
        # Stride 2 in all dims to halve resolution
        self.downsample = nn.Conv3d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_block(x, t_emb)
        skip = h
        h = self.downsample(h)
        return h, skip


class UpBlock(nn.Module):
    """Upsample: trilinear upsample, concat skip, ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv_block = ConvBlock3D(in_ch + skip_ch, out_ch, time_dim)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:
        # Upsample x to match skip's spatial dimensions
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        h = torch.cat([x, skip], dim=1)
        return self.conv_block(h, t_emb)


class ChunkUNet3D(nn.Module):
    """3D U-Net for chunk diffusion.

    Input channels:
    - block_embed_dim: learned embedding features of the block-id grid
    - 1: state volume (REAL=0, MASK=1, TARGET=2 per voxel)
    Total input: block_embed_dim + 1 channels

    Output: NUM_BLOCK_TYPES logits per voxel in the CENTER chunk only.
    """

    def __init__(
        self,
        out_channels: int = NUM_BLOCK_TYPES,
        base_channels: int = 48,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        time_dim: int = 128,
        block_embed_dim: int = 16,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.block_embed_dim = block_embed_dim
        self.block_embed = nn.Embedding(NUM_BLOCK_TYPES, block_embed_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Build channel sizes
        channels = [base_channels * m for m in channel_mults]

        # Initial projection
        self.input_conv = nn.Conv3d(block_embed_dim + 1, channels[0], 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        in_ch = channels[0]
        for ch in channels[1:]:
            self.down_blocks.append(DownBlock(in_ch, ch, time_dim))
            in_ch = ch

        # Bottleneck
        self.bottleneck = ConvBlock3D(in_ch, in_ch, time_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.up_blocks.append(
                UpBlock(channels[i], channels[i], channels[i - 1], time_dim)
            )

        # Output projection -- predicts logits for center chunk
        groups_final = min(8, channels[0])
        while channels[0] % groups_final != 0:
            groups_final -= 1
        self.output_norm = nn.GroupNorm(groups_final, channels[0])
        self.output_conv = nn.Conv3d(channels[0], out_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        state_vol: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Either:
                - (B, CHUNK_Y, 3*CHUNK_Z, 3*CHUNK_X) integer block ids, or
                - (B, C, CHUNK_Y, 3*CHUNK_Z, 3*CHUNK_X) precomputed features.
            t: (B,) integer timesteps
            state_vol: (B, 1, CHUNK_Y, 3*CHUNK_Z, 3*CHUNK_X) state channel

        Returns:
            (B, NUM_BLOCK_TYPES, CHUNK_Y, CHUNK_Z, CHUNK_X) logits for center chunk
        """
        # Convert integer grid to learned embeddings to avoid full one-hot expansion.
        if x.dim() == 4:
            x_feat = self.block_embed(x.long()).permute(0, 4, 1, 2, 3)
        else:
            x_feat = x

        # Concatenate input channels
        h = torch.cat([x_feat, state_vol], dim=1)

        # Time embedding
        t_emb = self.time_embed(t)

        # Input conv
        h = self.input_conv(h)

        # Encoder
        skips = []
        for down in self.down_blocks:
            h, skip = down(h, t_emb)
            skips.append(skip)

        # Bottleneck
        h = self.bottleneck(h, t_emb)

        # Decoder
        for up, skip in zip(self.up_blocks, reversed(skips)):
            h = up(h, skip, t_emb)

        # Output
        h = F.silu(self.output_norm(h))
        h = self.output_conv(h)

        # Crop to center chunk only
        h = h[
            :,
            :,
            :,
            CHUNK_Z : 2 * CHUNK_Z,
            CHUNK_X : 2 * CHUNK_X,
        ]

        return h  # (B, NUM_BLOCK_TYPES, CHUNK_Y, CHUNK_Z, CHUNK_X)
