"""
Discrete diffusion for categorical voxel data.

We use an "absorbing state" discrete diffusion process inspired by D3PM
(Austin et al., 2021), simplified for our use case.

The idea:
- Forward process: at each timestep, each voxel has an increasing probability
  of being replaced with a uniform random block type (noise).
- Reverse process: the model predicts the clean block type distribution
  for each voxel, and we denoise step by step.

This is conceptually similar to masked language modeling (BERT), but with a
gradual noising schedule rather than binary mask/no-mask.

Concretely:
- At t=0, data is clean (original blocks).
- At t=T, data is fully uniform random noise.
- For intermediate t, each voxel is independently either kept clean or
  replaced with uniform noise, with probability that increases with t.

The model predicts logits for the clean x_0 at each step, and we use the
posterior to sample x_{t-1}.
"""

import torch
import torch.nn.functional as F

from .palette import NUM_BLOCK_TYPES


class CategoricalDiffusion:
    """Absorbing-state discrete diffusion process for categorical data."""

    def __init__(
        self,
        num_classes: int = NUM_BLOCK_TYPES,
        num_timesteps: int = 200,
        noise_schedule: str = "cosine",
        device: str | torch.device = "cuda",
    ):
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.device = device

        # Build noise schedule: probability of replacing a voxel at timestep t
        # This goes from ~0 at t=0 to ~1 at t=T
        if noise_schedule == "cosine":
            steps = torch.linspace(0, 1, num_timesteps + 1, device=device)
            # Cosine schedule (Nichol & Dhariwal style, adapted for discrete)
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            # Probability that a voxel has been corrupted by timestep t
            self.corrupt_prob = (1.0 - alpha_bar[1:]).clamp(0, 0.999)
        elif noise_schedule == "linear":
            self.corrupt_prob = torch.linspace(0.0, 0.999, num_timesteps, device=device)
        else:
            raise ValueError(f"Unknown schedule: {noise_schedule}")

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Forward process: add noise to clean data.

        Args:
            x_0: (B, Y, Z, X) LongTensor of clean block indices (0 to num_classes-1).
            t:   (B,) LongTensor of timesteps (0 to num_timesteps-1).

        Returns:
            x_t: (B, Y, Z, X) LongTensor of noised block indices.
        """
        B = x_0.shape[0]
        # Get corruption probability for each sample in the batch
        prob = self.corrupt_prob[t]  # (B,)
        prob = prob[:, None, None, None]  # (B, 1, 1, 1) for broadcasting

        # Mask: which voxels get corrupted
        mask = torch.rand_like(x_0.float()) < prob  # (B, Y, Z, X) bool

        # Uniform random noise
        noise = torch.randint_like(x_0, 0, self.num_classes)

        # Apply: keep original where not masked, use noise where masked
        x_t = torch.where(mask, noise, x_0)
        return x_t

    def training_loss(
        self,
        model: torch.nn.Module,
        x_0: torch.Tensor,
        grid_int: torch.Tensor,
        state_vol: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the training loss.

        The model predicts x_0 from x_t. We use cross-entropy loss on the
        target chunk voxels.

        Args:
            model: The U-Net model.
            x_0: (B, CHUNK_Y, CHUNK_Z, CHUNK_X) clean target chunk (LongTensor).
            grid_int: (B, CHUNK_Y, 3*CHUNK_Z, 3*CHUNK_X) integer grid of block
                indices. The center chunk region will be replaced with the noised
                version before the model forward pass.
            state_vol: (B, 1, CHUNK_Y, 3*CHUNK_Z, 3*CHUNK_X) state channel.

        Returns:
            Scalar loss tensor.
        """
        from .palette import CHUNK_X, CHUNK_Y, CHUNK_Z

        B = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=x_0.device)

        # Forward process: noise the target chunk
        x_t = self.q_sample(x_0, t)

        # Write noised target into the integer grid center (in-place on the
        # caller's tensor -- the caller must not rely on center being clean).
        z_start, z_end = CHUNK_Z, 2 * CHUNK_Z
        x_start, x_end = CHUNK_X, 2 * CHUNK_X
        grid_int[:, :, z_start:z_end, x_start:x_end] = x_t

        # Model prediction: logits for clean x_0
        logits = model(grid_int, t, state_vol)  # (B, C, Y, Z, X)

        # Cross-entropy loss
        # Reshape for cross_entropy: (B*Y*Z*X, C) vs (B*Y*Z*X,)
        logits_flat = logits.permute(0, 2, 3, 4, 1).reshape(-1, self.num_classes)
        target_flat = x_0.reshape(-1)

        loss = F.cross_entropy(logits_flat, target_flat)
        return loss

    @torch.no_grad()
    def p_sample_step(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        t: int,
        grid_int: torch.Tensor,
        state_vol: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Single reverse diffusion step: predict x_{t-1} from x_t.

        Args:
            model: The U-Net model.
            x_t: (B, CHUNK_Y, CHUNK_Z, CHUNK_X) current noised target.
            t: Current timestep (integer).
            grid_int: Full integer grid with border chunks.
            state_vol: State channel.
            temperature: Sampling temperature (1.0 = normal, <1.0 = more confident).

        Returns:
            x_{t-1}: (B, CHUNK_Y, CHUNK_Z, CHUNK_X) less noised target.
        """
        from .palette import CHUNK_X, CHUNK_Z

        B = x_t.shape[0]
        t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)

        # Place x_t into grid (integer version)
        # grid_int is (B, Y, 3Z, 3X), x_t is (B, Y, Z, X)
        grid_input = grid_int.clone()
        z_start, z_end = CHUNK_Z, 2 * CHUNK_Z
        x_start, x_end = CHUNK_X, 2 * CHUNK_X
        grid_input[:, :, z_start:z_end, x_start:x_end] = x_t

        # Get model prediction (logits for x_0)
        # Model accepts integer grid directly and uses learned embeddings
        logits = model(grid_input, t_tensor, state_vol)  # (B, C, Y, Z, X)
        logits = logits / temperature

        # Convert to probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, Y, Z, X)

        if t == 0:
            # Final step: take argmax
            x_prev = probs.argmax(dim=1)
        else:
            # Intermediate step: sample from the predicted distribution,
            # but mix with the current x_t based on the noise schedule.
            # At high t: mostly use model prediction (lots of noise to remove)
            # At low t: mostly keep current state (refinement)

            # Sample from predicted x_0
            B, C, Y, Z, X = probs.shape
            probs_flat = probs.permute(0, 2, 3, 4, 1).reshape(-1, C)
            sampled_flat = torch.multinomial(probs_flat, 1).squeeze(1)
            x_0_pred = sampled_flat.reshape(B, Y, Z, X)

            # Re-noise to t-1 level (posterior)
            t_prev = t - 1
            if t_prev > 0:
                x_prev = self.q_sample(
                    x_0_pred,
                    torch.full((B,), t_prev, device=x_t.device, dtype=torch.long),
                )
            else:
                x_prev = x_0_pred

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        grid_int: torch.Tensor,
        state_vol: torch.Tensor,
        temperature: float = 1.0,
        callback=None,
    ) -> torch.Tensor:
        """Full reverse diffusion: generate a center chunk from pure noise.

        Args:
            model: The U-Net model.
            grid_int: (B, Y, 3Z, 3X) integer border context (center replaced each step).
            state_vol: (B, 1, Y, 3Z, 3X) state volume.
            temperature: Sampling temperature.
            callback: Optional function called at each step with (t, x_t).

        Returns:
            (B, CHUNK_Y, CHUNK_Z, CHUNK_X) generated center chunk.
        """
        from .palette import CHUNK_X, CHUNK_Y, CHUNK_Z

        B = grid_int.shape[0]
        device = grid_int.device

        # Start from pure noise
        x_t = torch.randint(
            0, self.num_classes, (B, CHUNK_Y, CHUNK_Z, CHUNK_X), device=device
        )

        for t in reversed(range(self.num_timesteps)):
            x_t = self.p_sample_step(model, x_t, t, grid_int, state_vol, temperature)
            if callback is not None:
                callback(t, x_t)

        return x_t
