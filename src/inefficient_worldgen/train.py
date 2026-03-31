"""
Training script for the chunk diffusion model.

Usage:
    uv run python -m inefficient_worldgen.train \
        --chunk-dir ./extracted_chunks \
        --epochs 100 \
        --batch-size 2 \
        --lr 1e-4
"""

import argparse
import time
import warnings
from pathlib import Path
import re
from typing import cast

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import ChunkKernelDataset, make_state_volume
from .diffusion import CategoricalDiffusion
from .unet3d import ChunkUNet3D


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # Dataset
    dataset = ChunkKernelDataset(args.chunk_dir, preload=True)
    if len(dataset) == 0:
        raise RuntimeError(
            "No valid 3x3 kernels found! Need more chunks with neighbors."
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # preloaded data, no need for workers
        pin_memory=True,
        drop_last=True,
    )

    # Model
    model = ChunkUNet3D(
        base_channels=args.base_channels,
        channel_mults=(1, 2, 4, 8),
        time_dim=128,
        block_embed_dim=args.block_embed_dim,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Diffusion process
    diffusion = CategoricalDiffusion(
        num_timesteps=args.num_timesteps,
        noise_schedule="cosine",
        device=device,
    )

    # Optimizer
    adamw_kwargs = {"lr": args.lr, "weight_decay": 1e-4}
    if device.type == "cuda":
        adamw_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    except Exception:
        adamw_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)

    # Gradient accumulation — effective batch = batch_size * accum_steps
    accum_steps = args.grad_accum_steps

    # LR scheduler (steps counted per optimizer step, not per micro-batch).
    # The scheduler constructor internally calls step() once (to set the
    # initial LR), which triggers a spurious "scheduler.step() before
    # optimizer.step()" warning.  Suppress it here.
    steps_per_epoch = len(dataloader) // accum_steps
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*lr_scheduler.step.*optimizer.step.*"
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * steps_per_epoch, eta_min=args.lr * 0.01
        )

    # AMP (automatic mixed precision)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Checkpointing
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume if checkpoint exists
    start_epoch = 0
    latest_ckpt = ckpt_dir / "latest.pt"
    epoch_ckpt = None
    if args.resume:
        if latest_ckpt.exists():
            epoch_ckpt = latest_ckpt
        else:
            candidates = []
            for p in ckpt_dir.glob("epoch_*.pt"):
                m = re.match(r"epoch_(\d+)\.pt$", p.name)
                if m:
                    candidates.append((int(m.group(1)), p))
            if candidates:
                epoch_ckpt = max(candidates, key=lambda x: x[0])[1]

        if epoch_ckpt is not None:
            print(f"Resuming from checkpoint: {epoch_ckpt}")
            ckpt = torch.load(epoch_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if "scaler" in ckpt and use_amp:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed from epoch {start_epoch}")
        else:
            print("--resume was set, but no checkpoint was found. Starting fresh.")
    elif latest_ckpt.exists() or any(ckpt_dir.glob("epoch_*.pt")):
        print("Checkpoint(s) found, but --resume was not set. Starting fresh.")

    # Compile only for execution speed; keep uncompiled `model` for checkpoint I/O.
    train_model: torch.nn.Module = model
    if device.type == "cuda":
        try:
            train_model = cast(
                torch.nn.Module,
                torch.compile(model, mode="reduce-overhead"),
            )
            print("Enabled torch.compile (mode=reduce-overhead)")
        except Exception as e:
            print(f"torch.compile unavailable, continuing without it: {e}")

    # Training loop
    eff_batch = args.batch_size * accum_steps
    print(f"\nStarting training for {args.epochs} epochs")
    print(f"  Micro-batch size: {args.batch_size}")
    print(f"  Gradient accumulation steps: {accum_steps}")
    print(f"  Effective batch size: {eff_batch}")
    print(f"  Samples: {len(dataset)}")
    print(f"  Micro-batches/epoch: {len(dataloader)}")
    print(f"  Optimizer steps/epoch: {steps_per_epoch}")
    print(f"  Timesteps: {args.num_timesteps}")
    print(f"  AMP (mixed precision): {use_amp}")
    print()

    for epoch in range(start_epoch, args.epochs):
        train_model.train()
        epoch_loss = 0.0
        num_batches = 0
        t_start = time.time()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            grid = batch["grid"].to(device, non_blocking=True)  # (B, Y, 3Z, 3X) uint8
            target = batch["target"].to(device, non_blocking=True)  # (B, Y, Z, X) long
            state_map = batch["state_map"].to(device, non_blocking=True)  # (B, 3, 3)

            # Build state volume
            state_vol = make_state_volume(state_map)

            # Forward pass under AMP autocast
            with torch.amp.autocast("cuda", enabled=use_amp):
                # Compute loss (passes integer grid; one-hot built inside with
                # the noised center already spliced in, avoiding a full clone)
                loss = diffusion.training_loss(train_model, target, grid, state_vol)
                # Scale loss by accumulation steps so gradients average correctly
                loss_scaled = loss / accum_steps

            # Backward (through scaler for AMP)
            scaler.scale(loss_scaled).backward()

            # Step optimizer every accum_steps micro-batches
            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}"
            )

        # Handle any leftover micro-batches at end of epoch
        if num_batches % accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - t_start
        print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}, time={elapsed:.1f}s")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "loss": avg_loss,
                "args": vars(args),
            }
            torch.save(ckpt, ckpt_dir / "latest.pt")
            torch.save(ckpt, ckpt_dir / f"epoch_{epoch + 1:04d}.pt")
            print(f"  Saved checkpoint (epoch {epoch + 1})")


def main():
    parser = argparse.ArgumentParser(description="Train chunk diffusion model")
    parser.add_argument(
        "--chunk-dir",
        type=str,
        required=True,
        help="Directory with extracted chunk .npy files",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint output directory",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--num-timesteps", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--block-embed-dim",
        type=int,
        default=16,
        help="Embedding channels for integer block ids",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch_size * this)",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
