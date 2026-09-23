"""Train OFormer Autoencoder on Burgers equation data (APEBench).

This script trains a mesh-free autoencoder to encode the incremental change
(delta = u_{t+1} - u_t) from the Burgers equation dataset into a latent space
and decode it back.

The autoencoder uses:
- SpatialEncoder2D: mesh-free coordinate-based attention encoder
- PointWiseDecoder2D: mesh-free cross-attention decoder

Training objective: MSE between original delta and reconstructed delta.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Make `python -m engine.train_oformer_autoencoder` work from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.base_model import get_model
from data import Parametric_2D_Dataset
from data.synthetic import canonical_to_native

# Import models to trigger registration
import models.OFormer  # noqa: F401


def load_yaml_config(config_path: Path, variant: str = "basic_config") -> dict:
    """Load a YAML config and resolve the specified variant (with YAML anchors)."""
    with open(config_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    if variant not in data:
        raise ValueError(f"Variant '{variant}' not found in {config_path}. Available: {list(data.keys())}")

    return data[variant]


def yaml_config_to_oformer_config(yaml_cfg: dict) -> dict:
    """Convert OFormer YAML config to framework OFormer adapter config."""
    return {
        "latent_dim": yaml_cfg.get("latent_dim", 256),
        "input_channels": yaml_cfg.get("input_channels", 2),
        "output_channels": yaml_cfg.get("output_channels", 2),
        "time_window": yaml_cfg.get("time_window", 10),
        "resolution": yaml_cfg.get("resolution", 128),
        "encoder_dim": yaml_cfg.get("encoder_dim", 512),
        "decoder_dim": yaml_cfg.get("decoder_dim", 256),
        "encoder_depth": yaml_cfg.get("encoder_depth", 4),
        "encoder_heads": yaml_cfg.get("encoder_heads", 8),
        "decoder_depth": yaml_cfg.get("decoder_depth", 4),
        "decoder_propagator_depth": yaml_cfg.get("decoder_propagator_depth", 4),
        "scale": yaml_cfg.get("scale", 16),
        "device": yaml_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    }


def _collate_fn_delta(batch, T_window: int = 10):
    """Collate function for delta encoding (u_{t+1} - u_t).

    Parametric_2D_Dataset returns (canonical_sample, para) where:
        canonical_sample: (Nx, Ny, T, C) — single sample, canonical layout (numpy)
        para: (1,) — PDE parameter (numpy)

    For delta encoding:
        - window: first T_window timesteps -> (T_window, B, C, Nx, Ny)
        - target: delta = u_{T_window} - u_{T_window-1} -> (B, C, Nx, Ny)
    """
    # Stack canonical samples: (B, Nx, Ny, T, C)
    canonical_batch = torch.from_numpy(np.stack([b[0] for b in batch])).float()  # (B, Nx, Ny, T, C)

    # Convert to native layout: canonical (B, Nx, Ny, T, C) -> native (T, B, C, Nx, Ny)
    full_seq = canonical_to_native(canonical_batch)  # (T, B, C, Nx, Ny)

    # Window: first T_window timesteps
    window = full_seq[:T_window].contiguous()  # (T_window, B, C, Nx, Ny)

    # Target: delta = u_{T_window} - u_{T_window-1}
    # u_{T_window} is at index T_window (0-indexed)
    # u_{T_window-1} is at index T_window - 1
    target = full_seq[T_window] - full_seq[T_window - 1]  # (B, C, Nx, Ny)

    return window, target


def main():
    p = argparse.ArgumentParser(description="Train OFormer Autoencoder on Burgers equation delta")
    p.add_argument("--config", type=Path, default=None, help="Path to YAML config file")
    p.add_argument("--variant", type=str, default="basic_config", help="YAML variant: basic_config, finetune, frozen, less_frozen")
    p.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    p.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    p.add_argument("--T-window", type=int, default=10, help="Time window length for training")
    p.add_argument("--T-out", type=int, default=50, help="Number of output timesteps from dataset")
    p.add_argument("--finish-sample", type=int, default=100, help="Max samples per file")
    p.add_argument("--data-file", type=int, choices=[0, 1], default=0, help="Which data file: 0=nu_0.1, 1=nu_0.01")
    p.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder (overrides config)")
    p.add_argument("--freeze-decoder", action="store_true", help="Freeze decoder (overrides config)")
    p.add_argument("--unfreeze", action="store_true", help="Unfreeze all (full training, overrides config)")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--weights", type=Path, default=None, help="Path to pretrained weights (optional)")
    p.add_argument("--output", type=Path, default=None, help="Path to save trained weights")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # --- Load YAML config if provided ---
    if args.config:
        print(f"Loading YAML config from {args.config} (variant: {args.variant})...")
        yaml_cfg = load_yaml_config(args.config, args.variant)
        of_config = yaml_config_to_oformer_config(yaml_cfg)
        # CLI overrides
        if args.freeze_encoder:
            of_config["freeze_encoder"] = True
        if args.freeze_decoder:
            of_config["freeze_decoder"] = True
        if args.unfreeze:
            of_config["freeze_encoder"] = False
            of_config["freeze_decoder"] = False
    else:
        # Default OFormer config (reduced for memory)
        of_config = {
            "latent_dim": 128,
            "input_channels": 2,
            "output_channels": 2,
            "time_window": args.T_window,
            "resolution": 128,
            "encoder_dim": 256,
            "decoder_dim": 128,
            "encoder_depth": 2,
            "encoder_heads": 4,
            "decoder_depth": 2,
            "decoder_propagator_depth": 2,
            "scale": 8,
            "device": args.device,
            "freeze_encoder": False,
            "freeze_decoder": False,
        }

    print("=" * 60)
    print("OFormer Autoencoder Training on Burgers Equation (Delta)")
    print("=" * 60)
    print(f"Config: {args.config or 'default OFormer'}")
    print(f"Variant: {args.variant}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"T-window: {args.T_window}")
    print(f"Model: latent_dim={of_config['latent_dim']}, encoder_dim={of_config['encoder_dim']}, "
          f"decoder_dim={of_config['decoder_dim']}, encoder_depth={of_config['encoder_depth']}, "
          f"decoder_depth={of_config['decoder_depth']}")
    print(f"Freezing: encoder={of_config.get('freeze_encoder', False)}, decoder={of_config.get('freeze_decoder', False)}")
    print()

    # --- 1. Build OFormer Autoencoder ---
    print("Building OFormer Autoencoder...")
    model = get_model("OFormer").build(of_config)
    print(f"  Total params: {model.num_parameters():,}")
    trainable = sum(p.numel() for p in model.native_model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    # --- 2. Load pretrained weights if provided ---
    weights_path = args.weights or of_config.get("pretrained_ckpt_path")
    if weights_path:
        print(f"\nLoading pretrained weights from {weights_path}...")
        model.load_weights(weights_path, strict=False)
        print("  Weights loaded (strict=False)")

    # --- 3. Configure freezing ---
    print("\nConfiguring parameter freezing...")
    if of_config.get("freeze_encoder"):
        print("  Freezing encoder...")
        for param in model._encoder.parameters():
            param.requires_grad = False
    if of_config.get("freeze_decoder"):
        print("  Freezing decoder...")
        for param in model._decoder.parameters():
            param.requires_grad = False

    if args.unfreeze:
        print("  Override: unfreezing all parameters...")
        for param in model.native_model.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.native_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.native_model.parameters())
    print(f"  Trainable params after config: {trainable:,} / {total:,}")

    # --- 4. Setup data loader ---
    data_files = [
        "/home/hamda/PhD_Thesis/dataset/Burgers_nu_0.1.hdf5",
        "/home/hamda/PhD_Thesis/dataset/Burgers_nu_0.01.hdf5",
    ]
    pde_params = [
        [0.1],  # nu
        [0.01],
    ]

    list_of_files = [data_files[args.data_file]]
    list_of_PDE_parameters = [pde_params[args.data_file]]
    file_name = "nu_0.1" if args.data_file == 0 else "nu_0.01"

    print(f"\nLoading dataset ({file_name})...")
    train_dataset = Parametric_2D_Dataset(
        list_files=list_of_files,
        list_para=list_of_PDE_parameters,
        T_out=args.T_out,
        finish_sample=args.finish_sample,
        normalize=True,
    )
    print(f"  Dataset size: {len(train_dataset)} samples")
    print(f"  Data shape per sample: {train_dataset.data.shape[1:]}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: _collate_fn_delta(batch, T_window=args.T_window),
        pin_memory=False,
    )

    # --- 5. Training loop ---
    device = torch.device(args.device)
    model._encoder.to(device).train()
    model._decoder.to(device).train()

    params = [p for p in model.native_model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()

    print("\nStarting training...")
    print("-" * 60)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (window, target_delta) in enumerate(train_loader):
            window = window.to(device, non_blocking=True)
            target_delta = target_delta.to(device, non_blocking=True)

            # Convert window to canonical sample format for the adapter
            # window is (T_window, B, C, Nx, Ny)
            T, B, C, Nx, Ny = window.shape
            canonical_fields = window.permute(1, 3, 4, 0, 2).cpu().numpy()  # (B, Nx, Ny, T, C)

            # Create canonical sample
            from core.base_model import CanonicalSample
            canonical_sample = CanonicalSample(fields=canonical_fields)

            # Forward pass through autoencoder
            native_input, coords = model.from_canonical(canonical_sample)
            native_input = native_input.to(device)
            coords = coords.to(device)

            # The target is delta in native format (B, C, Nx, Ny)
            # We need to convert it to our internal format (B, N, C)
            target_native = target_delta.reshape(B, C, Nx * Ny).permute(0, 2, 1)  # (B, N, C)

            # Encode
            latent = model._encoder(native_input, coords)
            # Decode
            reconstructed, _ = model._decoder(latent, coords)
            # reconstructed is (B, T*C, N) -> we want (B, N, C) for single step
            reconstructed = reconstructed.permute(0, 2, 1)  # (B, N, T*C)
            # Take the last timestep delta prediction
            reconstructed_delta = reconstructed[:, :, -C:]  # (B, N, C)

            # Loss: MSE between predicted delta and target delta
            loss = loss_fn(reconstructed_delta, target_native)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optim.step()

            epoch_loss += loss.item()
            n_batches += 1

            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{args.epochs} | Avg Loss: {avg_loss:.6f}")

    print("-" * 60)
    print("Training complete!")

    # --- 6. Save trained weights ---
    output_path = args.output or Path("/home/hamda/PhD_Thesis/weights/OFormer_autoencoder_burgers.pt")
    model.save_weights(output_path)
    print(f"\nSaved trained weights to {output_path}")


if __name__ == "__main__":
    main()