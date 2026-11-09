"""Finetune FactFormer on Burgers equation data.

This script:
1. Builds FactFormer with the config from YAML
2. Loads pretrained weights from /home/hamda/PhD_Thesis/weights/FactFormer_checkpoint.pth
3. Sets up data loader for Burgers equation (Parametric_2D_Dataset)
4. Configures finetuning: freeze encoder/propagator
5. Runs native training loop

The number of state channels C is set with --n-channels (default 2). The
FABlock2D is built from scratch with dim = dim_out = C, so the same script
handles Burgers (C=2, u/v) and 3-channel cases such as NS (C=3, u/v/p).

Usage:
    python3 -m engine.finetune_factformer --config configs/models/factformer_config.yaml --variant frozen --epochs 10 --lr 1e-4 --batch-size 1 --device cuda
    python3 -m engine.finetune_factformer --config configs/models/factformer_config.yaml --variant frozen --n-channels 3 --epochs 10 --lr 1e-4 --batch-size 1 --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Make `python -m engine.finetune_factformer` work from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.base_model import get_model
from data import Parametric_2D_Dataset
from data.synthetic import canonical_to_native

# Import models to trigger registration
import models.FactFormer  # noqa: F401


def load_yaml_config(config_path: Path, variant: str = "basic_config") -> dict:
    """Load a YAML config and resolve the specified variant (with YAML anchors).

    Args:
        config_path: Path to YAML file
        variant: Which anchor to use ('basic_config', 'finetune', 'frozen', 'less_frozen')

    Returns:
        Resolved config dictionary
    """
    with open(config_path, "r") as f:
        # Use yaml.SafeLoader which supports anchors/aliases
        data = yaml.load(f, Loader=yaml.FullLoader)

    if variant not in data:
        raise ValueError(f"Variant '{variant}' not found in {config_path}. Available: {list(data.keys())}")

    return data[variant]


def yaml_config_to_factformer_config(yaml_cfg: dict) -> dict:
    """Convert FactFormer YAML config to framework FactFormer adapter config.

    Maps YAML keys to adapter config keys.
    """
    return {
        "dim": yaml_cfg.get("dim", 128),
        "dim_head": yaml_cfg.get("dim_head", 64),
        "latent_dim": yaml_cfg.get("latent_dim", 256),
        "heads": yaml_cfg.get("heads", 8),
        "dim_out": yaml_cfg.get("dim_out", 128),
        "use_rope": yaml_cfg.get("use_rope", True),
        "kernel_multiplier": yaml_cfg.get("kernel_multiplier", 2),
        "scaling_factor": yaml_cfg.get("scaling_factor", 1.0),
        # Pretrained weights path from YAML
        "pretrained_ckpt_path": yaml_cfg.get("pretrained_ckpt_path", None),
        # Finetuning flags
        "freeze_encoder": yaml_cfg.get("freeze_encoder", False),
        "freeze_propagator": yaml_cfg.get("freeze_propagator", False),
    }


def nrmse(pred, target, eps=1e-8):
    """
    Normalized Mean Absolute Error (nMAE)

    Args:
        pred (torch.Tensor): predicted values
        target (torch.Tensor): ground truth values
        eps (float): small value to avoid division by zero

    Returns:
        torch.Tensor: scalar nMAE
    """
    mse = torch.mean(torch.abs(pred - target))**2
    norm = torch.mean(torch.abs(target))**2
    return torch.sqrt(mse / (norm + eps))


def _collate_fn(batch, T_window: int = 10):
    """Collate function for DataLoader to produce (window, target) in native layout.

    Parametric_2D_Dataset returns (canonical_sample, para) where:
        canonical_sample: (Nx, Ny, T, C) — single sample, canonical layout (numpy)
        para: (1,) — PDE parameter (numpy)

    For autoregressive training:
        - window: first T_window timesteps -> (T_window, B, C, Nx, Ny)
        - target: next timestep (T_window) -> (B, C, Nx, Ny)
    """
    # Stack canonical samples: (B, Nx, Ny, T, C)
    canonical_batch = torch.from_numpy(np.stack([b[0] for b in batch])).float()  # (B, Nx, Ny, T, C)

    # Convert to native layout
    # canonical (B, Nx, Ny, T, C) -> native (T, B, C, Nx, Ny)
    full_seq = canonical_to_native(canonical_batch)  # (T, B, C, Nx, Ny)

    # Window: first T_window timesteps
    window = full_seq[:T_window].contiguous()  # (T_window, B, C, Nx, Ny)

    # Target: next timestep (T_window index)
    target = full_seq[T_window].contiguous()  # (B, C, Nx, Ny)

    return window, target


def main():
    p = argparse.ArgumentParser(description="Finetune FactFormer on Burgers equation")
    p.add_argument("--config", type=Path, default=None, help="Path to YAML config file (e.g., configs/models/factformer_config.yaml)")
    p.add_argument("--variant", type=str, default="basic_config", help="YAML variant to use: basic_config, finetune, frozen, less_frozen")
    p.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    p.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    p.add_argument("--T-out", type=int, default=16, help="Number of output timesteps")
    p.add_argument("--T-window", type=int, default=10, help="Time window length for training (should match pretrained in_time_window)")
    p.add_argument("--finish-sample", type=int, default=10, help="Max samples per file (default small for memory)")
    p.add_argument("--data-file", type=int, choices=[0, 1], default=0, help="Which data file to use: 0=nu_0.1, 1=nu_0.01")
    p.add_argument("--n-channels", type=int, default=2, help="Number of physical state channels C in the data (2 = Burgers (u,v), 3 = NS (u,v,p)). Sets FABlock2D dim/dim_out.")
    p.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder (overrides config)")
    p.add_argument("--freeze-propagator", action="store_true", help="Freeze propagator (overrides config)")
    p.add_argument("--unfreeze", action="store_true", help="Unfreeze all (full finetune, overrides config)")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 for memory efficiency)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--weights", type=Path, default=None, help="Path to pretrained weights (default: from YAML or /home/hamda/PhD_Thesis/weights/FactFormer_checkpoint.pth)")
    p.add_argument("--output", type=Path, default=None, help="Path to save finetuned weights")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # --- Load YAML config if provided ---
    if args.config:
        print(f"Loading YAML config from {args.config} (variant: {args.variant})...")
        yaml_cfg = load_yaml_config(args.config, args.variant)
        ff_config = yaml_config_to_factformer_config(yaml_cfg)
        # CLI overrides
        ff_config["gradient_checkpointing"] = True  # Always enable for memory
        if args.freeze_encoder:
            ff_config["freeze_encoder"] = True
        if args.freeze_propagator:
            ff_config["freeze_propagator"] = True
        if args.unfreeze:
            ff_config["freeze_encoder"] = False
            ff_config["freeze_propagator"] = False
    else:
        # Default FactFormer config
        ff_config = {
            "dim": 128,
            "dim_head": 64,
            "latent_dim": 256,
            "heads": 8,
            "dim_out": 128,
            "use_rope": True,
            "kernel_multiplier": 2,
            "scaling_factor": 1.0,
            "gradient_checkpointing": True,
            "freeze_encoder": False,
            "freeze_propagator": False,
            "pretrained_ckpt_path": None,
        }

    # FABlock2D's `dim` is the input channel count and `dim_out` the output
    # channel count, so both must equal the data's C.
    ff_config["dim"] = args.n_channels
    ff_config["dim_out"] = args.n_channels  # Predict the same C channels

    print("=" * 60)
    print(f"FactFormer Finetuning ({args.n_channels} channels)")
    print("=" * 60)
    print(f"Config: {args.config or 'default FactFormer'}")
    print(f"Variant: {args.variant}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"T-window: {args.T_window}")
    print(f"Channels (C): {args.n_channels}")
    print(f"Model: dim={ff_config['dim']}, heads={ff_config['heads']}, dim_head={ff_config['dim_head']}, latent_dim={ff_config['latent_dim']}, dim_out={ff_config['dim_out']}")
    print(f"Finetuning: freeze_encoder={ff_config.get('freeze_encoder', False)}, freeze_propagator={ff_config.get('freeze_propagator', False)}")
    print()

    # --- 1. Build FactFormer ---
    print("Building FactFormer model...")
    # Add canonical shape overrides for smoke test compatibility
    ff_config["_T"] = args.T_window
    ff_config["_C"] = args.n_channels
    ff_config["_Nx"] = 128
    ff_config["_Ny"] = 128

    model = get_model("FactFormer").build(ff_config)
    print(f"  Total params: {model.num_parameters():,}")
    trainable = sum(p.numel() for p in model.native_model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    # --- 2. Load pretrained weights ---
    # Priority: CLI --weights > YAML pretrained_ckpt_path
    # Note: The FactFormer checkpoint is for the FULL model (encoder+propagator+decoder).
    # Our adapter only wraps a single FABlock2D (one encoder layer).
    # These are NOT directly compatible. Training from scratch is the default.
    weights_path = args.weights or ff_config.get("pretrained_ckpt_path")
    if weights_path:
        print(f"\nLoading pretrained weights from {weights_path}...")
        print("  WARNING: Full model checkpoint not directly compatible with FABlock2D adapter.")
        print("  Attempting partial load (encoder layers only)...")
        model.load_weights(weights_path, strict=False)
        print("  Weights loaded (strict=False)")
    else:
        print("\nNo pretrained weights specified - training from scratch.")

    # --- 3. Channels ---
    # No adaptation needed: the block is built from scratch with
    # dim = dim_out = n_channels, so its projections already match C.
    # (The old `_adapt_factformer_to_n_channels` helper only rewired the
    # block-level `to_in`, never the `PoolingReducer.to_in` layers inside
    # `to_x`/`to_y`, and produced a Linear(n_channels, dim) that broke C != 2.)
    print(f"\nModel built from scratch with C={args.n_channels} (no channel adaptation)")

    # --- 4. Configure finetuning ---
    print("\nConfiguring finetuning...")
    if ff_config.get("freeze_encoder"):
        print("  Freezing encoder layers...")
        # Note: FactFormerAdapter wraps FABlock2D which is a single encoder layer
        # The full encoder would have multiple FABlock2D layers
        for param in model.native_model.parameters():
            param.requires_grad = False
        # Keep output projection trainable
        if hasattr(model.native_model, 'to_out'):
            for param in model.native_model.to_out.parameters():
                param.requires_grad = True
    elif ff_config.get("freeze_propagator"):
        print("  Freezing propagator (not applicable to FABlock2D adapter)...")
    else:
        print("  Default: freezing encoder (only output projection trainable)...")
        for param in model.native_model.parameters():
            param.requires_grad = False
        if hasattr(model.native_model, 'to_out'):
            for param in model.native_model.to_out.parameters():
                param.requires_grad = True

    if args.unfreeze:
        print("  Override: unfreezing all parameters...")
        for param in model.native_model.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.native_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.native_model.parameters())
    print(f"  Trainable params after config: {trainable:,} / {total:,}")

    # --- 5. Setup data loader ---
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
        collate_fn=lambda batch: _collate_fn(batch, T_window=args.T_window),
        pin_memory=False,
    )

    # --- 6. Training loop (native) ---
    device = torch.device(args.device)
    model.native_model.to(device).train()

    optim = torch.optim.AdamW(
        [p for p in model.native_model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nrmse

    print("\nStarting training...")
    print("-" * 60)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (window, target) in enumerate(train_loader):
            window = window.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            pred = model.forward(window)
            loss = loss_fn(pred, target)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.native_model.parameters(), max_norm=1.0)
            optim.step()

            epoch_loss += loss.item()
            n_batches += 1

            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{args.epochs} | Avg Loss: {avg_loss:.6f}")

    print("-" * 60)
    print("Training complete!")

    # --- 7. Save finetuned weights ---
    output_path = args.output or Path(f"/home/hamda/PhD_Thesis/weights/FactFormer_finetuned_C{args.n_channels}.pt")
    model.save_weights(output_path)
    print(f"\nSaved finetuned weights to {output_path}")


if __name__ == "__main__":
    main()