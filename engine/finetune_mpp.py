"""Finetune MPP on Burgers equation data.

This script:
1. Builds MPP with the config from YAML (supports AViT-Ti, AViT-S, etc.)
2. Loads pretrained weights from /home/hamda/PhD_Thesis/weights/MPP_AViT_S
3. Sets up data loader for Burgers equation (Parametric_2D_Dataset)
4. Configures finetuning: freeze middle, unfreeze projections
5. Runs native training loop

Usage:
    python3 -m engine.finetune_mpp --config configs/models/mpp_avit_s_config.yaml --variant frozen --epochs 10 --lr 1e-4 --batch-size 1 --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Make `python -m engine.finetune_mpp` work from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.base_model import get_model
from data import Parametric_2D_Dataset
from data.synthetic import canonical_to_native

# Import models to trigger registration
import models.MPP  # noqa: F401


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


def yaml_config_to_mpp_config(yaml_cfg: dict) -> dict:
    """Convert MPP YAML config to framework MPP adapter config.

    Maps YAML keys to adapter config keys.
    """
    return {
        "patch_size": tuple(yaml_cfg.get("patch_size", [16, 16])),
        "embed_dim": yaml_cfg.get("embed_dim", 384),
        "processor_blocks": yaml_cfg.get("processor_blocks", 12),
        "n_states": yaml_cfg.get("n_states", 12),
        "num_heads": yaml_cfg.get("num_heads", 6),
        "space_type": yaml_cfg.get("space_type", "axial_attention"),
        "time_type": yaml_cfg.get("time_type", "attention"),
        "block_type": yaml_cfg.get("block_type", "axial"),
        "bias_type": yaml_cfg.get("bias_type", "rel"),
        "gradient_checkpointing": yaml_cfg.get("gradient_checkpointing", False),
        "drop_path": yaml_cfg.get("drop_path", 0.1),
        # Pretrained weights path from YAML
        "pretrained_ckpt_path": yaml_cfg.get("pretrained_ckpt_path", None),
        # Finetuning flags
        "freeze_middle": yaml_cfg.get("freeze_middle", False),
        "freeze_processor": yaml_cfg.get("freeze_processor", False),
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


def _adapt_to_n_channels(native_model, n_channels: int):
    """Adapt MPP model from 12 channels to n_channels by slicing projections.

    The pretrained model has:
    - space_bag: SubsampledLinear(12, 96) -> weight [96, 12], bias [96]
    - debed.out_kernel: ConvTranspose2d(96, 12, 4, 4) -> weight [96, 12, 4, 4], bias [12]

    We slice to keep only the first n_channels.
    """
    import torch.nn as nn

    # --- Adapt space_bag (input projection) ---
    old_space_bag = native_model.space_bag
    new_space_bag = type(old_space_bag)(
        dim_in=n_channels,
        dim_out=old_space_bag.dim_out,
        subsample_in=old_space_bag.subsample_in,
    )
    # Copy weights for first n_channels
    with torch.no_grad():
        new_space_bag.weight[:, :n_channels] = old_space_bag.weight[:, :n_channels]
        new_space_bag.bias[:] = old_space_bag.bias[:]
    native_model.space_bag = new_space_bag

    # --- Adapt debed (output projection) ---
    old_debed = native_model.debed
    # debed.out_kernel is a ConvTranspose2d parameter
    # debed.out_bias is a bias parameter
    new_out_kernel = nn.Parameter(old_debed.out_kernel[:, :n_channels, :, :].clone())
    new_out_bias = nn.Parameter(old_debed.out_bias[:n_channels].clone())
    native_model.debed.out_kernel = new_out_kernel
    native_model.debed.out_bias = new_out_bias
    native_model.debed.out_chans = n_channels


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

    # Convert to native MPP layout
    # canonical (B, Nx, Ny, T, C) -> native (T, B, C, Nx, Ny)
    full_seq = canonical_to_native(canonical_batch)  # (T, B, C, Nx, Ny)

    # Window: first T_window timesteps
    window = full_seq[:T_window].contiguous()  # (T_window, B, C, Nx, Ny)

    # Target: next timestep (T_window index)
    target = full_seq[T_window].contiguous()  # (B, C, Nx, Ny)

    return window, target


def main():
    p = argparse.ArgumentParser(description="Finetune MPP on Burgers equation")
    p.add_argument("--config", type=Path, default=None, help="Path to YAML config file (e.g., configs/models/mpp_avit_s_config.yaml)")
    p.add_argument("--variant", type=str, default="basic_config", help="YAML variant to use: basic_config, finetune, frozen, less_frozen")
    p.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    p.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    p.add_argument("--T-out", type=int, default=50, help="Number of output timesteps")
    p.add_argument("--T-window", type=int, default=10, help="Time window length for training (should match pretrained n_steps)")
    p.add_argument("--finish-sample", type=int, default=100, help="Max samples per file (default small for memory)")
    p.add_argument("--data-file", type=int, choices=[0, 1], default=0, help="Which data file to use: 0=nu_0.1, 1=nu_0.01")
    p.add_argument("--freeze-processor", action="store_true", help="Freeze processor blocks (overrides config)")
    p.add_argument("--unfreeze", action="store_true", help="Unfreeze all (full finetune, overrides config)")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 for memory efficiency)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--weights", type=Path, default=None, help="Path to pretrained weights (default: /home/hamda/PhD_Thesis/weights/MPP_AViT_S)")
    p.add_argument("--output", type=Path, default=None, help="Path to save finetuned weights")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # --- Load YAML config if provided ---
    if args.config:
        print(f"Loading YAML config from {args.config} (variant: {args.variant})...")
        yaml_cfg = load_yaml_config(args.config, args.variant)
        mpp_config = yaml_config_to_mpp_config(yaml_cfg)
        # CLI overrides
        mpp_config["gradient_checkpointing"] = True  # Always enable for memory
        if args.freeze_processor:
            mpp_config["freeze_processor"] = True
        if args.unfreeze:
            mpp_config["freeze_middle"] = False
            mpp_config["freeze_processor"] = False
    else:
        # Default AViT-S config (matches pretrained weights)
        mpp_config = {
            "patch_size": (16, 16),
            "embed_dim": 384,
            "processor_blocks": 12,
            "n_states": 12,  # Match pretrained weights
            "num_heads": 6,
            "space_type": "axial_attention",
            "time_type": "attention",
            "block_type": "axial",
            "bias_type": "rel",
            "gradient_checkpointing": True,
            "freeze_middle": False,
            "freeze_processor": args.freeze_processor,
            "unfreeze": args.unfreeze,
        }

    print("=" * 60)
    print("MPP Finetuning on Burgers Equation")
    print("=" * 60)
    print(f"Config: {args.config or 'default AViT-S'}")
    print(f"Variant: {args.variant}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"T-window: {args.T_window}")
    print(f"Model: embed_dim={mpp_config['embed_dim']}, blocks={mpp_config['processor_blocks']}, heads={mpp_config['num_heads']}, n_states={mpp_config['n_states']}")
    print(f"Finetuning: freeze_middle={mpp_config.get('freeze_middle', False)}, freeze_processor={mpp_config.get('freeze_processor', False)}")
    print()

    # --- 1. Build MPP ---
    print("Building MPP model...")
    model = get_model("MPP").build(mpp_config)
    print(f"  Total params: {model.num_parameters():,}")
    trainable = sum(p.numel() for p in model.native_model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    # --- 2. Load pretrained weights ---
    # Priority: CLI --weights > YAML pretrained_ckpt_path > default AViT-S path
    weights_path = args.weights or mpp_config.get("pretrained_ckpt_path") or Path("/home/hamda/PhD_Thesis/weights/MPP_AViT_S")
    print(f"\nLoading pretrained weights from {weights_path}...")
    model.load_weights(weights_path, strict=True)
    print("  Weights loaded (strict=True)")

    # --- 3. Adapt model to 2 channels (Burgers) ---
    print("\nAdapting model to 2 channels (Burgers)...")
    _adapt_to_n_channels(model.native_model, n_channels=2)
    print("  Model adapted to 2 channels")

    # --- 4. Set inference context (state_labels, bcs) ---
    state_labels = [0, 1]
    bcs = torch.zeros(1, 2, dtype=torch.long)
    model.set_inference_context(state_labels=state_labels, bcs=bcs)
    print(f"  state_labels: {state_labels}")
    print(f"  bcs shape: {bcs.shape}")

    # Print trainable params after adaptation
    trainable = sum(p.numel() for p in model.native_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.native_model.parameters())
    print(f"  Trainable params after adaptation: {trainable:,} / {total:,}")

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

    # --- 6. Configure finetuning ---
    print("\nConfiguring finetuning...")
    if mpp_config.get("freeze_middle"):
        print("  Freezing middle layers (space_bag + debed trainable)...")
        model.native_model.freeze_middle()
    elif mpp_config.get("freeze_processor"):
        print("  Freezing processor blocks (keep embed/debed trainable)...")
        model.native_model.freeze_processor()
    else:
        print("  Default: freeze_middle (only projections trainable)...")
        model.native_model.freeze_middle()

    if args.unfreeze:
        print("  Override: unfreezing all parameters...")
        model.native_model.unfreeze()

    trainable = sum(p.numel() for p in model.native_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.native_model.parameters())
    print(f"  Trainable params after config: {trainable:,} / {total:,}")

    # --- 7. Training loop (native) ---
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

    # --- 8. Save finetuned weights ---
    output_path = args.output or Path("/home/hamda/PhD_Thesis/weights/MPP_AViT_S_finetuned_burgers.pt")
    model.save_weights(output_path)
    print(f"\nSaved finetuned weights to {output_path}")


if __name__ == "__main__":
    main()
