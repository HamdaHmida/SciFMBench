"""OFormer adapter for mesh-free autoencoding.

This adapter wraps OFormer's coordinate-based encoders and decoders to create
a mesh-free autoencoder that can:
- Encode velocity fields from canonical grid [B, Nx, Ny, T, C] to latent [B, M, d]
- Decode latent representations back to arbitrary grids/resolutions

Uses:
- SpatialEncoder2D (mesh-free, coordinate-based attention) for encoding
- PointWiseDecoder2D (mesh-free, cross-attention decoder) for decoding

Both operate on point clouds with explicit coordinates, enabling resolution-invariant
encoding/decoding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from core.base_model import BaseModel, CanonicalSample, ModelMetadata, Mode, register, TrainingResult

# Import vendored OFormer modules
from models.OFormer.encoder_module import SpatialEncoder2D
from models.OFormer.decoder_module import PointWiseDecoder2D
from models.OFormer.attention_module import CrossLinearAttention


@register
class OFormerAdapter(BaseModel):
    """Mesh-free autoencoder adapter using OFormer components."""

    metadata = ModelMetadata(
        name="OFormer",
        family="neural_operator",
        supported_modes=[Mode.FROM_SCRATCH, Mode.FINETUNE, Mode.ZERO_SHOT],
        default_mode=Mode.FROM_SCRATCH,
        native_input_layout="(B, N, T*C) with coordinates (B, N, 2)",
        native_output_layout="(B, N, T*C) with coordinates (B, N, 2)",
        canonical_layout="(B, Nx, Ny, T, C) float32",
        paper="OFormer: On Learning Mesh-Free Operators",
    )

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        input_channels: int,
        output_channels: int,
        time_window: int,
        resolution: int,
        n_patches: int = 16,
        device: str = "cuda",
    ):
        self._encoder = encoder
        self._decoder = decoder
        self._latent_dim = latent_dim
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._time_window = time_window
        self._resolution = resolution
        self._n_patches = n_patches
        self._device = torch.device(device)
        self._grid_coords: Optional[Tensor] = None  # cached canonical grid coords

        # Move to device
        self._encoder.to(self._device)
        self._decoder.to(self._device)

    @property
    def native_model(self) -> nn.Module:
        """Return the encoder-decoder pair as a single module for inspection."""
        return nn.ModuleDict({"encoder": self._encoder, "decoder": self._decoder})

    @classmethod
    def build(cls, cfg: Dict[str, Any]) -> "OFormerAdapter":
        """Build OFormer autoencoder from config dict.

        Expected config keys:
            - latent_dim: latent dimension (default: 256)
            - input_channels: input channels (default: 2 for Burgers)
            - output_channels: output channels (default: 2)
            - time_window: number of timesteps (default: 10)
            - resolution: spatial resolution (default: 128)
            - encoder_dim: encoder embedding dim (default: 512)
            - decoder_dim: decoder latent channels (default: 256)
            - encoder_depth: encoder transformer depth (default: 4)
            - encoder_heads: encoder attention heads (default: 8)
            - decoder_depth: decoder depth (default: 4)
            - decoder_propagator_depth: decoder propagator depth (default: 4)
            - scale: attention scale (default: 16)
            - device: device (default: cuda if available)
        """
        # Extract config with defaults
        latent_dim = cfg.get("latent_dim", 256)
        input_channels = cfg.get("input_channels", 2)
        output_channels = cfg.get("output_channels", 2)
        time_window = cfg.get("time_window", 10)
        resolution = cfg.get("resolution", 128)
        encoder_dim = cfg.get("encoder_dim", 512)
        decoder_dim = cfg.get("decoder_dim", 256)
        encoder_depth = cfg.get("encoder_depth", 4)
        encoder_heads = cfg.get("encoder_heads", 8)
        decoder_depth = cfg.get("decoder_depth", 4)
        decoder_propagator_depth = cfg.get("decoder_propagator_depth", 4)
        scale = cfg.get("scale", 16)
        device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Build encoder: SpatialEncoder2D
        # Input: [B, N, C_in] where C_in = T * input_channels
        # Output: [B, N, latent_dim]
        encoder = SpatialEncoder2D(
            input_channels=time_window * input_channels,
            in_emb_dim=encoder_dim,
            out_seq_emb_dim=latent_dim,
            heads=encoder_heads,
            depth=encoder_depth,
            res=resolution,
            use_ln=True,
            emb_dropout=0.05,
        )

        # Build decoder: PointWiseDecoder2D
        # Input: latent [B, N, latent_dim] + coords [B, N, 2]
        # Output: [B, N, T * output_channels]
        decoder = PointWiseDecoder2D(
            latent_channels=latent_dim,
            out_channels=output_channels,
            out_steps=time_window,
            propagator_depth=decoder_propagator_depth,
            scale=scale,
            dropout=0.0,
        )

        return cls(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            input_channels=input_channels,
            output_channels=output_channels,
            time_window=time_window,
            resolution=resolution,
            n_patches=cfg.get("n_patches", 16),
            device=device,
        )

    def _get_canonical_coords(self, Nx: int, Ny: int) -> Tensor:
        """Generate canonical grid coordinates for (Nx, Ny) grid."""
        if self._grid_coords is not None:
            cached_Nx = int(self._grid_coords.shape[1] ** 0.5)
            if cached_Nx == Nx:
                return self._grid_coords

        x = torch.linspace(0, 1, Nx, device=self._device)
        y = torch.linspace(0, 1, Ny, device=self._device)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).reshape(1, Nx * Ny, 2)  # [1, N, 2]
        self._grid_coords = coords
        return coords

    def _canonical_to_native(self, canonical: CanonicalSample) -> Tuple[Tensor, Tensor]:
        """Convert canonical (B, Nx, Ny, T, C) to native (B, N, T*C) + coords (B, N, 2)."""
        fields = canonical.fields  # (B, Nx, Ny, T, C)
        # Ensure fields is a tensor
        if not isinstance(fields, Tensor):
            fields = torch.from_numpy(fields).float().to(self._device)
        B, Nx, Ny, T, C = fields.shape

        # Reshape to (B, Nx*Ny, T*C)
        native_input = fields.permute(0, 1, 2, 4, 3).reshape(B, Nx * Ny, T * C)

        # Get coordinates
        coords = self._get_canonical_coords(Nx, Ny).expand(B, -1, -1)  # (B, N, 2)

        return native_input, coords

    def _native_to_canonical(
        self, native_output: Tensor, coords: Tensor, Nx: int, Ny: int, T: int, C: int
    ) -> CanonicalSample:
        """Convert native (B, N, T*C) back to canonical (B, Nx, Ny, T, C)."""
        B = native_output.shape[0]
        # Reshape from (B, Nx*Ny, T*C) to (B, Nx, Ny, T, C)
        canonical_fields = native_output.reshape(B, Nx, Ny, T, C)
        return CanonicalSample(fields=canonical_fields, coords=coords[0])

    def forward(self, x: Union[Tensor, Tuple[Tensor, Tensor]]) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass through autoencoder.

        Args:
            x: Either:
               - A tuple of (input_tensor, coords)
                 - input_tensor: (B, N, T*C) - flattened velocity field (all timesteps concatenated)
                 - coords: (B, N, 2) - spatial coordinates
               - Or a single tensor in native layout (T, B, C, Nx, Ny) from smoke test

        Returns:
            For smoke test (single tensor input): just the reconstructed tensor (B, C, Nx, Ny)
            For normal use (tuple input): tuple of (reconstructed, latent)
               - reconstructed: (B, N, T*C)
               - latent: (B, N, latent_dim)
        """
        # Handle smoke test input: single tensor in native layout (T, B, C, Nx, Ny)
        if isinstance(x, Tensor) and x.dim() == 5:
            # Convert from native (T, B, C, Nx, Ny) to our internal format
            T, B, C, Nx, Ny = x.shape
            # Use ALL timesteps concatenated as input: (B, N, T*C)
            input_tensor = x.permute(1, 0, 2, 3, 4).reshape(B, Nx * Ny, T * C)
            coords = self._get_canonical_coords(Nx, Ny).expand(B, -1, -1)  # (B, N, 2)

            latent = self._encoder(input_tensor, coords)  # (B, N, latent_dim)
            reconstructed, _ = self._decoder(latent, coords)  # (B, T*C, N)
            reconstructed = reconstructed.permute(0, 2, 1)  # (B, N, T*C)

            # Return in format expected by smoke test: (B, C, Nx, Ny)
            # Take the last predicted timestep
            recon_last = reconstructed[:, :, -C:].reshape(B, Nx, Ny, C).permute(0, 3, 1, 2)
            return recon_last

        # Normal tuple input (from our own training/eval)
        input_tensor, coords = x
        B, N, _ = input_tensor.shape

        # Encode
        latent = self._encoder(input_tensor, coords)  # (B, N, latent_dim)

        # Decode
        # PointWiseDecoder2D expects: z [B, N, C], propagate_pos [B, N, 2]
        reconstructed, _ = self._decoder(latent, coords)

        # Reshape decoder output: [B, (T*C), N] -> [B, N, T*C]
        reconstructed = reconstructed.permute(0, 2, 1)  # (B, N, T*C)

        return reconstructed, latent

    def encode(self, canonical: CanonicalSample) -> Tuple[Tensor, Tensor]:
        """Encode canonical sample to latent.

        Returns:
            (latent, coords) where latent is (B, N, latent_dim)
        """
        native_input, coords = self._canonical_to_native(canonical)
        latent = self._encoder(native_input, coords)
        return latent, coords

    def decode(self, latent: Tensor, coords: Tensor, Nx: int, Ny: int) -> CanonicalSample:
        """Decode latent to canonical sample at target resolution."""
        B = latent.shape[0]
        T = self._time_window
        C = self._output_channels

        # Decode
        reconstructed, _ = self._decoder(latent, coords)
        reconstructed = reconstructed.permute(0, 2, 1)  # (B, N, T*C)

        # Convert to canonical
        return self._native_to_canonical(reconstructed, coords, Nx, Ny, T, C)

    def to_canonical(self, y_native: Union[Tensor, Tuple[Tensor, Tensor]]) -> CanonicalSample:
        """Convert native output to canonical form."""
        # Handle both tuple (reconstructed, latent) and single tensor (reconstructed)
        if isinstance(y_native, tuple):
            reconstructed = y_native[0]
        else:
            reconstructed = y_native

        # Shape could be (B, N, T*C) for autoencoder internal format or (B, C, Nx, Ny) for smoke test
        if reconstructed.dim() == 4:
            # Smoke test format: (B, C, Nx, Ny) - already in canonical-like shape
            B, C, Nx, Ny = reconstructed.shape
            # Just reshape to canonical (B, Nx, Ny, T, C) with T=1
            canonical_fields = reconstructed.permute(0, 2, 3, 1).unsqueeze(3)  # (B, Nx, Ny, 1, C)
            coords = self._get_canonical_coords(Nx, Ny).expand(B, -1, -1)
            return CanonicalSample(fields=canonical_fields, coords=coords[0])
        else:
            # Internal format: (B, N, T*C)
            B, N, _ = reconstructed.shape
            Nx = Ny = int(N**0.5)
            coords = self._get_canonical_coords(Nx, Ny).expand(B, -1, -1)
            return self._native_to_canonical(reconstructed, coords, Nx, Ny, self._time_window, self._output_channels)

    def from_canonical(self, sample: CanonicalSample) -> Tuple[Tensor, Tensor]:
        """Convert canonical sample to native input format."""
        return self._canonical_to_native(sample)

    def load_weights(self, source: str, *, strict: bool = True) -> None:
        """Load weights from checkpoint."""
        state_dict = torch.load(source, map_location=self._device, weights_only=False)
        if "encoder" in state_dict and "decoder" in state_dict:
            self._encoder.load_state_dict(state_dict["encoder"], strict=strict)
            self._decoder.load_state_dict(state_dict["decoder"], strict=strict)
        else:
            # Try loading as single state dict
            self._encoder.load_state_dict(state_dict, strict=strict)
            # Decoder might not be in checkpoint
            if hasattr(self, "_decoder_state_dict"):
                self._decoder.load_state_dict(self._decoder_state_dict, strict=strict)

    def save_weights(self, destination: str) -> None:
        """Save weights to checkpoint."""
        torch.save(
            {"encoder": self._encoder.state_dict(), "decoder": self._decoder.state_dict()},
            destination,
        )

    def train(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        *,
        cfg: Dict[str, Any],
        callbacks: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Train the autoencoder using native training loop."""
        from torch.utils.data import DataLoader

        epochs = cfg.get("epochs", 100)
        lr = cfg.get("lr", 1e-4)
        batch_size = cfg.get("batch_size", 4)
        weight_decay = cfg.get("weight_decay", 1e-5)
        device = self._device

        # Handle both dataset with __len__ and generators
        if hasattr(train_dataset, "__len__"):
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        else:
            # Generator - use directly
            train_loader = train_dataset

        # Setup optimizer
        params = list(self._encoder.parameters()) + list(self._decoder.parameters())
        optim = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}

        self._encoder.train()
        self._decoder.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                # Handle both tuple (window, target) from smoke test and custom batch formats
                if isinstance(batch, tuple) and len(batch) == 2:
                    window, target = batch
                    # window is (T, B, C, Nx, Ny) from smoke test
                    # Convert to our format
                    T, B, C, Nx, Ny = window.shape
                    input_tensor = window.permute(1, 0, 2, 3, 4).reshape(B, Nx * Ny, T * C)
                    coords = self._get_canonical_coords(Nx, Ny).expand(B, -1, -1).to(device)
                    input_tensor = input_tensor.to(device)

                    # Forward
                    reconstructed, _ = self.forward((input_tensor, coords))
                    loss = loss_fn(reconstructed, input_tensor)
                else:
                    # Original format from Parametric_2D_Dataset
                    canonical_batch = torch.stack([b[0] for b in batch]).float().to(device)
                    canonical_sample = CanonicalSample(fields=canonical_batch)

                    # Forward
                    native_input, coords = self.from_canonical(canonical_sample)
                    reconstructed, _ = self.forward((native_input, coords))

                    # Loss in native space
                    loss = loss_fn(reconstructed, native_input)

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optim.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

        return TrainingResult(history=history)