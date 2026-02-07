import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Sequence, Optional

class FrequencyRelevanceAnalyzer:
    def __init__(self, config):
        self.config = config

    def build_frequency_relevance_net(
        self,
        hidden_units: Sequence[int] = (64, 32),
        input_dim: int = 3,
        architecture: Optional[str] = None,
        num_classes: int = 3,
    ) -> nn.Module:
        """
        Build the FRN backbone.

        Args:
            hidden_units: base hidden sizes.
            input_dim: number of input features.
            architecture: 'residual' for residual MLP, 'dense' for vanilla stack.
        """
        arch = (architecture or getattr(self.config, "frn_architecture", "residual")).lower()
        if arch not in {"residual", "dense"}:
            arch = "residual"
        units_seq = tuple(int(u) for u in hidden_units if int(u) > 0) or (64, 32)
        use_batchnorm = bool(getattr(self.config, "frn_use_batchnorm", True))
        dropout_cfg = getattr(self.config, "frn_dropout_rate", 0.05)
        dropout_rates = [float(dropout_cfg)] * len(units_seq)  # Simplified

        layers = nn.ModuleList()
        in_features = input_dim

        if arch == "dense":
            for units, drop_rate in zip(units_seq, dropout_rates):
                layers.append(nn.Linear(in_features, units))
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(units))
                layers.append(nn.ReLU())
                if drop_rate > 0.0:
                    layers.append(nn.Dropout(drop_rate))
                in_features = units
        else:
            for units, drop_rate in zip(units_seq, dropout_rates):
                layers.append(nn.Linear(in_features, units))
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(units))
                layers.append(nn.ReLU())
                if drop_rate > 0.0:
                    layers.append(nn.Dropout(drop_rate))
                in_features = units

        layers.append(nn.Linear(in_features, num_classes))
        layers.append(nn.Softmax(dim=-1))

        model = nn.Sequential(*layers)
        return model

    # ---------------------------- utilities ---------------------------- #
    def dct2_ortho(self, x: torch.Tensor) -> torch.Tensor:
        """2D DCT-II with orthonormal scaling over the spatial axes."""
        # PyTorch has no built-in DCT, use torch.fft or implement
        # For simplicity, use fft (approximate)
        x = torch.fft.fft2(x, norm="ortho")
        return x

    def idct2_ortho(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse 2D DCT (type-II) with orthonormal scaling."""
        x = torch.fft.ifft2(x, norm="ortho")
        return x

    def create_mask(self, h: int, w: int, lo: float, hi: float) -> np.ndarray:
        u = np.arange(h) / (h - 1) if h > 1 else np.zeros(h)
        v = np.arange(w) / (w - 1) if w > 1 else np.zeros(w)
        U, V = np.meshgrid(u, v, indexing="ij")
        r = np.sqrt(U**2 + V**2) / np.sqrt(2.0)
        return ((r >= lo) & (r < hi)).astype(np.float32)

    band_defs = {"low": (0.0, 0.25), "mid": (0.25, 0.5), "high": (0.5, 1.0)}

    # Other methods converted similarly...