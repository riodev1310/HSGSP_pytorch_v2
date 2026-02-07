import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Sequence, Optional

from models.model_utils import ModelUtils

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

        layers = []
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

    def dct2_ortho(self, x: torch.Tensor) -> torch.Tensor:
        """2D DCT-II with orthonormal scaling over the spatial axes."""
        x = torch.fft.fft2(x, norm="ortho")
        return x

    def idct2_ortho(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse 2D DCT (type-II) with orthonormal scaling."""
        x = torch.fft.ifft2(x, norm="ortho")
        return x.real

    def create_mask(self, h: int, w: int, lo: float, hi: float) -> np.ndarray:
        u = np.arange(h) / (h - 1) if h > 1 else np.zeros(h)
        v = np.arange(w) / (w - 1) if w > 1 else np.zeros(w)
        U, V = np.meshgrid(u, v, indexing="ij")
        r = np.sqrt(U**2 + V**2) / np.sqrt(2.0)
        return ((r >= lo) & (r < hi)).astype(np.float32)

    band_defs = {"low": (0.0, 0.25), "mid": (0.25, 0.5), "high": (0.5, 1.0)}

    def band_energies_from_kernel(self, kernel_np: np.ndarray):
        """Return band energies, ratios, and totals for a Conv2D kernel."""
        h, w, _, cout = kernel_np.shape
        k_torch = torch.from_numpy(kernel_np).float()
        X = self.dct2_ortho(k_torch)
        totals = torch.zeros((cout,), dtype=torch.float64)
        band_E: Dict[str, torch.Tensor] = {}
        for band, (lo, hi) in self.band_defs.items():
            mask = self.create_mask(h, w, lo, hi)
            mask_torch = torch.from_numpy(mask).reshape(h, w, 1, 1).float()
            coeffs = X * mask_torch
            energy = torch.sqrt(torch.sum(torch.square(coeffs), dim=(0, 1, 2)))
            band_E[band] = energy.double()
            totals += energy.double()
        eps = 1e-12
        ratios = {band: band_E[band] / (totals + eps) for band in band_E}
        return band_E, ratios, totals

    def band_grad_energies_from_batch(
        self,
        model: nn.Module,
        layer: nn.Module,
        xb: torch.Tensor,
        yb: torch.Tensor,
        from_logits: bool,
    ) -> Dict[str, torch.Tensor]:
        """Gradient-based Taylor energies split per frequency band for one batch."""
        criterion = nn.CrossEntropyLoss()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        grad = layer.weight.grad
        if grad is None:
            return {band: torch.zeros((layer.out_channels,), dtype=torch.float64) for band in self.band_defs}

        G = self.dct2_ortho(grad)
        band_G: Dict[str, torch.Tensor] = {}
        for band, (lo, hi) in self.band_defs.items():
            mask = self.create_mask(G.shape[0], G.shape[1], lo, hi)
            mask_torch = torch.from_numpy(mask).reshape(G.shape[0], G.shape[1], 1, 1).float().to(G.device)
            coeffs = G * mask_torch
            energy = torch.sqrt(torch.sum(torch.square(coeffs), dim=(0, 1, 2)))
            band_G[band] = energy.double()
        return band_G

    def build_frn_training_set(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        max_batches: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Legacy smaller builder using only frequency ratios."""
        from_logits = self._model_outputs_logits(model)
        binary_targets = bool(getattr(self.config, "frn_low_vs_rest", False))
        num_classes = 2 if binary_targets else 3

        conv_layers = ModelUtils.get_conv_layers(model)
        if not conv_layers:
            return np.zeros((0, num_classes), np.float32), np.zeros((0, num_classes), np.float32)

        grad_acc: Dict[str, Dict[str, np.ndarray]] = {l.name: None for l in conv_layers if hasattr(l, 'name')}

        for xb, yb in dataloader:
            xb = xb.float()
            for layer in conv_layers:
                band_grads = self.band_grad_energies_from_batch(model, layer, xb, yb, from_logits)
                if grad_acc[layer.name] is None:
                    grad_acc[layer.name] = {band: val.numpy().copy() for band, val in band_grads.items()}
                else:
                    for band in self.band_defs:
                        grad_acc[layer.name][band] += band_grads[band].numpy()

        X_list: List[np.ndarray] = []
        Y_list: List[np.ndarray] = []
        for layer in conv_layers:
            weights = layer.weight.detach().numpy()
            band_E, ratios, _ = self.band_energies_from_kernel(weights)
            cout = weights.shape[0]
            grad_band = grad_acc.get(layer.name, {
                band: np.zeros((cout,), np.float64) for band in self.band_defs
            })

            grad_tot = np.zeros((cout,), np.float64)
            for band in self.band_defs:
                grad_tot += grad_band[band]
            eps = 1e-12
            Y = np.stack(
                [
                    grad_band["low"] / (grad_tot + eps),
                    grad_band["mid"] / (grad_tot + eps),
                    grad_band["high"] / (grad_tot + eps),
                ],
                axis=1,
            )
            if binary_targets:
                low_col = Y[:, 0:1]
                other_col = np.sum(Y[:, 1:], axis=1, keepdims=True)
                Y = np.concatenate([low_col, other_col], axis=1)
                Y = Y / (np.sum(Y, axis=1, keepdims=True) + eps)
            X = np.stack(
                [ratios["low"].numpy(), ratios["mid"].numpy(), ratios["high"].numpy()],
                axis=1,
            )
            mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
            if not np.any(mask):
                continue
            X_list.append(X[mask])
            Y_list.append(Y[mask])

        if not X_list:
            return np.zeros((0, num_classes), np.float32), np.zeros((0, num_classes), np.float32)

        X_all = np.concatenate(X_list, axis=0).astype(np.float32)
        Y_all = np.concatenate(Y_list, axis=0).astype(np.float32)
        return X_all, Y_all

    def _model_outputs_logits(self, model: nn.Module) -> bool:
        last_module = list(model.modules())[-1]
        if isinstance(last_module, nn.Softmax):
            return False
        return True