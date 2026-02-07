import torch
from typing import Dict, Iterable, Tuple, Union, Optional

def _dct2(x: torch.Tensor) -> torch.Tensor:
    """Apply orthonormal 2-D DCT (type-II) over the last two axes."""
    return torch.fft.fft2(x, norm="ortho")

def compute_spectral_entropy(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    shape = features.shape
    height, width = shape[1], shape[2]
    features = features.view(-1, height, width)
    coeffs = _dct2(features)
    energy = coeffs.pow(2)
    total = energy.sum(dim=[1, 2], keepdim=True) + eps
    probs = energy / total
    entropy = - (probs * torch.log(probs.clamp(min=eps))).sum(dim=[1, 2])
    return entropy.mean()

# Other functions converted similarly...