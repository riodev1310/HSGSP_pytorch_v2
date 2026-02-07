import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Set, Callable, Any
from dataclasses import dataclass

@dataclass
class LayerPruningConfig:
    """Configuration for pruning a specific layer"""
    layer_name: str
    original_filters: int
    filters_to_keep: int
    pruning_ratio: float
    importance_scores: np.ndarray
    mask: np.ndarray

class PruningStrategy:
    """
    Implements various pruning strategies for HSGSP
    """
    
    def __init__(self, config):
        self.config = config
        self.min_filters_per_layer = int(getattr(config, "hybrid_min_filters", 8))
        self.pruning_schedule = self._create_pruning_schedule()
    
    def _create_pruning_schedule(self) -> Dict[str, float]:
        """
        Create layer-wise pruning schedule
        Different layers may have different sensitivity to pruning
        """
        return {
            'early': 0.8,   # Keep 80% in early layers (more important)
            'middle': 0.6,  # Keep 60% in middle layers
            'late': 0.5,    # Keep 50% in late layers (can prune more)
        }
    
    def compute_layer_importance(self, 
                                layer_name: str,
                                layer_position: float) -> float:
        """
        Compute importance multiplier for a layer based on its position
        
        Args:
            layer_name: Name of the layer
            layer_position: Normalized position in network (0=first, 1=last)
            
        Returns:
            Importance multiplier (higher = more important)
        """
        if layer_position < 0.3:
            return self.pruning_schedule['early']
        elif layer_position < 0.7:
            return self.pruning_schedule['middle']
        else:
            return self.pruning_schedule['late']
    
    # Other methods converted to use torch...
    # (Omitted for brevity)