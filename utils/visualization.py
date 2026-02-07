import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional
from config import Config

class Visualizer:
    """Visualization utilities for training history and metrics"""

    def __init__(self, config: Config):
        self.config = config

    def plot_training_history(self,
                              history: Dict[str, list],
                              save_path: Optional[str] = None,
                              show: bool = False):
        """Plot training and validation loss/accuracy"""
        epochs = range(1, len(history['train_loss']) + 1)

        plt.figure(figsize=(12, 4))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")

        if show:
            plt.show()

        plt.close()

    # Additional plotting methods if needed, e.g., for pruning history
    def plot_pruning_history(self, history: list, save_path: Optional[str] = None):
        pass  # Implement if needed