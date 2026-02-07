import torch
import numpy as np
from typing import Dict, Tuple, Optional
import time

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, config):
        self.config = config
    
    def evaluate_model(self, 
                      model: torch.nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      dataset_name: str = "") -> Dict:
        """Model evaluation"""
        print(f"\nEvaluating model on {dataset_name}...")

        model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                total_correct += torch.sum(preds == labels).item()
                total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        # Per-class metrics, inference speed, complexity similar conversions...
        # (Omitted for brevity)
        return {'loss': avg_loss, 'accuracy': accuracy}