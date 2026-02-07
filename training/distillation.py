import torch
import torch.nn as nn
from typing import List, Optional

class Distiller(nn.Module):
    """
    Knowledge Distillation wrapper for PyTorch models.
    """

    def __init__(self,
                 student: nn.Module,
                 teacher: nn.Module,
                 alpha: float = 0.5,
                 temperature: float = 4.0,
                 name: Optional[str] = None):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.requires_grad_(False)

        self.alpha = float(alpha)
        self.temperature = float(temperature)

        self.student_loss_fn = nn.CrossEntropyLoss()
        self.distillation_loss_fn = nn.KLDivLoss(reduction="batchmean")

    def forward(self, x):
        return self.student(x)

    def train_step(self, x, y):
        # Implement in training loop
        pass

    # Similar to TF, but use in custom loop