import os
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
from datetime import datetime

@dataclass
class Config:
    """Configuration optimized for max accuracy on VGG16 CIFAR-10"""

    # ========== DATA CONFIGURATION ==========
    task: str = 'cifar10'  # 'cifar10', 'cifar100', or 'imagenet'
    validation_split: float = 0.1  # Giảm để có train set lớn hơn (45k samples for CIFAR; for ImageNet, use pre-split)
    data_augmentation: bool = True
    batch_size: int = 128  # 128 for CIFAR; recommend 256 for ImageNet on multi-GPU

    # Dataset specific
    num_classes_cifar10: int = 10
    num_classes_cifar100: int = 100
    num_classes_imagenet: int = 200
    input_shape_cifar10: Tuple[int, int, int] = (3, 32, 32)
    input_shape_cifar100: Tuple[int, int, int] = (3, 32, 32)
    input_shape_imagenet: Tuple[int, int, int] = (3, 64, 64)

    # ========== TRAINING CONFIGURATION ==========
    default_epochs: int = 300  # Tăng để converge tốt hơn (300 for CIFAR; 90-100 for ImageNet)
    initial_lr: float = 0.05  # Cao hơn cho SGD start nhanh (0.05 for CIFAR batch 128; 0.01 for ImageNet batch 256)
    pruned_growth_lr: float = 1e-4
    min_lr: float = 1e-5
    momentum: float = 0.9  # for SGD
    optimizer: str = 'sgd'  # Thay adamw bằng sgd cho acc cao hơn trên CIFAR và ImageNet

    # Learning rate schedule
    lr_schedule: str = 'step'  # Step decay đơn giản và hiệu quả cho VGG (decay at 30,60,90 for ImageNet)
    lr_warmup_epochs: int = 5
    lr_decay_rate: float = 0.1  # Decay by 0.1
    lr_decay_steps: int = 80  # Decay every 80 epochs (e.g., at 80, 160, 240 for CIFAR; adjust to 30 for ImageNet)

    # Early Stopping (giữ nguyên nhưng tăng patience)
    early_stopping_patience: int = 15  # Tăng để tránh stop sớm
    early_stopping_min_delta: float = 1e-4
    reduce_lr_patience: int = 10
    reduce_lr_factor: float = 0.5
    reduce_lr_min_delta: float = 1e-3
    fine_tune_lr_schedule: str = 'cosine'
    fine_tune_exp_decay: float = 0.8
    fine_tune_cosine_min_factor: float = 0.05
    fine_tune_linear_end_factor: float = 0.1
    fine_tune_step_decay_rate: float = 0.5
    fine_tune_step_decay_epochs: int = 5

    # Regularization
    l2_regularization: float = 5e-4  # Tăng nhẹ so với 1e-4 cho CIFAR; 1e-4 standard for ImageNet
    batch_norm_momentum: float = 0.99  # Tăng để smooth hơn

    dropout_rate: float = 0.2  # Giảm convolutional dropout (0.5 standard in VGG paper for FC layers)
    use_spatial_dropout: bool = False  # Disable để giữ feature spatial
    spatial_dropout_rate: float = 0.0
    fc_dropout_rate1: float = 0.3  # Giảm nhẹ
    fc_dropout_rate2: float = 0.2

    weight_decay: float = 5e-4  # Match l2_reg (1e-4 for ImageNet)

    # Label Smoothing
    label_smoothing: float = 0.1

    # ========== DISTILLATION CONFIGURATION ==========
    distill_alpha: float = 0.0  # Disable distillation (set 0 để focus vanilla training)
    distill_temperature: float = 2.5

    # ========== PRUNING CONFIGURATION ==========
    # Disable pruning để max acc (giữ full model)
    frequency_bands: Dict[str, tuple] = None
    complexity_weights: Dict[str, float] = None
    max_global_pruning_ratio: float = 0.0  # Set 0 để không prune
    min_global_keep: float = 1.0
    max_accuracy_drop: float = 0.0
    accuracy_guard_center: float = 0.92
    accuracy_guard_sharpness: float = 0.5

    simple_finetune_epochs: int = 20
    simple_finetune_lr: float = 1e-5

    # ========== AUGMENTATION CONFIGURATION ==========
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    mixup_prob: float = 0.5
    use_cutout: bool = True  # Thêm Cutout để boost acc (implement in data pipeline; use RandomErasing for ImageNet)
    cutout_length: int = 16  # Kích thước cutout cho 32x32 images (not applicable for ImageNet; use scale in transforms)

    # ========== HYBRID BASELINE CONFIGURATION ==========
    hybrid_iterations: int = 0  # Disable hybrid (set 0)
    hybrid_prune_fraction: float = 0.0
    hybrid_alpha: float = 0.5
    hybrid_kappa_beta: float = 0.1
    hybrid_initial_kappa_ratio: float = 0.5
    hybrid_mode: str = 'original'
    hybrid_min_filters: int = 8
    hybrid_finetune_epochs: int = 20
    hybrid_warmup_epochs: int = 0
    hybrid_warmup_lr: float = 2e-4
    hybrid_regrow_fraction: float = 0.0

    frequency_regularization_layers: int = 0  # Disable
    frequency_entropy_beta: float = 0.0
    frequency_entropy_target_batches: int = 8
    frequency_entropy_refresh_interval: int = 3
    frequency_entropy_layer_weights: Dict[str, float] = field(default_factory=dict)
   
    frn_validation_split: float = 0.2
    frn_plot_training: bool = True
    frn_feature_count: Optional[int] = 3
    frn_hidden_units: Tuple[int, ...] = (64, 32)
    frn_activation_batches: int = 512
    frn_min_validation_samples: int = 128
    frn_dropout_rate: float = 0.05
    frn_use_batchnorm: bool = False
    frn_use_activation_features: bool = False
    frn_low_vs_rest: bool = False
    frn_architecture: str = "dense"
    frn_ema_beta: float = 0.8
    frn_sharpen_gamma: float = 2.0
    frn_initial_lr: float = 1e-4
    frn_min_lr: float = 1e-5
    frn_epochs: int = 15
    frn_cosine_min_factor: float = 0.1
    frn_batch_size: int = 256
    frn_weight_clip: float = 2.0  

    # ========== PATHs CONFIGURATION ==========
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%d%m%Y_%H%M%S"))
    tensorboard_dir: str = field(init=False)
    results_dir: str = field(init=False)
    logs_dir: str = field(init=False)
    models_dir: str = field(init=False)
    plots_dir: str = field(init=False)

    def __post_init__(self):
        if self.frequency_bands is None:
            self.frequency_bands = {
                'low': (0.0, 0.25),
                'mid': (0.25, 0.5),
                'high': (0.5, 1.0)
            }

        # Build directories using the provided task and a timestamp run_id
        exp_root = f"./EXPERIMENT/{self.run_id}_{self.task}"
        self.tensorboard_dir = os.path.join(exp_root, "tensorboard_logs")
        self.results_dir = os.path.join(exp_root, "results")
        self.logs_dir = os.path.join(exp_root, "logs")
        self.models_dir = "/content/drive/MyDrive/ML_DL_models/imagenet_models"
        self.plots_dir = os.path.join(exp_root, "plots")

        # Create directories
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # Task-specific overrides for optimal settings
        if self.task == 'imagenet':
            self.batch_size = 256  # Larger batch for ImageNet
            self.default_epochs = 90  # Standard for VGG on ImageNet
            self.initial_lr = 0.01  # Adjusted for larger batch
            self.lr_decay_steps = 30  # Decay every 30 epochs (at 30,60,90)
            self.l2_regularization = 1e-4
            self.weight_decay = 1e-4
            self.use_mixup = False  # Optional; disable for vanilla max acc
            self.use_cutout = False  # Use RandomErasing in data loader instead