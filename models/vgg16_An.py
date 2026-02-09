import torch
import torch.nn as nn
from typing import Tuple
from config import Config

class VGG16(nn.Module):
    """VGG16 model"""

    def __init__(self, config):
        super(VGG16, self).__init__()
        self.config = config

    def build_vgg16_model(self, num_classes: int, input_shape: Tuple[int, int, int]) -> nn.Module:
        """Build VGG16 model with BatchNormalization"""
        momentum = self.config.batch_norm_momentum

        def conv_block(in_channels, out_channels, num_convs):
            block = []
            for _ in range(num_convs):
                block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                block.append(nn.BatchNorm2d(out_channels, momentum=momentum))
                block.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if self.config.use_spatial_dropout:
                block.append(nn.Dropout2d(p=self.config.spatial_dropout_rate))
            return nn.Sequential(*block)

        features = nn.Sequential(
            conv_block(3, 64, 2),
            conv_block(64, 128, 2),
            conv_block(128, 256, 3),
            conv_block(256, 512, 3),
            conv_block(512, 512, 3),
        )

        dense_drop1 = getattr(self.config, "fc_dropout_rate1", self.config.dropout_rate)
        dense_drop2 = getattr(self.config, "fc_dropout_rate2", self.config.dropout_rate)

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256 if num_classes == 10 else 512),
            nn.BatchNorm1d(256 if num_classes == 10 else 512, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dense_drop1),
            nn.Linear(256 if num_classes == 10 else 512, 512),
            nn.BatchNorm1d(512, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dense_drop2),
            nn.Linear(512, num_classes),
        )

        model = nn.Sequential(features, classifier)
        self._init_weights(model)
        return model

    def _init_weights(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self[0](x)