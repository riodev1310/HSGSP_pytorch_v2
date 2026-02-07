import torch
import torch.nn as nn
from torch.nn import init
from typing import Tuple
from config import Config

class VGG16(nn.Module):
    """VGG16 model"""

    def __init__(self, config):
        super(VGG16, self).__init__()
        self.config = config
        self.features = nn.Sequential()
        self.classifier = nn.Sequential()

    def build_vgg16_model(self, num_classes: int, input_shape: Tuple[int, int, int]) -> nn.Module:
        """Build VGG16 model with BatchNormalization"""
        # L2 regularization is handled in optimizer (weight_decay)
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

        features = [
            conv_block(3, 64, 2),
            conv_block(64, 128, 2),
            conv_block(128, 256, 3),
            conv_block(256, 512, 3),
            conv_block(512, 512, 3),
        ]
        self.features = nn.Sequential(*features)

        dense_drop1 = getattr(self.config, "fc_dropout_rate1", self.config.dropout_rate)
        dense_drop2 = getattr(self.config, "fc_dropout_rate2", self.config.dropout_rate)

        classifier = [
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ]
        if num_classes == 10:
            classifier.append(nn.Linear(512, 256))
        else:
            classifier.append(nn.Linear(512, 512))
        classifier.extend([
            nn.BatchNorm1d(classifier[-1].out_features, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dense_drop1),
            nn.Linear(classifier[-4].out_features, 512),
            nn.BatchNorm1d(512, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dense_drop2),
            nn.Linear(512, num_classes),
        ])
        self.classifier = nn.Sequential(*classifier)

        model = nn.Sequential(self.features, self.classifier)
        self._init_weights(model)
        return model

    def _init_weights(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x