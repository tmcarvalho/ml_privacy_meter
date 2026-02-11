import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_shape, num_classes=10, hidden_dims=(256, 256, 128)):
        super().__init__()

        layers = []
        prev_dim = in_shape

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
            ])
            prev_dim = dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        x = x.flatten(1)
        x = self.feature_extractor(x)
        return self.classifier(x)
