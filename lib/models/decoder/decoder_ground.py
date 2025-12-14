import torch.nn as nn
import torch.nn.functional as F


class GroundNormalHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Tanh()
        )

    def forward(self, x):  # x: (B, C, H, W)
        x = self.pool(x).view(x.size(0), -1)  # (B, C)
        x = self.head(x)                      # (B, 3), range [-1, 1]
        x = F.normalize(x, dim=-1)            # Ensure unit norm
        return x