import torch.nn as nn


def orthogonal_init(layer, gain=1.0):
    """正交初始化"""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class DoubleConv(nn.Module):
    """双层卷积，(3*3*3卷积+激活函数)*2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样，最大池化+双层卷积"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
