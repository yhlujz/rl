import torch
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


class PolicyUNet(nn.Module):
    """策略网络：输入21X21X9X3的状态，输出6个动作的概率"""

    def __init__(self):
        super().__init__()

        self.inc = DoubleConv(3, 8)  # 21x21x9x8
        self.down1 = Down(8, 16)  # 10x10x4x16
        self.down2 = Down(16, 32)  # 5x5x2x32=1600
        self.fc1 = nn.Linear(1600, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 6)
        # 初始化网络参数
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                orthogonal_init(m)
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3, gain=0.01)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2).view(-1, 1600)  # 展平为一维张量
        x4 = torch.tanh(self.fc1(x3))
        x5 = torch.tanh(self.fc2(x4))
        x6 = torch.softmax(self.fc3(x5), dim=1)
        return x6


class ValueUNet(nn.Module):
    """价值网络：输入21X21X9X3的状态，输出1个当前状态的价值"""

    def __init__(self):
        super().__init__()

        self.inc = DoubleConv(3, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.fc1 = nn.Linear(1600, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        # 初始化网络参数
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                orthogonal_init(m)
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2).view(-1, 1600)  # 展平为一维张量
        x4 = torch.tanh(self.fc1(x3))
        x5 = torch.tanh(self.fc2(x4))
        x6 = self.fc3(x5)
        return x6
