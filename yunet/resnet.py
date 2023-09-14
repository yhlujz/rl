import torch
import torch.nn as nn


def orthogonal_init(layer, gain=1.0):
    """正交初始化"""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Residual(nn.Module):
    """残差结构，双层卷积"""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv3d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = torch.tanh(self.conv1(X))
        Y = self.conv2(Y)
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return torch.tanh(Y)


class DoubleRes(nn.Module):
    """双层残差结构"""
    def __init__(self, input_channels, num_channels):
        super().__init__()
        self.double_res = nn.Sequential(
            Residual(input_channels, num_channels,
                     use_1x1conv=True),
            Residual(num_channels, num_channels)
        )

    def forward(self, x):
        return self.double_res(x)


class Down(nn.Module):
    """双层残差下采样结构"""
    def __init__(self, input_channels, num_channels):
        super().__init__()
        self.down = nn.Sequential(
            Residual(input_channels, num_channels,
                     use_1x1conv=True, strides=2),
            Residual(num_channels, num_channels)
        )

    def forward(self, x):
        return self.down(x)


class PolicyResNet(nn.Module):
    """策略网络：输入21X21X9X3的状态，输出6个动作的概率"""

    def __init__(self):
        super().__init__()

        self.inc = DoubleRes(3, 8)  # 21x21x9x8
        self.down1 = Down(8, 16)  # 11x11x5x16
        self.down2 = Down(16, 32)  # 6x6x3x32
        self.down3 = Down(32, 64)  # 3x3x2x64
        self.pool = nn.AdaptiveAvgPool3d(1)  # 1x1x1x64
        self.fc = nn.Linear(64, 6)
        # 初始化网络参数
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                orthogonal_init(m)
        orthogonal_init(self.fc, gain=0.01)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.pool(x4).view(-1, 64)  # 展平为一维张量
        x6 = torch.softmax(self.fc(x5), dim=1)
        return x6


class ValueResNet(nn.Module):
    """价值网络：输入21X21X9X3的状态，输出1个当前状态的价值"""

    def __init__(self):
        super().__init__()

        self.inc = DoubleRes(3, 8)  # 21x21x9x8
        self.down1 = Down(8, 16)  # 11x11x5x16
        self.down2 = Down(16, 32)  # 6x6x3x32
        self.down3 = Down(32, 64)  # 3x3x2x64
        self.pool = nn.AdaptiveAvgPool3d(1)  # 1x1x1x64
        self.fc = nn.Linear(64, 1)
        # 初始化网络参数
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                orthogonal_init(m)
        orthogonal_init(self.fc)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.pool(x4).view(-1, 64)  # 展平为一维张量
        x6 = self.fc(x5)
        return x6
