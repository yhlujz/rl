import torch
import torch.nn as nn


def orthogonal_init(layer, gain=1.0):
    """正交初始化"""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class DoubleConv(nn.Module):
    """双层卷积，(3*3*3卷积+tanh)*2"""

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


class DoubleConvRelu(nn.Module):
    """双层卷积，(3*3*3卷积+Relu)*2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class DownRelu(nn.Module):
    """下采样，最大池化+双层卷积"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConvRelu(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DoubleConvGelu(nn.Module):
    """双层卷积，(3*3*3卷积+GELU)*2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)


class DownGelu(nn.Module):
    """下采样，最大池化+双层卷积"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConvGelu(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


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


class DownRes(nn.Module):
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
