import torch
import torch.nn as nn
from .net_parts import (
    orthogonal_init,
    DoubleConv,
    Down,
    DoubleRes,
    DownRes,
    DoubleConvGelu,
    DownGelu,
)


class PolicyNet(nn.Module):
    """策略网络：输入21X21X9Xc的状态，输出n个动作的概率"""

    def __init__(self, action_num, state_channel, OI):
        super().__init__()

        self.inc = DoubleConv(state_channel, 8)  # 21x21x9x8
        self.down1 = Down(8, 16)  # 10x10x4x16
        self.down2 = Down(16, 32)  # 5x5x2x32=1600
        self.fc1 = nn.Linear(1600, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, action_num)
        if OI:
            # 正交初始化
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


class PolicyNetLight(nn.Module):
    """策略网络：输入21X21X9Xc的状态，输出n个动作的概率"""

    def __init__(self, action_num, state_channel, OI):
        super().__init__()

        self.inc = DoubleConv(state_channel, 2)  # 21x21x9x2
        self.down1 = Down(2, 4)  # 10x10x4x4
        self.down2 = Down(4, 8)  # 5x5x2x8=400
        self.fc1 = nn.Linear(400, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, action_num)
        if OI:
            # 正交初始化
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    orthogonal_init(m)
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2).view(-1, 400)  # 展平为一维张量
        x4 = torch.tanh(self.fc1(x3))
        x5 = torch.tanh(self.fc2(x4))
        x6 = torch.softmax(self.fc3(x5), dim=1)
        return x6


class PolicyNet2(nn.Module):
    """策略网络：输入27X27X9Xc的状态，输出n个动作的概率"""

    def __init__(self, action_num, state_channel, OI):
        super().__init__()

        self.inc = DoubleConv(state_channel, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.fc1 = nn.Linear(288, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_num)
        if OI:
            # 正交初始化
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    orthogonal_init(m)
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2).view(-1, 288)  # 展平为一维张量
        x4 = torch.tanh(self.fc1(x3))
        x5 = torch.tanh(self.fc2(x4))
        x6 = torch.softmax(self.fc3(x5), dim=1)
        return x6


class PolicyResNet(nn.Module):
    """策略网络：输入21X21X9Xc的状态，输出n个动作的概率"""

    def __init__(self, action_num, state_channel, OI):
        super().__init__()

        self.inc = DoubleRes(state_channel, 8)  # 21x21x9x8
        self.down1 = DownRes(8, 16)  # 11x11x5x16
        self.down2 = DownRes(16, 32)  # 6x6x3x32
        self.down3 = DownRes(32, 64)  # 3x3x2x64
        self.pool = nn.AdaptiveAvgPool3d(1)  # 1x1x1x64
        self.fc = nn.Linear(64, action_num)
        if OI:
            # 正交初始化
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


class PolicyNetStep(nn.Module):
    """策略网络：输入21X21X9Xc的状态，输出n个动作的概率"""

    def __init__(self, action_num, state_channel, OI):
        super().__init__()

        self.inc = DoubleConv(state_channel, 8)  # 21x21x9x8
        self.down1 = Down(8, 16)  # 10x10x4x16
        self.down2 = Down(16, 32)  # 5x5x2x32=1600
        self.fc1 = nn.Linear(1600, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(66, action_num)
        if OI:
            # 正交初始化
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    orthogonal_init(m)
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, x, y, z):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2).view(-1, 1600)  # 展平为一维张量
        x4 = torch.tanh(self.fc1(x3))
        x5 = torch.tanh(self.fc2(x4))
        x6 = torch.softmax(self.fc3(torch.cat((x5, y, z), dim=1)), dim=1)
        return x6


class PolicyNetStep2(nn.Module):
    """策略网络：输入21X21X9Xc的状态，输出n个动作的概率"""

    def __init__(self, action_num, state_channel, OI):
        super().__init__()

        self.inc = DoubleConv(state_channel, 8)  # 21x21x9x8
        self.down1 = Down(8, 16)  # 10x10x4x16
        self.down2 = Down(16, 32)  # 5x5x2x32=1600
        self.fc1 = nn.Linear(1600, 254)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, action_num)
        if OI:
            # 正交初始化
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    orthogonal_init(m)
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, x, y, z):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2).view(-1, 1600)  # 展平为一维张量
        x4 = torch.tanh(self.fc1(x3))
        x5 = torch.tanh(self.fc2(torch.cat((x4, y, z), dim=1)))
        x6 = torch.softmax(self.fc3(x5), dim=1)
        return x6


class PolicyNetStepGelu(nn.Module):
    """策略网络：输入21X21X9Xc的状态，输出n个动作的概率"""

    def __init__(self, action_num, state_channel, OI):
        super().__init__()

        self.inc = DoubleConvGelu(state_channel, 8)  # 21x21x9x8
        self.down1 = DownGelu(8, 16)  # 10x10x4x16
        self.down2 = DownGelu(16, 32)  # 5x5x2x32=1600
        self.fc1 = nn.Linear(1600, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(66, action_num)
        if OI:
            # 正交初始化
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    orthogonal_init(m)
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, x, y, z):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2).view(-1, 1600)  # 展平为一维张量
        x4 = nn.functional.gelu(self.fc1(x3))
        x5 = nn.functional.gelu(self.fc2(x4))
        x6 = torch.softmax(self.fc3(torch.cat((x5, y, z), dim=1)), dim=1)
        return x6
