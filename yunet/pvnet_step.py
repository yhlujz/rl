import torch
import torch.nn as nn
from .net_parts import *


class PolicyNetStep(nn.Module):
    """策略网络：输入21X21X9Xc的状态，输出6个动作的概率"""

    def __init__(self, state_channel):
        super().__init__()

        self.inc = DoubleConv(state_channel, 8)  # 21x21x9x8
        self.down1 = Down(8, 16)  # 10x10x4x16
        self.down2 = Down(16, 32)  # 5x5x2x32=1600
        self.fc1 = nn.Linear(1600, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(66, 6)
        # 初始化网络参数
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


class ValueNetStep(nn.Module):
    """价值网络：输入21X21X9Xc的状态，输出1个当前状态的价值"""

    def __init__(self, state_channel):
        super().__init__()

        self.inc = DoubleConv(state_channel, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.fc1 = nn.Linear(1600, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(66, 1)
        # 初始化网络参数
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                orthogonal_init(m)
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, x, y, z):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2).view(-1, 1600)  # 展平为一维张量
        x4 = torch.tanh(self.fc1(x3))
        x5 = torch.tanh(self.fc2(x4))
        x6 = self.fc3(torch.cat((x5, y, z), dim=1))
        return x6
