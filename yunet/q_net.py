import torch
import torch.nn as nn
from .net_parts import (
    orthogonal_init,
    DoubleConv,
    Down,
    DoubleConvRelu,
    DownRelu,
)


class QNet(nn.Module):
    """q值网络：输入21X21X9Xc的状态，输出n个动作的价值"""

    def __init__(self, action_num, state_channel, OI):
        super().__init__()

        self.inc = DoubleConv(state_channel, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
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
            orthogonal_init(self.fc3)

    def forward(self, x, y, z):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2).view(-1, 1600)  # 展平为一维张量
        x4 = torch.tanh(self.fc1(x3))
        x5 = torch.tanh(self.fc2(x4))
        x6 = self.fc3(torch.cat((x5, y, z), dim=1))
        return x6


class QNet2(nn.Module):
    """q值网络：输入21X21X9Xc的状态，输出n个动作的价值"""

    def __init__(self, action_num, state_channel, OI):
        super().__init__()

        self.inc = DoubleConv(state_channel, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
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
            orthogonal_init(self.fc3)

    def forward(self, x, y, z):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2).view(-1, 1600)  # 展平为一维张量
        x4 = torch.tanh(self.fc1(x3))
        x5 = torch.tanh(self.fc2(torch.cat((x4, y, z), dim=1)))
        x6 = self.fc3(x5)
        return x6


class VANet(nn.Module):
    """q值网络：输入21X21X9Xc的状态，输出n个动作的价值"""

    def __init__(self, action_num, state_channel, OI):
        super().__init__()

        self.inc = DoubleConv(state_channel, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.fc1 = nn.Linear(1600, 256)
        self.fc2 = nn.Linear(256, 64)  # 以上为共享网络部分
        self.fc3 = nn.Linear(66, 1)
        self.fc4 = nn.Linear(66, action_num)
        if OI:
            # 正交初始化
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    orthogonal_init(m)
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)

    def forward(self, x, y, z):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2).view(-1, 1600)  # 展平为一维张量
        x4 = torch.tanh(self.fc1(x3))
        x5 = torch.tanh(self.fc2(x4))
        V = self.fc3(torch.cat((x5, y, z), dim=1))
        A = self.fc4(torch.cat((x5, y, z), dim=1))
        Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q


class VANet2(nn.Module):
    """q值网络：输入21X21X9Xc的状态，输出n个动作的价值"""

    def __init__(self, action_num, state_channel, OI):
        super().__init__()

        self.inc = DoubleConv(state_channel, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.fc1 = nn.Linear(1600, 254)
        self.fc2 = nn.Linear(256, 64)  # 以上为共享网络部分
        self.fc3 = nn.Linear(64, 1)
        self.fc4 = nn.Linear(64, action_num)
        if OI:
            # 正交初始化
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    orthogonal_init(m)
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)

    def forward(self, x, y, z):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2).view(-1, 1600)  # 展平为一维张量
        x4 = torch.tanh(self.fc1(x3))
        x5 = torch.tanh(self.fc2(torch.cat((x4, y, z), dim=1)))
        V = self.fc3(x5)
        A = self.fc4(x5)
        Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q


class VANetRelu(nn.Module):
    """q值网络：输入21X21X9Xc的状态，输出n个动作的价值"""

    def __init__(self, action_num, state_channel, OI):
        super().__init__()

        self.inc = DoubleConvRelu(state_channel, 8)
        self.down1 = DownRelu(8, 16)
        self.down2 = DownRelu(16, 32)
        self.fc1 = nn.Linear(1600, 256)
        self.fc2 = nn.Linear(256, 64)  # 以上为共享网络部分
        self.fc3 = nn.Linear(66, 1)
        self.fc4 = nn.Linear(66, action_num)
        if OI:
            # 正交初始化
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    orthogonal_init(m)
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)

    def forward(self, x, y, z):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2).view(-1, 1600)  # 展平为一维张量
        x4 = torch.relu(self.fc1(x3))
        x5 = torch.relu(self.fc2(x4))
        V = self.fc3(torch.cat((x5, y, z), dim=1))
        A = self.fc4(torch.cat((x5, y, z), dim=1))
        Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q
