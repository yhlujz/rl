import torch
import torch.nn as nn
from .net_parts import (
    orthogonal_init,
    DoubleConv,
    Down,
)


class VANet(nn.Module):
    """q值网络：输入21X21X9Xc的状态，输出6个动作的价值"""

    def __init__(self, state_channel):
        super().__init__()

        self.inc = DoubleConv(state_channel, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.fc1 = nn.Linear(1600, 256)
        self.fc2 = nn.Linear(256, 64)  # 以上为共享网络部分
        self.fc3 = nn.Linear(66, 1)
        self.fc4 = nn.Linear(66, 6)
        # 初始化网络参数
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
