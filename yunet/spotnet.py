import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """双层卷积，(3*3*3卷积+BN+ReLU)*2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
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


class OutConv(nn.Module):
    """1*1*1卷积输出"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SpotNet(nn.Module):
    """输入160X160X64的CT图像，下采样得到三维空间中点的坐标"""

    def __init__(self):
        super().__init__()

        self.inc = DoubleConv(1, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.down4 = Down(64, 128)
        self.down5 = Down(128, 256)
        self.outc = OutConv(256, 60)
        self.fc1 = nn.Linear(3000, 60)
        self.fc2 = nn.Linear(60, 3)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.outc(x6).view(-1, 3000)  # 转化为只有batch维的一维张量
        x8 = F.relu(self.fc1(x7))
        x9 = self.fc2(x8)
        return x9
