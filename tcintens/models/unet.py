"""轻量 U-Net —— 用于 IR1 条状缺测的图像修复（image inpainting）。

输入 : (N, 2, H, W)  —— 通道 0 = 损坏图（缺测处为归一化后的哨兵值），通道 1 = 0/1 掩码
输出 : (N, 1, H, W)  —— 修复后的 IR1 亮温（与输入同归一化空间）

设计要点：
  - 标准编码器-解码器 + 跳跃连接，瓶颈不降分辨率过狠（96×96 输入下 4 次下采样到 6×6）。
  - 首层卷积 stride=1 保持空间，下采样用 MaxPool（比 stride 卷积对修复任务更稳）。
  - 通道数从 base=48 起，参数量约 3M，适配 96×96 小图，避免过拟合。
  - 输出层 1×1 卷积 + 无激活（回归亮温值，可为负/正，归一化后约 [-8, 1.3]）。

用法：
  from tcintens.models.unet import UNet
  model = UNet(in_channels=2, out_channels=1, base=48)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base=48):
        super().__init__()
        self.base = base
        self.enc1 = _conv_block(in_channels, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.enc4 = _conv_block(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _conv_block(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = _conv_block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv_block(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)                       # base
        e2 = self.enc2(self.pool(e1))           # base*2
        e3 = self.enc3(self.pool(e2))           # base*4
        e4 = self.enc4(self.pool(e3))           # base*8

        b = self.bottleneck(self.pool(e4))      # base*16

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)


if __name__ == "__main__":
    m = UNet(in_channels=2, out_channels=1, base=48)
    x = torch.randn(2, 2, 96, 96)
    with torch.no_grad():
        y = m(x)
    print("out shape:", tuple(y.shape))
    n = sum(p.numel() for p in m.parameters())
    print(f"参数: {n/1e6:.2f} M")
