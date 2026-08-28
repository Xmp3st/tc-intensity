# -*- coding: utf-8 -*-
"""
backbones.py —— 可插拔骨干网络（来自 torchvision）

支持的 backbone（在 configs 的 model.backbone 指定）：
  resnet18 / resnet34 / resnet50 / resnet101
  efficientnet_b0 / efficientnet_b1
  mobilenet_v3_small
  vit_b_16
  vgg16

每个 build_backbone 返回 (backbone, num_features)：
  backbone 已去掉原始分类头，输出「特征向量」；
  num_features 是特征维度，供后续接自定义头部。

新增：in_channels（输入通道数）。TCIR 卫星图为 4 通道（IR1/WV/VIS/PMW），
本框架默认取 3 通道（IR1+WV+PMW，弃用白天不稳定的 VIS）。
当 in_channels != 3 时，自动改写第一个卷积层以接受任意通道数，
并以「循环复用预训练 3 通道权重」的方式初始化，保留迁移学习信号。
"""
import torch
import torch.nn as nn
from torchvision import models


# 预训练权重枚举映射，便于按名字选择
_WEIGHTS = {
    "resnet18": models.ResNet18_Weights.DEFAULT,
    "resnet34": models.ResNet34_Weights.DEFAULT,
    "resnet50": models.ResNet50_Weights.DEFAULT,
    "resnet101": models.ResNet101_Weights.DEFAULT,
    "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
    "efficientnet_b1": models.EfficientNet_B1_Weights.DEFAULT,
    "mobilenet_v3_small": models.MobileNet_V3_Small_Weights.DEFAULT,
    "vit_b_16": models.ViT_B_16_Weights.DEFAULT,
    "vgg16": models.VGG16_Weights.DEFAULT,
}

_BUILDERS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b1": models.efficientnet_b1,
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "vit_b_16": models.vit_b_16,
    "vgg16": models.vgg16,
}


def _replace_first_conv(model: nn.Module, in_channels: int, pretrained: bool) -> bool:
    """递归找到模型中的「第一个 Conv2d」并替换成 in_channels 通道版本。

    返回是否发生了替换。
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            old: nn.Conv2d = child
            if old.in_channels == in_channels:
                return True  # 无需替换
            new = nn.Conv2d(
                in_channels, old.out_channels, old.kernel_size, old.stride,
                old.padding, old.dilation, old.groups,
                bias=(old.bias is not None),
            )
            with torch.no_grad():
                if pretrained:
                    # 循环复用原 3 通道权重，并按 sqrt(3/in) 缩放保持激活方差
                    idx = torch.tensor([i % 3 for i in range(in_channels)])
                    tiled = old.weight[:, idx, ...].clone()
                    tiled.mul_((3.0 / in_channels) ** 0.5)
                    new.weight.copy_(tiled)
                    if old.bias is not None:
                        new.bias.copy_(old.bias)
                else:
                    nn.init.kaiming_normal_(new.weight)
                    if old.bias is not None:
                        nn.init.zeros_(new.bias)
            setattr(model, name, new)
            return True
        if _replace_first_conv(child, in_channels, pretrained):
            return True
    return False


def build_backbone(name: str, pretrained: bool = True, in_channels: int = 3):
    if name not in _BUILDERS:
        raise ValueError(
            f"不支持的 backbone: {name}。可选: {list(_BUILDERS.keys())}"
        )
    weights = _WEIGHTS[name] if pretrained else None
    model = _BUILDERS[name](weights=weights)

    if name.startswith("resnet"):
        num_features = model.fc.in_features
        model.fc = nn.Identity()
    elif name.startswith("efficientnet"):
        num_features = model.classifier[1].in_features  # Linear 输入维度
        model.classifier = nn.Identity()
    elif name.startswith("mobilenet_v3"):
        num_features = model.classifier[0].in_features
        model.classifier = nn.Identity()
    elif name.startswith("vit_b_16"):
        num_features = model.heads.head.in_features
        model.heads = nn.Identity()
    elif name.startswith("vgg"):
        # torchvision VGG.forward: features -> avgpool(7x7) -> flatten -> classifier
        # 将 classifier 置为 Identity 后，forward 直接返回扁平化特征 (B, 512*7*7)
        # 多通道时须先替换首层卷积，否则下面的 dummy forward 会因通道数不符而报错
        if in_channels != 3:
            _replace_first_conv(model, in_channels, pretrained)
        model.classifier = nn.Identity()
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 32, 32)
            num_features = model(dummy).shape[1]
    else:
        raise ValueError(f"未知 backbone 结构: {name}")

    # 处理多通道输入（如 TCIR 的 4 通道卫星图）
    if in_channels != 3:
        _replace_first_conv(model, in_channels, pretrained)

    return model, num_features
