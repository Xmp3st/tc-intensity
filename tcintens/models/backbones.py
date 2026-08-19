# -*- coding: utf-8 -*-
"""
backbones.py —— 可插拔骨干网络（来自 torchvision）

支持的 backbone（在 configs 的 model.backbone 指定）：
  resnet18 / resnet34 / resnet50 / resnet101
  efficientnet_b0 / efficientnet_b1
  mobilenet_v3_small
  vit_b_16

每个 build_backbone 返回 (backbone, num_features)：
  backbone 已去掉原始分类头，输出「特征向量」；
  num_features 是特征维度，供后续接自定义头部。
"""
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
}


def build_backbone(name: str, pretrained: bool = True):
    if name not in _BUILDERS:
        raise ValueError(
            f"不支持的 backbone: {name}。可选: {list(_BUILDERS.keys())}"
        )
    weights = _WEIGHTS[name] if pretrained else None
    model = _BUILDERS[name](weights=weights)

    if name.startswith("resnet"):
        num_features = model.fc.in_features
        model.fc = nn.Identity()
        return model, num_features

    if name.startswith("efficientnet"):
        num_features = model.classifier[1].in_features  # Linear 输入维度
        model.classifier = nn.Identity()
        return model, num_features

    if name.startswith("mobilenet_v3"):
        num_features = model.classifier[0].in_features
        model.classifier = nn.Identity()
        return model, num_features

    if name.startswith("vit_b_16"):
        num_features = model.heads.head.in_features
        model.heads = nn.Identity()
        return model, num_features

    raise ValueError(f"未知 backbone 结构: {name}")
