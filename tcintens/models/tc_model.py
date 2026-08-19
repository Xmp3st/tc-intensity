# -*- coding: utf-8 -*-
"""
tc_model.py —— 热带气旋强度识别模型

把「骨干网络 + 任务头部」封装为一个 nn.Module：
  - 分类任务：Linear(num_features, num_classes)
  - 回归任务：Linear(num_features, reg_hidden) -> ReLU -> Dropout -> Linear(reg_hidden, 1)

支持冻结骨干（freeze_backbone）做小数据集迁移学习。
"""
import torch
import torch.nn as nn

from .backbones import build_backbone


class TCIntensityModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        task: str = "classification",
        num_classes: int = 6,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
        reg_hidden: int = 128,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.task = task
        self.backbone, self.num_features = build_backbone(backbone_name, pretrained)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        if task == "regression":
            self.head = nn.Sequential(
                nn.Linear(self.num_features, reg_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(reg_hidden, 1),
            )
        else:
            self.head = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        out = self.head(feat)
        if self.task == "regression":
            return out.squeeze(-1)  # (B,) 连续风速
        return out                    # (B, num_classes) logits


def build_model(cfg) -> TCIntensityModel:
    """按配置构造模型。"""
    m = cfg.to_dict()["model"]
    data_cfg = cfg.to_dict()["data"]
    task = data_cfg.get("task", "classification")
    num_classes = len(data_cfg.get("class_names", [])) if task == "classification" else 1
    model = TCIntensityModel(
        backbone_name=m.get("backbone", "resnet18"),
        task=task,
        num_classes=num_classes,
        pretrained=m.get("pretrained", True),
        freeze_backbone=m.get("freeze_backbone", False),
        dropout=m.get("dropout", 0.3),
        reg_hidden=m.get("reg_hidden", 128),
    )
    return model
