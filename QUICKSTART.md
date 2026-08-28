# QUICKSTART —— 极简上手（WPAC 主线）

> 目标：用 TCIR 西北太平洋（WPAC）卫星图训练「图像→风速 Vmax」回归模型，并做评估、可视化、推理。
> 数据已抽取为紧凑文件，**无需碰 30GB 原始 h5**。所有 python 用 `dl_env` 环境的解释器。

```bash
cd /home/yzm/tf/tc_intensity
PY=/home/yzm/tf/dl-env/bin/python3.11     # 环境 python（train.sh 已内置，无需手填）
CFG=configs/tcir_wpac_train.yaml
```

---

## 0. 前置检查（首次 / 换机器）

```bash
ls data/wpac_96.h5 data/wpac_compact.csv   # 数据是否就绪
# 若缺失，用 scripts/extract_wpac.py 抽取（见 README §4.1）
```

---

## 1. 训练

```bash
# 方式 A：一键脚本（默认 VGG16，40 轮，自动命名实验目录）
./train.sh

# 方式 B：直接入口（调参用 --set 覆盖）
$PY train.py -c $CFG --set model.backbone=vgg16
```

产物：`outputs/tcir_wpac_vgg16/tcir_wpac_vgg16/checkpoints/best.ckpt`

---

## 2. 评估（测试集指标）

```bash
$PY evaluate.py -c $CFG --set model.backbone=vgg16
```

> ⚠️ 必须带 `-c $CFG` 与 `--set model.backbone=vgg16`，否则数据/骨干不匹配。
> 指标写入 `outputs/tcir_wpac/tcir_wpac_v1/test_metrics.json`。

---

## 3. 误差分析（最大误差样本）

```bash
$PY scripts/error_analysis.py -c $CFG --set model.backbone=vgg16
```

产物：`test_predictions.csv`（逐样本）、`worst12.csv`（误差最大 12 个样本）。

---

## 4. 可视化

```bash
$PY scripts/visualize_test.py -c $CFG --set model.backbone=vgg16
# 可选参数：--samples 16 --scatter-n 2913 --out outputs/my_viz.png
```

产物：`outputs/tcir_wpac/tcir_wpac_v1/visualization.png`
（上方=抽样台风伪 RGB 图+Pred/True Vmax；下方=方形等距 Pred vs True 散点图）

---

## 5. 推理（新数据）

```bash
# 单张 .npy（H,W,C，通道顺序同配置 IR1/WV/PMW）
$PY predict.py --image frame.npy -c $CFG --set model.backbone=vgg16

# 文件夹批量 → CSV
$PY predict.py --folder ./frames/ -c $CFG --set model.backbone=vgg16 --out preds.csv

# 直接对紧凑 h5 按帧号推理
$PY predict.py --h5 data/wpac_96.h5 --indices 10260,9603 -c $CFG \
    --set model.backbone=vgg16 --out preds.csv
```

---

## 6. 换骨干 / 调参速查

```bash
./train.sh resnet18 60                       # resnet18，60 轮
./train.sh vgg16 40 train.lr=2e-4 train.batch_size=32   # 追加任意 --set

# 4 通道（含 VIS）
$PY train.py -c $CFG --set model.backbone=vgg16 \
    model.in_channels=4 data.tcir.channels='["IR1","WV","VIS","PMW"]'
```

---

## 7. 当前结果（VGG16，测试集 N=2913）

| 指标 | 值 |
|---|---|
| RMSE | 16.22 kn |
| MAE  | 11.93 kn |
| R²   | 0.7613 |

---

## 坑位速记

- **骨干必须显式指定**：当前 `tcir_wpac_v1` 是 VGG16 权重，所有 `evaluate/error_analysis/visualize/predict`
  都加 `--set model.backbone=vgg16`。
- 不同骨干用 `train.sh` 自动分目录（`tcir_wpac_vgg16` / `tcir_wpac_resnet18`），不会互相覆盖。
- 完整文档见 [README.md](README.md)；TCIR 数据集说明见 [docs/tcir_training.md](docs/tcir_training.md)。
