# QUICKSTART —— 极简上手（WPAC 主线）

> 目标：用 TCIR 西北太平洋（WPAC）卫星图训练「图像→风速 Vmax」回归模型，并做评估、可视化、推理。
> 数据已抽取为紧凑文件，**无需碰 30GB 原始 h5**。所有 python 用 `dl_env` 环境的解释器。

```bash
cd /home/yzm/tf/tc_intensity     # ⚠️ 换成你机器上的项目根目录
PY=/home/yzm/tf/dl-env/bin/python3.11     # 环境 python（train.sh 已内置，无需手填；其他机器改此路径）
CFG=configs/tcir_wpac_train.yaml
```

> 📌 路径约定：本文档所有相对路径（如 `data/wpac_96.h5`、`configs/...`）都以「项目根目录」为基准。
> `dl-env` 虚拟环境路径因机器而异，换机器时只需改上面两行 `cd` / `PY` 即可，命令本身不变。

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

# 方式 B：直接入口（config 默认即 vgg16）
$PY train.py -c $CFG
```

产物：`outputs/tcir_wpac/tcir_wpac_v1/checkpoints/best.ckpt`

> 📌 上面两种方式的默认产物目录一致（均落在 `outputs/tcir_wpac/tcir_wpac_v1/`）。

---

## 2. 评估（测试集指标）

```bash
$PY evaluate.py -c $CFG
```

> ⚠️ 必须带 `-c $CFG`；config 已默认 vgg16，与 `tcir_wpac_v1` 权重一致，无需额外覆盖。
> 指标写入 `outputs/tcir_wpac/tcir_wpac_v1/test_metrics.json`。

---

## 3. 误差分析（最大误差样本）

```bash
$PY scripts/error_analysis.py -c $CFG
```

产物：`test_predictions.csv`（逐样本）、`worst12.csv`（误差最大 12 个样本）。

---

## 4. 可视化

```bash
$PY scripts/visualize_test.py -c $CFG
# 可选参数：--samples 16 --scatter-n 2913 --out outputs/my_viz.png
```

产物：`outputs/tcir_wpac/tcir_wpac_v1/visualization.png`
（上方=抽样台风伪 RGB 图+Pred/True Vmax；下方=方形等距 Pred vs True 散点图）

---

## 5. 推理（新数据）

```bash
# 单张 .npy（H,W,C，通道顺序同配置 IR1/WV/PMW）
$PY predict.py --image frame.npy -c $CFG

# 文件夹批量 → CSV
$PY predict.py --folder ./frames/ -c $CFG --out preds.csv

# 直接对紧凑 h5 按「紧凑行号」推理（matrix_index，见 README §4 路径约定）
$PY predict.py --h5 data/wpac_96.h5 --indices 10260,9603 -c $CFG --out preds.csv
```

---

## 6. 换骨干 / 调参速查

```bash
./train.sh resnet18                       # resnet18，默认 40 轮
./train.sh vgg16 60                       # vgg16，60 轮
./train.sh vgg16 40 train.lr=2e-4 train.batch_size=32   # 追加任意 --set

# 4 通道（含 VIS）：需重算归一化统计（compute_tcir_stats.py）
$PY train.py -c $CFG \
    model.in_channels=4 data.tcir.channels='["IR1","WV","VIS","PMW"]'
```

> 📌 换其他骨干请用 `train.sh <骨干>`（自动建独立目录 `tcir_wpac_<骨干>`，如 `tcir_wpac_resnet18`），避免覆盖默认 VGG16 权重。

---

## 7. 当前结果（VGG16，测试集 N=2913）

| 指标 | 值 |
|---|---|
| RMSE | 16.22 kn |
| MAE  | 11.93 kn |
| R²   | 0.7613 |

---

## 坑位速记

- **默认即 VGG16**：`configs/tcir_wpac_train.yaml` 默认 `model.backbone=vgg16`，与 `tcir_wpac_v1` 权重一致，
  所有 `evaluate/error_analysis/visualize/predict` 直接 `-c $CFG` 即可，无需 `--set model.backbone=...`。
- 换其他骨干请用 `train.sh <骨干>`（自动建独立目录 `tcir_wpac_<骨干>`），避免覆盖。
- 完整文档见 [README.md](README.md)；TCIR 数据集说明见 [docs/tcir_training.md](docs/tcir_training.md)。
