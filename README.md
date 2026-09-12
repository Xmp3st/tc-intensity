# 北太平洋热带气旋强度识别框架（PyTorch）

> 通过卫星云图识别**北太平洋（西北太平洋）热带气旋**的强度，支持「分类（6 级 CMA 强度）」
> 与「回归（最大持续风速）」两种任务。整个框架**以 YAML 配置为中心**，调参无需改代码。
>
> 当前已在 **TCIR（WPAC 西北太平洋）数据集**上完成训练与评测（图像→风速 `Vmax`，单位 knot）。

---

## 1. 环境要求（已在当前 dl_env 验证）

| 组件 | 版本 |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.11.0+cu128 |
| Torchvision | 0.26.0+cu128 |
| CUDA | 12.8（GPU 可用，RTX 5060） |

依赖见 `requirements.txt`。在 WSL 中通过 `dl_env` 环境的 python 运行（无需手动 `conda activate`）：

```bash
# 环境路径（当前机器）
/home/yzm/tf/dl-env/bin/python3.11 <xxx.py> ...
```

---

## 2. 目录结构

```
tc_intensity/
├── configs/                 # 配置（调参入口）
│   ├── default.yaml         # 默认配置，最全注解
│   ├── classification.yaml   # 预设：分类任务（resnet50）
│   ├── regression_wind.yaml  # 预设：回归风速任务
│   ├── tcir_regression.yaml  # 预设：TCIR 多通道卫星图风速回归
│   └── tcir_wpac_train.yaml  # ✅ WPAC 正式训练/评测配置（当前主线）
├── tcintens/               # 框架核心包
│   ├── data/               # 数据集 + 数据增强（含 tcir.py 加载器）
│   ├── models/             # 可插拔骨干(支持多通道) + 强度预测头
│   ├── engine/             # 训练 / 评估 / 推理（train/evaluate/predict）
│   └── utils/              # 配置 / 日志 / 指标 / 杂项
├── scripts/                # 数据工具 & 分析脚本
│   ├── make_synthetic.py   # 生成合成数据（无真实数据也能跑通）
│   ├── prepare_data.py     # 把真实数据整理成 labels.csv
│   ├── download_tcir.py    # 下载 TCIR 数据集（约 13GB）
│   ├── compute_tcir_stats.py  # 计算 TCIR 逐通道均值/标准差（归一化用）
│   ├── extract_wpac.py     # ✅ 抽取 WPAC 子集为紧凑 h5 + csv + 统计
│   ├── visualize_test.py   # ✅ 测试集预测可视化（面板 + 散点图）
│   └── error_analysis.py   # ✅ 完整测试集评估 + 误差最大 top-k 样本
├── tests/
│   ├── test_pipeline.py     # 通用框架冒烟测试
│   └── test_tcir.py        # TCIR 加载器 + 多通道训练端到端测试
├── train.py / evaluate.py / predict.py   # 命令行入口
├── train.sh                 # ✅ 一键训练脚本（默认 VGG16，40 轮）
├── requirements.txt
└── docs/
    ├── github_upload.md     # GitHub 上传与仓库管理方案
    └── tcir_training.md     # TCIR 数据集：加载/训练方法/标注说明
```

---

## 3. 快速开始（TL;DR）

想直接跑通当前 WPAC 主线，只需三步（数据已抽取好，无需碰 30GB 原文件）：

```bash
cd /home/yzm/tf/tc_intensity      # 进入项目根（换成你机器上的项目目录）

./train.sh                          # 训练 VGG16（自动用 dl_env 的 python）

python evaluate.py -c configs/tcir_wpac_train.yaml        # 测试集指标

python predict.py --h5 data/wpac_96.h5 --indices 10260 \
    -c configs/tcir_wpac_train.yaml                  # 单样本推理
```

更完整的命令与说明见 **[QUICKSTART.md](QUICKSTART.md)**。

---

## 4. TCIR 数据集与任务

TCIR（Chen, Chen & Lin, KDD 2018）是「卫星图 → 台风强度」回归的公开基准：4 通道卫星图
（IR1/WV/VIS/PMW，201×201，含 NaN），标签为 best-track 风速/风圈/海压。

- **是否需要标注：不需要。** 数据集已自带标签（来自 JTWC/HURDAT2 best-track），直接监督回归即可。
- 加载与训练方法、坑点见 **[docs/tcir_training.md](docs/tcir_training.md)**。

**当前 WPAC 主线配置（`configs/tcir_wpac_train.yaml`）**

| 项 | 值 |
|---|---|
| 区域 | WPAC（西北太平洋），20059 帧 |
| 通道 | `IR1 / WV / PMW`（3 通道，伪 RGB：IR1→R、WV→G、PMW→B） |
| 输入尺寸 | 96 × 96 |
| 任务 | 回归 `Vmax`（单位 **knot**） |
| 切分 | 按风暴 ID 防泄漏（70 / 15 / 15） |
| 归一化 | 逐通道 mean/std `[260.9, 232.8, 0.64]` / `[32.8, 15.4, 1.71]` |
| 骨干 | `vgg16`（默认）+ resnet18 等可插拔 |

紧凑数据文件（已生成，无需重新抽取）：

```
data/wpac_96.h5          # (20059, 96, 96, 3) float32，约 2.2GB
data/wpac_compact.csv    # 逐帧元数据：data_set, ID, lon, lat, time, Vmax, ..., matrix_index
data_tcir_wpac_stats.json# 逐通道 mean/std
```

### 4.1 数据抽取（一次性，已有则跳过）

原始 h5 约 30GB 且 WSL 跨挂载盘读很慢，先抽取为紧凑本地文件。**本项目数据已抽取好，通常无需重跑**。

```bash
# 1) 准备输入（二选一）：
#    a) 把原始文件拷到 data/：
#       cp "/mnt/e/TCIR-ATLN_EPAC_WPAC.h5/TCIR-ATLN_EPAC_WPAC.h5" data/TCIR.h5
#    b) 或直接把 --h5 指向原始文件绝对/相对路径
#    data/wpac_info.csv 是随原始数据提供的「info+matrix 整理表」
#    （列：matrix_index[原始帧号], Vmax, ID, data_set），作为抽取脚本的输入
python scripts/extract_wpac.py \
    --h5 data/TCIR.h5 \
    --csv data/wpac_info.csv \
    --out-h5 data/wpac_96.h5 \
    --out-csv data/wpac_compact.csv \
    --channels IR1 WV PMW --resize 96 --start 27000 \
    --stats-out data_tcir_wpac_stats.json
```
> 说明：`--start 27000` 是加速项——原始 matrix 前段是 ATL/EPAC 等无关帧，WPAC 帧的原始
> `matrix_index` 约从 27322 起，从 27000 开始读即可跳过前部；脚本仍按 `wpac_info.csv` 的行严格筛选，
> 不会误包含非 WPAC 帧。输出 `wpac_compact.csv` 的 `matrix_index` 改为**紧凑行号**（见上方路径约定）。

---

## 5. 训练（Train）

### 5.1 一键脚本（推荐）

`train.sh` 自动用 `dl_env` 的 python、按骨干自动命名实验目录、带前置检查：

```bash
./train.sh                  # 默认 vgg16，40 轮
./train.sh resnet18         # 换 resnet18
./train.sh vgg16 60         # vgg16，60 轮
./train.sh vgg16 40 train.lr=2e-4 train.batch_size=32   # 追加任意 --set 覆盖
```

### 5.2 直接入口

```bash
python train.py -c configs/tcir_wpac_train.yaml          # 用 config 默认（vgg16）

# 调参（不改文件，命令行覆盖）
python train.py -c configs/tcir_wpac_train.yaml --set train.epochs=60 train.lr=2e-4
```

输出默认进入 `outputs/tcir_wpac/tcir_wpac_v1/`（VGG16，config 默认实验名，自动存 `best.ckpt`、训练日志、`config_used.yaml`）；换其他骨干（如 `train.sh resnet18`）则进入 `outputs/tcir_wpac/tcir_wpac_<骨干名>/`，互相隔离不覆盖。

---

## 6. 测试（Test / 评估）

### 6.1 正式测试集指标（2913 样本）

```bash
python evaluate.py -c configs/tcir_wpac_train.yaml
```

> ⚠️ `evaluate.py` **必须带 `-c 训练配置`**；只传 `--set experiment.name=xxx` 会退回 `default.yaml`
> 的数据配置，导致加载错误数据（这是早期踩过的坑，已固化进框架）。
> 当前 `configs/tcir_wpac_train.yaml` 默认 `model.backbone=vgg16`、实验名 `tcir_wpac_v1`，
> 与 `outputs/tcir_wpac/tcir_wpac_v1/checkpoints/best.ckpt`（VGG16 权重）一致，直接运行即可。
> 指标写入 `outputs/tcir_wpac/tcir_wpac_v1/test_metrics.json`。

**当前 VGG16 测试集指标（N=2913）**：RMSE = 16.22 kn · MAE = 11.93 kn · R² = 0.7613。

### 6.2 误差最大样本 + 逐样本预测

```bash
python scripts/error_analysis.py -c configs/tcir_wpac_train.yaml
```

输出（写入 experiment 目录）：
- `test_predictions.csv`：全部 2913 样本的 真实/预测/误差
- `worst12.csv`：误差最大的 12 个样本（含 风暴ID / matrix_index / 真实Vmax / 预测Vmax / 误差）

> 规律：误差最大的样本几乎都是强台风（Vmax 98–155 kn）被**严重低估**，属回归「向均值回归」现象。

### 6.3 测试集可视化

```bash
python scripts/visualize_test.py -c configs/tcir_wpac_train.yaml
# 可选：--samples 16 --scatter-n 2913 --out outputs/my_viz.png
```

生成 `visualization.png`：上方为抽样台风伪 RGB 图（标题含 Pred/True Vmax 与 Δ），底部为
**Pred vs True 方形等距散点图**（x/y 同范围，`y=x` 为 45° 对角线）。

---

## 7. 推理（Predict）—— 适配 TCIR 新数据

`predict.py` 已改造为 **TCIR 专用推理入口**：支持三通道 `.npy` 帧（含通道选择、逐通道归一化、
resize），以及对紧凑 h5 按帧号直接推理；输出 **knots**（并换算 m/s）。

```bash
# 单张 .npy（H,W,C，通道顺序同配置 IR1/WV/PMW）
python predict.py --image frame.npy -c configs/tcir_wpac_train.yaml

# 文件夹批量 → CSV
python predict.py --folder ./frames/ -c configs/tcir_wpac_train.yaml --out preds.csv

# 直接对紧凑 h5 按「紧凑行号」推理（无需先导出 .npy）
python predict.py --h5 data/wpac_96.h5 --indices 10260,9603 \
    -c configs/tcir_wpac_train.yaml --out preds.csv
```

> 验证：对 `matrix[10260]` 推理得到 53.70 kn，与测试集评估完全一致。

---

## 8. 参数调节（核心特性）

**所有超参数都在 YAML 里**。改文件，或用命令行 `--set` 临时覆盖（类型自动推断）：

```bash
python train.py -c configs/tcir_wpac_train.yaml --set train.epochs=60 train.lr=1e-4
python train.py -c configs/tcir_wpac_train.yaml --set model.freeze_backbone=true   # 小数据迁移
python train.py --print-config          # 仅打印最终生效配置，不训练
```

常用调节项：

| 配置路径 | 含义 | 可选值 |
|---|---|---|
| `model.backbone` | 骨干网络 | resnet18/34/50/101, efficientnet_b0/b1, mobilenet_v3_small, vit_b_16, **vgg16** |
| `model.pretrained` | 是否用预训练权重 | true / false |
| `model.freeze_backbone` | 冻结骨干只训头部 | true / false |
| `model.in_channels` | 输入通道数 | 3 / 4（通道可配，骨干首层会自动替换） |
| `model.dropout` | 头部 dropout | 0.0 ~ 0.5 |
| `data.task` | 任务类型 | classification / regression |
| `train.optimizer` | 优化器 | adam / adamw / sgd |
| `train.scheduler` | 学习率调度 | none / step / cosine / plateau |
| `train.loss` | 损失 | mse / huber |
| `train.mixed_precision` | 混合精度(省显存) | true / false |
| `train.early_stopping_patience` | 早停耐心 | 整数 |
| `augmentation.train.*` | 各类增强开关 | 设为 false 即关闭 |
| `data.tcir.channels` | 通道组合 | 如 `["IR1","WV","PMW"]` |

---

## 9. 强度分级说明（分类任务）

采用中国气象局（CMA）西北太平洋分级，由弱到强：

| 等级 | 含义 | 最大风速 (m/s) |
|---|---|---|
| TD | 热带低压 | < 17.2 |
| TS | 热带风暴 | 17.2 – 24.4 |
| STS | 强热带风暴 | 24.5 – 32.6 |
| TY | 台风 | 32.7 – 41.4 |
| STY | 强台风 | 41.5 – 50.9 |
| SuperTY | 超强台风 | ≥ 51.0 |

---

## 10. 测试（单元 / 冒烟）

```bash
python tests/test_pipeline.py     # 生成合成数据并跑 2 个 epoch，断言指标优于随机
python tests/test_tcir.py         # TCIR 加载器 + 多通道训练端到端（合成 HDF5，无需真实数据）
```

---

## 11. 上传到 GitHub

仓库初始化、提交规范与远端推送步骤见 **[docs/github_upload.md](docs/github_upload.md)**。

---

## 12. 常见问题（FAQ）

**Q：`evaluate.py` / `predict.py` 报 `KeyError` 或模型加载失败？**
A：最常见是「骨干不匹配」或「实验目录不对」。当前 `configs/tcir_wpac_train.yaml`
默认 `model.backbone=vgg16`，对应 `outputs/tcir_wpac/tcir_wpac_v1/checkpoints/best.ckpt`（VGG16 权重），
直接用 `-c configs/tcir_wpac_train.yaml` 即可。若你用 `train.sh resnet18` 训了新骨干，权重在
`outputs/tcir_wpac_resnet18/tcir_wpac_resnet18/`，需用 `-c` 同配置并 `--set model.backbone=resnet18`、
`experiment.name=tcir_wpac_resnet18` `experiment.output_dir=outputs/tcir_wpac_resnet18` 指向它。

**Q：想用 4 通道（含 VIS）？**
A：训练时 `--set model.in_channels=4 data.tcir.channels=["IR1","WV","VIS","PMW"]`，
骨干首层会自动按通道数替换；归一化统计需用 `compute_tcir_stats.py` 重新算。

**Q：强台风被严重低估？**
A：属回归向均值回归。可尝试对 Vmax 做 log 变换、分段/分位数损失，或高强段数据增广。

---

## 13. 术语表（专业名词解释）

| 术语 | 解释 |
|---|---|
| **TCIR** | *Dataset of Tropical Cyclone for Image-to-intensity Regression*（Chen, Chen & Lin, KDD 2018）：卫星图→台风强度的公开基准数据集，自带标签、无需人工标注。 |
| **WPAC / EPAC / ATL / SH** | 洋区缩写：西北太平洋 / 东太平洋 / 大西洋 / 南半球。本项目主线用 WPAC。 |
| **IR1 / WV / VIS / PMW** | 卫星通道：红外 1 / 水汽 / 可见光 / 被动微波。VIS 夜间缺失、白天不稳定，默认弃用；本项目用 IR1+WV+PMW 三通道。 |
| **Vmax** | 最大持续风速（Maximum Sustained Wind），本项目回归目标，单位 **knot**（节）。 |
| **knot / kn** | 节（海里/小时），航海与气象风速单位；1 kn ≈ 0.514 m/s。 |
| **best-track** | 事后由 JTWC/HURDAT2 等机构定出的「准真值」强度序列，含标注噪声（约 10 kn），被当作监督标签。 |
| **by_storm 防泄漏切分** | 按「风暴 ID」而非随机按「帧」切分 train/val/test，确保同一台风的连续帧只进一个集合，避免测试集偷看训练集、指标虚高。 |
| **归一化（逐通道 mean/std）** | 对每个通道用其均值/标准差标准化，使量纲一致；本项目用 TCIR 各通道统计，而非 ImageNet 的均值。 |
| **伪 RGB / 合成图** | 把多通道（IR1→R、WV→G、PMW→B）逐通道拉伸后拼成的可视化彩色图，仅用于展示，不参与训练。 |
| **混合精度 / AMP** | Automatic Mixed Precision：计算用 float16、关键累加用 float32，省显存、提速，精度几乎不掉。 |
| **RMSE / MAE / R²** | 回归指标：均方根误差 / 平均绝对误差 / 决定系数（越接近 1 越好）。 |
| **HDF5** | 一种支持超大数组的分层二进制格式（`.h5`），TCIR 用它存 matrix(N×201×201×4) 与 info 表。 |
| **dropout / 预训练骨干 / 早停 / Top-K / huber** | 训练组件：随机置零防过拟合 / 用 ImageNet 等预训练权重初始化 / 验证集不降则停 / 保留最优 K 个权重 / 对离群更稳健的损失。 |
| **回归向均值回归** | 模型在极端值（如强台风）上系统性偏向中间值的现象，是本项目强台风被低估的根因。 |
