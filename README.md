# 北太平洋热带气旋强度识别框架（PyTorch）

> 通过卫星云图识别**北太平洋（西北太平洋）热带气旋**的强度，支持「分类（6 级 CMA 强度）」
> 与「回归（最大持续风速 m/s）」两种任务。整个框架**以 YAML 配置为中心**，调参无需改代码。

---

## 1. 环境要求（已在当前 dl_env 验证）

| 组件 | 版本 |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.11.0+cu128 |
| Torchvision | 0.26.0+cu128 |
| CUDA | 12.8（GPU 可用） |

依赖见 `requirements.txt`。在 WSL 中通过前缀 conda 环境 `dl_env` 运行：

```bash
# 假设环境装在 /home/yzm/tf/dl-env
/home/yzm/tf/dl-env/bin/python3.11 train.py --config configs/default.yaml
```

---

## 2. 目录结构

```
tc_intensity/
├── configs/                 # 配置（调参入口）
│   ├── default.yaml         # 默认配置，最全注解
│   ├── classification.yaml   # 预设：分类任务（resnet50）
│   ├── regression_wind.yaml  # 预设：回归风速任务
│   └── tcir_regression.yaml  # 预设：TCIR 多通道卫星图风速回归
├── tcintens/               # 框架核心包
│   ├── data/               # 数据集 + 数据增强（含 tcir.py 加载器）
│   ├── models/             # 可插拔骨干(支持多通道) + 强度预测头
│   ├── engine/             # 训练 / 评估 / 推理
│   └── utils/              # 配置 / 日志 / 指标 / 杂项
├── scripts/                # 数据工具
│   ├── make_synthetic.py   # 生成合成数据（无真实数据也能跑通）
│   ├── prepare_data.py     # 把真实数据整理成 labels.csv
│   ├── download_tcir.py    # 下载 TCIR 数据集（约 13GB）
│   └── compute_tcir_stats.py  # 计算 TCIR 逐通道均值/标准差（归一化用）
├── tests/
│   ├── test_pipeline.py     # 通用框架冒烟测试
│   └── test_tcir.py        # TCIR 加载器 + 多通道训练端到端测试
├── train.py / evaluate.py / predict.py   # 命令行入口
├── requirements.txt
└── docs/
    ├── github_upload.md     # GitHub 上传与仓库管理方案
    └── tcir_training.md     # TCIR 数据集：加载/训练方法/标注说明
```

---

## 3. 快速开始

### 3.1 用合成数据一键验证（无需真实数据）

```bash
python scripts/make_synthetic.py --out-dir data/synthetic --per-class 60
python train.py --config configs/default.yaml
python evaluate.py --set experiment.name=tc_intensity_baseline
python predict.py --image data/synthetic/images/TY/TY_0000.png --set experiment.name=tc_intensity_baseline
```

### 3.2 接入你自己的数据

将数据整理为以下两种格式之一（详见 `docs/github_upload.md` 及 `scripts/prepare_data.py`）：

- **CSV 模式**（推荐）：一个 `labels.csv`，列含 `image_path,label,wind_speed_ms`。
- **文件夹模式**：`data/processed/<类别名>/*.png`，自动分层切分。

```bash
# 例：把 data/raw/<类别>/ 整理成 labels.csv
python scripts/prepare_data.py --src data/raw --out data/labels.csv
python train.py --set data.label_file=data/labels.csv
```

---

## 4. TCIR 官方卫星数据集（image-to-intensity 回归）

TCIR（Chen, Chen & Lin, KDD 2018）是「卫星图 → 台风强度」回归的公开基准：4 通道卫星图
（IR1/WV/VIS/PMW，201×201，含 NaN），标签为 best-track 风速/风圈/海压。

- **是否需要标注：不需要。** 数据集已自带标签（来自 JTWC/HURDAT2 best-track），直接监督回归即可。
- 加载与训练方法、坑点见 **[docs/tcir_training.md](docs/tcir_training.md)**。
- 专属配置 `configs/tcir_regression.yaml`；下载脚本 `scripts/download_tcir.py`。

```bash
python scripts/download_tcir.py --out data/
python train.py -c configs/tcir_regression.yaml \
    --set data.tcir.h5_path=data/TCIR-ATLN_EPAC_WPAC.h5
```

---

## 5. 参数调节（核心特性）

**所有超参数都在 YAML 里**。改文件，或用命令行 `--set` 临时覆盖（类型自动推断）：

```bash
# 换更大的骨干
python train.py --set model.backbone=resnet50

# 调学习率 / 批次 / 任务
python train.py --set train.lr=1e-4 train.batch_size=64 data.task=regression

# 冻结骨干做小数据集迁移学习
python train.py --set model.freeze_backbone=true

# 先只看最终生效配置，不训练
python train.py --print-config
```

常用调节项一览：

| 配置路径 | 含义 | 可选值 |
|---|---|---|
| `model.backbone` | 骨干网络 | resnet18/34/50/101, efficientnet_b0/b1, mobilenet_v3_small, vit_b_16 |
| `model.pretrained` | 是否用预训练权重 | true / false |
| `model.freeze_backbone` | 冻结骨干只训头部 | true / false |
| `model.dropout` | 头部 dropout | 0.0 ~ 0.5 |
| `data.task` | 任务类型 | classification / regression |
| `data.image_size` | 输入尺寸 | 如 224 |
| `train.optimizer` | 优化器 | adam / adamw / sgd |
| `train.scheduler` | 学习率调度 | none / step / cosine / plateau |
| `train.mixed_precision` | 混合精度(省显存) | true / false |
| `train.early_stopping_patience` | 早停耐心 | 整数 |
| `augmentation.train.*` | 各类增强开关 | 设为 false 即关闭 |

---

## 5. 强度分级说明（分类任务）

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

## 6. 测试

```bash
python tests/test_pipeline.py     # 生成合成数据并跑 2 个 epoch，断言指标优于随机
python tests/test_tcir.py         # TCIR 加载器 + 多通道训练端到端（合成 HDF5，无需真实数据）
```

---

## 6. 上传到 GitHub

仓库初始化、提交规范与远端推送步骤见 **[docs/github_upload.md](docs/github_upload.md)**。
