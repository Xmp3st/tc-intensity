# TCIR 数据集：加载、训练方法与标注说明

TCIR = *Dataset of Tropical Cyclone for Image-to-intensity Regression*（Chen, Chen & Lin, KDD 2018）。
官方页：<https://www.csie.ntu.edu.tw/~htlin/program/TCIR/> ｜ 代码镜像：<https://github.com/BoyoChen/TCIR>

本框架已内置 TCIR 加载器（`tcintens/data/tcir.py`）与专用配置 `configs/tcir_regression.yaml`，
可直接做「卫星图 → 最大持续风速」回归。

---

## 1. 数据长什么样

| 项目 | 内容 |
|---|---|
| 输入 | 4 通道卫星图，**201×201**，台风眼居中，半径约 7°（网格约 4 km） |
| 通道 | `IR1`(红外) / `WV`(水汽) / `VIS`(可见光) / `PMW`(被动微波) |
| 标签 | `intensity`(最大持续风速, **knot**, 主目标) / `size`(35kt 风圈半径均值, nmi) / `pres`(最低海压, hPa) / 中心经纬度 |
| 规模 | 6 区域共 **1285 个台风 / 70501 帧**（ATL 13707、WPAC 20061、EPAC 13615、SH 18434 …） |
| 格式 | 单一 HDF5，两个 key：`matrix`(N×201×201×4, 含 NaN) + `info`(pandas 表) |
| 标签来源 | JTWC best-track(WPAC) / HURDAT2(EPAC, ATL)，2003–2016 |

**NaN**：部分网格缺测，官方建议「插值」或「置 0」处理。

---

## 2. 是否需要人工标注？—— 不需要 ✅

TCIR 是**有监督回归**数据集，**标签已经在发布包里**：

- 标签（`intensity/size/pres/位置`）由 JTWC / HURDAT2 的 **best-track** 事后定强给出，
  不是实时估算，被学界当作「准真值（ground truth）」使用。
- 你**不需要**做任何人工标框/标级；只要把 `data.tcir.target` 指到 `intensity`（默认）即可。
- 唯一要注意的：best-track 本身仍有噪声（强度误差约 10 knot 内视为可接受），
  所以**不要期待模型超过这个标注噪声下限**。

> 一句话：**这个数据集开箱即用、完全标注，直接训练即可，无需标注。**

---

## 3. 如何加载（load）这份数据

### 3.1 下载（约 13 GB，分 3 个分卷）

```bash
# 方式 A：脚本一键下载 + 解压（NTU 镜像）
python scripts/download_tcir.py --out data/

# 方式 B：Google Drive 镜像
# https://drive.google.com/drive/folders/19Jo9iLi2b5qFoPan78SyojBxDHr-egJw

# 方式 C：手动 wget（三个文件任选，按区域）
wget https://learner.csie.ntu.edu.tw/~boyochen/TCIR/TCIR-ATLN_EPAC_WPAC.h5.tar.gz
wget https://learner.csie.ntu.edu.tw/~boyochen/TCIR/TCIR-CPAC_IO_SH.h5.tar.gz
wget https://learner.csie.ntu.edu.tw/~boyochen/TCIR/TCIR-ALL_2017.h5.tar.gz
tar -xzf TCIR-*.tar.gz -C data/
```

### 3.2 用官方方式读（参考）

```python
import numpy as np, pandas as pd, h5py
data_info = pd.read_hdf("data/TCIR.h5", key="info", mode="r")   # 标签表
with h5py.File("data/TCIR.h5", "r") as hf:
    data_matrix = hf["matrix"][:]      # 注意：一次性读全会占数十 GB 内存！
```

### 3.3 用本框架加载器（推荐，懒加载）

本框架**不一次性载入 `matrix`**，而是按样本索引从 HDF5 切片读取，配合 DataLoader 多进程也安全：

```python
from tcintens.data.tcir import TCIRDataset, create_tcir_datasets
from tcintens.utils.config import load_config

cfg = load_config("configs/tcir_regression.yaml")   # 或直接用 dict
datasets, meta = create_tcir_datasets(cfg)
# datasets["train"/"val"/"test"] 即标准 torch Dataset，返回 (Tensor(C,H,W), float风速)
x, y = datasets["train"][0]
print(x.shape, y)        # torch.Size([3, 128, 128])  <风速(knot)>
```

直接训练：

```bash
python train.py -c configs/tcir_regression.yaml \
    --set data.tcir.h5_path=data/TCIR-ATLN_EPAC_WPAC.h5
```

> 真实数据读取 `info` 表优先用 `pd.read_hdf`（需 `pytables`：`pip install tables`）；
> 若未安装，加载器会自动回退到纯 h5py 解析，仍能读出标签。

---

## 4. 训练方法（怎么训最好）

### 4.1 任务定义
**图像→强度回归**：输入多通道卫星图，输出最大持续风速（knot / 也可换算 m/s）。
可用 `data.tcir.target` 切换到 `size`(风圈半径) 或 `pres`(海压) 任务。

### 4.2 官方基线：Rotation-blended CNN（R-CNN, KDD 2018）
论文的核心思路是**利用台风的旋转等变性**：

1. 对每张输入做 0°/90°/180°/270° 旋转，训练 4 个结构相同的 CNN；
2. 推理时把预测旋转回原方位再**平均**，等价于「旋转集成」；
3. 这样模型不必死记某个朝向，显著降误差。

本框架用「可插拔骨干(resnet 等) + 回归头」实现**单模型强基线**，
并通过 `augmentation.train.rotate` 做旋转增强、`horizontal_flip` 做翻转增强，
逼近 R-CNN 的旋转鲁棒性思想（要完全复现 R-CNN 可再做 4 向推理平均）。

### 4.3 预处理要点（务必做）

| 步骤 | 建议 | 本框架对应配置 |
|---|---|---|
| 通道选择 | 默认 `IR1+WV+PMW`，**弃用 VIS**（白天极不稳定、夜间缺失） | `data.tcir.channels` |
| 缺失值 | `zero`(置0) 或 `interp`(按行线性插值) | `data.tcir.nan_mode` |
| 归一化 | 卫星通道尺度差异大，**不要用 ImageNet 的 mean/std**；用 `scripts/compute_tcir_stats.py` 算逐通道 mean/std 后归一化 | `data.tcir.normalize` + `norm_mean/norm_std` |
| 尺寸 | 201 太大可降到 128/96 降算力 | `data.tcir.resize` |
| 南半球 | SH 台风**旋转方向相反**，若混训要注意（建议分区域训或分区域测试） | `data.tcir.regions` |

### 4.4 切分防泄漏（关键！）

同一台风的连续帧高度相关。**绝不能随机按帧切分**，否则验证集会「偷看」训练集。
本框架默认**按风暴 ID（`info` 的 `id` 列）分组切分** train/val/test：

```yaml
data:
  tcir:
    split_strategy: "by_storm"   # 同一台风的帧只会出现在一个集合
    storm_col: "id"
```

### 4.5 损失与指标

- 损失：`mse`（默认）或 `huber`（对离群/标注噪声更稳）— `train.loss`
- 指标：RMSE / MAE / R² 自动计算并打印（回归任务的早停 monitor 为 −RMSE）

### 4.6 训练技巧

- **预训练骨干 + 微调**：`model.pretrained=true`，小学习率（`lr≈1e-3`）起手；
  样本少可 `model.freeze_backbone=true` 先只训头部。
- **混合精度**：`train.mixed_precision=true`（当前 RTX 5060 + CUDA 12.8 已验证可用，省显存提速）。
- **早停 + Top-K 权重**：`train.early_stopping_patience` / `save_top_k`。
- **关注区域**：你重点在北太平洋，可用
  `data.tcir.regions: ["WPAC","EPAC","ATL"]` 只取这些区域训练/评估。

---

## 5. 端到端最小示例（已在本框架验证）

```bash
# 1) 准备数据（真实数据需先下载，见 §3.1）
python scripts/download_tcir.py --out data/

# 2) （可选）算归一化统计
python scripts/compute_tcir_stats.py \
    --h5 data/TCIR-ATLN_EPAC_WPAC.h5 --channels IR1 WV PMW \
    --out data/tcir_stats.json

# 3) 训练（改 h5_path 指向你的文件）
python train.py -c configs/tcir_regression.yaml \
    --set data.tcir.h5_path=data/TCIR-ATLN_EPAC_WPAC.h5

# 4) 评估 / 推理（务必带 -c，否则退回 default.yaml 加载错误数据）
python evaluate.py -c configs/tcir_regression.yaml
python predict.py --image <某帧.npy> -c configs/tcir_regression.yaml
# 或直接对 h5 按帧号推理：
python predict.py --h5 data/TCIR-ATLN_EPAC_WPAC.h5 --indices 0 -c configs/tcir_regression.yaml
```

> 无真实数据时，本仓库 `tests/test_tcir.py` 会用**同 schema 的合成 HDF5**
> （含 NaN、含 info 表）验证：懒加载、NaN 处理、按风暴防泄漏切分、
> 多通道模型前向+反向、**完整训练引擎跑通**。可直接：
> ```bash
> python tests/test_tcir.py
> ```

---

## 6. 常见坑

- **内存爆炸**：别用 `hf["matrix"][:]` 一次性读全量（70k×201×201×4×4B ≈ 数十 GB）。本加载器按索引切片。
- **VIS 噪声**：白天正常、夜间几乎全 NaN，建议默认弃用。
- **标注噪声下限**：best-track 强度误差约 10 knot，模型 RMSE 难以长期低于此。
- **SH 旋转**：南半球气旋旋转方向相反，混训/评估需小心。
- **单位**：TCIR `intensity` 是 **knot**；若想用 m/s，设 `data.tcir.label_unit: "ms"`（自动 ×0.514444）。

> 专业术语（TCIR / WPAC / Vmax / best-track / by_storm 防泄漏 / knot / 混合精度 / RMSE 等）
> 见 [README.md §13 术语表](../README.md#13-术语表)。
