# 泄漏对照实验：无损坏子集 · resnet50 · 8:2 划分

> 实验日期：2026-10-09 ｜ 目的：**仅作测试**——量化「随机划分（不防泄露）」相对
> 「按风暴划分（防泄露）」会把测试指标抬高多少。
> 数据：`data/wpac_clean.csv`（`data/wpac_96.h5` 中**完全无损坏**的帧，见 §2）。
> 骨干：resnet50（沿用 `scripts/train_backbones.py` 选出的最优骨干）。
> 训练：40 轮、batch 64、AdamW 1e-3 + cosine、AMP；**无验证集**（`data.split=[0.8,0,0.2]`），
> `best.ckpt` 取最后一轮；`missing_band=none`（输入本就无缺测，无需补全）。
> 3 个 seed（42/43/44），`split_seed` 固定 42（各 seed 只变初始化与增强，划分不变 → 可配对）。

---

## 1. 结论速览

在**同一批 15986 帧无损坏图像**上，仅把划分方式从「按风暴（防泄露）」换成「随机（不防泄露）」，
测试集 RMSE 从 **14.770 → 8.495**（**−6.275 kn，配对 t = −98.1，p = 1e-4**），
R² 从 **0.808 → 0.931**，MAE 从 **10.99 → 6.00**。

> **随机划分把测试指标抬高了约 42.5%（以 RMSE 相对降幅计）。**
> 这是「同一批数据、只换划分」的受控对比，因此这 6.3 kn 反映了**泄漏本身的量级**
> （严格说应视为**上界**，理由见 §3 第 2 点）。

对论文/汇报的直接含义：**任何不做风暴级分组的随机划分结果都不可信**——
本项目坚持 `split_strategy=by_storm` 是必要的（`split_seed=42` 固定），
否则会把模型的真实泛化能力高估一倍。

---

## 2. 「完全无损坏」子集怎么来的

脚本 `scripts/make_clean_subset.py` 扫描全部 20059 帧，用项目自身的损坏检测参数
（sentinel=0、`low_thr=[150,150,-999]`、`min_blob=20`、`dilate=1`）统计每帧「坏道数」。

| 口径 | 定义 | 帧数 | 占比 | 风暴数 | Vmax 均值 |
|---|---|---|---|---|---|
| **project（本次采用）** | 坏道数 == 0（IR1/WV 无 ≥20px 缺测连通块，且三通道无整通道缺失） | **15986** | 79.7% | 379 | 55.82 |
| strict | IR1/WV 逐像素零缺测（`min_blob=1, dilate=0`） | 14830 | 73.9% | 377 | 55.77 |
| 全量 | — | 20059 | 100% | 379 | 54.84 |

坏道数分布：0 道 15986 帧、1 道 3981 帧、2 道 92 帧 —— 即约 **20.3% 的帧带缺测**，
与既往记录的「IR1 缺测覆盖 19.8% 帧」一致。

统计明细：`logs/clean_subset_stats.json`。

---

## 3. 两种划分的「泄漏解剖」

同为 8:2（`data.split=[0.8, 0.0, 0.2]`）、同为上表 15986 帧，只改 `split_strategy`：

| 划分 | n_train | n_test | 测试帧中「训练期见过该风暴」比例 | 每个测试风暴在训练集的曝光帧数（均值/中位/最大） | 训练/测试 Vmax 均值 |
|---|---|---|---|---|---|
| **random（不防泄露）** | 12789 | 3197 | **99.97%** | **34.2 / 30.5 / 119** | 55.93 / 55.40 |
| **by_storm（防泄露）** | 12821 | 3165 | **0%** | 0 / 0 / 0 | 54.97 / **59.28** |

两点值得注意：

1. 随机划分下，几乎每一个测试帧所属的风暴都在训练集中出现过，且平均曝光 **34 帧**
   —— 模型有充分机会记住「这台风暴长什么样、大约多强」，这就是泄漏的具体形态。
2. 两组**测试集不是同一批数据**：by_storm 的测试风暴更强（59.28 vs 55.40 kn），
   因此两组 RMSE 之差同时含「泄漏」与「轻微分布偏移」两个成分，方向一致
   —— 即该差值应视为泄漏效应的**上界估计**。

明细：`logs/clean82_anatomy.json`。

---

## 4. 结果

![泄漏对照](../outputs/tcir_wpac/clean82_compare.png)

### 4.1 逐 run

| 臂 | seed | 测试 RMSE | MAE | R² |
|---|---|---|---|---|
| **random（不防泄露）** | 42 | **8.450** | 5.999 | 0.9318 |
| | 43 | **8.591** | 6.098 | 0.9295 |
| | 44 | **8.444** | 5.892 | 0.9319 |
| **by_storm（防泄露）** | 42 | **14.849** | 11.054 | 0.8062 |
| | 43 | **14.775** | 10.981 | 0.8081 |
| | 44 | **14.686** | 10.923 | 0.8104 |

### 4.2 均值与配对差

| 臂 | RMSE 均值 | MAE 均值 | R² 均值 |
|---|---|---|---|
| random（不防泄露） | **8.495 ± 0.083** | **5.997** | **0.9311** |
| by_storm（防泄露） | **14.770 ± 0.081** | **10.986** | **0.8082** |
| 差（random − by_storm） | **−6.275** | **−4.989** | **+0.1229** |

配对（3 seed，split_seed 固定 42）：

| seed | Δ RMSE |
|---|---|
| 42 | −6.398 |
| 43 | −6.184 |
| 44 | −6.242 |

**配对 t = −98.07（df=2，p = 1.0e-4）**，3/3 同号且极稳（各 seed 内 σ 仅 0.08）。
random / by_storm 的 RMSE 之比 = **0.575**，即随机划分把误差压掉 **42.5%**。

### 4.3 读数

1. **泄漏效应量级巨大且高度稳定**：6.3 kn 的差距远超 seed 噪声（±0.08），
   3 个 seed 的 Δ 落在 −6.18 ~ −6.40 的窄带内。这不是运气，是划分本身带来的系统性虚高。
2. **R² 从 0.808 虚增到 0.931**：如果只看 R²，会得出「模型已接近可用」的错误结论。
   本次对照说明**这类 R² 0.93 的结果在同一批数据上完全可以靠泄漏刷出来**。
3. **泄漏的机制不是「测试帧本身进过训练」**（`frame_overlap = 0`，逐帧无重叠），
   而是**测试帧所属的风暴被模型见过**（99.97% 的测试帧、平均 34 帧/风暴曝光）。
   换句话说泄漏发生在**风暴身份/个体强度**这一层，模型记住了「这台风暴大概多强」。
4. **两组测试集并非同一批数据**，这使结论更保守而不是更激进：
   by_storm 的测试风暴平均更强（59.28 vs 55.40 kn）、数量更少（76 vs 372 台），
   按既往结论「强台风更难预测」，by_storm 本就吃亏 ⇒ 6.3 kn 应视为
   **泄漏效应的上界估计**（含少量「测试集更难」的成分）。
5. **两组训练分布几乎相同**（train Vmax 55.93 vs 54.97，std 32.8 vs 32.4），
   训练算量也相当（12789 vs 12821 帧），因此差距只能归因于划分方式。

### 4.4 与前作口径的关系（勿直接比较）

本实验的 by_storm 臂 = **14.770**，高于既有的 `bb_resnet50_s42` = **13.384**，
但这**不代表 resnet50 变差了**——两者协议不同：

| 项 | `bb_resnet50_s42`（既有） | 本实验 by_storm 臂 |
|---|---|---|
| 训练数据 | 全部 20059 帧（含缺测，`missing_band=interp`） | 仅 15986 帧无损坏 |
| 验证集 | 有（10.7%，早停选模） | **无**（`best` = 最后一轮，40 轮固定） |
| 测试集 | 2893 帧（含缺测，损坏过滤后） | 3165 帧（**不含缺测**，76 台未见风暴） |
| 测试集 Vmax 均值 | ≈54 | **59.28** |

即：本实验的测试集**换了数据、换了风暴、并去掉了验证集早停**。若要在同一口径下比较
backbone，仍应以 `docs/backbone_compare.md` 为准；本实验只回答「划分方式的影响」。

---

## 5. 复现

```bash
P=/home/yzm/tf/dl-env/bin/python3.11

# 1) 生成无损坏子集 CSV
$P scripts/make_clean_subset.py

# 2) 训练 + 测试（6 组 = 2 臂 × 3 seed）
for ARM in leak bystorm; do
  if [ "$ARM" = "leak" ]; then STRAT=random; else STRAT=by_storm; fi
  for S in 42 43 44; do
    $P tcir.py train -c configs/tcir_wpac_train.yaml \
       --set data.tcir.csv_path=data/wpac_clean.csv \
       --set data.tcir.split_strategy=$STRAT \
       --set data.split=[0.8,0.0,0.2] \
       --set model.backbone=resnet50 \
       --set train.epochs=40 \
       --set experiment.seed=$S \
       --set experiment.name=tcir_clean82_${ARM}_s$S
    $P tcir.py eval -c configs/tcir_wpac_train.yaml \
       --set model.backbone=resnet50 --set experiment.name=tcir_clean82_${ARM}_s$S
  done
done

# 3) 汇总
$P scripts/clean82_summary.py
```

⚠️ 两个坑：
- `eval` **必须带 `--set model.backbone=resnet50`**：`align_preprocessing` 只对齐预处理/
  划分类键，不含 `model.backbone`，否则会用默认 vgg16 去加载 resnet 权重报
  `Error(s) in loading state_dict`。
- `--set data.tcir.missing_band=none` **不能写**：`_coerce` 会把字符串 `none` 解析成
  Python `None`，导致 `missing_band != "none"` 分支被误触发。用配置默认的 `"none"` 即可。

---

## 6. 产物

| 路径 | 说明 |
|---|---|
| `data/wpac_clean.csv` / `wpac_clean_strict.csv` | 无损坏子集（project / strict 口径） |
| `logs/clean_subset_stats.json` | 子集规模与标签统计 |
| `logs/clean82_anatomy.json` | 泄漏解剖（曝光量、标签分布） |
| `outputs/tcir_wpac/tcir_clean82_{leak,bystorm}_s{42,43,44}/` | 6 组训练产物（含 `config_used.yaml`） |
| `outputs/tcir_wpac/clean82_compare.{png,csv,json}` | 汇总图与配对统计 |
| `scripts/make_clean_subset.py` / `clean82_check.py` / `clean82_anatomy.py` / `clean82_summary.py` | 本实验脚本 |
