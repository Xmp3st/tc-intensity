# 多 backbone 对比实验（96×96×3 · WPAC · Vmax 回归）

> 目的：在**完全相同的输入、预处理、数据划分与超参**下，横向比较不同骨干网络，
> 为主干选型提供依据。
>
> 内容：§1–§4 骨干对比；**§5 输入级消融 raw vs corrected（4 骨干 × 2 补全方法 = 8 组）**；
> §6 复现；§7 产物。

## 1. 实验协议

| 项目 | 取值 |
|---|---|
| 数据 | `data/wpac_96.h5` (20059, 96, 96, 3)，通道 **IR1 / WV / PMW** |
| 标签 | `data/wpac_compact.csv` 的 `Vmax`（TCIR/JTWC，1-min 持续风速，knots） |
| 划分 | `split_strategy=by_storm`，**`split_seed=42`**（按风暴 ID 防泄漏） |
| 划分规模 | 损坏过滤后 train / val / test = **14062 / 3008 / 2893** |
| 缺失修正 | `missing_band=interp`（IR1/WV 区域插值，`low_thr=150K`，`min_blob=20`，`dilate=1`） |
| 损坏过滤 | 坏道数≥2 ｜ 全图缺测面积≥1/2 ｜ 单道≥2/3，三者 OR，作用于 train/val/test |
| 归一化 | 逐通道 `mean=[260.864, 232.83, 0.638]`、`std=[32.849, 15.389, 1.71]` |
| 增强 | 水平翻转 + 旋转 ±15° |
| 训练 | 40 epochs、早停 patience=10、batch 64、AdamW lr 1e-3、weight_decay 1e-2、cosine、MSE、AMP(bf16) |
| 头部 | `Linear(nf, 64) → ReLU → Dropout(0.3) → Linear(64, 1)`，`freeze_backbone=false` |
| 唯一变量 | `model.backbone` |

**划分为何固定**：train/val/test 的风暴归属只由 `data.tcir.split_seed=42` 决定；
`experiment.seed` 仅影响权重初始化和数据加载顺序。因此换 backbone、换 seed 都不会挪动
任何一张图所属的子集。各 run 的 `config_used.yaml` 中 `split_seed` 均为 42。

## 2. 结果

以**测试集**为准（验证集仅用于早停与选模型，不与测试集混用）：

| 骨干 | 参数量 | best epoch | Val RMSE | Val MAE | Val R² | **Test RMSE** | Test MAE | Test R² | 耗时 |
|---|---|---|---|---|---|---|---|---|---|
| **resnet50** | 23.6 M | 26 | 12.871 | 9.587 | 0.8463 | **13.384** | 10.182 | 0.8365 | 48 min |
| resnet18 | 11.2 M | 17 | 13.126 | 9.701 | 0.8401 | 14.478 | 10.830 | 0.8087 | 37 min |
| efficientnet_b0 | 4.1 M | 20 | 13.450 | 9.715 | 0.8321 | 14.145 | 10.546 | 0.8174 | 41 min |
| resnet34 | 21.3 M | 7 | 13.722 | 10.299 | 0.8253 | 14.245 | 10.702 | 0.8148 | 23 min |
| vgg16 \* | 138.4 M | 27 | 13.065 | 9.794 | 0.8416 | 13.539 | 10.293 | 0.8330 | 66 min |

\* **vgg16 为参考基线，口径不同**：来自已有 `outputs/tcir_wpac/tcir_wpac_v2_3x/`
（**80 轮 + patience 20**），并非本轮 40 轮协议下重跑。数据划分与预处理完全相同，
仅轮数/耐心不同，比较时须注意。

![backbone comparison](../outputs/tcir_wpac/tcir_wpac_bb/backbone_compare.png)

## 3. 结论

1. **resnet50 是本轮最优**：Test RMSE **13.384 kn**，比 vgg16 参考基线（13.539，但轮数是 2 倍）
   低 **0.155 kn**，而参数量只有其 **1/5.9**。R² 0.8365 也是本轮最高。
2. **轻量骨干性价比突出**：`efficientnet_b0` 仅 **4.1 M** 参数，Test RMSE 14.145 kn，
   与 21.3 M 的 resnet34（14.245）相当、且优于 11.2 M 的 resnet18（14.478）。
   若部署算力受限，它是首选。
3. **同族内并非越深越好**：resnet34（21.3 M）反而略差于 resnet50，且 best epoch 只有 7，
   早停触发得太早 —— 说明它在当前 lr/增强下收敛不稳（val 曲线在 13.7–16.8 间反复跳动），
   不排除是**超参对该网络不最优**，而非容量不足。
4. **resnet18 的 val→test 落差异常**：val 13.126（本轮最好之一）但 test 14.478，
   落差 **+1.35 kn**，明显大于其余模型的 +0.5~0.7 kn。这说明其 val 表现有一定偶然性，
   用 val 选模型时对这个网络要更谨慎。此结论已通过**独立干净重跑复核**：重跑后的
   val 13.1259 / test 14.4778 与原结果**逐位一致**，落差是模型固有特性，非中断所致。
5. **都触发了早停**：4 个模型分别在 **27 / 17 / 36 / 30** 轮停下（顺序同上表：
   resnet18 / resnet34 / resnet50 / efficientnet_b0），均未跑满 40 轮，
   说明 40 轮上限对这批骨干是充裕的。

## 4. 注意事项（可比性）

- **vgg16 轮数口径不同**（80 轮 vs 40 轮）。若需要严格同轮数对比，可用
  `--backbones vgg16 --epochs 40 --patience 10` 补跑一次（约 20 min）。
- **resnet18 已做独立干净重跑复核**：首轮训练在 epoch 19 时因 WSL 重启被杀，曾从
  `last.ckpt` 断点续训完成。为排除续训干扰，已删除该目录并**从零重跑一次**
  （`--no-resume`，40 轮 + patience 10，27 轮早停）。重跑结果与续训结果**完全一致**
  （val RMSE 13.1259、test RMSE 14.4778），确认最终选中的 epoch 17 权重未受中断影响，
  表中数据以这次干净重跑为准。旧目录指标留存于 `logs/backup_bb/resnet18_v1_interrupted_*`。
- **ViT-B/16 未参与**：torchvision 0.26 强制要求 224×224 输入（96² 直接断言失败），
  要跑只能改用 224 输入，协议就不再一致了。

## 5. 输入级消融：raw vs corrected（4 骨干 × 2 补全方法 = 8 组）

**目的**：论文 4.4 节已有 vgg16 的输入级消融（原始花屏 raw → 插值修正 corrected）。
本节把该消融扩展到本轮的 4 个新骨干，并把「修正」这一侧取**两种补全方法**，
得到 **4 模型 × 2 方法 = 8 组** raw vs corrected 对比。

### 5.1 协议

同一权重、同一批测试样本，**只改「缺测怎么处理」这一项输入构造**：

| 输入 | 构造 |
|---|---|
| **raw (glitch)** | `missing_band=none` —— 缺测区保持原始 0 值（花屏） |
| **corrected** | `missing_band=interp` + `missing_region_method = griddata \| crosschannel` |

- **统计口径 = glitch 子集 n=637**（测试集中做过修正的样本）。干净帧两种输入完全相同，
  混进全测试集只会稀释差距（全测试集 Δ 恰 = glitch Δ × 637/2893）。
- ⚠️ 权重是**用 griddata 输入训练**的，所以 `crosschannel` 一列属于
  **「评估侧换补全方法」= 模型 × 评估方法错配**，用来检验模型对补全风格的敏感度
  （与 `docs/inpaint_improvement.md` 第 5.1 节的交叉诊断同源）。

### 5.2 结果（glitch 子集，n=637）

| 骨干 | 补全方法 | raw RMSE | **corrected RMSE** | Δ (raw−corr) | 相对改善 | corr MAE | corr R² | 逐样本改善 |
|---|---|---|---|---|---|---|---|---|
| resnet18 | griddata | 15.163 | 14.061 | +1.102 | 7.3% | 10.448 | 0.830 | 345 (54.2%) |
| resnet18 | crosschannel | 15.163 | **13.902** | **+1.261** | 8.3% | 10.298 | 0.834 | 348 (54.6%) |
| resnet34 | griddata | 14.806 | 13.148 | +1.659 | 11.2% | 10.092 | 0.851 | 375 (58.9%) |
| resnet34 | crosschannel | 14.806 | **13.120** | **+1.686** | 11.4% | 10.057 | 0.852 | 373 (58.6%) |
| resnet50 | griddata | 13.648 | 12.526 | +1.122 | 8.2% | 9.493 | 0.865 | 337 (52.9%) |
| resnet50 | crosschannel | 13.648 | **12.493** | **+1.155** | 8.5% | 9.475 | 0.866 | 352 (55.3%) |
| efficientnet_b0 | griddata | 14.519 | **12.981** | **+1.539** | 10.6% | 9.588 | 0.855 | 346 (54.3%) |
| efficientnet_b0 | crosschannel | 14.519 | 13.074 | +1.445 | 10.0% | 9.678 | 0.853 | 338 (53.1%) |

参考：**vgg16**（80 轮口径）同口径消融 raw **14.438** → corrected **12.264**，Δ **+2.17 kn**。

全测试集（n=2893，含 2256 张干净帧，差距被 637/2893 稀释）：

| 骨干 | griddata raw → corr | Δ | crosschannel raw → corr | Δ |
|---|---|---|---|---|
| resnet18 | 14.721 → 14.478 | +0.243 | 14.721 → 14.444 | +0.277 |
| resnet34 | 14.598 → 14.245 | +0.354 | 14.598 → 14.239 | +0.360 |
| resnet50 | 13.624 → 13.384 | +0.239 | 13.624 → 13.377 | +0.246 |
| efficientnet_b0 | 14.471 → 14.145 | +0.326 | 14.471 → 14.164 | +0.307 |

### 5.3 关键读数

1. **8/8 组修复收益为正**（Δ = +1.10 ~ +1.69 kn，相对改善 7.3%~11.4%）
   —— 在**输入层面**，插值修复的价值是确定的、不依赖骨干网络的。
   vgg16 的 Δ（+2.17）比 4 个新骨干都大，但它是 80 轮口径，不宜直接并列。
2. **两种补全方法差异极小**：|Δgriddata − Δcrosschannel| ≤ **0.159 kn**（resnet18），
   其余 ≤ 0.09，且方向不一致（resnet18/34/50 是 crosschannel 略好，
   efficientnet_b0 是 griddata 略好）。⇒ 再次印证**模型对补全风格不敏感**——
   这与 `docs/inpaint_improvement.md` 的结论一致：换补全方法**推不动下游**。
3. **骨干排序稳定**：在**有缺测**的样本上 corrected RMSE 排序
   resnet50 (12.49) < efficientnet_b0 (12.98) < resnet34 (13.12) < resnet18 (13.90)，
   与全测试集排序（13.384 < 14.145 < 14.245 < 14.478）**完全一致**
   ⇒ 缺测子集的结论可直接外推到整体选型。
4. **修复收益高度集中在尾部**（本轮最重要的发现）：

   | 分组 | raw RMSE | corrected RMSE | 变化 |
   |---|---|---|---|
   | 原始误差最大的 **10%（64 张）** | 30.3 ~ 35.4 | 24.3 ~ 30.1 | **↓15% ~ 20%** |
   | 其余 **90%（573 张）** | 9.6 ~ 11.1 | 9.9 ~ 11.1 | **基本中性，部分略升** |

   MSE 归因：那 64 张贡献了总 MSE 降幅的 **95% ~ 101%**
   （>100% 表示尾部增益还要反过来补偿其余 573 张的轻微恶化）。
   逐样本看：**53%~59% 的样本 |误差| 变小，41%~47% 变大**，
   中位改善仅 **0.13 ~ 0.62 kn**。
   ⇒ **修复不是「普遍性小幅提升」，而是「救回少数严重花屏样本」。**
5. 全测试集 Δ 恰等于 glitch Δ × 637/2893（干净帧不受影响），因此只有 +0.24~+0.36。

### 5.4 图

![raw vs corrected](../outputs/tcir_wpac/tcir_wpac_bb/corrraw_summary.png)

### 5.5 `improve` 对 `true` Vmax 的线性拟合

把逐样本修复收益 `improve = |err_raw| − |err_corrected|` 对真值 `true_vmax` 做最小二乘
直线拟合（`scripts/improve_vs_true_fit.py`，8 组 = 4 骨干 × 2 补全方法，n=637）。

| 骨干 | 补全方法 | 斜率 | 截距 | r | R² | p | 过零(kn) |
|---|---|---|---|---|---|---|---|
| ResNet18 | griddata | 0.0269 | −0.75 | 0.172 | 0.030 | 1e-5 | 27.9 |
| ResNet18 | crosschannel | 0.0237 | −0.42 | 0.151 | 0.023 | 1e-4 | 17.6 |
| ResNet34 | griddata | 0.0307 | −0.46 | 0.191 | 0.036 | 6e-7 | 15.0 |
| ResNet34 | crosschannel | 0.0293 | −0.34 | 0.179 | 0.032 | 1e-6 | 11.7 |
| **ResNet50** | griddata | **0.0334** | −1.29 | 0.210 | 0.044 | 3e-8 | 38.6 |
| ResNet50 | crosschannel | 0.0303 | −1.09 | 0.200 | 0.040 | 3e-7 | 36.1 |
| EffNet-B0 | griddata | 0.0289 | −0.90 | 0.174 | 0.030 | 1e-5 | 31.2 |
| EffNet-B0 | crosschannel | 0.0271 | −0.89 | 0.169 | 0.029 | 3e-5 | 32.8 |

**读数**

1. **8/8 组斜率全为正，且高度一致**（0.024 ~ 0.033，均值 **0.0288**）。即
   **真值强度每高 1 kn，修复带来的 |误差| 改善多 0.029 kn**——收益随强度近似线性增长。
2. **统计显著但解释力弱**：8/8 组 p < 0.01（第 1e-4 ~ 3e-8 量级），但 R² 仅
   **0.023 ~ 0.044**（均值 0.033）。⇒ 强度只是修复收益的一个**弱预测因子**，
   逐样本方差远大于趋势本身。Spearman ρ 同号但更小（0.06 ~ 0.15），说明趋势不是离群点造的。
3. **外推到低强度会出现负值**：拟合线过零点在 **12 ~ 39 kn**，即
   弱于约 20~40 kn 的风暴，修复的期望收益变为**负**——与 5.3 节「not-improved 全是
   30~60 kn 弱台风」互证。参考强度上的拟合预测（ResNet50 / griddata）：

   | true Vmax | 40 | 60 | 80 | 100 | 120 |
   |---|---|---|---|---|---|
   | 预测 improve (kn) | +0.05 | +0.71 | +1.38 | +2.05 | +2.72 |

4. **分箱均值呈「低强度≈0 或负 → 中高强度单调抬升 → 120 kn 达峰后回落」**，
   峰值出现在 100~120 kn（约 +3 ~ +4 kn），140 kn 箱回到约 +2 kn。
   ⇒ 修复对**强台风**收益最大，对超强台风增益趋缓（raw 时它们已大量被低估，
   修复能纠回一部分，但上限受模型自身表达力限制）。
5. **两种补全方法的拟合几乎重合**（斜率差 ≤ 0.003、截距差 ≤ 0.2），再次印证
   **模型对补全风格不敏感**，趋势由数据结构（缺测只在 IR1、且强台风缺测更致命）决定。

> 一句话：`improve ≈ 0.029 · Vmax − 0.85`（8 组平均），但 R² 仅 0.03 ——
> **方向明确、强度不可靠**，不能拿它做逐样本的修复收益预测。

![improve ~ true 拟合](../outputs/tcir_wpac/tcir_wpac_bb/corrraw_summary_fit.png)

### 5.6 `improve` 与缺测位置（中心 vs 边缘）的关系

**动机**：TCIR 图像以风暴中心为画面中心，因此「缺测落在画面中心区」≈「缺测盖住了风暴核心」。
上一节已知 `improve` 与强度弱相关；本节检验**位置**是否是更强的判据。
脚本 `scripts/improve_vs_position.py`（n=637，resnet50）。

**位置指标**（96×96 归一化，越大越靠近中心）

| 指标 | 含义 |
|---|---|
| `central_frac` | 缺测像素落在中心圆盘（半径 24 px = 0.25·min(H,W)）内的比例 ← **主指标** |
| `c_min` | 缺测像素到画面中心的**最小**距离 / 半对角线（0 = 盖住中心） |
| `c_cent` | 缺测质心到画面中心距离 / 半对角线 |
| `row_off` | \|质心行 − 中心行\| / 中心行（IR1 是横向条带，衡量其垂直位置） |

**分组对比**（`central_frac > 0` 即为「中心组」）

| 补全方法 | 中心组 (n=370) | 边缘组 (n=267) | 差 | Welch t | p | Mann-Whitney p |
|---|---|---|---|---|---|---|
| griddata | **+1.262** | **−0.234** | **+1.496** | 3.91 | <0.001 | <0.001 |
| crosschannel | **+1.212** | **−0.123** | **+1.334** | 3.65 | <0.001 | 0.001 |

**相关系数（griddata）**

| 指标 | Pearson | Spearman | 偏相关（控 Vmax） | 偏相关（控 Vmax + 面积） |
|---|---|---|---|---|
| `central_frac` | +0.116 (p=3e-3) | +0.146 | **+0.137** | +0.116 (p<0.01) |
| `c_min` | −0.186 (p=2e-6) | −0.185 | **−0.231** | −0.131 (p<0.01) |
| `c_cent` | −0.168 (p=2e-5) | −0.150 | −0.201 | −0.101 (p=0.01) |
| `row_off` | −0.141 (p=3e-4) | −0.146 | −0.177 | −0.125 (p<0.01) |
| `area_frac` | +0.198 (p=5e-7) | +0.129 | +0.242 | −0.000 (p=1.00) |

**控制强度后的分层**（griddata，中心组 − 边缘组）

| true Vmax | n | 中心组 | 边缘组 | 差 |
|---|---|---|---|---|
| 0–50 kn | 315 | −0.115 (n=208) | −0.984 (n=107) | +0.870 |
| 50–75 kn | 130 | +1.130 (n=72) | −0.209 (n=58) | +1.339 |
| 75–200 kn | 192 | **+4.550** (n=90) | +0.539 (n=102) | **+4.011** |

**读数**

1. **位置比强度是更有效的判据**。中心 vs 边缘的 improve 差 **+1.50 kn**（p<0.001），
   分组间差异量级与 §5.5 线性趋势（R² 仅 0.03）完全不在一个层次上。
2. **位置有独立于强度和面积的效应**：同时控制 Vmax 与缺测面积后，`c_min` / `row_off` /
   `central_frac` 的偏相关仍为 **±0.10~0.13 且 p<0.01**。
   注意 `area_frac` 在控制后会掉到 0 —— 说明**面积效应基本被位置吸收**（大缺口往往也盖住中心），
   而位置效应反而剩下独立成分。
3. **混杂方向对结论不利，反而加强结论**：中心组的平均强度**更低**（52.3 vs 65.0 kn），
   而 improve 随强度递增 ⇒ 若只看强度，中心组应当**更差**。实际却显著更好 ⇒ 位置效应真实且被低估。
4. **两组原始误差几乎相同**（raw 平均 |误差| 10.06 vs 10.22 kn），差别只在**修复能否纠回**。
   ⇒ 位置影响的不是「错得有多离谱」，而是「这个错误能不能被插值救回来」。
5. **效应随强度放大**：弱台风档位置差仅 +0.87 kn，强台风档达 **+4.01 kn**。
   机理上合理：强台风的判据信息高度集中在眼墙/中心区，中心一旦缺测，raw 输入会严重低估
   （见下方案例：83 kn 被压到 31 kn）；而中心未受损时，边缘缺测几乎不改变模型的判断。
6. **两种补全方法结论一致**（差 +1.50 / +1.33），说明这是**数据结构的性质**，与补全算法无关。

**典型案例**（上：中心缺测收益最大；下：边缘缺测收益最差）

| ID | true | raw (Δ) | 修复后 (Δ) | improve | 位置 |
|---|---|---|---|---|---|
| 200911W | 83 | 31.3 (−51.7) | 80.7 (−2.3) | **+49.4** | central_frac 0.24 |
| 200911W | 110 | 51.9 (−58.1) | 93.6 (−16.4) | **+41.7** | central_frac 0.31 |
| 200705W | 90 | 52.9 (−37.1) | 98.8 (+8.8) | **+28.3** | central_frac 0.36 |
| 201119W | 30 | 39.3 (+9.3) | 54.2 (+24.2) | **−15.0** | central_frac 0.00 |
| 200726W | 20 | 44.7 (+24.7) | 58.0 (+38.0) | **−13.3** | central_frac 0.00 |
| 201323W | 100 | 107.7 (+7.7) | 118.5 (+18.4) | **−10.7** | central_frac 0.00 |

![中心 vs 边缘案例](../outputs/tcir_wpac/tcir_wpac_bb/improve_vs_position_cases.png)

> 一句话：**「缺测盖住中心（风暴核心）→ 修复收益大；缺测只在边缘 → 修复收益≈0 甚至为负」**，
> 且该规律在控制强度与缺失面积后依然成立、并随强度放大。
> 这比 §5.5 的强度线性关系强得多，是判断「该不该修」的更实用判据。

![improve vs 位置](../outputs/tcir_wpac/tcir_wpac_bb/improve_vs_position.png)

## 6. 复现

```bash
# 训练（断点续训默认开启；中断后重跑同一条命令即可续上）
python scripts/train_backbones.py -c configs/tcir_wpac_train.yaml \
    --backbones resnet18,resnet34,resnet50,efficientnet_b0 \
    --epochs 40 --patience 10 --missing-band interp \
    --out-name tcir_wpac_bb --eval-test

# 图表
python scripts/backbone_compare_plot.py

# 输入级消融 raw vs corrected（4 骨干 × 2 补全方法 = 8 组）
P=/home/yzm/tf/dl-env/bin/python3.11
BB=outputs/tcir_wpac/tcir_wpac_bb
for M in resnet18 resnet34 resnet50 efficientnet_b0; do
  for MET in griddata crosschannel; do
    EXTRA=""
    [ "$MET" = "crosschannel" ] && EXTRA="--set data.tcir.missing_region_method=crosschannel"
    $P scripts/corrected_vs_raw.py -c configs/tcir_wpac_train.yaml \
      --set model.backbone=$M \
      --set experiment.name=bb_${M}_s42 --set experiment.output_dir=$BB \
      $EXTRA --out-dir $BB/corrraw/${M}__${MET} --show 12
  done
done
$P scripts/corrraw_summary.py      # 汇总表 + 四联图

# improve ~ true 线性拟合（8 组）
$P scripts/improve_vs_true_fit.py

# improve ~ 缺测位置（中心 vs 边缘，resnet50）
$P scripts/improve_vs_position.py
```

> ⚠️ 跑 `corrected_vs_raw.py` 必须带 `--set model.backbone=<骨干>`，
> 否则会用主配置默认的 vgg16 去加载 resnet/efficientnet 权重，报
> `Error(s) in loading state_dict`。
> `crosschannel` 通过 `--set` 显式覆盖一致性键 —— `align_preprocessing` 在
> 「显式覆盖且与训练值冲突」时**保留用户选择、只告警**，正好用于这种交叉评估。

## 7. 产物

| 路径 | 说明 |
|---|---|
| `logs/backbone_compare.csv` / `.json` | 汇总指标（机器可读） |
| `outputs/tcir_wpac/tcir_wpac_bb/backbone_compare.png` | 对比图（RMSE 柱状 + 参数量-精度散点） |
| `outputs/tcir_wpac/tcir_wpac_bb/bb_<name>_s42/` | 各模型 `best.ckpt`、`config_used.yaml`、`train.log`、`test_metrics.json` |
| `logs/training_runs.csv` | 每次训练追加一行（含本次 4 条） |
| `outputs/tcir_wpac/tcir_wpac_bb/corrraw/<骨干>__<方法>/` | 8 组消融的 `corrected_vs_raw.{json,csv,png}` + 指标图（各自独立目录，互不覆盖） |
| `outputs/tcir_wpac/tcir_wpac_bb/corrraw_summary.{png,csv,json}` | 8 组汇总四联图与明细表 |
| `outputs/tcir_wpac/tcir_wpac_bb/corrraw_summary_fit.{png,csv,json}` | `improve ~ true_vmax` 线性拟合（8 组统计 + 分箱曲线） |
| `outputs/tcir_wpac/tcir_wpac_bb/improve_vs_position.{png,csv,json}` | `improve ~ 缺测位置` 六联图 / 逐样本位置指标 / 统计 |
| `outputs/tcir_wpac/tcir_wpac_bb/improve_vs_position_cases.png` | 中心缺测 vs 边缘缺测 典型案例（IR1 底图 + 掩码红色叠加） |
| `outputs/tcir_wpac/tcir_wpac_bb/corrraw/<骨干>__<方法>/hard_cases_top6.{png,csv}` | 逐样本极端案例（improved / not-improved 各 6 张） |
