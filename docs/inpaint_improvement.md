# 部分缺失图像的补全改进：方法、验证与结论

> 实验日期：2026-10-08 ｜ 数据：`data/wpac_96.h5`（20059 帧，IR1/WV/PMW，96×96）
> 固定划分：`split_seed=42` → train/val/test = **14062 / 3008 / 2893**（损坏过滤 53/23/20）
> 除被检验项外，所有对照实验其余配置完全一致（vgg16 / 80 轮 / patience 20 / batch 64 / AdamW 1e-3）。
>
> **一句话结论：补全精度可以大幅改进，但下游 Vmax 精度无法借此提升 —— 四个补全方法 × 三 seed
> 的配对实验（`griddata` / `crosschannel` / `crosschannel+feather3` / `harmonic`）显示
> 三个替代方案全部更差，默认继续用 `griddata`。**
>
> **2026-10-13 补充（学习式修复）：** 新训练的 U-Net 修复网络（`docs/inpaint_unet_report.md`）
> 在**缺测区像素级**显著优于 griddata（RMSE −20.2%、胜率 97%），但**下游已两级验证**
> （推理级 a3 臂 `docs/a3_repair_compare.md`、训练级 a5 臂 `docs/a5_unet_train_report.md`）**均未显著优于 griddata**
> （p=0.34 / 0.82，见 §8.6）⇒ **默认仍用 `griddata`**。

---

## 摘要（先看结论）

1. **补全精度确实被大幅改进**：新方法 `crosschannel` 相比原实现 `griddata`
   在合成 benchmark 上 IR1 RMSE **11.11 → 8.71 K（−22%）**、WV **10.88 → 5.75 K（−47%）**，
   速度快 **16 倍**，且在「缺失区外围被涂抹污染」的模拟下不再崩溃（130 K → 9 K）。
2. **但单次端到端 A/B 反而变差**（test RMSE +0.34 kn）。扩展到 **3v3 配对**后确认：
   val 上两者打平（Δ +0.014, p≈0.92），**test 上 3 个 seed 一致为负收益**
   （Δ 均值 +0.309, 配对 t≈3.9, p≈0.06）。
3. **根因不是补全值不准，而是边界接缝**：`crosschannel` 的填补值来自**另一通道**，
   在缺测区边界留下台阶——实测接缝比中位数 **1.21 vs griddata 1.00**、p90 **3.13 vs 1.71**。
   交叉诊断（模型×评估补全错配）显示模型**不依赖补全风格**，故排除「跨通道捷径」，
   指向训练输入统计特性的偏移。
4. **修复方案：边界羽化（feather）**。按「到缺测边界的距离」把跨通道预测与平滑的空间
   调和插值线性融合。`feather=3` 时**像素级两个指标同时改善**：IR1 RMSE **7.92 → 6.02 K**，
   接缝中位数 **1.21 → 0.99**（回到 griddata 水平）。
5. **但羽化同样救不回下游**（第 7 节，3v3 已完成）：`crosschannel+feather=3` 的
   test 均值 **14.285**，比 griddata 差 **+0.431**、比纯 crosschannel 还差 **+0.123**；
   val 也从「打平」变成「明显变差」（+0.190）。
   ⇒ **「补全精度/接缝」不是决定下游的唯一中介变量**，该方向的改进路线到此为止。
6. **最终建议**：**`griddata` 保持为默认**（v2 口径 test RMSE 13.54 不变）。
   三个替代方案（crosschannel / +feather3 / harmonic）在 3v3 上**全部劣于 griddata**
   （+0.309 / +0.431 / +0.441）。**本方向结案**，后续换杠杆（见第 8.4 节）。
7. **2026-10-13 补充（学习式修复）——首次在像素级"碾压"插值**：用「干净帧 × 真实条状掩码」
   合成训练集训得 U-Net（输入 2ch = 损坏图 + 掩码，输出 1ch 修复图），在测试年干净帧上缺测区
   RMSE **9.231 → 7.369 K（−20.2%，胜率 97%，t=15.6）**、MAE 6.189 → 4.800 K（−22.4%）。
   但**像素级优于 ≠ 下游更好**——**首轮下游验证（a3 臂）已证实**：
   损坏帧 RMSE 17.08→修复后 12.6~12.8，但 U-Net 与 griddata 在下游**无显著差异**（t=−0.96, p=0.34，
   胜率 51.6%），优势只在大面积损坏（>8%）才明显。详见 §8.6、`docs/a3_repair_compare.md`。

---

## 1. 缺失形态诊断（`scripts/analyze_missing_shape.py`）

全量扫描 20059 帧，两个通道的缺测**形态完全不同**：

| 通道 | 有缺测的帧 | 块形态 | 单帧缺失面积 |
|---|---|---|---|
| IR1 | 3969（**19.8%**） | **横向细条带**（长宽比中位 0.027，81.3% 贴边） | p50 仅 1.15% |
| WV | 188（**0.94%**） | 罕见但**大块**（p90 面积 5250 px，近方块） | 发生时 p90 达 **84%** |

⇒ 不该用同一套策略：细条带适合沿法向的 1D/调和插值，大块缺失必须借助通道间相关性。

**另一个关键实测**（比较缺测块外围 ring1~4 与帧内有效均值）：

| 通道 | ring1 偏差 | 明显偏低(<−0.5σ)的帧占比 |
|---|---|---|
| IR1 | −0.10σ | 18% |
| **WV** | **−1.21σ** | **61%** |

bilinear 下采样把 WV 缺测区**外围一圈像素也拉低了**。纯空间插值会把这些被污染的像素
当真值使用——这是空间方法对 WV 天然不利的原因。

产出：`outputs/missing_shape_analysis.png` / `missing_shape_stats.json`

---

## 2. 有真值的合成 benchmark（`scripts/benchmark_inpaint.py`）

真实缺测无 ground truth，故**提取真实缺失形态的掩码**、施加到**干净帧**上构造伪缺失样本。
缓存：`data/inpaint_bench_cache.npz`（400 干净帧 + 250 IR1 掩码 + 172 WV 掩码，面积分位覆盖 0.5%~92%）。

| 方法 | IR1 | WV | 模拟边缘污染·IR1 | ·WV | 耗时/mask |
|---|---|---|---|---|---|
| mean（均值兜底） | 25.54 | 11.98 | 26.85 | 13.27 | — |
| **griddata（原实现）** | 11.11 | 10.88 | **132.1** | **118.5** | **72 ms** |
| row_interp | 26.93 | 11.76 | 117.3 | 114.4 | — |
| col_interp | 11.03 | 10.42 | 127.4 | 74.9 | — |
| harmonic | 10.39 | 9.86 | 127.4 | 74.7 | 11 ms |
| **crosschannel（新）** | **8.71** | **5.75** | **13.88** | **9.23** | **~1 ms** |

- `row_interp` 在 IR1 上极差（26.9）→ 印证 IR1 是**横向**条带（行内无有效像素）。
- 边缘污染会让**纯空间插值彻底崩溃**（130 K ≈ 把 260 K 的亮温填成 130 K）。
- 跨通道回归的多项式阶数：IR1 deg1 8.31 / deg2 7.94 / deg3 7.43；WV deg1 3.78 / deg2 3.33 / deg3 3.15
  → 取 **deg=2**（兼顾稳健）。

---

## 3. 实现（`tcintens/data/band_health.py`）

### 3.1 `harmonic_fill()` — 调和插值
迭代求解带 Dirichlet 边界的拉普拉斯方程（最小化缺失区内梯度能量）。
关键优化：用**列向 1D 插值**作初值，对细条带几乎已是正解，100 次迭代即收敛（均值初值需 300+）。
速度比 griddata 快约 6.5 倍、精度高约 7%。

### 3.2 `inpaint_regions_crosschannel()` — 跨通道回归 + 调和兜底
1. 按缺测**面积升序**逐通道修复 —— 先修好的通道可作后修通道的参考；
2. 用参考通道的**全部有效像素**拟合 `y = f(ref)`（deg=2，归一化坐标避免 260 K 量级病态），
   再把缺测位置的 ref 值代回。**不依赖缺测区周边（可能被涂抹污染的）像素**；
3. `robust=True`：一轮 3×MAD 残差剔除后重拟合；
4. **夹取保护**：预测值 clip 到有效值极差 ±25%，杜绝负亮温等非物理输出；
5. 无可用参考通道 → 退回 `harmonic_fill`；
6. **参考通道白名单 `ref_idx` 必须排除 PMW** —— PMW 的 0 是合法观测值，混入回归会发散。

### 3.3 `feather` — 边界羽化（本报告的核心新增）
`feather > 0` 时，用「到缺测边界的距离」做权重，把跨通道预测与空间调和插值融合：

```
alpha = clip(distance_to_boundary / feather, 0, 1)     # 边界=0，深度>=feather 处=1
out[mask] = alpha * crosschannel_pred + (1 - alpha) * harmonic_fill
```

动因见第 5 节的接缝诊断。效果见第 6 节。

### 3.4 配置与一致性
新增配置键 `data.tcir.missing_region_feather`（默认 0 = 旧行为，**向后兼容**），
并已加入 `utils/repro.py::CONSISTENCY_KEYS`，训练/评估/可视化自动对齐。
相关键：`missing_region_method` / `missing_region_deg` / `missing_region_iters` / `missing_region_feather`。

---

## 4. 端到端 A/B：补全更准，但下游没变好

### 4.1 单跑（seed 42，唯一变量 = 补全方法）

| 指标 | griddata | crosschannel | Δ |
|---|---|---|---|
| val RMSE | 13.1905 | 13.4632 | +0.273 |
| **test RMSE** | **14.158** | **14.499** | **+0.341** |
| test MAE / R² | 10.596 / 0.817 | 10.823 / 0.808 | +0.227 / −0.009 |
| 修正组 RMSE (n=637) | 13.239 | 13.336 | +0.097 |
| 干净组 RMSE (n=2256) | 14.407 | 14.811 | +0.404 |

**退步主要在「干净组」**——而干净组根本无缺测、补全碰不到它们 ⇒ 这是**模型层面**的效应。

### 4.2 3v3 配对（`scripts/inpaint_3v3_plot.py`）

| seed | griddata val → test | crosschannel val → test | Δtest (cc−gd) |
|---|---|---|---|
| 42 | 13.191 → 14.158 | 13.463 → 14.499 | +0.341 |
| 43 | 13.071 → 13.863 | 12.946 → 14.022 | +0.159 |
| 44 | 13.065 → 13.539 | 12.959 → 13.966 | +0.427 |
| **均值** | 13.109 → **13.853** | 13.123 → **14.162** | **+0.309** |

- val：Δ 均值 **+0.014**（配对 t≈0.11，p≈0.92）→ **无差异**；
- test：Δ 均值 **+0.309**，3/3 同号，配对 **t≈3.92（df=2，p≈0.06）** → 提示性负收益；
- 分组（3 seed 均值）：修正组 12.66 → 13.00（+0.34）、干净组 14.17 → 14.47（+0.30）
  → **两组一致变差**，进一步确认是全局模型质量偏移。

产出：`outputs/inpaint_3v3_compare.png`

---

## 5. 交叉诊断与根因

### 5.1 交叉诊断（模型 × 评估补全方法错配，`repro.py` 在显式 `--set` 冲突时保留用户选择）

| 训练用 | 评估用 | test RMSE |
|---|---|---|
| griddata | griddata | 14.158 |
| griddata | crosschannel | 14.136 |
| crosschannel | griddata | 14.538 |
| crosschannel | crosschannel | 14.499 |

**换评估时的补全方法只差 0.02~0.04 kn** ⇒ 模型对补全风格**不敏感**，
排除「模型学到跨通道捷径」。差异来自**训练输入的统计特性**。

### 5.2 接缝诊断（`scripts/seam_diagnostic.py`）

指标 `seam_ratio` = 边界环带梯度幅值 / 帧内背景梯度幅值（1.0 = 与背景一致）。

| 通道 | 方法 | 中位数 | 均值 | p90 | >1.5 占比 |
|---|---|---|---|---|---|
| IR1 | griddata | 0.998 | 1.202 | 1.711 | 15.0% |
| IR1 | harmonic | 0.916 | 1.109 | 1.555 | 12.0% |
| IR1 | **crosschannel** | **1.213** | **1.807** | **3.130** | **29.0%** |
| WV | griddata | 1.693 | 2.151 | 3.557 | 57.6% |
| WV | **crosschannel** | **1.991** | **2.700** | **4.732** | **73.3%** |

`crosschannel` 的填补值数值上最准，却引入最明显的边界台阶——因为它的纹理来自**另一通道**。
benchmark 只衡量数值精度，测不到这种伪影，而下游 CNN 对伪影敏感。
**这就是「补全更准却下游更差」的机制。**

产出：`outputs/seam_diagnostic.json`

---

## 6. 修复：边界羽化（`scripts/verify_feather.py`）

| feather | IR1 RMSE | WV RMSE | 接缝中位 | 接缝 p90 | >1.5 占比 |
|---|---|---|---|---|---|
| 0（原 crosschannel） | 7.923 | 3.347 | 1.213 | 3.130 | 29.0% |
| 2 | 6.640 | 3.072 | 1.031 | 2.094 | 21.7% |
| **3** | **6.021** | **2.962** | **0.993** | **1.847** | **17.7%** |
| 5 | 6.134 | 3.067 | 0.960 | 1.671 | 13.7% |

`feather=3` **同时**把精度提到最高、把接缝降到 griddata 水平。
机制解释：对外侧薄条带，羽化让平滑的空间解主导（那里空间插值本就接近正解）；
对内部大块，跨通道精度得以保留；两者误差部分独立，加权融合还能进一步降方差。

### 管线接入验证（`scripts/smoke_feather_pipeline.py`）
- 划分不变：**14062 / 3008 / 2893**（剔除 53/23/20）；
- `finite=True`，取值范围与 griddata 一致；
- IR1/WV 极小值与 griddata **逐位相同**（−7.941 / −3.201，对应 0.00 K）——
  是 `min_blob=20` 之下的细小残余 0 斑点，本就在掩码之外，与方法无关。

---

## 7. feather=3 的下游 3v3 验证（已完成 —— 结论：仍然负收益）

命令（三支串行，训完即 eval test）：

```bash
for S in 42 43 44; do
  python tcir.py train -c configs/tcir_wpac_train.yaml \
    --set data.tcir.missing_band=interp \
    --set data.tcir.missing_region_method=crosschannel \
    --set data.tcir.missing_region_feather=3 \
    --set experiment.seed=$S --set experiment.name=tcir_wpac_ccf_s$S
  python tcir.py eval -c configs/tcir_wpac_train.yaml --set experiment.name=tcir_wpac_ccf_s$S
done
```

产物目录：`outputs/tcir_wpac/tcir_wpac_ccf_s{42,43,44}/`（共 2h02m，均正常收敛）。

### 7.1 三臂总表（每臂 3 seed 配对，仅补全方法不同）

由 `scripts/inpaint_arms_compare.py` 汇总（同时补跑了三臂的 `grouped_eval.py`）：

| 臂 | val 均值 | test 均值 | 修正组 (n=637) | 干净组 (n=2256) | Δtest vs griddata | 配对 t |
|---|---|---|---|---|---|---|
| **griddata（沿用默认）** | **13.109** | **13.853** | **12.655** | **14.173** | — | — |
| crosschannel | 13.123 | 14.162 | 12.997 | 14.473 | +0.309 | 3.92 |
| crosschannel + feather=3 | 13.299 | 14.285 | 13.261 | 14.559 | **+0.431** | 1.53 |

逐 seed Δtest：

| 臂 | s42 | s43 | s44 |
|---|---|---|---|
| crosschannel | +0.341 | +0.159 | +0.427 |
| crosschannel + feather=3 | +0.195 | +0.107 | **+0.993** |

### 7.2 羽化确实修掉了接缝，但没修好下游

- `ccf − cc` 配对差 = **−0.146 / −0.053 / +0.566**（2/3 个 seed 略好，s44 大幅变差）
  → 均值 **+0.123**，配对 t≈0.55（df=2）→ **与纯 crosschannel 无显著差异**。
- 更关键的是 **val 也从「打平」变成「变差」**：13.123 → **13.299**（+0.176）。
  val 与 test 同方向 ⇒ 不是过拟合/划分噪声，而是**训练输入分布本身**的效应。
- 分组看：修正组 12.655 → 13.261（+0.606）、干净组 14.173 → 14.559（+0.386）
  —— **两个组都变差，且修正组更差**。干净组根本不含缺测、补全碰不到它却同样退化，
  再次印证第 4.1 / 5.1 节的判断：**效应作用在全局模型质量上，而不在被补的那些像素上。**

### 7.3 为什么「像素级双赢」没能传导到下游

第 6 节的 benchmark 与接缝诊断都是**输入侧**指标；它们只能证明「补得更准、更不像伪影」，
**不能**保证 CNN 学得更好。已排除的解释与剩余事实：

| 假设 | 检验 | 结论 |
|---|---|---|
| 模型学到「跨通道捷径」，换了补全风格就失效 | 模型×评估补全错配（第 5.1 节），仅差 0.02~0.04 | **排除** |
| 补全值不准 | benchmark 上 crosschannel 全面更优 | **排除** |
| 边界台阶伪影 | feather=3 后接缝已回到 griddata 水平（0.993 vs 0.998） | **排除**（修了也没用） |
| 训练输入分布偏移（补全像素的**高阶统计**，如局部纹理/梯度分布） | val/test 同向退化、干净组同步退化、接缝已修仍退化 | **剩余解释** |

⇒ 结论：**在 96² / VGG16 / 该数据规模下，「把缺测补得更准更平滑」不是提升下游 Vmax 的有效杠杆。**
这是一个有价值的负结果，已如实写入第 8 节建议。

### 7.4 `harmonic` 对照（已完成 —— 同样负收益，本方向结案）

`harmonic`（纯空间、迭代解拉普拉斯）是最后一个候选：benchmark 上它
**略优于 griddata**（IR1 10.39 vs 11.11，WV 9.86 vs 10.88）、**接缝最好**
（中位 0.916 vs 0.998）、且**快 6.5×**。命令：

```bash
for S in 42 43 44; do
  python tcir.py train -c configs/tcir_wpac_train.yaml \
    --set data.tcir.missing_band=interp --set data.tcir.missing_region_method=harmonic \
    --set experiment.seed=$S --set experiment.name=tcir_wpac_harm_s$S
  python tcir.py eval -c configs/tcir_wpac_train.yaml --set experiment.name=tcir_wpac_harm_s$S
done
```

结果（`outputs/tcir_wpac/tcir_wpac_harm_s{42,43,44}/`，1h51m）：

| seed | val | test | 修正组 | 干净组 |
|---|---|---|---|---|
| 42 | 13.185 | 14.370 | 13.365 | 14.641 |
| 43 | 13.291 | 14.219 | 13.800 | 14.335 |
| 44 | 12.884 | 14.295 | 12.914 | 14.661 |
| **均值** | **13.120** | **14.294** | 13.360 | 14.546 |

**`harmonic` 给出了本轮最有意思的一组数**：
- **val 与 griddata 几乎完全打平**（13.120 vs 13.109，Δ **+0.011**），
  是唯一在 val 上不输 griddata 的替代方案；
- **但 test 仍差 +0.441**（逐 seed +0.212 / +0.356 / +0.756，配对 t≈2.71）。

这正好把第 7.3 节的判断钉死：**val 打平甚至更好的补全，test 依然更差**，
说明问题不在「补全质量」这条轴上。

---

## 7.5 四臂总表（最终）

`scripts/inpaint_arms_compare.py` 汇总（四臂 × 三 seed，全部同一划分 / 同参）：

| 臂 | val 均值 | test 均值 | 修正组 | 干净组 | Δtest vs griddata | 配对 t | test−val 落差 |
|---|---|---|---|---|---|---|---|
| **griddata（沿用默认）** | **13.109** | **13.853** | **12.655** | **14.173** | — | — | **+0.745** |
| crosschannel | 13.123 | 14.162 | 12.997 | 14.473 | +0.309 | 3.92 | +1.039 |
| crosschannel + feather=3 | 13.299 | 14.285 | 13.261 | 14.559 | +0.431 | 1.53 | +0.986 |
| harmonic | 13.120 | 14.294 | 13.360 | 14.546 | +0.441 | 2.71 | **+1.174** |

逐 seed Δtest：

| 臂 | s42 | s43 | s44 |
|---|---|---|---|
| crosschannel | +0.341 | +0.159 | +0.427 |
| crosschannel + feather=3 | +0.195 | +0.107 | **+0.993** |
| harmonic | +0.212 | +0.356 | **+0.756** |

**三个替代方案全部为正（更差），12 个配对点中 11 个为正**（唯一的负点是 ccf 的 s42/s43）。
另一个稳定规律：**所有替代方案的 val→test 落差都比 griddata 大**
（+0.75 vs +0.99~+1.17）⇒ 替代补全让模型**泛化更差**，而不只是「这一次差一点」。

图：`outputs/inpaint_arms_compare.png`

---

## 8. 结论与建议

### 8.1 判据汇总

| 判据 | 结论 |
|---|---|
| 补全数值精度（benchmark） | **三个新方法都优于 griddata**（crosschannel 最优，IR1 −22%、WV −47%） |
| 计算开销 | crosschannel 快 16×（~1 ms vs 72 ms）、harmonic 快 6.5×（11 ms） |
| 抗边缘污染 | crosschannel **质变**（130 K → 9 K），空间方法全崩 |
| 边界伪影 | harmonic 最好（0.916）、griddata 0.998、crosschannel 最差（1.213）；**feather=3 可修到 0.993** |
| **下游 Vmax（最终判据）** | **三个替代全部劣于 griddata**：+0.309 / +0.431 / +0.441 |
| 泛化落差 test−val | griddata **+0.745 最小**；替代方案 +0.99 ~ +1.17 |

### 8.2 最终结论

> **在 96² / VGG16 / 该数据规模下，「把缺测补得更准」与下游 Vmax 精度之间没有正向因果。**
> 三个在输入侧全面占优（更准 / 更快 / 抗污染 / 接缝更好）的补全方法，
> 在下游 3v3 配对实验中**无一例外地变差**；其中 `harmonic` 甚至在 val 上与 griddata 完全打平，
> test 却仍差 0.44 kn。已逐项排除「补全值不准」「边界伪影」「跨通道捷径」三个中介假设，
> 剩余解释指向**补全像素的高阶统计（局部纹理 / 梯度分布）改变了训练输入的分布**。

### 8.3 处置

1. **默认保持 `griddata`**，v2 口径（test RMSE 13.54 / MAE 10.29 / R² 0.833）**不变**；
   补全改动**不进入主实验与论文主表**。
2. `harmonic` / `crosschannel`（含 `feather`）保留为**工程可选实现**，仅在
   「需抗边缘污染」或「追求数据加载速度」时使用，并在代码注释中标注**已知不改善下游精度**。
3. **不再在补全方向投入**。

### 8.4 后续方向（补全之外）

| 方向 | 已有线索 |
|---|---|
| 骨干网络 | **resnet50 test 13.384**（优于 vgg16 参考 13.539，参数仅 1/5.9）；efficientnet_b0 仅 4.1M 达 14.145（性价比最高） |
| 损失函数 | 分档加权 / 强台风加权——双方文献均观察到强台风被系统性低估 |
| 输入信息 | 多通道（补回 VIS）、多分辨率（128²/160² 消融已有配置）、时次上下文 |
| 数据量 | 目前仅用 WPAC 子集，全海域扩展 |
| **学习式修复** | U-Net 像素级大幅超越插值（缺测区 RMSE −20%，见 §8.6）；**下游两级验证（推理级 a3 + 训练级 a5）均与 griddata 无显著差异**（p=0.34 / 0.82）。按真实面积分布重训亦无改善（§8.6.1）⇒ 欲提升应换杠杆 |

### 8.5 方法学收获（可写进论文）

> **仅以「重构误差 / 伪影指标」评价补全方法是不充分的。**
> 本文构造了有真值的合成 benchmark 与接缝伪影指标，两者一致指向 crosschannel 更优，
> 但同一改动在下游回归任务上稳定变差（3 seed 配对）。
> 因此在「补全 → 下游任务」的链路中，**必须以下游指标做最终判据**，
> 并至少使用 3 个随机种子做配对比较（本文单 seed 结论曾与 3v3 结论方向相反）。



### 8.6 补充（2026-10-13）：学习式修复（U-Net）——像素级胜出，下游已完整验证（训练级）

前文四个**插值类**方法都止步于"补得更准但下游更差"。作为对照，我们用**学习式**修复
（U-Net 直接学「损坏图 + 掩码 → 修复图」的映射）做了第一次尝试：**像素级明显更强，但下游仍未验证。**

| 指标（亮温 K，越低越好） | griddata | U-Net | 改善 | U-Net 胜率 | 配对 t |
|---|---|---|---|---|---|
| **缺测区 RMSE** | 9.231 | **7.369** | **+1.861（−20.2%）** | **97.0%** | t=15.61 |
| **缺测区 MAE** | 6.189 | **4.800** | **+1.389（−22.4%）** | **98.8%** | t=15.59 |
| 全图 RMSE | 2.822 | **2.427** | +0.395（−14.0%） | 88.8% | t=7.97 |
| 全图 MAE | **0.607** | 0.819 | −0.212 | 10.2% | t=−20.65 |

- **协议**：训练年（2003–2013）干净帧 **12502** × 真实条状掩码 **1155**（面积 5–40%，**掩码库只取自训练年**），
  在线合成（水平镜像 + 纵向平移 ±24px，掩码处置哨兵）；评估用测试年（2014–2016）干净帧 **N=500**
  人工贴掩码，逐样本配对。模型 17.46 M 参数、L1 缺测区 4× 加权、60 轮，best_roi 0.1452。
- **对缺测面积不敏感**（改善 +1.57 ~ +2.05 K，各档胜率 ≥95.9%）。U-Net 大胜的个例全是
  **griddata 插值彻底失效**（误差 22–29 K，缺测切断强梯度）——与其"救回少数严重花屏样本"的机制一致。
- ⚠️ **全图 MAE 上 griddata 更好是"构造性"的，不是真实质量差距**：griddata 只改缺测像素、掩码外误差恒 0，
  其全图 MAE ≈ 缺测区 MAE × 面积占比（6.189 × 0.101 ≈ 0.62）。**评修复质量只看缺测区指标。**
- ⚠️ **下游验证（已完成两级）**：
  - **推理级（a3 臂，`docs/a3_repair_compare.md`）**：在「仅干净图训练」的 a3 上，损坏帧 RMSE
    **17.08 → 修复后 12.6~12.8**（修复与否是决定性因素）；但 **U-Net 相比 griddata 无显著下游增益**
    ——损坏帧 RMSE 12.611 vs 12.836、配对 |误差| t=−0.96（p=0.34）、胜率 51.6%，
    优势只在大面积损坏（>8%，n=56）才明显（−1.15 kn）。
  - **训练级（a5 臂，`docs/a5_unet_train_report.md`）**：把 U-Net 修复做进**训练与测试的输入侧**
    （离线预修复缓存 `data/ir1_unet_repaired.npy` + 同口径 by_year），a5 三口径全部最优
    （全集 12.466 / 干净 12.438 / 损坏 12.654），但相对 a2(griddata) 只赢 **0.05–0.08 kn、p=0.82（不显著）**；
    a1→a5 的损坏帧改善（13.499→12.654，−0.85 kn）说明**"修不修"是决定性的，"用什么修"不是**。
  ⇒ 两级验证都指向同一结论：**像素级准 ≠ 下游更好**。
- 完整报告：`docs/inpaint_unet_report.md`；产物：`outputs/tcir_inpaint_unet/`。

**8.6.1 追加（2026-10-13）：按真实缺测面积分布重训 —— 否定结果。**
上面这版 U-Net 的训练掩码池被筛成**面积 5%~40%**（中位 8.05%），而真实损坏帧的缺测面积
中位只有 **3.17%**（全集 3969 帧实测；p90 10.4%）⇒ 训练分布系统性偏大。
按真实面积直方图**重采样掩码池**（新池 3973 个、中位 3.23%，总变差距离 0.0066）并重训 60 轮后：

| 视角 | 旧 U-Net（5~40%） | 新 U-Net（真实分布） | 判定 |
|---|---|---|---|
| 像素级缺测区 RMSE（真实分布掩码池 N=600） | **6.591 K** | 6.752 K | 新**差 +0.16 K**（t=10.9，p=2.5e−25，7/7 面积档新都更差） |
| 下游损坏帧 RMSE（a3 臂，n=519） | 12.611 kn | **12.603 kn** | 差 −0.008 kn（p=0.32，不可区分） |

⇒ **面积分布错配不是瓶颈**：决定泛化的不是"分布对齐"而是**训练难度/覆盖面**——
大缺测才是逼网络学"跨长程重建 IR1 结构"的强信号，只喂小洞会让它退化成"邻域局部平滑"。
**U-Net 择优仍用旧权重**（`outputs/tcir_inpaint_unet/`）；要提升应往**更难/更广**的分布走。
详见 `docs/unet_areamatch_report.md`。

---

## 9. 复现命令一览

```bash
export HOME=/home/yzm PYTHONPATH=/home/yzm/tf/tc_intensity
P=/home/yzm/tf/dl-env/bin/python3.11

# 形态诊断 / benchmark / 真实帧对比 / 接缝诊断 / 羽化验证
$P scripts/analyze_missing_shape.py
$P scripts/benchmark_inpaint.py            # 加 --edge-blur 模拟边缘污染
$P scripts/compare_inpaint_methods.py
$P scripts/seam_diagnostic.py --n 300
$P scripts/verify_feather.py

# 图表
$P scripts/inpaint_ab_plot.py              # 单跑 A/B 四联图
$P scripts/inpaint_3v3_plot.py             # 3v3 配对图（griddata vs crosschannel）
$P scripts/inpaint_arms_compare.py         # 多臂 3v3 汇总图（自动并入已完成的臂）

# 分组评估（修正组 vs 干净组，9 个模型）
$P scripts/grouped_eval.py -c configs/tcir_wpac_train.yaml \
   --set experiment.name=<NAME> --set experiment.output_dir=<OUT_DIR>

# 训练 / 评估
$P tcir.py train -c configs/tcir_wpac_train.yaml \
   --set data.tcir.missing_band=interp \
   --set data.tcir.missing_region_method=crosschannel \
   --set data.tcir.missing_region_feather=3 \
   --set experiment.name=<NAME>
$P tcir.py eval -c configs/tcir_wpac_train.yaml --set experiment.name=<NAME>
```

## 10. 产物文件清单

| 文件 | 说明 |
|---|---|
| `tcintens/data/band_health.py` | `harmonic_fill` / `inpaint_regions_crosschannel` / `feather` |
| `tcintens/data/tcir.py` | 新增 `missing_region_feather` 配置键与透传 |
| `tcintens/utils/repro.py` | `missing_region_feather` 加入一致性键 |
| `tcintens/utils/plotting.py` | `use_cjk_font()`（解决 matplotlib 中文缺字） |
| `scripts/analyze_missing_shape.py` | 缺失形态诊断 |
| `scripts/benchmark_inpaint.py` | 有真值的补全 benchmark |
| `scripts/compare_inpaint_methods.py` | 真实帧三方法对比 |
| `scripts/seam_diagnostic.py` | 接缝伪影诊断 |
| `scripts/verify_feather.py` | 羽化效果验证 |
| `scripts/smoke_feather_pipeline.py` | 管线接入冒烟 |
| `scripts/inpaint_ab_plot.py` / `inpaint_3v3_plot.py` | 对照图 |
| `scripts/inpaint_arms_compare.py` | 多臂 3v3 汇总图 + 控制台总表（自动并入已完成的臂） |
| `outputs/missing_shape_analysis.png` | 缺失形态 |
| `outputs/inpaint_benchmark*.png/json` | benchmark |
| `outputs/inpaint_real_compare.png` | 真实帧对比 |
| `outputs/seam_diagnostic.json` | 接缝诊断 |
| `outputs/inpaint_ab_compare.png` | 单跑 A/B |
| `outputs/inpaint_3v3_compare.png` | 3v3 配对（griddata vs crosschannel） |
| `outputs/inpaint_arms_compare.png` | 多臂 3v3 汇总（griddata / crosschannel / +feather3 / harmonic） |
| `outputs/tcir_wpac/tcir_wpac_ccf_s{42,43,44}/` | crosschannel+feather=3 的三次训练与评估 |
| `outputs/tcir_wpac/tcir_wpac_harm_s{42,43,44}/` | harmonic 的三次训练与评估 |
| `tcintens/models/unet.py` | U-Net 修复网络（2ch→1ch，17.46M，见 §8.6） |
| `scripts/build_inpaint_pairs.py` / `train_inpaint_unet.py` / `eval_inpaint_unet.py` | U-Net 修复的数据构建 / 训练 / 评估 |
| `outputs/tcir_inpaint_unet/` | U-Net 修复模型（best.ckpt）+ `eval_summary.{csv,json}` + `cases/*.png` |
| `docs/inpaint_unet_report.md` | U-Net 修复实验完整报告 |
| `scripts/build_inpaint_pairs_matched.py` + `logs/inpaint_pairs_areamatch*.{npz,json,png}` | 按真实缺测面积分布重采样掩码池（§8.6.1） |
| `outputs/tcir_inpaint_unet_areamatch/` + `docs/unet_areamatch_report.md` | 真实分布重训版 U-Net（否定结果，§8.6.1） |
| `scripts/eval_unet_areamatch_compare.py` | 旧/新 U-Net 与 griddata 的像素级三法同图对比 |
| `scripts/a3_repair_compare.py` + `outputs/tcir_wpac_ir1/a3_repair_compare/` | a3 臂下 griddata vs U-Net **下游强度回归**对比（`docs/a3_repair_compare.md`） |
| `scripts/build_unet_repaired_ir1.py` + `data/ir1_unet_repaired.npy` | 离线预修复 IR1 缓存（供 a5 臂训练/评估用） |
| `outputs/tcir_wpac_ir1/ir1_year_a5_unet/` + `docs/a5_unet_train_report.md` | a5 臂：U-Net 修复做进训练管线的**训练级**下游验证 |

## 11. 踩坑记录

1. **跨通道方法的参考通道必须检测自身缺失**：首版对比脚本只检测目标通道，
   参考通道自己的 0 值混进了回归拟合 → WV 大洞被补成 **−52 K**。
   改为一并检测后消失。跨通道方法的前提是「参考通道在参与拟合的位置是真值」。
2. **PMW 的 0 是合法观测值**，不能作参考通道，否则被大量 0 污染。
3. **多项式外推**：真实缺测常在分布边缘（台风眼墙），必须加夹取保护。
4. **matplotlib 中文**：WSL 唯一系统 CJK 字体 `Droid Sans Fallback` 不含拉丁数字，
   且 matplotlib 不跨字体逐字形回退 → 用 `tcintens/utils/plotting.py::use_cjk_font()`
   注册 Windows 挂载字体（`simhei.ttf` 等）。标注中也**不能用 Unicode 减号 `−`(U+2212)**。
5. **一致性守卫的 `--set` 语义**：显式 `--set` 冲突时守卫**保留用户选择、仅告警**——
   这正好可以用来做「模型×评估补全错配」的交叉诊断。
6. **U-Net 训练（2026-10-13，见 §8.6）**：`torch.from_numpy` 不接受负 stride 数组
   （水平镜像 `m[:, ::-1]` 需 `.copy()`）；torch 2.11 起 `torch.cuda.amp.*` 已弃用，
   改用 `torch.amp.GradScaler("cuda")` / `torch.amp.autocast("cuda")`。
7. **判断训练是否真卡死要看 checkpoint 的 mtime**，不要看日志（stdout 块缓冲会滞后）；
   本项目曾在回合间因 WSL GPU stall 卡死，用 `last.ckpt` 断点续训无损恢复。
