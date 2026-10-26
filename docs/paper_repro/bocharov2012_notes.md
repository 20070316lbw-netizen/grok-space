# Bocharov & Thiesson (2012/2013) 论文整理与复现笔记

> 标注约定：**【原文】** = 论文明确写出的内容（附页码）；**【解读】** = 我的理解/推断；**【假设】** = 论文未给出、复现时自行设定的参数或做法。
> 页码：Ch = Springer 章节版 *Pattern Recognition – Applications and Methods* (AISC 204, 2013) pp. 93–107；Cf = ICPRAM 2012 会议版 Proceedings pp. 116–125。

## 0. 全文来源

| 版本 | 标题 | 获取方式 | 本地文件 |
|---|---|---|---|
| 会议版 ICPRAM 2012（DOI 10.5220/0003758601160125） | *Non-parametric Segmentation of Regime-Switching Time Series with Oblique Switching Trees* | 微软研究院官网 PDF：https://www.microsoft.com/en-us/research/wp-content/uploads/2012/02/ICPRAM_BocharovThiesson.pdf | `papers/bocharov_thiesson_icpram2012_conference.pdf`（10 页）+ 文本 `.txt`（全文已读；仅本地保存，不提交到仓库） |
| 章节版 AISC 204, pp. 93–107（DOI 10.1007/978-3-642-36530-0_8） | *Segmentation of Nonstationary Time Series with Geometric Clustering* | 奥尔堡大学 VBN 仓储的出版社 PDF（https://vbn.aau.dk/en/publications/segmentation-of-nonstationary-time-series-with-geometric-clusteri）。该 PDF 文件链接对脚本下载返回 403，全文通过 studyres.com 上该 VBN PDF 的文字转录（https://studyres.com/doc/8072848/）完整阅读 | 未存 PDF（下载被拒）；内容与会议版逐节核对过 |

两版的方法、实验、Table 1/2 数字**完全一致**。章节版的差异：标题与摘要改写（"geometric clustering"、"nonstationary"）；第 6 节补充"时间切分不提升"的结论可能不推广到股价以外的数据；第 7 节补充未来做非高斯（如 Student-t/离散）叶子的设想。会议版摘要里写的是 "a few targeted splits on time"。

PDF 受版权保护，**不放入代码 zip**。

## 1. 研究目标与分析对象（重点）

- **【原文】** 目标：为 ART（autoregressive tree，自回归树，Meek et al. 2002）模型提出新的切分候选生成方法——用谱聚类在 "phase view" 上找**一个**斜切分（oblique split），并在 "trace view" 上找少量时间切分点；用来替代标准的 AxialGaussian（每个自回归项 15 个高斯分位数轴向切分点）。（Ch pp. 93–95；Cf p.116–117）
- **【原文】** 被预测的量：**每只股票的日收盘价** x_t 本身（"daily closing prices for 1495 stocks"，Ch p.102；Cf p.122）。模型叶子是价格**水平**上的高斯 AR(p)：x_t | x_{t-p..t-1} ~ N(m + Σ b_j x_{t-j}, σ²)（Ch p.96；Cf p.117）。
- **【原文】** 预测步长：**一步**（one-step forecast），holdout = 每条序列**最后 5 个观测**（一个交易周），对每个 holdout 观测做一步预测（Ch pp.102–103；Cf p.122）。
- **【原文】** 评价指标：sequential predictive score = holdout 上一步预测对数似然的平均；方法的总分 = 所有序列该分数的平均。论文强调对数似然"同时评估估计的准确性和不确定性估计的准确性"（Ch p.103；Cf p.122）。
- **【解读】** 因此：分析目标**不是**收益率或涨跌方向，也不是分段本身；分段（regime）是手段，最终评价的是**下一日收盘价（价格水平）的概率预测质量**。对数似然有价格单位（美元），所以跨股票平均会被高价股/低价股的尺度影响——论文未讨论这一点。
- **【解读】** 每只股票单独建模（约 145 个训练点），不是横截面模型；没有外生变量（"We do not complicate the experiments with the use of exogenous regressors"，Cf p.122）。时间（time index）只能作为切分变量，不做回归项（Ch p.96；Cf p.118）。

## 2. ART 模型定义

- **【原文】** AR(p)：p(x_t | x_1^{t-1}) = p(x_t | x_{t-p}^{t-1}) ~ N(m + Σ_{j=1}^p b_j x_{t-j}, σ²)，参数 θ = (m, b_1..b_p, σ²)（Ch p.96；Cf p.117）。
- **【原文】** ART = AR 的 regime-switching 推广：一棵切换回归树决定每个时间步用哪个叶子 AR 模型；自回归项既作切分输入又作叶子回归的预测变量（Ch p.96；Cf p.117）。
- **【原文】** 标准 ART 切分：轴向 X_i ≤ c，X_i 为某个自回归项或时间变量（Ch p.96；Cf p.118）。本文扩展：斜切分 Σ_i a_i X_i ≤ c，求和覆盖模型中**全部**自回归项（式(1)，Ch p.97；Cf p.118）。
- **【原文】** 附录论证：若 X_{t-i} − A_t（A_t 为 p 个滞后的均值）与 A_t 弱相关且方差相当，则每个 regime 的协方差 Σ^{(c)} 有一个主轴接近对角线 d=(1/√p,…,1/√p)，故轴向切分难以最优地分开两个 regime（Ch p.97 & Appendix pp.106–107；Cf p.119 & p.125）。**【解读】** 对价格水平来说，滞后值高度共线，数据沿对角线分布，这正是斜切分优于轴向切分的动机。

## 3. 学习算法（逐步）

1. **【原文】 逐阶加入自回归项**：从无回归项的平凡模型开始，每次贪心加入"比当前再往前一步"的自回归项，直到评分不再提高（Ch p.97；Cf p.119）。
2. **【原文】 phase view**：把序列转成 T − p 个样本 (x_{t-p}, …, x_{t-1}, x_t)（含目标）（Ch p.98；Cf p.119）。
3. **【原文】 建树与评分**：用 Meek et al. (2002) 的贝叶斯评分；它惩罚复杂度，**无需剪枝阶段**（Ch p.98；Cf p.119）。具体评分公式和先验**本文未给出**（"described in detail in that paper"）。**【解读】** 贪心生长：对每个叶子，在候选切分中选得分提升最大的；不提升就不切。
4. **【原文】 谱聚类（Fig. 1）**（Ch pp.98–99；Cf p.120）：
   - 亲和度矩阵 A，D = diag(Σ_j a_ij)；L = D − A，L_norm = I − D^{-1/2} A D^{-1/2}；改用 L'_norm = I − L_norm = D^{-1/2} A D^{-1/2}，取**第二大**特征向量 e（等价于 Fiedler 向量）；
   - 对 e 的元素做二聚类（"e.g. by the largest gap in values"），样本 i 归入 e_i 所在的类；
   - 相比 Ng–Jordan–Weiss：只做二分类、只用前两个特征向量、无需行归一化。
5. **【原文】 亲和度**：∝ 1 / (1 + ||p_1 − p_2||_2)，L2 范数，作用在 phase view（**含目标 x_t**）上；无外生变量时所有变量同尺度，无需标准化（Ch p.100；Cf p.121）。**【解读】** 原文分母里 "||p1 − p2||2" 的 "2" 指 L2 范数下标，不是平方；但排版上存在歧义 → 复现中两种都实现，在 search 年上选。
6. **【原文】 斜切分候选**：把聚类结果投影到回归项空间（去掉 x_t），用**单层感知机最小化总误分类数**求分隔超平面 → 得到唯一一个斜切分候选。选感知机是因为它相对 Fisher 判别对离群点更不敏感（Ch p.100；Cf p.121）。
7. **【原文】 时间切分**（Ch pp.101–102；Cf p.121）：
   - **Fixed**：商业实现用"较少的等间距时间点"作候选（点数未给出）；
   - **TimeSpectral（trace view）**：把聚类标签按时间排序，**每次相邻样本跨簇时**提出一个时间切分点 t ≤ c；
   - 时间不作为谱聚类的维度。
8. **【原文】 AxialGaussian**：每个回归项 15 个切分点，按高斯分位数（Chickering et al. 2001）；原文验证了它与穷举轴向切分在 10% 的序列上无显著差异（Ch p.96, p.103；Cf p.118, p.123）。
9. **【原文】 复杂度**：亲和度 O(n²)，特征分解直接法 O(n³)，可用 IRLM（Lanczos）O(mh+nh)，可对亲和度设阈值稀疏化；感知机 O(nh)；trace view 额外 O(n)（Ch pp.100–101；Cf p.121）。
10. **【解读】 预测**：holdout 观测 t 的回归项取实际已观测的 x_{t-1..t-p}（"sequential"），沿树走到叶子，用叶子高斯 AR 给出预测分布；时间切分 t ≤ c 对未来时点总是落在"最新"一侧（原文也指出时间切分让预测只用最新段的信息，Ch p.100；Cf p.121）。

## 4. 数据、协议、基线与指标（论文原协议）

- **【原文】** 数据：S&P 1500 截至 2008-01-01 的成分中 1495 只股票（5 只因数据不足剔除），每条约 150 个交易日、截至 2008-02-01 的日收盘价，来源 Yahoo!（Ch p.102；Cf p.122）。
- **【原文】** 切分：最后 5 个观测为 holdout，其余训练（Ch p.102；Cf p.122）。
- **【原文】** 方法：回归项切分 {AxialGaussian, ObliqueSpectral} × 时间切分 {NoSplit, Fixed, TimeSpectral}，外加 **Baseline**（"a very weak baseline, namely the method not allowing any splits"）；所有方法用同一贝叶斯评分、同一阶数搜索、同一建树方法，只是候选切分不同（Ch pp.102–103；Cf p.122）。
- **【原文】** 指标：平均 sequential predictive score；单侧符号检验（逐序列比较，不计平局，Binomial(0.5)）（Ch p.103；Cf p.122）。

## 5. 论文结果（原样引用）

**Table 1（Ch p.103；Cf p.123）平均 sequential predictive score**

| 回归项切分 | 时间切分 | 平均分 |
|---|---|---|
| Baseline | Baseline | −3.07 |
| AxialGaussian | NoSplit | −1.73 |
| AxialGaussian | Fixed | −1.72 |
| AxialGaussian | TimeSpectral | −1.74 |
| ObliqueSpectral | NoSplit | −1.45 |
| ObliqueSpectral | Fixed | −1.46 |
| ObliqueSpectral | TimeSpectral | −1.44 |

**Table 2（Ch p.104；Cf p.123）逐序列胜负计数（前者更高 / 后者更高，不含平局）**

| | Baseline / AxialGaussian | Baseline / ObliqueSpectral | AxialGaussian / ObliqueSpectral |
|---|---|---|---|
| NoSplit | 118 / 959 | 74 / 1168 | 462 / 615 |
| Fixed | 114 / 990 | 79 / 1182 | 226 / 418 |
| SpectralTime | 122 / 955 | 71 / 1171 | 473 / 604 |

- **【原文】** 符号检验在所有情况下以 < 10^-5 的显著性拒绝相等（Ch p.104；Cf p.123）。
- **【原文】** AxialGaussian→ObliqueSpectral 的提升相对 Baseline→AxialGaussian 的提升"still above 20%"（Ch p.103；Cf p.123）。**【解读】** 按 Table 1：(1.73−1.45)/(3.07−1.73) ≈ 0.21。
- **【原文】** 加时间切分（Fixed 或 TimeSpectral）**不提升**；作者认为轴向/斜切分已足够灵活，并说明该结论未必推广到非股价序列（Ch p.104；Cf p.123）。
- **【原文】** 对所有回归项子集穷举斜切分只带来不显著的提升（Ch p.104；Cf p.123–124）。
- **【解读】** Table 2 的 Fixed 一行 226/418 合计只有 644，远少于其他行（≈1077），意味着该比较中有约 850 个平局，原文未解释。

## 6. 歧义与未明确之处

1. 贝叶斯评分的具体形式与先验（只引用 Meek et al. 2002）。
2. Baseline 的确切含义："不允许任何切分"——是否仍做阶数搜索（AR(p)）？如果是，−3.07 与 −1.73 的差距对股价来说大得惊人（单个 AR(1) 在 145 天价格上通常已接近随机游走）。可能 Baseline 叶子更弱（如无回归项），**无法从原文确定**。
3. 亲和度分母是 L2 距离还是距离平方（排版歧义）。
4. 最大阶数、叶子最小样本数、树深限制均未给出。
5. 感知机的具体训练算法（迭代次数、是否 pocket）。
6. "largest gap" 是否限制两侧最小样本数（否则可能只分出一个离群点）。
7. Fixed 时间切分的等间距点数。
8. 不同阶数 p 的评分比较时，样本集合（T − p 个）不同，如何对齐不清楚。
9. 对数似然以价格单位计，跨股票平均受价格水平影响；是否做了标准化未说明。
10. holdout 预测时叶子参数是否用后验预测（Student-t）还是点估计高斯，未说明。

## 7. 复现实现与假设（`src/paper_repro/`）

| 环节 | 实现 | 性质 |
|---|---|---|
| 叶子模型 | 带截距线性回归，共轭 Normal-Inverse-Gamma 先验 | 【假设】替代 Meek 2002 |
| 评分 | 叶子 log 边际似然之和；切分仅在增益 > 0 时接受，无剪枝 | 【原文】精神 +【假设】形式 |
| 先验 | 经验贝叶斯：m0 = 根节点 OLS 系数，V0 = (X'X_root/n_root)^{-1}（单位信息，n0=1），σ² ~ IG(1, s²_root)，所有叶子共用 | 【假设】 |
| 阶数搜索 | p = 0,1,2,… 逐阶加入，评分不升即停；max_p = 8；所有 p 使用同一组目标 t ≥ max_p，保证评分可比 | 【原文】+【假设】 |
| 谱聚类 | 稠密 A、D^{-1/2}AD^{-1/2} 的第二大特征向量（numpy eigh，n≈140 无需 Lanczos）；最大间隙切分且两侧 ≥ min_leaf | 【原文】+【假设】 |
| 亲和度 | 1/(1+‖·‖)，原始价格（不标准化）；平方距离变体作为超参 | 【原文】+【假设】 |
| 斜切分 | pocket 感知机（标准化输入、类均值差初始化、100 epoch），转换回原尺度 a·x ≤ c | 【原文】+【假设】 |
| AxialGaussian | 每个回归项 μ + σ·Φ^{-1}(i/16), i=1..15 | 【原文】 |
| Fixed | 节点时间范围内 15 个等间距点 | 【假设】点数 |
| TimeSpectral | 按时间排序的聚类标签每次跳变处提一个 t ≤ t_k | 【原文】 |
| 停止规则 | min_leaf（search 上从 {10,20,30} 选），max_depth = 6 | 【假设】 |
| 预测/评分 | 叶子后验预测（Student-t）对数密度，holdout 逐点用真实滞后值 | 【解读】+【假设】 |
| 符号检验 | 单侧精确二项检验，不计平局 | 【原文】 |
| Baseline | 不允许任何切分、但仍做阶数搜索的 AR(p) | 【解读】（见歧义 2） |

## 8. 本项目协议（用户决定）与论文原协议的差异

- **论文**：1495 只股票 × 1 个窗口（≈150 天截至 2008-02-01），最后 5 天 holdout，无超参调节。
- **本项目**：S&P 500 当前成分（500 只有数据，存在幸存者偏差），2016–2026 复权收盘价（yfinance auto_adjust）；年块切分 seed 42、80/20：eval 年 = 2017、2024，其余为 search。
  - episode = 同一股票、同一自然年内连续 150 个交易日（145 训练 + 5 holdout，与论文同规模）；每个 (股票, 年) 取该年前 150 天和最后 150 天两个 episode（2026 年数据截至 2026-10-23，两者部分重叠）。
  - episode 完全落在单一年块内，所以目标期不跨越切分边界；跨边界的样本直接不构造（即丢弃）。
  - 阶段 1（search 年）：每个方法在自己的小网格上选超参；阶段 2（eval 年）：每个方法用选定超参一次性评估。
- 额外参照（论文中没有）：高斯随机游走 RandomWalk(ref)；以及尺度调整对数似然 = LL + log(P_ref)（P_ref 为最后一个训练日收盘价，即对 x/P_ref 的密度，消除价格量纲），以及相对 RMSE。

## 9. 复现结果（全部来自 `outputs/repro_bocharov/` 的实际输出）

### 9.1 合成数据（每个场景 20 个随机种子，训练 400 点；`synthetic_summary.csv`）

| 场景 | 方法 | 平均叶子数 | 叶子 vs 真实 regime ARI | 纯度 | holdout 平均 LL |
|---|---|---|---|---|---|
| S1 四段交替（类似 Fig.2） | Baseline | 1.00 | 0.000 | 0.510 | −2.175 |
| | AxialGaussian+NoSplit | 3.85 | 0.704 | 0.977 | −2.103 |
| | ObliqueSpectral+NoSplit | 2.40 | **0.862** | 0.981 | **−2.085** |
| | ObliqueSpectral+TimeSpectral | 3.50 | 0.759 | 0.981 | −2.080 |
| | 根节点谱聚类（p=2）本身 | – | 0.931 | 0.982 | – |
| S2 SETAR，对角线边界 | Baseline | 1.00 | 0.050 | 0.827 | −1.524 |
| | AxialGaussian+NoSplit | 2.15 | **0.694** | 0.986 | −1.429 |
| | ObliqueSpectral+NoSplit | 2.05 | 0.510 | 0.979 | −1.429 |
| | 根节点谱聚类（p=2）本身 | – | 0.418 | 0.954 | – |
| S3 纯随机游走 | 各方法 | 1.05–1.95 | – | – | −1.371 ~ −1.380 |

结论：
- 谱聚类 + 斜切分能恢复按段交替、可在 phase view 中分离的 regime（S1：根节点聚类 ARI 0.93，斜切分树 ARI 0.86，叶子更少）。
- S2 的 regime 由滞后值的线性组合决定，而 phase view 聚类主要按价格水平分组；这里斜切分不如轴向切分（ARI 0.51 vs 0.69），两者预测 LL 相同。
- 在随机游走上，NoSplit 方法基本不切（约 1.05–1.15 个叶子）。

### 9.2 S&P 500（本项目协议；500 只股票，search 8752 个 episode，eval 1940 个 episode）

超参（仅在 search 上选择）：所有切分方法都选 min_leaf = 30（网格上界）。距离是否平方：Oblique+NoSplit/Fixed 选非平方，两个 TimeSpectral 方法选平方。

**Table 1 对照（eval 2017+2024，n=1940）**

| 方法 | 平均 LL（价格单位，同论文） | 中位数 | 尺度调整 LL | 平均叶子数 | 平均 p |
|---|---|---|---|---|---|
| Baseline（AR(p)，不切） | **−1.6459** | −1.4419 | **2.7484** | 1.00 | 1.05 |
| AxialGaussian+NoSplit | −1.6949 | −1.4802 | 2.6994 | 1.95 | 1.15 |
| AxialGaussian+Fixed | −1.7182 | −1.5115 | 2.6760 | 2.41 | 1.10 |
| AxialGaussian+TimeSpectral | −1.7109 | −1.5032 | 2.6833 | 2.20 | 1.12 |
| ObliqueSpectral+NoSplit | −1.6730 | −1.4382 | 2.7212 | 1.58 | 1.10 |
| ObliqueSpectral+Fixed | −1.7160 | −1.5088 | 2.6782 | 2.33 | 1.06 |
| ObliqueSpectral+TimeSpectral | −1.6997 | −1.4784 | 2.6945 | 1.94 | 1.08 |
| RandomWalk(ref，论文中没有) | −1.6642 | −1.4347 | 2.7300 | – | – |

search 年（n=8752）排序相同：Baseline −1.8150 > Oblique+NoSplit −1.8197 > RandomWalk −1.8286 ≈ Axial+NoSplit −1.8282 > 其余。

**Table 2 对照（eval，前者更高 / 后者更高，不含平局；单侧 p 值检验"后者更好"）**

| | Baseline / Axial | Baseline / Oblique | Axial / Oblique |
|---|---|---|---|
| NoSplit | 741 / 530（p=1.0） | 487 / 334（p=1.0） | 517 / 676（**p=2e-6**） |
| Fixed | 1022 / 597（p=1.0） | 976 / 559（p=1.0） | 313 / 335（p=0.20） |
| TimeSpectral | 898 / 554（p=1.0） | 724 / 437（p=1.0） | 416 / 537（**p=5e-5**） |

search 年 Axial / Oblique：NoSplit 2403 / 2634（p=6e-4），Fixed 1231 / 1372（p=0.003），TimeSpectral 1739 / 1942（p=4e-4）。

分年（eval，尺度调整 LL）：2017 Baseline 3.146、Oblique+NoSplit 3.120、Axial+NoSplit 3.097；2024 Baseline 2.372、Oblique+NoSplit 2.344、Axial+NoSplit 2.323（`sp500_eval_by_year_scaleadj.csv`）。

**诊断（不参与选择）**：若把 Baseline 改成常数均值模型（p=0，不切），平均 LL：eval −3.714，search −3.830（`sp500_diag_constant_mean_baseline.csv`），与论文 Baseline 的 −3.07 同一量级。

### 9.3 结论

- **复现成功的部分**：斜切分 vs 轴向切分的相对结论成立。在 search 和 eval 上，ObliqueSpectral 都显著多次胜过 AxialGaussian（NoSplit 与 TimeSpectral，p ≤ 6e-4），且用更少的叶子。"时间切分不提升"也成立：Fixed 和 TimeSpectral 都比 NoSplit 差。
- **未复现的部分**：论文中"任何切分方法都大幅优于 Baseline（−3.07 → −1.45）"。在我们的实现中，不切分的 AR(p) Baseline 才是最好的 ART 变体，切分模型在 holdout 上反而略差（过拟合）。所有方法都与随机游走处于同一水平（平均选到 p≈1，系数接近 1）。
- **【解读】** 最可能的原因：
  1. 论文的 Baseline 很可能远弱于 AR(p)。我们的诊断显示常数均值模型约 −3.7，接近 −3.07。
  2. 我们的贝叶斯评分/先验与 Meek et al. 2002 不同，对切分的惩罚强度不同。
  3. 数据期间（2016–2026 vs 2007 年末–2008 年初）与股票池不同。
  4. 价格单位的 LL 本身不可跨研究直接比较。


## 10. 收益率版本复现（2026-10-26，returns_exp.py）

### 10.1 设置（与 §9 价格版唯一的区别是目标序列）
- 序列：r_t = log(P_t/P_{t-1})。目标为 r_t，回归量为 r_{t-1..t-p}。ART 和谱聚类都作用在收益率元组上。
- 样本：每个 episode 取同一自然年内 151 个价格，得到 150 个收益率（145 训练 + 5 验证）。每个 ticker-年取首、尾两个 episode。样本覆盖同样的 500 只股票。
- 切分：年块切分，seed 42。只在 search 年调参（n=8752）；eval 年为 2017 和 2024，只跑一次（n=1940）。
- 方法：Baseline AR(p)（不分裂），以及 Oblique/Axial × NoSplit/TimeSpectral/Fixed，超参网格与 §9 相同。
- 参考基线：
  - ZeroMean：N(0, σ̂²)，相当于价格随机游走。
  - ConstMean：N(μ̂, σ̂²)。
- 注意：p=0 时没有回归量，NoSplit 方法无法分裂，此时等同于 Baseline（表中 ties 很多就是这个原因）。
- 变体 **[STD]**：先把收益率除以训练段标准差 s 再建模，预测时变换回原尺度（LL − log s）。目的是让谱聚类亲和度 1/(1+d) 不再因为 d≈0.02 而接近常数。
- 输出：outputs/repro_bocharov/returns/（主结果）和 returns_std/（标准化变体）。

### 10.2 Eval（2017+2024，n=1940）——主版本（原始收益率）

| 方法 | mean logscore | median | MSE×1e4 | MAE×1e2 | 方向命中 | 分裂占比 |
|---|---|---|---|---|---|---|
| Baseline AR(p) | **2.7514** | 3.0166 | 3.401 | 1.121 | 0.469 | 0 |
| Axial NoSplit | 2.7205 | 3.0246 | 3.563 | 1.142 | 0.472 | 0.40 |
| Axial Fixed | 2.7092 | 2.9924 | 3.490 | 1.148 | 0.488 | 0.76 |
| Axial TimeSpectral | 2.6971 | 2.9746 | 3.519 | 1.144 | 0.491 | 0.75 |
| Oblique NoSplit | 2.7490 | 3.0174 | 3.422 | 1.126 | 0.466 | 0.08 |
| Oblique Fixed | 2.7191 | 2.9948 | 3.456 | 1.141 | 0.485 | 0.69 |
| Oblique TimeSpectral | 2.6980 | 2.9634 | 3.434 | 1.139 | 0.485 | 0.71 |
| ZeroMean(ref) | 2.7437 | 3.0237 | **3.297** | **1.099** | – | – |
| ConstMean(ref) | 2.7313 | 3.0185 | 3.381 | 1.117 | 0.471 | – |

[STD] 变体的 eval 结果与上表基本一致：
- Oblique NoSplit 为 2.7465，分裂占比 0.15。
- Oblique Fixed 为 2.7153，Oblique TimeSpectral 为 2.6979。
- Axial TimeSpectral 为 2.6968，其余 Axial 方法与上表相同。

### 10.3 Sign tests（每个样本的 logscore 胜/负）

| 比较 | Eval 主版本 | Eval [STD] | Search 主版本 |
|---|---|---|---|
| Oblique vs Axial, NoSplit | 410 vs 377，p=0.13 | 400 vs 374，p=0.18 | 1329 vs 1687（Axial 更好，p=4e-11） |
| Oblique vs Axial, Fixed | 291 vs 264，p=0.14 | 291 vs 256，p=0.07 | 922 vs 796（Oblique 更好，p=1e-3） |
| Oblique vs Axial, TimeSpectral | 201 vs 212，ns | 197 vs 215，ns | 597 vs 589，ns |
| Axial Fixed vs Baseline | 664 vs 812（Baseline 更好，p=6e-5） | 同 | 3991 vs 2840（Fixed 更好，p=2e-44） |
| Axial TimeSpectral vs Baseline | 596 vs 854（p=7e-12） | 585 vs 856 | 3778 vs 2937（TimeSpectral 更好） |
| Oblique Fixed vs Baseline | 602 vs 739（p=1e-4） | 606 vs 751 | 3810 vs 2647（Fixed 更好，p=6e-48） |
| Oblique TimeSpectral vs Baseline | 522 vs 845（p=1e-18） | 523 vs 830 | 3611 vs 2844（TimeSpectral 更好，p=7e-22） |
| NoSplit (Axial/Oblique) vs Baseline | ns / ns | ns / ns | 两者都更好（p≤6e-11） |
| ZeroMean vs Baseline | 1299 vs 641（ZeroMean 更好，p=1e-51） | 同 | 5332 vs 3420（p=9e-94） |
| ConstMean vs Baseline | 1180 vs 760（ConstMean 更好，p=6e-22） | 同 | 5319 vs 3433 |

说明：
- 按均值，Baseline 最好；按胜场和中位数，ZeroMean 最好。这说明 Baseline 的均值优势来自尾部样本：Student-t 预测分布在极端日的得分更好。
- Search 年的均值：Oblique Fixed 最高（2.6454），Baseline 为 2.6281。

### 10.4 按年拆分：logscore 相对 Baseline 的差值（all_years 文件含全部年份）
| 年份 | Oblique Fixed | Axial Fixed | Axial NoSplit |
|---|---|---|---|
| 2018 | +0.060 | +0.053 | −0.016 |
| 2020 | **+0.177** | **+0.177** | +0.061 |
| 2022 | +0.034 | +0.030 | 0.000 |
| 2017 (eval) | −0.037 | −0.043 | −0.012 |
| 2024 (eval) | −0.028 | −0.042 | −0.049 |
| 2021 / 2026 | −0.056 / −0.055 | −0.056 / −0.071 | −0.011 / −0.029 |

Search 年里分裂方法的优势几乎全部来自波动冲击年（2020、2018、2022）。在平静年份，分裂方法反而落后。因此 eval 的结论不是偶然，而是年份决定的。数据见 returns/by_year_delta_vs_baseline_all_years.csv。

### 10.5 分段是否捕捉了波动率 regime
指标 var_ratio = 叶子 E[σ²] 的最大值 / 最小值，与相同叶子大小的随机划分对比。以下为 Eval 主版本结果：

| 方法 | var_ratio 中位数 | 随机划分中位数 | 高于随机划分的占比 |
|---|---|---|---|
| Axial NoSplit | 3.52 | 2.00 | 0.90 |
| Axial Fixed | 3.98 | 2.05 | 0.94 |
| Axial TimeSpectral | 3.14 | 1.75 | 0.93 |
| Oblique NoSplit | 2.83 | 1.80 | 0.84 |
| Oblique Fixed | 3.65 | 2.01 | 0.91 |
| Oblique TimeSpectral | 3.01 | 1.72 | 0.90 |

Search 年结果相近（占比 0.85–0.97），[STD] 也相近。结论：分裂确实主要在切分波动率水平，而不是切分均值动态。

### 10.6 与价格版（§9）对照（eval，n=1940）
| | 价格版 | 收益率版 |
|---|---|---|
| Baseline | −1.6459（尺度调整后 2.7484） | 2.7514 |
| 随机游走 / ZeroMean | −1.6642（2.7300） | 2.7437 |
| Oblique NoSplit | −1.6730（2.7212） | 2.7490 |
| Axial NoSplit | −1.6949（2.6994） | 2.7205 |
| Oblique vs Axial (NoSplit) | 676 vs 517，p=2e-6 | 410 vs 377，p=0.13 |
| Oblique vs Axial (TimeSpectral) | 537 vs 416，p=5e-5 | 201 vs 212，ns |
| 时间分裂 vs 不分裂 | 更差 | 更差（search 年在冲击年更好） |
| 分裂方法 vs Baseline | 全部更差 | eval 全部更差；search 全部更好 |

尺度调整后的价格 LL 与收益率 LL 量纲近似可比，因为 log(P_t/P_ref) 和 log(P_t/P_{t-1}) 的 Jacobian 都约等于 1。两个版本的 Baseline 只差约 0.003。

### 10.7 解读
1. 收益率上几乎不存在可预测的均值：方向命中率 ≤0.50，MSE 最低的是 ZeroMean，大部分 p 被选为 0。
2. ART 在收益率上的作用，基本等同于一个分段常数波动率模型。它在波动冲击年（2020 等）明显占优，在平静年因为多估了参数而吃亏。
3. 论文的 "oblique > axial" 只在价格版成立。在收益率上，二者差异不显著，原因是可以斜切的回归量很少。标准化以后结论不变，所以并不是亲和度尺度造成的。
4. 下一步若要利用波动率结构，应该直接比较 GARCH/EWMA 等波动率模型，而不是继续沿用 AR 树。
