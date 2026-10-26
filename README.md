# sp500-trend：S&P 500 线性趋势扫描 + 论文复现

仓库分为两个**相互独立**的部分（同一个 uv 项目、两个 Python 包）：

| 部分 | 包 | 命令 | 内容 |
|---|---|---|---|
| A. 我们自己的处理流程 | `src/trend_scan/` | `uv run trend-scan ...` | 成分股、日线下载、趋势区间扫描与线性拟合、年块切分实验记录、报告、ML 时点特征 |
| B. 论文复现 | `src/paper_repro/` | `uv run paper-repro ...` | Bocharov & Thiesson (2012/2013) 谱聚类斜切分 ART 的复现（价格版 + 收益率版） |

依赖方向是单向的：`paper_repro` 只从 `trend_scan` 读取路径常量与价格缓存（`trend_scan.paths`），`trend_scan` 不导入 `paper_repro`。

> ⚠️ trend_scan stage 1 **不假设**任何预测能力或盈利能力。检测出的上涨/下跌区间需要用到窗口末日价格，
> 属于**事后（ex-post）描述**，**不能**视为当时可执行的交易信号。

## 目录结构

```
sp500_trend/
├── pyproject.toml            # uv 项目（hatchling 构建），定义 trend-scan / paper-repro 两个命令
├── uv.lock                   # 锁定的依赖版本（已提交）
├── .python-version           # 3.13
├── configs/
│   └── trend_scan/default.yaml   # trend_scan 参数网格、分析区间、切分、特征配置
├── src/
│   ├── trend_scan/           # A. 我们的流程
│   │   ├── cli.py            #   命令行入口（原 run.py）
│   │   ├── paths.py          #   路径常量（仓库根目录；可用 SP500_TREND_ROOT 覆盖）
│   │   ├── universe.py       #   股票池：维基百科抓取 / 自定义列表（含 GICS 行业列）
│   │   ├── data.py           #   yfinance 下载、缓存、失败记录
│   │   ├── scan.py           #   趋势区间识别（detection，事后）
│   │   ├── fit.py            #   对数价格 OLS 拟合（前缀和闭式解，全向量化）
│   │   ├── conditions.py     #   条件函数占位 condition(features)->bool（stage 2）
│   │   ├── experiment.py     #   参数网格、随机年块切分、实验记录
│   │   ├── features.py       #   时点（point-in-time）ML 特征数据集
│   │   ├── features_check.py #   特征完整性检查（无前视 / 与 scan 一致 / 标签）
│   │   └── report.py         #   图表与自动报告（只读真实输出）
│   └── paper_repro/          # B. 论文复现
│       ├── cli.py            #   命令行入口
│       ├── bayes.py, spectral.py, art.py   # 叶子 NIG 评分、谱聚类、ART 学习器
│       ├── synthetic.py      #   合成数据验证
│       ├── sp500.py          #   价格水平版 S&P 500 实验
│       └── returns_exp.py    #   日对数收益率版实验
├── docs/paper_repro/bocharov2012_notes.md   # 论文中文笔记 + 复现结果（§9 价格版，§10 收益率版）
├── experiments/              # [提交] 每次 scan 的实验目录与 index*.csv（小文件）
├── reports/                  # [提交] report.md、PNG 图表、汇总 CSV
├── outputs/repro_bocharov/   # [提交汇总表；逐样本 *.csv.gz / episodes.csv 不提交] 论文复现输出
├── data/                     # [不提交] 价格缓存、processed、features（可重建，约 600 MB）
├── papers/                   # [不提交] 论文 PDF / 全文 txt（版权材料，仅本地）
└── logs/                     # [不提交] 运行日志
```

`data/` 下的布局（运行后生成）：`data/raw/universe/`（成分股列表）、`data/raw/prices/`（每只股票一个 parquet + `_download_log.csv`、`_failed.csv`）、
`data/processed/w{N}_u{up}_d{down}/`（up/down parquet + CSV 抽样）、`data/processed/ticker_groups.csv`、`data/features/year=YYYY/`（ML 特征）。

## 安装（uv）

```bash
uv sync                       # 按 uv.lock 创建 .venv 并以可编辑模式安装本项目（Python 3.13）
```

- 依赖全部在 `[project.dependencies]`（pandas、numpy、requests、lxml、yfinance、pyarrow、matplotlib、PyYAML）；
  paper_repro 实际只用到 numpy/pandas/pyarrow，但两部分共用一个环境，未拆分 optional 组。
- 不再维护 `requirements.txt`。需要 pip 时可临时生成：`uv export --no-hashes --no-dev -o requirements.txt`。
- 所有输出路径相对于仓库根目录（`trend_scan.paths.ROOT`），与在哪个目录下运行命令无关。

## A. trend_scan 使用方法

```bash
uv run trend-scan universe                 # 抓取维基百科当前成分股
uv run trend-scan download                 # 下载日线（已缓存的跳过）
uv run trend-scan download --refresh       # 强制重新下载
uv run trend-scan scan                     # 跑参数网格 + 生成报告（默认 --config configs/trend_scan/default.yaml）
uv run trend-scan report                   # 仅重新生成报告与图表
uv run trend-scan all                      # 下载 + 扫描 + 报告
uv run trend-scan features                 # 构建 ML 特征数据集 + 完整性检查
uv run trend-scan features --check-only    # 仅重跑检查（结果见 reports/features_sanity.json）
# 调试：--limit 10 只处理前 10 只；自定义股票池：--tickers-file my_list.csv（含 ticker 列）
```

## A. trend_scan：方法说明

### A1. 股票池
- 来源：维基百科 *List of S&P 500 companies*（requests + 浏览器 User-Agent + `pandas.read_html`）。
- 代码转换为 yfinance 格式（`BRK.B -> BRK-B`），保存到 `data/raw/universe/sp500_constituents_YYYYMMDD.csv`（含 `scrape_date`）。
- **幸存者偏差**：v1 只用*当前*成分股，不含历史上被剔除或已退市的股票。越往前的历史，样本越偏向“后来存活并壮大”的公司。
  `universe.load_universe(source="file", tickers_file=...)` / `--tickers-file` 可接入任意股票列表（例如历史成分股+退市股），流程其余部分不变。

### A2. 日线数据
- yfinance，`period="max"`，`auto_adjust=True`：使用复权后的 `Close`（拆股 + 分红调整，近似总收益口径），保证跨拆股/分红的收益可比。
- 注意：复权价格会随未来新的分红/拆股整体回溯调整，因此下载时间记录在 `_download_log.csv`，不同时间下载的数据可能略有差异。
- 分批下载（默认 50 只/批），失败的进行多轮重试，最后逐只兜底；仍失败的写入 `_failed.csv`。幂等：已缓存的跳过，除非 `--refresh`。

### A2b. 分析区间：2016-01-01 至最新（约 10 年）
- 下载仍缓存全部历史（`period="max"`），但扫描与拟合只使用 `configs/trend_scan/default.yaml` 中 `data.start`/`data.end` 指定区间（默认 `2016-01-01` ~ 最新可用日）。
- **为什么只用近 10 年？** 股票池是*当前*成分股，回溯越久，样本越集中于“后来存活并进入指数”的公司，幸存者偏差越大；
  限定在近 10 年可以显著降低这种偏差（但不能消除：期间被剔除/退市的公司仍未包含）。另一个附带好处是：
  Yahoo 早期历史中的明显错误报价（例如 NVR 1993、HUBB 1994 的单日数十倍跳变）均在 2016 年以前。
- 区间内少于 `min_history_days`（默认 100）个交易日的新上市/新分拆股票不参与扫描。
- 数据质量：区间内单日 |对数收益| > ln2 的跳变写入 `reports/data_quality_flags.csv`，**仅标记不删除**，需人工核查。

### A3. 趋势区间识别（`trend_scan/scan.py`）
- 固定长度 N 的窗口（N = 窗口内交易日个数，含首尾），逐日滑动；`cum_ret = P_end / P_start - 1`。
- `cum_ret >= +8%` → `up`；`cum_ret <= -8%` → `down`；其余不标记。阈值与 N 均可配置；保留所有满足条件的窗口（允许重叠）。
- **默认窗口网格：5, 10, 20, 30, 40 个交易日（不超过 40）。** 原因：使用 yfinance 的*当前*成分股存在幸存者偏差，
  而存活下来的成分股在长期上天然偏向上涨趋势，窗口越长，这种偏差对“上涨/下跌区间”的数量和形态影响越大
  （长窗口更容易把长期上行的漂移计入上涨区间）。因此 stage 1 只研究较短窗口；更长窗口需在接入历史成分股/退市股后再研究。

### A4. 线性拟合（`trend_scan/fit.py`）
- 对窗口内 `log(price)` 关于交易日序号 `x = 0..N-1` 做 OLS。
- 记录：`slope`、`intercept`（x=0 处拟合对数价格）、`r2`、`resid_std`（自由度 N-2）、`max_abs_resid`、`slope_tstat`、
  `slope_annual = slope × 252`、`growth_annual = exp(slope×252) - 1`。
- 实现：用 `y, j·y, y²` 的前缀和一次性得到所有滑动窗口的闭式 OLS 解（已与 `np.polyfit` 抽样核对，误差 < 1e-7）；
  最大绝对残差仅对被标记窗口用 `sliding_window_view` 计算。
- 注意：重叠窗口高度相关，样本数不等于独立观测数；`slope_tstat` 仅为窗口内描述量（残差有自相关），不应做显著性解读。

### A5. 参数扫描与实验记录（`trend_scan/experiment.py`）
- 每个（窗口长度 × 阈值对）组合为一个实验，id 形如 `20261026T110000_w20_u0.08_d-0.08_<配置哈希>`。
- 每个实验目录保存：完整配置、元数据（时间戳、数据日期范围、股票数、包版本、切分信息）、汇总统计
  （各 split 的 up/down 数量、占全部窗口比例、累计收益/斜率均值与中位数、R² 分位数、通过各 R² 阈值的比例等）、年份→split 映射。
- `experiments/index.csv` 每次运行追加，便于跨实验比较。

### A6. 时间切分：随机年块切分（固定种子，80/20）
- 将分析区间（默认 2016–2026，共 11 个年块）按**日历年**（`block_unit` 可选 year / quarter / month）划分为块，以 `seed=42` 随机把约 **80%** 的年份分到 `search`（参数搜索），约 **20%** 分到 `eval`（独立评估）。
- 每个窗口的首日与末日所在年份必须同属一个 split，否则标记为 `excluded`（不参与 search/eval 统计）。年份→split 映射保存在每个实验的 `split_mapping.csv` 与 `metadata.json`。
- **为什么按块而不是按单个窗口随机切分？** 滑动窗口逐日移动、大量重叠：相邻窗口共享绝大部分价格。
  若按窗口随机分配，search 与 eval 中会出现几乎相同的窗口，评估集“看过”搜索集的数据（信息泄漏），评估结果会被高估。
  按整年分块并剔除跨不同 split 的窗口，可保证两组样本在时间上不重叠；同时随机抽年比单一日期截断更能覆盖不同的市场环境。
- 局限：相邻年份之间仍存在市场状态的连续性（非完全独立）；年份映射依赖数据起止年份，更换股票池后请以保存的映射为准。

### A6b. 行业分组（GICS）
- 分组信息取自已保存的维基百科成分股表：`GICS Sector`（11 个一级行业）与 `GICS Sub-Industry`，
  映射保存为 `data/processed/ticker_groups.csv`，两列（`sector`、`sub_industry`）都会写入每个 `up/down.parquet`。
- 配置 `grouping.by: sector`（默认）或 `sub_industry` 决定汇总按哪一列分组；`grouping.r2_threshold` 默认 0.8。
- 每个实验额外输出 `experiments/<id>/summary_by_<group>.csv`，并追加到 `experiments/index_by_group.csv`；
  统计口径为 窗口长度 × split（all / search / eval）× 分组：组内股票数、全部窗口数、up/down 数量与占比、up/down 比值、
  斜率中位数、R² 中位数、R² ≥ 阈值占比。
- 报告图表：`reports/heatmap_median_r2_<group>_x_window.png`（search）及 `_eval.png`（各分组 × 窗口长度的上涨/下跌 R² 中位数），
  `reports/counts_by_<group>_w20.png`（窗口 20 下各组 up/down 数量）；明细见 `reports/summary_by_<group>_latest_run.csv`。
- **注意**：① 各行业股票数差异大（约 21–82 只），样本量不同，小组统计波动更大；
  ② 分组是*当前* GICS 分类，公司历史上可能属于其他行业（例如 2018 年 GICS 调整新建“通信服务”），
  这与幸存者偏差叠加，行业间比较只能作描述参考；③ sub_industry 组数多、每组股票很少，统计更不稳定。

### A7. 检测 / 拟合 / 预测 的区分
- `scan.py`（检测）：用窗口内未来价格事后识别，**仅用于描述**。
- `fit.py`（拟合）：描述窗口内价格路径的线性程度。
- `conditions.py`（未来条件函数）：只能使用判断时刻之前的信息（如过去 N 日的累计收益、斜率、R²）构造特征，
  当前仅为占位接口，未接入任何交易或回测；参数只能在 search 块上选择，在 eval 块上一次性评估。

### A8. ML 特征数据集（`trend_scan/features.py`，时点 / point-in-time）
每行一个 (ticker, date)，date ∈ [2016-01-01, 最新]，与 scan 同一批 501 只股票。读取：
`pd.read_parquet("data/features")`（可加 `filters=[("year", ">=", 2020)]`）。完整列定义见 `data/features_dictionary.md`。

**特征（只用 ≤ t 的价格）**：对每个 N ∈ {5,10,20,30,40}，以 t 为**末日**的回看窗口 [t-N+1, t]：
- `cum_ret_N` = P_t / P_(t-N+1) − 1；`slope_N`、`r2_N`、`resid_std_N`、`slope_tstat_N`：log 价格对 0..N-1 的 OLS；
- `resid_last_N`：t 当天相对拟合线的残差（log 价格偏离）；
- `in_up_N` / `in_down_N`：`cum_ret_N` ≥ +8% / ≤ −8%（与 scan 标签同一定义，但**窗口在 t 结束**，因此在 t 时刻可知）；
- 历史不足 N 天时为 NaN。热身（warm-up）使用 2016 年以前的历史价格，因此 2016 年初的行也有值。
- `feat_window_crosses_split_40`：40 日回看窗口跨越了 search/eval 年块边界（特征本身不含未来信息，但其价格来自相邻块）。

**标签（使用未来数据，统一以 `fwd_` 为前缀）**：`fwd_ret_H` = P_(t+H)/P_t − 1，`fwd_logret_H` 为对数版本，H ∈ {5,10,20}；
每只股票最后 H 行为 NaN。`fwd_crosses_split_H` = t 与 t+H 不在同一 split。

**防泄漏规则**：
1. 训练时只能把非 `fwd_` 列当作输入；`fwd_*` 只能作为目标。`adj_close` 为非平稳原始价格，不建议直接入模。
2. 在 search 年份上训练 / 调参时，删除 `fwd_crosses_split_H == True` 的行（否则标签会用到 eval 年份的价格）；
   更严格的做法是同时删除 `feat_window_crosses_split_40 == True` 的行。
3. 标签窗口重叠（相邻日的 `fwd_ret_20` 共享 19 天），样本高度自相关：交叉验证需按时间分块，并做 purge（删除标签区间与验证集重叠的训练样本）+ embargo（验证集之后空出 ≥ H 天）；不要用随机 K 折。
4. eval 年份（2017、2024）只做一次性最终评估。
5. `sector` / `sub_industry` 为**当前**分类，股票池为**当前**成分股：存在幸存者偏差与事后分类偏差，回测结果会偏乐观。

**完整性检查**（`trend_scan/features_check.py`）：随机抽样行，用截断到 t 的价格序列重算 + `np.polyfit` 直接拟合对比；
scan 中每个以 t 为末日的 ±8% 窗口与 t 行特征逐一对比（数值、标志、计数）；`fwd_ret_H` 与平移后的价格对比；
`fwd_crosses_split_H` 与切分映射对比。`slope_tstat` 在 R²≈1 时对极小 SSE 敏感，容差放宽至 1e-3（相对）。

## B. paper_repro：Bocharov & Thiesson (2012/2013) 谱聚类斜切分 ART

- 论文笔记（中文，含原文/解读/假设标注与页码，以及全部复现结果）：`docs/paper_repro/bocharov2012_notes.md`。
  论文 PDF / 全文 txt 不随仓库分发（本地放在被忽略的 `papers/`）。
- 实现：`src/paper_repro/`（纯 numpy，无 scipy 依赖）。sp500 / returns 需要先有 trend_scan 的价格缓存（`uv run trend-scan download`）。

```bash
uv run paper-repro synthetic               # 合成数据验证 → outputs/repro_bocharov/synthetic_*.csv（约 1 分钟）
uv run paper-repro sp500 [--ticker-frac F] # 价格水平版：search 年选超参 → eval 2017/2024 一次性评估 → outputs/repro_bocharov/sp500_*（8 进程约 20–40 分钟）
uv run paper-repro returns [--standardize] # 日对数收益率版（含 ZeroMean/ConstMean 参考与波动率 regime 诊断）→ outputs/repro_bocharov/returns/（或 returns_std/），约 15 分钟
```

- `paper-repro` 会自动设置 `OMP_NUM_THREADS=1`（若未设置），等价于旧版命令行中手动设置的环境变量。
- 也可直接 `uv run python -m paper_repro.sp500 [ticker_frac]` / `python -m paper_repro.returns_exp [--standardize]` / `python -m paper_repro.synthetic [out_dir]`。
- 价格版预测目标与论文一致：单只股票日收盘价（价格水平）的一步预测，指标为 holdout 5 天的平均一步预测对数似然（sequential predictive score）+ 单侧符号检验；
  收益率版目标为 r_t = log(P_t/P_{t-1})，另报 MSE/MAE/方向命中率。结论摘要见笔记 §9、§10。

## 已知局限
- 幸存者偏差（见上，已通过“近 10 年 + 窗口 ≤ 40 日”加以限制，但未消除）；yfinance 数据质量与复权口径依赖 Yahoo。
- 年块只有 11 个，eval 只有约 2 个年份，评估结果受这两年市场环境影响很大。
- 重叠窗口导致样本强相关；统计量仅作描述。
- ML 特征的标签窗口重叠、行业为当前分类；MRNA 2026-08-19 等 6 处大幅跳变仅标记未剔除（见 reports/data_quality_flags.csv），会进入特征与标签。
