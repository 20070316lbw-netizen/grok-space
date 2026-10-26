"""时点（point-in-time）特征数据集：每行 = (ticker, date)。

特征（前缀无 fwd_）只使用 date 当日及之前的复权收盘价：
  对每个 N，以 t 为窗口*末日*的回看窗口 [t-N+1, t]（与 scan.py 窗口定义一致：N 个交易日，含首尾）。
标签（前缀 fwd_）使用 t 之后的价格，只能作为监督学习的目标，绝不能作为特征。

复用 fit.rolling_ols_log 的前缀和闭式 OLS：rolling 结果按窗口起点 i 索引，
回看窗口末日 e 对应起点 i = e-N+1。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import fit as fitmod
from .experiment import block_ids
from .paths import EXPERIMENTS_DIR, ROOT

log = logging.getLogger(__name__)
FEATURES_DIR = ROOT / "data" / "features"


# ----------------------------------------------------------------------------- 计算
def trailing_features(price: np.ndarray, n: int, up: float, down: float,
                      ann_days: int = 252) -> dict[str, np.ndarray]:
    """对每个日期 t（索引 e）计算以 t 结尾的 N 日回看特征；e < N-1 时为 NaN。"""
    T = len(price)
    out = {k: np.full(T, np.nan) for k in ("cum_ret", "slope", "r2", "resid_std", "resid_last", "slope_tstat")}
    if T < n:
        return out
    ols = fitmod.rolling_ols_log(price, n, ann_days)
    e = np.arange(n - 1, T)
    out["cum_ret"][e] = price[e] / price[e - n + 1] - 1.0
    for k in ("slope", "r2", "resid_std", "slope_tstat"):
        out[k][e] = ols[k]
    # 最后一点残差：log P_t - (intercept + slope*(N-1))
    out["resid_last"][e] = np.log(price[e]) - (ols["intercept"] + ols["slope"] * (n - 1))
    return out


def forward_returns(price: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """fwd_ret_H = P_{t+H}/P_t - 1 及对数版本；末尾不足 H 天为 NaN。"""
    T = len(price)
    r = np.full(T, np.nan)
    if T > h:
        r[: T - h] = price[h:] / price[: T - h] - 1.0
    return r, np.log1p(r)


def load_split_mapping(cfg: dict, first_date, last_date) -> dict[int, str]:
    """优先读取最近一次实验保存的 split_mapping.csv（保证与扫描实验一致），否则按配置重新生成。"""
    from .experiment import make_block_split
    unit = cfg["split"].get("block_unit", "year")
    files = sorted(EXPERIMENTS_DIR.glob("*/split_mapping.csv"))
    computed = make_block_split(first_date, last_date, cfg["split"])
    if files:
        m = pd.read_csv(files[-1])
        saved = dict(zip(m["block_id"].astype(int), m["split"]))
        if saved != computed:
            log.warning("已保存的 split 映射与按当前配置计算的不一致，使用已保存映射：%s", files[-1])
        return saved
    return computed


def build_ticker_frame(t: str, s_full: pd.Series, start: pd.Timestamp, fc: dict,
                       mapping: dict[int, str], unit: str, dtype) -> pd.DataFrame:
    p = s_full.to_numpy(dtype=float)
    dates = s_full.index.to_numpy(dtype="datetime64[ns]")
    keep = dates >= np.datetime64(start)
    cols: dict[str, np.ndarray] = {}
    for n in fc["windows"]:
        f = trailing_features(p, n, fc["up"], fc["down"])
        for k, v in f.items():
            cols[f"{k}_{n}"] = v
        cr = f["cum_ret"]
        valid = np.isfinite(cr)
        cols[f"in_up_{n}"] = np.where(valid, cr >= fc["up"], np.nan)
        cols[f"in_down_{n}"] = np.where(valid, cr <= fc["down"], np.nan)
    # 回看窗口跨 split 标记（以最长窗口为准）：特征窗口跨越 search/eval 年份边界
    bid = block_ids(dates, unit)

    def split_of(b):
        return np.array([mapping.get(int(x), "outside") for x in b], dtype=object)
    sp = split_of(bid)
    nmax = max(fc["windows"])
    cross_feat = np.full(len(p), np.nan)
    if len(p) >= nmax:
        e = np.arange(nmax - 1, len(p))
        cross_feat[e] = (sp[e - nmax + 1] != sp[e])
    cols[f"feat_window_crosses_split_{nmax}"] = cross_feat
    # 标签
    for h in fc["horizons"]:
        r, lr = forward_returns(p, h)
        cols[f"fwd_ret_{h}"] = r
        cols[f"fwd_logret_{h}"] = lr
        c = np.full(len(p), np.nan)
        if len(p) > h:
            c[: len(p) - h] = (sp[h:] != sp[: len(p) - h])
        cols[f"fwd_crosses_split_{h}"] = c
    df = pd.DataFrame(cols)
    df.insert(0, "date", s_full.index)
    df.insert(0, "ticker", t)
    df["adj_close"] = p
    df["split"] = sp
    df = df[keep].reset_index(drop=True)
    for c in df.columns:
        if c.startswith(("in_up_", "in_down_", "fwd_crosses_split_", "feat_window_crosses_split_")):
            df[c] = df[c].astype("boolean") if df[c].notna().all() else \
                pd.array(np.where(df[c].isna(), None, df[c] == 1), dtype="boolean")
        elif df[c].dtype == float:
            df[c] = df[c].astype(dtype)
    return df


def feature_dictionary(fc: dict) -> pd.DataFrame:
    up, down = fc["up"], fc["down"]
    rows = [("ticker", "id", "股票代码（yfinance 格式）", "否"),
            ("date", "id", "交易日 t", "否"),
            ("year", "id/分区", "date 的年份（parquet 分区列）", "否"),
            ("sector", "context", "当前 GICS Sector（维基百科，存在事后分类偏差）", "否（但为当前分类，有偏）"),
            ("sub_industry", "context", "当前 GICS Sub-Industry", "否（但为当前分类，有偏）"),
            ("split", "context", "date 所在年份的 search/eval 归属（seed 42，80/20 年块）", "否"),
            ("adj_close", "raw", "复权收盘价 P_t（非平稳，不建议直接作为特征）", "否")]
    for n in fc["windows"]:
        rows += [
            (f"cum_ret_{n}", "feature", f"P_t / P_(t-{n - 1}) - 1，回看 {n} 个交易日（含 t）", "否"),
            (f"slope_{n}", "feature", f"窗口 [t-{n - 1}, t] 内 log(P) 对 x=0..{n - 1} 的 OLS 斜率（日对数斜率）", "否"),
            (f"r2_{n}", "feature", "同一回归的 R²", "否"),
            (f"resid_std_{n}", "feature", f"同一回归的残差标准差（自由度 {n}-2）", "否"),
            (f"resid_last_{n}", "feature", "最后一点（t）的残差 log P_t - 拟合值", "否"),
            (f"slope_tstat_{n}", "feature", "斜率 t 统计量（残差自相关，仅作描述性强度指标）", "否"),
            (f"in_up_{n}", "feature(flag)", f"cum_ret_{n} >= {up}（scan 上涨标签的时点版本：以 t 为窗口末日）", "否"),
            (f"in_down_{n}", "feature(flag)", f"cum_ret_{n} <= {down}", "否"),
        ]
    nmax = max(fc["windows"])
    rows.append((f"feat_window_crosses_split_{nmax}", "leakage-flag",
                 f"{nmax} 日回看窗口首日与 t 的年份属于不同 split（特征使用了另一 split 的价格）", "否"))
    for h in fc["horizons"]:
        rows += [
            (f"fwd_ret_{h}", "LABEL", f"P_(t+{h}) / P_t - 1（未来 {h} 个交易日简单收益）", "是（未来数据，仅作标签）"),
            (f"fwd_logret_{h}", "LABEL", f"log(P_(t+{h}) / P_t)", "是（未来数据，仅作标签）"),
            (f"fwd_crosses_split_{h}", "leakage-flag", f"t 与 t+{h} 的年份属于不同 split；训练时应剔除", "是（依赖未来日期）"),
        ]
    return pd.DataFrame(rows, columns=["column", "role", "definition", "uses_future_data"])


def build_features(cfg: dict, tickers: list[str], groups: pd.DataFrame | None, out_dir: Path = FEATURES_DIR) -> dict:
    """主入口：生成并写出分区 parquet + 特征字典，返回元数据。"""
    from .data import load_close
    t0 = time.time()
    fc = {"windows": [5, 10, 20, 30, 40], "horizons": [5, 10, 20], "up": 0.08, "down": -0.08,
          "warmup_history": True, "dtype": "float32", **(cfg.get("features") or {})}
    dc = cfg["data"]
    start = pd.Timestamp(fc.get("start") or dc.get("start") or "1900-01-01")
    end = fc.get("end") or dc.get("end")
    dtype = np.dtype(fc["dtype"])
    # 股票集合与扫描一致：分析区间内至少 min_history_days 个交易日
    in_range = load_close(tickers, dc["price_field"], dc["min_history_days"], str(start.date()), end)
    full = load_close(list(in_range), dc["price_field"], 1, None, end) if fc["warmup_history"] else in_range
    all_d = pd.DatetimeIndex(sorted(set().union(*[s.index[[0, -1]] for s in in_range.values()])))
    mapping = load_split_mapping(cfg, all_d.min(), all_d.max())
    unit = cfg["split"].get("block_unit", "year")
    frames = [build_ticker_frame(t, full[t], start, fc, mapping, unit, dtype) for t in in_range]
    df = pd.concat(frames, ignore_index=True)
    if groups is not None:
        g = groups.set_index("ticker")
        df.insert(2, "sector", df["ticker"].map(g["sector"]).fillna("Unknown"))
        df.insert(3, "sub_industry", df["ticker"].map(g["sub_industry"]).fillna("Unknown"))
    for c in ("ticker", "sector", "sub_industry", "split"):
        if c in df:
            df[c] = df[c].astype("category")
    df["year"] = df["date"].dt.year.astype("int16")
    t_build = time.time() - t0
    import shutil
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    df.to_parquet(out_dir, partition_cols=["year"], index=False)
    fd = feature_dictionary(fc)
    fd.to_csv(out_dir.parent / "features_dictionary.csv", index=False)
    md = ["| column | role | definition | uses_future_data |", "|---|---|---|---|"] + \
         [f"| `{r.column}` | {r.role} | {r.definition} | {r.uses_future_data} |" for r in fd.itertuples()]
    (out_dir.parent / "features_dictionary.md").write_text("# 特征字典\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    meta = {"rows": len(df), "n_tickers": int(df["ticker"].nunique()), "n_columns": df.shape[1],
            "first_date": str(df["date"].min().date()), "last_date": str(df["date"].max().date()),
            "columns": list(df.columns), "config": {k: v for k, v in fc.items()},
            "split_mapping": {str(1970 + k) if unit == "year" else str(k): v for k, v in sorted(mapping.items())},
            "build_seconds": round(t_build, 1), "total_seconds": round(time.time() - t0, 1)}
    (out_dir.parent / "features_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    log.info("特征数据集：%d 行 × %d 列，%d 只股票，用时 %.1fs", len(df), df.shape[1], meta["n_tickers"], meta["total_seconds"])
    return meta
