"""参数扫描与实验记录。

每个 (窗口长度, 阈值对) 组合为一个实验，实验 id = <run时间戳>_w<N>_u<up>_d<down>_<cfg哈希>。
输出：
  data/processed/w{N}_u{up}_d{down}/{up,down}.parquet  (+ CSV 抽样)
  experiments/<id>/config.yaml, metadata.json, summary.json, summary.csv
  experiments/index.csv（每个实验×split×label 一行，追加写入）
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import fit as fitmod
from . import scan as scanmod
from .paths import EXPERIMENTS_DIR, PROCESSED_DIR

log = logging.getLogger(__name__)
SPLITS = ("search", "eval", "excluded")
# 分组键 -> 结果表中的列名（来自维基百科成分股表的 GICS 分类，均为“当前”分类）
GROUP_COLS = {"sector": "sector", "sub_industry": "sub_industry"}
WIKI_GROUP_SRC = {"sector": "GICS Sector", "sub_industry": "GICS Sub-Industry"}


def load_groups(universe_df: pd.DataFrame) -> pd.DataFrame:
    """从成分股表提取 ticker -> sector / sub_industry 映射（缺失则为 Unknown）。"""
    out = pd.DataFrame({"ticker": universe_df["ticker"]})
    for key, src in WIKI_GROUP_SRC.items():
        out[GROUP_COLS[key]] = universe_df[src].fillna("Unknown") if src in universe_df else "Unknown"
    return out


def summarize_groups(df: pd.DataFrame, totals: pd.DataFrame, groups: pd.DataFrame, group_col: str,
                     r2_main: float = 0.8) -> pd.DataFrame:
    """按 split × 分组 统计：up/down 数量、比值、斜率/R^2 中位数、R^2>=阈值占比、组内股票数。"""
    tot = totals.merge(groups[["ticker", group_col]], on="ticker", how="left")
    tot[group_col] = tot[group_col].fillna("Unknown")
    rows = []
    for split in ("all", "search", "eval"):
        d = df if split == "all" else df[df["split"] == split]
        base = (tot[list(SPLITS)].sum(axis=1) if split == "all" else tot[split])
        tot_g = tot.assign(_base=base).groupby(group_col)
        n_tk = tot_g["ticker"].nunique()
        n_win = tot_g["_base"].sum()
        for g in sorted(n_tk.index):
            x = d[d[group_col] == g]
            r = {"split": split, group_col: g, "n_tickers_in_group": int(n_tk[g]),
                 "n_windows_total": int(n_win[g])}
            for lab in ("up", "down"):
                y = x[x["label"] == lab]
                r[f"{lab}_count"] = int(len(y))
                r[f"{lab}_share_of_windows"] = len(y) / n_win[g] if n_win[g] else np.nan
                r[f"{lab}_n_tickers"] = int(y["ticker"].nunique())
                r[f"{lab}_slope_median"] = float(y["slope"].median()) if len(y) else np.nan
                r[f"{lab}_r2_median"] = float(y["r2"].median()) if len(y) else np.nan
                r[f"{lab}_share_r2_ge_{r2_main}"] = float((y["r2"] >= r2_main).mean()) if len(y) else np.nan
            r["up_down_ratio"] = r["up_count"] / r["down_count"] if r["down_count"] else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------- 时间切分：按日历块（默认年）随机分配 search / eval ----------------
def block_ids(dates: np.ndarray, unit: str = "year") -> np.ndarray:
    """日期 -> 整数块编号（year: 1970 年起的年序号；quarter；month）。"""
    d = np.asarray(dates, dtype="datetime64[ns]")
    if unit == "year":
        return d.astype("datetime64[Y]").astype(np.int64)
    m = d.astype("datetime64[M]").astype(np.int64)
    if unit == "month":
        return m
    if unit == "quarter":
        return m // 3
    raise ValueError(f"未知 block_unit: {unit}")


def block_label(bid: int, unit: str) -> str:
    if unit == "year":
        return str(1970 + bid)
    if unit == "quarter":
        return f"{1970 + bid // 4}Q{bid % 4 + 1}"
    return str(np.datetime64(int(bid), "M"))


def make_block_split(first_date, last_date, split_cfg: dict) -> dict[int, str]:
    """用固定种子把数据区间内的所有块随机分配到 search/eval，返回 {block_id: split}。"""
    unit = split_cfg.get("block_unit", "year")
    seed = int(split_cfg.get("seed", 42))
    frac = float(split_cfg.get("search_fraction", 0.8))
    b0, b1 = block_ids(np.array([np.datetime64(pd.Timestamp(first_date)),
                                 np.datetime64(pd.Timestamp(last_date))]), unit)
    blocks = np.arange(b0, b1 + 1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(blocks)
    n_search = int(round(frac * len(blocks)))
    search = set(perm[:n_search].tolist())
    return {int(b): ("search" if b in search else "eval") for b in blocks}


def mapping_table(mapping: dict[int, str], unit: str) -> pd.DataFrame:
    return pd.DataFrame({"block_id": list(mapping.keys()),
                         "block": [block_label(b, unit) for b in mapping],
                         "split": list(mapping.values())}).sort_values("block_id")


def assign_split(start: np.ndarray, end: np.ndarray, split_cfg: dict,
                 mapping: dict[int, str]) -> np.ndarray:
    """窗口首日和末日所在块必须属于同一 split，否则标记为 excluded（避免跨 split 的重叠泄漏）。"""
    unit = split_cfg.get("block_unit", "year")
    bs, be = block_ids(start, unit), block_ids(end, unit)
    lo = min(mapping)
    lut = np.array([mapping.get(b, "excluded") for b in range(lo, max(mapping) + 1)], dtype=object)
    ss = lut[np.clip(bs - lo, 0, len(lut) - 1)]
    se = lut[np.clip(be - lo, 0, len(lut) - 1)]
    return np.where(ss == se, ss, "excluded").astype(object)


def scan_universe(closes: dict[str, pd.Series], n: int, up: float, down: float,
                  split_cfg: dict, mapping: dict[int, str], ann_days: int = 252,
                  groups: pd.DataFrame | None = None):
    """对全部股票执行 检测 + 拟合，返回 (labeled_df, totals_df)。"""
    frames, totals = [], []
    for t, s in closes.items():
        p = s.to_numpy(dtype=float)
        if len(p) < n:
            continue
        dates = s.index.to_numpy(dtype="datetime64[ns]")
        # --- detection（事后） ---
        starts, cr, labels, n_all = scanmod.detect_windows(p, n, up, down)
        # 各 split 的全部窗口数（含未标记），作为基数
        all_split = assign_split(dates[: n_all], dates[n - 1:], split_cfg, mapping)
        vc = pd.Series(all_split).value_counts()
        totals.append({"ticker": t, **{k: int(vc.get(k, 0)) for k in SPLITS}})
        if len(starts) == 0:
            continue
        # --- fitting ---
        ols = fitmod.rolling_ols_log(p, n, ann_days)
        sel = {k: v[starts] for k, v in ols.items()}
        mar = fitmod.max_abs_residual(p, n, starts, sel["slope"], sel["intercept"])
        sd, ed = dates[starts], dates[starts + n - 1]
        frames.append(pd.DataFrame({
            "ticker": t, "start_date": sd, "end_date": ed, "window": n,
            "cum_return": cr, "label": labels, **sel, "max_abs_resid": mar,
            "split": assign_split(sd, ed, split_cfg, mapping),
        }))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        if groups is not None and len(groups):
            g = groups.set_index("ticker")
            for col in GROUP_COLS.values():
                if col in g.columns:
                    df[col] = df["ticker"].map(g[col]).fillna("Unknown").astype("category")
        df["ticker"] = df["ticker"].astype("category")
        df["label"] = df["label"].astype("category")
        df["split"] = df["split"].astype("category")
    return df, pd.DataFrame(totals)


def summarize(df: pd.DataFrame, totals: pd.DataFrame, r2_thresholds: list[float]) -> list[dict]:
    rows = []
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    for split in ("all",) + SPLITS:
        d = df if split == "all" else df[df["split"] == split]
        base = int(totals[list(SPLITS)].to_numpy().sum()) if split == "all" else int(totals[split].sum())
        for lab in ("up", "down"):
            x = d[d["label"] == lab]
            r = {"split": split, "label": lab, "n_windows_total": base, "count": int(len(x)),
                 "share_of_windows": (len(x) / base) if base else np.nan,
                 "n_tickers": int(x["ticker"].nunique()) if len(x) else 0}
            if len(x):
                r.update({
                    "first_end_date": str(x["end_date"].min().date()),
                    "last_end_date": str(x["end_date"].max().date()),
                    "cum_return_mean": float(x["cum_return"].mean()),
                    "cum_return_median": float(x["cum_return"].median()),
                    "slope_mean": float(x["slope"].mean()),
                    "slope_median": float(x["slope"].median()),
                    "slope_annual_median": float(x["slope_annual"].median()),
                    "slope_tstat_median": float(x["slope_tstat"].median()),
                    "resid_std_median": float(x["resid_std"].median()),
                    "max_abs_resid_median": float(x["max_abs_resid"].median()),
                    "r2_mean": float(x["r2"].mean()),
                })
                for q, v in zip(qs, x["r2"].quantile(qs).to_numpy()):
                    r[f"r2_q{int(q*100):02d}"] = float(v)
                for th in r2_thresholds:
                    r[f"share_r2_ge_{th}"] = float((x["r2"] >= th).mean())
            rows.append(r)
    return rows


def _pkg_versions() -> dict:
    import matplotlib, pyarrow, yfinance  # noqa
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "yfinance": yfinance.__version__,
            "pyarrow": pyarrow.__version__, "matplotlib": matplotlib.__version__, "pyyaml": yaml.__version__}


def combo_key(n: int, up: float, down: float) -> str:
    return f"w{n}_u{up:g}_d{down:g}"


def run_grid(cfg: dict, closes: dict[str, pd.Series], universe_meta: dict,
             config_path: str | None = None, groups: pd.DataFrame | None = None) -> list[str]:
    run_ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
    split_cfg = cfg["split"]
    r2_th = cfg["fit"]["r2_thresholds"]
    ann = cfg["fit"].get("annualization_days", 252)
    n_sample = cfg.get("output", {}).get("save_csv_sample_rows", 2000)
    all_dates = pd.DatetimeIndex(sorted(set().union(*[s.index[[0, -1]] for s in closes.values()])))
    data_range = {"first_date": str(all_dates.min().date()), "last_date": str(all_dates.max().date()),
                  "n_tickers_used": len(closes)}
    mapping = make_block_split(all_dates.min(), all_dates.max(), split_cfg)
    unit = split_cfg.get("block_unit", "year")
    mtab = mapping_table(mapping, unit)
    split_info = {"method": "random_block", "block_unit": unit, "seed": split_cfg.get("seed", 42),
                  "search_fraction": split_cfg.get("search_fraction", 0.8),
                  "n_blocks": len(mapping), "n_search_blocks": int((mtab.split == "search").sum()),
                  "search_blocks": mtab[mtab.split == "search"].block.tolist(),
                  "eval_blocks": mtab[mtab.split == "eval"].block.tolist()}
    log.info("时间切分：%d 个块，search %d / eval %d", len(mapping), split_info["n_search_blocks"],
             len(mapping) - split_info["n_search_blocks"])
    gkey = (cfg.get("grouping") or {}).get("by", "sector")
    if gkey not in GROUP_COLS:
        raise ValueError(f"grouping.by 必须是 {list(GROUP_COLS)} 之一")
    group_col = GROUP_COLS[gkey]
    r2_main = (cfg.get("grouping") or {}).get("r2_threshold", 0.8)
    index_rows, group_rows, ids = [], [], []
    for n in cfg["scan"]["window_lengths"]:
        for th in cfg["scan"]["thresholds"]:
            up, down = float(th["up"]), float(th["down"])
            key = combo_key(n, up, down)
            exp_id = f"{run_ts}_{key}_{cfg_hash}"
            t0 = time.time()
            df, totals = scan_universe(closes, n, up, down, split_cfg, mapping, ann, groups)
            t_scan = time.time() - t0
            pdir = PROCESSED_DIR / key
            pdir.mkdir(parents=True, exist_ok=True)
            for lab in ("up", "down"):
                x = df[df["label"] == lab].reset_index(drop=True)
                x.to_parquet(pdir / f"{lab}.parquet", index=False)
                x.sample(min(n_sample, len(x)), random_state=0).sort_values(["ticker", "start_date"]) \
                    .to_csv(pdir / f"{lab}_sample.csv", index=False)
            totals.to_csv(pdir / "window_totals_by_ticker.csv", index=False)
            summ = summarize(df, totals, r2_th)
            gsum = None
            if groups is not None and group_col in df.columns:
                totals.merge(groups, on="ticker", how="left").to_csv(pdir / "window_totals_by_ticker.csv", index=False)
                gsum = summarize_groups(df, totals, groups, group_col, r2_main)
            edir = EXPERIMENTS_DIR / exp_id
            edir.mkdir(parents=True, exist_ok=True)
            exp_cfg = {**cfg, "this_experiment": {"window": n, "up": up, "down": down}}
            with open(edir / "config.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(exp_cfg, f, allow_unicode=True, sort_keys=False)
            meta = {"experiment_id": exp_id, "group_by": gkey if groups is not None else None, "run_timestamp": run_ts, "config_hash": cfg_hash,
                    "config_path": config_path, "window": n, "up_threshold": up, "down_threshold": down,
                    "data_range": data_range, "universe": universe_meta,
                    "processed_dir": str(pdir), "scan_fit_seconds": round(t_scan, 2),
                    "package_versions": _pkg_versions(),
                    "note": "ex-post descriptive only; windows use future data within window"}
            mtab.to_csv(edir / "split_mapping.csv", index=False)
            if gsum is not None:
                gsum.to_csv(edir / f"summary_by_{gkey}.csv", index=False)
                group_rows.append(gsum.assign(experiment_id=exp_id, run_timestamp=run_ts, window=n,
                                              up=up, down=down, group_by=gkey)
                                  .rename(columns={group_col: "group"}))
            meta["split"] = split_info
            (edir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            (edir / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
            pd.DataFrame(summ).to_csv(edir / "summary.csv", index=False)
            for r in summ:
                index_rows.append({"experiment_id": exp_id, "run_timestamp": run_ts, "window": n,
                                   "up": up, "down": down, **r})
            ids.append(exp_id)
            log.info("[%s] 完成：up=%d down=%d 用时 %.1fs", exp_id,
                     (df["label"] == "up").sum(), (df["label"] == "down").sum(), t_scan)
    if group_rows:
        gidx = EXPERIMENTS_DIR / "index_by_group.csv"
        newg = pd.concat(group_rows, ignore_index=True)
        if gidx.exists():
            newg = pd.concat([pd.read_csv(gidx), newg], ignore_index=True)
        newg.to_csv(gidx, index=False)
    idx = EXPERIMENTS_DIR / "index.csv"
    new = pd.DataFrame(index_rows)
    if idx.exists():
        new = pd.concat([pd.read_csv(idx), new], ignore_index=True)
    new.to_csv(idx, index=False)
    return ids
