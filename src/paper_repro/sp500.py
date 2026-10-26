"""在本项目 S&P 500 日线（复权收盘价，2016–2026）上近似复现论文 §5–6 的实验。

论文原协议（§5, pp.102–103）：每只股票 ≈150 个交易日收盘价（截至 2008-02-01），最后 5 天为 holdout，
对每个 holdout 观测做一步预测，得分 = 平均对数似然（sequential predictive score），再对所有序列平均；方法间用单侧符号检验。

本项目协议（用户决定）：年块切分 seed 42、80/20（eval 年 = 2017、2024，其余为 search）。
- "episode" = 同一只股票、同一自然年内连续 150 个交易日：前 145 天训练、后 5 天 holdout（与论文同规模）。
  每个 (ticker, 年) 取两个 episode：该年前 150 个交易日、该年最后 150 个交易日（年内交易日 < 150 则跳过）。
  episode 完全落在单一年块内 → 目标期不会跨越切分边界（跨边界样本自然被丢弃）。
- 阶段 1（search 年）：每个方法在小网格上选超参（min_leaf；谱方法另含距离是否平方），按 search 平均得分选择。
- 阶段 2（eval 年 2017/2024）：每个方法只用选定的超参跑一次，报告结果。
"""
from __future__ import annotations

import json
import os
import sys
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .art import ARTLearner, METHODS, method_config, sign_test

from trend_scan.paths import PRICES_DIR as PRICES, ROOT  # 单向依赖：paper_repro -> trend_scan
EVAL_YEARS = {2017, 2024}
EP_LEN, HOLD = 150, 5


def build_episodes(start="2016-01-01", end="2026-12-31"):
    eps = []
    for f in sorted(PRICES.glob("*.parquet")):
        s = pd.read_parquet(f, columns=["Close"])["Close"].dropna()
        s = s[(s.index >= start) & (s.index <= end) & (s > 0)]
        for yr, g in s.groupby(s.index.year):
            if len(g) < EP_LEN:
                continue
            for tag, seg in (("first", g.iloc[:EP_LEN]), ("last", g.iloc[-EP_LEN:])):
                eps.append(dict(ticker=f.stem, year=int(yr), tag=tag, split="eval" if yr in EVAL_YEARS else "search",
                                hold_start=str(seg.index[-HOLD].date()), hold_end=str(seg.index[-1].date()),
                                x=seg.values.astype(float)))
    return eps


def grid_for(method: str):
    if method == "Baseline":
        return [dict()]
    ml = [10, 20, 30]
    if "Oblique" in method or "TimeSpectral" in method:
        return [dict(min_leaf=m, affinity_squared=a) for m, a in product(ml, [False, True])]
    return [dict(min_leaf=m) for m in ml]


def rw_score(x):
    """额外参照（论文中没有）：高斯随机游走，sd 取训练期一阶差分。"""
    tr = x[:-HOLD]
    d = np.diff(tr); mu, sd = d.mean(), d.std(ddof=1)
    out = []
    for t in range(len(x) - HOLD, len(x)):
        z = x[t] - x[t - 1] - mu
        out.append(-0.5 * np.log(2 * np.pi * sd * sd) - 0.5 * z * z / (sd * sd))
    return float(np.mean(out))


def _job(args):
    ep_i, x, tasks = args
    res = []
    hold = np.arange(len(x) - HOLD, len(x))
    for method, hp in tasks:
        L = ARTLearner(method_config(method, **hp)).fit(x[:-HOLD])
        s, e = L.sequential_score(x, hold)
        res.append(dict(ep=ep_i, method=method, hp=json.dumps(hp, sort_keys=True), score=s,
                        sq_err=float((e ** 2).mean()), p=L.p, n_leaves=L.n_leaves))
    res.append(dict(ep=ep_i, method="RandomWalk(ref)", hp="{}", score=rw_score(x), sq_err=float(((x[hold] - x[hold - 1]) ** 2).mean()),
                    p=np.nan, n_leaves=np.nan))
    return res


def run_phase(eps, idx, tasks_fn, procs):
    jobs = [(i, eps[i]["x"], tasks_fn(i)) for i in idx]
    out = []
    with Pool(procs) as pool:
        for k, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=4)):
            out.extend(r)
            if (k + 1) % 500 == 0:
                print(f"  {k + 1}/{len(jobs)}", flush=True)
    return pd.DataFrame(out)


def main(out_dir="outputs/repro_bocharov", ticker_frac=1.0, procs=8, seed=42):
    out = ROOT / out_dir; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    eps = build_episodes()
    tick = sorted({e["ticker"] for e in eps})
    if ticker_frac < 1:
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(tick, int(round(len(tick) * ticker_frac)), replace=False))
        eps = [e for e in eps if e["ticker"] in keep]
    meta = pd.DataFrame([{k: v for k, v in e.items() if k != "x"} | {"last_train_close": e["x"][-HOLD - 1]} for e in eps])
    meta["ep"] = np.arange(len(meta))
    meta.to_csv(out / "sp500_episodes.csv", index=False)
    s_idx = meta.index[meta.split == "search"].tolist(); e_idx = meta.index[meta.split == "eval"].tolist()
    print(f"episodes: search={len(s_idx)} eval={len(e_idx)} tickers={meta.ticker.nunique()}", flush=True)

    all_tasks = [(m, hp) for m in METHODS for hp in grid_for(m)]
    search = run_phase(eps, s_idx, lambda i: all_tasks, procs)
    search.to_csv(out / "sp500_search_detail.csv.gz", index=False)
    sel = (search[search.method != "RandomWalk(ref)"].groupby(["method", "hp"]).score.mean().reset_index()
           .sort_values("score", ascending=False).groupby("method").head(1).set_index("method"))
    grid_tab = search.groupby(["method", "hp"]).score.mean().unstack(0).round(4)
    grid_tab.to_csv(out / "sp500_search_grid.csv")
    chosen = {m: json.loads(sel.loc[m, "hp"]) for m in METHODS}
    (out / "sp500_selected_hparams.json").write_text(json.dumps(chosen, indent=2))
    print("selected:", chosen, f"[{time.time() - t0:.0f}s]", flush=True)

    ev = run_phase(eps, e_idx, lambda i: [(m, chosen[m]) for m in METHODS], procs)
    ev.to_csv(out / "sp500_eval_detail.csv.gz", index=False)
    srch_sel = pd.concat([search[(search.method == m) & (search.hp == json.dumps(chosen[m], sort_keys=True))] for m in METHODS]
                         + [search[search.method == "RandomWalk(ref)"]])
    report(meta, srch_sel, ev, out)
    print(f"done [{time.time() - t0:.0f}s]")


def summarize(meta, df):
    df = df.merge(meta[["ep", "last_train_close", "year"]], on="ep")
    df["score_scaleadj"] = df.score + np.log(df.last_train_close)    # 对 x/P_ref 的对数似然（Jacobian），与价格量纲无关
    df["rel_rmse_vs_price"] = np.sqrt(df.sq_err) / df.last_train_close
    order = list(METHODS) + ["RandomWalk(ref)"]
    t = df.groupby("method").agg(n=("score", "size"), score_mean=("score", "mean"), score_median=("score", "median"),
                                 scaleadj_mean=("score_scaleadj", "mean"), rel_rmse_mean=("rel_rmse_vs_price", "mean"),
                                 p_mean=("p", "mean"), leaves_mean=("n_leaves", "mean")).reindex(order)
    return t, df


def pairwise(df):
    W = df.pivot(index="ep", columns="method", values="score")
    rows = []
    for ts in ["NoSplit", "Fixed", "TimeSpectral"]:
        ax, ob = f"AxialGaussian+{ts}", f"ObliqueSpectral+{ts}"
        for a, b in [("Baseline", ax), ("Baseline", ob), (ax, ob)]:
            aw, bw, p = sign_test(W[a].values, W[b].values)
            rows.append(dict(time_split=ts, A=a, B=b, A_better=aw, B_better=bw, ties=int(len(W) - aw - bw), p_one_sided_B_better=p))
    for a, b in [("RandomWalk(ref)", "Baseline"), ("RandomWalk(ref)", "ObliqueSpectral+NoSplit"), ("RandomWalk(ref)", "AxialGaussian+NoSplit")]:
        aw, bw, p = sign_test(W[a].values, W[b].values)
        rows.append(dict(time_split="-", A=a, B=b, A_better=aw, B_better=bw, ties=int(len(W) - aw - bw), p_one_sided_B_better=p))
    return pd.DataFrame(rows)


def report(meta, search, ev, out):
    for name, df in [("search", search), ("eval", ev)]:
        t, d = summarize(meta, df)
        t.round(4).to_csv(out / f"sp500_{name}_table1.csv")
        pw = pairwise(d); pw.to_csv(out / f"sp500_{name}_table2_signtests.csv", index=False)
        if name == "eval":
            byyr = d.groupby(["year", "method"]).score_scaleadj.mean().unstack(0).reindex(t.index).round(4)
            byyr.to_csv(out / "sp500_eval_by_year_scaleadj.csv")
        with pd.option_context("display.width", 250, "display.max_columns", 20):
            print(f"\n=== {name} ===\n", t.round(4), "\n", pw)


if __name__ == "__main__":
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    main(ticker_frac=frac)
