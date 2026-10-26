"""收益率版本复现：序列 = 日对数收益 r_t = log(P_t/P_{t-1})；目标 r_t，回归项 r_{t-1..t-p}；
谱聚类在收益率 phase 元组 (r_t, r_{t-1..t-p}) 上进行。其他设置与价格版相同：
- 500 只股票；episode = 同一自然年内连续 151 个交易日价格 → 150 个年内收益（145 训练 + 5 验证），每 (股票, 年) 取年首/年末两个；
- 年块切分 seed 42（eval 年 2017、2024）；search 上按平均 LL 选超参，eval 一次性评估；
- 方法：Baseline AR(p) 不切 + {AxialGaussian, ObliqueSpectral} × {NoSplit, Fixed, TimeSpectral}；
- 参照：ZeroMean 高斯（= 价格随机游走，σ² = mean(r²)）、ConstMean 高斯（样本均值/标准差）。
变体 --standardize：每个 episode 用训练期收益标准差 s 缩放（r/s）后建模，LL 加 −log s 换算回原收益尺度，
  点预测乘回 s —— 对线性高斯叶子和经验先验几乎等变，主要改变的是谱聚类亲和度 1/(1+d) 的尺度（原始收益 d≈0.02，亲和度几乎全≈1）。
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from trend_scan.paths import ROOT  # 单向依赖：paper_repro -> trend_scan

from . import bayes
from .art import ARTLearner, METHODS, method_config, sign_test
from .sp500 import PRICES, EVAL_YEARS, grid_for

EP_PRICES, HOLD = 151, 5
REFS = ["ZeroMean(ref)", "ConstMean(ref)"]


def build_episodes():
    eps = []
    for f in sorted(PRICES.glob("*.parquet")):
        s = pd.read_parquet(f, columns=["Close"])["Close"].dropna()
        s = s[(s.index >= "2016-01-01") & (s > 0)]
        for yr, g in s.groupby(s.index.year):
            if len(g) < EP_PRICES:
                continue
            for tag, seg in (("first", g.iloc[:EP_PRICES]), ("last", g.iloc[-EP_PRICES:])):
                r = np.diff(np.log(seg.values.astype(float)))
                eps.append(dict(ticker=f.stem, year=int(yr), tag=tag, split="eval" if yr in EVAL_YEARS else "search",
                                hold_start=str(seg.index[-HOLD].date()), hold_end=str(seg.index[-1].date()), x=r))
    return eps


def vol_diag(L: ARTLearner, rng):
    """叶子残差方差（后验 E[σ²]=bn/(an-1)）的 max/min 比；以及同样大小的随机划分下的同一比值（零假设参照）。"""
    lv = ARTLearner.leaves(L.root)
    if len(lv) < 2:
        return np.nan, np.nan
    d = L.data
    prior = bayes.make_prior(d.X, d.y, n0=L.cfg.prior_n0)
    s2 = [lf.post.bn / (lf.post.an - 1) for lf in lv]
    perm = rng.permutation(len(d.y)); s2r, k = [], 0
    for lf in lv:
        ii = perm[k:k + len(lf.idx)]; k += len(lf.idx)
        po = bayes.posterior(d.X[ii], d.y[ii], prior); s2r.append(po.bn / (po.an - 1))
    return float(max(s2) / min(s2)), float(max(s2r) / min(s2r))


def _job(args):
    ep_i, r, tasks, standardize = args
    rng = np.random.default_rng(ep_i)
    tr = r[:-HOLD]
    s = float(tr.std(ddof=1)) if standardize else 1.0
    x = r / s
    hold = np.arange(len(r) - HOLD, len(r))
    y = r[hold]
    out = []
    for method, hp in tasks:
        L = ARTLearner(method_config(method, **hp)).fit(x[:-HOLD])
        ll, e = L.sequential_score(x, hold)
        pred = y - e * s
        vr, vr_rand = vol_diag(L, rng)
        out.append(dict(ep=ep_i, method=method, hp=json.dumps(hp, sort_keys=True), score=ll - np.log(s),
                        mse=float(((y - pred) ** 2).mean()), mae=float(np.abs(y - pred).mean()),
                        hits=int((np.sign(pred) == np.sign(y)).sum()), n_dir=int(((pred != 0) & (y != 0)).sum()),
                        p=L.p, n_leaves=L.n_leaves, var_ratio=vr, var_ratio_random=vr_rand))
    # 参照基线（在原始收益上，不受 standardize 影响）
    sd0 = float(np.sqrt((tr ** 2).mean()))
    mu, sd = float(tr.mean()), float(tr.std(ddof=1))
    for name, m_, s_ in (("ZeroMean(ref)", 0.0, sd0), ("ConstMean(ref)", mu, sd)):
        ll = float(np.mean(-0.5 * np.log(2 * np.pi * s_ ** 2) - 0.5 * (y - m_) ** 2 / s_ ** 2))
        pred = np.full(HOLD, m_)
        out.append(dict(ep=ep_i, method=name, hp="{}", score=ll, mse=float(((y - pred) ** 2).mean()),
                        mae=float(np.abs(y - pred).mean()),
                        hits=int((np.sign(pred) == np.sign(y)).sum()) if m_ != 0 else 0,
                        n_dir=int(((pred != 0) & (y != 0)).sum()), p=np.nan, n_leaves=np.nan,
                        var_ratio=np.nan, var_ratio_random=np.nan))
    return out


def run_phase(eps, idx, tasks_fn, procs, standardize):
    jobs = [(i, eps[i]["x"], tasks_fn(i), standardize) for i in idx]
    out = []
    with Pool(procs) as pool:
        for k, res in enumerate(pool.imap_unordered(_job, jobs, chunksize=4)):
            out.extend(res)
            if (k + 1) % 1000 == 0:
                print(f"  {k + 1}/{len(jobs)}", flush=True)
    return pd.DataFrame(out)


ORDER = list(METHODS) + REFS


def table1(df):
    g = df.groupby("method")
    t = pd.DataFrame({
        "n": g.size(), "logscore_mean": g.score.mean(), "logscore_median": g.score.median(),
        "mse_x1e4": g.mse.mean() * 1e4, "mae_x1e2": g.mae.mean() * 1e2,
        "hit_rate": g.hits.sum() / g.n_dir.sum().replace(0, np.nan),
        "p_mean": g.p.mean(), "leaves_mean": g.n_leaves.mean(), "share_split": g.n_leaves.apply(lambda v: (v > 1).mean()),
    }).reindex(ORDER)
    return t


def pairwise(df):
    W = df.pivot(index="ep", columns="method", values="score")
    rows = []
    def add(a, b, tag):
        aw, bw, p = sign_test(W[a].values, W[b].values)
        rows.append(dict(group=tag, A=a, B=b, A_better=aw, B_better=bw, ties=int(len(W) - aw - bw), p_one_sided_B_better=p,
                         p_one_sided_A_better=sign_test(W[b].values, W[a].values)[2]))
    for ts in ["NoSplit", "Fixed", "TimeSpectral"]:
        add(f"AxialGaussian+{ts}", f"ObliqueSpectral+{ts}", "oblique_vs_axial")
    for m in list(METHODS)[1:]:
        add("Baseline", m, "vs_baseline")
    for ref in REFS:
        for m in ["Baseline", "AxialGaussian+NoSplit", "ObliqueSpectral+NoSplit"]:
            add(ref, m, "vs_reference")
    return pd.DataFrame(rows)


def vol_table(df):
    d = df[df.var_ratio.notna()]
    g = d.groupby("method")
    return pd.DataFrame({"n_split_models": g.size(), "var_ratio_median": g.var_ratio.median(),
                         "var_ratio_random_median": g.var_ratio_random.median(),
                         "share_ratio_gt_random": g.apply(lambda v: (v.var_ratio > v.var_ratio_random).mean()),
                         "share_ratio_gt_2": g.apply(lambda v: (v.var_ratio > 2).mean())}).reindex(list(METHODS)[1:])


def main(standardize=False, procs=8):
    sub = "returns_std" if standardize else "returns"
    out = ROOT / "outputs" / "repro_bocharov" / sub; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    eps = build_episodes()
    meta = pd.DataFrame([{k: v for k, v in e.items() if k != "x"} for e in eps]); meta["ep"] = np.arange(len(meta))
    meta.to_csv(out / "episodes.csv", index=False)
    s_idx = meta.index[meta.split == "search"].tolist(); e_idx = meta.index[meta.split == "eval"].tolist()
    print(f"[{sub}] episodes search={len(s_idx)} eval={len(e_idx)} tickers={meta.ticker.nunique()}", flush=True)
    all_tasks = [(m, hp) for m in METHODS for hp in grid_for(m)]
    search = run_phase(eps, s_idx, lambda i: all_tasks, procs, standardize)
    search.to_csv(out / "search_detail.csv.gz", index=False)
    grid = search[~search.method.isin(REFS)].groupby(["method", "hp"]).score.mean()
    grid.unstack(0).round(5).to_csv(out / "search_grid.csv")
    chosen = {m: json.loads(grid.loc[m].idxmax()) for m in METHODS}
    (out / "selected_hparams.json").write_text(json.dumps(chosen, indent=2))
    print("selected:", chosen, f"[{time.time() - t0:.0f}s]", flush=True)
    ev = run_phase(eps, e_idx, lambda i: [(m, chosen[m]) for m in METHODS], procs, standardize)
    ev.to_csv(out / "eval_detail.csv.gz", index=False)
    srch = pd.concat([search[(search.method == m) & (search.hp == json.dumps(chosen[m], sort_keys=True))] for m in METHODS]
                     + [search[search.method.isin(REFS)]])
    for name, df in (("search", srch), ("eval", ev)):
        t = table1(df); t.round(5).to_csv(out / f"{name}_table1.csv")
        pw = pairwise(df); pw.to_csv(out / f"{name}_signtests.csv", index=False)
        vt = vol_table(df); vt.round(4).to_csv(out / f"{name}_vol_regimes.csv")
        if name == "eval":
            dd = df.merge(meta[["ep", "year"]], on="ep")
            dd.groupby(["year", "method"]).score.mean().unstack(0).reindex(ORDER).round(5).to_csv(out / "eval_by_year_logscore.csv")
        with pd.option_context("display.width", 250, "display.max_columns", 20):
            print(f"\n=== {sub} {name} ===\n", t.round(5), "\n", pw, "\n", vt.round(3), flush=True)
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main(standardize="--standardize" in sys.argv)
