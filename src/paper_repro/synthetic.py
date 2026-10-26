"""合成状态切换序列验证：已知真实 regime，检查 (1) 根节点谱聚类、(2) 学到的树叶子 是否恢复 regime。

S1 "分段"（类似论文 Fig.2 的 4 段交替）：A,B,A,B 各 100 点；
    A: 均值回复到 20，phi=0.8，噪声 sd=1；B: 均值回复到 30，phi=0.3，噪声 sd=2。
S2 "SETAR（斜边界）"：regime 由 x_{t-1}+x_{t-2} > 0 决定（边界为对角线方向，与论文附录的对角主轴论证一致）；
    高: x_t = 0.9 x_{t-1} + 0.5 + e；低: x_t = 0.9 x_{t-1} - 0.5 + e，e~N(0,1.5^2)，且两 regime 噪声不同（高 1.0 / 低 2.0）。
S3 "纯随机游走"（无 regime，检查不会乱切）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .art import ARTLearner, METHODS, method_config
from .spectral import fiedler_split


def gen_s1(rng):
    segs, regs = [], []
    x = 20.0
    for k in range(4):
        A = k % 2 == 0
        mu, phi, sd = (20, 0.8, 1.0) if A else (30, 0.3, 2.0)
        for _ in range(100):
            x = mu + phi * (x - mu) + rng.normal(0, sd)
            segs.append(x); regs.append(0 if A else 1)
    # 再接 50 个 B 作为 holdout（延续最后一段）
    for _ in range(50):
        x = 30 + 0.3 * (x - 30) + rng.normal(0, 2.0)
        segs.append(x); regs.append(1)
    return np.array(segs), np.array(regs)


def gen_s2(rng, T=450):
    x = [0.0, 0.0]; regs = [0, 0]
    for _ in range(T - 2):
        hi = x[-1] + x[-2] > 0
        x.append(0.9 * x[-1] + (0.5 if hi else -0.5) + rng.normal(0, 1.0 if hi else 2.0))
        regs.append(int(hi))
    return np.array(x), np.array(regs)  # regs[t] = 生成 x_t 时的 regime


def gen_s3(rng, T=450):
    return np.cumsum(rng.normal(0, 1, T)) + 50, np.zeros(T, int)


def ari(a, b):
    from math import comb
    a = np.asarray(a); b = np.asarray(b)
    ct = pd.crosstab(a, b).values
    s = sum(comb(int(v), 2) for v in ct.ravel())
    sa = sum(comb(int(v), 2) for v in ct.sum(1)); sb = sum(comb(int(v), 2) for v in ct.sum(0))
    tot = comb(len(a), 2); exp = sa * sb / tot; mx = (sa + sb) / 2
    return 1.0 if mx == exp else (s - exp) / (mx - exp)


def purity(leaf, truth):
    ct = pd.crosstab(leaf, truth).values
    return ct.max(1).sum() / ct.sum()


def run(n_seeds=20, n_train=400):
    rows = []
    gens = {"S1_segments": gen_s1, "S2_setar_oblique": gen_s2, "S3_random_walk": gen_s3}
    for gname, g in gens.items():
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 + seed)
            x, reg = g(rng)
            hold = np.arange(n_train, len(x))
            for m in METHODS:
                L = ARTLearner(method_config(m, min_leaf=20)).fit(x[:n_train])
                s, e = L.sequential_score(x, hold)
                tr = L.data.t
                leaf = L.train_leaf_assignment()
                row = dict(scenario=gname, seed=seed, method=m, p=L.p, n_leaves=L.n_leaves, holdout_ll=s,
                           holdout_rmse=float(np.sqrt((e ** 2).mean())),
                           leaf_ari=ari(leaf, reg[tr]) if gname != "S3_random_walk" else np.nan,
                           leaf_purity=purity(leaf, reg[tr]) if gname != "S3_random_walk" else np.nan)
                rows.append(row)
            # 根节点谱聚类本身（p=2 phase view）的 regime 恢复
            if gname != "S3_random_walk":
                from .art import make_data
                d = make_data(x[:n_train], 2, 8)
                r = fiedler_split(d.P, 20)
                rows.append(dict(scenario=gname, seed=seed, method="[root spectral clustering, p=2]",
                                 leaf_ari=ari(r[0], reg[d.t]) if r else np.nan,
                                 leaf_purity=purity(r[0], reg[d.t]) if r else np.nan))
    return pd.DataFrame(rows)


def main(out=None):
    from trend_scan.paths import PAPER_OUTPUTS_DIR
    out = Path(out) if out else PAPER_OUTPUTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    df = run()
    df.to_csv(f"{out}/synthetic_detail.csv", index=False)
    summ = df.groupby(["scenario", "method"], sort=False).agg(
        p=("p", "mean"), n_leaves=("n_leaves", "mean"), holdout_ll=("holdout_ll", "mean"),
        holdout_rmse=("holdout_rmse", "mean"), leaf_ari=("leaf_ari", "mean"), leaf_purity=("leaf_purity", "mean")).round(3)
    summ.to_csv(f"{out}/synthetic_summary.csv")
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(summ)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
