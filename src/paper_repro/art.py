"""ART（自回归树）模型 + 论文的切分提议器。

论文对应：
- §2：叶子为 AR(p) 高斯模型；切分可作用于自回归项（axial：X_i <= c；oblique：sum a_i X_i <= c）或时间（t <= c）。
- §3：逐阶（stagewise）加入自回归项 x_{t-1}, x_{t-2}, ...，直到评分不再提高；每个阶数下学习一棵回归树；
      贝叶斯评分（Meek et al. 2002）避免过拟合，因此无剪枝。
- §4.2：ObliqueSpectral —— 对 phase view (x_t, x_{t-1..t-p}) 谱聚类 → 投影到自回归项空间用感知机求超平面 → 唯一一个斜切分候选。
- §4.3：TimeSpectral —— trace view：按时间顺序追踪聚类标签，每次跨簇处提出一个时间切分候选；
         Fixed —— 等间距时间点（商业实现的近似，点数未给出 → 假设 15）。
- §2.1：AxialGaussian —— 每个自回归项按高斯分位数提出 15 个切分点（Chickering et al. 2001）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from . import bayes
from .spectral import fiedler_split, pocket_perceptron

GAUSS_Q = np.array([NormalDist().inv_cdf(i / 16) for i in range(1, 16)])


@dataclass
class Config:
    reg_split: str = "oblique"      # none | axial | oblique
    time_split: str = "none"        # none | fixed | spectral
    max_p: int = 8                  # 最大自回归阶数（论文未给出 → 假设）
    min_leaf: int = 20              # 叶子最少样本（论文未给出 → 假设，在 search 上调）
    max_depth: int = 6
    n_fixed_time: int = 15
    affinity_squared: bool = False
    gap_rule: str = "gap"
    prior_n0: float = 1.0
    seed: int = 0


@dataclass
class Node:
    idx: np.ndarray
    post: bayes.Posterior | None = None
    pred: tuple | None = None       # ("axial", j, c) / ("oblique", a, c) / ("time", c)
    left: "Node | None" = None
    right: "Node | None" = None
    cluster_lab: np.ndarray | None = None

    def is_leaf(self):
        return self.pred is None


def eval_pred(pred, R: np.ndarray, tt: np.ndarray) -> np.ndarray:
    kind = pred[0]
    if kind == "axial":
        return R[:, pred[1]] <= pred[2]
    if kind == "oblique":
        return R @ pred[1] <= pred[2]
    return tt <= pred[1]


@dataclass
class Data:
    y: np.ndarray          # 目标 x_t
    R: np.ndarray          # 自回归项 (n, p)：x_{t-1}..x_{t-p}
    X: np.ndarray          # 设计矩阵（含截距）
    t: np.ndarray          # 时间索引
    P: np.ndarray          # phase view (n, p+1)：(x_t, x_{t-1}, ..., x_{t-p})


def make_data(x: np.ndarray, p: int, start: int) -> Data:
    T = len(x)
    tt = np.arange(start, T)
    R = np.column_stack([x[tt - j] for j in range(1, p + 1)]) if p else np.empty((len(tt), 0))
    y = x[tt]
    return Data(y=y, R=R, X=bayes.design(R), t=tt, P=np.column_stack([y, R]))


class ARTLearner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.n_candidates = 0

    # ---------------- 切分候选 ----------------
    def _cluster(self, d: Data, idx: np.ndarray):
        r = fiedler_split(d.P[idx], self.cfg.min_leaf, self.cfg.affinity_squared, self.cfg.gap_rule)
        return None if r is None else r[0]

    def candidates(self, d: Data, idx: np.ndarray):
        cfg, out = self.cfg, []
        p = d.R.shape[1]
        need_cluster = (cfg.reg_split == "oblique" and p > 0) or cfg.time_split == "spectral"
        lab = self._cluster(d, idx) if need_cluster else None
        if cfg.reg_split == "axial" and p > 0:
            for j in range(p):
                v = d.R[idx, j]
                mu, sd = v.mean(), v.std()
                for c in np.unique(mu + sd * GAUSS_Q):
                    out.append(("axial", j, float(c)))
        if cfg.reg_split == "oblique" and p > 0 and lab is not None:
            a, c, _ = pocket_perceptron(d.R[idx], lab, seed=cfg.seed)
            out.append(("oblique", a, float(c)))
        if cfg.time_split == "fixed":
            lo, hi = d.t[idx].min(), d.t[idx].max()
            for c in np.linspace(lo, hi, cfg.n_fixed_time + 2)[1:-1]:
                out.append(("time", float(c)))
        if cfg.time_split == "spectral" and lab is not None:
            order = np.argsort(d.t[idx])
            ls, ts = lab[order], d.t[idx][order]
            jumps = np.nonzero(np.diff(ls) != 0)[0]
            for k in jumps:
                out.append(("time", float(ts[k])))      # t <= ts[k] | t > ts[k]
        return out

    # ---------------- 建树 ----------------
    def grow(self, d: Data, idx: np.ndarray, prior, depth: int) -> Node:
        node = Node(idx=idx, post=bayes.posterior(d.X[idx], d.y[idx], prior))
        if depth >= self.cfg.max_depth or len(idx) < 2 * self.cfg.min_leaf:
            return node
        best = (0.0, None, None, None)
        for pred in self.candidates(d, idx):
            self.n_candidates += 1
            m = eval_pred(pred, d.R[idx], d.t[idx])
            nl = int(m.sum())
            if nl < self.cfg.min_leaf or len(idx) - nl < self.cfg.min_leaf:
                continue
            il, ir = idx[m], idx[~m]
            pl = bayes.posterior(d.X[il], d.y[il], prior)
            pr_ = bayes.posterior(d.X[ir], d.y[ir], prior)
            gain = pl.logml + pr_.logml - node.post.logml
            if gain > best[0]:
                best = (gain, pred, il, ir)
        if best[1] is not None:
            node.pred = best[1]
            node.left = self.grow(d, best[2], prior, depth + 1)
            node.right = self.grow(d, best[3], prior, depth + 1)
        return node

    @staticmethod
    def leaves(node: Node):
        if node.is_leaf():
            return [node]
        return ARTLearner.leaves(node.left) + ARTLearner.leaves(node.right)

    def fit_p(self, x: np.ndarray, p: int):
        d = make_data(x, p, self.cfg.max_p)        # 所有阶数使用同一组目标（t >= max_p），评分可比（假设）
        prior = bayes.make_prior(d.X, d.y, n0=self.cfg.prior_n0)
        root = self.grow(d, np.arange(len(d.y)), prior, 0)
        score = sum(l.post.logml for l in self.leaves(root))
        return root, score, d

    def fit(self, x: np.ndarray):
        """§3 逐阶加入自回归项：p = 0, 1, 2, ...，评分不提高即停止。"""
        best = None
        for p in range(0, self.cfg.max_p + 1):
            root, score, d = self.fit_p(x, p)
            if best is not None and score <= best[1]:
                break
            best = (root, score, d, p)
        self.root, self.score, self.data, self.p = best
        self.n_leaves = len(self.leaves(self.root))
        return self

    # ---------------- 预测 ----------------
    def route(self, r: np.ndarray, t: int) -> Node:
        node = self.root
        while not node.is_leaf():
            go_left = bool(eval_pred(node.pred, r[None, :], np.array([t]))[0])
            node = node.left if go_left else node.right
        return node

    def sequential_score(self, x_full: np.ndarray, t_hold: np.ndarray):
        """§5：对每个 holdout 观测做一步预测（自回归项用实际已观测值），返回平均对数似然与误差。"""
        lls, errs = [], []
        for t in t_hold:
            r = np.array([x_full[t - j] for j in range(1, self.p + 1)])
            leaf = self.route(r, int(t))
            xd = np.concatenate([[1.0], r])
            lls.append(bayes.predictive_logpdf(xd, float(x_full[t]), leaf.post))
            errs.append(float(x_full[t]) - bayes.predictive_mean(xd, leaf.post))
        return float(np.mean(lls)), np.array(errs)

    def train_leaf_assignment(self):
        """训练样本所在叶子编号（用于合成数据的分段评估）。"""
        lab = np.empty(len(self.data.y), int)
        for k, lf in enumerate(self.leaves(self.root)):
            lab[lf.idx] = k
        return lab


METHODS = {
    "Baseline":                      Config(reg_split="none", time_split="none"),
    "AxialGaussian+NoSplit":         Config(reg_split="axial", time_split="none"),
    "AxialGaussian+Fixed":           Config(reg_split="axial", time_split="fixed"),
    "AxialGaussian+TimeSpectral":    Config(reg_split="axial", time_split="spectral"),
    "ObliqueSpectral+NoSplit":       Config(reg_split="oblique", time_split="none"),
    "ObliqueSpectral+Fixed":         Config(reg_split="oblique", time_split="fixed"),
    "ObliqueSpectral+TimeSpectral":  Config(reg_split="oblique", time_split="spectral"),
}


def method_config(name: str, **over) -> Config:
    import dataclasses
    return dataclasses.replace(METHODS[name], **over)


def sign_test(a: np.ndarray, b: np.ndarray):
    """单侧符号检验：b 优于 a 的次数 vs a 优于 b 的次数（不计平局），返回 (a_wins, b_wins, p 值[H1: b 更好])。"""
    import math
    aw, bw = int((a > b).sum()), int((b > a).sum())
    n = aw + bw
    if n == 0:
        return aw, bw, 1.0
    # P(X >= bw), X ~ Bin(n, 0.5)
    logs = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2) for k in range(bw, n + 1)]
    m = max(logs)
    return aw, bw, float(min(1.0, math.exp(m) * sum(math.exp(v - m) for v in logs)))
