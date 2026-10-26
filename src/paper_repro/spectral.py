"""论文 §4.1 Fig.1：简化归一化谱聚类；§4.2：口袋感知机求分离超平面。"""
from __future__ import annotations

import numpy as np


def affinity(P: np.ndarray, squared: bool = False) -> np.ndarray:
    """a_ij ∝ 1 / (1 + ||p_i - p_j||_2)（论文原文；squared=True 为平方距离变体，作为假设备选）。"""
    sq = (P * P).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * P @ P.T, 0.0)
    d = d2 if squared else np.sqrt(d2)
    return 1.0 / (1.0 + d)


def fiedler_split(P: np.ndarray, min_side: int, squared: bool = False, rule: str = "gap"):
    """返回 (labels∈{0,1}, eigvec) 或 None。

    L'norm = D^{-1/2} A D^{-1/2}（= I - Lnorm），取第二大特征向量；
    rule="gap"：按特征向量排序后最大间隙切分（论文 "e.g. by the largest gap"），并要求两侧 ≥ min_side（假设）；
    rule="sign"：按正负号切分（备选）。
    """
    n = len(P)
    if n < 2 * min_side:
        return None
    A = affinity(P, squared)
    dinv = 1.0 / np.sqrt(A.sum(1))
    M = dinv[:, None] * A * dinv[None, :]
    w, V = np.linalg.eigh(M)               # 升序；最大为 1（对应常数向量 D^{1/2}1）
    e = V[:, -2]
    if rule == "sign":
        lab = (e > 0).astype(int)
    else:
        order = np.argsort(e)
        es = e[order]
        gaps = np.diff(es)
        valid = np.arange(min_side - 1, n - min_side)   # 间隙位置 k：左侧 k+1 个
        if len(valid) == 0:
            return None
        k = valid[np.argmax(gaps[valid])]
        lab = np.zeros(n, int)
        lab[order[k + 1:]] = 1
    if lab.sum() < min_side or (n - lab.sum()) < min_side:
        return None
    return lab, e


def pocket_perceptron(X: np.ndarray, lab: np.ndarray, epochs: int = 100, seed: int = 0):
    """单层感知机（口袋算法）最小化总误分类数。返回原尺度下的 (a, c)：预测 a·x <= c 为一侧。"""
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    Zb = np.column_stack([Z, np.ones(len(Z))])
    yy = np.where(lab == 1, 1.0, -1.0)
    # 初始化：类均值差方向（对应 Fisher 方向的简化），阈值取中点
    m1, m0 = Z[lab == 1].mean(0), Z[lab == 0].mean(0)
    w = np.append(m1 - m0, -0.5 * float((m1 - m0) @ (m1 + m0)))
    best_w, best_err = w.copy(), int((np.sign(Zb @ w) != yy).sum())
    for _ in range(epochs):
        for i in rng.permutation(len(Zb)):
            if yy[i] * (Zb[i] @ w) <= 0:
                w = w + yy[i] * Zb[i]
        err = int((np.sign(Zb @ w) != yy).sum())
        if err < best_err:
            best_w, best_err = w.copy(), err
            if err == 0:
                break
    # 原尺度：w·((x-mu)/sd) + b > 0 为类 1；转成 a·x <= c 为类 0
    a = best_w[:-1] / sd
    c = -(best_w[-1] - float(a @ mu))
    return a, c, best_err
