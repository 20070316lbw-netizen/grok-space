"""叶子模型：x_t ~ N(m + sum_j b_j x_{t-j}, sigma^2)，共轭 Normal-Inverse-Gamma 先验。

论文使用 Meek et al. (2002) 的贝叶斯评分但未给出细节（本文只说 "described in detail in that paper"）。
这里的假设（见 notes 第 7 节）：
- beta | s2 ~ N(m0, s2 * V0)，s2 ~ IG(a0, b0)，所有叶子共用同一先验（经验贝叶斯：取自根节点数据）；
- m0 = 根节点 OLS 系数，V0 = n0^{-1} * (X'X_root / n_root)^{-1}（"单位信息"先验，等效样本量 n0）；
- a0 = 1，b0 = a0 * s2_root。
节点得分 = log 边际似然；切分得分 = 子节点得分之和 - 父节点得分（>0 才切分，无剪枝）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LOG2PI = math.log(2 * math.pi)


@dataclass
class Prior:
    m0: np.ndarray
    V0inv: np.ndarray
    logdetV0: float
    a0: float
    b0: float


def design(Xreg: np.ndarray) -> np.ndarray:
    """加截距列。Xreg: (n, p)。"""
    return np.column_stack([np.ones(len(Xreg)), Xreg]) if Xreg.size else np.ones((len(Xreg), 1))


def make_prior(X: np.ndarray, y: np.ndarray, n0: float = 1.0, a0: float = 1.0, ridge: float = 1e-8) -> Prior:
    n, k = X.shape
    XtX = X.T @ X + ridge * np.eye(k) * max(1.0, np.trace(X.T @ X) / k)
    beta = np.linalg.solve(XtX, X.T @ y)
    resid = y - X @ beta
    s2 = max(float(resid @ resid) / max(n - k, 1), 1e-12)
    V0inv = n0 * XtX / n
    sign, logdet = np.linalg.slogdet(V0inv)
    return Prior(m0=beta, V0inv=V0inv, logdetV0=-logdet, a0=a0, b0=a0 * s2)


@dataclass
class Posterior:
    mn: np.ndarray
    Vn: np.ndarray
    an: float
    bn: float
    logml: float


def posterior(X: np.ndarray, y: np.ndarray, pr: Prior) -> Posterior:
    n = len(y)
    Vninv = pr.V0inv + X.T @ X
    rhs = pr.V0inv @ pr.m0 + X.T @ y
    L = np.linalg.cholesky(Vninv)
    mn = np.linalg.solve(Vninv, rhs)
    logdetVn = -2.0 * np.log(np.diag(L)).sum()
    an = pr.a0 + n / 2.0
    bn = pr.b0 + 0.5 * (float(y @ y) + float(pr.m0 @ pr.V0inv @ pr.m0) - float(mn @ Vninv @ mn))
    bn = max(bn, 1e-300)
    logml = (-0.5 * n * LOG2PI + 0.5 * (logdetVn - pr.logdetV0)
             + pr.a0 * math.log(pr.b0) - an * math.log(bn) + math.lgamma(an) - math.lgamma(pr.a0))
    return Posterior(mn=mn, Vn=np.linalg.inv(Vninv), an=an, bn=bn, logml=logml)


def predictive_logpdf(x: np.ndarray, y: float, post: Posterior) -> float:
    """后验预测分布（Student-t，自由度 2an）在 y 处的对数密度。x 含截距。"""
    nu = 2.0 * post.an
    loc = float(x @ post.mn)
    scale2 = post.bn / post.an * (1.0 + float(x @ post.Vn @ x))
    z2 = (y - loc) ** 2 / scale2
    return (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi * scale2)
            - (nu + 1) / 2 * math.log1p(z2 / nu))


def predictive_mean(x: np.ndarray, post: Posterior) -> float:
    return float(x @ post.mn)
