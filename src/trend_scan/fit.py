"""线性拟合（fitting）：对窗口内 log(价格) 关于交易日序号 x=0..N-1 做 OLS。

使用前缀和（cumsum）闭式解，一次性得到所有滑动窗口的斜率/截距/R^2/残差标准差/t 统计量，
再只对被标记的窗口用 sliding_window_view 计算最大绝对残差。
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def rolling_ols_log(price: np.ndarray, n: int, ann_days: int = 252) -> dict[str, np.ndarray]:
    """对所有窗口起点 i（0..T-n）计算 OLS 指标，返回 dict of arrays（长度 T-n+1）。"""
    y_raw = np.log(np.asarray(price, dtype=float))
    T = len(y_raw)
    m = T - n + 1
    if m <= 0:
        return {}
    c = y_raw.mean()           # 去中心化以降低前缀和的数值误差
    y = y_raw - c
    j = np.arange(T, dtype=float)
    C1 = np.concatenate([[0.0], np.cumsum(y)])
    Cj = np.concatenate([[0.0], np.cumsum(j * y)])
    C2 = np.concatenate([[0.0], np.cumsum(y * y)])
    i = np.arange(m, dtype=float)
    a, b = np.arange(m), np.arange(m) + n
    Sy = C1[b] - C1[a]
    Sjy = Cj[b] - Cj[a]
    Syy = C2[b] - C2[a]
    xbar = (n - 1) / 2.0
    Sxx = n * (n * n - 1) / 12.0
    Sxy_c = (Sjy - i * Sy) - xbar * Sy          # sum (x - xbar) * y
    Syy_c = Syy - Sy * Sy / n                   # sum (y - ybar)^2
    slope = Sxy_c / Sxx
    intercept = Sy / n - slope * xbar + c       # 窗口首日（x=0）处的拟合 log 价格
    sse = np.maximum(Syy_c - Sxy_c * Sxy_c / Sxx, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(Syy_c > 1e-14, 1.0 - sse / Syy_c, np.nan)
        dof = max(n - 2, 1)
        resid_std = np.sqrt(sse / dof)
        se_slope = resid_std / np.sqrt(Sxx)
        tstat = np.where(se_slope > 0, slope / se_slope, np.nan)
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": np.clip(r2, 0.0, 1.0),
        "resid_std": resid_std,
        "slope_tstat": tstat,
        "slope_annual": slope * ann_days,                 # 年化对数斜率
        "growth_annual": np.expm1(slope * ann_days),      # 对应年化增长率
    }


def max_abs_residual(price: np.ndarray, n: int, starts: np.ndarray,
                     slope: np.ndarray, intercept: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """对指定窗口计算最大绝对残差（log 尺度）。slope/intercept 已按 starts 取好。"""
    y = np.log(np.asarray(price, dtype=float))
    W = sliding_window_view(y, n)
    x = np.arange(n, dtype=float)
    out = np.empty(len(starts))
    for k in range(0, len(starts), chunk):
        s = starts[k:k + chunk]
        fitted = intercept[k:k + chunk, None] + slope[k:k + chunk, None] * x
        out[k:k + chunk] = np.abs(W[s] - fitted).max(axis=1)
    return out
