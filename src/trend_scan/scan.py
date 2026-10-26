"""趋势区间识别（detection）——事后（ex-post）描述。

固定长度 N 的滑动窗口沿交易日逐日移动；窗口含 N 个交易日（首日 i，末日 i+N-1）。
累计收益 cum_ret = P_end / P_start - 1。
  cum_ret >= up_threshold   -> 'up'
  cum_ret <= down_threshold -> 'down'
  其他                      -> 不标记

!!! 重要：判断一个窗口是否为上涨/下跌区间需要用到窗口末日价格，
即在窗口起点时刻属于“未来数据”。因此本模块输出只能用于事后描述性分析，
不能直接视为当时可执行的交易信号。可执行条件请见 conditions.py（仅可使用截至判断时刻的信息）。
"""
from __future__ import annotations

import numpy as np

UP, DOWN = "up", "down"


def window_cum_returns(price: np.ndarray, n: int) -> np.ndarray:
    """返回所有窗口（起点 i=0..T-n）的累计收益，长度 T-n+1。"""
    price = np.asarray(price, dtype=float)
    if len(price) < n:
        return np.empty(0)
    return price[n - 1:] / price[: len(price) - n + 1] - 1.0


def detect_windows(price: np.ndarray, n: int, up: float, down: float):
    """返回 (starts, cum_ret, labels)，仅包含被标记为 up/down 的窗口（允许重叠）。"""
    cr = window_cum_returns(price, n)
    is_up = cr >= up
    is_down = cr <= down
    starts = np.flatnonzero(is_up | is_down)
    labels = np.where(is_up[starts], UP, DOWN)
    return starts, cr[starts], labels, len(cr)
