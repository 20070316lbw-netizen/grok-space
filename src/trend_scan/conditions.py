"""条件函数占位（stage 2 预留）。

与 scan.py 的区别：
- scan.py 的“上涨/下跌区间”是用窗口末日价格事后识别的（ex-post），含未来信息；
- 本模块的条件函数将来只能使用“判断时刻及之前”的信息构造特征（例如用过去 N 日
  计算的累计收益、回归斜率、R^2），才可能成为可执行的信号。

具体条件、交易方向、持仓规则与评价指标尚未确定，这里仅提供接口与示例，
并且参数选择必须只在 search 区间完成、在 eval 区间做一次性评估。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ConditionParams:
    min_cum_return: float | None = None
    min_slope: float | None = None
    min_r2: float | None = None


def condition(features: Mapping[str, float], params: ConditionParams = ConditionParams()) -> bool:
    """示例条件：features 需包含 cum_return / slope / r2（必须是“截至当日”可得的回看特征）。

    当前仅为占位实现，未接入任何回测流程。
    """
    checks = [
        (params.min_cum_return, features.get("cum_return")),
        (params.min_slope, features.get("slope")),
        (params.min_r2, features.get("r2")),
    ]
    for thr, val in checks:
        if thr is not None and (val is None or val < thr):
            return False
    return True
