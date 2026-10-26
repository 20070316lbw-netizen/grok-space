"""日线数据：yfinance 批量下载（period='max', auto_adjust=True），按股票缓存 parquet。

- 复权选择：auto_adjust=True，Close 为经拆股与分红调整后的价格（近似总收益口径）。
  注意 yfinance 的复权价格会随新的分红/拆股事件整体回溯调整，因此记录下载时间。
- 幂等：已存在的 parquet 默认跳过，--refresh 时重新下载。
- 失败的股票写入 data/raw/prices/_failed.csv；所有状态写入 _download_log.csv。
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .paths import PRICES_DIR

log = logging.getLogger(__name__)
COLS = ["Open", "High", "Low", "Close", "Volume"]


def price_path(ticker: str) -> Path:
    return PRICES_DIR / f"{ticker}.parquet"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[[c for c in COLS if c in df.columns]].copy()
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _save(ticker: str, df: pd.DataFrame) -> None:
    df.to_parquet(price_path(ticker))


def _download_batch(batch: list[str]) -> dict[str, pd.DataFrame]:
    raw = yf.download(batch, period="max", auto_adjust=True, actions=False,
                      group_by="ticker", threads=True, progress=False)
    out = {}
    for t in batch:
        try:
            sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        sub = _clean(sub)
        if not sub.empty:
            out[t] = sub
    return out


def _download_single(ticker: str) -> pd.DataFrame:
    return _clean(yf.Ticker(ticker).history(period="max", auto_adjust=True, actions=False))


def download_prices(tickers: list[str], refresh: bool = False, batch_size: int = 50,
                    max_retries: int = 3, pause: float = 2.0) -> pd.DataFrame:
    """下载并缓存，返回下载日志 DataFrame。"""
    now = dt.datetime.now().isoformat(timespec="seconds")
    status: dict[str, dict] = {}
    todo = []
    for t in tickers:
        if price_path(t).exists() and not refresh:
            status[t] = {"ticker": t, "status": "cached"}
        else:
            todo.append(t)
    log.info("共 %d 只，需下载 %d 只（缓存跳过 %d）", len(tickers), len(todo), len(tickers) - len(todo))

    pending = list(todo)
    for attempt in range(1, max_retries + 1):
        if not pending:
            break
        failed = []
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            try:
                got = _download_batch(batch)
            except Exception as e:  # 网络/限流等
                log.warning("批次下载异常(第%d轮): %s", attempt, e)
                got = {}
            for t in batch:
                if t in got:
                    _save(t, got[t])
                    status[t] = {"ticker": t, "status": "ok", "attempt": attempt, "mode": "batch"}
                else:
                    failed.append(t)
            log.info("第%d轮 批次 %d-%d：成功 %d/%d", attempt, i, i + len(batch), len(got), len(batch))
            time.sleep(pause)
        pending = failed
        if pending:
            log.warning("第%d轮后仍失败 %d 只，等待后重试", attempt, len(pending))
            time.sleep(pause * 2 * attempt)

    # 逐只兜底
    still = []
    for t in pending:
        df = pd.DataFrame()
        for k in range(max_retries):
            try:
                df = _download_single(t)
                if not df.empty:
                    break
            except Exception as e:
                log.warning("%s 单只下载异常: %s", t, e)
            time.sleep(pause * (k + 1))
        if df.empty:
            still.append(t)
            status[t] = {"ticker": t, "status": "failed"}
        else:
            _save(t, df)
            status[t] = {"ticker": t, "status": "ok", "attempt": max_retries + 1, "mode": "single"}

    rows = []
    for t in tickers:
        r = dict(status.get(t, {"ticker": t, "status": "failed"}))
        p = price_path(t)
        if p.exists():
            d = pd.read_parquet(p, columns=["Close"])
            r.update(rows=len(d), first_date=d.index.min().date().isoformat(),
                     last_date=d.index.max().date().isoformat())
        r["checked_at"] = now
        rows.append(r)
    logdf = pd.DataFrame(rows)
    # 合并历史下载日志
    lp = PRICES_DIR / "_download_log.csv"
    logdf.to_csv(lp, index=False)
    pd.DataFrame({"ticker": still, "checked_at": now}).to_csv(PRICES_DIR / "_failed.csv", index=False)
    log.info("下载完成：ok/cached %d，失败 %d", (logdf.status != "failed").sum(), len(still))
    return logdf


def load_close(tickers: list[str], field: str = "Close", min_days: int = 100,
               start: str | None = None, end: str | None = None) -> dict[str, pd.Series]:
    """读取缓存价格并截取分析区间 [start, end]，返回 {ticker: Series(Date->price)}。"""
    out = {}
    for t in tickers:
        p = price_path(t)
        if not p.exists():
            continue
        s = pd.read_parquet(p, columns=[field])[field].astype(float)
        s = s[np.isfinite(s) & (s > 0)]
        if start:
            s = s[s.index >= pd.Timestamp(start)]
        if end:
            s = s[s.index <= pd.Timestamp(end)]
        if len(s) >= min_days:
            out[t] = s
    return out


def data_quality_flags(closes: dict[str, pd.Series], max_abs_logret: float = float(np.log(2))) -> pd.DataFrame:
    """标记单日 |log 收益| 超过阈值的异常跳变（仅标记，不删除），便于人工核查 Yahoo 数据错误。"""
    rows = []
    for t, s in closes.items():
        r = np.log(s).diff()
        for d, v in r[r.abs() > max_abs_logret].items():
            i = s.index.get_loc(d)
            rows.append({"ticker": t, "date": d.date().isoformat(), "prev_close": float(s.iloc[i - 1]),
                         "close": float(s.iloc[i]), "simple_return": float(np.expm1(v))})
    return pd.DataFrame(rows, columns=["ticker", "date", "prev_close", "close", "simple_return"])
