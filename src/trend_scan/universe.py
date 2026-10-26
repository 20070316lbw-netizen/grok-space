"""股票池：抓取维基百科当前 S&P 500 成分股，或加载自定义股票列表。

注意：维基百科给出的是 *当前* 成分股，用于长历史研究存在幸存者偏差。
本模块的 load_universe() 支持 source=file，以便后续接入历史成分股/退市股票列表。
"""
from __future__ import annotations

import datetime as dt
import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from .paths import UNIVERSE_DIR

log = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def to_yf_ticker(sym: str) -> str:
    """维基百科代码 -> yfinance 代码（如 BRK.B -> BRK-B）。"""
    return str(sym).strip().upper().replace(".", "-")


def scrape_wikipedia(save: bool = True) -> pd.DataFrame:
    resp = requests.get(WIKI_URL, headers={"User-Agent": UA}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    df = None
    for t in tables:
        if "Symbol" in t.columns:
            df = t.copy()
            break
    if df is None:
        raise RuntimeError("未在维基百科页面中找到含 Symbol 列的成分股表格")
    df = df.rename(columns={"Symbol": "wiki_symbol"})
    df.insert(0, "ticker", df["wiki_symbol"].map(to_yf_ticker))
    today = dt.date.today().isoformat()
    df["scrape_date"] = today
    df["source_url"] = WIKI_URL
    df = df.drop_duplicates("ticker").reset_index(drop=True)
    if save:
        dated = UNIVERSE_DIR / f"sp500_constituents_{today.replace('-', '')}.csv"
        df.to_csv(dated, index=False)
        df.to_csv(UNIVERSE_DIR / "sp500_constituents_latest.csv", index=False)
        log.info("成分股 %d 只，已保存 %s", len(df), dated)
    return df


def load_universe(source: str = "wikipedia", tickers_file: str | None = None,
                  refresh: bool = False) -> pd.DataFrame:
    """返回至少含 ticker 列的 DataFrame。

    source='wikipedia'：优先使用已缓存的 latest 文件（除非 refresh）。
    source='file'：读取自定义 CSV（需含 ticker 列，可附带 start/end 有效期等字段，供未来扩展）。
    """
    if source == "file":
        if not tickers_file:
            raise ValueError("source=file 需要提供 tickers_file")
        df = pd.read_csv(tickers_file)
        df["ticker"] = df["ticker"].map(to_yf_ticker)
        return df
    latest = UNIVERSE_DIR / "sp500_constituents_latest.csv"
    if latest.exists() and not refresh:
        return pd.read_csv(latest)
    return scrape_wikipedia(save=True)
