"""论文复现命令行入口（[project.scripts] paper-repro = paper_repro.cli:main）。

uv run paper-repro synthetic [--out DIR]              # 合成状态切换数据验证 → outputs/repro_bocharov/synthetic_*.csv
uv run paper-repro sp500 [--ticker-frac F]            # 价格水平版 S&P 500 实验 → outputs/repro_bocharov/sp500_*
uv run paper-repro returns [--standardize]            # 日对数收益率版 → outputs/repro_bocharov/returns[_std]/

数据依赖：sp500 / returns 读取 trend_scan 下载的价格缓存（先运行 `uv run trend-scan download`）。
"""
from __future__ import annotations

import argparse
import os


def main():
    # 多进程实验：每个进程单线程 BLAS 更快（等价于旧版 README 中的 OMP_NUM_THREADS=1）。须在导入 numpy 前设置。
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ap = argparse.ArgumentParser(prog="paper-repro")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("synthetic", help="合成数据验证")
    p.add_argument("--out", default=None, help="输出目录（默认 outputs/repro_bocharov）")
    p = sub.add_parser("sp500", help="价格水平版 S&P 500 实验")
    p.add_argument("--ticker-frac", type=float, default=1.0, help="抽样股票比例（调试用）")
    p.add_argument("--procs", type=int, default=8)
    p = sub.add_parser("returns", help="日对数收益率版实验")
    p.add_argument("--standardize", action="store_true", help="标准化收益率变体 → returns_std/")
    p.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "synthetic":
        from paper_repro import synthetic
        synthetic.main(a.out)
    elif a.cmd == "sp500":
        from paper_repro import sp500
        sp500.main(ticker_frac=a.ticker_frac, procs=a.procs)
    elif a.cmd == "returns":
        from paper_repro import returns_exp
        returns_exp.main(standardize=a.standardize, procs=a.procs)


if __name__ == "__main__":
    main()
