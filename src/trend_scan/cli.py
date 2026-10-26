"""命令行入口（[project.scripts] trend-scan = trend_scan.cli:main）。

uv run trend-scan universe [--refresh-universe]
uv run trend-scan download [--refresh] [--limit K] [--tickers-file f.csv]
uv run trend-scan scan [--config configs/trend_scan/default.yaml] [--limit K]
uv run trend-scan report
uv run trend-scan features [--check-only]   # 生成 ML 时点特征数据集 + 完整性检查
uv run trend-scan all
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time

from trend_scan import data, experiment, report, universe
from trend_scan.paths import DEFAULT_CONFIG, LOGS_DIR, ROOT

log = logging.getLogger("run")


def _setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "run.log", encoding="utf-8")])


def get_universe(args, cfg):
    u = cfg.get("universe", {}) if cfg else {}
    src = "file" if args.tickers_file else u.get("source", "wikipedia")
    tf = args.tickers_file or u.get("tickers_file")
    df = universe.load_universe(src, tf, refresh=getattr(args, "refresh_universe", False))
    tickers = df["ticker"].tolist()
    if args.limit:
        tickers = tickers[: args.limit]
    meta = {"source": src, "tickers_file": tf, "n_tickers": len(tickers),
            "scrape_date": str(df["scrape_date"].iloc[0]) if "scrape_date" in df else None}
    return tickers, meta, df


def du(path) -> str:
    return subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True).stdout.split()[0]


def main():
    _setup_logging()
    ap = argparse.ArgumentParser(prog="trend-scan")
    ap.add_argument("cmd", choices=["universe", "download", "scan", "report", "features", "all"])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--refresh", action="store_true", help="重新下载价格（忽略缓存）")
    ap.add_argument("--refresh-universe", action="store_true", help="重新抓取维基百科成分股")
    ap.add_argument("--tickers-file", default=None, help="自定义股票列表 CSV（含 ticker 列）")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 K 只（调试）")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--check-only", action="store_true", help="features：只运行完整性检查，不重建")
    args = ap.parse_args()
    cfg = experiment.load_config(args.config)
    timings = {}

    if args.cmd == "universe":
        df = universe.scrape_wikipedia(save=True)
        print(f"成分股 {len(df)} 只")
        return
    tickers, umeta, udf = get_universe(args, cfg)

    if args.cmd in ("download", "all"):
        t0 = time.time()
        lg = data.download_prices(tickers, refresh=args.refresh, batch_size=args.batch_size)
        timings["download_seconds"] = round(time.time() - t0, 1)
        print(lg["status"].value_counts().to_string())

    if args.cmd in ("scan", "all"):
        t0 = time.time()
        dc = cfg["data"]
        closes = data.load_close(tickers, dc["price_field"], dc["min_history_days"],
                                 dc.get("start"), dc.get("end"))
        flags = data.data_quality_flags(closes)
        flags.to_csv(ROOT / "reports" / "data_quality_flags.csv", index=False)
        log.info("分析区间 %s ~ %s，可用股票 %d，异常跳变标记 %d 条", dc.get("start"), dc.get("end"),
                 len(closes), len(flags))
        timings["load_seconds"] = round(time.time() - t0, 1)
        umeta["n_tickers_with_data"] = len(closes)
        umeta["n_tickers_excluded_short_history"] = len(tickers) - len(closes)
        umeta["analysis_start"], umeta["analysis_end"] = dc.get("start"), dc.get("end")
        t0 = time.time()
        groups = experiment.load_groups(udf)
        groups.to_csv(ROOT / "data" / "processed" / "ticker_groups.csv", index=False)
        ids = experiment.run_grid(cfg, closes, umeta, args.config, groups)
        timings["scan_fit_total_seconds"] = round(time.time() - t0, 1)
        print("\n".join(ids))

    if args.cmd == "features":
        from trend_scan import features, features_check
        groups = experiment.load_groups(udf)
        if not args.check_only:
            meta = features.build_features(cfg, tickers, groups)
            print(json.dumps({"rows": meta["rows"], "n_tickers": meta["n_tickers"],
                              "seconds": meta["total_seconds"]}, ensure_ascii=False))
        t0 = time.time()
        chk = features_check.run_checks(cfg)
        print(json.dumps({"checks_passed": chk["all_passed"], "check_seconds": round(time.time() - t0, 1)}))
        return

    if args.cmd in ("report", "all", "scan"):
        lt = LOGS_DIR / "last_timings.json"
        if not timings and lt.exists():
            timings_for_report = json.loads(lt.read_text())
        else:
            timings_for_report = timings
        extra = {**timings_for_report, "data/raw 磁盘占用": du(ROOT / "data" / "raw"),
                 "data/processed 磁盘占用": du(ROOT / "data" / "processed")}
        out = report.build_report(extra)
        print("report:", out)
    if timings:
        (LOGS_DIR / "last_timings.json").write_text(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
