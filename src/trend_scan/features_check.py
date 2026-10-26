"""特征数据集完整性检查：无前视（截断重算）、与 scan 结果一致、标签定义正确。结果写入 reports/features_sanity.json。"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import features as F
from .data import load_close
from .experiment import combo_key
from .paths import PROCESSED_DIR, REPORTS_DIR

log = logging.getLogger(__name__)
METRICS = ["cum_ret", "slope", "r2", "resid_std", "slope_tstat"]


def _rel_close(a, b, rtol=2e-4, atol=2e-6):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both_nan = np.isnan(a) & np.isnan(b)
    return bool(np.all(both_nan | np.isclose(a, b, rtol=rtol, atol=atol)))


def run_checks(cfg: dict, n_samples: int = 40, seed: int = 0) -> dict:
    fc = {"windows": [5, 10, 20, 30, 40], "horizons": [5, 10, 20], "up": 0.08, "down": -0.08,
          **(cfg.get("features") or {})}
    df = pd.read_parquet(F.FEATURES_DIR)
    df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    res: dict = {"rows": len(df)}
    rng = np.random.default_rng(seed)

    # 1) 无前视：随机抽样 (ticker, t)，用截断到 t 的价格重新计算，并与 np.polyfit 直接计算对比
    idx = rng.choice(len(df), n_samples, replace=False)
    full = load_close(df["ticker"].unique().tolist(), cfg["data"]["price_field"], 1)
    ok_trunc, ok_poly, details = True, True, []
    for i in idx:
        r = df.iloc[i]
        s = full[r["ticker"]]
        p_trunc = s[s.index <= r["date"]].to_numpy(float)          # 只保留 <= t 的价格
        for n in fc["windows"]:
            f = F.trailing_features(p_trunc, n, fc["up"], fc["down"])
            stored = [r[f"{m}_{n}"] for m in METRICS + ["resid_last"]]
            recomputed = [f[m][-1] for m in METRICS + ["resid_last"]]
            ok1 = _rel_close(stored, recomputed)
            ok_trunc &= ok1
            if len(p_trunc) >= n:
                y = np.log(p_trunc[-n:]); x = np.arange(n)
                b, a = np.polyfit(x, y, 1); e = y - (a + b * x)
                r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
                ok2 = _rel_close([r[f"slope_{n}"], r[f"r2_{n}"], r[f"resid_last_{n}"], r[f"cum_ret_{n}"]],
                                 [b, r2, e[-1], p_trunc[-1] / p_trunc[-n] - 1])
                ok_poly &= ok2
            if not ok1:
                details.append({"ticker": r["ticker"], "date": str(r["date"].date()), "n": n})
    res["no_lookahead_truncation"] = {"samples": int(n_samples), "windows": fc["windows"], "passed": bool(ok_trunc),
                                      "polyfit_match": bool(ok_poly), "failures": details[:10]}

    # 2) 与 scan 结果一致：scan 中以 t 为末日的窗口 == 时点特征在 t 的值
    df["_pos"] = df.groupby("ticker").cumcount()        # 在分析区间内的位置（用于对齐 scan 的窗口起点约束）
    scan_res = {}
    for n in fc["windows"]:
        key = combo_key(n, fc["up"], fc["down"])
        out = {}
        for lab in ("up", "down"):
            p = PROCESSED_DIR / key / f"{lab}.parquet"
            if not p.exists():
                continue
            s = pd.read_parquet(p, columns=["ticker", "end_date", "cum_return"] + METRICS[1:])
            s = s.rename(columns={"cum_return": "cum_ret"})
            s["ticker"] = s["ticker"].astype(str)
            m = s.merge(df[["ticker", "date", f"in_{lab}_{n}"] + [f"{k}_{n}" for k in METRICS]],
                        left_on=["ticker", "end_date"], right_on=["ticker", "date"], how="left")
            maxdiff = {k: float(np.nanmax(np.abs(m[k].to_numpy(float) - m[f"{k}_{n}"].to_numpy(float))
                                          / np.maximum(1.0, np.abs(m[k].to_numpy(float))))) for k in METRICS}
            flag_ok = bool(m[f"in_{lab}_{n}"].fillna(False).all())
            feat_cnt = int((df[f"in_{lab}_{n}"].fillna(False) & (df["_pos"] >= n - 1)).sum())
            out[lab] = {"scan_windows": len(s), "matched_rows": int(m["date"].notna().sum()),
                        "flag_all_true": flag_ok, "count_equal": feat_cnt == len(s),
                        "feature_flag_count": feat_cnt, "max_rel_diff": maxdiff,
                        "passed": bool(flag_ok and feat_cnt == len(s) and m["date"].notna().all()
                                       and max(v for k, v in maxdiff.items() if k != "slope_tstat") < 1e-4
                                       and maxdiff["slope_tstat"] < 1e-3),
                        "tolerance": "max_rel_diff < 1e-4（slope_tstat < 1e-3：R²≈1 时 SSE 极小，前缀和闭式解的抵消误差会放大到 t 值）"}
        scan_res[n] = out
    res["scan_consistency"] = scan_res

    # 3) 标签：fwd_ret_H == adj_close(t+H)/adj_close(t) - 1（按 ticker 平移），且每只股票最后 H 行为 NaN
    lab_res = {}
    g = df.groupby("ticker")["adj_close"]
    for h in fc["horizons"]:
        exp = (g.shift(-h) / df["adj_close"] - 1).to_numpy(float)
        got = df[f"fwd_ret_{h}"].to_numpy(float)
        tail_nan = bool(df.groupby("ticker")[f"fwd_ret_{h}"].apply(lambda x: x.iloc[-h:].isna().all()).all())
        cross_exp = df.groupby("ticker")["split"].shift(-h).astype(object) != df["split"].astype(object)
        cross_got = df[f"fwd_crosses_split_{h}"]
        valid = df[f"fwd_ret_{h}"].notna()
        lab_res[h] = {"fwd_ret_matches_shifted_price": _rel_close(exp, got, rtol=1e-5, atol=1e-6),
                      "last_h_rows_nan": tail_nan,
                      "crosses_split_matches": bool((cross_exp[valid] == cross_got[valid]).all()),
                      "crosses_split_share": float(cross_got[valid].mean())}
    res["labels"] = lab_res

    # 4) 列角色：特征列中不含 fwd_ 前缀
    feat_cols = [c for c in df.columns if c.split("_")[0] in ("cum", "slope", "r2", "resid", "in")]
    res["no_fwd_in_feature_columns"] = not any(c.startswith("fwd_") for c in feat_cols)

    res["all_passed"] = bool(res["no_lookahead_truncation"]["passed"] and res["no_lookahead_truncation"]["polyfit_match"]
                             and all(v["passed"] for d in scan_res.values() for v in d.values())
                             and all(v["fwd_ret_matches_shifted_price"] and v["last_h_rows_nan"] and v["crosses_split_matches"]
                                     for v in lab_res.values())
                             and res["no_fwd_in_feature_columns"])

    # 汇总统计：缺失率 + window=20 描述统计
    num = [c for c in df.columns if df[c].dtype.kind == "f"]
    nan_rate = df[num].isna().mean().rename("nan_rate")
    nan_rate.to_csv(REPORTS_DIR / "features_nan_rates.csv")
    w20 = [c for c in df.columns if c.endswith("_20") and df[c].dtype.kind == "f"] + \
          [f"fwd_ret_{h}" for h in fc["horizons"]]
    desc = df[w20].astype(float).describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    desc.to_csv(REPORTS_DIR / "features_describe_w20.csv")
    flags = {c: float(df[c].astype(float).mean()) for c in df.columns if c.startswith(("in_up_", "in_down_", "feat_window"))}
    res["flag_true_rates"] = flags
    res["rows_by_split"] = df["split"].astype(str).value_counts().to_dict()
    (REPORTS_DIR / "features_sanity.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    log.info("特征检查 all_passed=%s", res["all_passed"])
    return res
