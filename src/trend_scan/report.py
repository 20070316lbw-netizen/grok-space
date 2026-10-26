"""图表与中文报告（只读取真实输出文件生成，不手写任何数字）。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import EXPERIMENTS_DIR, PROCESSED_DIR, REPORTS_DIR


def latest_run(index: pd.DataFrame) -> pd.DataFrame:
    return index[index["run_timestamp"] == index["run_timestamp"].max()]


def plot_r2_hist(key: str, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, split in zip(axes, ("search", "eval")):
        for lab, color in (("up", "tab:green"), ("down", "tab:red")):
            d = pd.read_parquet(PROCESSED_DIR / key / f"{lab}.parquet", columns=["r2", "split"])
            d = d[d["split"] == split]
            ax.hist(d["r2"], bins=50, range=(0, 1), density=True, alpha=0.5, color=color,
                    label=f"{lab} (n={len(d):,}, median={d['r2'].median():.3f})")
        ax.set_title(f"{key}  split={split}")
        ax.set_xlabel("R² of OLS on log(price)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_counts(run: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    d = run[run["split"].isin(["search", "eval"])]
    wins = sorted(d["window"].unique())
    x = np.arange(len(wins))
    w = 0.2
    k = 0
    for split in ("search", "eval"):
        for lab, color in (("up", "tab:green"), ("down", "tab:red")):
            s = d[(d.split == split) & (d.label == lab)].set_index("window").reindex(wins)
            hatch = "" if split == "search" else "//"
            axes[0].bar(x + (k - 1.5) * w, s["count"], w, color=color, hatch=hatch,
                        alpha=0.8, label=f"{split}-{lab}")
            axes[1].plot(wins, s["r2_q50"], marker="o", color=color,
                         linestyle="-" if split == "search" else "--", label=f"{split}-{lab}")
            k += 1
    axes[0].set_xticks(x, [str(v) for v in wins])
    axes[0].set_xlabel("window length N (trading days)")
    axes[0].set_ylabel("labeled windows (overlapping)")
    axes[0].set_title("Count of labeled windows by window length")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("window length N")
    axes[1].set_ylabel("median R²")
    axes[1].set_title("Median R² by window length")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _short(name: str, k: int = 28) -> str:
    return name if len(name) <= k else name[: k - 1] + "…"


def plot_group_heatmap(g: pd.DataFrame, out: Path, split: str = "search") -> Path:
    """median R^2 热力图：分组 × 窗口长度，上涨/下跌各一幅。"""
    d = g[g["split"] == split]
    groups = sorted(d["group"].unique())
    wins = sorted(d["window"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.45 * len(groups) + 2), sharey=True)
    for ax, lab in zip(axes, ("up", "down")):
        m = d.pivot_table(index="group", columns="window", values=f"{lab}_r2_median").reindex(groups)[wins]
        im = ax.imshow(m.to_numpy(), aspect="auto", cmap="viridis", vmin=0.55, vmax=0.85)
        ax.set_xticks(range(len(wins)), [str(w) for w in wins])
        ax.set_yticks(range(len(groups)), [_short(x) for x in groups])
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                            color="white" if v < 0.72 else "black")
        ax.set_title(f"median R² ({lab}), split={split}")
        ax.set_xlabel("window length N")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_group_counts(g: pd.DataFrame, window: int, out: Path) -> Path:
    """window=N 下各分组 up/down 数量（search / eval）。"""
    d = g[g["window"] == window]
    groups = sorted(d["group"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.4 * len(groups) + 2), sharey=True)
    y = np.arange(len(groups))
    for ax, split in zip(axes, ("search", "eval")):
        s = d[d["split"] == split].set_index("group").reindex(groups)
        ax.barh(y - 0.2, s["up_count"], 0.4, color="tab:green", label="up")
        ax.barh(y + 0.2, s["down_count"], 0.4, color="tab:red", label="down")
        for yi, (u, dn, nt) in enumerate(zip(s["up_count"], s["down_count"], s["n_tickers_in_group"])):
            ax.text(max(u, dn) * 1.01, yi, f"up/down={(u / dn) if dn else float('nan'):.2f}, n_tk={nt}", va="center", fontsize=7)
        ax.set_yticks(y, [_short(x) for x in groups])
        ax.set_title(f"window={window}, split={split}: labeled windows by group")
        ax.set_xlabel("count (overlapping windows)")
        ax.margins(x=0.35)
        ax.legend(fontsize=8)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def build_report(extra: dict | None = None) -> Path:
    index = pd.read_csv(EXPERIMENTS_DIR / "index.csv")
    run = latest_run(index)
    charts = []
    keys = run[["window", "up", "down"]].drop_duplicates()
    k20 = keys[keys.window == 20]
    if len(k20):
        r = k20.iloc[0]
        key = f"w{int(r.window)}_u{r.up:g}_d{r.down:g}"
        charts.append(plot_r2_hist(key, REPORTS_DIR / f"r2_hist_{key}.png"))
    charts.append(plot_counts(run, REPORTS_DIR / "counts_and_median_r2_by_window.png"))

    group_tab, gby = None, None
    gidx = EXPERIMENTS_DIR / "index_by_group.csv"
    if gidx.exists():
        gall = pd.read_csv(gidx)
        g = gall[gall["run_timestamp"].astype(str) == str(run["run_timestamp"].iloc[0])]
        if len(g):
            gby = g["group_by"].iloc[0]
            charts.append(plot_group_heatmap(g, REPORTS_DIR / f"heatmap_median_r2_{gby}_x_window.png", "search"))
            charts.append(plot_group_heatmap(g, REPORTS_DIR / f"heatmap_median_r2_{gby}_x_window_eval.png", "eval"))
            w0 = 20 if 20 in set(g["window"]) else int(g["window"].min())
            charts.append(plot_group_counts(g, w0, REPORTS_DIR / f"counts_by_{gby}_w{w0}.png"))
            gcols = ["window", "split", "group", "n_tickers_in_group", "up_count", "down_count", "up_down_ratio",
                     "up_slope_median", "down_slope_median", "up_r2_median", "down_r2_median",
                     "up_share_r2_ge_0.8", "down_share_r2_ge_0.8"]
            gt = g[g["split"].isin(["search", "eval"])][[c for c in gcols if c in g.columns]]
            gt.to_csv(REPORTS_DIR / f"summary_by_{gby}_latest_run.csv", index=False)
            group_tab = gt[gt["window"] == w0]

    exp_id = run["experiment_id"].iloc[0]
    meta = json.loads((EXPERIMENTS_DIR / exp_id / "metadata.json").read_text())
    cols = ["window", "split", "label", "n_windows_total", "count", "share_of_windows", "n_tickers",
            "cum_return_median", "slope_median", "slope_annual_median", "r2_q25", "r2_q50", "r2_q75",
            "share_r2_ge_0.8", "resid_std_median", "max_abs_resid_median"]
    tab = run[[c for c in cols if c in run.columns]].copy()
    tab.to_csv(REPORTS_DIR / "summary_latest_run.csv", index=False)

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) < 100 else f"{v:,.0f}"
        return f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)

    lines = ["# S&P 500 线性趋势扫描 —— 实验报告（自动生成）", "",
             "> 本报告所有数字均由 `experiments/index.csv` 与实验目录自动读取生成。",
             "> 检测到的上涨/下跌区间是**事后（ex-post）描述**，使用了窗口内的未来价格，不是可执行的交易信号。", "",
             f"- 运行时间戳：`{run['run_timestamp'].iloc[0]}`",
             f"- 数据区间：{meta['data_range']['first_date']} ~ {meta['data_range']['last_date']}，"
             f"参与扫描股票数 {meta['data_range']['n_tickers_used']}",
             f"- 时间切分：随机块切分，块单位 `{meta['split']['block_unit']}`，seed={meta['split']['seed']}，"
             f"search 块 {meta['split']['n_search_blocks']} / 共 {meta['split']['n_blocks']}",
             f"- search 年份：{', '.join(meta['split']['search_blocks'])}",
             f"- eval 年份：{', '.join(meta['split']['eval_blocks'])}", ""]
    if extra:
        lines += ["## 运行信息", ""] + [f"- {k}：{v}" for k, v in extra.items()] + [""]
    lines += ["## 各参数组合汇总（search / eval / excluded）", "",
              "| " + " | ".join(tab.columns) + " |", "|" + "---|" * len(tab.columns)]
    for _, r in tab.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in r.to_list()) + " |")
    if group_tab is not None:
        lines += ["", f"## 分组统计（按 `{gby}`，window={int(group_tab['window'].iloc[0])}；全部窗口长度见 "
                  f"`reports/summary_by_{gby}_latest_run.csv`）", "",
                  "> 注意：各组股票数不同（样本量差异大）；分组为维基百科上的**当前** GICS 分类，历史上公司可能属于其他行业。", "",
                  "| " + " | ".join(group_tab.columns) + " |", "|" + "---|" * len(group_tab.columns)]
        for _, r in group_tab.iterrows():
            lines.append("| " + " | ".join(fmt(v) for v in r.to_list()) + " |")
    lines += ["", "## 图表", ""] + [f"![{c.name}]({c.name})" for c in charts] + [""]
    out = REPORTS_DIR / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
