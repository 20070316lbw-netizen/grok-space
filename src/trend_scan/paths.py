"""项目路径常量。

ROOT 默认为仓库根目录（src/trend_scan/paths.py 向上两级；uv sync 以可编辑模式安装，路径不变）。
可用环境变量 SP500_TREND_ROOT 覆盖（例如非可编辑安装时）。
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("SP500_TREND_ROOT") or Path(__file__).resolve().parents[2])
CONFIGS_DIR = ROOT / "configs"
DEFAULT_CONFIG = CONFIGS_DIR / "trend_scan" / "default.yaml"
PAPER_OUTPUTS_DIR = ROOT / "outputs" / "repro_bocharov"
DATA_RAW = ROOT / "data" / "raw"
UNIVERSE_DIR = DATA_RAW / "universe"
PRICES_DIR = DATA_RAW / "prices"
PROCESSED_DIR = ROOT / "data" / "processed"
EXPERIMENTS_DIR = ROOT / "experiments"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

for _p in (UNIVERSE_DIR, PRICES_DIR, PROCESSED_DIR, EXPERIMENTS_DIR, REPORTS_DIR, LOGS_DIR):
    _p.mkdir(parents=True, exist_ok=True)
