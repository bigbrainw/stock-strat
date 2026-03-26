"""Canonical symbols and default backtest parameters (2317 only)."""

from pathlib import Path

SYMBOL_TWSE = "2317"
SYMBOL_YAHOO = "2317.TW"

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CACHE_DIR = PROJECT_ROOT / "data" / "cache"

DEFAULT_START = "2015-01-01"
DEFAULT_END = "2026-03-26"

# First backtest milestone
INITIAL_CAPITAL = 100_000.0
FEE_PCT_PER_TRADE = 0.001  # 0.1% on traded notional (entry + exit each pay fee in simulation)
RSI_PERIOD = 14
RSI_ENTRY = 30.0
RSI_EXIT = 50.0

FINMIND_BASE = "https://api.finmindtrade.com/api/v4/data"
