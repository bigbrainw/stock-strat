"""Canonical symbols and default backtest parameters."""

from pathlib import Path

# Hon Hai
SYMBOL_TWSE = "2317"
SYMBOL_YAHOO = "2317.TW"

# TSMC 台積電
SYMBOL_TSMC = "2330"
SYMBOL_TSMC_YAHOO = "2330.TW"

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CACHE_DIR = PROJECT_ROOT / "data" / "cache"

DEFAULT_START = "2015-01-01"
DEFAULT_END = "2026-03-26"

# First backtest milestone
INITIAL_CAPITAL = 100_000.0

# --- Taiwan cash equity (台股現股), typical retail costs (verify on your broker’s page) ---
# Brokerage (手續費): statutory max 0.1425% of trade amount per side (買賣各一次).
# Discounts are common (e.g. 6折、3折、2折); 永豐「大戶投」等電子下單常見 2 折 → 0.1425% × 0.2.
# Ref: broker fee schedule (e.g. https://www.sinotrade.com.tw/newweb/Fee_Rate ) — rates change; confirm live.
TW_BROKERAGE_RATE_CAP = 0.001425  # 0.1425%
TW_COMMISSION_SINOPAC_DIGITAL_2折 = TW_BROKERAGE_RATE_CAP * 0.2  # ≈ 0.0285% per side

# Securities transaction tax (證券交易稅): on *sell* amount only for stocks; general rate 0.3% (現股).
# Day-trade (當沖) may use reduced rate (e.g. 0.15%) — not modeled unless you pass --sell-tax.
# Ref: MOF / eTax summaries — confirm current law.
TW_SECURITIES_TRANSACTION_TAX_STOCK = 0.003  # 0.3%

# Defaults for `run_rsi_backtest` when using commission + sell tax (not legacy single fee)
DEFAULT_COMMISSION_PCT = TW_COMMISSION_SINOPAC_DIGITAL_2折
DEFAULT_SELL_TAX_PCT = TW_SECURITIES_TRANSACTION_TAX_STOCK

# Legacy: single rate on buy cash and sell gross (no separate 證交稅); kept for stress/sweeps
FEE_PCT_PER_TRADE = 0.001  # 0.1% — educational; walkforward/stress default
RSI_PERIOD = 14
RSI_ENTRY = 30.0
RSI_EXIT = 50.0

FINMIND_BASE = "https://api.finmindtrade.com/api/v4/data"
