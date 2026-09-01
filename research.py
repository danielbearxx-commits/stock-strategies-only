"""每日獨立研究版選股報告。

執行：uv run python research.py

這支腳本與 main.py 分開，會讀同一張 Watchlist，另外計算市場／類股／
量價／基本面／新聞的研究分，不會改寫既有 Signals 或改變 BUY/WATCH/SKIP。
"""
import os
import sys
import time
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from stock_strategies.research import run_research
from stock_strategies.research_notify import format_research_messages
from stock_strategies.sheet import append_research, read_watchlist
from stock_strategies.notify import send_telegram


REQUIRED_ENV = [
    "FINMIND_TOKEN",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "GOOGLE_SHEET_ID",
    "GOOGLE_CREDS_JSON",
]


def main():
    missing = [key for key in REQUIRED_ENV if not os.environ.get(key)]
    if missing:
        print(f"❌ 缺少環境變數: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"[{datetime.now()}] 讀取 watchlist...")
    watchlist = read_watchlist()
    print(f"  → {len(watchlist)} 檔啟用中")
    print("執行獨立研究分析（市場、類股、量價、基本面、新聞）...")

    market, results = run_research(watchlist)
    today = datetime.now().strftime("%Y-%m-%d")
    for result in results:
        result["date"] = today

    valid = [r for r in results if r.get("status") == "ok"]
    print(f"  → 可分析 {len(valid)}/{len(results)} 檔")
    if valid:
        top = max(valid, key=lambda r: r.get("short_score", 0))
        print(
            f"  → 短線最高: {top['stock_id']} {top['name']} "
            f"{top.get('short_score', 0)} 分"
        )

    print("寫回 Google Sheet (Research)...")
    append_research(results)

    print("發送 Telegram 研究版報告...")
    for message in format_research_messages(market, results):
        send_telegram(message)
        time.sleep(0.5)

    print("✅ 研究版完成（不影響既有 main.py）")


if __name__ == "__main__":
    main()


