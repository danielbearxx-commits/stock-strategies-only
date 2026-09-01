import os
import json

import gspread
from google.oauth2.service_account import Credentials


def get_gsheet():
    creds_json = os.environ["GOOGLE_CREDS_JSON"]
    creds_dict = json.loads(creds_json)
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc.open_by_key(os.environ["GOOGLE_SHEET_ID"])


def read_watchlist() -> list[dict]:
    """從 Google Sheet Watchlist 分頁讀股票清單"""
    sh = get_gsheet()
    ws = sh.worksheet("Watchlist")
    rows = ws.get_all_records()
    enabled = [
        r for r in rows
        if str(r.get("enabled", "")).upper() in ("TRUE", "1", "YES")
    ]
    return enabled


def append_signals(signals: list[dict]):
    """把結果寫回 Signals 分頁"""
    if not signals:
        return
    sh = get_gsheet()
    try:
        ws = sh.worksheet("Signals")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Signals", rows=1000, cols=20)
        ws.append_row([
            "date", "stock_id", "name", "action", "signal_score",
            "entry_price", "stop_loss_price", "target_price",
            "rr_ratio", "position_pct", "winrate", "samples",
            "tech_signals", "risk_notes"
        ])

    rows = []
    for s in signals:
        c = s.get("components", {})
        rows.append([
            s.get("date", ""),
            s.get("stock_id", ""),
            s.get("name", ""),
            s.get("action", ""),
            s.get("signal_score", ""),
            s.get("entry_price", ""),
            s.get("stop_loss_price", ""),
            s.get("target_price", ""),
            s.get("risk_reward_ratio", ""),
            s.get("position_size_pct", ""),
            c.get("backtest_winrate", ""),
            c.get("backtest_samples", ""),
            ", ".join(c.get("tech_signals", [])),
            " / ".join(s.get("risk_notes", [])),
        ])
    ws.append_rows(rows)


RESEARCH_HEADERS = [
    "date", "stock_id", "name", "category", "price",
    "short_score", "short_recommendation", "short_reasons",
    "long_score", "long_recommendation", "long_reasons",
    "ret5", "ret20", "ret60", "relative20", "vol_ratio",
    "above_ma20", "above_ma60", "fundamental_score", "eps", "roe",
    "category_score", "news_score", "news_label", "news_title",
    "news_link", "news_published", "risk_notes",
]


def append_research(records: list[dict]):
    """把獨立研究版結果寫入 Research 分頁。每次執行追加一批紀錄。"""
    if not records:
        return
    sh = get_gsheet()
    try:
        ws = sh.worksheet("Research")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Research", rows=2000, cols=len(RESEARCH_HEADERS))
        ws.append_row(RESEARCH_HEADERS)

    rows = []
    for r in records:
        rows.append([r.get(header, "") for header in RESEARCH_HEADERS])
    ws.append_rows(rows)


def _ensure_watchlist_headers(ws) -> list[str]:
    """讀第一列 headers，沒 headers 就建好 stock_id/name/enabled 三欄。"""
    values = ws.get_all_values()
    if not values:
        headers = ["stock_id", "name", "enabled"]
        ws.append_row(headers)
        return headers
    headers = [h.strip() for h in values[0]]
    if "stock_id" not in headers or "enabled" not in headers:
        # 既有 sheet 有資料但 schema 不符，謹慎處理 — 不擅自改 headers
        return headers
    return headers


def add_to_watchlist(stock_id: str, name: str = "") -> dict:
    """加一檔到 Watchlist 分頁。

    若該 stock_id 已存在但 enabled=FALSE → 直接改回 TRUE（重啟用）
    若已存在且 enabled=TRUE → 不重複加，回傳 status='exists'
    若不存在 → append 新 row（enabled=TRUE）
    """
    sh = get_gsheet()
    ws = sh.worksheet("Watchlist")
    headers = _ensure_watchlist_headers(ws)

    sid_col = headers.index("stock_id") + 1  # gspread 是 1-based
    name_col = headers.index("name") + 1 if "name" in headers else None
    en_col = headers.index("enabled") + 1

    rows = ws.get_all_records()
    for i, r in enumerate(rows, start=2):  # row 1 是 header
        if str(r.get("stock_id", "")).strip() == str(stock_id).strip():
            current = str(r.get("enabled", "")).upper()
            if current in ("TRUE", "1", "YES"):
                return {
                    "status": "exists",
                    "stock_id": stock_id,
                    "name": r.get("name", name),
                }
            ws.update_cell(i, en_col, "TRUE")
            return {
                "status": "reenabled",
                "stock_id": stock_id,
                "name": r.get("name", name),
            }

    # 不存在 → append
    new_row = [""] * len(headers)
    new_row[sid_col - 1] = str(stock_id)
    if name_col is not None:
        new_row[name_col - 1] = name
    new_row[en_col - 1] = "TRUE"
    ws.append_row(new_row)
    return {"status": "added", "stock_id": stock_id, "name": name}


def remove_from_watchlist(stock_id: str) -> dict:
    """把 Watchlist 該 stock_id 的 enabled 改成 FALSE（軟刪除，保留歷史）"""
    sh = get_gsheet()
    ws = sh.worksheet("Watchlist")
    headers = _ensure_watchlist_headers(ws)
    if "enabled" not in headers:
        return {"status": "no_enabled_column"}
    en_col = headers.index("enabled") + 1

    rows = ws.get_all_records()
    for i, r in enumerate(rows, start=2):
        if str(r.get("stock_id", "")).strip() == str(stock_id).strip():
            ws.update_cell(i, en_col, "FALSE")
            return {"status": "disabled", "stock_id": stock_id}
    return {"status": "not_found", "stock_id": stock_id}


def read_latest_signals(limit: int = 50) -> list[dict]:
    """從 Signals 分頁讀最近 N 筆紀錄（依 row 順序，最後 N 筆）。

    若該分頁不存在 → 回空 list（代表還沒跑過）
    """
    sh = get_gsheet()
    try:
        ws = sh.worksheet("Signals")
    except gspread.WorksheetNotFound:
        return []
    rows = ws.get_all_records()
    if not rows:
        return []
    return rows[-limit:][::-1]  # 最新的在最前面


PERFORMANCE_HEADERS = [
    "signal_date", "stock_id", "name", "entry_close", "entry_open",
    "t5_date", "t5_close", "t5_ret",
    "t10_date", "t10_close", "t10_ret",
    "t20_date", "t20_close", "t20_ret",
    "hit_target", "hit_stop", "status",
]


def read_performance() -> list[dict]:
    """讀取 Performance 分頁的所有追蹤紀錄（若尚未建立則回空 list）"""
    sh = get_gsheet()
    try:
        ws = sh.worksheet("Performance")
    except gspread.WorksheetNotFound:
        return []
    return ws.get_all_records()


def write_performance(records: list[dict]):
    """整張 Performance 分頁清空重寫（紀錄數不多，效率 OK）"""
    sh = get_gsheet()
    try:
        ws = sh.worksheet("Performance")
        ws.clear()
    except gspread.WorksheetNotFound:
        rows_alloc = max(2000, len(records) + 100)
        ws = sh.add_worksheet(
            title="Performance", rows=rows_alloc, cols=len(PERFORMANCE_HEADERS)
        )

    ws.append_row(PERFORMANCE_HEADERS)
    if not records:
        return

    rows = [[r.get(h, "") for h in PERFORMANCE_HEADERS] for r in records]
    ws.append_rows(rows)
