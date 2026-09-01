"""獨立研究版選股引擎。

這個模組刻意不修改既有 V3.2/V3.3 訊號。它用另一組透明、可解釋的
市場與個股指標產生「短線研究分」與「中長線研究分」，並把新聞只當作
低權重的輔助訊號。新聞抓不到時會回中性，不會把缺資料當成利空。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

from .data import get_fundamental, get_price_history
from .datasources import get_index_history


POSITIVE_WORDS = (
    "營收創高", "營收成長", "獲利成長", "上調", "新高", "接單", "擴產",
    "法人大買", "買超", "看好", "達標", "訂單", "需求增", "成長", "利多",
    "上修", "強勁", "good", "upgrade", "beat",
)
NEGATIVE_WORDS = (
    "營收下滑", "獲利下滑", "下修", "虧損", "衰退", "減產", "砍單",
    "法人賣超", "賣超", "利空", "訴訟", "延後", "風險", "警示",
    "下調", "疲弱", "跌停", "bad", "downgrade", "miss",
)


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return float(max(low, min(high, value)))


def _scale(value: float | None, low: float, high: float, neutral: float = 50.0) -> float:
    if value is None or not np.isfinite(value) or high <= low:
        return neutral
    return _clip((float(value) - low) / (high - low) * 100.0)


def _as_float(value, default: float | None = None) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _return(close: pd.Series, periods: int) -> float | None:
    if len(close) <= periods:
        return None
    base = _as_float(close.iloc[-periods - 1])
    latest = _as_float(close.iloc[-1])
    if base in (None, 0) or latest is None:
        return None
    return latest / base - 1.0


def _parse_news_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def fetch_news(stock_id: str, name: str, days: int = 7) -> dict:
    """抓 Google News RSS 的最近標題；失敗時回傳中性結果。"""
    query = quote_plus(f'"{stock_id}" {name} 台股')
    url = (
        "https://news.google.com/rss/search?q=" + query
        + "&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    )
    neutral = {
        "score": 50.0,
        "label": "無法判讀",
        "title": "",
        "link": "",
        "published": "",
        "status": "unavailable",
    }
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "stock-strategies-research/1.0"},
            timeout=8,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except (requests.RequestException, ET.ParseError, ValueError):
        return neutral

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    items = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        date_text = (item.findtext("pubDate") or "").strip()
        published = _parse_news_date(date_text)
        if published is not None and published < cutoff:
            continue
        if title:
            items.append({"title": title, "link": link, "published": date_text})

    if not items:
        neutral["status"] = "no_recent_news"
        return neutral

    title = items[0]["title"]
    lower_title = title.lower()
    positive = sum(word.lower() in lower_title for word in POSITIVE_WORDS)
    negative = sum(word.lower() in lower_title for word in NEGATIVE_WORDS)
    score = _clip(50.0 + (positive - negative) * 12.0)
    label = "利多傾向" if score >= 62 else "利空傾向" if score <= 38 else "中性"
    return {
        "score": round(score, 1),
        "label": label,
        "title": title,
        "link": items[0]["link"],
        "published": items[0]["published"],
        "status": "ok",
    }


def market_snapshot(index: pd.DataFrame) -> dict:
    """由加權指數近月線、季線與報酬率判斷市場環境。"""
    if index is None or index.empty or "close" not in index.columns:
        return {
            "regime": "資料不足", "score": 50.0, "ret20": None,
            "note": "⚪ 加權指數資料不足，市場分數採中性",
        }
    close = pd.to_numeric(index["close"], errors="coerce").dropna().reset_index(drop=True)
    if len(close) < 60:
        return {
            "regime": "資料不足", "score": 50.0, "ret20": _return(close, 20),
            "note": "⚪ 加權指數資料不足，市場分數採中性",
        }
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1])
    latest = float(close.iloc[-1])
    ret20 = _return(close, 20)
    if latest > ma20 > ma60:
        regime, score, note = "偏多", 75.0, "🟢 加權指數站上月線與季線"
    elif latest > ma20:
        regime, score, note = "中性偏多", 62.0, "🟡 加權指數站上月線但季線尚未完全轉強"
    elif latest < ma20 < ma60:
        regime, score, note = "偏空", 30.0, "🔴 加權指數跌破月線與季線"
    else:
        regime, score, note = "中性偏空", 42.0, "🟠 加權指數跌破月線，方向仍分歧"
    return {
        "regime": regime, "score": score, "ret20": ret20,
        "latest": latest, "ma20": ma20, "ma60": ma60,
        "note": note,
    }


def _fundamental_snapshot(stock_id: str) -> dict:
    try:
        raw = get_fundamental(stock_id)
    except Exception:
        raw = {"eps": {}, "roe": {}}
    eps_map = raw.get("eps", {}) or {}
    roe_map = raw.get("roe", {}) or {}
    eps = _as_float(eps_map[max(eps_map)]) if eps_map else None
    roe = _as_float(roe_map[max(roe_map)]) if roe_map else None
    eps_score = 75.0 if eps is not None and eps >= 5 else 60.0 if eps is not None and eps > 0 else 40.0 if eps is not None else 50.0
    roe_score = 75.0 if roe is not None and roe >= 15 else 60.0 if roe is not None and roe > 8 else 40.0 if roe is not None else 50.0
    return {
        "eps": round(eps, 2) if eps is not None else None,
        "roe": round(roe, 2) if roe is not None else None,
        "score": round((eps_score + roe_score) / 2, 1),
    }


def analyze_stock(
    stock_id: str,
    name: str,
    category: str,
    price: pd.DataFrame,
    market: dict,
    category_score: float,
    news: dict,
    fundamentals: dict,
) -> dict:
    """分析單檔股票，回傳可直接寫入 Sheet/Telegram 的純 dict。"""
    if price is None or price.empty or "close" not in price.columns:
        return {"stock_id": stock_id, "name": name, "category": category, "status": "資料不足"}
    close = pd.to_numeric(price["close"], errors="coerce").dropna().reset_index(drop=True)
    if len(close) < 60:
        return {"stock_id": stock_id, "name": name, "category": category, "status": "資料不足"}

    ret5, ret20, ret60 = _return(close, 5), _return(close, 20), _return(close, 60)
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1])
    latest = float(close.iloc[-1])
    above20 = latest > ma20
    above60 = latest > ma60

    volume = pd.to_numeric(price.get("volume", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(volume) >= 25:
        vol_ratio = float(volume.iloc[-5:].mean() / max(volume.iloc[-25:-5].mean(), 1))
    else:
        vol_ratio = 1.0
    volatility = float(close.pct_change().tail(20).std()) if len(close) >= 21 else 0.05
    relative20 = ret20 - market["ret20"] if ret20 is not None and market.get("ret20") is not None else None

    momentum = _scale(ret5, -0.15, 0.20) * 0.35 + _scale(ret20, -0.25, 0.35) * 0.65
    trend = (75.0 if above20 and above60 else 62.0 if above20 else 38.0 if above60 else 25.0)
    volume_score = _scale(vol_ratio, 0.6, 2.0)
    relative_score = _scale(relative20, -0.15, 0.20)
    stability = _scale(-volatility, -0.12, -0.015)

    short_score = (
        market["score"] * 0.10 + momentum * 0.25 + trend * 0.20
        + volume_score * 0.15 + relative_score * 0.15
        + category_score * 0.10 + news["score"] * 0.05
    )
    long_score = (
        market["score"] * 0.10 + _scale(ret60, -0.35, 0.80) * 0.15
        + trend * 0.20 + stability * 0.10 + relative_score * 0.10
        + category_score * 0.10 + fundamentals["score"] * 0.20
        + news["score"] * 0.05
    )

    def recommendation(score: float) -> str:
        return "優先研究" if score >= 70 else "觀察" if score >= 55 else "暫不列入"

    short_reasons = []
    if ret20 is not None and ret20 > 0.05:
        short_reasons.append(f"20日動能{ret20 * 100:+.1f}%")
    if above20 and above60:
        short_reasons.append("站上月線與季線")
    if vol_ratio >= 1.2:
        short_reasons.append(f"量能{vol_ratio:.1f}倍")
    if relative20 is not None and relative20 > 0.03:
        short_reasons.append("強於大盤")
    if news["label"] != "無法判讀" and news["label"] != "中性":
        short_reasons.append(f"新聞{news['label']}")
    long_reasons = []
    if ret60 is not None and ret60 > 0.10:
        long_reasons.append(f"60日趨勢{ret60 * 100:+.1f}%")
    if fundamentals["eps"] is not None and fundamentals["eps"] > 0:
        long_reasons.append(f"最新年度EPS {fundamentals['eps']:.1f}")
    if fundamentals["roe"] is not None and fundamentals["roe"] > 8:
        long_reasons.append(f"ROE {fundamentals['roe']:.1f}%")
    if category_score >= 60:
        long_reasons.append("類股相對強")

    risks = []
    if market["score"] < 45:
        risks.append("大盤偏弱")
    if not above20:
        risks.append("股價跌破月線")
    if volatility > 0.06:
        risks.append("近20日波動偏大")
    if news["label"] == "利空傾向":
        risks.append("新聞標題偏利空，需人工確認")

    return {
        "stock_id": stock_id,
        "name": name,
        "category": category,
        "status": "ok",
        "price": round(latest, 2),
        "ret5": round(ret5 * 100, 2) if ret5 is not None else None,
        "ret20": round(ret20 * 100, 2) if ret20 is not None else None,
        "ret60": round(ret60 * 100, 2) if ret60 is not None else None,
        "above_ma20": above20,
        "above_ma60": above60,
        "vol_ratio": round(vol_ratio, 2),
        "relative20": round(relative20 * 100, 2) if relative20 is not None else None,
        "fundamental_score": round(fundamentals["score"], 1),
        "eps": fundamentals["eps"],
        "roe": fundamentals["roe"],
        "category_score": round(category_score, 1),
        "news_score": round(news["score"], 1),
        "news_label": news["label"],
        "news_title": news["title"],
        "news_link": news["link"],
        "news_published": news["published"],
        "short_score": round(_clip(short_score), 1),
        "long_score": round(_clip(long_score), 1),
        "short_recommendation": recommendation(short_score),
        "long_recommendation": recommendation(long_score),
        "short_reasons": "、".join(short_reasons) or "動能訊號不明顯",
        "long_reasons": "、".join(long_reasons) or "中長線條件尚未集中",
        "risk_notes": "、".join(risks) or "目前無明顯警示",
    }


def run_research(watchlist: list[dict]) -> tuple[dict, list[dict]]:
    """執行完整研究版掃描；回傳 market 與結果。"""
    index = get_index_history("TAIEX", start=(datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d"))
    market = market_snapshot(index)
    prepared = []
    for row in watchlist:
        sid = str(row.get("stock_id", "")).strip()
        name = str(row.get("name", "")).strip()
        category = str(row.get("category", "其他")).strip() or "其他"
        try:
            price = get_price_history(sid, years=1)
        except Exception:
            price = pd.DataFrame()
        ret20 = None
        if not price.empty and "close" in price.columns:
            close = pd.to_numeric(price["close"], errors="coerce").dropna()
            ret20 = _return(close, 20)
        prepared.append({"row": row, "sid": sid, "name": name, "category": category,
                         "price": price, "ret20": ret20})

    by_category: dict[str, list[float]] = {}
    for item in prepared:
        if item["ret20"] is not None:
            by_category.setdefault(item["category"], []).append(item["ret20"])
    category_scores = {
        category: _scale(float(np.mean(values)), -0.20, 0.30)
        for category, values in by_category.items()
    }

    results = []
    for item in prepared:
        news = fetch_news(item["sid"], item["name"])
        fundamentals = _fundamental_snapshot(item["sid"])
        results.append(analyze_stock(
            item["sid"], item["name"], item["category"], item["price"], market,
            category_scores.get(item["category"], 50.0), news, fundamentals,
        ))
    results.sort(key=lambda r: (-(r.get("short_score") or 0), -(r.get("long_score") or 0)))
    return market, results


