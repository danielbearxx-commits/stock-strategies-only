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
from .datasources import (
    get_index_history,
    get_institutional,
    get_margin,
    get_month_revenue,
    get_shareholding,
    get_valuation,
)


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

SEVERE_NEGATIVE_WORDS = (
    "虧損", "砍單", "停工", "舞弊", "下修", "訴訟", "減產", "警示", "跌停",
)

# 先用便宜的量價資料掃描全清單，再把各週期前幾名交給慢速資料源。
# 這是為了避免免費 API 限流，也避免 GitHub Actions 超過 20 分鐘。
DEEP_RESEARCH_LIMIT = 5


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


def _keyword_hits(text: str, words: tuple[str, ...]) -> list[str]:
    """找關鍵字並避免短字重複計入長字（例如「成長」/「營收成長」）。"""
    lower = text.lower()
    hits: list[tuple[int, int, str]] = []
    for word in sorted(words, key=len, reverse=True):
        needle = word.lower()
        start = 0
        while True:
            found = lower.find(needle, start)
            if found < 0:
                break
            end = found + len(needle)
            if not any(found < other_end and end > other_start
                       for other_start, other_end, _ in hits):
                hits.append((found, end, word))
            start = end
    return [word for _, _, word in sorted(hits)]


def _news_item_score(title: str) -> tuple[float, list[str], list[str]]:
    positive = _keyword_hits(title, POSITIVE_WORDS)
    negative = _keyword_hits(title, NEGATIVE_WORDS)
    score = _clip(50.0 + (len(positive) - len(negative)) * 15.0)
    return score, positive, negative


def fetch_news(stock_id: str, name: str, days: int = 7) -> dict:
    """抓 Google News RSS 最近多則標題，做具時間衰減的輔助判讀。

    不讀全文、不把缺新聞當利空；只把新聞當總分低權重因子，並保留標題與來源
    讓使用者人工確認。"""
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
        "news_count": 0,
        "news_24h_count": 0,
        "news_sources": "",
        "news_trend": "無法判讀",
        "news_positive_hits": 0,
        "news_negative_hits": 0,
        "news_risk_flags": "",
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
        source = (item.findtext("source") or "").strip()
        published = _parse_news_date(date_text)
        if published is not None and published < cutoff:
            continue
        if title:
            items.append({
                "title": title, "link": link, "published": date_text,
                "parsed": published, "source": source,
            })

    if not items:
        neutral["status"] = "no_recent_news"
        return neutral

    now = datetime.now(timezone.utc)
    items.sort(key=lambda item: item["parsed"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    scored = []
    for item in items[:10]:
        item_score, positive, negative = _news_item_score(item["title"])
        age_hours = ((now - item["parsed"]).total_seconds() / 3600
                     if item["parsed"] else 72.0)
        weight = 1.5 if age_hours <= 24 else 1.0 if age_hours <= 72 else 0.6
        scored.append((item, item_score, weight, positive, negative))

    score = _clip(sum(item_score * weight for _, item_score, weight, _, _ in scored)
                  / max(sum(weight for _, _, weight, _, _ in scored), 1.0))
    label = "利多傾向" if score >= 62 else "利空傾向" if score <= 38 else "中性"
    latest = scored[0][0]
    positive_hits = [word for _, _, _, positive, _ in scored for word in positive]
    negative_hits = [word for _, _, _, _, negative in scored for word in negative]
    recent_scores = [item_score for item, item_score, _, _, _ in scored
                     if item["parsed"] and (now - item["parsed"]).total_seconds() <= 72 * 3600]
    older_scores = [item_score for item, item_score, _, _, _ in scored
                   if not item["parsed"] or (now - item["parsed"]).total_seconds() > 72 * 3600]
    if recent_scores and older_scores:
        delta = float(np.mean(recent_scores) - np.mean(older_scores))
        trend = "改善" if delta >= 8 else "轉弱" if delta <= -8 else "持平"
    elif len(scored) >= 2:
        trend = "近期有新訊"
    else:
        trend = "單一新聞"
    sources = []
    for item, *_ in scored:
        if item["source"] and item["source"] not in sources:
            sources.append(item["source"])
    return {
        "score": round(score, 1),
        "label": label,
        "title": latest["title"],
        "link": latest["link"],
        "published": latest["published"],
        "status": "ok",
        "news_count": len(scored),
        "news_24h_count": sum(
            1 for item, *_ in scored
            if item["parsed"] and (now - item["parsed"]).total_seconds() <= 24 * 3600
        ),
        "news_sources": "、".join(sources[:5]),
        "news_trend": trend,
        "news_positive_hits": len(positive_hits),
        "news_negative_hits": len(negative_hits),
        "news_risk_flags": "、".join(dict.fromkeys(
            word for word in negative_hits if word in SEVERE_NEGATIVE_WORDS
        )),
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
        "available": eps is not None or roe is not None,
    }


def _empty_advanced() -> dict:
    return {
        "flow_score": 50.0, "flow_consistency": None, "foreign_flow_score": 50.0,
        "revenue_score": 50.0, "revenue_yoy": None, "revenue_mom": None,
        "valuation_score": 50.0, "per": None, "pbr": None,
        "margin_score": 50.0, "margin_ratio": None, "margin_change_pct": None,
        "ownership_score": 50.0, "foreign_ratio": None,
        "available": 0,
    }


def _latest_numeric(df: pd.DataFrame, column: str) -> float | None:
    if df is None or df.empty or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return _as_float(values.iloc[-1]) if len(values) else None


def _advanced_snapshot(stock_id: str, days: int = 180) -> dict:
    """讀取較慢更新的基本面／籌碼因子；任何單一資料源失敗都回中性。"""
    result = _empty_advanced()
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        institutional = get_institutional(stock_id, start)
    except Exception:
        institutional = pd.DataFrame()
    if not institutional.empty and "total_net" in institutional.columns:
        net = pd.to_numeric(institutional["total_net"], errors="coerce").dropna()
        if len(net):
            sign_consistency = float(np.sign(net.tail(20)).mean())
            result["flow_consistency"] = round(sign_consistency, 3)
            result["flow_score"] = round(_scale(sign_consistency, -1, 1), 1)
            if "foreign_net" in institutional.columns:
                foreign = pd.to_numeric(institutional["foreign_net"], errors="coerce").dropna()
                if len(foreign):
                    result["foreign_flow_score"] = round(
                        _scale(float(np.sign(foreign.tail(20)).mean()), -1, 1), 1
                    )
            result["available"] += 1

    try:
        revenue = get_month_revenue(stock_id, start)
    except Exception:
        revenue = pd.DataFrame()
    if not revenue.empty:
        yoy = _latest_numeric(revenue, "yoy")
        mom = _latest_numeric(revenue, "mom")
        result["revenue_yoy"] = round(yoy * 100, 2) if yoy is not None else None
        result["revenue_mom"] = round(mom * 100, 2) if mom is not None else None
        yoy_score = _scale(yoy, -0.20, 0.40)
        mom_score = _scale(mom, -0.20, 0.20)
        result["revenue_score"] = round(yoy_score * 0.7 + mom_score * 0.3, 1)
        result["available"] += 1

    try:
        valuation = get_valuation(stock_id, start)
    except Exception:
        valuation = pd.DataFrame()
    if not valuation.empty:
        per = _latest_numeric(valuation, "per")
        pbr = _latest_numeric(valuation, "pbr")
        result["per"] = round(per, 2) if per is not None else None
        result["pbr"] = round(pbr, 2) if pbr is not None else None
        parts = []
        if per is not None and per > 0:
            parts.append(100 - _scale(per, 8, 60))
        if pbr is not None and pbr > 0:
            parts.append(100 - _scale(pbr, 0.8, 8))
        if parts:
            result["valuation_score"] = round(float(np.mean(parts)), 1)
            result["available"] += 1

    try:
        margin = get_margin(stock_id, start)
    except Exception:
        margin = pd.DataFrame()
    if not margin.empty:
        ratio = _latest_numeric(margin, "short_margin_ratio")
        balance = pd.to_numeric(margin.get("margin_balance", pd.Series(dtype=float)), errors="coerce").dropna()
        result["margin_ratio"] = round(ratio, 4) if ratio is not None else None
        if len(balance) >= 20 and balance.iloc[-20] > 0:
            result["margin_change_pct"] = round((balance.iloc[-1] / balance.iloc[-20] - 1) * 100, 2)
        if ratio is not None:
            result["margin_score"] = round(_scale(-ratio, -1.0, -0.02), 1)
            result["available"] += 1

    try:
        ownership = get_shareholding(stock_id, start)
    except Exception:
        ownership = pd.DataFrame()
    if not ownership.empty:
        foreign_ratio = _latest_numeric(ownership, "foreign_ratio")
        result["foreign_ratio"] = round(foreign_ratio, 2) if foreign_ratio is not None else None
        ratios = pd.to_numeric(ownership.get("foreign_ratio", pd.Series(dtype=float)), errors="coerce").dropna()
        if len(ratios) >= 2:
            result["ownership_score"] = round(_scale(float(ratios.iloc[-1] - ratios.iloc[-2]), -2, 2), 1)
            result["available"] += 1

    return result


def _technical_quality(close: pd.Series, price: pd.DataFrame) -> dict:
    """補充 RSI、ATR、突破與回撤，避免只看單一報酬率。"""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
    if pd.isna(gain) or pd.isna(loss):
        rsi = None
    elif loss == 0:
        rsi = 100.0
    else:
        rsi = float(100 - 100 / (1 + gain / loss))

    high = pd.to_numeric(price.get("high", close), errors="coerce")
    low = pd.to_numeric(price.get("low", close), errors="coerce")
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_pct = _as_float((true_range.rolling(14).mean().iloc[-1] / close.iloc[-1]))
    high60 = float(close.tail(60).max())
    latest = float(close.iloc[-1])
    breakout = bool(latest >= high60 * 0.995 and _return(close, 20) is not None and _return(close, 20) > 0)
    drawdown60 = latest / high60 - 1 if high60 else None
    if rsi is None:
        rsi_score = 50.0
    elif 50 <= rsi <= 70:
        rsi_score = 80.0
    elif 40 <= rsi <= 75:
        rsi_score = 65.0
    elif rsi > 80:
        rsi_score = 35.0
    else:
        rsi_score = 40.0
    breakout_score = 85.0 if breakout else 55.0 if latest >= high60 * 0.95 else 40.0
    return {
        "rsi14": round(rsi, 2) if rsi is not None else None,
        "atr_pct": round(atr_pct * 100, 2) if atr_pct is not None else None,
        "drawdown60": round(drawdown60 * 100, 2) if drawdown60 is not None else None,
        "breakout60": breakout,
        "technical_quality": round(rsi_score * 0.6 + breakout_score * 0.4, 1),
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
    advanced: dict | None = None,
    research_stage: str = "深入",
) -> dict:
    """分析單檔股票，回傳可直接寫入 Sheet/Telegram 的純 dict。"""
    if price is None or price.empty or "close" not in price.columns:
        return {
            "stock_id": stock_id, "name": name, "category": category,
            "status": "資料不足", "research_stage": research_stage,
        }
    close = pd.to_numeric(price["close"], errors="coerce").dropna().reset_index(drop=True)
    if len(close) < 60:
        return {
            "stock_id": stock_id, "name": name, "category": category,
            "status": "資料不足", "research_stage": research_stage,
        }

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
    technical = _technical_quality(close, price)
    advanced = advanced or _empty_advanced()

    momentum = _scale(ret5, -0.15, 0.20) * 0.35 + _scale(ret20, -0.25, 0.35) * 0.65
    trend = (75.0 if above20 and above60 else 62.0 if above20 else 38.0 if above60 else 25.0)
    volume_score = _scale(vol_ratio, 0.6, 2.0)
    relative_score = _scale(relative20, -0.15, 0.20)
    stability = _scale(-volatility, -0.12, -0.015)

    short_score = (
        market["score"] * 0.10 + momentum * 0.20 + trend * 0.15
        + volume_score * 0.10 + relative_score * 0.10
        + category_score * 0.10 + advanced["flow_score"] * 0.10
        + technical["technical_quality"] * 0.10 + news["score"] * 0.05
    )
    long_score = (
        market["score"] * 0.08 + _scale(ret60, -0.35, 0.80) * 0.12
        + trend * 0.15 + stability * 0.10 + relative_score * 0.08
        + category_score * 0.08 + fundamentals["score"] * 0.17
        + advanced["revenue_score"] * 0.10 + advanced["valuation_score"] * 0.05
        + advanced["flow_score"] * 0.05 + news["score"] * 0.02
    )
    mid_score = (
        market["score"] * 0.08 + _scale(ret20, -0.25, 0.45) * 0.15
        + _scale(ret60, -0.35, 0.80) * 0.15 + trend * 0.15
        + stability * 0.08 + relative_score * 0.10
        + category_score * 0.08 + fundamentals["score"] * 0.10
        + advanced["revenue_score"] * 0.07 + advanced["flow_score"] * 0.03
        + news["score"] * 0.01
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
    if technical["breakout60"]:
        short_reasons.append("接近60日突破")
    if technical["rsi14"] is not None and 50 <= technical["rsi14"] <= 70:
        short_reasons.append(f"RSI {technical['rsi14']:.0f}健康")
    if advanced["flow_consistency"] is not None and advanced["flow_consistency"] > 0.2:
        short_reasons.append("法人近20日偏買")
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
    if advanced["revenue_yoy"] is not None and advanced["revenue_yoy"] > 0:
        long_reasons.append(f"月營收年增{advanced['revenue_yoy']:+.1f}%")
    mid_reasons = []
    if ret20 is not None and ret20 > 0.03:
        mid_reasons.append(f"20日趨勢{ret20 * 100:+.1f}%")
    if ret60 is not None and ret60 > 0.08:
        mid_reasons.append(f"60日趨勢{ret60 * 100:+.1f}%")
    if above20 and above60:
        mid_reasons.append("月線與季線同向")
    if advanced["revenue_yoy"] is not None and advanced["revenue_yoy"] > 0:
        mid_reasons.append(f"營收年增{advanced['revenue_yoy']:+.1f}%")

    risks = []
    if market["score"] < 45:
        risks.append("大盤偏弱")
    if not above20:
        risks.append("股價跌破月線")
    if volatility > 0.06:
        risks.append("近20日波動偏大")
    if technical["rsi14"] is not None and technical["rsi14"] > 80:
        risks.append("RSI過熱")
    if technical["drawdown60"] is not None and technical["drawdown60"] < -15:
        risks.append("距60日高點回落超過15%")
    if advanced["margin_ratio"] is not None and advanced["margin_ratio"] > 0.8:
        risks.append("券資比偏高")
    if advanced["margin_change_pct"] is not None and advanced["margin_change_pct"] > 25:
        risks.append("融資餘額近月明顯增加")
    if news["label"] == "利空傾向":
        risks.append("新聞標題偏利空，需人工確認")

    available_sources = [
        True, fundamentals.get("available", False), advanced.get("available", 0) > 0,
        news.get("status") == "ok",
    ]
    confidence = round(sum(available_sources) / len(available_sources), 2)
    overall_score = round((short_score + long_score) / 2, 1)
    if short_score >= 70 and long_score >= 70:
        consensus = "短長線共識"
    elif short_score >= 70:
        consensus = "短線優先"
    elif long_score >= 70:
        consensus = "中長線優先"
    else:
        consensus = "等待確認"

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
        "rsi14": technical["rsi14"],
        "atr_pct": technical["atr_pct"],
        "drawdown60": technical["drawdown60"],
        "breakout60": technical["breakout60"],
        "technical_quality": technical["technical_quality"],
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
        "news_count": news.get("news_count", 0),
        "news_24h_count": news.get("news_24h_count", 0),
        "news_sources": news.get("news_sources", ""),
        "news_trend": news.get("news_trend", "無法判讀"),
        "news_positive_hits": news.get("news_positive_hits", 0),
        "news_negative_hits": news.get("news_negative_hits", 0),
        "news_risk_flags": news.get("news_risk_flags", ""),
        "flow_score": advanced["flow_score"],
        "flow_consistency": advanced["flow_consistency"],
        "foreign_flow_score": advanced["foreign_flow_score"],
        "revenue_score": advanced["revenue_score"],
        "revenue_yoy": advanced["revenue_yoy"],
        "revenue_mom": advanced["revenue_mom"],
        "valuation_score": advanced["valuation_score"],
        "per": advanced["per"],
        "pbr": advanced["pbr"],
        "margin_score": advanced["margin_score"],
        "margin_ratio": advanced["margin_ratio"],
        "margin_change_pct": advanced["margin_change_pct"],
        "ownership_score": advanced["ownership_score"],
        "foreign_ratio": advanced["foreign_ratio"],
        "short_score": round(_clip(short_score), 1),
        "long_score": round(_clip(long_score), 1),
        "mid_score": round(_clip(mid_score), 1),
        "short_recommendation": recommendation(short_score),
        "mid_recommendation": recommendation(mid_score),
        "long_recommendation": recommendation(long_score),
        "mid_reasons": "、".join(mid_reasons) or "中期趨勢尚未集中",
        "overall_score": overall_score,
        "consensus": consensus,
        "data_confidence": confidence,
        "data_quality": "完整" if confidence >= 0.75 else "部分資料" if confidence >= 0.5 else "資料偏少",
        "research_stage": research_stage,
        "short_reasons": "、".join(short_reasons) or "動能訊號不明顯",
        "long_reasons": "、".join(long_reasons) or "中長線條件尚未集中",
        "risk_notes": "、".join(risks) or "目前無明顯警示",
    }


def run_research(watchlist: list[dict]) -> tuple[dict, list[dict]]:
    """先初篩全清單，再深入研究各週期前幾名。

    初篩只使用市場、量價、類股相對強弱，避免對全部股票呼叫慢速資料源。
    短／中／長各取前五名後合併，最多深入 15 檔；其餘股票仍會保留初篩排名。
    """
    index = get_index_history("TAIEX", start=(datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d"))
    market = market_snapshot(index)
    prepared = []
    for index_no, row in enumerate(watchlist, 1):
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
        print(f"  → 量價初篩 {index_no}/{len(watchlist)} {sid} {name}", flush=True)

    by_category: dict[str, list[float]] = {}
    for item in prepared:
        if item["ret20"] is not None:
            by_category.setdefault(item["category"], []).append(item["ret20"])
    category_scores = {
        category: _scale(float(np.mean(values)), -0.20, 0.30)
        for category, values in by_category.items()
    }

    # 用 watchlist 自身做市場廣度補充：不只看加權指數，也看候選股有多少站上月線。
    valid_prices = [item["price"] for item in prepared if not item["price"].empty]
    if valid_prices:
        above20_flags = []
        ret20_values = []
        for price in valid_prices:
            close = pd.to_numeric(price.get("close", pd.Series(dtype=float)), errors="coerce").dropna()
            if len(close) >= 20:
                above20_flags.append(float(close.iloc[-1] > close.rolling(20).mean().iloc[-1]))
                value = _return(close, 20)
                if value is not None:
                    ret20_values.append(value)
        if above20_flags:
            breadth_pct = float(np.mean(above20_flags) * 100)
            market["breadth_pct"] = round(breadth_pct, 1)
            market["breadth_score"] = round(breadth_pct, 1)
            market["score"] = round(market["score"] * 0.8 + breadth_pct * 0.2, 1)
            market["breadth_note"] = f"自選股站上月線 {breadth_pct:.0f}%"
        if ret20_values:
            market["watchlist_median_ret20"] = round(float(np.median(ret20_values) * 100), 2)

    neutral_news = {
        "score": 50.0, "label": "未深入抓取", "title": "", "link": "",
        "published": "", "status": "screening_only",
    }
    neutral_fundamentals = {
        "score": 50.0, "eps": None, "roe": None, "available": False,
    }
    neutral_advanced = _empty_advanced()

    # 第一階段：全清單快速初篩，不抓新聞、財報、籌碼等慢速資料。
    screening = []
    for item in prepared:
        screening.append(analyze_stock(
            item["sid"], item["name"], item["category"], item["price"], market,
            category_scores.get(item["category"], 50.0), neutral_news,
            neutral_fundamentals, neutral_advanced, "初篩",
        ))

    valid_screening = [r for r in screening if r.get("status") == "ok"]
    short_ranked = sorted(valid_screening, key=lambda r: r.get("short_score", 0), reverse=True)
    mid_ranked = sorted(valid_screening, key=lambda r: r.get("mid_score", 0), reverse=True)
    long_ranked = sorted(valid_screening, key=lambda r: r.get("long_score", 0), reverse=True)
    deep_ids = {
        r["stock_id"]
        for ranking in (short_ranked, mid_ranked, long_ranked)
        for r in ranking[:DEEP_RESEARCH_LIMIT]
    }
    print(
        f"  → 各週期前{DEEP_RESEARCH_LIMIT}名聯集，深入研究 {len(deep_ids)} 檔",
        flush=True,
    )

    results = []
    deep_done = 0
    for item in prepared:
        if item["price"].empty:
            news = {**neutral_news, "label": "無法判讀", "status": "price_unavailable"}
            fundamentals = neutral_fundamentals
            advanced = neutral_advanced
            stage = "資料不足"
        elif item["sid"] in deep_ids:
            deep_done += 1
            print(
                f"  → 深入研究 {deep_done}/{len(deep_ids)} "
                f"{item['sid']} {item['name']}",
                flush=True,
            )
            news = fetch_news(item["sid"], item["name"])
            fundamentals = _fundamental_snapshot(item["sid"])
            advanced = _advanced_snapshot(item["sid"])
            stage = "深入"
        else:
            news = neutral_news
            fundamentals = neutral_fundamentals
            advanced = neutral_advanced
            stage = "初篩"
        results.append(analyze_stock(
            item["sid"], item["name"], item["category"], item["price"], market,
            category_scores.get(item["category"], 50.0), news, fundamentals, advanced,
            stage,
        ))
    results.sort(key=lambda r: (-(r.get("short_score") or 0), -(r.get("long_score") or 0)))
    return market, results
