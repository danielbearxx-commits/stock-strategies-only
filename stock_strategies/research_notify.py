"""研究版 Telegram 報告格式。"""
from __future__ import annotations

from datetime import datetime
import re


def _clean(value: str, limit: int = 150) -> str:
    text = re.sub(r"[\r\n*_`\[\]]", " ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def _ok(results: list[dict]) -> list[dict]:
    return [r for r in results if r.get("status") == "ok"]


def format_research_messages(market: dict, results: list[dict]) -> list[str]:
    """回傳三則可直接送 Telegram 的研究版訊息。"""
    today = datetime.now().strftime("%Y/%m/%d")
    valid = _ok(results)
    short = sorted(valid, key=lambda r: r.get("short_score", 0), reverse=True)
    mid = sorted(valid, key=lambda r: r.get("mid_score", 0), reverse=True)
    long = sorted(valid, key=lambda r: r.get("long_score", 0), reverse=True)
    deep = [r for r in valid if r.get("research_stage") == "深入"]
    priority_short = sum(r.get("short_recommendation") == "優先研究" for r in valid)
    priority_long = sum(r.get("long_recommendation") == "優先研究" for r in valid)

    msg1 = [
        f"📚 *獨立研究版選股* {today}",
        f"掃描 {len(results)} 檔 | 可分析 {len(valid)} 檔",
        f"深入研究 {len(deep)} 檔 | 其餘為快速初篩",
        "",
        f"🌡️ 市場環境：*{_clean(market.get('regime', '資料不足'))}*",
        _clean(market.get("note", "市場資料不足")),
        "",
        f"📈 短線：優先研究 {priority_short} 檔",
        f"📊 中長線：優先研究 {priority_long} 檔",
        "",
        "🔎 *短線前五名*",
    ]
    for i, r in enumerate(short[:5], 1):
        msg1.append(
            f"{i}. {r['stock_id']} {r['name']} "
            f"{r['short_score']}分｜{r['short_recommendation']}"
        )
    msg1 += ["", "🔎 *中期前五名*"]
    for i, r in enumerate(mid[:5], 1):
        msg1.append(
            f"{i}. {r['stock_id']} {r['name']} "
            f"{r['mid_score']}分｜{r['mid_recommendation']}"
        )
    msg1 += ["", "🔎 *中長線前五名*"]
    for i, r in enumerate(long[:5], 1):
        msg1.append(
            f"{i}. {r['stock_id']} {r['name']} "
            f"{r['long_score']}分｜{r['long_recommendation']}｜{r.get('consensus', '等待確認')}"
        )
    msg1 += ["", "_研究分數是篩選工具，不是買進指令；新聞需人工閱讀原文。_"]

    msg2 = ["📈 *研究版｜短線詳細*", ""]
    for r in [r for r in short if r.get("research_stage") == "深入"][:8]:
        msg2 += [
            f"*{r['stock_id']} {r['name']}*｜{r['short_score']}分｜{r['short_recommendation']}｜{r.get('research_stage', '')}",
            f"5日 {r.get('ret5', 'N/A')}%｜20日 {r.get('ret20', 'N/A')}%｜"
            f"量能 {r.get('vol_ratio', 'N/A')}倍｜相對大盤 {r.get('relative20', 'N/A')}%｜"
            f"RSI {r.get('rsi14', 'N/A')}",
            f"理由：{_clean(r.get('short_reasons', ''))}",
            f"籌碼：法人分 {r.get('flow_score', 'N/A')}｜新聞 {r.get('news_trend', 'N/A')} "
            f"({r.get('news_count', 0)}則)",
            f"風險：{_clean(r.get('risk_notes', ''))}",
        ]
        if r.get("news_title"):
            msg2.append(f"新聞：{_clean(r['news_title'])}")
            if r.get("news_link"):
                msg2.append(f"來源：{r['news_link']}")
        msg2.append("")

    msg3 = ["📊 *研究版｜中期／中長線詳細*", ""]
    msg3.append("【中期】")
    for r in [r for r in mid if r.get("research_stage") == "深入"][:5]:
        msg3 += [
            f"*{r['stock_id']} {r['name']}*｜{r['mid_score']}分｜{r['mid_recommendation']}",
            f"20日 {r.get('ret20', 'N/A')}%｜60日 {r.get('ret60', 'N/A')}%｜{_clean(r.get('mid_reasons', ''))}",
            "",
        ]
    msg3.append("【中長線】")
    for r in [r for r in long if r.get("research_stage") == "深入"][:8]:
        msg3 += [
            f"*{r['stock_id']} {r['name']}*｜{r['long_score']}分｜{r['long_recommendation']}",
            f"60日 {r.get('ret60', 'N/A')}%｜EPS {r.get('eps', 'N/A')}｜"
            f"ROE {r.get('roe', 'N/A')}%｜營收年增 {r.get('revenue_yoy', 'N/A')}%｜"
            f"類股分 {r.get('category_score', 'N/A')}",
            f"理由：{_clean(r.get('long_reasons', ''))}",
            f"估值：PER {r.get('per', 'N/A')}｜PBR {r.get('pbr', 'N/A')}｜"
            f"資料可信度 {r.get('data_confidence', 'N/A')}",
            f"風險：{_clean(r.get('risk_notes', ''))}",
            "",
        ]

    return ["\n".join(msg1), "\n".join(msg2), "\n".join(msg3)]
