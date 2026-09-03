import pandas as pd
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

from stock_strategies import research
from tests.conftest import make_price_df


def test_market_snapshot_classifies_rising_index():
    dates = pd.bdate_range("2024-01-01", periods=80)
    close = [17000 + i * 20 for i in range(80)]
    index = pd.DataFrame({"date": dates, "close": close})

    result = research.market_snapshot(index)

    assert result["regime"] == "偏多"
    assert result["score"] == 75.0
    assert result["ret20"] > 0


def test_fetch_news_returns_neutral_when_no_recent_items(monkeypatch):
    class Response:
        content = b"<rss><channel></channel></rss>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(research.requests, "get", lambda *args, **kwargs: Response())

    result = research.fetch_news("2330", "台積電")

    assert result["score"] == 50.0
    assert result["status"] == "no_recent_news"


def test_fetch_news_aggregates_recent_headlines_with_recency(monkeypatch):
    now = datetime.now(timezone.utc)
    recent = format_datetime(now - timedelta(hours=2), usegmt=True)
    older = format_datetime(now - timedelta(days=5), usegmt=True)
    xml = f"""<rss><channel>
      <item><title>台積電營收創高、接單成長</title><link>https://a.example</link>
        <pubDate>{recent}</pubDate><source>來源A</source></item>
      <item><title>市場關注需求</title><link>https://b.example</link>
        <pubDate>{older}</pubDate><source>來源B</source></item>
    </channel></rss>""".encode()

    class Response:
        content = xml

        def raise_for_status(self):
            return None

    monkeypatch.setattr(research.requests, "get", lambda *args, **kwargs: Response())

    result = research.fetch_news("2330", "台積電")

    assert result["status"] == "ok"
    assert result["news_count"] == 2
    assert result["news_24h_count"] == 1
    assert result["news_sources"] == "來源A、來源B"
    assert result["label"] == "利多傾向"
    assert result["news_trend"] in {"改善", "持平", "近期有新訊"}


def test_analyze_stock_produces_short_and_long_research_scores():
    price = make_price_df(n=140, base=100)
    market = {"score": 75.0, "ret20": 0.05, "regime": "偏多"}
    news = {
        "score": 62.0, "label": "利多傾向", "title": "營收創高、接單成長",
        "link": "https://example.com/news", "published": "", "status": "ok",
    }
    fundamentals = {"score": 75.0, "eps": 8.0, "roe": 18.0}

    result = research.analyze_stock(
        "2330", "台積電", "AI", price, market, 70.0, news, fundamentals
    )

    assert result["status"] == "ok"
    assert 0 <= result["short_score"] <= 100
    assert 0 <= result["long_score"] <= 100
    assert result["short_recommendation"] in {"優先研究", "觀察", "暫不列入"}
    assert result["long_recommendation"] in {"優先研究", "觀察", "暫不列入"}
    assert result["news_title"] == "營收創高、接單成長"
    assert "rsi14" in result
    assert "technical_quality" in result
    assert "overall_score" in result
    assert result["consensus"] in {"短長線共識", "短線優先", "中長線優先", "等待確認"}


def test_advanced_snapshot_combines_flow_revenue_and_valuation(monkeypatch):
    monkeypatch.setattr(
        research, "get_institutional",
        lambda *args, **kwargs: pd.DataFrame({"total_net": [10, 20, 30], "foreign_net": [5, 1, 8]}),
    )
    monkeypatch.setattr(
        research, "get_month_revenue",
        lambda *args, **kwargs: pd.DataFrame({"yoy": [0.15], "mom": [0.04]}),
    )
    monkeypatch.setattr(
        research, "get_valuation",
        lambda *args, **kwargs: pd.DataFrame({"per": [20], "pbr": [2]}),
    )
    monkeypatch.setattr(
        research, "get_margin",
        lambda *args, **kwargs: pd.DataFrame({"short_margin_ratio": [0.2, 0.2], "margin_balance": [100, 105]}),
    )
    monkeypatch.setattr(
        research, "get_shareholding",
        lambda *args, **kwargs: pd.DataFrame({"foreign_ratio": [40, 41]}),
    )

    result = research._advanced_snapshot("2330")

    assert result["available"] == 5
    assert result["revenue_yoy"] == 15.0
    assert result["per"] == 20.0
    assert result["foreign_ratio"] == 41.0


def test_run_research_deep_analyzes_only_union_of_top_horizons(monkeypatch):
    price = make_price_df(n=140, base=100)
    index = pd.DataFrame({"close": [17000 + i * 20 for i in range(80)]})
    calls = {"news": 0, "fundamental": 0, "advanced": 0}

    monkeypatch.setattr(research, "get_index_history", lambda *args, **kwargs: index)
    monkeypatch.setattr(research, "get_price_history", lambda *args, **kwargs: price)

    def fake_news(*args, **kwargs):
        calls["news"] += 1
        return {"score": 50.0, "label": "中性", "title": "", "link": "",
                "published": "", "status": "ok"}

    monkeypatch.setattr(research, "fetch_news", fake_news)
    monkeypatch.setattr(
        research, "_fundamental_snapshot",
        lambda *args, **kwargs: calls.__setitem__("fundamental", calls["fundamental"] + 1)
        or {"score": 50.0, "eps": None, "roe": None, "available": False},
    )
    monkeypatch.setattr(
        research, "_advanced_snapshot",
        lambda *args, **kwargs: calls.__setitem__("advanced", calls["advanced"] + 1)
        or research._empty_advanced(),
    )

    watchlist = [
        {"stock_id": str(1000 + i), "name": f"股票{i}", "category": "AI"}
        for i in range(20)
    ]
    _, results = research.run_research(watchlist)

    assert len(results) == 20
    assert sum(r["research_stage"] == "深入" for r in results) == 5
    assert sum(r["research_stage"] == "初篩" for r in results) == 15
    assert calls == {"news": 5, "fundamental": 5, "advanced": 5}
    assert all("mid_score" in result for result in results)
