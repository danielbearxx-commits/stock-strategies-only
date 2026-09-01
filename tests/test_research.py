import pandas as pd

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


