"""Tests for the vetted banking knowledge base in knowledge.kb."""

import pytest

from knowledge.kb import TOPICS, answer, article, load_articles, numbers_supported, search


def test_all_articles_load() -> None:
    articles = load_articles()
    assert len(articles) == 12
    assert len({item["id"] for item in articles}) == 12


def test_topics_match_json_order() -> None:
    assert TOPICS == tuple(item["id"] for item in load_articles())
    assert len(TOPICS) == 12


def test_every_topic_has_an_article() -> None:
    for topic in TOPICS:
        item = article(topic)
        assert item is not None
        assert item["zh"]
        assert item["en"]
        assert item["source"]


def test_article_unknown_is_none() -> None:
    assert article("no_such_topic") is None


def test_search_anti_fraud() -> None:
    found = search("被冒充警察要求转到安全账户")
    assert found is not None
    assert found["id"] == "anti_fraud"


def test_search_fx_quota() -> None:
    found = search("how much can I buy in dollars per year")
    assert found is not None
    assert found["id"] == "fx_quota"


def test_search_no_match() -> None:
    assert search("weather today") is None


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("lang", ["zh", "en"])
def test_answer_contains_source(topic: str, lang: str) -> None:
    text = answer(topic, lang)
    source = article(topic)["source"]
    assert source in text


def test_answer_unknown_topic_raises() -> None:
    with pytest.raises(ValueError):
        answer("no_such_topic", "en")


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("lang", ["zh", "en"])
def test_answer_numbers_supported(topic: str, lang: str) -> None:
    assert numbers_supported(answer(topic, lang), topic) is True


def test_unsupported_number_rejected() -> None:
    assert numbers_supported("insured up to ¥600,000", "deposit_insurance") is False
