"""Split-adjustment math (offline)."""

import pandas as pd

from fetch.fetchStockData import _split_adjust
from fetch.fetchSECFilings import _risk_excerpt, _mdna_excerpt


def test_split_adjust_backadjusts_pre_split_bars():
    # raw: 40, 40 pre-split, then a 4:1 split, then 10, 10 -> all become 10
    df = pd.DataFrame({
        "Open": [40.0, 40.0, 10.0, 10.0],
        "High": [40.0, 40.0, 10.0, 10.0],
        "Low": [40.0, 40.0, 10.0, 10.0],
        "Close": [40.0, 40.0, 10.0, 10.0],
        "Stock Splits": [0.0, 0.0, 4.0, 0.0],
    })
    out = _split_adjust(df)
    assert out["Close"].tolist() == [10.0, 10.0, 10.0, 10.0]


def test_split_adjust_is_ratio_preserving():
    # a uniform post-series split scales every bar equally -> return ratios unchanged
    df = pd.DataFrame({
        "Open": [100.0, 110.0, 121.0],
        "High": [100.0, 110.0, 121.0],
        "Low": [100.0, 110.0, 121.0],
        "Close": [100.0, 110.0, 121.0],
        "Stock Splits": [0.0, 0.0, 0.0],
    })
    out = _split_adjust(df)
    c = out["Close"].tolist()
    assert c[1] / c[0] == c[2] / c[1]  # 10% steps preserved


def test_split_adjust_no_column_is_noop():
    df = pd.DataFrame({"Close": [1.0, 2.0]})
    assert _split_adjust(df)["Close"].tolist() == [1.0, 2.0]


def test_risk_excerpt_skips_toc_and_crossref():
    # A realistic 10-K shape: TOC entry, then a preamble cross-reference, then
    # the actual Item 1A section with substantial content.
    toc = "Table of Contents Item 1A. Risk Factors 5 Item 1B. Unresolved Staff Comments 17 "
    preamble = "Item 1A of this Form 10-K under the heading Risk Factors. The Company assumes no obligation. "
    real = "PART I Item 1. Business overview text. Item 1A. Risk Factors UNIQUE_RISK_MARKER " + ("risk prose " * 400) + " Item 1B. Unresolved Staff Comments none. "
    text = toc + preamble + real

    excerpt, quality = _risk_excerpt(text)
    assert quality == "section"
    assert excerpt.startswith("Item 1A. Risk Factors UNIQUE_RISK_MARKER")  # the real section
    assert "Unresolved Staff Comments 17" not in excerpt[:50]            # not the TOC


def test_risk_excerpt_keyword_fallback_is_flagged():
    # No proper Item 1A section, but the phrase appears -> a flagged fallback slice.
    text = "intro text " * 100 + " other risk factors may apply to the business. " + ("tail " * 100)
    excerpt, quality = _risk_excerpt(text)
    assert quality == "fallback"
    assert excerpt


def test_risk_excerpt_empty_when_no_section_or_keyword():
    # No section and no keyword -> empty + flag, NEVER a blind cover-page slice.
    excerpt, quality = _risk_excerpt("Some filing text with no item headers at all. " * 50)
    assert quality == "empty"
    assert excerpt == ""


def test_mdna_excerpt_finds_section_not_toc():
    toc = "Table of Contents Item 7. Management's Discussion and Analysis 21 Item 7A. Market Risk 27 "
    real = ("Item 7. Management's Discussion and Analysis MDNA_MARKER " + ("results prose " * 400)
            + " Item 7A. Quantitative Disclosures end. ")
    excerpt, quality = _mdna_excerpt(toc + real)
    assert quality == "section"
    assert "MDNA_MARKER" in excerpt[:80]
    assert "Market Risk 27" not in excerpt[:50]  # not the TOC entry


def test_mdna_excerpt_empty_when_absent():
    assert _mdna_excerpt("a filing with no mdna section " * 50) == ("", "empty")
