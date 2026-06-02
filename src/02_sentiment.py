"""
Step 2: Sentiment scoring and brand tagging on filtered comments.

Reads:  data/filtered_comments.parquet
Writes: data/scored_comments.parquet
"""

import re
import polars as pl
from pathlib import Path
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from keywords import ANTHROPIC_RE, OPENAI_RE

DATA_DIR = Path(__file__).parent.parent / "data"

analyzer = SentimentIntensityAnalyzer()


def score_batch(texts: list[str]) -> list[float]:
    return [analyzer.polarity_scores(t or "")["compound"] for t in texts]


def brand_context(mentions_anthropic: bool, mentions_openai: bool) -> str:
    if mentions_anthropic and mentions_openai:
        return "both"
    if mentions_anthropic:
        return "anthropic_only"
    if mentions_openai:
        return "openai_only"
    return "neither"  # in-thread comment that doesn't name either brand directly


def sentiment_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def main():
    path = DATA_DIR / "filtered_comments.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run 01_download_filter.py first — {path} not found")

    df = pl.read_parquet(path)
    print(f"Loaded {len(df):,} comments")

    texts = df["text_clean"].to_list()

    print("Scoring sentiment (VADER)...")
    chunk = 5000
    compounds = []
    for i in tqdm(range(0, len(texts), chunk)):
        compounds.extend(score_batch(texts[i : i + chunk]))

    print("Tagging brand mentions...")
    mentions_anthropic = [bool(ANTHROPIC_RE.search(t or "")) for t in texts]
    mentions_openai    = [bool(OPENAI_RE.search(t or ""))    for t in texts]

    df = df.with_columns([
        pl.Series("compound",            compounds),
        pl.Series("mentions_anthropic",  mentions_anthropic),
        pl.Series("mentions_openai",     mentions_openai),
    ])

    df = df.with_columns([
        pl.struct(["mentions_anthropic", "mentions_openai"])
          .map_elements(
              lambda r: brand_context(r["mentions_anthropic"], r["mentions_openai"]),
              return_dtype=pl.Utf8
          )
          .alias("brand_context"),
        pl.col("compound")
          .map_elements(sentiment_label, return_dtype=pl.Utf8)
          .alias("sentiment"),
    ])

    out = DATA_DIR / "scored_comments.parquet"
    df.write_parquet(out)
    print(f"\nSaved {out} ({len(df):,} rows)")

    # Quick distribution summary
    print("\nSentiment distribution:")
    print(df.group_by("sentiment").agg(pl.len().alias("count")).sort("count", descending=True))
    print("\nBrand context distribution:")
    print(df.group_by("brand_context").agg(pl.len().alias("count")).sort("count", descending=True))


if __name__ == "__main__":
    main()
