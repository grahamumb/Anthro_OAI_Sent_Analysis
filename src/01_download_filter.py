"""
Step 1: Query HuggingFace HN parquet via DuckDB and save filtered subsets.

Produces:
  data/filtered_stories.parquet  — AI-topic stories (2022-01-01 onward)
  data/filtered_comments.parquet — AI-keyword comments (in-thread + off-topic branch)
  data/dead_comments.parquet     — shadowbanned comments mentioning AI terms (separate)
"""

import os
import sys
import duckdb
import polars as pl
from pathlib import Path
from dotenv import load_dotenv
from keywords import STORY_TITLE_FILTER, COMMENT_TEXT_FILTER

load_dotenv(Path(__file__).parent.parent / ".env")

HF_TOKEN  = os.getenv("HF_TOKEN", "")
DATA_DIR  = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

START_DATE = "2022-01-01"

# Dataset is partitioned by year/month: data/{year}/{year}-{month}.parquet
# List the specific year globs we need rather than a recursive wildcard
HN_YEARS   = ["2022", "2023", "2024", "2025", "2026"]
HN_PARQUET_LIST = [
    f"hf://datasets/open-index/hacker-news/data/{y}/{y}-*.parquet"
    for y in HN_YEARS
]
# DuckDB read_parquet accepts a list, but also a glob. We'll build a union query.
def _parquet_sources() -> str:
    """Return a SQL-safe list literal for read_parquet([...])."""
    quoted = ", ".join(f"'{p}'" for p in HN_PARQUET_LIST)
    return f"[{quoted}]"


def get_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    if HF_TOKEN:
        # DuckDB ≥0.10 uses the secrets API for HuggingFace auth
        conn.execute(f"""
            CREATE SECRET hf_secret (
                TYPE huggingface,
                token '{HF_TOKEN}'
            )
        """)
    return conn


def fetch_stories(conn: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    print("Fetching AI-topic stories...")
    sql = f"""
        SELECT
            id,
            "by",
            CAST(time AS TIMESTAMP) AS time,
            title,
            url,
            score,
            descendants
        FROM read_parquet({_parquet_sources()})
        WHERE type = 1
          AND deleted IS DISTINCT FROM 1
          AND time >= '{START_DATE}'
          AND ({STORY_TITLE_FILTER})
        ORDER BY time
    """
    df = conn.execute(sql).pl()
    print(f"  → {len(df):,} stories")
    return df


def fetch_comments(conn: duckdb.DuckDBPyConnection, story_ids: list[int]) -> pl.DataFrame:
    """
    Two branches:
      A) Any comment whose parent story is in our story set (in-thread signal)
      B) Any comment anywhere on HN that mentions AI terms (off-topic injection signal)
    Dead (shadowbanned) comments are collected separately.
    """
    ids_csv = ",".join(str(i) for i in story_ids)

    print("Fetching AI-keyword comments (live)...")
    sql_live = f"""
        SELECT
            id,
            "by",
            CAST(time AS TIMESTAMP) AS time,
            parent,
            text,
            score,
            CASE
                WHEN parent IN ({ids_csv}) THEN 'in_thread'
                ELSE 'off_topic_mention'
            END AS source
        FROM read_parquet({_parquet_sources()})
        WHERE type = 2
          AND time >= '{START_DATE}'
          AND (dead IS DISTINCT FROM 1)
          AND (
              parent IN ({ids_csv})
              OR ({COMMENT_TEXT_FILTER})
          )
        ORDER BY time
    """
    df_live = conn.execute(sql_live).pl()
    print(f"  → {len(df_live):,} live comments")

    print("Fetching dead (shadowbanned) AI comments...")
    sql_dead = f"""
        SELECT
            id,
            "by",
            CAST(time AS TIMESTAMP) AS time,
            parent,
            text,
            score
        FROM read_parquet({_parquet_sources()})
        WHERE type = 2
          AND time >= '{START_DATE}'
          AND dead = 1
          AND ({COMMENT_TEXT_FILTER})
        ORDER BY time
    """
    df_dead = conn.execute(sql_dead).pl()
    print(f"  → {len(df_dead):,} dead comments")

    return df_live, df_dead


def clean_html(text: str) -> str:
    from bs4 import BeautifulSoup
    if not text:
        return ""
    return BeautifulSoup(text, "lxml").get_text(separator=" ").strip()


def main():
    conn = get_conn()

    stories = fetch_stories(conn)
    story_ids = stories["id"].to_list()

    comments_live, comments_dead = fetch_comments(conn, story_ids)

    print("Cleaning HTML from comment text...")
    comments_live = comments_live.with_columns(
        pl.col("text").map_elements(clean_html, return_dtype=pl.Utf8).alias("text_clean")
    )
    comments_dead = comments_dead.with_columns(
        pl.col("text").map_elements(clean_html, return_dtype=pl.Utf8).alias("text_clean")
    )

    out_stories  = DATA_DIR / "filtered_stories.parquet"
    out_comments = DATA_DIR / "filtered_comments.parquet"
    out_dead     = DATA_DIR / "dead_comments.parquet"

    stories.write_parquet(out_stories)
    comments_live.write_parquet(out_comments)
    comments_dead.write_parquet(out_dead)

    print(f"\nSaved:")
    print(f"  {out_stories}  ({len(stories):,} rows)")
    print(f"  {out_comments}  ({len(comments_live):,} rows)")
    print(f"  {out_dead}  ({len(comments_dead):,} rows)")


if __name__ == "__main__":
    main()
