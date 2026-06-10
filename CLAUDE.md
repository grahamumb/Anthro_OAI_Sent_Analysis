# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Python data pipeline that analyzes Hacker News sentiment toward Anthropic vs OpenAI, and looks for coordinated inauthentic behavior (persistent opinion pushers). It pulls HN data from the HuggingFace `open-index/hacker-news` parquet dataset via DuckDB, scores comments with VADER, builds per-user profiles with a composite "suspicion score", and renders charts.

There is no build system, linter, or test suite — just sequential scripts.

## Commands

```bash
pip install -r requirements.txt          # install dependencies
cp .env.example .env                     # then set HF_TOKEN (HuggingFace read token)

python src/run_all.py                    # full pipeline, all 4 steps in order
python src/01_download_filter.py         # any step can be run individually
```

Steps must run in numeric order on first run — each reads the parquet output of the previous one from `data/` and raises `FileNotFoundError` if it's missing. Once `data/` is populated, later steps can be re-run independently (useful when iterating on analysis or charts without re-downloading).

Run scripts from the repo root. The scripts resolve `data/` and `charts/` relative to their own file location, but step imports (`from keywords import ...`) rely on the script's directory being on `sys.path`, which `python src/NN_*.py` provides.

## Pipeline architecture

Stages communicate only through parquet files in `data/` (gitignored, but persists across sessions in long-lived containers — check before re-downloading, step 1 is the slow/expensive one):

1. **`01_download_filter.py`** — DuckDB queries over `hf://datasets/open-index/hacker-news/...` (httpfs + HF secret from `HF_TOKEN`). Filters stories/comments from 2022-01-01 by AI keyword. Comments carry a `source` column: `in_thread` (parent story is an AI story) vs `off_topic_mention` (brand name injected into a non-AI thread). Dead/shadowbanned comments go to a separate file. HTML is stripped into `text_clean`.
   → `filtered_stories.parquet`, `filtered_comments.parquet`, `dead_comments.parquet`
2. **`02_sentiment.py`** — VADER compound score per comment; tags `mentions_anthropic`/`mentions_openai` and derives `brand_context` (`anthropic_only` / `openai_only` / `both` / `neither`) and a `sentiment` label (±0.05 compound thresholds).
   → `scored_comments.parquet`
3. **`03_analysis.py`** — per-user profiles (sentiment skew, brand lean, off-topic rate, etc.), a weighted composite `suspicion_score` (tunables `MIN_AI_COMMENTS`, `SUSPICION_TOP_N`, `SIM_THRESHOLD` at top of file), weekly temporal sentiment with 8-week rolling baseline and deviation z-scores, TF-IDF cosine-similarity talking-point pairs, and a thread co-appearance network.
   → `user_profiles.parquet`, `suspicious_users.parquet`, `temporal_sentiment.parquet`, `talking_point_clusters.parquet`
4. **`04_visualize.py`** — 9 charts to `charts/` (matplotlib/seaborn PNGs + plotly HTML for the similarity network and suspicion leaderboard). Uses the `Agg` backend; never needs a display.
5. **`05_consistency_report.py`** — interactive HTML report of the top 50 most *consistently positive* users per brand (Wilson lower bound of positive share among brand-exclusive comments; click a user to see their comments/stories in a side panel). Only needs step 2's output. Supports `--data-dir`/`--out` overrides, which the other steps don't.
   → `charts/10_consistency_report.html`, `data/consistent_boosters.parquet`

**`keywords.py`** is the shared vocabulary module: brand term lists, compiled regexes (used by steps 2–3), generated SQL `LIKE` clauses (used by step 1), and `LAUNCH_EVENTS` product-launch dates overlaid on time-series charts. Adding a brand term here propagates to both the SQL filter and the Python tagging — but step 1 must be re-run for the SQL side to take effect.

## Conventions and gotchas

- Data flows DuckDB → polars (I/O) → pandas (analysis). Steps 3–4 convert to pandas immediately via `.to_pandas()`; keep parquet read/write in polars.
- `data/` and `charts/` are gitignored output directories created by the scripts; don't commit their contents.
- `postmortem.md` documents a prior failed session. Its environment-specific claims (no Python installed) may not apply to the current container — verify with `command -v python3` rather than assuming. Its general lessons stand: `ls` before guessing file layout, and `data/` may already hold results from previous runs.
- Several metrics are explicitly proxies, noted in comments/columns: `account_age_proxy_days` (first AI comment, not real account age) and topic concentration (no full-HN denominator — relative ranking only). Preserve these caveats when extending the analysis.
