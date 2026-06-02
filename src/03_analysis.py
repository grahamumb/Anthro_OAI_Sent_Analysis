"""
Step 3: User profile analysis and coordinated inauthentic behavior (CIB) detection.

Focus: persistent opinion pushers (employees/advocates), not bot bursts.

Reads:  data/scored_comments.parquet
        data/filtered_stories.parquet
Writes: data/user_profiles.parquet
        data/suspicious_users.parquet
        data/temporal_sentiment.parquet
        data/talking_point_clusters.parquet
"""

import re
import numpy as np
import pandas as pd
import polars as pl
import networkx as nx
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from keywords import ANTHROPIC_RE, OPENAI_RE, LAUNCH_EVENTS

DATA_DIR = Path(__file__).parent.parent / "data"

MIN_AI_COMMENTS = 5     # minimum comments to include a user in profile analysis
SUSPICION_TOP_N = 200   # how many users to inspect in detail for CIB signals
SIM_THRESHOLD   = 0.70  # cosine similarity threshold for "shared talking points"


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    comments = pl.read_parquet(DATA_DIR / "scored_comments.parquet").to_pandas()
    stories  = pl.read_parquet(DATA_DIR / "filtered_stories.parquet").to_pandas()
    comments["time"] = pd.to_datetime(comments["time"])
    stories["time"]  = pd.to_datetime(stories["time"])
    return comments, stories


# ---------------------------------------------------------------------------
# 2. User profiles
# ---------------------------------------------------------------------------

def build_user_profiles(comments: pd.DataFrame) -> pd.DataFrame:
    """
    Per-user metrics that characterise persistent opinion pushers.
    """
    g = comments.groupby("by")

    profiles = pd.DataFrame({
        "username":          g["by"].first(),
        "total_ai_comments": g["id"].count(),
        "avg_sentiment":     g["compound"].mean(),
        "sentiment_std":     g["compound"].std().fillna(0),
        "pct_positive":      g.apply(lambda x: (x["sentiment"] == "positive").mean()),
        "pct_negative":      g.apply(lambda x: (x["sentiment"] == "negative").mean()),
        "pct_neutral":       g.apply(lambda x: (x["sentiment"] == "neutral").mean()),
        "mentions_anthropic_count": g["mentions_anthropic"].sum(),
        "mentions_openai_count":    g["mentions_openai"].sum(),
        "threads_appeared_in":      g["parent"].nunique(),
        "first_ai_comment":  g["time"].min(),
        "last_ai_comment":   g["time"].max(),
        "off_topic_count":   g.apply(lambda x: (x["source"] == "off_topic_mention").sum()),
    }).reset_index(drop=True)

    profiles["days_active_in_ai"] = (
        profiles["last_ai_comment"] - profiles["first_ai_comment"]
    ).dt.days.clip(lower=1)

    profiles["ai_comments_per_day"] = (
        profiles["total_ai_comments"] / profiles["days_active_in_ai"]
    )

    # Sentiment skew: +1 = all positive, -1 = all negative
    profiles["sentiment_skew"] = profiles["pct_positive"] - profiles["pct_negative"]

    # Brand lean
    total_brand = (profiles["mentions_anthropic_count"] + profiles["mentions_openai_count"]).clip(lower=1)
    profiles["anthropic_ratio"] = profiles["mentions_anthropic_count"] / total_brand
    profiles["openai_ratio"]    = profiles["mentions_openai_count"]    / total_brand

    def brand_lean(row):
        if row["mentions_anthropic_count"] == 0 and row["mentions_openai_count"] == 0:
            return "neither"
        if row["anthropic_ratio"] > 0.7:
            lean = "anthropic"
        elif row["openai_ratio"] > 0.7:
            lean = "openai"
        else:
            lean = "mixed"
        # "mixed" means both brands — just return mixed; don't apply sentiment prefix
        if lean == "mixed":
            return "mixed"
        if row["sentiment_skew"] > 0.3:
            return f"pro_{lean}"
        if row["sentiment_skew"] < -0.3:
            return f"anti_{lean}"
        return f"neutral_{lean}"

    profiles["brand_lean"] = profiles.apply(brand_lean, axis=1)

    # Off-topic injection rate
    profiles["off_topic_rate"] = (
        profiles["off_topic_count"] / profiles["total_ai_comments"]
    )

    # Account age proxy: earliest AI comment as lower-bound for account age
    # (real account age needs full HN history, but this gives us a floor)
    profiles["account_age_proxy_days"] = (
        pd.Timestamp.now() - profiles["first_ai_comment"]
    ).dt.days

    return profiles


def compute_topic_concentration(comments: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Fraction of a user's AI-related comments out of their total HN comments.
    Requires the full comment set — here we use the full scored_comments as numerator.
    For true concentration we'd need total HN comment count, but we can note this
    in output as a lower-bound metric.
    """
    profiles = profiles.copy()
    profiles["topic_concentration_note"] = (
        "total_ai_comments / total_HN_comments — denominator requires full HN history; "
        "use as relative ranking only"
    )
    return profiles


def suspicion_score(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Composite suspicion score for persistent opinion pushing.
    Higher = more likely to be an advocate/employee.
    """
    p = profiles[profiles["total_ai_comments"] >= MIN_AI_COMMENTS].copy()

    # Normalise each signal to [0, 1]
    def norm(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.0, index=s.index)

    # High volume × high extremity × low std (consistent one-directional)
    p["s_volume"]      = norm(p["total_ai_comments"])
    p["s_extremity"]   = norm(p["sentiment_skew"].abs())
    p["s_consistency"] = 1 - norm(p["sentiment_std"])   # low std → high score
    p["s_off_topic"]   = norm(p["off_topic_rate"])
    p["s_breadth"]     = norm(p["threads_appeared_in"])

    p["suspicion_score"] = (
        0.30 * p["s_volume"]      +
        0.25 * p["s_extremity"]   +
        0.20 * p["s_consistency"] +
        0.15 * p["s_off_topic"]   +
        0.10 * p["s_breadth"]
    )

    return p.sort_values("suspicion_score", ascending=True).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Temporal sentiment aggregation
# ---------------------------------------------------------------------------

def temporal_sentiment(comments: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly mean sentiment per brand context, plus comment volume.
    """
    df = comments.copy()
    df["week"] = df["time"].dt.to_period("W").dt.start_time

    agg = (
        df.groupby(["week", "brand_context"])
        .agg(
            mean_compound=("compound", "mean"),
            count=("id", "count"),
            pct_positive=("sentiment", lambda x: (x == "positive").mean()),
            pct_negative=("sentiment", lambda x: (x == "negative").mean()),
            pct_neutral=("sentiment",  lambda x: (x == "neutral").mean()),
            score_weighted_sentiment=("compound", lambda x: np.average(
                x, weights=df.loc[x.index, "score"].clip(lower=0) + 1
            )),
        )
        .reset_index()
    )

    # Rolling 8-week baseline and deviation
    for ctx in agg["brand_context"].unique():
        mask = agg["brand_context"] == ctx
        roll = agg.loc[mask, "mean_compound"].rolling(8, min_periods=2).mean()
        std  = agg.loc[mask, "mean_compound"].rolling(8, min_periods=2).std()
        agg.loc[mask, "rolling_mean"]   = roll.values
        agg.loc[mask, "rolling_std"]    = std.values
        agg.loc[mask, "deviation_zscore"] = (
            (agg.loc[mask, "mean_compound"] - roll) / std.clip(lower=1e-4)
        ).values

    return agg


# ---------------------------------------------------------------------------
# 4. Talking-point similarity (cross-account, spread over time)
# ---------------------------------------------------------------------------

def talking_point_clusters(comments: pd.DataFrame, top_users: list[str]) -> pd.DataFrame:
    """
    For the top suspicious users: compute TF-IDF corpus per user, find pairs
    with cosine similarity > threshold. High similarity = shared talking points.
    """
    user_corpora = (
        comments[comments["by"].isin(top_users)]
        .groupby("by")["text_clean"]
        .apply(lambda texts: " ".join(t for t in texts if t))
        .reset_index()
    )
    user_corpora = user_corpora[user_corpora["text_clean"].str.len() > 50]

    if len(user_corpora) < 2:
        return pd.DataFrame()

    vec = TfidfVectorizer(max_features=5000, sublinear_tf=True, ngram_range=(1, 2))
    tfidf = vec.fit_transform(user_corpora["text_clean"])
    sim_matrix = cosine_similarity(tfidf)

    users = user_corpora["by"].tolist()
    pairs = []
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            if sim_matrix[i, j] >= SIM_THRESHOLD:
                pairs.append({
                    "user_a":     users[i],
                    "user_b":     users[j],
                    "similarity": round(sim_matrix[i, j], 4),
                })

    return pd.DataFrame(pairs).sort_values("similarity", ascending=False)


# ---------------------------------------------------------------------------
# 5. Co-appearance network
# ---------------------------------------------------------------------------

def build_coappearance_network(comments: pd.DataFrame, top_users: list[str]) -> nx.Graph:
    """
    Bipartite projection: users who repeatedly comment on the same threads form edges.
    """
    sub = comments[comments["by"].isin(top_users)][["by", "parent"]]
    G = nx.Graph()
    G.add_nodes_from(top_users)

    thread_user_map: dict[int, list[str]] = {}
    for _, row in sub.iterrows():
        thread_user_map.setdefault(row["parent"], []).append(row["by"])

    for thread_id, users_in_thread in thread_user_map.items():
        unique_users = list(set(users_in_thread))
        for i in range(len(unique_users)):
            for j in range(i + 1, len(unique_users)):
                u, v = unique_users[i], unique_users[j]
                if G.has_edge(u, v):
                    G[u][v]["weight"] += 1
                else:
                    G.add_edge(u, v, weight=1)

    return G


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    comments, stories = load()
    print(f"Loaded {len(comments):,} comments, {len(stories):,} stories")

    print("\nBuilding user profiles...")
    profiles = build_user_profiles(comments)
    profiles = compute_topic_concentration(comments, profiles)

    print("Computing suspicion scores...")
    profiles_scored = suspicion_score(profiles)

    out_profiles = DATA_DIR / "user_profiles.parquet"
    pl.from_pandas(profiles_scored).write_parquet(out_profiles)
    print(f"Saved {out_profiles} ({len(profiles_scored):,} users)")

    # Top suspicious users
    top = profiles_scored.nlargest(SUSPICION_TOP_N, "suspicion_score")
    top_users = top["username"].tolist()
    out_sus = DATA_DIR / "suspicious_users.parquet"
    pl.from_pandas(top).write_parquet(out_sus)
    print(f"Saved {out_sus} (top {len(top)} suspicious users)")

    print("\nComputing temporal sentiment...")
    temp = temporal_sentiment(comments)
    out_temp = DATA_DIR / "temporal_sentiment.parquet"
    pl.from_pandas(temp).write_parquet(out_temp)
    print(f"Saved {out_temp}")

    print("\nComputing talking-point similarity clusters...")
    clusters = talking_point_clusters(comments, top_users)
    out_clusters = DATA_DIR / "talking_point_clusters.parquet"
    pl.from_pandas(clusters).write_parquet(out_clusters)
    print(f"Saved {out_clusters} ({len(clusters):,} high-similarity pairs)")

    # Summary output
    print("\n--- Top 20 Suspicious Users ---")
    cols = ["username", "total_ai_comments", "avg_sentiment", "sentiment_std",
            "sentiment_skew", "brand_lean", "off_topic_rate", "suspicion_score"]
    print(top[cols].head(20).to_string(index=False))

    print("\n--- Talking-point pairs (similarity > threshold) ---")
    if not clusters.empty:
        print(clusters.head(20).to_string(index=False))
    else:
        print("No pairs above threshold.")


if __name__ == "__main__":
    main()
