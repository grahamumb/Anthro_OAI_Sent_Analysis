"""
Step 4: Generate all analysis charts.

Reads:  data/scored_comments.parquet
        data/user_profiles.parquet
        data/suspicious_users.parquet
        data/temporal_sentiment.parquet
        data/talking_point_clusters.parquet
Writes: charts/*.png  (and charts/*.html for interactive plots)
"""

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from keywords import LAUNCH_EVENTS

DATA_DIR   = Path(__file__).parent.parent / "data"
CHARTS_DIR = Path(__file__).parent.parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")
BRAND_COLORS = {
    "anthropic_only": "#da7756",   # Anthropic orange-ish
    "openai_only":    "#10a37f",   # OpenAI green
    "both":           "#7c5cbf",   # purple = both
    "neither":        "#aaaaaa",
}
LEAN_COLORS = {
    "pro_anthropic":    "#da7756",
    "anti_anthropic":   "#f7b89c",
    "pro_openai":       "#10a37f",
    "anti_openai":      "#7dd3bc",
    "mixed":            "#7c5cbf",
    "neutral_anthropic": "#e8c5b5",
    "neutral_openai":   "#b7e5d8",
    "neither":          "#cccccc",
}


def load_all():
    comments     = pl.read_parquet(DATA_DIR / "scored_comments.parquet").to_pandas()
    profiles     = pl.read_parquet(DATA_DIR / "user_profiles.parquet").to_pandas()
    suspicious   = pl.read_parquet(DATA_DIR / "suspicious_users.parquet").to_pandas()
    temporal     = pl.read_parquet(DATA_DIR / "temporal_sentiment.parquet").to_pandas()
    try:
        clusters = pl.read_parquet(DATA_DIR / "talking_point_clusters.parquet").to_pandas()
    except Exception:
        clusters = pd.DataFrame()

    comments["time"]  = pd.to_datetime(comments["time"])
    temporal["week"]  = pd.to_datetime(temporal["week"])
    return comments, profiles, suspicious, temporal, clusters


def add_event_lines(ax, alpha=0.6):
    for date_str, label in LAUNCH_EVENTS:
        x = pd.Timestamp(date_str)
        ax.axvline(x, color="#555555", linestyle="--", alpha=alpha, linewidth=0.8)
        ax.text(x, ax.get_ylim()[1], label, rotation=90, fontsize=6,
                va="top", ha="right", color="#555555", alpha=alpha)


# ---------------------------------------------------------------------------
# Chart 1: Time-series sentiment — Anthropic vs OpenAI
# ---------------------------------------------------------------------------

def chart_sentiment_timeseries(temporal: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Weekly Sentiment: Anthropic vs OpenAI Mentions on HN", fontsize=13)

    for ax, ctx, label, color in [
        (axes[0], "anthropic_only", "Anthropic-only comments", BRAND_COLORS["anthropic_only"]),
        (axes[1], "openai_only",    "OpenAI-only comments",    BRAND_COLORS["openai_only"]),
    ]:
        sub = temporal[temporal["brand_context"] == ctx].sort_values("week")
        if sub.empty:
            ax.set_title(f"{label} — no data")
            continue

        ax.plot(sub["week"], sub["mean_compound"], color=color, linewidth=1.5, label="Mean compound")
        ax.fill_between(sub["week"],
                        sub["mean_compound"] - sub["rolling_std"].fillna(0),
                        sub["mean_compound"] + sub["rolling_std"].fillna(0),
                        alpha=0.15, color=color)
        ax.plot(sub["week"], sub["rolling_mean"], color=color, linewidth=2.5,
                linestyle="--", alpha=0.7, label="8-week rolling mean")

        # Annotate high-deviation weeks
        flagged = sub[sub["deviation_zscore"].abs() > 2.5]
        ax.scatter(flagged["week"], flagged["mean_compound"],
                   color="red", zorder=5, s=40, label="Anomaly (>2.5σ)")

        ax.set_title(label, fontsize=10)
        ax.set_ylabel("VADER compound")
        ax.set_ylim(-0.5, 0.5)
        ax.legend(fontsize=8)
        add_event_lines(ax)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    out = CHARTS_DIR / "01_sentiment_timeseries.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Chart 2: Comment volume heatmap by sentiment bucket
# ---------------------------------------------------------------------------

def chart_volume_heatmap(temporal: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Weekly Comment Volume by Sentiment (Anthropic vs OpenAI)", fontsize=12)

    comments_raw = pl.read_parquet(DATA_DIR / "scored_comments.parquet").to_pandas()
    comments_raw["time"] = pd.to_datetime(comments_raw["time"])
    comments_raw["week"] = comments_raw["time"].dt.to_period("W").dt.start_time

    for ax, ctx, label in [
        (axes[0], "anthropic_only", "Anthropic"),
        (axes[1], "openai_only",    "OpenAI"),
    ]:
        sub = comments_raw[comments_raw["brand_context"] == ctx]
        if sub.empty:
            ax.set_title(f"{label} — no data")
            continue

        pivot = sub.pivot_table(index="week", columns="sentiment",
                                values="id", aggfunc="count", fill_value=0)
        pivot = pivot.reindex(columns=["positive", "neutral", "negative"])
        pivot.index = pd.to_datetime(pivot.index)

        sns.heatmap(pivot.T, ax=ax, cmap="RdYlGn_r" if label == "Anthropic" else "RdYlGn",
                    cbar_kws={"label": "comment count"})
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Week")
        ax.set_ylabel("Sentiment")
        # Show every 8th week label
        xticks = ax.get_xticklabels()
        for i, t in enumerate(xticks):
            if i % 8 != 0:
                t.set_visible(False)

    plt.tight_layout()
    out = CHARTS_DIR / "02_volume_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Chart 3: Top opinionated users
# ---------------------------------------------------------------------------

def chart_top_users(profiles: pd.DataFrame):
    top = profiles.nlargest(30, "total_ai_comments")
    top = top.sort_values("avg_sentiment")

    colors = [LEAN_COLORS.get(lean, "#aaaaaa") for lean in top["brand_lean"]]

    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(top["username"], top["total_ai_comments"], color=colors)
    ax.set_xlabel("Total AI-keyword comments")
    ax.set_title("Top 30 Users by AI Comment Volume\n(color = brand lean + sentiment direction)", fontsize=11)

    legend_handles = [mpatches.Patch(color=v, label=k) for k, v in LEAN_COLORS.items()
                      if k in top["brand_lean"].values]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right")

    plt.tight_layout()
    out = CHARTS_DIR / "03_top_users.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Chart 4: User suspicion scatter
# ---------------------------------------------------------------------------

def chart_suspicion_scatter(profiles: pd.DataFrame):
    p = profiles[profiles["total_ai_comments"] >= 5].copy()
    colors = [LEAN_COLORS.get(l, "#aaaaaa") for l in p["brand_lean"]]
    sizes  = np.clip(p["total_ai_comments"] * 2, 10, 300)

    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(
        p["account_age_proxy_days"],
        p["sentiment_std"],
        c=colors,
        s=sizes,
        alpha=0.6,
        edgecolors="white",
        linewidths=0.3,
    )
    ax.set_xlabel("Account age proxy (days since first AI comment → now)")
    ax.set_ylabel("Sentiment std (low = suspiciously consistent opinion)")
    ax.set_title("User Suspicion Landscape\n"
                 "Bottom-left quadrant: newer accounts with consistently one-directional sentiment",
                 fontsize=11)

    # Annotate top-N suspicious users
    top_sus = p.nlargest(15, "suspicion_score")
    for _, row in top_sus.iterrows():
        ax.annotate(row["username"], (row["account_age_proxy_days"], row["sentiment_std"]),
                    fontsize=6, alpha=0.8, xytext=(3, 3), textcoords="offset points")

    legend_handles = [mpatches.Patch(color=v, label=k) for k, v in LEAN_COLORS.items()
                      if k in p["brand_lean"].values]
    ax.legend(handles=legend_handles, fontsize=7, loc="upper right")

    plt.tight_layout()
    out = CHARTS_DIR / "04_suspicion_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Chart 5: Topic concentration histogram
# ---------------------------------------------------------------------------

def chart_topic_concentration(profiles: pd.DataFrame):
    p = profiles[profiles["total_ai_comments"] >= 5].copy()

    # Since we don't have full HN history, use relative ranking
    # Split by brand lean for comparison
    fig, ax = plt.subplots(figsize=(10, 5))

    for lean, color in [("pro_anthropic", LEAN_COLORS["pro_anthropic"]),
                        ("pro_openai",    LEAN_COLORS["pro_openai"]),
                        ("mixed",         LEAN_COLORS["mixed"])]:
        sub = p[p["brand_lean"] == lean]["ai_comments_per_day"]
        if not sub.empty:
            ax.hist(sub, bins=40, alpha=0.5, color=color, label=lean, density=True)

    ax.set_xlabel("AI comments per active day (proxy for topic concentration)")
    ax.set_ylabel("Density")
    ax.set_title("Comment Rate Distribution by Brand Lean\n"
                 "High rate + strong lean = advocate signal", fontsize=11)
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = CHARTS_DIR / "05_topic_concentration.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Chart 6: Off-topic injection timeline
# ---------------------------------------------------------------------------

def chart_off_topic_timeline(comments: pd.DataFrame):
    off = comments[comments["source"] == "off_topic_mention"].copy()
    if off.empty:
        print("No off-topic comments found — skipping chart 6")
        return

    off["week"] = off["time"].dt.to_period("W").dt.start_time

    agg = (
        off.groupby(["week", "brand_context"])
        .agg(count=("id", "count"), avg_compound=("compound", "mean"))
        .reset_index()
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("Off-Topic AI Brand Mentions Over Time\n"
                 "(comments in non-AI threads that inject AI brand names)", fontsize=12)

    for ax, ctx, label, color in [
        (axes[0], "anthropic_only", "Anthropic injections", BRAND_COLORS["anthropic_only"]),
        (axes[1], "openai_only",    "OpenAI injections",    BRAND_COLORS["openai_only"]),
    ]:
        sub = agg[agg["brand_context"] == ctx].sort_values("week")
        if sub.empty:
            ax.set_title(f"{label} — no data")
            continue
        ax.bar(sub["week"], sub["count"], width=6, color=color, alpha=0.7, label="Volume")
        ax2 = ax.twinx()
        ax2.plot(sub["week"], sub["avg_compound"], color="black", linewidth=1.2,
                 linestyle="--", alpha=0.7, label="Avg sentiment")
        ax2.set_ylabel("Avg sentiment", fontsize=8)
        ax2.set_ylim(-0.5, 0.5)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Comment count")
        add_event_lines(ax)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    out = CHARTS_DIR / "06_off_topic_timeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Chart 7: Talking-point similarity network (interactive HTML)
# ---------------------------------------------------------------------------

def chart_similarity_network(clusters: pd.DataFrame, profiles: pd.DataFrame):
    if clusters.empty:
        print("No talking-point pairs above threshold — skipping chart 7")
        return

    G = nx.Graph()
    for _, row in clusters.iterrows():
        G.add_edge(row["user_a"], row["user_b"], weight=row["similarity"])

    lean_map = profiles.set_index("username")["brand_lean"].to_dict()
    pos = nx.spring_layout(G, seed=42, k=1.5)

    # Plotly interactive graph
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_colors = [LEAN_COLORS.get(lean_map.get(n, "neither"), "#aaaaaa") for n in G.nodes()]
    node_text  = [f"{n}<br>lean: {lean_map.get(n, 'unknown')}" for n in G.nodes()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.8, color="#aaaaaa"), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                             text=list(G.nodes()), textposition="top center",
                             marker=dict(size=10, color=node_colors, line=dict(width=1)),
                             hovertext=node_text, hoverinfo="text"))
    fig.update_layout(title="Talking-Point Similarity Network (suspicious users)",
                      showlegend=False, hovermode="closest",
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

    out = CHARTS_DIR / "07_similarity_network.html"
    fig.write_html(str(out))
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Chart 8: Sentiment divergence (observed vs rolling baseline)
# ---------------------------------------------------------------------------

def chart_sentiment_divergence(temporal: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("Sentiment Divergence from Baseline\n"
                 "(observed minus 8-week rolling mean; red = unexplained spike)", fontsize=12)

    for ax, ctx, label, color in [
        (axes[0], "anthropic_only", "Anthropic", BRAND_COLORS["anthropic_only"]),
        (axes[1], "openai_only",    "OpenAI",    BRAND_COLORS["openai_only"]),
    ]:
        sub = temporal[temporal["brand_context"] == ctx].sort_values("week")
        if sub.empty:
            ax.set_title(f"{label} — no data")
            continue

        divergence = sub["mean_compound"] - sub["rolling_mean"]
        colors = ["red" if abs(z) > 2.5 else color
                  for z in sub["deviation_zscore"].fillna(0)]

        ax.bar(sub["week"], divergence, width=6, color=colors, alpha=0.75)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Sentiment divergence")
        add_event_lines(ax)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    out = CHARTS_DIR / "08_sentiment_divergence.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Chart 9: Suspicion leaderboard (interactive)
# ---------------------------------------------------------------------------

def chart_suspicion_leaderboard(suspicious: pd.DataFrame):
    top = suspicious.nlargest(40, "suspicion_score")

    fig = px.bar(
        top,
        x="suspicion_score",
        y="username",
        orientation="h",
        color="brand_lean",
        color_discrete_map=LEAN_COLORS,
        hover_data=["total_ai_comments", "avg_sentiment", "sentiment_std",
                    "off_topic_rate", "threads_appeared_in"],
        title="Top 40 Suspicious Users by Persistent Opinion Pushing Score",
        labels={"suspicion_score": "Suspicion Score", "username": ""},
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    out = CHARTS_DIR / "09_suspicion_leaderboard.html"
    fig.write_html(str(out))
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    comments, profiles, suspicious, temporal, clusters = load_all()

    print("\nGenerating charts...")
    chart_sentiment_timeseries(temporal)
    chart_volume_heatmap(temporal)
    chart_top_users(profiles)
    chart_suspicion_scatter(profiles)
    chart_topic_concentration(profiles)
    chart_off_topic_timeline(comments)
    chart_similarity_network(clusters, profiles)
    chart_sentiment_divergence(temporal)
    chart_suspicion_leaderboard(suspicious)

    print(f"\nAll charts saved to {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
