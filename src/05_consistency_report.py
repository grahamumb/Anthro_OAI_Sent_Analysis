"""
Step 5: Interactive "consistent boosters" report.

Ranks users by how *consistently positive* they are toward one brand,
using the Wilson score lower bound of their positive-comment share
(rewards both consistency and volume; a 3-for-3 user doesn't outrank
a 40-for-45 user). Comments are attributed to a brand only when they
mention that brand exclusively (brand_context == '*_only'), so VADER
sentiment can't be misattributed in comparison comments.

Reads:  data/scored_comments.parquet
        data/filtered_stories.parquet   (optional — user's submitted stories)
Writes: charts/10_consistency_report.html
        data/consistent_boosters.parquet

Usage:  python src/05_consistency_report.py [--data-dir DIR] [--out FILE]
"""

import argparse
import json
import math
from pathlib import Path

import polars as pl

ROOT = Path(__file__).parent.parent

TOP_N        = 50   # users per brand
MIN_COMMENTS = 5    # minimum brand-exclusive comments to be ranked
MAX_COMMENTS_EMBEDDED = 300  # per user, newest first, to bound HTML size

BRANDS = {
    "anthropic": {"context": "anthropic_only", "label": "Anthropic", "color": "#da7756"},
    "openai":    {"context": "openai_only",    "label": "OpenAI",    "color": "#10a37f"},
}


def wilson_lower_bound(positives: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for a Bernoulli proportion."""
    if n == 0:
        return 0.0
    p = positives / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def rank_brand(comments: pl.DataFrame, context: str) -> pl.DataFrame:
    """Top users by Wilson lower bound of positive share among brand-exclusive comments."""
    sub = comments.filter(pl.col("brand_context") == context)

    stats = (
        sub.group_by("by")
        .agg(
            n=pl.len(),
            positives=(pl.col("sentiment") == "positive").sum(),
            negatives=(pl.col("sentiment") == "negative").sum(),
            avg_compound=pl.col("compound").mean(),
            first_comment=pl.col("time").min(),
            last_comment=pl.col("time").max(),
        )
        .filter(pl.col("n") >= MIN_COMMENTS)
    )

    stats = stats.with_columns(
        pl.struct(["positives", "n"])
        .map_elements(lambda r: wilson_lower_bound(r["positives"], r["n"]), return_dtype=pl.Float64)
        .alias("wilson_lb"),
        (pl.col("positives") / pl.col("n")).alias("pct_positive"),
        (pl.col("negatives") / pl.col("n")).alias("pct_negative"),
    )

    return stats.sort("wilson_lb", descending=True).head(TOP_N)


def user_payload(username: str, comments: pl.DataFrame, stories: pl.DataFrame, context: str) -> dict:
    """All embeddable detail for one user: brand comments + submitted stories."""
    rows = (
        comments.filter((pl.col("by") == username) & (pl.col("brand_context") == context))
        .sort("time", descending=True)
        .head(MAX_COMMENTS_EMBEDDED)
    )
    user_comments = [
        {
            "id": r["id"],
            "time": str(r["time"])[:10],
            "compound": round(r["compound"], 3),
            "sentiment": r["sentiment"],
            "source": r["source"],
            "text": r["text_clean"] or "",
        }
        for r in rows.iter_rows(named=True)
    ]

    user_stories = []
    if stories is not None:
        srows = stories.filter(pl.col("by") == username).sort("time", descending=True)
        user_stories = [
            {
                "id": r["id"],
                "time": str(r["time"])[:10],
                "title": r["title"] or "",
                "score": r["score"],
            }
            for r in srows.iter_rows(named=True)
        ]

    return {"comments": user_comments, "stories": user_stories}


def build_report_data(comments: pl.DataFrame, stories: pl.DataFrame) -> dict:
    data = {}
    for key, cfg in BRANDS.items():
        ranked = rank_brand(comments, cfg["context"])
        users = []
        for r in ranked.iter_rows(named=True):
            detail = user_payload(r["by"], comments, stories, cfg["context"])
            users.append({
                "username": r["by"],
                "n": r["n"],
                "positives": r["positives"],
                "negatives": r["negatives"],
                "pct_positive": round(r["pct_positive"], 3),
                "pct_negative": round(r["pct_negative"], 3),
                "avg_compound": round(r["avg_compound"], 3),
                "wilson_lb": round(r["wilson_lb"], 4),
                "first_comment": str(r["first_comment"])[:10],
                "last_comment": str(r["last_comment"])[:10],
                **detail,
            })
        data[key] = users
        print(f"  {cfg['label']}: {len(users)} users ranked "
              f"(min {MIN_COMMENTS} brand-exclusive comments)")
    return data


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>HN Consistent Boosters — Anthropic vs OpenAI</title>
<style>
  :root {
    --anthropic: #da7756; --openai: #10a37f;
    --bg: #16161a; --panel: #1f1f25; --panel2: #26262e;
    --text: #e8e6e3; --muted: #9a97a0; --border: #34343e;
    --pos: #4caf7d; --neg: #e05c5c; --neu: #8a8a94;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.45 -apple-system, "Segoe UI", Roboto, sans-serif;
         background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; }
  header { padding: 14px 20px; border-bottom: 1px solid var(--border); }
  header h1 { margin: 0; font-size: 17px; }
  header p { margin: 4px 0 0; color: var(--muted); font-size: 12px; }
  .layout { flex: 1; display: flex; min-height: 0; }

  .sidebar { width: 460px; min-width: 360px; border-right: 1px solid var(--border);
             display: flex; flex-direction: column; }
  .tabs { display: flex; }
  .tab { flex: 1; padding: 10px; text-align: center; cursor: pointer;
         border-bottom: 3px solid transparent; font-weight: 600; color: var(--muted); }
  .tab.active-anthropic { color: var(--anthropic); border-color: var(--anthropic); }
  .tab.active-openai    { color: var(--openai);    border-color: var(--openai); }
  .search { padding: 8px 12px; }
  .search input { width: 100%; padding: 7px 10px; border-radius: 6px; border: 1px solid var(--border);
                  background: var(--panel); color: var(--text); }
  .userlist { flex: 1; overflow-y: auto; }
  .user-row { padding: 9px 14px; cursor: pointer; border-bottom: 1px solid var(--border);
              display: flex; align-items: center; gap: 10px; }
  .user-row:hover { background: var(--panel); }
  .user-row.selected { background: var(--panel2); }
  .rank { width: 26px; color: var(--muted); font-size: 12px; text-align: right; flex-shrink: 0; }
  .uinfo { flex: 1; min-width: 0; }
  .uname { font-weight: 600; }
  .ustats { font-size: 11.5px; color: var(--muted); margin-top: 1px; }
  .bar { width: 90px; height: 8px; border-radius: 4px; background: var(--panel2);
         overflow: hidden; flex-shrink: 0; display: flex; }
  .bar .p { background: var(--pos); height: 100%; }
  .bar .x { background: var(--neg); height: 100%; }

  .detail { flex: 1; overflow-y: auto; padding: 18px 24px; min-width: 0; }
  .placeholder { color: var(--muted); margin-top: 40px; text-align: center; }
  .detail h2 { margin: 0 0 2px; font-size: 18px; }
  .detail h2 a { color: inherit; text-decoration: none; border-bottom: 1px dotted var(--muted); }
  .summary { color: var(--muted); font-size: 12.5px; margin-bottom: 14px; }
  .section-title { font-size: 13px; text-transform: uppercase; letter-spacing: .06em;
                   color: var(--muted); margin: 18px 0 8px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
          padding: 10px 13px; margin-bottom: 9px; }
  .card .meta { display: flex; gap: 10px; align-items: center; font-size: 11.5px;
                color: var(--muted); margin-bottom: 6px; flex-wrap: wrap; }
  .badge { padding: 1px 8px; border-radius: 10px; font-weight: 600; font-size: 11px; }
  .badge.positive { background: rgba(76,175,125,.18); color: var(--pos); }
  .badge.negative { background: rgba(224,92,92,.18);  color: var(--neg); }
  .badge.neutral  { background: rgba(138,138,148,.18); color: var(--neu); }
  .badge.off { background: rgba(124,92,191,.2); color: #b39ddb; }
  .card .text { white-space: pre-wrap; word-wrap: break-word; font-size: 13px; }
  .card a.hn { margin-left: auto; color: var(--muted); text-decoration: none; font-size: 11.5px; }
  .card a.hn:hover { color: var(--text); }
  .truncnote { color: var(--muted); font-size: 12px; margin: 6px 0 14px; }
</style>
</head>
<body>
<header>
  <h1>HN Consistent Boosters — top __TOP_N__ most consistently positive users per brand</h1>
  <p>Ranked by Wilson lower bound (95%) of positive share among comments that mention only that brand.
     Minimum __MIN_COMMENTS__ brand-exclusive comments. Click a user to inspect their comments and submissions.</p>
</header>
<div class="layout">
  <div class="sidebar">
    <div class="tabs">
      <div class="tab" id="tab-anthropic" onclick="setBrand('anthropic')">Anthropic (__N_ANTHROPIC__)</div>
      <div class="tab" id="tab-openai" onclick="setBrand('openai')">OpenAI (__N_OPENAI__)</div>
    </div>
    <div class="search"><input id="search" placeholder="Filter usernames…" oninput="renderList()"></div>
    <div class="userlist" id="userlist"></div>
  </div>
  <div class="detail" id="detail">
    <div class="placeholder">Select a user to see their comments and posts.</div>
  </div>
</div>
<script>
const DATA = __DATA_JSON__;
let brand = 'anthropic';
let selected = null;

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function setBrand(b) {
  brand = b; selected = null;
  document.getElementById('search').value = '';
  renderList(); renderDetail();
}

function renderList() {
  const q = document.getElementById('search').value.toLowerCase();
  const list = document.getElementById('userlist');
  document.getElementById('tab-anthropic').className = 'tab' + (brand==='anthropic' ? ' active-anthropic' : '');
  document.getElementById('tab-openai').className    = 'tab' + (brand==='openai'    ? ' active-openai'    : '');
  list.innerHTML = DATA[brand].map((u, i) => {
    if (q && !u.username.toLowerCase().includes(q)) return '';
    const pw = Math.round(u.pct_positive * 100), nw = Math.round(u.pct_negative * 100);
    return `<div class="user-row${u.username===selected?' selected':''}" onclick="select('${esc(u.username).replace(/'/g,"\\\\'")}')">
      <div class="rank">${i+1}</div>
      <div class="uinfo">
        <div class="uname">${esc(u.username)}</div>
        <div class="ustats">${u.positives}/${u.n} positive · avg ${u.avg_compound.toFixed(2)} · WLB ${u.wilson_lb.toFixed(3)}</div>
      </div>
      <div class="bar" title="${pw}% positive / ${nw}% negative"><div class="p" style="width:${pw}%"></div><div class="x" style="width:${nw}%"></div></div>
    </div>`;
  }).join('');
}

function select(name) {
  selected = name;
  renderList(); renderDetail();
}

function renderDetail() {
  const el = document.getElementById('detail');
  if (!selected) { el.innerHTML = '<div class="placeholder">Select a user to see their comments and posts.</div>'; return; }
  const u = DATA[brand].find(x => x.username === selected);
  if (!u) { el.innerHTML = ''; return; }
  const brandLabel = brand === 'anthropic' ? 'Anthropic' : 'OpenAI';

  let html = `<h2><a href="https://news.ycombinator.com/user?id=${encodeURIComponent(u.username)}" target="_blank">${esc(u.username)}</a></h2>
    <div class="summary">${u.n} ${brandLabel}-only comments (${u.positives} positive, ${u.negatives} negative) ·
      avg compound ${u.avg_compound.toFixed(3)} · Wilson LB ${u.wilson_lb.toFixed(4)} ·
      active ${u.first_comment} → ${u.last_comment}</div>`;

  if (u.stories.length) {
    html += `<div class="section-title">Submitted AI stories (${u.stories.length})</div>`;
    html += u.stories.map(s => `<div class="card">
      <div class="meta"><span>${s.time}</span><span>${s.score ?? 0} points</span>
        <a class="hn" href="https://news.ycombinator.com/item?id=${s.id}" target="_blank">view on HN ↗</a></div>
      <div class="text">${esc(s.title)}</div></div>`).join('');
  }

  html += `<div class="section-title">${brandLabel}-only comments (newest first)</div>`;
  if (u.n > u.comments.length) {
    html += `<div class="truncnote">Showing the ${u.comments.length} most recent of ${u.n} comments.</div>`;
  }
  html += u.comments.map(c => `<div class="card">
    <div class="meta">
      <span>${c.time}</span>
      <span class="badge ${c.sentiment}">${c.sentiment} ${c.compound.toFixed(2)}</span>
      ${c.source === 'off_topic_mention' ? '<span class="badge off">off-topic thread</span>' : ''}
      <a class="hn" href="https://news.ycombinator.com/item?id=${c.id}" target="_blank">view on HN ↗</a>
    </div>
    <div class="text">${esc(c.text)}</div></div>`).join('');

  el.innerHTML = html;
  el.scrollTop = 0;
}

renderList();
</script>
</body>
</html>
"""


def render_html(data: dict, out_path: Path):
    # "</" inside embedded strings would terminate the <script> block
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = (
        HTML_TEMPLATE
        .replace("__DATA_JSON__", payload)
        .replace("__TOP_N__", str(TOP_N))
        .replace("__MIN_COMMENTS__", str(MIN_COMMENTS))
        .replace("__N_ANTHROPIC__", str(len(data["anthropic"])))
        .replace("__N_OPENAI__", str(len(data["openai"])))
    )
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "charts" / "10_consistency_report.html")
    args = ap.parse_args()

    scored = args.data_dir / "scored_comments.parquet"
    if not scored.exists():
        raise FileNotFoundError(f"Run 01–02 first — {scored} not found")

    comments = pl.read_parquet(scored)
    stories_path = args.data_dir / "filtered_stories.parquet"
    stories = pl.read_parquet(stories_path) if stories_path.exists() else None
    print(f"Loaded {len(comments):,} scored comments")

    print("Ranking consistent boosters...")
    data = build_report_data(comments, stories)

    # Flat ranking table for downstream use
    flat = pl.DataFrame([
        {k: u[k] for k in ("username", "n", "positives", "negatives", "pct_positive",
                           "pct_negative", "avg_compound", "wilson_lb",
                           "first_comment", "last_comment")} | {"brand": b}
        for b in BRANDS for u in data[b]
    ])
    flat_out = args.data_dir / "consistent_boosters.parquet"
    flat.write_parquet(flat_out)
    print(f"Saved {flat_out} ({len(flat)} rows)")

    render_html(data, args.out)


if __name__ == "__main__":
    main()
