import re

ANTHROPIC_TERMS = [
    "claude", "anthropic", "opus", "sonnet", "haiku",
    "claude code", "claude 3", "claude 4",
]

OPENAI_TERMS = [
    "openai", "open ai", "chatgpt", "chat gpt",
    "gpt-4", "gpt4", "gpt-3", "gpt3", "gpt-4o", "gpt4o",
    "codex", "o1", "o3", "o3-mini", "o1-mini",
    "dall-e", "dalle", "sora", "oai",
]

ALL_TERMS = ANTHROPIC_TERMS + OPENAI_TERMS

# Compiled regexes for fast matching
_ANTHROPIC_PAT = r'\b(?:' + '|'.join(re.escape(t) for t in ANTHROPIC_TERMS) + r')\b'
_OPENAI_PAT    = r'\b(?:' + '|'.join(re.escape(t) for t in OPENAI_TERMS) + r')\b'

ANTHROPIC_RE = re.compile(_ANTHROPIC_PAT, re.IGNORECASE)
OPENAI_RE    = re.compile(_OPENAI_PAT,    re.IGNORECASE)

# DuckDB LIKE conditions (one string to drop into SQL)
def _like_clauses(col: str, terms: list[str]) -> str:
    return ' OR '.join(f"lower({col}) LIKE '%{t}%'" for t in terms)

STORY_TITLE_FILTER  = _like_clauses('title', ALL_TERMS)
COMMENT_TEXT_FILTER = _like_clauses('text',  ALL_TERMS)

# Key product launch dates for event overlays on charts
LAUNCH_EVENTS = [
    ("2022-11-30", "ChatGPT launch"),
    ("2023-03-14", "GPT-4 launch"),
    ("2023-03-14", "Claude 1 launch"),
    ("2023-07-11", "Claude 2 launch"),
    ("2024-03-04", "Claude 3 launch"),
    ("2024-05-13", "GPT-4o launch"),
    ("2024-06-20", "Claude 3.5 Sonnet"),
    ("2025-02-24", "Claude 3.7 Sonnet"),
    ("2025-05-22", "Claude 4 / Claude Code launch"),
    ("2025-05-16", "OpenAI Codex launch"),
]
