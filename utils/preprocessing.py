"""User-input preprocessing: file-path extraction and guardrail classification.

These functions run *before* the LLM to make the system more reliable:
- :func:`extract_paths` deterministically pulls file paths out of user text.
- :func:`is_reconciliation_related` is a fast keyword check that can short-
  circuit off-topic queries without wasting an LLM call.
"""

import re
from pathlib import Path

# File extensions the agent can handle.
VALID_EXTENSIONS: frozenset[str] = frozenset({
    ".xlsx", ".xls", ".csv", ".json", ".txt", ".parquet",
})

# Pattern for a quoted or bare path ending in a valid extension.
_PATH_PATTERN = re.compile(
    r"""
    (?:"([^"]+)"|'([^']+)'|`([^`]+)`|         # quoted paths
     (file:///[^\s,;)]+)                      # file:// URIs
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_BARE_PATH_PATTERN = re.compile(
    r"""
    (?:^|[\s:;,])
    (
        (?:[^\s"']*\\)?[^\s"']*
        \.(?:xlsx|xls|csv|json|txt|parquet)
    )
    (?:$|[\s:;,])
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Keywords that strongly indicate a reconciliation task.
_RECON_KEYWORDS: frozenset[str] = frozenset({
    "reconcile", "compare", "match", "diff", "difference",
    "source", "target", "file", "xlsx", "csv", "json", "parquet",
    "analyze", "report",
})

# Keywords that clearly indicate off-topic queries (no reconciliation intent).
_OFFTOPIC_KEYWORDS: frozenset[str] = frozenset({
    "weather", "cook", "recipe", "poem", "code", "write a",
    "color", "paint", "song", "movie", "game", "sport",
})

POLITE_DECLINE: str = (
    "I am a Data Integrity Test Agent. I can only help with reconciling "
    "data files (XLSX, CSV, JSON, TXT, Parquet). Please provide Source "
    "and Target file paths for reconciliation."
)


def extract_paths(text: str) -> list[str]:
    """Extract file paths from *text* using regex patterns.

    Returns a list of unique, deduplicated path strings in the order they
    were first seen.  Only paths ending in a supported extension
    (``.xlsx``, ``.csv``, ``.json``, ``.txt``, ``.parquet``) are returned.
    """
    seen: set[str] = set()
    paths: list[str] = []

    def _add(p: str) -> None:
        p_stripped = p.strip().strip("\"'`")
        ext = Path(p_stripped).suffix.lower()
        if ext not in VALID_EXTENSIONS:
            return
        if p_stripped not in seen:
            seen.add(p_stripped)
            paths.append(p_stripped)

    for match in _PATH_PATTERN.finditer(text):
        for group in match.groups():
            if group:
                _add(group)

    for match in _BARE_PATH_PATTERN.finditer(text):
        _add(match.group(1))

    return paths


def is_reconciliation_related(text: str) -> bool:
    """Quick keyword-based classification of user intent.

    Returns ``True`` if the text appears reconciliation-related, ``False``
    if it appears off-topic, and ``True`` for ambiguous input (pass through
    to the LLM).  Reconciliation keywords take priority over off-topic ones.
    """
    lower = text.lower().strip()

    # Strong reconciliation signal — takes priority over off-topic.
    if any(kw in lower for kw in _RECON_KEYWORDS):
        return True

    # Has file-like paths
    if extract_paths(text):
        return True

    # Strong off-topic signal
    if any(kw in lower for kw in _OFFTOPIC_KEYWORDS):
        return False

    # Ambiguous — pass through to LLM
    return True
