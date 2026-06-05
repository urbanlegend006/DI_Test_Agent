"""tools package.

Exposes a module-level :class:`SessionContext` instance named ``SESSION_STATE``
that tools use to share data across the reconciliation workflow. The context is
a typed dataclass (not a bare ``dict``) so that field access is explicit and
type-checkable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


@dataclass
class SessionContext:
    """Typed context shared between tools during a reconciliation session.

    All fields default to ``None`` / empty so freshly created contexts are
    cheap and safe to use as a sentinel "no data" state.
    """

    # DataFrames
    source_df: Optional[pd.DataFrame] = None
    target_df: Optional[pd.DataFrame] = None
    comp_source: Optional[pd.DataFrame] = None
    comp_target: Optional[pd.DataFrame] = None

    # Column-alignment metadata
    align_meta: Optional[dict[str, Any]] = None

    # File metadata
    source_filename: Optional[str] = None
    target_filename: Optional[str] = None
    source_fullpath: Optional[str] = None
    target_fullpath: Optional[str] = None

    # Reconciliation inputs
    detected_keys: Optional[list[str]] = None
    key_profile: Optional[list[dict[str, Any]]] = None
    primary_key_cols: Optional[list[str]] = None
    tolerance_settings: Optional[dict[str, Any]] = None

    # Reconciliation output
    reconciliation_results: Optional[dict[str, Any]] = None
    report_path: Optional[str] = None
    report_format: Optional[str] = None

    # Workflow state
    workflow_state: str = "awaiting_paths"

    # UI state
    _recon_table_shown: bool = False

    # Session tracking (populated by agent.py)
    session_thread_id: str = ""
    _turn_count: int = 0
    _session_generation: int = 0
    _extracted_paths: Optional[list[str]] = None

    def reset(self) -> None:
        """Reset every field back to its declared default value.

        This creates a fresh :class:`SessionContext` and copies its field values
        into ``self``. Equivalent to a full re-initialization, but avoids
        breaking references held by callers.
        """
        defaults = SessionContext()
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(defaults, f))


# Module-level instance shared by all tools and the CLI.
SESSION_STATE = SessionContext()

