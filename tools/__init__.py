"""tools package.

Exposes a module-level :class:`SessionContext` instance named ``SESSION_STATE``
that tools use to share data across the reconciliation workflow. The context is
a typed dataclass (not a bare ``dict``) so that field access is explicit and
type-checkable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    align_meta: Optional[dict] = None

    # File metadata
    source_filename: Optional[str] = None
    target_filename: Optional[str] = None
    source_fullpath: Optional[str] = None
    target_fullpath: Optional[str] = None

    # Reconciliation inputs
    detected_keys: Optional[list] = None
    primary_key_cols: Optional[list] = None
    tolerance_settings: Optional[dict] = None

    # Reconciliation output
    reconciliation_results: Optional[dict] = None

    # UI state
    _recon_table_shown: bool = False

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

