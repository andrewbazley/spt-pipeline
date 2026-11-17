"""Shared utilities for the SPT pipeline v2.

Small, dependency-light helpers used across modules: column resolution,
path discovery, condition/experiment inference, and JSON serialization.
Keeping these in one module prevents the silent divergence that plagued the
three-script era, where the same helper was copied and edited independently.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------

def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first column matching any candidate.

    Matching is exact first, then case-insensitive, then substring. This
    tolerates the many header conventions produced by trackpy, TrackMate, and
    hand-edited CSVs without silently grabbing the wrong column.
    """
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in low:
            return low[cand.lower()]
        for c in cols:
            if cand.lower() in c.lower():
                return c
    return None


# Column-name aliases so trajectory tables from different trackers (trackpy,
# MOSAIC ParticleTracker, TrackMate, hand-edited CSVs) all normalize to the
# canonical ``particle/frame/x/y/mass`` schema the vectorized code expects.
# Matching is exact first, then case-insensitive; substring matching is
# deliberately avoided so columns like ``mean_x_px`` are never mistaken for x.
TRACK_COLUMN_ALIASES = {
    "particle": ("particle", "Trajectory", "trajectory", "track", "track_id",
                 "trackid", "traj", "TRACK_ID"),
    "frame":    ("frame", "Frame", "FRAME", "t", "slice", "frame_idx", "frame_number"),
    "x":        ("x", "X", "x_px", "POSITION_X", "pos_x"),
    "y":        ("y", "Y", "y_px", "POSITION_Y", "pos_y"),
    "mass":     ("mass", "m0", "total_intensity", "intensity"),
}


def canonicalize_track_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename tracker-specific columns to the canonical schema (non-destructive).

    Only renames when the canonical name is absent and an alias is present, so
    trackpy tables (already ``particle/frame/x/y``) pass through untouched. This
    is what lets the pipeline consume MOSAIC ``Traj_*.csv`` (``Trajectory``,
    ``Frame``, ``x``, ``y``) without a separate reader.
    """
    cols = list(df.columns)
    lower = {}
    for c in cols:
        lower.setdefault(str(c).lower(), c)
    rename = {}
    for canon, aliases in TRACK_COLUMN_ALIASES.items():
        if canon in cols:
            continue
        for a in aliases:
            if a in cols and a not in rename:
                rename[a] = canon
                break
            src = lower.get(a.lower())
            if src is not None and src not in rename and src != canon:
                rename[src] = canon
                break
    # Never collide with an existing canonical column.
    rename = {k: v for k, v in rename.items() if v not in df.columns}
    return df.rename(columns=rename) if rename else df


def normalize_track_table(track_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten any index and coerce core numeric columns.

    trackpy frequently returns tables indexed by ``frame`` (and keeps ``frame``
    as a column too, e.g. after ``filter_stubs``), or by ``(frame, particle)``.
    Naively calling ``reset_index()`` then raises "cannot insert frame, already
    exists". We therefore drop any index level that duplicates a column and
    promote the rest, leaving a flat integer-indexed table for the vectorized
    code downstream.
    """
    out = track_df.copy()
    idx_names = [n for n in out.index.names if n is not None]
    # Drop index levels already present as columns (avoids the collision).
    dup_levels = [n for n in idx_names if n in out.columns]
    if dup_levels:
        out = out.reset_index(level=dup_levels, drop=True)
    # Promote any remaining named levels into columns.
    remaining = [n for n in out.index.names if n is not None]
    if remaining:
        out = out.reset_index()
    else:
        out = out.reset_index(drop=True)
    out = canonicalize_track_columns(out)
    for col in ("frame", "particle", "x", "y", "mass"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------

IMAGE_EXTS = (".nd2", ".tif", ".tiff")


def should_ignore_path(path_like) -> bool:
    """True for hidden / AppleDouble / macOS bundle paths."""
    p = Path(path_like)
    return any(
        part.startswith(".") or part.startswith("._") or part == "__MACOSX"
        for part in p.parts
    )


def rglob_filtered(root: Path, pattern: str) -> List[Path]:
    """Recursive glob skipping hidden/AppleDouble files, sorted for determinism."""
    return sorted(
        (p for p in Path(root).rglob(pattern) if p.is_file() and not should_ignore_path(p)),
        key=lambda p: str(p),
    )


def discover_image_files(root: Path, exclude_patterns: Optional[Sequence[str]] = None) -> List[Path]:
    """Find all supported image files under ``root``.

    ``exclude_patterns`` are case-insensitive substrings matched against the
    file name (e.g. ``bead``, ``flatfield``) to skip calibration movies.
    """
    exclude = [e.lower() for e in (exclude_patterns or [])]
    files: List[Path] = []
    for ext in IMAGE_EXTS:
        files.extend(rglob_filtered(root, f"*{ext}"))
    keep = []
    for p in sorted({p for p in files}, key=lambda p: str(p)):
        name = p.name.lower()
        if any(tok in name for tok in exclude):
            continue
        keep.append(p)
    return keep


def discover_track_files(root: Path) -> List[Path]:
    """Find all ``*_tracks.csv`` files under ``root`` (for --from-tracks scans)."""
    return rglob_filtered(Path(root), "*_tracks.csv")


def movie_id_from_path(path: Path) -> str:
    """Derive a stable movie id from a file stem, stripping known suffixes."""
    stem = Path(path).stem
    for suffix in ("_tracks", "_particles", "_diffusion_coefficients"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


# ---------------------------------------------------------------------------
# Condition / experiment inference
# ---------------------------------------------------------------------------

DEFAULT_CONDITION_TOKENS = [
    "NT", "Tg", "DMSO", "untreated", "treated", "control", "stress",
    "BiP", "WT", "KO", "mock", "vehicle",
]

_REPLICATE_RE = re.compile(r"(?:^|[_\-])(?:rep|replicate|r|movie|mov|fov|pos|cell|s)?\d+$", re.IGNORECASE)


def load_groupnames_condition_map(root: Path) -> Dict[str, str]:
    """Read a legacy ``groupnames.csv`` (folder,condition) if present."""
    group_file = Path(root) / "groupnames.csv"
    if not group_file.exists() or should_ignore_path(group_file):
        return {}
    try:
        gdf = pd.read_csv(group_file, low_memory=False)
    except Exception:
        return {}
    if not {"folder", "condition"}.issubset(gdf.columns):
        return {}
    out: Dict[str, str] = {}
    for _, row in gdf.iterrows():
        folder = str(row["folder"]).strip()
        if folder.endswith("_tracks"):
            folder = folder[:-7]
        condition = str(row["condition"]).strip()
        if folder and condition:
            out[folder] = condition
    return out


def infer_condition(name: str, tokens: Optional[Sequence[str]] = None) -> Optional[str]:
    """Match a filename/folder against condition tokens (longest token wins)."""
    tokens = list(tokens) if tokens else DEFAULT_CONDITION_TOKENS
    text = str(name)
    hits = []
    for tok in tokens:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(tok)}(?![A-Za-z0-9])", text, re.IGNORECASE):
            hits.append(tok)
    if not hits:
        return None
    return sorted(hits, key=len, reverse=True)[0]


_DATE_PATTERNS = [
    (re.compile(r"(20\d{2})[_\-.]?(\d{2})[_\-.]?(\d{2})"), (1, 2, 3)),   # 2025-04-08 / 20250408
    (re.compile(r"(?<!\d)(\d{2})[_\-.](\d{2})[_\-.](20\d{2})"), (3, 1, 2)),  # 04_08_2025
    (re.compile(r"(?<!\d)(\d{2})[_\-](\d{2})(?!\d)"), (None, 1, 2)),      # experiment_04_08 (month_day)
]


def infer_experiment(name: str, default_year: Optional[int] = None) -> Optional[str]:
    """Parse an ISO date (YYYY-MM-DD) from a folder/file name if possible."""
    text = str(name)
    for regex, order in _DATE_PATTERNS:
        m = regex.search(text)
        if not m:
            continue
        yi, mi, di = order
        year = m.group(yi) if yi else (str(default_year) if default_year else None)
        if year is None:
            continue
        month, day = m.group(mi), m.group(di)
        try:
            dt = datetime(int(year), int(month), int(day))
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return None


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy / Path objects into JSON-serializable types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        # Sorted so run-parameter JSONs are byte-stable across runs and can be
        # diffed to see what actually changed between two analyses.
        return sorted(to_jsonable(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def dump_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(to_jsonable(obj), fh, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Misc numeric helpers
# ---------------------------------------------------------------------------

def nan_median(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmedian(arr)) if arr.size else float("nan")


def nan_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def resolve_n_jobs(requested: Optional[int]) -> int:
    """Clamp a requested worker count to the available CPUs (>=1)."""
    import os
    cpu = os.cpu_count() or 1
    if not requested or requested <= 0:
        return max(1, min(4, cpu))
    return max(1, min(int(requested), cpu))
