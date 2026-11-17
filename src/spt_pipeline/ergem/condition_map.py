"""Read a FICD ER-GEM condition map (JSON) into the core manifest schema.

The LUMEN organisational pipeline emits ``*_condition_map.json`` with the shape::

    conditions[] -> replicates[] -> movies[]

Each movie already carries the located ``trajectory_csv`` (MOSAIC ``Traj_*.csv``,
most-recent version), ``raw_nd2``, ``ficd_marker_nd2`` and ``clone``; each
replicate carries ``experiment_date`` and an ``analysis_provenance`` block with
the FICD expression table. We flatten this to one manifest row per movie so the
existing pipeline phases operate unchanged, adding a few factor columns
(genotype/complementation/treatment/clone/...) used by the FICD contrasts.

Calibration (``mpp``/``fps``) is resolved per movie in this order:
  1. an explicit value in the JSON (future LUMEN field), then
  2. acquisition metadata read from the movie's ND2/TIF (preferred over user
     entry, which can be fallible), then
  3. the supplied defaults (0.11 um/px, 100 fps for this campaign).
If none resolves and no positive default is given, the value is left blank and
core ``manifest.validate`` will refuse to proceed with an actionable message.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .. import manifest as core_manifest
from . import DEFAULT_MPP, DEFAULT_FPS, EXTRA_COLUMNS

ALL_COLUMNS = list(core_manifest.COLUMNS) + EXTRA_COLUMNS


def load_condition_map(path: Path) -> Dict[str, Any]:
    """Load and lightly validate a condition-map JSON."""
    path = Path(path)
    with open(path) as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "conditions" not in data:
        raise ValueError(f"{path.name} is not a condition map (missing 'conditions').")
    return data


def _first(*vals):
    for v in vals:
        if v not in (None, ""):
            return v
    return None


def _resolve_calibration(
    movie: Dict[str, Any],
    replicate: Dict[str, Any],
    default_mpp: Optional[float],
    default_fps: Optional[float],
    read_metadata: bool,
) -> Dict[str, Optional[float]]:
    """Return {'mpp':..,'fps':..} using JSON -> file metadata -> defaults."""
    # 1) explicit JSON fields, if a future map provides them
    mpp = _first(movie.get("mpp"), movie.get("microns_per_pixel"),
                 replicate.get("mpp"))
    fps = _first(movie.get("fps"), movie.get("frames_per_second"),
                 replicate.get("fps"))
    # 2) acquisition metadata from the movie file (GEM stream preferred)
    if read_metadata and (mpp is None or fps is None):
        for key in ("raw_nd2", "trajectory_csv", "ficd_marker_nd2"):
            p = movie.get(key)
            if not p:
                continue
            fp = Path(p)
            if fp.suffix.lower() not in (".nd2", ".tif", ".tiff") or not fp.exists():
                continue
            meta = core_manifest._extract_image_metadata(fp)
            mpp = mpp if mpp is not None else meta.get("mpp")
            fps = fps if fps is not None else meta.get("fps")
            if mpp is not None and fps is not None:
                break
    # 3) campaign defaults
    if mpp is None:
        mpp = default_mpp
    if fps is None:
        fps = default_fps
    return {"mpp": mpp, "fps": fps}


def iter_movie_rows(
    cmap: Dict[str, Any],
    default_mpp: Optional[float] = DEFAULT_MPP,
    default_fps: Optional[float] = DEFAULT_FPS,
    read_metadata: bool = True,
    correlation_only: bool = False,
    require_trajectory: bool = True,
) -> List[Dict[str, Any]]:
    """Flatten conditions->replicates->movies into manifest-row dicts."""
    rows: List[Dict[str, Any]] = []
    for cond in cmap.get("conditions", []):
        cond_id = cond.get("condition_id")
        genotype = cond.get("genotype")
        compl = cond.get("complementation")
        treat = cond.get("treatment_code") or cond.get("treatment")
        for rep in cond.get("replicates", []):
            exp = rep.get("experiment_date") or rep.get("replicate_id")
            prov = rep.get("analysis_provenance", {}) or {}
            expr_table = prov.get("ficd_expression_measurement_table") or ""
            for mv in rep.get("movies", []):
                traj = mv.get("trajectory_csv") or ""
                if require_trajectory and not traj:
                    continue
                has_red = bool(mv.get("has_paired_red_image"))
                if correlation_only and not has_red:
                    continue
                cal = _resolve_calibration(mv, rep, default_mpp, default_fps, read_metadata)
                rows.append({
                    "movie_id": mv.get("movie_id"),
                    "source_path": mv.get("raw_nd2") or traj,
                    "file_format": "mosaic_traj",
                    "condition": cond_id,
                    "experiment": exp,
                    "tracks_csv": traj,
                    "diff_csv": "",
                    "mpp": cal["mpp"] if cal["mpp"] is not None else "",
                    "fps": cal["fps"] if cal["fps"] is not None else "",
                    "n_frames": "",
                    "n_channels": "",
                    "notes": "",
                    # --- FICD factor columns (additive) ---
                    "genotype": genotype or "",
                    "complementation": compl or "",
                    "treatment_code": treat or "",
                    "clone": mv.get("clone") or "",
                    "has_paired_red": bool(has_red),
                    "ficd_expr_table": expr_table,
                    "ficd_marker_nd2": mv.get("ficd_marker_nd2") or "",
                    "condition_id": cond_id,
                    "replicate_id": rep.get("replicate_id") or "",
                })
    return rows


def to_manifest(
    cmap: Dict[str, Any],
    default_mpp: Optional[float] = DEFAULT_MPP,
    default_fps: Optional[float] = DEFAULT_FPS,
    read_metadata: bool = True,
    correlation_only: bool = False,
) -> pd.DataFrame:
    """Build a manifest DataFrame (core columns + FICD factor columns)."""
    rows = iter_movie_rows(cmap, default_mpp, default_fps, read_metadata, correlation_only)
    df = pd.DataFrame(rows, columns=ALL_COLUMNS)
    # De-duplicate movie ids defensively (the map should already be unique).
    if not df.empty and df["movie_id"].duplicated().any():
        seen: Dict[str, int] = {}
        new = []
        for mid in df["movie_id"]:
            if mid in seen:
                seen[mid] += 1
                new.append(f"{mid}__{seen[mid]}")
            else:
                seen[mid] = 0
                new.append(mid)
        df["movie_id"] = new
    return df


def from_json(
    json_path: Path,
    out_manifest: Optional[Path] = None,
    default_mpp: Optional[float] = DEFAULT_MPP,
    default_fps: Optional[float] = DEFAULT_FPS,
    read_metadata: bool = True,
    correlation_only: bool = False,
) -> pd.DataFrame:
    """Load a condition map and (optionally) write a manifest CSV."""
    cmap = load_condition_map(json_path)
    df = to_manifest(cmap, default_mpp, default_fps, read_metadata, correlation_only)
    if out_manifest is not None:
        core_manifest.write(df, Path(out_manifest))
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per condition: replicate/movie/marker counts (for a scan log)."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("condition")
    out = pd.DataFrame({
        "n_replicates": g["experiment"].nunique(),
        "n_movies": g.size(),
        "n_with_paired_red": g["has_paired_red"].apply(lambda s: int(pd.Series(s).astype(bool).sum())),
        "clones": g["clone"].apply(lambda s: ",".join(sorted({x for x in s if x}))),
    }).reset_index()
    return out
