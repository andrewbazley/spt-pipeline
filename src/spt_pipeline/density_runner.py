"""Measure density for a whole manifest, independent of tracking.

Two entry points into the same machinery:

* **After tracking**, when ``{movie}_detections.csv.gz`` exists, density is
  re-measured from the pre-link detections. Changing the field radius costs
  seconds and needs no re-tracking.
* **On data tracked before this existed**, there is no detection table, so
  density falls back to the linked tracks. Every row is stamped
  ``density_source = "post_link"``.

The fallback is not equivalent, and the difference is one-directional.
Linking discards detections it cannot join and stub filtering discards short
tracks; both happen most in crowded fields. So post-link density **understates
crowding exactly where crowding matters most**, which makes a dense cell look
more like a sparse one and weakens the correction rather than inventing one.
That is the safer direction to be wrong in, but it is still wrong, and the flag
travels with every row so no figure can quietly imply otherwise.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import density as density_mod
from . import manifest as manifest_mod
from . import utils

#: Per-movie outputs.
PER_TRACK_SUFFIX = "_density_per_track.csv"
PER_CELL_SUFFIX = "_density_per_cell.csv"
DETECTIONS_SUFFIX = "_detections.csv.gz"


def detections_path(tracks_csv: Path, movie_id: str) -> Path:
    return Path(tracks_csv).with_name(f"{movie_id}{DETECTIONS_SUFFIX}")


def load_detections(tracks_csv: Path, movie_id: str) -> Optional[pd.DataFrame]:
    """The saved pre-link detection table, or None if tracking never wrote one."""
    p = detections_path(tracks_csv, movie_id)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except (OSError, pd.errors.ParserError):
        return None
    return df if not df.empty else None


def measure_manifest(
    manifest_df: pd.DataFrame, *, field_radius_px: float = density_mod.DEFAULT_FIELD_RADIUS_PX,
    default_mpp: float = 0.183, default_search_range_px: float = 5.0,
    segment: bool = True, force: bool = False, progress: bool = True,
) -> Dict[str, Any]:
    """Measure density for every movie with tracks; write per-movie tables."""
    from .compare import _tracking_context

    ctx = _tracking_context(manifest_df)
    ctx_range = ctx.get("search_range")
    results: List[Dict[str, Any]] = []
    n_pre = n_post = n_skip = n_fail = 0

    for _, row in manifest_df.iterrows():
        if not manifest_mod.has_tracks(row):
            continue
        movie_id = str(row["movie_id"])
        tracks_csv = Path(str(row["tracks_csv"]))
        out_csv = tracks_csv.with_name(f"{movie_id}{PER_TRACK_SUFFIX}")
        if out_csv.exists() and not force:
            n_skip += 1
            continue

        mpp, _fps = manifest_mod.row_mpp_fps(row, default_mpp, 50.0)
        search_range = float(_movie_search_range(tracks_csv, movie_id)
                             or ctx_range or default_search_range_px)
        try:
            tracks = utils.normalize_track_table(pd.read_csv(tracks_csv, low_memory=False))
            det = load_detections(tracks_csv, movie_id)
            res = density_mod.measure_movie(
                det, tracks, search_range_px=search_range, mpp=mpp,
                movie_id=movie_id, field_radius_px=field_radius_px, segment=segment)
        except Exception as exc:
            n_fail += 1
            results.append({"movie_id": movie_id, "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}"})
            continue

        per_track = res["per_track"]
        if per_track.empty:
            n_fail += 1
            results.append({"movie_id": movie_id, "status": "failed",
                            "error": "no per-track density produced"})
            continue
        per_track.to_csv(out_csv, index=False)
        cells = res["cells"]
        if isinstance(cells, pd.DataFrame) and not cells.empty:
            cells.to_csv(tracks_csv.with_name(f"{movie_id}{PER_CELL_SUFFIX}"), index=False)

        seg = res.get("segmentation")
        if res["density_source"] == "pre_link":
            n_pre += 1
        else:
            n_post += 1
        results.append({
            "movie_id": movie_id, "status": "ok",
            "density_source": res["density_source"],
            "n_tracks": int(len(per_track)),
            "n_cells": int(getattr(seg, "n_cells", 0)),
            "cells_resolved": bool(getattr(seg, "resolved", False)),
            "segmentation_note": getattr(seg, "note", ""),
            "step_scale_px": res["step_scale_px"],
            "search_range_px": search_range,
            "median_link_safety_ratio": float(
                pd.to_numeric(per_track["link_safety_ratio"], errors="coerce").median()),
        })
        if progress:
            print(f"  [{res['density_source']:>9}] {movie_id}: "
                  f"{len(per_track)} tracks, {getattr(seg, 'n_cells', 0)} cell(s)")

    return {"results": results, "n_pre_link": n_pre, "n_post_link": n_post,
            "n_cached": n_skip, "n_failed": n_fail,
            "field_radius_px": float(field_radius_px)}


def _movie_search_range(tracks_csv: Path, movie_id: str) -> Optional[float]:
    """The linking search range this movie was actually tracked with."""
    cj = Path(tracks_csv).with_name(f"{movie_id}_context.json")
    if not cj.exists():
        return None
    d = utils.load_json(cj)
    v = (d.get("params") or {}).get("search_range")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def load_pool(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Every movie's per-track density table, labelled by condition."""
    frames = []
    for _, row in manifest_df.iterrows():
        if not manifest_mod.has_tracks(row):
            continue
        movie_id = str(row["movie_id"])
        p = Path(str(row["tracks_csv"])).with_name(f"{movie_id}{PER_TRACK_SUFFIX}")
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except (OSError, pd.errors.ParserError):
            continue
        if df.empty:
            continue
        df["movie_id"] = movie_id
        df["condition"] = str(row["condition"])
        df["experiment"] = str(row.get("experiment", ""))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_cell_pool(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Every movie's per-cell table, labelled by condition."""
    frames = []
    for _, row in manifest_df.iterrows():
        if not manifest_mod.has_tracks(row):
            continue
        movie_id = str(row["movie_id"])
        p = Path(str(row["tracks_csv"])).with_name(f"{movie_id}{PER_CELL_SUFFIX}")
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except (OSError, pd.errors.ParserError):
            continue
        if df.empty:
            continue
        df["condition"] = str(row["condition"])
        df["experiment"] = str(row.get("experiment", ""))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
