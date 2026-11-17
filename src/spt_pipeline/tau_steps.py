"""Phase B: fixed-tau displacement engine.

This is the correctness centerpiece of v2. For a chosen lag ``tau`` (in
frames), every frame ``f`` in a track is paired with frame ``f + tau`` *if both
have detections*, yielding the sliding-window displacement distribution -- the
published-style step-size object (Object 2 in the design doc).

The computation is a single vectorized self-merge-join per tau (the approach
proven in the base pipeline's compare-conditions), which is ~10-50x faster than
the per-particle Python loop it replaces, and correct by construction: a
displacement exists at tau iff both endpoints were detected.

Key efficiency properties:

* Tracks CSV is read **once** per movie; all requested taus computed in a loop
  over the in-memory table.
* Per-movie, per-tau CSVs are cached; ``skip-if-exists`` means adding a new tau
  only computes that tau.
* Movies are independent -> trivially parallel across processes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import utils

TAU_STEP_COLUMNS = [
    "particle", "frame_a", "frame_b", "tau_frames", "tau_s",
    "dx_px", "dy_px", "step_px", "step_um",
]


def tau_filename(movie_id: str, tau_frames: int) -> str:
    return f"{movie_id}_tau_steps_{int(tau_frames):03d}.csv"


def compute_tau_displacements(
    tracks: pd.DataFrame,
    tau_frames: int,
    mpp: float,
    fps: float,
    particles: Optional[Iterable] = None,
    non_overlapping: bool = False,
) -> pd.DataFrame:
    """Vectorized fixed-tau displacements for a single track table.

    Parameters
    ----------
    tracks : DataFrame with ``particle``, ``frame``, ``x``, ``y`` columns.
    tau_frames : lag in frames (>= 1).
    mpp, fps : microns-per-pixel and frames-per-second for physical units.
    particles : optional whitelist of particle ids to keep (track filtering).
    non_overlapping : if True, keep only start frames that are multiples of tau
        within each track's first frame, giving statistically independent
        samples at reduced yield.

    Returns a DataFrame with :data:`TAU_STEP_COLUMNS`.
    """
    tau_frames = int(tau_frames)
    if tau_frames < 1:
        raise ValueError("tau_frames must be >= 1")

    t = utils.normalize_track_table(tracks)
    needed = {"particle", "frame", "x", "y"}
    if not needed.issubset(t.columns) or t.empty:
        return pd.DataFrame(columns=TAU_STEP_COLUMNS)

    t = t.dropna(subset=["particle", "frame", "x", "y"]).copy()
    t["frame_int"] = t["frame"].round().astype(np.int64)
    if particles is not None:
        keep = set(particles)
        t = t[t["particle"].isin(keep)]
    if t.empty:
        return pd.DataFrame(columns=TAU_STEP_COLUMNS)

    a = t[["particle", "frame_int", "x", "y"]].copy()
    a["frame_target"] = a["frame_int"] + tau_frames
    b = t[["particle", "frame_int", "x", "y"]].rename(
        columns={"x": "x_b", "y": "y_b", "frame_int": "frame_b"})

    merged = a.merge(
        b, left_on=["particle", "frame_target"], right_on=["particle", "frame_b"],
        how="inner", copy=False,
    )
    if merged.empty:
        return pd.DataFrame(columns=TAU_STEP_COLUMNS)

    dx = merged["x_b"].to_numpy() - merged["x"].to_numpy()
    dy = merged["y_b"].to_numpy() - merged["y"].to_numpy()
    step_px = np.sqrt(dx * dx + dy * dy)

    out = pd.DataFrame({
        "particle": merged["particle"].to_numpy(),
        "frame_a": merged["frame_int"].to_numpy(),
        "frame_b": merged["frame_b"].to_numpy(),
        "tau_frames": tau_frames,
        "tau_s": tau_frames / float(fps),
        "dx_px": dx,
        "dy_px": dy,
        "step_px": step_px,
        "step_um": step_px * float(mpp),
    })

    if non_overlapping:
        # Independent samples: keep start frames on a tau-grid anchored at each
        # track's first frame.
        first = out.groupby("particle")["frame_a"].transform("min")
        keep = ((out["frame_a"] - first) % tau_frames) == 0
        out = out[keep]

    return out.sort_values(["particle", "frame_a"]).reset_index(drop=True)[TAU_STEP_COLUMNS]


def _surviving_particles(diff_csv: Optional[Path], min_nspots: int, min_alpha: Optional[float]) -> Optional[set]:
    """Return the set of particle ids passing filters, or None if no filtering."""
    if not diff_csv or (min_nspots <= 0 and min_alpha is None):
        return None
    try:
        d = pd.read_csv(diff_csv, low_memory=False)
    except (OSError, pd.errors.ParserError):
        return None
    if "particle" not in d.columns:
        return None
    mask = pd.Series(True, index=d.index)
    if min_nspots > 0 and "n_spots" in d.columns:
        mask &= pd.to_numeric(d["n_spots"], errors="coerce").fillna(0) >= min_nspots
    if min_alpha is not None and "alpha" in d.columns:
        mask &= pd.to_numeric(d["alpha"], errors="coerce").fillna(-np.inf) >= min_alpha
    return set(d.loc[mask, "particle"].tolist())


def process_movie(
    movie_id: str,
    tracks_csv: Path,
    tau_list: Sequence[int],
    mpp: float,
    fps: float,
    out_dir: Path,
    diff_csv: Optional[Path] = None,
    min_nspots: int = 0,
    min_alpha: Optional[float] = None,
    non_overlapping: bool = False,
    force: bool = False,
) -> Dict[str, object]:
    """Compute (and cache) tau-step CSVs for one movie across all requested taus.

    Reads the tracks CSV once; writes ``{movie_id}_tau_steps_{NNN}.csv`` per tau.
    Skips taus whose output already exists unless ``force``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    taus = sorted({int(x) for x in tau_list})
    todo = taus if force else [
        tau for tau in taus if not (out_dir / tau_filename(movie_id, tau)).exists()
    ]
    result = {"movie_id": movie_id, "computed": [], "skipped": sorted(set(taus) - set(todo)),
              "n_displacements": {}, "status": "ok", "error": None}
    if not todo:
        return result

    try:
        tracks = pd.read_csv(tracks_csv, low_memory=False)
    except (OSError, pd.errors.ParserError) as exc:
        result["status"] = "failed"
        result["error"] = f"could not read tracks CSV: {exc}"
        return result

    particles = _surviving_particles(diff_csv, min_nspots, min_alpha)

    for tau in todo:
        df = compute_tau_displacements(
            tracks, tau, mpp=mpp, fps=fps, particles=particles,
            non_overlapping=non_overlapping,
        )
        out_path = out_dir / tau_filename(movie_id, tau)
        df.to_csv(out_path, index=False)
        result["computed"].append(tau)
        result["n_displacements"][tau] = int(len(df))
    return result
