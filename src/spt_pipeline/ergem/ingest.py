"""Turn MOSAIC ``Traj_*.csv`` into the per-movie products the pipeline uses.

Only what downstream figures/outputs consume is computed and written:

* ``*_tracks.csv``  -- trimmed (particle/frame/x/y[/mass]); read by
  ``compute tau-steps`` and lets core find the MSD file by name.
* ``*_diffusion_coefficients.csv`` -- per-particle ``alpha`` and ``D_fit`` (D at
  the fit lag, e.g. D_100ms). Both derive from the in-memory IMSD.
* ``*_MSD_vs_tau.csv`` -- ensemble MSD-vs-tau, built from IMSD (consumed by core
  ``compare-conditions``).
* ``*_context.json`` -- provenance.

Deliberately NOT computed/written (vestigial for figures/outputs):
EMSD (core prefers imsd_mean), and the ``*_IMSD.csv`` / ``*_EMSD.csv`` /
``*_step_sizes.csv`` files, plus dwell_time/total_mass columns. IMSD is computed
in memory only, vectorised (no per-particle loop), and is the source of both
alpha and D.

Movies are independent, so ingest runs across a process pool with a progress
bar. The manifest is written once at the end (not per movie).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .. import diffusion as core_diffusion
from .. import utils
from . import _util

TRACKS_SUFFIX = "_tracks.csv"
_TRACK_COLS = ["particle", "frame", "x", "y"]

# printed once per ingest run
INGEST_NOTE = ("computed: tracks(trimmed), diffusion (alpha + D at fit lag), "
               "MSD_vs_tau (from IMSD). skipped as unused: EMSD, IMSD.csv/EMSD.csv/"
               "step_sizes.csv, dwell/mass columns.")


# ---------------------------------------------------------------------------
# Vectorised IMSD (overlapping estimator; trackpy-compatible definition)
# ---------------------------------------------------------------------------

def compute_imsd(tracks: pd.DataFrame, mpp: float, fps: float, max_lagtime: int = 25) -> pd.DataFrame:
    """Per-particle MSD vs lag as a DataFrame (index = lag seconds, columns =
    particle, values = MSD in um^2). Vectorised: one self-merge + groupby per
    lag, assembled column-wise (no per-particle scalar assignment)."""
    t = utils.normalize_track_table(tracks)
    if t.empty or not {"particle", "frame", "x", "y"}.issubset(t.columns):
        return pd.DataFrame()
    t = t.dropna(subset=["particle", "frame", "x", "y"]).copy()
    t["frame"] = t["frame"].round().astype(np.int64)
    mpp2 = float(mpp) ** 2
    base = t[["particle", "frame", "x", "y"]]
    right = base.rename(columns={"frame": "frame_b", "x": "x_b", "y": "y_b"})
    per_lag: Dict[float, pd.Series] = {}
    for lag in range(1, int(max(1, max_lagtime)) + 1):
        left = base.copy()
        left["frame_b"] = left["frame"] + lag
        m = left.merge(right, on=["particle", "frame_b"], how="inner", copy=False)
        if m.empty:
            continue
        sq = ((m["x_b"].to_numpy() - m["x"].to_numpy()) ** 2
              + (m["y_b"].to_numpy() - m["y"].to_numpy()) ** 2) * mpp2
        per_lag[lag / float(fps)] = pd.Series(sq).groupby(m["particle"].to_numpy()).mean()
    if not per_lag:
        return pd.DataFrame()
    imsd = pd.DataFrame(per_lag)            # index = particle, columns = tau_s
    imsd = imsd.sort_index(axis=1).T        # index = tau_s, columns = particle
    imsd.index.name = "lag_time"
    return imsd


# ---------------------------------------------------------------------------
# Per-movie ingest
# ---------------------------------------------------------------------------

def ingest_movie(
    movie_id: str,
    trajectory_csv: Path,
    out_dir: Path,
    mpp: float,
    fps: float,
    diffusion_tau: int = 10,
    max_lagtime: int = 25,
    force: bool = False,
) -> Dict[str, object]:
    out_dir = Path(out_dir)
    movie_dir = out_dir / movie_id
    tracks_file = movie_dir / f"{movie_id}{TRACKS_SUFFIX}"
    diff_file = movie_dir / f"{movie_id}_diffusion_coefficients.csv"
    res: Dict[str, object] = {"movie_id": movie_id, "status": "ok",
                              "tracks_csv": str(tracks_file), "diff_csv": str(diff_file),
                              "n_tracks": 0, "error": None}
    if not force and tracks_file.exists() and tracks_file.stat().st_size > 0 and diff_file.exists():
        res["status"] = "skipped"
        try:
            res["n_tracks"] = int(pd.read_csv(tracks_file, usecols=["particle"])["particle"].nunique())
        except Exception:
            pass
        return res
    try:
        raw = pd.read_csv(trajectory_csv, low_memory=False)
    except (OSError, pd.errors.ParserError) as exc:
        res["status"] = "failed"; res["error"] = f"could not read Traj CSV: {exc}"
        return res

    tracks = utils.normalize_track_table(raw)
    if tracks.empty or not {"particle", "frame", "x", "y"}.issubset(tracks.columns):
        res["status"] = "failed"
        res["error"] = f"no particle/frame/x/y after normalization (cols={list(raw.columns)})"
        return res

    keep = [c for c in _TRACK_COLS + ["mass"] if c in tracks.columns]
    tracks = tracks[keep]

    movie_dir.mkdir(parents=True, exist_ok=True)
    imsd = compute_imsd(tracks, mpp, fps, max_lagtime=max_lagtime)

    tracks.to_csv(tracks_file, index=False)
    diff_df = core_diffusion.build_diffusion_table(tracks, imsd, fps, mpp, diffusion_tau)
    diff_df.to_csv(diff_file, index=False)
    # MSD_vs_tau from IMSD only (EMSD intentionally omitted -> emsd_um2 = NaN)
    msd_summary = core_diffusion.build_msd_summary(imsd, pd.Series(dtype=float))
    msd_summary.to_csv(movie_dir / f"{movie_id}_MSD_vs_tau.csv", index=False)

    utils.dump_json(
        {"movie_id": movie_id, "source": "mosaic_traj", "trajectory_csv": str(trajectory_csv),
         "mpp": mpp, "fps": fps, "diffusion_tau_frames": diffusion_tau,
         "n_tracks": int(tracks["particle"].nunique())},
        movie_dir / f"{movie_id}_context.json")
    res["n_tracks"] = int(tracks["particle"].nunique())
    return res


def _ingest_worker(job: dict) -> dict:
    return ingest_movie(**job)


def ingest_all(
    manifest_df: pd.DataFrame,
    out_dir: Path,
    default_mpp: float,
    default_fps: float,
    diffusion_tau: int = 10,
    max_lagtime: int = 25,
    force: bool = False,
    manifest_path: Optional[Path] = None,
    n_jobs: int = _util.DEFAULT_N_JOBS,
    progress: bool = True,
) -> List[Dict[str, object]]:
    """Ingest every movie in parallel; update the manifest once at the end."""
    from .. import manifest as core_manifest
    jobs = []
    skipped_rows = []
    for _, row in manifest_df.iterrows():
        traj = str(row.get("tracks_csv", "")).strip()
        if not traj:
            skipped_rows.append({"movie_id": row["movie_id"], "status": "failed",
                                 "error": "no trajectory_csv in manifest",
                                 "tracks_csv": "", "diff_csv": ""})
            continue
        mpp, fps = core_manifest.row_mpp_fps(row, default_mpp, default_fps)
        jobs.append(dict(movie_id=str(row["movie_id"]), trajectory_csv=Path(traj),
                         out_dir=Path(out_dir), mpp=mpp, fps=fps,
                         diffusion_tau=diffusion_tau, max_lagtime=max_lagtime, force=force))
    print(f"  ingest note: {INGEST_NOTE}", flush=True)
    results = _util.pmap(_ingest_worker, jobs, n_jobs=n_jobs,
                         desc="ingest", mode="process", progress=progress)
    results += skipped_rows

    # single manifest update (avoid per-row full-CSV rewrites)
    idx = {str(m): i for i, m in enumerate(manifest_df["movie_id"])}
    for r in results:
        if r["status"] in ("ok", "skipped"):
            i = idx.get(str(r["movie_id"]))
            if i is not None:
                manifest_df.iat[i, manifest_df.columns.get_loc("tracks_csv")] = r["tracks_csv"]
                manifest_df.iat[i, manifest_df.columns.get_loc("diff_csv")] = r["diff_csv"]
    if manifest_path is not None:
        core_manifest.write(manifest_df, Path(manifest_path))
    return results
