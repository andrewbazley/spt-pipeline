"""One row per track, carrying every quantity the pipeline can filter on.

This is the table the comparison stage actually works from. It is assembled
once per movie from files that already exist -- the diffusion table, the cached
fixed-tau displacements, and the marker-intensity table -- and written next to
them as ``{movie}_track_metrics.csv``. Nothing here re-reads a tracks CSV, so
building it for a 680k-track dataset costs seconds, and re-running the
comparison under different filters costs nothing at all.

Mobility columns, and why there are several
-------------------------------------------
``D_fit_um2_s``
    MSD(tau_fit) / (4 * tau_fit) at a **single** lag, tau_fit = the
    ``--diffusion-tau`` used at tracking. ``tau_fit_s`` records that lag in
    seconds, because "D" without its tau is not a number you can compare.

``v_path_um_s``
    Mean speed along the path: total distance travelled divided by time
    observed. Intuitive, but it is the most localization-error-inflated of the
    mobility measures -- every noise excursion adds path length, and a confined
    particle jittering in place still accrues a nonzero speed. Roughly, the
    error floor is ``~2*sigma_loc/dt``: at 100 Hz and 40 nm precision that is
    about 8 um/s of pure noise. Read it against the step medians below.

``step_median_um_tau{N}``
    Median displacement at lag N frames. Median, not mean, so one bad link
    cannot move it; no fit, so no fit assumptions. At larger N the localization
    error contributes proportionally less, which is why both a short and a long
    lag are kept: if a result holds at tau=1 and dies at tau=5, it was noise.

Because these disagree in informative ways on subdiffusive data, they are all
kept rather than one being chosen on the user's behalf.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import manifest as manifest_mod
from . import utils

#: Written beside the other per-movie outputs.
SUFFIX = "_track_metrics.csv"

_TAU_RE = re.compile(r"_tau_steps_(\d+)\.csv$")

#: Columns carried through from the diffusion table when present.
_DIFF_COLUMNS = ("particle", "n_spots", "dwell_time", "alpha", "mass_per_frame",
                 "total_mass", "mean_x_px", "mean_y_px", "tau_fit_frames",
                 "tau_fit_s", "msd_fit_um2", "D_fit_um2_s", "D_fit_um2_s_corr",
                 "tau_last_s", "D_last_um2_s")

#: Columns carried through from the per-track density table when present.
_DENSITY_COLUMNS = ("particle", "track_median_field_localizations_n",
                    "track_median_field_localizations_per_um2",
                    "track_p90_field_localizations_n",
                    "track_median_neighbors_R_n", "track_median_neighbors_2R_n",
                    "track_median_nn_distance_px", "link_safety_ratio",
                    "p_link_ambiguous", "track_fraction_field_window_truncated",
                    "cell_id", "cell_median_field_localizations_n",
                    "cell_link_safety_ratio", "cell_n_tracks", "cell_area_um2",
                    "n_cells_in_field", "density_source")

#: Columns carried through from the marker table when present.
_MARKER_COLUMNS = ("particle", "marker_intensity", "marker_intensity_median",
                   "marker_intensity_sd", "x_range_px", "y_range_px",
                   "dist_to_edge_px", "frac_disk_in_frame",
                   "n_localizations_measured")


def _pkey(value) -> str:
    """Particle id as text, so int/float round-trips across CSVs still join."""
    try:
        f = float(value)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(value)


def _read(path: Path, columns: Sequence[str]) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
    except (OSError, pd.errors.ParserError):
        return None
    if df.empty or "particle" not in df.columns:
        return None
    keep = [c for c in columns if c in df.columns]
    out = df[keep].copy()
    out["_pk"] = out["particle"].map(_pkey)
    return out.drop(columns=["particle"])


def discover_tau_files(movie_dir: Path, movie_id: str) -> Dict[int, Path]:
    """Cached fixed-tau displacement CSVs for one movie, keyed by lag."""
    out: Dict[int, Path] = {}
    for p in Path(movie_dir).glob(f"{movie_id}_tau_steps_*.csv"):
        m = _TAU_RE.search(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def path_speed_and_steps(tau_files: Dict[int, Path], fps: float
                         ) -> Optional[pd.DataFrame]:
    """Per-track mean path speed and median step size, from cached tau-steps.

    Mean path speed comes from the tau=1 table: summing consecutive-frame
    displacements *is* the path length, and dividing by (n_steps / fps) gives
    the mean speed over the intervals actually observed. Gaps bridged by
    ``--memory`` produce no tau=1 step, so they are correctly excluded from
    both the distance and the time rather than being interpolated across.
    """
    frames: List[pd.DataFrame] = []
    for tau, path in sorted(tau_files.items()):
        try:
            df = pd.read_csv(path, usecols=lambda c: c in ("particle", "step_um"))
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        if df.empty or "step_um" not in df.columns:
            continue
        df["_pk"] = df["particle"].map(_pkey)
        g = df.groupby("_pk")["step_um"]
        part = pd.DataFrame({f"step_median_um_tau{tau}": g.median(),
                             f"step_mean_um_tau{tau}": g.mean(),
                             f"n_steps_tau{tau}": g.size()})
        if tau == 1:
            # Mean speed along the path = total distance / time observed.
            part["path_length_um"] = g.sum()
            part["v_path_um_s"] = g.sum() / (g.size() / float(fps))
        frames.append(part)
    if not frames:
        return None
    out = pd.concat(frames, axis=1)
    out.index.name = "_pk"
    return out.reset_index()


def build_movie(row: pd.Series, fps: float, force: bool = False) -> Optional[Path]:
    """Assemble and cache the per-track metrics table for one movie."""
    tracks_csv = str(row.get("tracks_csv", "")).strip()
    if not tracks_csv:
        return None
    tpath = Path(tracks_csv)
    movie_id = str(row["movie_id"])
    out_path = tpath.with_name(f"{movie_id}{SUFFIX}")
    if out_path.exists() and not force:
        return out_path

    diff = _read(Path(str(row.get("diff_csv", "")).strip() or "."), _DIFF_COLUMNS)
    if diff is None:
        return None

    steps = path_speed_and_steps(discover_tau_files(tpath.parent, movie_id), fps)
    if steps is not None:
        diff = diff.merge(steps, on="_pk", how="left")

    mk = _read(tpath.with_name(f"{movie_id}_marker_intensity.csv"), _MARKER_COLUMNS)
    if mk is not None:
        diff = diff.merge(mk, on="_pk", how="left")

    dn = _read(tpath.with_name(f"{movie_id}_density_per_track.csv"), _DENSITY_COLUMNS)
    if dn is not None:
        diff = diff.merge(dn, on="_pk", how="left")

    diff = diff.rename(columns={"_pk": "particle"})
    diff.insert(0, "movie_id", movie_id)
    diff.to_csv(out_path, index=False)
    return out_path


def build_all(manifest_df: pd.DataFrame, default_fps: float = 50.0,
              force: bool = False) -> Dict[str, int]:
    """Build the metrics table for every movie that has tracking output."""
    n_built = n_skipped = n_failed = 0
    for _, row in manifest_df.iterrows():
        if not manifest_mod.has_tracks(row):
            continue
        _mpp, fps = manifest_mod.row_mpp_fps(row, 0.183, default_fps)
        existed = Path(str(row["tracks_csv"])).with_name(
            f"{row['movie_id']}{SUFFIX}").exists()
        try:
            p = build_movie(row, fps, force=force)
        except Exception:
            n_failed += 1
            continue
        if p is None:
            n_failed += 1
        elif existed and not force:
            n_skipped += 1
        else:
            n_built += 1
    return {"built": n_built, "cached": n_skipped, "failed": n_failed}


def load_pool(manifest_df: pd.DataFrame, default_fps: float = 50.0,
              build_missing: bool = True) -> pd.DataFrame:
    """Every movie's per-track metrics, labelled with condition and experiment."""
    frames = []
    for _, row in manifest_df.iterrows():
        if not manifest_mod.has_tracks(row):
            continue
        movie_id = str(row["movie_id"])
        p = Path(str(row["tracks_csv"])).with_name(f"{movie_id}{SUFFIX}")
        if not p.exists() and build_missing:
            _mpp, fps = manifest_mod.row_mpp_fps(row, 0.183, default_fps)
            try:
                build_movie(row, fps)
            except Exception:
                pass
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, low_memory=False)
        except (OSError, pd.errors.ParserError):
            continue
        if df.empty:
            continue
        df["movie_id"] = movie_id
        df["condition"] = str(row["condition"])
        df["experiment"] = str(row.get("experiment", ""))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Stable, readable column order: identity, then mobility, then everything else.
    lead = [c for c in ("condition", "experiment", "movie_id", "particle",
                        "D_fit_um2_s", "D_fit_um2_s_corr", "tau_fit_s",
                        "v_path_um_s", "step_median_um_tau1", "step_median_um_tau5",
                        "alpha", "n_spots", "mass_per_frame", "marker_intensity",
                        "track_median_field_localizations_n", "link_safety_ratio",
                        "cell_id", "density_source")
            if c in out.columns]
    return out[lead + [c for c in out.columns if c not in lead]]


def available_taus(pool: pd.DataFrame) -> List[int]:
    """Lags for which a median-step column exists in the pool."""
    out = []
    for c in pool.columns:
        m = re.fullmatch(r"step_median_um_tau(\d+)", str(c))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)
