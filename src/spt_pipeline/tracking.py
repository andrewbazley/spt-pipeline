"""Phase A: per-movie tracking (heavy, run once, cached).

Wraps trackpy (``batch`` -> ``link`` -> ``filter_stubs``) for each movie, using
per-movie ``mpp``/``fps`` from the manifest (falling back to CLI defaults). For
each movie it writes tracks, adjacent-frame steps (Object 1), fixed-tau
displacements (Object 2, at ``tau_steps`` -- computed here by default so the
common path is one command), the diffusion table, IMSD/EMSD, an MSD-vs-tau
summary, localization QC figures, diagnostic plots, and a context JSON, then
reports the paths so the manifest can be updated.

Movies are fully independent, so a run parallelizes trivially across processes
(``n_jobs``). Existing, non-empty outputs are skipped unless ``force``.

No-track movies
---------------
A movie where nothing clears the detection thresholds, or where no track
survives the stub filter, returns status ``"no_tracks"`` -- *not* ``"failed"``.
An empty field of view is an experimental result, not a crash: the run
continues, the manifest keeps a blank ``tracks_csv`` for that row, and every
downstream stage skips it. Only genuine faults (unreadable file, trackpy
exception) are failures.
"""
from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import density as density_mod
from . import diffusion, imaging, plotting, qc, utils

try:
    import trackpy as tp  # type: ignore
    HAVE_TRACKPY = True
except ImportError:  # pragma: no cover
    HAVE_TRACKPY = False


#: Default fixed-tau lags computed as part of tracking. 1 frame is the
#: adjacent-step reference; 5 and 10 frames probe whether the apparent
#: mobility is scale-free or whether confinement bends the distribution at
#: longer lags. Additional taus can be added later with ``compute tau-steps``
#: without re-tracking.
DEFAULT_TAUS: Tuple[int, ...] = (1, 5, 10)

#: Re-exported so the CLI has one place to read the default from.
DEFAULT_FIELD_RADIUS_PX = density_mod.DEFAULT_FIELD_RADIUS_PX


@dataclass
class TrackingParams:
    diameter: int = 7
    minmass: float = 40.0
    search_range: int = 5
    memory: int = 0
    min_separation: int = 3
    percentile: float = 90.0
    min_track_length: int = 4
    max_lagtime: int = 25
    diffusion_tau: int = 5
    channel: int = 0
    mpp: float = 0.183
    fps: float = 50.0
    #: Fixed-tau lags to compute inline. Empty tuple disables the step.
    tau_steps: Tuple[int, ...] = DEFAULT_TAUS
    #: Emit the localization/linking QC panel set alongside the tracks.
    qc_plots: bool = True
    #: Frame number(s) for the QC detections overlay.
    qc_detections_overlay_frames: Tuple[int, ...] = tuple(qc.DEFAULT_OVERLAY_FRAMES)
    #: Measure local density and segment cells from the pre-link detections.
    density: bool = True
    #: Crowding neighbourhood radius, in pixels.
    density_field_radius_px: float = density_mod.DEFAULT_FIELD_RADIUS_PX
    #: Keep the annotated pre-link detection table on disk. It is the only
    #: input from which density can be measured correctly later, so re-running
    #: the density analysis with a different radius does not need re-tracking.
    save_detections: bool = True

    def __post_init__(self) -> None:
        # Tolerate a list from argparse; keep the field hashable/picklable.
        self.tau_steps = tuple(sorted({int(t) for t in (self.tau_steps or ()) if int(t) >= 1}))
        self.qc_detections_overlay_frames = tuple(
            sorted({int(f) for f in (self.qc_detections_overlay_frames or ()) if int(f) >= 0}))

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["tau_steps"] = list(self.tau_steps)
        d["qc_detections_overlay_frames"] = list(self.qc_detections_overlay_frames)
        return d


@dataclass
class TrackingResult:
    movie_id: str
    status: str = "ok"
    tracks_csv: str = ""
    diff_csv: str = ""
    n_tracks: int = 0
    error: Optional[str] = None
    outputs: List[str] = field(default_factory=list)
    #: Taus whose displacement CSVs exist for this movie after the run.
    taus: List[int] = field(default_factory=list)
    #: Scalar localization-QC diagnostics (see :mod:`spt_pipeline.qc`).
    qc: Dict[str, Any] = field(default_factory=dict)
    #: Crowding summary: cells found, median link-safety ratio, source.
    density: Dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """True when this movie contributes data downstream."""
        return self.status in ("ok", "skipped") and bool(self.tracks_csv)


@contextlib.contextmanager
def _trackpy_quiet():
    """Quiet trackpy's per-frame logging and its harmless numeric warnings.

    trackpy's uncertainty estimate computes ``noise / mass`` and emits a
    divide-by-zero RuntimeWarning whenever a detection has zero mass; the result
    is unused here, so we suppress the warning (and the corresponding numpy
    divide/invalid states) to keep the console clean.
    """
    import warnings
    names = ["trackpy", "trackpy.feature", "trackpy.feature.batch",
             "trackpy.linking", "trackpy.linking.linking"]
    saved = [(logging.getLogger(n), logging.getLogger(n).level) for n in names]
    for logger, _ in saved:
        logger.setLevel(logging.WARNING)
    try:
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            yield
    finally:
        for logger, lvl in saved:
            logger.setLevel(lvl)


def build_adjacent_steps(tracks: pd.DataFrame, fps: float, mpp: float) -> pd.DataFrame:
    """Object 1: displacement between consecutive linked detections per track."""
    cols = ["particle", "frame_prev", "frame", "frame_delta", "dx_px", "dy_px",
            "step_px", "step_um", "tau_s"]
    t = utils.normalize_track_table(tracks)
    if t.empty or not {"particle", "frame", "x", "y"}.issubset(t.columns):
        return pd.DataFrame(columns=cols)
    t = t.sort_values(["particle", "frame"]).reset_index(drop=True)
    t["frame_prev"] = t.groupby("particle")["frame"].shift(1)
    t["x_prev"] = t.groupby("particle")["x"].shift(1)
    t["y_prev"] = t.groupby("particle")["y"].shift(1)
    s = t.dropna(subset=["frame_prev", "x_prev", "y_prev"]).copy()
    if s.empty:
        return pd.DataFrame(columns=cols)
    s["frame_prev"] = s["frame_prev"].astype(int)
    s["frame"] = s["frame"].astype(int)
    s["frame_delta"] = s["frame"] - s["frame_prev"]
    s["dx_px"] = s["x"] - s["x_prev"]
    s["dy_px"] = s["y"] - s["y_prev"]
    s["step_px"] = np.sqrt(s["dx_px"] ** 2 + s["dy_px"] ** 2)
    s["step_um"] = s["step_px"] * float(mpp)
    s["tau_s"] = s["frame_delta"] / float(fps)
    return s[cols]


def _attach_alpha(tracks: pd.DataFrame, imsd: pd.DataFrame) -> pd.DataFrame:
    out = utils.normalize_track_table(tracks)
    amap = diffusion.alpha_map_from_imsd(imsd)
    lookup = {str(k): v for k, v in amap.items()}
    out["alpha"] = out["particle"].map(lambda p: amap.get(p, lookup.get(str(p), np.nan)))
    return out


def track_movie(
    movie_id: str, source_path: Path, file_format: str, params: TrackingParams,
    out_dir: Path, force: bool = False,
) -> TrackingResult:
    """Track one movie end-to-end and write all per-movie outputs."""
    if not HAVE_TRACKPY:
        return TrackingResult(movie_id, status="failed", error="trackpy not installed")

    movie_dir = Path(out_dir) / movie_id
    movie_dir.mkdir(parents=True, exist_ok=True)
    tracks_file = movie_dir / f"{movie_id}_tracks.csv"
    diff_file = movie_dir / f"{movie_id}_diffusion_coefficients.csv"
    no_tracks_file = movie_dir / f"{movie_id}_NO_TRACKS.json"

    res = TrackingResult(movie_id, tracks_csv=str(tracks_file), diff_csv=str(diff_file))

    if not force and tracks_file.exists() and tracks_file.stat().st_size > 0 \
            and diff_file.exists():
        res.status = "skipped"
        try:
            res.n_tracks = int(pd.read_csv(tracks_file, usecols=["particle"])
                               ["particle"].nunique())
        except Exception:
            pass
        # Late-added taus are filled in without re-tracking.
        if params.tau_steps:
            res.taus = _ensure_tau_steps(movie_id, tracks_file, movie_dir, params, force=False)
        return res

    # A previous run already established that this movie yields nothing. Honour
    # that cached verdict instead of paying for the full localization pass again.
    if not force and no_tracks_file.exists():
        info = utils.load_json(no_tracks_file)
        return _no_tracks_result(movie_id, reason=str(info.get("reason", "no tracks")))

    try:
        frames = imaging.load_movie(source_path, file_format, channel=params.channel)
    except Exception as exc:
        res.status = "failed"; res.error = f"load failed: {exc}"
        return res

    try:
        with _trackpy_quiet():
            batch_kwargs = dict(diameter=params.diameter, minmass=params.minmass,
                                percentile=params.percentile, noise_size=1, engine="numba",
                                # Force single-process locate: movies are already
                                # parallelized across processes, and trackpy's own
                                # pool cannot spawn inside a daemon worker
                                # ("daemonic processes are not allowed to have
                                # children"). processes=1 keeps it in-process.
                                processes=1)
            if params.min_separation and params.min_separation > 0:
                batch_kwargs["separation"] = params.min_separation
            particles = tp.batch(frames, **batch_kwargs)
        if particles is None or particles.empty:
            # Empty field of view: record it, emit whatever QC is possible, and
            # let the run continue. Not an error.
            _write_no_tracks_marker(no_tracks_file, movie_id, source_path, params,
                                    reason="no particles located",
                                    n_detections=0, n_linked_tracks=0)
            return _no_tracks_result(movie_id, reason="no particles located")
        particles = utils.normalize_track_table(particles)
        with _trackpy_quiet():
            linked = tp.link(particles, search_range=params.search_range, memory=params.memory)
        tracks = tp.filter_stubs(linked, params.min_track_length)
        tracks = utils.normalize_track_table(tracks)
        if tracks.empty:
            # Detections existed but nothing linked into a track long enough to
            # measure. Still a legitimate outcome; keep the QC figures, because
            # they are exactly what diagnoses an over-strict threshold.
            n_linked = 0
            try:
                n_linked = int(utils.normalize_track_table(linked)["particle"].nunique())
            except Exception:
                pass
            if params.qc_plots:
                try:
                    qc.localization_qc(
                        movie_id, movie_dir, particles=particles, tracks=None,
                        frames=frames, diameter=params.diameter, minmass=params.minmass,
                        min_track_length=params.min_track_length,
                        overlay_frames=params.qc_detections_overlay_frames)
                except Exception:
                    pass
            _write_no_tracks_marker(no_tracks_file, movie_id, source_path, params,
                                    reason="no track survived filter_stubs",
                                    n_detections=int(len(particles)),
                                    n_linked_tracks=n_linked)
            return _no_tracks_result(movie_id, reason="no track survived filter_stubs")
        # A prior no-track verdict is now obsolete.
        no_tracks_file.unlink(missing_ok=True)

        dwell = tracks.groupby("particle")["frame"].count()
        tracks["dwell_time"] = tracks["particle"].map(dwell)
        if "mass" in tracks.columns:
            tracks["total_mass"] = tracks.groupby("particle")["mass"].transform("sum")

        imsd = tp.imsd(tracks, params.mpp, params.fps, max_lagtime=params.max_lagtime)
        emsd = tp.emsd(tracks, params.mpp, params.fps, max_lagtime=params.max_lagtime)
        tracks = _attach_alpha(tracks, imsd)

        # write core tables
        tracks.to_csv(tracks_file, index=False)
        imsd.to_csv(movie_dir / f"{movie_id}_IMSD.csv")
        emsd.to_csv(movie_dir / f"{movie_id}_EMSD.csv")

        steps = build_adjacent_steps(tracks, params.fps, params.mpp)
        steps.to_csv(movie_dir / f"{movie_id}_step_sizes.csv", index=False)

        diff_df = diffusion.build_diffusion_table(tracks, imsd, params.fps, params.mpp,
                                                  params.diffusion_tau)
        diff_df.to_csv(diff_file, index=False)

        msd_summary = diffusion.build_msd_summary(imsd, emsd)
        msd_summary.to_csv(movie_dir / f"{movie_id}_MSD_vs_tau.csv", index=False)

        # Object 2: fixed-tau displacements, computed here so the standard path
        # is a single command. `compute tau-steps` remains available for adding
        # further lags later without re-tracking.
        if params.tau_steps:
            res.taus = _ensure_tau_steps(movie_id, tracks_file, movie_dir, params,
                                         force=force, tracks=tracks)

        # local density, cells, and linking risk -- measured on the pre-link
        # detections, which is the only place crowding is represented honestly
        if params.density:
            try:
                res.density = _measure_density(
                    movie_id, movie_dir, particles, tracks, params,
                    frame_shape=_frame_shape(frames))
            except Exception as exc:
                res.density = {"error": f"{type(exc).__name__}: {exc}"}

        # localization / linking QC
        if params.qc_plots:
            try:
                res.qc = qc.localization_qc(
                    movie_id, movie_dir, particles=particles, tracks=tracks,
                    frames=frames, diameter=params.diameter, minmass=params.minmass,
                    min_track_length=params.min_track_length,
                    overlay_frames=params.qc_detections_overlay_frames)
            except Exception:
                res.qc = {}

        # diagnostic plots
        _movie_plots(movie_id, movie_dir, frames, tracks, steps, diff_df, msd_summary)

        # per-movie context
        utils.dump_json(
            {"movie_id": movie_id, "source_path": str(source_path),
             "params": params.to_dict(), "n_tracks": int(tracks["particle"].nunique()),
             "taus_computed": list(res.taus), "qc": res.qc, "density": res.density},
            movie_dir / f"{movie_id}_context.json",
        )

        res.n_tracks = int(tracks["particle"].nunique())
        res.outputs = [str(p) for p in movie_dir.glob(f"{movie_id}_*")]
        return res
    except Exception as exc:
        res.status = "failed"; res.error = f"tracking error: {exc}"
        return res


def _frame_shape(frames) -> Optional[Tuple[int, int]]:
    """(height, width) of the movie, or None if it cannot be determined."""
    try:
        a = np.asarray(frames[0])
        while a.ndim > 2:
            a = a[0]
        return (int(a.shape[0]), int(a.shape[1]))
    except Exception:
        return None


def _measure_density(movie_id, movie_dir, particles, tracks, params, frame_shape):
    """Density, cells and link risk for one movie; writes the per-movie tables.

    Runs on ``particles`` -- the detections *before* linking and stub filtering.
    Linking drops what it cannot join and stub filtering drops what is short,
    and both happen most in crowded fields, so measuring afterwards would
    understate crowding exactly where it matters.
    """
    res = density_mod.measure_movie(
        particles, tracks, search_range_px=params.search_range, mpp=params.mpp,
        movie_id=movie_id, field_radius_px=params.density_field_radius_px,
        frame_shape=frame_shape, segment=True)

    per_track = res["per_track"]
    if not per_track.empty:
        per_track.to_csv(movie_dir / f"{movie_id}_density_per_track.csv", index=False)
    cells = res["cells"]
    if isinstance(cells, pd.DataFrame) and not cells.empty:
        cells.to_csv(movie_dir / f"{movie_id}_density_per_cell.csv", index=False)

    if params.save_detections and particles is not None and len(particles):
        keep = [c for c in ("frame", "x", "y", "mass", "size", "ecc", "signal",
                            "field_localizations_n", "neighbors_R_n",
                            "neighbors_2R_n", "nn_distance_px", "cell_id")
                if c in res.get("annotated_tracks", particles).columns]
        det = density_mod.measure_detection_density(
            particles, search_range_px=params.search_range,
            field_radius_px=params.density_field_radius_px, mpp=params.mpp,
            frame_shape=frame_shape)
        seg = res.get("segmentation")
        if seg is not None and getattr(seg, "n_cells", 0):
            det["cell_id"] = density_mod.assign_cells(det, seg)
        cols = [c for c in keep if c in det.columns] or list(det.columns)
        det[cols].to_csv(movie_dir / f"{movie_id}_detections.csv.gz",
                         index=False, compression="gzip")

    seg = res.get("segmentation")
    safety = pd.to_numeric(per_track.get("link_safety_ratio"), errors="coerce") \
        if not per_track.empty else pd.Series(dtype=float)
    out = {"density_source": res["density_source"],
           "field_radius_px": float(params.density_field_radius_px),
           "step_scale_px": res["step_scale_px"],
           "n_cells": int(getattr(seg, "n_cells", 0)),
           "cells_resolved": bool(getattr(seg, "resolved", False)),
           "segmentation_note": getattr(seg, "note", ""),
           "median_link_safety_ratio": (float(safety.median())
                                        if safety.notna().any() else None),
           "median_field_localizations_n": (
               float(pd.to_numeric(per_track["track_median_field_localizations_n"],
                                   errors="coerce").median())
               if not per_track.empty else None)}
    return out


# ---------------------------------------------------------------------------
# No-track bookkeeping
# ---------------------------------------------------------------------------

def _no_tracks_result(movie_id: str, reason: str) -> TrackingResult:
    """A movie that yielded nothing: reported, skipped, never fatal."""
    return TrackingResult(movie_id, status="no_tracks", tracks_csv="", diff_csv="",
                          n_tracks=0, error=reason)


def _write_no_tracks_marker(path: Path, movie_id: str, source_path, params: TrackingParams,
                            *, reason: str, n_detections: int, n_linked_tracks: int) -> None:
    """Persist why a movie produced no tracks, so re-runs skip it cheaply.

    Also records the thresholds in force at the time: if the verdict later looks
    wrong, the file says exactly which cutoff to relax.
    """
    try:
        utils.dump_json(
            {"movie_id": movie_id, "source_path": str(source_path), "reason": reason,
             "n_detections": int(n_detections), "n_linked_tracks": int(n_linked_tracks),
             "params": params.to_dict(),
             "hint": "Lower --minmass/--percentile or --min-track-length, or raise "
                     "--range/--memory, then re-run with --force to revisit this movie."},
            path)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Inline fixed-tau displacements
# ---------------------------------------------------------------------------

def _ensure_tau_steps(movie_id: str, tracks_file: Path, movie_dir: Path,
                      params: TrackingParams, force: bool = False,
                      tracks: Optional[pd.DataFrame] = None) -> List[int]:
    """Compute any missing fixed-tau displacement CSVs for this movie.

    Delegates to :mod:`spt_pipeline.tau_steps` so there is exactly one
    implementation of the merge-join; this wrapper only decides *which* taus
    still need doing and reuses the in-memory track table when tracking just
    produced it (avoiding a pointless re-read of a file we wrote a moment ago).
    """
    from . import tau_steps as ts_mod
    taus = list(params.tau_steps)
    if not taus:
        return []
    todo = taus if force else [
        t for t in taus if not (movie_dir / ts_mod.tau_filename(movie_id, t)).exists()
    ]
    if todo and tracks is None:
        try:
            tracks = pd.read_csv(tracks_file, low_memory=False)
        except (OSError, pd.errors.ParserError):
            return [t for t in taus
                    if (movie_dir / ts_mod.tau_filename(movie_id, t)).exists()]
    for tau in todo:
        try:
            df = ts_mod.compute_tau_displacements(
                tracks, tau, mpp=params.mpp, fps=params.fps)
            df.to_csv(movie_dir / ts_mod.tau_filename(movie_id, tau), index=False)
        except Exception:
            continue
    return [t for t in taus if (movie_dir / ts_mod.tau_filename(movie_id, t)).exists()]


def _movie_plots(movie_id, movie_dir, frames, tracks, steps, diff_df, msd_summary):
    try:
        if steps is not None and not steps.empty:
            plotting.histogram(steps["step_um"].to_numpy(),
                               movie_dir / f"{movie_id}_step_hist.png",
                               xlabel="Adjacent step (µm)", title=f"{movie_id} steps")
        if diff_df is not None and not diff_df.empty and "D_fit_um2_s" in diff_df.columns:
            d = pd.to_numeric(diff_df["D_fit_um2_s"], errors="coerce").to_numpy()
            d = d[np.isfinite(d) & (d > 0)]
            if d.size:
                plotting.histogram(np.log10(d), movie_dir / f"{movie_id}_logD_hist.png",
                                   xlabel="log₁₀ D (µm²/s)", title=f"{movie_id} log D")
        if msd_summary is not None and not msd_summary.empty:
            plotting.msd_vs_tau(msd_summary, movie_dir / f"{movie_id}_MSD_vs_tau.png",
                                loglog=True, title=f"{movie_id} MSD vs τ")
        _tracks_overlay(movie_id, movie_dir, frames, tracks)
    except Exception:
        pass


def _tracks_overlay(movie_id, movie_dir, frames, tracks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plotting import _PLOT_LOCK
    with _PLOT_LOCK:
        try:
            first = np.array(frames[0], dtype=np.float32)
            if first.ndim > 2:
                first = first[0]
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(first, cmap="gray")
            for _, grp in tracks.groupby("particle"):
                g = grp.sort_values("frame")
                ax.plot(g["x"], g["y"], linewidth=0.8, alpha=0.8)
            ax.set_title(f"Tracks — {movie_id}"); ax.axis("off")
            fig.savefig(movie_dir / f"{movie_id}_tracks_overlay.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            plt.close("all")


# ---------------------------------------------------------------------------
# Parallel driver (module-level worker for picklability)
# ---------------------------------------------------------------------------

def _worker(job):
    (movie_id, source_path, file_format, params_dict, out_dir, force) = job
    params = TrackingParams(**params_dict)
    r = track_movie(movie_id, Path(source_path), file_format, params, Path(out_dir), force)
    return r.__dict__


def track_all(
    manifest_df: pd.DataFrame, out_dir: Path, defaults: TrackingParams,
    n_jobs: int = 4, force: bool = False, progress: bool = True,
) -> List[TrackingResult]:
    """Track every movie in the manifest, parallel across processes."""
    from . import manifest as manifest_mod
    jobs = []
    for _, row in manifest_df.iterrows():
        mpp, fps = manifest_mod.row_mpp_fps(row, defaults.mpp, defaults.fps)
        p = TrackingParams(**{**defaults.to_dict(), "mpp": mpp, "fps": fps})
        jobs.append((row["movie_id"], row["source_path"], row["file_format"],
                     p.to_dict(), str(out_dir), force))

    n_jobs = utils.resolve_n_jobs(n_jobs)
    results: List[Dict] = []
    if n_jobs == 1 or len(jobs) == 1:
        for j in jobs:
            results.append(_worker(j))
            if progress:
                print(f"  [{results[-1]['status']:>7}] {results[-1]['movie_id']}", flush=True)
    else:
        # 'spawn' is safest cross-platform (required on Windows/macOS).
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_jobs) as pool:
            for r in pool.imap_unordered(_worker, jobs):
                results.append(r)
                if progress:
                    print(f"  [{r['status']:>7}] {r['movie_id']}", flush=True)
    return [TrackingResult(**r) for r in results]
