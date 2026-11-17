"""Localization and linking QC figures, written to ``qc_plots/`` per movie.

These mirror the diagnostics from the trackpy parameter-tuning notebook, so a
parameter choice made interactively there can be checked in bulk here. Panels
are combined into strips where they are read together:

``{movie}_qc_feature_properties.png``
    One strip: mass (log y), size, signal, ecc. The four numbers trackpy gives
    per detection, side by side.

``{movie}_qc_subpixel_bias.png``
    One strip: fractional part of the fitted x and y centroids. Flat is
    correct. Peaks at 0 or 0.5 mean the centroid is snapping to pixel centres,
    which inflates short displacements and biases D upward. Caused by a
    ``diameter`` that is too small.

``{movie}_qc_ecc_vs_mass.png``
    Real diffraction-limited spots form a compact low-eccentricity cloud.

``{movie}_qc_features_per_frame.png``
    Detections per frame across the movie: bleaching, focus loss, drift.

``{movie}_qc_detections_frame{N}.png``
    The detections drawn on the frame itself, for each requested frame.

``{movie}_qc_track_lengths.png``
    Track-length distribution with the stub-filter threshold marked.

Every function is defensive: a missing column or empty table skips that panel
rather than failing the movie. QC must never be able to break tracking.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .plotting import OKABE_ITO, _PLOT_LOCK, apply_house_style

#: Subdirectory, relative to the movie folder, holding these figures.
QC_DIRNAME = "qc_plots"

#: Default frame(s) for the detections overlay.
DEFAULT_OVERLAY_FRAMES: Sequence[int] = (100,)

#: Per-detection properties shown in the combined strip, in order.
FEATURE_COLUMNS = ("mass", "size", "signal", "ecc")

#: Columns plotted on a log y-axis (heavy-tailed counts).
_LOG_Y = {"mass"}

_AXIS_LABELS = {
    "mass": "Mass (a.u.)",
    "size": "Size (px)",
    "signal": "Signal (a.u.)",
    "ecc": "Eccentricity",
}


def _num(df: Optional[pd.DataFrame], col: str) -> Optional[np.ndarray]:
    """Finite numeric values of a column, or None if unusable."""
    if df is None or df.empty or col not in df.columns:
        return None
    v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return v if v.size else None


def _save(fig, path: Path) -> None:
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _nbins(n: int) -> int:
    return int(min(80, max(20, np.sqrt(max(n, 1)))))


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def feature_properties(particles: pd.DataFrame, out_png: Path, *,
                       minmass: Optional[float] = None) -> bool:
    """Combined strip of mass / size / signal / ecc histograms."""
    import matplotlib.pyplot as plt
    cols = [c for c in FEATURE_COLUMNS if _num(particles, c) is not None]
    if not cols:
        return False
    fig, axes = plt.subplots(1, len(cols), figsize=(3.3 * len(cols), 3.1))
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, cols):
        v = _num(particles, col)
        ax.hist(v, bins=_nbins(v.size), color=OKABE_ITO[0], alpha=0.85)
        if col == "mass" and minmass is not None and np.isfinite(minmass):
            ax.axvline(float(minmass), color=OKABE_ITO[3], linestyle="--", linewidth=1.3)
        if col in _LOG_Y:
            ax.set_yscale("log")
        ax.set_xlabel(_AXIS_LABELS.get(col, col))
        ax.set_ylabel("Detections" if ax is axes[0] else "")
    _save(fig, out_png)
    return True


def subpixel_bias(particles: pd.DataFrame, out_png: Path) -> bool:
    """Fractional-centroid histograms for x and y, side by side."""
    import matplotlib.pyplot as plt
    x, y = _num(particles, "x"), _num(particles, "y")
    if x is None or y is None:
        return False
    nb = 25
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    for ax, v, lab in ((axes[0], x, "x"), (axes[1], y, "y")):
        counts, edges = np.histogram(np.mod(v, 1.0), bins=nb, range=(0.0, 1.0))
        p = counts / counts.sum() if counts.sum() else counts.astype(float)
        ax.bar(0.5 * (edges[:-1] + edges[1:]), p, width=1.0 / nb,
               color=OKABE_ITO[0], alpha=0.85, edgecolor="none")
        ax.axhline(1.0 / nb, color="0.35", linestyle="--", linewidth=1.0)
        ax.set_xlabel(f"Subpixel position, {lab} (px)")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("Fraction of detections")
    _save(fig, out_png)
    return True


def ecc_vs_mass(particles: pd.DataFrame, out_png: Path) -> bool:
    """Eccentricity against mass; debris and merged spots sit high in ecc."""
    import matplotlib.pyplot as plt
    if particles is None or particles.empty or \
            not {"ecc", "mass"}.issubset(particles.columns):
        return False
    d = particles[["ecc", "mass"]].apply(pd.to_numeric, errors="coerce").dropna()
    if d.empty:
        return False
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.scatter(d["ecc"], d["mass"], s=5, alpha=min(0.4, max(0.02, 2000.0 / len(d))),
               color=OKABE_ITO[0], edgecolors="none")
    ax.set_xlabel("Eccentricity")
    ax.set_ylabel("Mass (a.u.)")
    _save(fig, out_png)
    return True


def mass_vs_size(particles: pd.DataFrame, out_png: Path) -> bool:
    """Mass against apparent spot size; real spots form a compact cloud."""
    import matplotlib.pyplot as plt
    if particles is None or particles.empty or \
            not {"mass", "size"}.issubset(particles.columns):
        return False
    d = particles[["mass", "size"]].apply(pd.to_numeric, errors="coerce").dropna()
    if d.empty:
        return False
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.scatter(d["size"], d["mass"], s=5, alpha=min(0.4, max(0.02, 2000.0 / len(d))),
               color=OKABE_ITO[2], edgecolors="none")
    ax.set_xlabel("Size (px)")
    ax.set_ylabel("Mass (a.u.)")
    _save(fig, out_png)
    return True


def features_per_frame(particles: pd.DataFrame, out_png: Path, *,
                       tracks: Optional[pd.DataFrame] = None) -> bool:
    """Detections per frame, and how many of them survive into tracks."""
    import matplotlib.pyplot as plt
    if particles is None or particles.empty or "frame" not in particles.columns:
        return False
    det = (pd.to_numeric(particles["frame"], errors="coerce").dropna().astype(int)
           .value_counts().sort_index())
    if det.empty:
        return False
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.plot(det.index.to_numpy(), det.to_numpy(), linewidth=1.0,
            color=OKABE_ITO[0], label="detected")
    if tracks is not None and not tracks.empty and "frame" in tracks.columns:
        lk = (pd.to_numeric(tracks["frame"], errors="coerce").dropna().astype(int)
              .value_counts().sort_index())
        if not lk.empty:
            ax.plot(lk.index.to_numpy(), lk.to_numpy(), linewidth=1.0,
                    color=OKABE_ITO[3], label="in tracks")
            ax.legend(loc="best")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Features")
    _save(fig, out_png)
    return True


def detections_overlay(frame, particles: pd.DataFrame, out_png: Path, *,
                       frame_no: int, diameter: int = 7) -> bool:
    """Detections circled on one frame of the movie."""
    import matplotlib.pyplot as plt
    if frame is None or particles is None or particles.empty:
        return False
    if not {"x", "y"}.issubset(particles.columns):
        return False
    sub = particles
    if "frame" in particles.columns:
        sub = particles[pd.to_numeric(particles["frame"], errors="coerce") == frame_no]
    if sub.empty:
        return False
    img = np.asarray(frame, dtype=float)
    while img.ndim > 2:
        img = img[0]
    # Percentile clip, as in the tuning notebook: a few hot pixels must not
    # flatten the field to black and hide the spots being checked.
    lo, hi = np.nanpercentile(img, [1, 99.9])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap="gray", vmin=lo, vmax=max(hi, lo + 1e-9))
    ax.scatter(sub["x"], sub["y"], s=(2.6 * float(diameter)) ** 2, facecolors="none",
               edgecolors=OKABE_ITO[1], linewidths=0.9)
    ax.set_title(f"Frame {frame_no}")
    ax.set_axis_off()
    ax.grid(False)
    _save(fig, out_png)
    return True


def track_length_hist(tracks: pd.DataFrame, out_png: Path, *,
                      min_track_length: Optional[int] = None) -> bool:
    """Track-length distribution with the stub-filter threshold marked."""
    import matplotlib.pyplot as plt
    if tracks is None or tracks.empty or "particle" not in tracks.columns:
        return False
    lens = tracks.groupby("particle")["frame"].count().to_numpy(dtype=float)
    lens = lens[np.isfinite(lens)]
    if lens.size == 0:
        return False
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    hi = int(np.percentile(lens, 99.5))
    ax.hist(lens, bins=np.arange(0.5, max(hi, 2) + 1.5, 1.0),
            color=OKABE_ITO[2], alpha=0.85)
    if min_track_length:
        ax.axvline(float(min_track_length) - 0.5, color=OKABE_ITO[3],
                   linestyle="--", linewidth=1.3)
    ax.set_yscale("log")
    ax.set_xlabel("Track length (frames)")
    ax.set_ylabel("Tracks")
    _save(fig, out_png)
    return True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def localization_qc(
    movie_id: str,
    movie_dir: Path,
    particles: Optional[pd.DataFrame] = None,
    tracks: Optional[pd.DataFrame] = None,
    frames: Any = None,
    *,
    diameter: Optional[int] = None,
    minmass: Optional[float] = None,
    min_track_length: Optional[int] = None,
    overlay_frames: Sequence[int] = DEFAULT_OVERLAY_FRAMES,
) -> Dict[str, Any]:
    """Write the QC panel set for one movie into ``movie_dir/qc_plots/``.

    ``frames`` is the loaded movie (any sequence indexable by frame number);
    only the requested ``overlay_frames`` are touched. A requested frame beyond
    the end of the movie is skipped rather than clamped -- silently drawing a
    different frame than asked for would make the figure a lie.

    Returns a summary dict of panels written plus the scalar diagnostics, which
    the caller stores in the per-movie context JSON so movies can be compared
    without opening figures.
    """
    qc_dir = Path(movie_dir) / QC_DIRNAME
    qc_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    def _try(name: str, fn) -> None:
        try:
            if fn():
                written.append(name)
        except Exception:
            pass

    with _PLOT_LOCK:
        apply_house_style()
        _try("feature_properties", lambda: feature_properties(
            particles, qc_dir / f"{movie_id}_qc_feature_properties.png", minmass=minmass))
        _try("subpixel_bias", lambda: subpixel_bias(
            particles, qc_dir / f"{movie_id}_qc_subpixel_bias.png"))
        _try("ecc_vs_mass", lambda: ecc_vs_mass(
            particles, qc_dir / f"{movie_id}_qc_ecc_vs_mass.png"))
        _try("mass_vs_size", lambda: mass_vs_size(
            particles, qc_dir / f"{movie_id}_qc_mass_vs_size.png"))
        _try("features_per_frame", lambda: features_per_frame(
            particles, qc_dir / f"{movie_id}_qc_features_per_frame.png", tracks=tracks))
        _try("track_lengths", lambda: track_length_hist(
            tracks, qc_dir / f"{movie_id}_qc_track_lengths.png",
            min_track_length=min_track_length))

        n_frames = len(frames) if frames is not None else 0
        for fno in overlay_frames or ():
            fno = int(fno)
            if not (0 <= fno < n_frames):
                continue
            _try(f"detections_frame{fno}", (lambda f=fno: detections_overlay(
                frames[f], particles, qc_dir / f"{movie_id}_qc_detections_frame{f:04d}.png",
                frame_no=f, diameter=diameter or 7)))

    summary: Dict[str, Any] = {"movie_id": movie_id, "panels": written,
                               "dir": QC_DIRNAME}
    summary.update(qc_metrics(particles, tracks, minmass=minmass))
    return summary


def qc_metrics(particles: Optional[pd.DataFrame], tracks: Optional[pd.DataFrame],
               *, minmass: Optional[float] = None) -> Dict[str, Any]:
    """Scalar QC diagnostics, for logging and cross-movie comparison.

    ``subpixel_bias_max_dev`` is the largest deviation of the fractional-centroid
    histogram from uniform, scaled so 0 is perfectly flat and 1 means a bin holds
    twice its uniform share. Above ~0.8 indicates pixel biting.
    """
    out: Dict[str, Any] = {}
    x, y = _num(particles, "x"), _num(particles, "y")
    if x is not None and y is not None:
        nb = 25
        devs = []
        for v in (x, y):
            counts, _ = np.histogram(np.mod(v, 1.0), bins=nb, range=(0.0, 1.0))
            p = counts / counts.sum() if counts.sum() else counts.astype(float)
            devs.append(float(np.max(np.abs(p - 1.0 / nb)) * nb))
        out["subpixel_bias_max_dev"] = max(devs)
    m = _num(particles, "mass")
    if m is not None:
        out["n_detections"] = int(m.size)
        out["mass_median"] = float(np.median(m))
        if minmass is not None and np.isfinite(minmass):
            out["frac_detections_above_minmass"] = float((m >= float(minmass)).mean())
    if particles is not None and not particles.empty and "frame" in particles.columns:
        det = (pd.to_numeric(particles["frame"], errors="coerce").dropna().astype(int)
               .value_counts().sort_index())
        n = len(det)
        if n >= 20:
            head = float(det.iloc[: max(1, n // 10)].mean())
            tail = float(det.iloc[-max(1, n // 10):].mean())
            out["bleach_ratio_last_first_decile"] = (tail / head) if head > 0 else None
    if tracks is not None and not tracks.empty and "particle" in tracks.columns:
        lens = tracks.groupby("particle")["frame"].count().to_numpy(dtype=float)
        if lens.size:
            out["n_tracks"] = int(lens.size)
            out["track_length_median"] = float(np.median(lens))
            out["track_length_max"] = float(np.max(lens))
    return out
