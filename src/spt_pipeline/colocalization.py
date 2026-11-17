#!/usr/bin/env python3
"""
SPT Two-Color Co-Diffusion Pipeline
====================================
Standalone script for bead-based channel registration and two-color
co-diffusion analysis, designed to consume track CSVs produced by
``spt-pipeline track trackpy`` (output format: x, y, frame, particle, mass, ...).

Subcommands
-----------
    register    Compute affine transform from a two-channel bead image.
    pair        Apply transform, find co-diffusing track pairs.
    analyze     Summarize co-diffusion statistics with random-overlap controls.

Usage examples
--------------
    # Step 1: derive registration from bead image
    spt-coloc register beads.tif -o registration/

    # Step 2: pair tracks across channels
    spt-coloc pair ch1/ ch2/ registration/transform.json \\
        --mpp 0.122 --fps 50 -o coloc/

    # Step 3: analyze co-diffusion with shuffle control
    spt-coloc analyze coloc/ --n-shuffles 500 -o coloc/
"""

from __future__ import annotations

# Annotations are deferred so that type hints naming optional dependencies
# (skimage's AffineTransform) do not need those dependencies to be installed
# just to import this module. Without it, `spt-coloc --help` fails on a
# core-only install.

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import ndimage, optimize
from scipy.spatial import cKDTree

try:
    import tifffile
    HAVE_TIFFFILE = True
except ImportError:
    HAVE_TIFFFILE = False

try:
    import trackpy as tp
    HAVE_TRACKPY = True
except ImportError:
    HAVE_TRACKPY = False

try:
    from skimage.transform import AffineTransform, warp
    from skimage.registration import phase_cross_correlation
    HAVE_SKIMAGE = True
except ImportError:
    # Names are bound to None so that anything referencing them at module scope
    # gets a clear failure at call time, not an obscure NameError at import.
    AffineTransform = warp = phase_cross_correlation = None
    HAVE_SKIMAGE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _save_fig(fig, path: str, dpi: int = 300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def _should_ignore_path(path_like) -> bool:
    p = Path(path_like)
    return any(
        part.startswith(".") or part.startswith("._") or part == "__MACOSX"
        for part in p.parts
    )


def _discover_track_csvs(root: Path) -> List[Path]:
    """Find *_tracks.csv files under root (one level of movie subdirs)."""
    hits = sorted(root.rglob("*_tracks.csv"))
    return [p for p in hits if p.is_file() and not _should_ignore_path(p)]

# ---------------------------------------------------------------------------
# REGISTER subcommand
# ---------------------------------------------------------------------------

def _bandpass_filter(img: np.ndarray, psf_sigma_px: float) -> np.ndarray:
    """DoG bandpass: enhances point sources, suppresses background."""
    small = ndimage.uniform_filter(img.astype(np.float64), size=int(2 * round(psf_sigma_px) + 1))
    large = ndimage.uniform_filter(img.astype(np.float64), size=int(4 * round(psf_sigma_px) + 1))
    bp = small - large
    bp[bp < 0] = 0
    return bp


def _find_peaks(bp: np.ndarray, threshold_sigma: float = 4.0, min_sep: int = 5) -> np.ndarray:
    """Detect local maxima above threshold in a bandpass-filtered image.

    Returns Nx2 array of (x, y) centroids (sub-pixel via centroid refinement).
    """
    thr = bp.mean() + threshold_sigma * bp.std()
    mask = bp > thr

    local_max = ndimage.maximum_filter(bp, size=2 * min_sep + 1)
    peaks = (bp == local_max) & mask

    labeled, n = ndimage.label(peaks)
    coords = []
    half = min_sep
    ny, nx = bp.shape
    for lbl in range(1, n + 1):
        ys, xs = np.where(labeled == lbl)
        cy, cx = int(ys.mean()), int(xs.mean())
        y0, y1 = max(cy - half, 0), min(cy + half + 1, ny)
        x0, x1 = max(cx - half, 0), min(cx + half + 1, nx)
        roi = bp[y0:y1, x0:x1]
        if roi.sum() == 0:
            continue
        ry, rx = np.mgrid[y0:y1, x0:x1]
        total = roi.sum()
        sub_x = float((rx * roi).sum() / total)
        sub_y = float((ry * roi).sum() / total)
        coords.append([sub_x, sub_y])
    return np.array(coords) if coords else np.empty((0, 2))


def _match_beads(
    ref_xy: np.ndarray,
    mov_xy: np.ndarray,
    bulk_shift: np.ndarray,
    max_dist_px: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match bead centroids by nearest neighbor after applying a bulk shift.

    Returns (ref_matched, mov_matched) arrays of paired coordinates.
    """
    shifted = mov_xy + bulk_shift
    tree = cKDTree(shifted)
    ref_matched, mov_matched = [], []
    used = set()
    for i, pt in enumerate(ref_xy):
        dist, idx = tree.query(pt)
        if dist < max_dist_px and idx not in used:
            ref_matched.append(ref_xy[i])
            mov_matched.append(mov_xy[idx])
            used.add(idx)
    return np.array(ref_matched), np.array(mov_matched)


def cmd_register(args):
    """Compute affine transform from a two-channel bead image."""
    if not HAVE_TIFFFILE:
        raise SystemExit("tifffile is required for register")
    if not HAVE_SKIMAGE:
        raise SystemExit("scikit-image is required for register")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Load bead image --
    raw = tifffile.imread(args.bead_image)
    log.info("Raw bead image shape: %s  dtype: %s", raw.shape, raw.dtype)

    # Expect either (C, Z, Y, X), (C, Y, X), or (Z, Y, X, C) etc.
    # User said: two-channel image with Z-stacks acquired separately per
    # wavelength -> likely (C, Z, Y, X) or (2, nZ, Y, X).
    img = raw.astype(np.float64)

    # Determine channel axis heuristically
    if img.ndim == 4:
        # (C, Z, Y, X) or (Z, C, Y, X)
        if img.shape[0] == 2:
            ch_ref = img[args.ref_channel]  # (Z, Y, X)
            ch_mov = img[1 - args.ref_channel]
        elif img.shape[1] == 2:
            ch_ref = img[:, args.ref_channel]
            ch_mov = img[:, 1 - args.ref_channel]
        else:
            raise SystemExit(
                f"Cannot identify channel axis in shape {img.shape}. "
                "Expected one axis of size 2."
            )
    elif img.ndim == 3 and img.shape[0] == 2:
        # (C, Y, X) -- single z-plane, two channels
        ch_ref = img[args.ref_channel][np.newaxis]
        ch_mov = img[1 - args.ref_channel][np.newaxis]
    elif img.ndim == 3:
        raise SystemExit(
            f"3D array with shape {img.shape}. If this is a single-channel "
            "Z-stack, both channels must be in one file."
        )
    else:
        raise SystemExit(f"Unexpected image dimensions: {img.shape}")

    log.info(
        "Channel stacks: ref=%s  mov=%s",
        ch_ref.shape,
        ch_mov.shape,
    )

    # -- Max-project --
    proj_ref = ch_ref.max(axis=0) if ch_ref.ndim == 3 else ch_ref
    proj_mov = ch_mov.max(axis=0) if ch_mov.ndim == 3 else ch_mov

    # -- Bandpass and detect --
    psf_sigma = args.psf_nm / args.pixel_nm
    bp_ref = _bandpass_filter(proj_ref, psf_sigma)
    bp_mov = _bandpass_filter(proj_mov, psf_sigma)

    peaks_ref = _find_peaks(bp_ref, threshold_sigma=args.threshold_sigma, min_sep=args.min_sep)
    peaks_mov = _find_peaks(bp_mov, threshold_sigma=args.threshold_sigma, min_sep=args.min_sep)
    log.info("Detected peaks: ref=%d  mov=%d", len(peaks_ref), len(peaks_mov))

    if len(peaks_ref) < 4 or len(peaks_mov) < 4:
        raise SystemExit("Too few beads detected. Adjust --threshold-sigma.")

    # -- Bulk shift via phase cross-correlation --
    shift_yx, _, _ = phase_cross_correlation(bp_ref, bp_mov, upsample_factor=10)
    bulk_shift = np.array([shift_yx[1], shift_yx[0]])  # (dx, dy)
    log.info("Bulk pixel shift (dx, dy): %.2f, %.2f", bulk_shift[0], bulk_shift[1])

    # -- Match beads --
    ref_matched, mov_matched = _match_beads(
        peaks_ref, peaks_mov, bulk_shift, max_dist_px=args.match_radius
    )
    log.info("Matched bead pairs: %d", len(ref_matched))

    if len(ref_matched) < 3:
        raise SystemExit(
            f"Only {len(ref_matched)} bead pairs matched (need >= 3). "
            "Try increasing --match-radius or adjusting --threshold-sigma."
        )

    # -- Fit affine transform: maps mov -> ref --
    tform = AffineTransform()
    success = tform.estimate(mov_matched, ref_matched)
    if not success:
        raise SystemExit("Affine fit failed.")

    # -- Compute residuals --
    transformed = tform(mov_matched)
    residuals = np.sqrt(((transformed - ref_matched) ** 2).sum(axis=1))
    reg_error_px = float(residuals.mean())
    reg_error_nm = reg_error_px * args.pixel_nm
    log.info(
        "Registration error: %.3f px (%.1f nm) mean, %.3f px (%.1f nm) max",
        reg_error_px,
        reg_error_nm,
        residuals.max(),
        residuals.max() * args.pixel_nm,
    )

    # -- Save transform --
    transform_data = {
        "matrix": tform.params.tolist(),
        "ref_channel": int(args.ref_channel),
        "n_pairs": int(len(ref_matched)),
        "residual_mean_px": float(reg_error_px),
        "residual_mean_nm": float(reg_error_nm),
        "residual_max_px": float(residuals.max()),
        "residual_max_nm": float(residuals.max() * args.pixel_nm),
        "pixel_nm": float(args.pixel_nm),
        "psf_nm": float(args.psf_nm),
        "bead_image": str(args.bead_image),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    tf_path = out_dir / "transform.json"
    with open(tf_path, "w") as fh:
        json.dump(transform_data, fh, indent=2)
    log.info("Saved transform to %s", tf_path)

    # -- Diagnostic plots --
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: overlay of max projections with matched pairs
    axes[0].imshow(proj_ref, cmap="Reds", alpha=0.6)
    axes[0].imshow(proj_mov, cmap="Greens", alpha=0.4)
    for r, m in zip(ref_matched, mov_matched):
        axes[0].plot([r[0], m[0]], [r[1], m[1]], "w-", lw=0.5, alpha=0.7)
    axes[0].scatter(ref_matched[:, 0], ref_matched[:, 1], c="red", s=8, zorder=3, label="ref")
    axes[0].scatter(mov_matched[:, 0], mov_matched[:, 1], c="lime", s=8, zorder=3, label="mov")
    axes[0].set_title(f"Matched bead pairs (n={len(ref_matched)})")
    axes[0].legend(fontsize=8)
    axes[0].axis("off")

    # Panel 2: quiver plot of displacement field
    dx = ref_matched[:, 0] - mov_matched[:, 0]
    dy = ref_matched[:, 1] - mov_matched[:, 1]
    axes[1].quiver(
        mov_matched[:, 0], mov_matched[:, 1], dx, dy,
        angles="xy", scale_units="xy", scale=1, color="cyan", width=0.003,
    )
    axes[1].set_xlim(0, proj_ref.shape[1])
    axes[1].set_ylim(proj_ref.shape[0], 0)
    axes[1].set_aspect("equal")
    axes[1].set_title("Displacement field (mov -> ref)")

    # Panel 3: residual histogram
    axes[2].hist(residuals * args.pixel_nm, bins=max(10, len(residuals) // 5), edgecolor="k")
    axes[2].axvline(reg_error_nm, color="red", ls="--", label=f"mean={reg_error_nm:.1f} nm")
    axes[2].set_xlabel("Residual (nm)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Registration residuals")
    axes[2].legend(fontsize=8)

    _save_fig(fig, str(out_dir / "registration_diagnostic.png"))

    # Save matched pairs for inspection
    pairs_df = pd.DataFrame({
        "ref_x": ref_matched[:, 0],
        "ref_y": ref_matched[:, 1],
        "mov_x": mov_matched[:, 0],
        "mov_y": mov_matched[:, 1],
        "residual_px": residuals,
        "residual_nm": residuals * args.pixel_nm,
    })
    pairs_df.to_csv(out_dir / "matched_bead_pairs.csv", index=False)
    log.info("Registration complete.")

# ---------------------------------------------------------------------------
# PAIR subcommand
# ---------------------------------------------------------------------------

def _load_transform(path: str) -> Tuple[AffineTransform, Dict]:
    with open(path) as fh:
        data = json.load(fh)
    tform = AffineTransform(matrix=np.array(data["matrix"]))
    return tform, data


def _transform_coords(df: pd.DataFrame, tform: AffineTransform) -> pd.DataFrame:
    """Apply affine transform to x, y columns in-place; return copy."""
    df = df.copy()
    xy = np.column_stack([df["x"].values, df["y"].values])
    xy_t = tform(xy)
    df["x"] = xy_t[:, 0]
    df["y"] = xy_t[:, 1]
    df["x_orig"] = xy[:, 0]
    df["y_orig"] = xy[:, 1]
    return df


def _load_all_tracks(root: Path) -> pd.DataFrame:
    """Load and concatenate all *_tracks.csv files under root.

    Adds a 'movie' column with the parent folder name and re-indexes particle
    IDs to be globally unique (movie + original particle).
    """
    csv_files = _discover_track_csvs(root)
    if not csv_files:
        raise SystemExit(f"No *_tracks.csv found under {root}")
    dfs = []
    for p in csv_files:
        df = pd.read_csv(p, low_memory=False)
        movie_name = p.parent.name
        df["movie"] = movie_name
        df["particle_orig"] = df["particle"]
        df["particle"] = df["movie"] + "_" + df["particle"].astype(str)
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    log.info("Loaded %d localizations from %d CSVs under %s", len(out), len(csv_files), root)
    return out


def _pair_per_frame(
    ref_df: pd.DataFrame,
    mov_df: pd.DataFrame,
    max_dist_px: float,
) -> pd.DataFrame:
    """Nearest-neighbor pairing of localizations per frame.

    Returns a DataFrame with one row per frame-level pair containing columns
    from both channels (suffixed _ref / _mov).
    """
    frames = np.intersect1d(ref_df["frame"].unique(), mov_df["frame"].unique())
    records = []
    for fr in frames:
        rr = ref_df[ref_df["frame"] == fr]
        mm = mov_df[mov_df["frame"] == fr]
        if rr.empty or mm.empty:
            continue
        ref_xy = rr[["x", "y"]].values
        mov_xy = mm[["x", "y"]].values
        tree = cKDTree(mov_xy)
        dists, idxs = tree.query(ref_xy)
        for i, (d, j) in enumerate(zip(dists, idxs)):
            if d <= max_dist_px:
                rec = {
                    "frame": int(fr),
                    "x_ref": ref_xy[i, 0],
                    "y_ref": ref_xy[i, 1],
                    "particle_ref": rr.iloc[i]["particle"],
                    "mass_ref": rr.iloc[i].get("mass", np.nan),
                    "x_mov": mov_xy[j, 0],
                    "y_mov": mov_xy[j, 1],
                    "particle_mov": mm.iloc[j]["particle"],
                    "mass_mov": mm.iloc[j].get("mass", np.nan),
                    "pair_dist_px": d,
                }
                records.append(rec)
    return pd.DataFrame(records)


def _identify_codiffusing_pairs(
    frame_pairs: pd.DataFrame,
    min_coframes: int = 3,
) -> pd.DataFrame:
    """Identify track pairs that remain co-localized over multiple frames.

    A co-diffusing pair is defined as two specific track IDs that appear
    paired in >= min_coframes consecutive or near-consecutive frames.
    Returns a summary DataFrame with one row per co-diffusing pair.
    """
    if frame_pairs.empty:
        return pd.DataFrame()

    grouped = (
        frame_pairs
        .groupby(["particle_ref", "particle_mov"])
        .agg(
            n_coframes=("frame", "count"),
            first_frame=("frame", "min"),
            last_frame=("frame", "max"),
            mean_dist_px=("pair_dist_px", "mean"),
            median_dist_px=("pair_dist_px", "median"),
            max_dist_px=("pair_dist_px", "max"),
        )
        .reset_index()
    )
    codiff = grouped[grouped["n_coframes"] >= min_coframes].copy()
    codiff["frame_span"] = codiff["last_frame"] - codiff["first_frame"] + 1
    codiff["coframe_fraction"] = codiff["n_coframes"] / codiff["frame_span"]
    codiff = codiff.sort_values("n_coframes", ascending=False).reset_index(drop=True)
    return codiff


def _displacement_correlation(
    frame_pairs: pd.DataFrame,
    ref_tracks: pd.DataFrame,
    mov_tracks: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-step displacement vector correlation for co-localized pairs.

    For each frame-pair in which both tracks have a displacement from the
    previous frame, computes the cosine similarity and magnitude ratio of
    the two displacement vectors.
    """
    records = []
    for (pr, pm), grp in frame_pairs.groupby(["particle_ref", "particle_mov"]):
        if len(grp) < 2:
            continue
        grp = grp.sort_values("frame")
        frames = grp["frame"].values
        for k in range(1, len(frames)):
            f_prev, f_curr = frames[k - 1], frames[k]
            if f_curr - f_prev > 2:
                continue
            r0 = grp[grp["frame"] == f_prev].iloc[0]
            r1 = grp[grp["frame"] == f_curr].iloc[0]
            dx_ref = r1["x_ref"] - r0["x_ref"]
            dy_ref = r1["y_ref"] - r0["y_ref"]
            dx_mov = r1["x_mov"] - r0["x_mov"]
            dy_mov = r1["y_mov"] - r0["y_mov"]
            mag_ref = np.sqrt(dx_ref**2 + dy_ref**2)
            mag_mov = np.sqrt(dx_mov**2 + dy_mov**2)
            if mag_ref > 0 and mag_mov > 0:
                cos_sim = (dx_ref * dx_mov + dy_ref * dy_mov) / (mag_ref * mag_mov)
            else:
                cos_sim = np.nan
            records.append({
                "particle_ref": pr,
                "particle_mov": pm,
                "frame": f_curr,
                "dx_ref": dx_ref,
                "dy_ref": dy_ref,
                "dx_mov": dx_mov,
                "dy_mov": dy_mov,
                "mag_ref": mag_ref,
                "mag_mov": mag_mov,
                "cos_similarity": cos_sim,
                "mag_ratio": min(mag_ref, mag_mov) / max(mag_ref, mag_mov) if max(mag_ref, mag_mov) > 0 else np.nan,
            })
    return pd.DataFrame(records)


def cmd_pair(args):
    """Apply channel registration and identify co-diffusing track pairs."""
    if not HAVE_SKIMAGE:
        raise SystemExit("scikit-image is required")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tform, tf_meta = _load_transform(args.transform)
    reg_error_px = tf_meta["residual_mean_px"]
    coloc_radius = args.coloc_radius if args.coloc_radius else 3.0 * reg_error_px
    log.info(
        "Registration error: %.3f px. Co-localization radius: %.2f px (%.1f nm)",
        reg_error_px,
        coloc_radius,
        coloc_radius * tf_meta["pixel_nm"],
    )

    ref_root = Path(args.ref_dir)
    mov_root = Path(args.mov_dir)

    ref_tracks = _load_all_tracks(ref_root)
    mov_tracks = _load_all_tracks(mov_root)

    # Transform moving channel into reference coordinate space
    mov_tracks_t = _transform_coords(mov_tracks, tform)
    log.info("Transformed %d moving-channel localizations", len(mov_tracks_t))

    # Save transformed tracks
    mov_tracks_t.to_csv(out_dir / "mov_tracks_transformed.csv", index=False)

    # -- Per-frame pairing --
    # Pair within each movie separately (movies in ch1 and ch2 should correspond)
    ref_movies = ref_tracks["movie"].unique()
    mov_movies = mov_tracks_t["movie"].unique()

    all_frame_pairs = []
    all_codiff = []
    all_dispcorr = []

    for rmov in ref_movies:
        # Match movie names: strip trailing suffixes that might differ
        ref_sub = ref_tracks[ref_tracks["movie"] == rmov]
        # Try to find corresponding movie in moving channel
        # Common pattern: same base name, tracked independently
        mmov = rmov  # assume same name; if dirs differ, user should ensure consistency
        mov_sub = mov_tracks_t[mov_tracks_t["movie"] == mmov]
        if mov_sub.empty:
            log.warning("No matching movie '%s' in moving channel; skipping", mmov)
            continue

        fp = _pair_per_frame(ref_sub, mov_sub, max_dist_px=coloc_radius)
        if fp.empty:
            log.info("Movie %s: 0 frame-level pairs", rmov)
            continue
        fp["movie"] = rmov
        log.info("Movie %s: %d frame-level pairs", rmov, len(fp))

        cd = _identify_codiffusing_pairs(fp, min_coframes=args.min_coframes)
        if not cd.empty:
            cd["movie"] = rmov
            log.info("Movie %s: %d co-diffusing track pairs", rmov, len(cd))

        dc = _displacement_correlation(fp, ref_sub, mov_sub)
        if not dc.empty:
            dc["movie"] = rmov

        all_frame_pairs.append(fp)
        all_codiff.append(cd)
        all_dispcorr.append(dc)

    frame_pairs_all = pd.concat(all_frame_pairs, ignore_index=True) if all_frame_pairs else pd.DataFrame()
    codiff_all = pd.concat(all_codiff, ignore_index=True) if all_codiff else pd.DataFrame()
    dispcorr_all = pd.concat(all_dispcorr, ignore_index=True) if all_dispcorr else pd.DataFrame()

    frame_pairs_all.to_csv(out_dir / "frame_pairs.csv", index=False)
    codiff_all.to_csv(out_dir / "codiffusing_pairs.csv", index=False)
    dispcorr_all.to_csv(out_dir / "displacement_correlation.csv", index=False)

    # -- Save pairing metadata --
    pair_meta = {
        "transform_file": str(args.transform),
        "ref_dir": str(args.ref_dir),
        "mov_dir": str(args.mov_dir),
        "coloc_radius_px": float(coloc_radius),
        "coloc_radius_nm": float(coloc_radius * tf_meta["pixel_nm"]),
        "min_coframes": int(args.min_coframes),
        "mpp": float(args.mpp),
        "fps": float(args.fps),
        "n_frame_pairs": int(len(frame_pairs_all)),
        "n_codiffusing_pairs": int(len(codiff_all)),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "pair_meta.json", "w") as fh:
        json.dump(pair_meta, fh, indent=2)

    # -- Diagnostic plots --
    _plot_pairing_diagnostics(
        frame_pairs_all, codiff_all, dispcorr_all,
        ref_tracks, mov_tracks_t,
        coloc_radius, tf_meta["pixel_nm"], args.fps,
        out_dir,
    )

    log.info("Pairing complete. Outputs in %s", out_dir)


def _plot_pairing_diagnostics(
    frame_pairs, codiff, dispcorr,
    ref_tracks, mov_tracks,
    coloc_radius, pixel_nm, fps,
    out_dir,
):
    """Generate diagnostic figures for co-diffusion pairing."""
    # 1. Pair distance histogram
    if not frame_pairs.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(
            frame_pairs["pair_dist_px"] * pixel_nm,
            bins=50, edgecolor="k", alpha=0.7,
        )
        ax.axvline(
            coloc_radius * pixel_nm, color="red", ls="--",
            label=f"radius={coloc_radius * pixel_nm:.0f} nm",
        )
        ax.set_xlabel("Pair distance (nm)")
        ax.set_ylabel("Count")
        ax.set_title("Frame-level pair distances")
        ax.legend(fontsize=8)
        _save_fig(fig, str(out_dir / "pair_distance_histogram.png"))

    # 2. Co-diffusion duration histogram
    if not codiff.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        durations = codiff["n_coframes"] / fps
        ax.hist(durations * 1000, bins=range(0, int(durations.max() * 1000) + 100, 20), edgecolor="k", alpha=0.7)
        ax.set_xlabel("Co-diffusion duration (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"Co-diffusion residence times (n={len(codiff)} pairs)")
        _save_fig(fig, str(out_dir / "codiffusion_duration_histogram.png"))

    # 3. Displacement correlation scatter
    if not dispcorr.empty and "cos_similarity" in dispcorr.columns:
        valid = dispcorr.dropna(subset=["cos_similarity"])
        if not valid.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].hist(valid["cos_similarity"], bins=50, edgecolor="k", alpha=0.7)
            axes[0].axvline(0, color="gray", ls=":")
            axes[0].set_xlabel("Cosine similarity")
            axes[0].set_ylabel("Count")
            axes[0].set_title("Displacement vector correlation")

            axes[1].scatter(
                valid["mag_ref"] * pixel_nm,
                valid["mag_mov"] * pixel_nm,
                c=valid["cos_similarity"], cmap="coolwarm", vmin=-1, vmax=1,
                s=4, alpha=0.5,
            )
            axes[1].plot(
                [0, valid["mag_ref"].max() * pixel_nm],
                [0, valid["mag_ref"].max() * pixel_nm],
                "k--", lw=0.5,
            )
            axes[1].set_xlabel("Ref step size (nm)")
            axes[1].set_ylabel("Mov step size (nm)")
            axes[1].set_title("Step magnitude correlation")
            cb = fig.colorbar(axes[1].collections[0], ax=axes[1], shrink=0.8)
            cb.set_label("Cosine similarity")
            _save_fig(fig, str(out_dir / "displacement_correlation.png"))

    # 4. Spatial map of co-diffusing pairs
    if not codiff.empty and not frame_pairs.empty:
        fig, ax = plt.subplots(figsize=(8, 8))
        # Plot all localizations faintly
        ax.scatter(ref_tracks["x"], ref_tracks["y"], s=0.1, c="gray", alpha=0.05)
        # Highlight co-diffusing tracks
        for _, row in codiff.iterrows():
            pr, pm = row["particle_ref"], row["particle_mov"]
            sub = frame_pairs[
                (frame_pairs["particle_ref"] == pr) & (frame_pairs["particle_mov"] == pm)
            ].sort_values("frame")
            if len(sub) >= 2:
                ax.plot(sub["x_ref"], sub["y_ref"], "r-", lw=0.8, alpha=0.7)
                ax.plot(sub["x_mov"], sub["y_mov"], "g-", lw=0.8, alpha=0.7)
        ax.set_aspect("equal")
        ax.set_title(f"Co-diffusing pair trajectories (n={len(codiff)})")
        ax.invert_yaxis()
        _save_fig(fig, str(out_dir / "codiffusion_spatial_map.png"))


# ---------------------------------------------------------------------------
# ANALYZE subcommand -- statistics and shuffle controls
# ---------------------------------------------------------------------------

def _shuffle_control(
    ref_tracks: pd.DataFrame,
    mov_tracks: pd.DataFrame,
    coloc_radius: float,
    min_coframes: int,
    n_shuffles: int = 200,
) -> Dict[str, Any]:
    """Estimate random co-localization rate by circular frame-shifting.

    For each shuffle iteration, the moving-channel frame indices are shifted
    by a random offset (wrapping around), and co-diffusion pairing is
    repeated. This preserves the spatial structure and track continuity of
    both channels while destroying any true temporal correlation.
    """
    max_frame = max(ref_tracks["frame"].max(), mov_tracks["frame"].max())
    rng = np.random.default_rng(42)

    n_pairs_shuffled = []
    n_codiff_shuffled = []
    for i in range(n_shuffles):
        offset = rng.integers(1, max_frame)
        mov_shifted = mov_tracks.copy()
        mov_shifted["frame"] = (mov_shifted["frame"] + offset) % (max_frame + 1)

        fp = _pair_per_frame(ref_tracks, mov_shifted, max_dist_px=coloc_radius)
        n_pairs_shuffled.append(len(fp))
        if not fp.empty:
            cd = _identify_codiffusing_pairs(fp, min_coframes=min_coframes)
            n_codiff_shuffled.append(len(cd))
        else:
            n_codiff_shuffled.append(0)

        if (i + 1) % 50 == 0:
            log.info("  Shuffle %d/%d", i + 1, n_shuffles)

    return {
        "n_shuffles": n_shuffles,
        "frame_pairs_per_shuffle": n_pairs_shuffled,
        "codiff_pairs_per_shuffle": n_codiff_shuffled,
        "mean_frame_pairs": float(np.mean(n_pairs_shuffled)),
        "std_frame_pairs": float(np.std(n_pairs_shuffled)),
        "mean_codiff_pairs": float(np.mean(n_codiff_shuffled)),
        "std_codiff_pairs": float(np.std(n_codiff_shuffled)),
    }


def cmd_analyze(args):
    """Compute co-diffusion statistics with shuffle-based random-overlap controls."""
    out_dir = Path(args.coloc_dir)

    # Load pairing outputs
    meta_path = out_dir / "pair_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"pair_meta.json not found in {out_dir}. Run 'pair' first.")
    with open(meta_path) as fh:
        pair_meta = json.load(fh)

    frame_pairs_path = out_dir / "frame_pairs.csv"
    codiff_path = out_dir / "codiffusing_pairs.csv"
    dispcorr_path = out_dir / "displacement_correlation.csv"

    frame_pairs = pd.read_csv(frame_pairs_path, low_memory=False) if frame_pairs_path.exists() else pd.DataFrame()
    codiff = pd.read_csv(codiff_path, low_memory=False) if codiff_path.exists() else pd.DataFrame()
    dispcorr = pd.read_csv(dispcorr_path, low_memory=False) if dispcorr_path.exists() else pd.DataFrame()

    mpp = pair_meta.get("mpp", 0.122)
    fps = pair_meta.get("fps", 50)
    pixel_nm = pair_meta.get("coloc_radius_nm", 0) / pair_meta.get("coloc_radius_px", 1) if pair_meta.get("coloc_radius_px", 0) > 0 else 122.0
    coloc_radius = pair_meta["coloc_radius_px"]

    stats_out = {
        "total_frame_pairs": int(len(frame_pairs)),
        "total_codiff_pairs": int(len(codiff)),
        "coloc_radius_px": coloc_radius,
        "coloc_radius_nm": coloc_radius * pixel_nm,
    }

    # Displacement correlation summary
    if not dispcorr.empty and "cos_similarity" in dispcorr.columns:
        valid_cos = dispcorr["cos_similarity"].dropna()
        stats_out["displacement_correlation"] = {
            "n_steps": int(len(valid_cos)),
            "mean_cos_similarity": float(valid_cos.mean()),
            "median_cos_similarity": float(valid_cos.median()),
            "fraction_cos_gt_0.5": float((valid_cos > 0.5).mean()),
            "fraction_cos_gt_0": float((valid_cos > 0).mean()),
        }

    # Co-diffusion duration summary
    if not codiff.empty:
        durations_s = codiff["n_coframes"] / fps
        stats_out["codiffusion_duration"] = {
            "n_pairs": int(len(codiff)),
            "mean_frames": float(codiff["n_coframes"].mean()),
            "median_frames": float(codiff["n_coframes"].median()),
            "max_frames": int(codiff["n_coframes"].max()),
            "mean_ms": float(durations_s.mean() * 1000),
            "median_ms": float(durations_s.median() * 1000),
            "mean_coframe_fraction": float(codiff["coframe_fraction"].mean()),
        }

    # -- Shuffle control --
    if args.n_shuffles > 0:
        log.info("Running %d shuffle iterations for random-overlap control...", args.n_shuffles)

        # Reload tracks for shuffling
        ref_root = Path(pair_meta["ref_dir"])
        mov_root = Path(pair_meta["mov_dir"])
        tform, tf_meta = _load_transform(pair_meta["transform_file"])

        ref_tracks = _load_all_tracks(ref_root)
        mov_tracks = _load_all_tracks(mov_root)
        mov_tracks_t = _transform_coords(mov_tracks, tform)

        # Run per-movie
        ref_movies = ref_tracks["movie"].unique()
        agg_shuffle = {"frame_pairs_per_shuffle": [], "codiff_pairs_per_shuffle": []}
        for rmov in ref_movies:
            ref_sub = ref_tracks[ref_tracks["movie"] == rmov]
            mov_sub = mov_tracks_t[mov_tracks_t["movie"] == rmov]
            if mov_sub.empty:
                continue
            shuf = _shuffle_control(
                ref_sub, mov_sub,
                coloc_radius=coloc_radius,
                min_coframes=args.min_coframes or pair_meta.get("min_coframes", 3),
                n_shuffles=args.n_shuffles,
            )
            # Accumulate across movies
            for i in range(args.n_shuffles):
                if i >= len(agg_shuffle["frame_pairs_per_shuffle"]):
                    agg_shuffle["frame_pairs_per_shuffle"].append(0)
                    agg_shuffle["codiff_pairs_per_shuffle"].append(0)
                agg_shuffle["frame_pairs_per_shuffle"][i] += shuf["frame_pairs_per_shuffle"][i]
                agg_shuffle["codiff_pairs_per_shuffle"][i] += shuf["codiff_pairs_per_shuffle"][i]

        shuffle_fp = np.array(agg_shuffle["frame_pairs_per_shuffle"])
        shuffle_cd = np.array(agg_shuffle["codiff_pairs_per_shuffle"])

        actual_fp = stats_out["total_frame_pairs"]
        actual_cd = stats_out["total_codiff_pairs"]

        stats_out["shuffle_control"] = {
            "n_shuffles": args.n_shuffles,
            "frame_pairs_shuffle_mean": float(shuffle_fp.mean()),
            "frame_pairs_shuffle_std": float(shuffle_fp.std()),
            "frame_pairs_actual": actual_fp,
            "frame_pairs_enrichment": float(actual_fp / shuffle_fp.mean()) if shuffle_fp.mean() > 0 else float("inf"),
            "frame_pairs_z_score": float((actual_fp - shuffle_fp.mean()) / shuffle_fp.std()) if shuffle_fp.std() > 0 else float("inf"),
            "codiff_pairs_shuffle_mean": float(shuffle_cd.mean()),
            "codiff_pairs_shuffle_std": float(shuffle_cd.std()),
            "codiff_pairs_actual": actual_cd,
            "codiff_pairs_enrichment": float(actual_cd / shuffle_cd.mean()) if shuffle_cd.mean() > 0 else float("inf"),
            "codiff_pairs_z_score": float((actual_cd - shuffle_cd.mean()) / shuffle_cd.std()) if shuffle_cd.std() > 0 else float("inf"),
            "codiff_p_value_upper": float((shuffle_cd >= actual_cd).mean()),
        }

        # Plot shuffle distributions
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(shuffle_fp, bins=30, edgecolor="k", alpha=0.6, label="Shuffled")
        axes[0].axvline(actual_fp, color="red", lw=2, label=f"Actual ({actual_fp})")
        axes[0].set_xlabel("Frame-level pairs")
        axes[0].set_ylabel("Shuffle iterations")
        axes[0].set_title("Frame pairs: actual vs. random")
        axes[0].legend(fontsize=8)

        axes[1].hist(shuffle_cd, bins=30, edgecolor="k", alpha=0.6, label="Shuffled")
        axes[1].axvline(actual_cd, color="red", lw=2, label=f"Actual ({actual_cd})")
        axes[1].set_xlabel("Co-diffusing pairs")
        axes[1].set_ylabel("Shuffle iterations")
        axes[1].set_title("Co-diffusing pairs: actual vs. random")
        axes[1].legend(fontsize=8)

        _save_fig(fig, str(out_dir / "shuffle_control.png"))

    # Save summary
    stats_path = out_dir / "codiffusion_stats.json"
    with open(stats_path, "w") as fh:
        json.dump(_to_jsonable(stats_out), fh, indent=2)
    log.info("Analysis summary saved to %s", stats_path)

    # Print summary to console
    print("\n" + "=" * 60)
    print("CO-DIFFUSION SUMMARY")
    print("=" * 60)
    print(f"  Frame-level pairs:     {stats_out['total_frame_pairs']}")
    print(f"  Co-diffusing pairs:    {stats_out['total_codiff_pairs']}")
    print(f"  Co-loc radius:         {coloc_radius:.2f} px ({coloc_radius * pixel_nm:.0f} nm)")
    if "codiffusion_duration" in stats_out:
        d = stats_out["codiffusion_duration"]
        print(f"  Mean co-diff duration: {d['mean_ms']:.0f} ms ({d['mean_frames']:.1f} frames)")
    if "displacement_correlation" in stats_out:
        dc = stats_out["displacement_correlation"]
        print(f"  Mean cos(theta):       {dc['mean_cos_similarity']:.3f}")
        print(f"  Fraction cos > 0:      {dc['fraction_cos_gt_0']:.2%}")
    if "shuffle_control" in stats_out:
        sc = stats_out["shuffle_control"]
        print(f"  Shuffle enrichment:    {sc['codiff_pairs_enrichment']:.1f}x")
        print(f"  Shuffle z-score:       {sc['codiff_pairs_z_score']:.1f}")
        print(f"  Shuffle p (upper):     {sc['codiff_p_value_upper']:.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two-color SPT co-diffusion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- register --
    reg = subparsers.add_parser("register", help="Compute affine transform from bead image")
    reg.add_argument("bead_image", help="Path to two-channel bead TIF (Z-stack or single plane)")
    reg.add_argument("-o", "--output-dir", default="registration", help="Output directory")
    reg.add_argument("--pixel-nm", type=float, default=122.0, help="Camera pixel size in nm")
    reg.add_argument("--psf-nm", type=float, default=250.0, help="Approximate PSF sigma in nm")
    reg.add_argument("--ref-channel", type=int, default=0, help="Reference channel index (0 or 1)")
    reg.add_argument("--threshold-sigma", type=float, default=4.0, help="Detection threshold in sigma above mean")
    reg.add_argument("--min-sep", type=int, default=5, help="Minimum separation between peaks (px)")
    reg.add_argument("--match-radius", type=float, default=5.0, help="Max distance for bead matching after bulk shift (px)")
    reg.set_defaults(func=cmd_register)

    # -- pair --
    pair = subparsers.add_parser("pair", help="Find co-diffusing track pairs across channels")
    pair.add_argument("ref_dir", help="Directory with reference-channel track CSVs (ch1/)")
    pair.add_argument("mov_dir", help="Directory with moving-channel track CSVs (ch2/)")
    pair.add_argument("transform", help="Path to transform.json from register step")
    pair.add_argument("-o", "--output-dir", default="coloc", help="Output directory")
    pair.add_argument("--coloc-radius", type=float, default=None, help="Co-localization radius in px (default: 3x registration error)")
    pair.add_argument("--min-coframes", type=int, default=3, help="Minimum co-localized frames to call co-diffusion")
    pair.add_argument("--mpp", type=float, default=0.122, help="Microns per pixel")
    pair.add_argument("--fps", type=float, default=50.0, help="Frames per second")
    pair.set_defaults(func=cmd_pair)

    # -- analyze --
    ana = subparsers.add_parser("analyze", help="Co-diffusion statistics and shuffle controls")
    ana.add_argument("coloc_dir", help="Directory with pair outputs (from pair step)")
    ana.add_argument("--n-shuffles", type=int, default=200, help="Number of shuffle iterations for control")
    ana.add_argument("--min-coframes", type=int, default=None, help="Override min-coframes for shuffle (default: from pair_meta)")
    ana.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
