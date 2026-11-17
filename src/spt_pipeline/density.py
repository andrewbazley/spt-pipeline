"""Local particle density, cell identity, and mis-linking risk.

Why this exists
---------------
Two cells imaged under the same condition can carry very different GEM
densities -- unequal transfection, a brighter cell next to a dim one in the same
field. Density is not a nuisance you can ignore, because it changes the answer
in two independent ways:

* **Biologically**, a crowded ER is a different environment from a sparse one.
* **Technically**, crowding makes linking ambiguous. When particles are closer
  together than they move between frames, the linker sometimes joins the wrong
  pair, which truncates real tracks and fabricates displacements. That biases
  mobility *downward* in exactly the cells that express most.

So a difference between conditions can be manufactured entirely by a difference
in density. This module measures density, attributes it to cells, quantifies the
linking risk, and lets the comparison be run at matched density.

Where density is measured
-------------------------
On the **pre-link detection table**, before linking and stub filtering. This
matters: linking discards detections it cannot join, and stub filtering discards
short tracks, both of which happen preferentially in crowded fields. Measuring
after linking therefore understates crowding exactly where it is worst.

For movies tracked before this module existed there is no detection table, so
density falls back to the linked tracks and every output is stamped
``density_source = "post_link"``. It is better than nothing and clearly worse
than the real thing.

The two axes
------------
``field density`` -- localizations within ``field_radius_px`` of each
localization, in the same frame. The biology-facing number: how crowded is the
ER around this particle.

``link safety`` -- nearest-neighbour spacing divided by the movie's typical
single-frame displacement. Dimensionless, so it is comparable across
acquisitions with different pixel sizes, frame rates and search ranges. Below
roughly 2-3 the linker is guessing.

The denominator is deliberately the **movie's** median step, not each track's
own. Dividing by a track's own displacement would make every fast track score as
risky, manufacturing precisely the mobility-density correlation this module
exists to rule out.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import utils

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None

#: Radius of the crowding neighbourhood, in pixels. At 0.11 um/px this is a
#: 1.1 um radius (3.8 um^2) -- roughly an ER subdomain, not a whole cell.
DEFAULT_FIELD_RADIUS_PX = 10.0

#: Track column used for density stratification and matching by default.
DEFAULT_MATCH_COLUMN = "track_median_field_localizations_n"

#: Below this, nearest neighbours are closer than a few typical steps and the
#: linker is choosing between candidates rather than following one particle.
LINK_SAFETY_WARN = 3.0

DETECTION_DENSITY_COLUMNS = [
    "density_field_radius_px", "density_search_range_px",
    "field_localizations_n", "field_neighbors_n",
    "field_localizations_per_px2", "field_localizations_per_um2",
    "neighbors_R_n", "neighbors_2R_n", "nn_distance_px",
    "field_window_truncated",
]

TRACK_DENSITY_COLUMNS = [
    "track_field_radius_px",
    "track_median_field_localizations_n", "track_mean_field_localizations_n",
    "track_p90_field_localizations_n",
    "track_median_field_localizations_per_um2",
    "track_p90_field_localizations_per_um2",
    "track_median_neighbors_R_n", "track_median_neighbors_2R_n",
    "track_p90_neighbors_2R_n", "track_fraction_frames_neighbor_2R",
    "track_isolated_2R_all_frames", "track_fraction_field_window_truncated",
    "track_median_nn_distance_px", "link_safety_ratio", "p_link_ambiguous",
    "cell_id", "cell_median_field_localizations_n", "cell_link_safety_ratio",
    "cell_n_tracks", "cell_area_um2", "n_cells_in_field", "density_source",
]


# ---------------------------------------------------------------------------
# Neighbour geometry
# ---------------------------------------------------------------------------

def _neighbor_counts(xy: np.ndarray, radius: float) -> np.ndarray:
    """Number of *other* points within ``radius`` of each row of ``xy``."""
    if xy.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    if cKDTree is None:  # pragma: no cover - scipy is a hard dependency
        d2 = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=2)
        return (d2 <= float(radius) ** 2).sum(axis=1).astype(np.int64) - 1
    tree = cKDTree(xy)
    try:
        counts = tree.query_ball_point(xy, r=float(radius), return_length=True)
    except TypeError:  # pragma: no cover - older scipy
        counts = np.fromiter((len(v) for v in tree.query_ball_point(xy, r=float(radius))),
                             dtype=np.int64, count=xy.shape[0])
    return np.asarray(counts, dtype=np.int64) - 1


def _nn_distance(xy: np.ndarray) -> np.ndarray:
    """Distance to the closest *other* point, per row. NaN when alone."""
    n = xy.shape[0]
    if n < 2:
        return np.full(n, np.nan)
    if cKDTree is None:  # pragma: no cover
        d2 = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        return np.sqrt(d2.min(axis=1))
    d, _ = cKDTree(xy).query(xy, k=2)
    return np.asarray(d[:, 1], dtype=float)


def measure_detection_density(
    detections: pd.DataFrame, *, search_range_px: float,
    field_radius_px: float = DEFAULT_FIELD_RADIUS_PX, mpp: float = 0.11,
    frame_shape: Optional[Tuple[int, int]] = None,
) -> pd.DataFrame:
    """Annotate every localization with the crowding it sat in.

    Counts are strictly within-frame: a localization from another frame is never
    counted as a neighbour, even at identical coordinates. ``frame_shape`` is
    ``(height, width)`` and only flags neighbourhoods clipped by the image edge;
    counts are never extrapolated past the edge, because inventing particles
    outside the field of view would understate nothing and overstate everything.
    """
    search_range_px = float(search_range_px)
    field_radius_px = float(field_radius_px)
    mpp = float(mpp)
    if search_range_px <= 0 or field_radius_px <= 0 or mpp <= 0:
        raise ValueError("search_range_px, field_radius_px and mpp must be > 0")

    out = utils.normalize_track_table(detections)
    if not {"frame", "x", "y"}.issubset(out.columns):
        raise ValueError("detections must contain frame, x and y")
    if out.empty:
        for c in DETECTION_DENSITY_COLUMNS:
            out[c] = pd.Series(dtype=float)
        return out

    out = out.dropna(subset=["frame", "x", "y"]).reset_index(drop=True)
    n = len(out)
    n_r = np.zeros(n, dtype=np.int64)
    n_2r = np.zeros(n, dtype=np.int64)
    n_field = np.zeros(n, dtype=np.int64)
    nn = np.full(n, np.nan)

    for _frame, index in out.groupby("frame", sort=False).groups.items():
        idx = np.asarray(list(index), dtype=np.int64)
        xy = out.loc[idx, ["x", "y"]].to_numpy(dtype=float)
        n_r[idx] = _neighbor_counts(xy, search_range_px)
        n_2r[idx] = _neighbor_counts(xy, 2.0 * search_range_px)
        n_field[idx] = _neighbor_counts(xy, field_radius_px)
        nn[idx] = _nn_distance(xy)

    area_px2 = np.pi * field_radius_px ** 2
    area_um2 = np.pi * (field_radius_px * mpp) ** 2
    localizations = n_field + 1
    out["density_field_radius_px"] = field_radius_px
    out["density_search_range_px"] = search_range_px
    out["field_localizations_n"] = localizations
    out["field_neighbors_n"] = n_field
    out["field_localizations_per_px2"] = localizations / area_px2
    out["field_localizations_per_um2"] = localizations / area_um2
    out["neighbors_R_n"] = n_r
    out["neighbors_2R_n"] = n_2r
    out["nn_distance_px"] = nn

    if frame_shape is None:
        out["field_window_truncated"] = False
    else:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        x = out["x"].to_numpy(float); y = out["y"].to_numpy(float)
        out["field_window_truncated"] = (
            (x < field_radius_px) | (x > (w - 1 - field_radius_px)) |
            (y < field_radius_px) | (y > (h - 1 - field_radius_px)))
    return out


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellSegmentation:
    """Labelled cells in one field of view, and how confident we are."""
    labels: np.ndarray              # 2-D label image on the binned grid
    bin_px: float                   # grid spacing, pixels
    n_cells: int
    occupancy: float                # fraction of the frame called cell
    note: str                       # plain-language caveat, or ""

    @property
    def resolved(self) -> bool:
        """False when the field could not be split into separate cells."""
        return self.n_cells > 1


def _otsu(values: np.ndarray) -> float:
    """Otsu threshold, implemented here to avoid a scikit-image dependency."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0
    hist, edges = np.histogram(v, bins=256)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    with np.errstate(invalid="ignore", divide="ignore"):
        m0 = np.cumsum(hist * centres) / w0
        m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / w1
        between = w0 * w1 * (m0 - m1) ** 2
    between[~np.isfinite(between)] = -np.inf
    return float(centres[int(np.argmax(between))])


def segment_cells(
    detections: pd.DataFrame, frame_shape: Tuple[int, int], *, mpp: float,
    bin_um: float = 1.0, smooth_um: float = 2.5, min_cell_area_um2: float = 25.0,
    confluent_occupancy: float = 0.60,
) -> CellSegmentation:
    """Find cells by where localizations accumulate over the whole movie.

    Collapse every localization onto a coarse grid, smooth it, threshold with
    Otsu, and label connected components. Cells are where GEMs are; background
    is where they are not. No fluorescence image is needed.

    **When cells touch, they become one label, and that is reported rather than
    forced apart.** A confluent field of identical-density cells genuinely is one
    density field for our purposes, and splitting it with a watershed would
    invent boundaries the data does not contain. ``note`` says so, and
    ``resolved`` is False, so a downstream per-cell comparison can decline to
    treat the result as separate cells.
    """
    from scipy import ndimage

    h, w = int(frame_shape[0]), int(frame_shape[1])
    bin_px = max(1.0, float(bin_um) / float(mpp))
    ny, nx = max(1, int(np.ceil(h / bin_px))), max(1, int(np.ceil(w / bin_px)))
    d = utils.normalize_track_table(detections)
    if d.empty or not {"x", "y"}.issubset(d.columns):
        return CellSegmentation(np.zeros((ny, nx), dtype=int), bin_px, 0, 0.0,
                                "no localizations")
    x = pd.to_numeric(d["x"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(d["y"], errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    counts, _, _ = np.histogram2d(y[ok], x[ok], bins=[ny, nx],
                                  range=[[0, h], [0, w]])
    sigma = max(0.5, float(smooth_um) / float(bin_um))
    smooth = ndimage.gaussian_filter(counts.astype(float), sigma=sigma)
    if not np.isfinite(smooth).any() or smooth.max() <= 0:
        return CellSegmentation(np.zeros((ny, nx), dtype=int), bin_px, 0, 0.0,
                                "no localizations")

    # Threshold on log1p, not on the raw counts. Otsu on raw counts splits the
    # *brightest* structure from everything else, so a sparse cell beside a
    # crowded one gets classified as background -- discarding exactly the
    # comparison this module exists to enable. The log compresses the
    # cell-to-cell density difference while leaving the gap between "some
    # localizations" and "none" intact, so the split lands on cell vs background.
    log_img = np.log1p(smooth)
    thr = _otsu(log_img[log_img > 0])
    mask = log_img > max(thr, np.log1p(smooth.max()) * 0.02)
    mask = ndimage.binary_closing(mask, np.ones((3, 3)))
    mask = ndimage.binary_fill_holes(mask)
    labels, n = ndimage.label(mask)

    # Drop specks: debris and stray detections are not cells.
    px_area_um2 = (bin_px * mpp) ** 2
    if n:
        sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
        too_small = np.where(sizes * px_area_um2 < float(min_cell_area_um2))[0] + 1
        for lab in too_small:
            labels[labels == lab] = 0
        # Renumber so ids are contiguous.
        keep = [l for l in range(1, n + 1) if l not in set(too_small.tolist())]
        remap = {old: i + 1 for i, old in enumerate(keep)}
        out = np.zeros_like(labels)
        for old, new in remap.items():
            out[labels == old] = new
        labels, n = out, len(keep)

    occupancy = float((labels > 0).mean())
    note = ""
    # A grid only a handful of bins across cannot resolve anything: the
    # smoothing kernel spans the whole field and every cell merges. Say so
    # rather than returning a confident-looking single label.
    if min(ny, nx) < 20:
        note = (f"binned grid is only {nx}x{ny}; too coarse to separate cells "
                f"(field is {w * mpp:.0f}x{h * mpp:.0f} um at bin_um={bin_um:g}). "
                f"Treated as one density field.")
        return CellSegmentation(labels, bin_px, min(int(n), 1), occupancy, note)
    if n == 0:
        note = "no region passed the cell-size threshold"
    elif n == 1 and occupancy >= confluent_occupancy:
        note = (f"one region covering {occupancy:.0%} of the field: cells are "
                f"touching or confluent, treated as a single density field")
    elif n == 1:
        note = "a single cell in the field"
    return CellSegmentation(labels, bin_px, int(n), occupancy, note)


def assign_cells(df: pd.DataFrame, seg: CellSegmentation) -> pd.Series:
    """Cell label for each row of a table carrying ``x``/``y``. 0 = background."""
    if df.empty or seg.labels.size == 0:
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    x = pd.to_numeric(df["x"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(float)
    ny, nx = seg.labels.shape
    ix = np.clip(np.floor(x / seg.bin_px), 0, nx - 1)
    iy = np.clip(np.floor(y / seg.bin_px), 0, ny - 1)
    good = np.isfinite(ix) & np.isfinite(iy)
    out = np.zeros(len(df), dtype=int)
    out[good] = seg.labels[iy[good].astype(int), ix[good].astype(int)]
    return pd.Series(out, index=df.index)


def cell_summary(tracks: pd.DataFrame, seg: CellSegmentation, *, mpp: float,
                 movie_id: str = "") -> pd.DataFrame:
    """One row per detected cell: area, track count, density."""
    if seg.n_cells == 0:
        return pd.DataFrame(columns=["movie_id", "cell_id", "cell_area_um2",
                                     "cell_n_tracks", "cell_median_field_localizations_n",
                                     "cell_tracks_per_um2", "n_cells_in_field",
                                     "cell_segmentation_note"])
    px_area_um2 = (seg.bin_px * mpp) ** 2
    rows = []
    per_track = (tracks.drop_duplicates("particle")
                 if "particle" in tracks.columns else tracks)
    for cid in range(1, seg.n_cells + 1):
        area = float((seg.labels == cid).sum() * px_area_um2)
        sub = per_track[per_track.get("cell_id", pd.Series(dtype=int)) == cid] \
            if "cell_id" in per_track.columns else per_track.iloc[0:0]
        med = (float(pd.to_numeric(sub["track_median_field_localizations_n"],
                                   errors="coerce").median())
               if "track_median_field_localizations_n" in sub.columns and len(sub)
               else np.nan)
        rows.append({"movie_id": movie_id, "cell_id": cid, "cell_area_um2": area,
                     "cell_n_tracks": int(len(sub)),
                     "cell_median_field_localizations_n": med,
                     "cell_tracks_per_um2": (len(sub) / area) if area > 0 else np.nan,
                     "n_cells_in_field": seg.n_cells,
                     "cell_segmentation_note": seg.note})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-track summaries
# ---------------------------------------------------------------------------

def movie_step_scale_px(tracks: pd.DataFrame) -> float:
    """Median single-frame displacement in a movie, in pixels.

    The denominator of the link-safety ratio. A *movie-level* scale on purpose:
    using each track's own displacement would score every fast track as risky
    and so bake a mobility-density correlation into the QC metric itself.
    """
    t = utils.normalize_track_table(tracks)
    if t.empty or not {"particle", "frame", "x", "y"}.issubset(t.columns):
        return np.nan
    t = t.sort_values(["particle", "frame"])
    dx = t.groupby("particle")["x"].diff()
    dy = t.groupby("particle")["y"].diff()
    df_ = t.groupby("particle")["frame"].diff()
    step = np.hypot(dx, dy)[df_ == 1]
    step = step[np.isfinite(step)]
    return float(np.median(step)) if step.size else np.nan


def summarize_track_density(tracks: pd.DataFrame, *, step_scale_px: float = np.nan,
                            density_source: str = "pre_link") -> pd.DataFrame:
    """One crowding summary row per trajectory."""
    t = utils.normalize_track_table(tracks)
    needed = {"particle", "field_localizations_n", "neighbors_R_n", "neighbors_2R_n"}
    if t.empty or not needed.issubset(t.columns):
        return pd.DataFrame(columns=["particle", *TRACK_DENSITY_COLUMNS])

    def _num(g, col):
        if col not in g.columns:
            return np.array([])
        v = pd.to_numeric(g[col], errors="coerce").to_numpy(float)
        return v[np.isfinite(v)]

    med = lambda v: float(np.median(v)) if v.size else np.nan          # noqa: E731
    mean = lambda v: float(np.mean(v)) if v.size else np.nan           # noqa: E731
    p90 = lambda v: float(np.percentile(v, 90)) if v.size else np.nan  # noqa: E731

    rows = []
    for particle, g in t.groupby("particle", sort=False, dropna=True):
        field_n = _num(g, "field_localizations_n")
        field_um2 = _num(g, "field_localizations_per_um2")
        nr = _num(g, "neighbors_R_n")
        n2r = _num(g, "neighbors_2R_n")
        nn = _num(g, "nn_distance_px")
        radius = _num(g, "density_field_radius_px")
        trunc = pd.Series(g.get("field_window_truncated", False)).fillna(False).astype(bool)
        nn_med = med(nn)
        nr_med = med(nr)
        rows.append({
            "particle": particle,
            "track_field_radius_px": med(radius),
            "track_median_field_localizations_n": med(field_n),
            "track_mean_field_localizations_n": mean(field_n),
            "track_p90_field_localizations_n": p90(field_n),
            "track_median_field_localizations_per_um2": med(field_um2),
            "track_p90_field_localizations_per_um2": p90(field_um2),
            "track_median_neighbors_R_n": nr_med,
            "track_median_neighbors_2R_n": med(n2r),
            "track_p90_neighbors_2R_n": p90(n2r),
            "track_fraction_frames_neighbor_2R": (float(np.mean(n2r > 0))
                                                  if n2r.size else np.nan),
            "track_isolated_2R_all_frames": (bool(np.all(n2r == 0))
                                             if n2r.size else False),
            "track_fraction_field_window_truncated": float(trunc.mean()),
            "track_median_nn_distance_px": nn_med,
            # Dimensionless linking headroom: how many typical steps separate a
            # particle from its nearest competitor.
            "link_safety_ratio": (nn_med / step_scale_px
                                  if np.isfinite(nn_med) and np.isfinite(step_scale_px)
                                  and step_scale_px > 0 else np.nan),
            # Poisson probability that at least one competitor sits inside the
            # linker's search disk.
            "p_link_ambiguous": (float(1.0 - np.exp(-max(nr_med, 0.0)))
                                 if np.isfinite(nr_med) else np.nan),
            "cell_id": (int(pd.to_numeric(g["cell_id"], errors="coerce").median())
                        if "cell_id" in g.columns and g["cell_id"].notna().any() else 0),
            "density_source": density_source,
        })
    return pd.DataFrame(rows, columns=["particle", *TRACK_DENSITY_COLUMNS])


def measure_movie(
    detections: Optional[pd.DataFrame], tracks: pd.DataFrame, *,
    search_range_px: float, mpp: float, movie_id: str = "",
    field_radius_px: float = DEFAULT_FIELD_RADIUS_PX,
    frame_shape: Optional[Tuple[int, int]] = None,
    segment: bool = True,
) -> Dict[str, Any]:
    """Density, cells and linking risk for one movie.

    ``detections`` is the pre-link table when available. Passing ``None`` falls
    back to the linked tracks and stamps ``density_source = "post_link"``, which
    understates crowding because unlinked and stub-filtered detections are
    already gone. The flag travels with every row so the compromise is never
    invisible in a downstream figure.
    """
    source = "pre_link" if detections is not None and len(detections) else "post_link"
    base = detections if source == "pre_link" else tracks
    if base is None or len(base) == 0:
        return {"per_track": pd.DataFrame(columns=["particle", *TRACK_DENSITY_COLUMNS]),
                "cells": pd.DataFrame(), "segmentation": None,
                "density_source": source, "step_scale_px": np.nan}

    if frame_shape is None:
        b = utils.normalize_track_table(base)
        frame_shape = (int(np.ceil(pd.to_numeric(b["y"], errors="coerce").max())) + 1,
                       int(np.ceil(pd.to_numeric(b["x"], errors="coerce").max())) + 1)

    dens = measure_detection_density(base, search_range_px=search_range_px,
                                     field_radius_px=field_radius_px, mpp=mpp,
                                     frame_shape=frame_shape)
    seg = (segment_cells(dens, frame_shape, mpp=mpp) if segment
           else CellSegmentation(np.zeros((1, 1), int), 1.0, 0, 0.0, "segmentation off"))

    # Carry density onto the linked tracks. When density came from the linked
    # tracks themselves this is a no-op join; when it came from the pre-link
    # detections, each track's localizations are matched on (frame, x, y).
    tl = utils.normalize_track_table(tracks)
    if source == "pre_link":
        key = ["frame", "x", "y"]
        cols = key + [c for c in DETECTION_DENSITY_COLUMNS if c in dens.columns]
        tl = tl.merge(dens[cols].drop_duplicates(key), on=key, how="left")
    else:
        tl = dens
    if seg.n_cells:
        tl["cell_id"] = assign_cells(tl, seg)

    step_scale = movie_step_scale_px(tl)
    per_track = summarize_track_density(tl, step_scale_px=step_scale,
                                        density_source=source)
    if not per_track.empty:
        per_track.insert(1, "movie_id", movie_id)
        per_track = attach_cell_density(per_track, seg, mpp=mpp)
    cells = cell_summary(tl.merge(per_track, on="particle", how="left",
                                  suffixes=("", "_t")) if not per_track.empty else tl,
                         seg, mpp=mpp, movie_id=movie_id)
    return {"per_track": per_track, "cells": cells, "segmentation": seg,
            "density_source": source, "step_scale_px": step_scale,
            "annotated_tracks": tl}


def attach_cell_density(per_track: pd.DataFrame, seg: "CellSegmentation",
                        *, mpp: float) -> pd.DataFrame:
    """Give every track the density of the *cell* it came from.

    Per-track local density and cell density answer different questions, and
    the difference matters. Mis-linking risk is largely a property of the whole
    cell: a track at the sparse rim of a crowded cell was still linked in a
    crowded field, because the linker considered every particle in the frame,
    not just the ones nearby. Stratifying only on a track's own neighbourhood
    would put that rim track in a low-density bin while it still carries the
    crowded cell's depressed mobility -- and the confound would survive the
    correction.

    So both are exposed. ``track_median_field_localizations_n`` asks "how
    crowded was this particle's immediate surroundings"; the ``cell_*`` columns
    ask "how crowded was the cell it lived in". For comparing cells of similar
    expression, the cell-level columns are the right axis.
    """
    if per_track.empty:
        return per_track
    out = per_track.copy()
    for c in ("cell_median_field_localizations_n", "cell_link_safety_ratio",
              "cell_n_tracks", "cell_area_um2", "n_cells_in_field"):
        out[c] = np.nan
    out["n_cells_in_field"] = int(getattr(seg, "n_cells", 0) or 0)
    if "cell_id" not in out.columns:
        return out
    px_area_um2 = (getattr(seg, "bin_px", 1.0) * mpp) ** 2
    labels = getattr(seg, "labels", None)
    for cid, g in out.groupby("cell_id"):
        if not cid:            # 0 = background; leave as NaN rather than guess
            continue
        m = out["cell_id"] == cid
        out.loc[m, "cell_median_field_localizations_n"] = float(pd.to_numeric(
            g["track_median_field_localizations_n"], errors="coerce").median())
        out.loc[m, "cell_link_safety_ratio"] = float(pd.to_numeric(
            g["link_safety_ratio"], errors="coerce").median())
        out.loc[m, "cell_n_tracks"] = int(len(g))
        if labels is not None and labels.size:
            out.loc[m, "cell_area_um2"] = float((labels == cid).sum() * px_area_um2)
    # A field that could not be split is one cell by definition, so the whole
    # field becomes the cell -- which is the honest reading, not a failure.
    if not out["cell_median_field_localizations_n"].notna().any():
        out["cell_median_field_localizations_n"] = float(pd.to_numeric(
            out["track_median_field_localizations_n"], errors="coerce").median())
        out["cell_link_safety_ratio"] = float(pd.to_numeric(
            out["link_safety_ratio"], errors="coerce").median())
        out["cell_n_tracks"] = int(len(out))
    return out


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DensityMatchResult:
    assignments: pd.DataFrame
    balance: pd.DataFrame


def _particle_token(value) -> str:
    try:
        f = float(value)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(value)


def track_uid(movie_id, particle) -> str:
    """Stable key; particle ids collide between movies, this does not."""
    return f"{movie_id}::{_particle_token(particle)}"


def add_track_uid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["track_uid"] = [track_uid(m, p) for m, p in zip(out["movie_id"], out["particle"])]
    return out


def density_strata(values: pd.Series, mode: str = "log2") -> pd.Series:
    """Bin a density-like column into comparable strata.

    ``log2`` uses power-of-two bins, which suits counts spanning orders of
    magnitude and keeps boundaries identical across conditions -- essential,
    since quantile bins would put a sparse track from one condition in the same
    stratum as a crowded track from another, defeating the point of matching.
    """
    v = pd.to_numeric(values, errors="coerce").to_numpy(float)
    labels = np.full(len(v), "invalid", dtype=object)
    good = np.isfinite(v) & (v > 0)
    if mode == "log2":
        e = np.floor(np.log2(v[good])).astype(int)
        labels[good] = [f"[{2 ** k:g},{2 ** (k + 1):g})" for k in e]
    else:
        raise ValueError(f"unknown stratum mode: {mode}")
    return pd.Series(labels, index=values.index, dtype="object")


def _length_strata(values: pd.Series, edges: Sequence[int]) -> pd.Series:
    e = sorted({int(x) for x in edges})
    if len(e) < 2:
        raise ValueError("length_bin_edges needs at least two distinct edges")
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=e,
                  right=False, include_lowest=True).astype(str)


def density_match_tracks(
    track_table: pd.DataFrame, *, density_column: str = DEFAULT_MATCH_COLUMN,
    length_bin_edges: Optional[Sequence[int]] = None,
    min_tracks_per_condition_per_stratum: int = 1, seed: int = 1729,
) -> DensityMatchResult:
    """Coarsened exact matching of tracks within each experiment.

    In every experiment and density stratum, keep the same number of tracks from
    every condition -- the smallest condition's count. What survives is a subset
    whose density distribution is the same in each arm by construction, so a
    remaining difference in mobility cannot be a density difference.

    The cost is data: the discarded fraction can be large and is uneven across
    conditions. That is why this is optional and the stratified comparison,
    which keeps everything, is the default.
    """
    required = {"movie_id", "particle", "condition", "experiment", density_column}
    missing = required - set(track_table.columns)
    if missing:
        raise ValueError(f"density matching is missing columns: {sorted(missing)}")
    if min_tracks_per_condition_per_stratum < 1:
        raise ValueError("min_tracks_per_condition_per_stratum must be >= 1")

    cols = list(required)
    if length_bin_edges is not None:
        if "n_spots" not in track_table.columns:
            raise ValueError("length matching requires n_spots")
        cols.append("n_spots")
    tracks = add_track_uid(track_table[cols].drop_duplicates(["movie_id", "particle"]))
    tracks["condition"] = tracks["condition"].astype(str)
    tracks["experiment"] = tracks["experiment"].astype(str)
    tracks["density_match_stratum"] = density_strata(tracks[density_column])
    if length_bin_edges is not None:
        tracks["density_match_stratum"] += "|L=" + _length_strata(
            tracks["n_spots"], length_bin_edges)
    tracks["density_match_selected"] = False
    tracks["density_match_reason"] = "no_shared_stratum"

    rng = np.random.default_rng(int(seed))
    balance_rows: List[dict] = []
    for experiment, exp in tracks.groupby("experiment", sort=True):
        conditions = sorted(exp["condition"].unique())
        if len(conditions) < 2:
            tracks.loc[exp.index, "density_match_reason"] = "one_condition_in_experiment"
            continue
        for stratum, st in exp.groupby("density_match_stratum", sort=True):
            counts = st.groupby("condition").size().reindex(conditions, fill_value=0)
            target = int(counts.min())
            usable = stratum != "invalid" and target >= min_tracks_per_condition_per_stratum
            for condition, before in counts.items():
                balance_rows.append({"experiment": experiment,
                                     "density_match_stratum": stratum,
                                     "condition": condition, "n_before": int(before),
                                     "n_matched": target if usable else 0})
            if not usable:
                continue
            for condition in conditions:
                cand = st.index[st["condition"] == condition].to_numpy()
                chosen = rng.choice(cand, size=target, replace=False)
                tracks.loc[chosen, "density_match_selected"] = True
                tracks.loc[chosen, "density_match_reason"] = "selected"
                rest = np.setdiff1d(cand, chosen, assume_unique=False)
                tracks.loc[rest, "density_match_reason"] = "surplus_in_shared_stratum"

    keep = ["track_uid", "movie_id", "particle", "condition", "experiment",
            density_column, "density_match_stratum", "density_match_selected",
            "density_match_reason"]
    if "n_spots" in tracks.columns:
        keep.insert(6, "n_spots")
    return DensityMatchResult(
        tracks[keep].sort_values(["experiment", "condition", "track_uid"])
        .reset_index(drop=True),
        pd.DataFrame(balance_rows))


def apply_density_match(table: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Filter a track- or displacement-level table by saved match assignments."""
    if table is None or table.empty:
        return table.copy() if table is not None else pd.DataFrame()
    if not {"movie_id", "particle"}.issubset(table.columns):
        raise ValueError("table must contain movie_id and particle")
    t = add_track_uid(table)
    sel = set(assignments.loc[assignments["density_match_selected"].astype(bool),
                              "track_uid"].astype(str))
    return t[t["track_uid"].isin(sel)].drop(columns="track_uid").reset_index(drop=True)


def origin_aligned(tracks: pd.DataFrame, mpp: float, max_tracks: int = 400,
                   seed: int = 0) -> pd.DataFrame:
    """Translate every track to start at the origin, in microns.

    The basis of the 'flower plot': with the starting point removed, the only
    thing left is how far and how tortuously each particle wandered, so two
    populations can be compared by eye without any summary statistic in between.
    """
    t = utils.normalize_track_table(tracks)
    if t.empty or not {"particle", "frame", "x", "y"}.issubset(t.columns):
        return pd.DataFrame(columns=["particle", "frame", "dx_um", "dy_um"])
    t = t.sort_values(["particle", "frame"])
    ids = t["particle"].unique()
    if len(ids) > max_tracks:
        ids = np.random.default_rng(seed).choice(ids, max_tracks, replace=False)
        t = t[t["particle"].isin(ids)]
    g = t.groupby("particle")
    out = pd.DataFrame({
        "particle": t["particle"].to_numpy(),
        "frame": t["frame"].to_numpy(),
        "dx_um": (t["x"] - g["x"].transform("first")).to_numpy() * float(mpp),
        "dy_um": (t["y"] - g["y"].transform("first")).to_numpy() * float(mpp),
    })
    return out
