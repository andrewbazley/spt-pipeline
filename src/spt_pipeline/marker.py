"""Expression-marker intensity: pairing, sampling, and per-track aggregation.

A *marker image* is a single static frame of an expression reporter (FICD,
tdTomato, …) acquired in register with an ER-GEM movie. This module answers one
question per track: how much marker signal surrounded that particle?

Pairing
-------
Marker rows are identified by ``marker`` in the manifest's ``notes``. Each
tracking movie is matched to one of them, in this order:

1. **``marker_image`` column** -- if populated, it is ground truth, full stop.
2. **Stem appendage** -- ``488__001`` pairs with ``488__001_FICD``, within the
   same directory.
3. **Trailing index, same directory only** -- ``488__001.nd2`` pairs with
   ``560__001.nd2`` when both sit in the same folder. The differing prefix is
   the channel; the shared trailing number is the field of view.
4. **Trailing index in a marker subfolder** -- ``C11-157_DMSO/488__001.nd2``
   pairs with ``C11-157_DMSO/FICD_marker/561-Exp__001.nd2``. Only immediate
   subdirectories of the movie's own folder are considered, and only ones whose
   name looks like a marker folder.
5. **Sole occupant** -- a movie folder containing exactly one movie and exactly
   one marker pairs them, no number needed.

Pairing never leaves the movie's own folder or its marker subfolder. Two
condition folders both containing ``488__001`` are different fields of view
that happen to share a filename, and silently pairing across them would attach
one cell's expression level to another cell's tracks. Anything ambiguous is
reported and skipped.

Intensity
---------
The value assigned is the **mean over the detection disk**, not the single
pixel under the localization. A single pixel is shot-noise dominated, and
``rint(x)`` rounding couples the measurement to subpixel bias -- the same
artifact the QC panels exist to detect. Averaging over the ~37 px of a
``diameter=7`` disk cuts noise ~6x and is insensitive to that rounding.

It is computed by convolving the marker image *once* with a normalized disk
kernel, giving a full-resolution image in which every pixel already holds the
disk mean centred there. Assignment is then a single vectorized index per
localization. This is exact -- no spatial binning -- and costs a few
milliseconds regardless of how many localizations there are. A validity mask is
convolved with the same kernel so disks overhanging the frame edge return a
correct partial mean, reported alongside ``frac_disk_in_frame``.

Aggregation
-----------
One value per track, as the mean of its per-localization disk means: the
average marker signal the particle actually experienced along its path. The
spread across the track is kept too, because a track crossing an expression
boundary is a different object from one sitting in uniform signal. Position and
quality columns are recorded so tracks can be filtered by location, intensity,
or edge overlap later without re-measuring anything.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import manifest as manifest_mod
from . import utils

#: Tokens stripped from a stem when testing for the appendage convention.
MARKER_TOKENS = ("marker", "ficd", "tdtomato", "td_tomato", "tdtom", "expression",
                 "expr", "reporter")

#: Per-track columns written by :func:`aggregate_per_track`.
TRACK_COLUMNS = [
    "particle", "marker_intensity", "marker_intensity_median", "marker_intensity_sd",
    "marker_intensity_min", "marker_intensity_max", "n_localizations_measured",
    "mean_x_px", "mean_y_px", "x_min_px", "x_max_px", "y_min_px", "y_max_px",
    "x_range_px", "y_range_px", "dist_to_edge_px", "frac_disk_in_frame",
]

_TRAILING_INDEX = re.compile(r"(\d+)\s*$")


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pairing:
    """One movie's marker assignment, with the reason it was made."""
    movie_id: str
    marker_movie_id: str = ""
    marker_path: str = ""
    method: str = "unmatched"
    detail: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.marker_path)


def _stem(path_text: str) -> str:
    return Path(str(path_text)).stem


def _strip_marker_tokens(stem: str) -> str:
    """Normalize a stem for appendage matching: lowercase, tokens removed."""
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", str(stem).lower()) if p]
    kept = [p for p in parts if p not in MARKER_TOKENS]
    return "".join(kept)


def trailing_index(text: str) -> Optional[str]:
    """The trailing run of digits in a stem, e.g. ``488__001`` -> ``001``.

    Kept as a string with leading zeros intact, then compared numerically, so
    ``001`` and ``1`` match but ``001`` and ``010`` do not.
    """
    m = _TRAILING_INDEX.search(str(text))
    return m.group(1) if m else None


def _same_index(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    try:
        return int(a) == int(b)
    except ValueError:
        return a == b


def _directory(row: pd.Series) -> str:
    """The resolved directory a row's image lives in."""
    sp = str(row.get("source_path", "")).strip()
    return str(Path(sp).parent.resolve()) if sp else ""


def _is_marker_subdir_of(candidate: str, movie_dir: str) -> bool:
    """True when ``candidate`` is an immediate marker-named child of ``movie_dir``.

    Restricted to one level and to folder names containing a marker token, so a
    sibling condition folder can never qualify. ``FICD_marker/`` and
    ``markers/`` do; ``C11-158_Tg/`` does not.
    """
    if not candidate or not movie_dir:
        return False
    c, m = Path(candidate), Path(movie_dir)
    if c.parent != m:
        return False
    name = c.name.lower()
    return any(tok in name for tok in MARKER_TOKENS)


def pair_markers(manifest_df: pd.DataFrame) -> List[Pairing]:
    """Assign a marker image to every trackable movie in the manifest.

    Returns one :class:`Pairing` per movie, including the failures, so the
    caller can report exactly which movies went unmatched and why.
    """
    markers = manifest_mod.marker_rows(manifest_df)
    movies = manifest_mod.movie_rows(manifest_df)
    out: List[Pairing] = []
    if movies.empty:
        return out

    # Index markers by directory once.
    by_dir: Dict[str, List[pd.Series]] = {}
    for _, mrow in markers.iterrows():
        by_dir.setdefault(_directory(mrow), []).append(mrow)
    movies_by_dir: Dict[str, int] = {}
    for _, row in movies.iterrows():
        movies_by_dir[_directory(row)] = movies_by_dir.get(_directory(row), 0) + 1

    for _, row in movies.iterrows():
        mid = str(row["movie_id"])

        # 1. Explicit column wins outright.
        explicit = str(row.get("marker_image", "")).strip()
        if explicit:
            out.append(Pairing(mid, _stem(explicit), explicit, "marker_image column"))
            continue

        d = _directory(row)
        # Markers sitting in the movie folder, plus any in a marker-named
        # subfolder of it (the common "C11-157_DMSO/FICD_marker/" layout).
        candidates = list(by_dir.get(d, []))
        sub_dirs = [k for k in by_dir
                    if k and _is_marker_subdir_of(k, d)]
        for k in sub_dirs:
            candidates.extend(by_dir[k])
        if not candidates:
            out.append(Pairing(mid, method="unmatched",
                               detail=f"no marker image in {d or '(unknown directory)'} "
                                      f"or a marker subfolder of it"))
            continue

        src_stem = _stem(row.get("source_path", "")) or mid

        # 2. Stem appendage, same directory.
        key = _strip_marker_tokens(src_stem)
        named = [m for m in candidates
                 if key and _strip_marker_tokens(_stem(m.get("source_path", ""))) == key]
        if len(named) == 1:
            out.append(Pairing(mid, str(named[0]["movie_id"]),
                               str(named[0]["source_path"]), "stem appendage"))
            continue
        if len(named) > 1:
            out.append(Pairing(mid, method="ambiguous",
                               detail=f"{len(named)} markers share the stem key '{key}'"))
            continue

        # 3. Trailing index, same directory only.
        idx = trailing_index(src_stem)
        indexed = [m for m in candidates
                   if _same_index(idx, trailing_index(_stem(m.get("source_path", ""))))]
        if len(indexed) == 1:
            hit = indexed[0]
            same = _directory(hit) == d
            out.append(Pairing(mid, str(hit["movie_id"]), str(hit["source_path"]),
                               "trailing index in directory" if same
                               else "trailing index in marker subfolder"))
            continue
        if len(indexed) > 1:
            out.append(Pairing(mid, method="ambiguous",
                               detail=f"{len(indexed)} markers share index '{idx}' in {d}"))
            continue

        # 5. Sole occupant: one movie and one marker in the directory.
        if len(candidates) == 1 and movies_by_dir.get(d, 0) == 1:
            out.append(Pairing(mid, str(candidates[0]["movie_id"]),
                               str(candidates[0]["source_path"]),
                               "sole marker in directory"))
            continue

        out.append(Pairing(
            mid, method="unmatched",
            detail=(f"{len(candidates)} marker(s) in {d}, none matching stem or index "
                    f"'{idx}'; set marker_image to resolve")))
    return out


def pairing_report(pairings: Sequence[Pairing]) -> pd.DataFrame:
    return pd.DataFrame([{"movie_id": p.movie_id, "marker_movie_id": p.marker_movie_id,
                          "marker_path": p.marker_path, "method": p.method,
                          "matched": p.ok, "detail": p.detail} for p in pairings])


# ---------------------------------------------------------------------------
# Disk-mean reference image
# ---------------------------------------------------------------------------

def disk_kernel(diameter: float) -> np.ndarray:
    """Normalized flat disk kernel approximating trackpy's detection mask.

    trackpy's ``diameter`` is the full extent of the feature mask, so the radius
    is ``diameter / 2``. Pixels whose centre lies within that radius are
    included, matching how trackpy builds its circular mask.
    """
    r = max(float(diameter) / 2.0, 0.5)
    n = int(np.ceil(r))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    k = ((xx ** 2 + yy ** 2) <= r ** 2).astype(float)
    total = k.sum()
    return k / total if total else k


def disk_mean_image(image: np.ndarray, diameter: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel disk mean of ``image``, plus the in-frame fraction of each disk.

    One convolution over the whole image replaces a per-localization
    neighbourhood extraction. Returns ``(mean_image, coverage)`` where
    ``coverage`` is the fraction of the disk lying inside the frame, so partial
    disks at the border are correct means over their valid part and can still be
    filtered out by the caller.

    Non-finite pixels (saturated/masked) are excluded from both sums, so they
    reduce coverage rather than poisoning the mean.
    """
    img = np.asarray(image, dtype=float)
    k = disk_kernel(diameter)
    valid = np.isfinite(img)
    filled = np.where(valid, img, 0.0)
    try:
        from scipy.ndimage import convolve as _convolve

        def C(a):
            return _convolve(a, k, mode="constant", cval=0.0)
    except ImportError:                                     # pragma: no cover
        def C(a):
            from numpy.fft import irfft2, rfft2
            s = (a.shape[0] + k.shape[0] - 1, a.shape[1] + k.shape[1] - 1)
            out = irfft2(rfft2(a, s) * rfft2(k, s), s)
            oy, ox = k.shape[0] // 2, k.shape[1] // 2
            return out[oy:oy + a.shape[0], ox:ox + a.shape[1]]

    num = C(filled)
    cov = C(valid.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cov > 0, num / cov, np.nan)
    return mean, np.clip(cov, 0.0, 1.0)


def sample_disk_means(mean_img: np.ndarray, coverage: np.ndarray,
                      x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Look up the disk mean and coverage at each ``(x, y)`` localization.

    Nearest-pixel indexing is used deliberately: the disk-mean image is already
    smooth on the scale of the disk, so interpolating between its pixels changes
    the value by far less than the localization uncertainty itself. Out-of-frame
    or non-finite coordinates yield NaN rather than a clamped edge value.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(x.shape, np.nan)
    cov = np.full(x.shape, np.nan)
    h, w = mean_img.shape
    xi = np.rint(x); yi = np.rint(y)
    ok = (np.isfinite(xi) & np.isfinite(yi)
          & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h))
    if ok.any():
        xii = xi[ok].astype(np.int64)
        yii = yi[ok].astype(np.int64)
        out[ok] = mean_img[yii, xii]
        cov[ok] = coverage[yii, xii]
    return out, cov


# ---------------------------------------------------------------------------
# Per-track aggregation
# ---------------------------------------------------------------------------

def aggregate_per_track(tracks: pd.DataFrame, shape: Tuple[int, int]) -> pd.DataFrame:
    """Collapse per-localization intensities to exactly one row per track.

    ``tracks`` must already carry ``marker_intensity_localization`` and
    ``marker_disk_coverage``. The output has one row per particle and no
    duplicate particle ids, so a downstream merge cannot fan out the diffusion
    table.
    """
    if tracks.empty or "particle" not in tracks.columns:
        return pd.DataFrame(columns=TRACK_COLUMNS)
    t = tracks.copy()
    t["_v"] = pd.to_numeric(t.get("marker_intensity_localization"), errors="coerce")
    t["_c"] = pd.to_numeric(t.get("marker_disk_coverage"), errors="coerce")
    g = t.groupby("particle", sort=True)
    h, w = shape
    out = pd.DataFrame({
        "particle": list(g.groups.keys()),
        "marker_intensity": g["_v"].mean().to_numpy(),
        "marker_intensity_median": g["_v"].median().to_numpy(),
        "marker_intensity_sd": g["_v"].std(ddof=1).to_numpy(),
        "marker_intensity_min": g["_v"].min().to_numpy(),
        "marker_intensity_max": g["_v"].max().to_numpy(),
        "n_localizations_measured": g["_v"].count().to_numpy(),
        "mean_x_px": g["x"].mean().to_numpy(),
        "mean_y_px": g["y"].mean().to_numpy(),
        "x_min_px": g["x"].min().to_numpy(),
        "x_max_px": g["x"].max().to_numpy(),
        "y_min_px": g["y"].min().to_numpy(),
        "y_max_px": g["y"].max().to_numpy(),
        "frac_disk_in_frame": g["_c"].min().to_numpy(),
    })
    out["x_range_px"] = out["x_max_px"] - out["x_min_px"]
    out["y_range_px"] = out["y_max_px"] - out["y_min_px"]
    # Closest approach to any frame edge, for spatial filtering.
    out["dist_to_edge_px"] = np.minimum.reduce([
        out["x_min_px"].to_numpy(), out["y_min_px"].to_numpy(),
        (w - 1) - out["x_max_px"].to_numpy(), (h - 1) - out["y_max_px"].to_numpy()])
    return out[TRACK_COLUMNS]


# ---------------------------------------------------------------------------
# Per-movie driver
# ---------------------------------------------------------------------------

@dataclass
class MarkerResult:
    movie_id: str
    status: str = "ok"
    n_tracks: int = 0
    n_localizations: int = 0
    marker_movie_id: str = ""
    method: str = ""
    error: Optional[str] = None
    per_track_csv: str = ""


def load_marker_image(path: Path, file_format: str = "", channel: int = 0) -> np.ndarray:
    """Load a marker image as a 2-D array, reducing any stack/channel axis."""
    from . import imaging
    arr = np.asarray(imaging.load_movie(Path(path), file_format or None, channel=channel))
    while arr.ndim > 2:
        # A marker is a single frame; if extra axes survive, take the first
        # plane of the largest leading axis rather than averaging, which would
        # silently blend channels.
        arr = arr[0]
    return arr


def measure_movie(
    movie_row: pd.Series, marker_path: Path, *, diameter: float,
    marker_channel: int = 0, marker_format: str = "",
    append: bool = True,
) -> MarkerResult:
    """Measure marker intensity for one movie's tracks and write the outputs.

    Writes ``{movie}_marker_intensity.csv`` (one row per track) beside the
    tracking outputs and, when ``append``, adds the per-localization column to
    the tracks CSV and the per-track value to the diffusion CSV. Both appends
    are idempotent: re-running replaces the columns rather than duplicating
    them, so this command is safe to repeat with a different disk diameter.
    """
    mid = str(movie_row["movie_id"])
    res = MarkerResult(mid)
    tracks_path = Path(str(movie_row.get("tracks_csv", "")).strip() or ".")
    if not tracks_path.exists():
        res.status, res.error = "skipped", "no tracks CSV"
        return res
    try:
        image = load_marker_image(marker_path, marker_format, marker_channel)
        if image.ndim != 2:
            raise ValueError(f"marker image is not 2-D (shape {image.shape})")
        mean_img, coverage = disk_mean_image(image, diameter)

        tracks = utils.normalize_track_table(pd.read_csv(tracks_path, low_memory=False))
        if not {"particle", "x", "y"}.issubset(tracks.columns):
            res.status, res.error = "skipped", "tracks CSV lacks particle/x/y"
            return res

        vals, cov = sample_disk_means(mean_img, coverage,
                                      tracks["x"].to_numpy(), tracks["y"].to_numpy())
        tracks["marker_intensity_localization"] = vals
        tracks["marker_disk_coverage"] = cov

        per_track = aggregate_per_track(tracks, image.shape)
        per_track.insert(1, "movie_id", mid)
        out_csv = tracks_path.with_name(f"{mid}_marker_intensity.csv")
        per_track.to_csv(out_csv, index=False)
        res.per_track_csv = str(out_csv)

        if append:
            tracks.to_csv(tracks_path, index=False)
            diff_path = Path(str(movie_row.get("diff_csv", "")).strip() or ".")
            if diff_path.exists():
                diff = pd.read_csv(diff_path, low_memory=False)
                if "particle" in diff.columns:
                    lut = per_track.set_index(
                        per_track["particle"].map(_particle_key))["marker_intensity"]
                    diff = diff.drop(columns=[c for c in ("marker_intensity",)
                                              if c in diff.columns])
                    diff["marker_intensity"] = diff["particle"].map(
                        lambda v: lut.get(_particle_key(v), np.nan))
                    diff.to_csv(diff_path, index=False)

        res.n_tracks = int(len(per_track))
        res.n_localizations = int(np.isfinite(vals).sum())
        return res
    except Exception as exc:
        res.status, res.error = "failed", f"{type(exc).__name__}: {exc}"
        return res


def _particle_key(value: Any) -> str:
    """Make integer-looking particle ids agree across independently read CSVs."""
    try:
        f = float(value)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(value)


def measure_all(
    manifest_df: pd.DataFrame, *, diameter: float, marker_channel: int = 0,
    append: bool = True, manifest_path: Optional[Path] = None,
) -> Tuple[List[MarkerResult], pd.DataFrame]:
    """Pair and measure every movie in the manifest that has a marker image.

    When ``manifest_path`` is given, the resolved marker path is written back
    into each movie's ``marker_image`` cell, so the pairing is recorded once and
    the next run does not have to re-derive it (and the user can correct it by
    hand if a match was wrong).
    """
    pairings = pair_markers(manifest_df)
    by_id = {str(r["movie_id"]): r for _, r in manifest_df.iterrows()}
    marker_meta = {str(r["movie_id"]): r for _, r in
                   manifest_mod.marker_rows(manifest_df).iterrows()}
    results: List[MarkerResult] = []

    for p in pairings:
        if not p.ok:
            results.append(MarkerResult(p.movie_id, status="unmatched",
                                        method=p.method, error=p.detail))
            continue
        row = by_id[p.movie_id]
        mrow = marker_meta.get(p.marker_movie_id)
        fmt = str(mrow.get("file_format", "")) if mrow is not None else ""
        r = measure_movie(row, Path(p.marker_path), diameter=diameter,
                          marker_channel=marker_channel, marker_format=fmt,
                          append=append)
        r.marker_movie_id = p.marker_movie_id
        r.method = p.method
        results.append(r)
        if manifest_path is not None and r.status == "ok":
            manifest_mod.update_row_outputs(manifest_path, p.movie_id,
                                            marker_image=p.marker_path)

    return results, pairing_report(pairings)


def load_marker_pool(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Collect every movie's per-track marker table, labelled by condition.

    Reads the standalone ``*_marker_intensity.csv`` files rather than the
    diffusion tables, so the position and quality columns survive for filtering.
    """
    frames = []
    for _, row in manifest_df.iterrows():
        tc = str(row.get("tracks_csv", "")).strip()
        if not tc:
            continue
        mid = str(row["movie_id"])
        p = Path(tc).with_name(f"{mid}_marker_intensity.csv")
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except (OSError, pd.errors.ParserError):
            continue
        if df.empty:
            continue
        df["movie_id"] = mid
        df["condition"] = str(row["condition"])
        df["experiment"] = str(row.get("experiment", ""))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
