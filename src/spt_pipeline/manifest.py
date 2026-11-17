"""Portable manifest handling: single source of truth for dataset metadata.

Workflow:

1. ``scan`` walks input directories, discovers image files (or existing
   ``*_tracks.csv`` with ``from_tracks=True``), extracts acquisition metadata,
   and infers ``condition``/``experiment``. It writes a *draft* CSV plus a scan
   log documenting every inference.
2. The user reviews the CSV and corrects any ``FILL_IN`` values.
3. Every subsequent stage calls :func:`load` + :func:`validate` before doing
   any work, so mistakes halt with actionable messages rather than producing
   silently wrong figures.

Portability contract
--------------------
``source_path``, ``tracks_csv``, and ``diff_csv`` are stored in the CSV as
paths relative to the directory containing the manifest.  ``load()`` resolves
those paths against the manifest directory and returns ordinary local paths to
the rest of the pipeline.  ``write()`` converts paths back to portable
manifest-relative form, so read-modify-write operations do not reintroduce
machine-specific absolute paths.

This means a manifest and its dataset can move together between Windows,
macOS, and Linux without changing the CSV.  If the dataset is mounted at a
different root while the manifest remains elsewhere, pass ``path_base`` to
``load()`` as the mounted dataset root.  Older manifests containing absolute
paths remain readable on the machine that created them, but should be migrated
by loading them on that machine and writing them with this module before
moving them between operating systems.

The manifest is a plain CSV; no special tooling is required to edit it.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import os
import re
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import utils

FILL = "FILL_IN"

COLUMNS = [
    "movie_id", "source_path", "file_format", "condition", "experiment",
    "tracks_csv", "diff_csv", "marker_image", "mpp", "fps", "n_frames",
    "n_channels", "notes",
]
REQUIRED = ["movie_id", "source_path", "file_format", "condition", "experiment"]
PATH_COLUMNS = ("source_path", "tracks_csv", "diff_csv", "marker_image")

#: Token in ``notes`` marking a row as an expression-marker image rather than a
#: tracking movie. Matched as a whole word so "no_marker_found" does not count.
MARKER_NOTE = "marker"
_MARKER_NOTE_RE = re.compile(r"(?:^|[\s,;|])marker(?:$|[\s,;|=])", re.IGNORECASE)

#: Sentinel written into ``notes`` when tracking produced nothing usable. A row
#: carrying this (or simply a blank ``tracks_csv``) is *skipped*, never an error:
#: a movie with no particles above threshold is a legitimate experimental
#: outcome, not a pipeline failure.
NO_TRACKS_NOTE = "no_tracks"

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")
_WINDOWS_UNC = re.compile(r"^(?:\\\\|//)")


# ---------------------------------------------------------------------------
# Portable path handling
# ---------------------------------------------------------------------------

def _text(value: Any) -> str:
    """Return a CSV cell as clean text, treating NaN/None as empty."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _looks_absolute_any_os(value: str) -> bool:
    """Recognize POSIX, Windows-drive, and Windows-UNC absolute paths."""
    return bool(value and (
        Path(value).is_absolute()
        or _WINDOWS_DRIVE.match(value)
        or _WINDOWS_UNC.match(value)
    ))


def _normalise_separators(value: str) -> str:
    """Use forward slashes in the CSV; Python accepts them on all platforms."""
    return value.replace("\\", "/")


def _strip_root(normalized: str) -> str:
    """Drop the OS-specific root from a normalized path, keeping the tail.

    ``C:/data/exp/m.nd2`` -> ``data/exp/m.nd2``
    ``//server/share/exp/m.nd2`` -> ``server/share/exp/m.nd2``
    ``/Volumes/data/exp/m.nd2`` -> ``Volumes/data/exp/m.nd2``
    """
    s = normalized
    if _WINDOWS_DRIVE.match(s):
        s = s[2:]
    return s.lstrip("/")


#: How far below the anchor to look for a re-anchoring match, and how many
#: directories to enumerate before giving up. Deep enough for "the manifest sits
#: a couple of levels above the movie folders"; bounded so a manifest pointed at
#: a huge share cannot trigger a full-tree walk.
_REBASE_MAX_DEPTH = 4
_REBASE_MAX_DIRS = 4000


def _anchor_dirs(base_dir: Path, max_depth: int = _REBASE_MAX_DEPTH) -> List[Path]:
    """``base_dir`` plus its descendant directories, breadth-first, bounded.

    Enumerated once per load and reused across rows: a 400-movie manifest must
    not re-walk the tree 400 times.
    """
    out = [base_dir]
    frontier = [base_dir]
    for _ in range(max_depth):
        nxt = []
        for d in frontier:
            try:
                for child in d.iterdir():
                    if not child.is_dir() or child.name.startswith("."):
                        continue
                    out.append(child)
                    nxt.append(child)
                    if len(out) >= _REBASE_MAX_DIRS:
                        return out
            except (OSError, PermissionError):
                continue
        if not nxt:
            break
        frontier = nxt
    return out


def _rebase_onto(normalized: str, base_dir: Path,
                 anchors: Optional[Sequence[Path]] = None) -> Optional[Path]:
    """Re-anchor a foreign path by matching its deepest tail under ``base_dir``.

    This is what makes a manifest genuinely self-referencing rather than merely
    relative. A manifest written on Windows may carry
    ``//research-cifs/…/Andrew/20260805 Expt/C11-158_DMSO/488__001.nd2`` while
    the same share is mounted on macOS at ``/Volumes/…/20260805 Expt`` with the
    manifest sitting inside it. The stored prefix is meaningless here, but the
    *tail* -- ``C11-158_DMSO/488__001.nd2`` -- is the dataset-internal location,
    and that is what we recover.

    Tails are tried longest-first so the most specific match wins; a bare
    filename is only accepted when nothing deeper matched. The cheap case (the
    anchor is the dataset root) resolves in one ``exists()`` call. Only if that
    fails do we consider descendant directories, so the fallback cost is paid
    exclusively by manifests that need it.
    """
    parts = [p for p in _strip_root(normalized).split("/") if p not in ("", ".")]
    if not parts:
        return None
    base_dir = Path(base_dir)

    # Fast path: the tail hangs directly off the anchor.
    for start in range(len(parts)):
        candidate = base_dir.joinpath(*parts[start:])
        if candidate.exists():
            return candidate

    # Slow path: the anchor sits above the data with intervening directories.
    if anchors is None:
        anchors = _anchor_dirs(base_dir)
    for start in range(len(parts)):
        tail = parts[start:]
        # A bare filename matched against every descendant is too weak to trust.
        if len(tail) < 2:
            break
        for anchor in anchors:
            candidate = anchor.joinpath(*tail)
            if candidate.exists():
                return candidate
    return None


def _resolve_manifest_path(value: Any, base_dir: Path,
                           anchors: Optional[Sequence[Path]] = None) -> str:
    """Resolve one manifest path for the current runtime.

    Resolution order, first hit wins:

    1. Relative path anchored to ``base_dir`` -- the normal, portable case.
    2. The path exactly as stored, if it exists natively.
    3. The deepest tail of the stored path that exists under ``base_dir``
       (:func:`_rebase_onto`) -- recovers manifests written on another OS or
       against a different mount point.

    If none resolve, the normalized native form is returned so validation can
    report something recognizable rather than a mangled path.
    """
    raw = _text(value)
    if not raw:
        return ""

    normalized = _normalise_separators(raw)
    base_dir = Path(base_dir)

    if not _looks_absolute_any_os(raw):
        anchored = (base_dir / Path(normalized))
        if anchored.exists():
            return str(anchored.resolve())
        # A relative path that does not resolve may still be re-anchorable: the
        # manifest may sit deeper or shallower in the tree than when written.
        rebased = _rebase_onto(normalized, base_dir, anchors)
        return str(rebased.resolve()) if rebased else str(anchored)

    native = Path(normalized)
    # Trust an absolute path only if it actually exists on this machine;
    # otherwise it is a leftover from wherever the manifest was written.
    if native.is_absolute() and native.exists():
        return str(native.expanduser().resolve())
    rebased = _rebase_onto(normalized, base_dir, anchors)
    if rebased is not None:
        return str(rebased.resolve())
    return str(native.expanduser()) if native.is_absolute() else normalized


def _serialise_manifest_path(value: Any, base_dir: Path,
                             problems: Optional[List[str]] = None) -> str:
    """Convert a runtime path to a manifest-relative POSIX path.

    Already-relative values are normalized and preserved. Absolute paths are
    made relative to the manifest directory when possible.

    Two cases cannot produce a clean relative path, and both are recorded in
    ``problems`` for the caller to report once rather than once per row:

    * the manifest and the data sit on different volumes (a Windows ``C:``
      manifest beside a ``\\\\server`` share), where ``relpath`` raises;
    * the manifest is not an ancestor of the data, so the relative path climbs
      out through ``..`` and only works while both trees keep their relative
      arrangement.

    In both cases the absolute path is preserved. That is no longer fatal --
    :func:`_resolve_manifest_path` can re-anchor by tail on read -- but it is
    worth telling the user, because putting the manifest inside the dataset
    makes the problem disappear.
    """
    raw = _text(value)
    if not raw:
        return ""

    normalized = _normalise_separators(raw)
    if not _looks_absolute_any_os(raw):
        return normalized

    candidate = Path(normalized)
    if not candidate.is_absolute():
        if problems is not None:
            problems.append(f"foreign absolute path preserved as-is: {raw}")
        return normalized

    try:
        rel = Path(os.path.relpath(candidate.resolve(), base_dir.resolve())).as_posix()
    except (OSError, ValueError):
        if problems is not None:
            problems.append(f"different volume from the manifest: {raw}")
        return normalized
    if rel.startswith(".."):
        if problems is not None:
            problems.append(f"outside the manifest directory: {raw}")
    return rel


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _extract_image_metadata(path: Path) -> Dict[str, Any]:
    """Best-effort frame/channel/pixel-size extraction; never raises."""
    meta: Dict[str, Any] = {"n_frames": None, "n_channels": None, "mpp": None, "fps": None}
    ext = path.suffix.lower()
    try:
        if ext == ".nd2":
            import nd2  # type: ignore
            with nd2.ND2File(str(path)) as f:
                sizes = dict(f.sizes)
                meta["n_frames"] = int(sizes.get("T", 1))
                meta["n_channels"] = int(sizes.get("C", 1))
                vox = getattr(f, "voxel_size", lambda: None)()
                if vox is not None and getattr(vox, "x", None):
                    meta["mpp"] = float(vox.x)
                # frame interval (seconds) -> fps. Try the experiment loop's
                # nominal period first, then fall back to actual frame timestamps
                # (more reliable across ND2 versions/acquisition modes).
                try:
                    for loop in f.experiment:
                        params = getattr(loop, "parameters", None)
                        period_ms = getattr(params, "periodMs", None)
                        if period_ms:
                            meta["fps"] = round(1000.0 / float(period_ms), 4)
                            break
                except Exception:
                    pass
                if not meta["fps"] and meta["n_frames"] and meta["n_frames"] > 1:
                    try:
                        def _t(i):
                            fm = f.frame_metadata(i)
                            return fm.channels[0].time.relativeTimeMs
                        # median dt over the first few frames
                        n = min(6, meta["n_frames"])
                        times = [_t(i) for i in range(n)]
                        diffs = np.diff(times)
                        diffs = diffs[diffs > 0]
                        if diffs.size:
                            meta["fps"] = round(1000.0 / float(np.median(diffs)), 4)
                    except Exception:
                        pass
        elif ext in (".tif", ".tiff"):
            import tifffile  # type: ignore
            with tifffile.TiffFile(str(path)) as tf:
                series = tf.series[0]
                shape = series.shape
                axes = series.axes
                if "T" in axes:
                    meta["n_frames"] = int(shape[axes.index("T")])
                elif "Z" in axes:
                    meta["n_frames"] = int(shape[axes.index("Z")])
                else:
                    meta["n_frames"] = int(len(tf.pages))
                meta["n_channels"] = int(shape[axes.index("C")]) if "C" in axes else 1
                # Pixel size: prefer ImageJ metadata, then TIFF resolution tags.
                try:
                    ij = getattr(tf, "imagej_metadata", None) or {}
                    if ij.get("spacing"):
                        meta["mpp"] = float(ij["spacing"])
                    else:
                        page = tf.pages[0]
                        xres = page.tags.get("XResolution")
                        unit = page.tags.get("ResolutionUnit")
                        if xres is not None and xres.value and xres.value[0]:
                            num, den = xres.value
                            per_unit = float(num) / float(den) if den else 0.0
                            if per_unit > 0:
                                microns_per_unit = 1e4 if (unit and int(unit.value) == 3) else 1.0
                                meta["mpp"] = microns_per_unit / per_unit
                    # Frame rate: ImageJ 'fps' or frame interval.
                    if ij.get("fps"):
                        meta["fps"] = round(float(ij["fps"]), 4)
                    elif ij.get("finterval"):
                        fi = float(ij["finterval"])
                        if fi > 0:
                            meta["fps"] = round(1.0 / fi, 4)
                except Exception:
                    pass
    except Exception:
        pass
    return meta


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    manifest: pd.DataFrame
    log: Dict[str, Any]


def scan(
    input_dirs: Sequence[Path],
    default_mpp: Optional[float] = None,
    default_fps: Optional[float] = None,
    exclude_patterns: Optional[Sequence[str]] = None,
    condition_tokens: Optional[Sequence[str]] = None,
    from_tracks: bool = False,
) -> ScanResult:
    """Scan directories and build a draft manifest DataFrame + inference log."""
    rows: List[Dict[str, Any]] = []
    inferences: List[Dict[str, Any]] = []
    n_ok = n_fill = 0

    for d in input_dirs:
        root = Path(d)
        group_map = utils.load_groupnames_condition_map(root)
        if from_tracks:
            files = utils.discover_track_files(root)
        else:
            files = utils.discover_image_files(root, exclude_patterns)

        for f in files:
            mid = utils.movie_id_from_path(f)
            ext = f.suffix.lower().lstrip(".")

            # Condition: groupnames.csv -> tokens -> FILL_IN
            cond = group_map.get(mid) or group_map.get(f.stem)
            cond_source = "groupnames.csv" if cond else None
            if not cond:
                cond = utils.infer_condition(f.stem, condition_tokens) or \
                       utils.infer_condition(f.parent.name, condition_tokens)
                cond_source = "token match" if cond else None
            if not cond:
                cond = FILL
                cond_source = "unresolved"

            # Experiment: date in folder -> date in filename -> file mtime
            exp = utils.infer_experiment(f.parent.name) or utils.infer_experiment(f.stem)
            exp_source = "date parse" if exp else None
            if not exp:
                try:
                    import datetime as _dt
                    exp = _dt.date.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d")
                    exp_source = "file mtime (verify!)"
                except OSError:
                    exp, exp_source = FILL, "unresolved"

            if from_tracks:
                imeta = {"n_frames": None, "n_channels": None, "mpp": None, "fps": None}
                tracks_csv: Optional[str] = str(f.resolve())
                diff_guess = f.with_name(f.name.replace("_tracks.csv", "_diffusion_coefficients.csv"))
                diff_csv = str(diff_guess.resolve()) if diff_guess.exists() else ""
                source_path = str(f.resolve())
                file_format = "tracks"
            else:
                imeta = _extract_image_metadata(f)
                tracks_csv, diff_csv = "", ""
                source_path = str(f.resolve())
                file_format = ext

            mpp = imeta["mpp"] if imeta["mpp"] else default_mpp
            fps = imeta["fps"] if imeta["fps"] else default_fps

            rows.append({
                "movie_id": mid,
                "source_path": source_path,
                "file_format": file_format,
                "condition": cond,
                "experiment": exp,
                "tracks_csv": tracks_csv or "",
                "diff_csv": diff_csv or "",
                "mpp": mpp if mpp is not None else "",
                "fps": fps if fps is not None else "",
                "n_frames": imeta["n_frames"] if imeta["n_frames"] is not None else "",
                "n_channels": imeta["n_channels"] if imeta["n_channels"] is not None else "",
                "notes": "",
            })
            if cond == FILL or exp == FILL:
                n_fill += 1
            else:
                n_ok += 1
            inferences.append({
                "movie_id": mid, "source_path": source_path,
                "condition": cond, "condition_source": cond_source,
                "experiment": exp, "experiment_source": exp_source,
                "mpp": mpp, "fps": fps,
            })

    df = pd.DataFrame(rows, columns=COLUMNS)
    # De-duplicate movie ids across directories (append a numeric suffix)
    if not df.empty and df["movie_id"].duplicated().any():
        seen: Dict[str, int] = {}
        new_ids = []
        for mid in df["movie_id"]:
            if mid in seen:
                seen[mid] += 1
                new_ids.append(f"{mid}__{seen[mid]}")
            else:
                seen[mid] = 0
                new_ids.append(mid)
        df["movie_id"] = new_ids

    log = {
        "input_dirs": [str(Path(d).resolve()) for d in input_dirs],
        "from_tracks": from_tracks,
        "n_movies": int(len(df)),
        "n_resolved": int(n_ok),
        "n_need_attention": int(n_fill),
        "default_mpp": default_mpp,
        "default_fps": default_fps,
        "exclude_patterns": list(exclude_patterns or []),
        "inferences": inferences,
    }
    return ScanResult(manifest=df, log=log)


def write(
    df: pd.DataFrame,
    path: Path,
    portable: bool = True,
    path_base: Optional[Path] = None,
) -> None:
    """Write a manifest, storing file paths relative to the manifest.

    Parameters
    ----------
    df:
        Manifest DataFrame.  It may contain absolute paths from ``scan()`` or
        paths already resolved by ``load()``.
    path:
        Destination manifest CSV.  Its parent is the default path anchor.
    portable:
        Keep the default ``True``.  Set ``False`` only for an intentional
        legacy export containing absolute paths.
    path_base:
        Optional directory to use as the path anchor instead of
        ``path.parent``.  This is useful when a manifest is stored in an
        analysis folder while the dataset has a separate stable root.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()

    if portable:
        base_dir = Path(path_base).expanduser().resolve() if path_base else path.resolve().parent
        problems: List[str] = []
        for col in PATH_COLUMNS:
            if col in out.columns:
                out[col] = out[col].map(
                    lambda value: _serialise_manifest_path(value, base_dir, problems))
        if problems:
            # One warning per write, not one per row: a 400-movie manifest with
            # a misplaced anchor would otherwise bury the console.
            kinds = sorted({p.split(":")[0] for p in problems})
            warnings.warn(
                f"{len(problems)} manifest path(s) could not be stored relative to "
                f"{str(base_dir)!r} ({'; '.join(kinds)}). They are preserved as absolute "
                f"paths and will still be recovered on another machine by tail matching, "
                f"but the manifest is only fully portable when it lives inside the data "
                f"directory -- or when --data-root points at that directory.",
                RuntimeWarning, stacklevel=2,
            )

    out.to_csv(path, index=False)


def load(
    path: Path,
    resolve_paths: bool = True,
    path_base: Optional[Path] = None,
) -> pd.DataFrame:
    """Load a manifest and resolve its portable paths for the current OS.

    Every value in ``PATH_COLUMNS`` is resolved against the manifest's own
    directory: first as a relative path, then -- if that misses -- by matching
    the deepest tail of the stored path under the anchor. That second step is
    what lets a manifest written on Windows against ``\\\\server\\share\\...``
    be read on macOS where the same data is mounted elsewhere, without editing
    the CSV.

    Use ``path_base`` when the dataset is mounted at a different root from the
    directory containing the manifest. Set ``resolve_paths=False`` only for
    tooling that needs the raw CSV values.
    """
    path = Path(path).expanduser()
    df = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""

    if resolve_paths:
        base_dir = Path(path_base).expanduser().resolve() if path_base else path.resolve().parent
        # Directory listing for the re-anchoring fallback, built at most once
        # per load and only when some path actually needs it.
        anchor_cache: Dict[str, List[Path]] = {}

        def _anchors() -> List[Path]:
            if "dirs" not in anchor_cache:
                anchor_cache["dirs"] = _anchor_dirs(base_dir)
            return anchor_cache["dirs"]

        class _LazyAnchors(Sequence):
            """Defers the directory walk until the fallback path is reached."""
            def __getitem__(self, i):   return _anchors()[i]
            def __len__(self):          return len(_anchors())

        lazy = _LazyAnchors()
        for col in PATH_COLUMNS:
            if col in df.columns:
                df[col] = df[col].map(lambda value: _resolve_manifest_path(value, base_dir, lazy))
        # Useful to callers that need to report where the portable anchor came
        # from, without adding another CSV column or changing the schema.
        df.attrs["manifest_path"] = str(path.resolve())
        df.attrs["manifest_base"] = str(base_dir)
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ManifestError(ValueError):
    """Raised when a manifest fails validation. Message lists every problem."""


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def validate(
    df: pd.DataFrame,
    default_mpp: Optional[float] = None,
    default_fps: Optional[float] = None,
    require_tracks: bool = False,
) -> List[str]:
    """Validate a manifest. Raises :class:`ManifestError` on hard failures.

    Returns a list of non-fatal warnings. ``require_tracks`` enforces that
    every row has an existing ``tracks_csv`` (used by Phase B / C).
    """
    errors: List[str] = []
    warnings: List[str] = []

    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise ManifestError(f"Manifest missing required columns: {missing_cols}")

    if df.empty:
        raise ManifestError("Manifest is empty; nothing to process.")

    # FILL_IN sentinels
    for col in ("condition", "experiment"):
        bad = df.index[df[col].astype(str).str.strip().eq(FILL)].tolist()
        if bad:
            errors.append(f"{len(bad)} row(s) still have {FILL} in '{col}' "
                          f"(movie_id: {', '.join(df.loc[bad, 'movie_id'].head(5))}...)")

    # Unique movie_id
    dup = df["movie_id"][df["movie_id"].duplicated()].unique().tolist()
    if dup:
        errors.append(f"Duplicate movie_id values: {dup[:10]}")

    # source_path existence
    missing_sources = []
    for _, row in df.iterrows():
        sp = str(row["source_path"]).strip()
        if sp and not Path(sp).exists():
            missing_sources.append((str(row["movie_id"]), sp))
    if missing_sources:
        shown = "\n      ".join(f"{mid}: {sp}" for mid, sp in missing_sources[:5])
        more = f"\n      ... and {len(missing_sources) - 5} more" if len(missing_sources) > 5 else ""
        base = df.attrs.get("manifest_base")
        hint = (f"\n    Paths were resolved against: {base}"
                f"\n    If the data lives elsewhere on this machine, point --data-root at the "
                f"directory that contains the movie folders." if base else "")
        errors.append(f"{len(missing_sources)} source_path(s) could not be found:"
                      f"\n      {shown}{more}{hint}")

    # tracks_csv when required.
    #
    # A *blank* tracks_csv means tracking legitimately yielded nothing for that
    # movie (no particles above threshold, or no track surviving the length
    # cutoff). That is data, not a defect, so the row is skipped with a warning
    # and downstream stages simply exclude it. A *populated but missing* path,
    # by contrast, means the manifest points at a file that should exist and
    # does not -- a real error worth halting on.
    if require_tracks:
        # Marker images and single-frame acquisitions are never tracked, so
        # their blank tracks_csv is expected and must not be reported as a gap.
        trackable = [r for _, r in df.iterrows() if is_trackable(r)]
        n_non_track = len(df) - len(trackable)
        n_blank = 0
        for row in trackable:
            tc = str(row.get("tracks_csv", "")).strip()
            if not tc:
                n_blank += 1
                continue
            if not Path(tc).exists():
                errors.append(f"tracks_csv listed but not found for {row['movie_id']}: '{tc}'")
        if n_non_track:
            warnings.append(f"{n_non_track} row(s) are marker images or single-frame "
                            f"acquisitions; excluded from tracking by design.")
        if n_blank:
            warnings.append(
                f"{n_blank} of {len(trackable)} movie(s) have no tracks_csv (tracking found "
                f"no qualifying tracks); these rows are skipped for this analysis."
            )
        if trackable and n_blank == len(trackable):
            errors.append("No movie in the manifest has tracks; nothing to analyze. "
                          "Run 'track trackpy' first, or relax the detection thresholds.")
        if not trackable:
            errors.append("Manifest contains no trackable movies (every row is a marker "
                          "image or a single frame).")

    # mpp/fps positivity (row value or CLI default)
    for _, row in df.iterrows():
        mpp = _num(row.get("mpp", "")) if str(row.get("mpp", "")).strip() else default_mpp
        fps = _num(row.get("fps", "")) if str(row.get("fps", "")).strip() else default_fps
        if mpp is None or mpp <= 0:
            errors.append(f"mpp missing/invalid for {row['movie_id']} and no positive default given.")
        if fps is None or fps <= 0:
            errors.append(f"fps missing/invalid for {row['movie_id']} and no positive default given.")

    # Warnings: coverage (over tracking movies only -- marker images are not
    # replicates and must not inflate the per-condition counts)
    cov = df[df.apply(is_trackable, axis=1)] if not df.empty else df
    for cond, sub in (cov.groupby("condition") if not cov.empty else []):
        n_exp = sub["experiment"].nunique()
        if n_exp < 2:
            warnings.append(f"Condition '{cond}' has only {n_exp} experiment(s); "
                            f"between-condition inference will be weak.")
        if len(sub) < 2:
            warnings.append(f"Condition '{cond}' has only {len(sub)} movie(s).")

    # Warning: inconsistent mpp/fps within an experiment
    for exp, sub in df.groupby("experiment"):
        for col in ("mpp", "fps"):
            vals = {round(v, 6) for v in (_num(x) for x in sub[col]) if v is not None}
            if len(vals) > 1:
                warnings.append(f"Experiment '{exp}' has differing {col} values: {sorted(vals)}")

    if errors:
        raise ManifestError(
            "Manifest validation failed:\n  - " + "\n  - ".join(errors)
        )
    return warnings


# ---------------------------------------------------------------------------
# Row accessors / in-place updates
# ---------------------------------------------------------------------------

def row_mpp_fps(row: pd.Series, default_mpp: float, default_fps: float) -> tuple:
    mpp = _num(row.get("mpp", "")) if str(row.get("mpp", "")).strip() else None
    fps = _num(row.get("fps", "")) if str(row.get("fps", "")).strip() else None
    return (mpp if mpp else default_mpp, fps if fps else default_fps)


def is_marker_row(row: pd.Series) -> bool:
    """True when this row is an expression-marker image, not a tracking movie."""
    return bool(_MARKER_NOTE_RE.search(_text(row.get("notes", ""))))


def n_frames_of(row: pd.Series) -> Optional[int]:
    """Frame count from the manifest, or None when it was never recorded."""
    v = _num(row.get("n_frames", ""))
    return int(v) if v is not None and v > 0 else None


def is_single_frame(row: pd.Series) -> bool:
    """True when the row is known to hold a single image frame.

    Conservative: a blank ``n_frames`` returns False rather than guessing, so a
    movie whose metadata failed to parse is still offered to the tracker (which
    will report it properly) instead of being silently dropped.
    """
    n = n_frames_of(row)
    return n is not None and n <= 1


def is_trackable(row: pd.Series) -> bool:
    """True when this row is a candidate for particle tracking.

    Marker images and any single-frame acquisition are excluded: there is no
    time axis to link along, so tracking them is meaningless rather than merely
    unproductive. They stay in the manifest because the marker analysis needs
    them; they are simply never handed to the tracker.
    """
    return not is_marker_row(row) and not is_single_frame(row)


def skip_tracking_reason(row: pd.Series) -> Optional[str]:
    """Why a row is not a tracking candidate, or None if it is one."""
    if is_marker_row(row):
        return "expression-marker image"
    if is_single_frame(row):
        return f"single frame (n_frames={n_frames_of(row)})"
    return None


def marker_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Subset of the manifest that holds expression-marker images."""
    if df.empty:
        return df
    out = df[df.apply(is_marker_row, axis=1)].copy()
    out.attrs.update(df.attrs)
    return out


def movie_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Subset of the manifest that is a candidate for tracking."""
    if df.empty:
        return df
    out = df[df.apply(is_trackable, axis=1)].copy()
    out.attrs.update(df.attrs)
    return out


def has_tracks(row: pd.Series) -> bool:
    """True when a manifest row points at a tracks CSV that exists on disk.

    The single predicate every consumer should use, so "this movie produced no
    tracks" is interpreted identically everywhere instead of being re-derived
    (and re-diverging) per module.
    """
    tc = _text(row.get("tracks_csv", ""))
    return bool(tc) and Path(tc).exists()


def rows_with_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """Subset of the manifest whose rows have usable tracks."""
    if df.empty:
        return df
    mask = df.apply(has_tracks, axis=1)
    out = df[mask].copy()
    out.attrs.update(df.attrs)
    return out


def skipped_movies(df: pd.DataFrame) -> List[str]:
    """movie_ids excluded because tracking produced nothing usable."""
    if df.empty:
        return []
    return [str(r["movie_id"]) for _, r in df.iterrows() if not has_tracks(r)]


def update_row_outputs(path: Path, movie_id: str, tracks_csv: str = "",
                       diff_csv: str = "", note: str = "",
                       marker_image: str = "") -> None:
    """Update tracks_csv/diff_csv (and optionally notes) for one movie.

    Read-modify-write so an interrupted tracking run can resume and skip
    already-tracked movies. Intended to be called from the main process
    (not from parallel workers) to avoid write races. Because :func:`load`
    resolves and :func:`write` re-relativizes, the round trip preserves the
    manifest's portable form rather than baking in absolute paths.
    """
    df = load(path)
    mask = df["movie_id"] == movie_id
    if not mask.any():
        return
    if tracks_csv:
        df.loc[mask, "tracks_csv"] = tracks_csv
    if diff_csv:
        df.loc[mask, "diff_csv"] = diff_csv
    if marker_image:
        df.loc[mask, "marker_image"] = marker_image
    if note:
        existing = _text(df.loc[mask, "notes"].iloc[0]) if "notes" in df.columns else ""
        parts = [p for p in existing.split("; ") if p and p != note]
        df.loc[mask, "notes"] = "; ".join(parts + [note])
    write(df, path)
