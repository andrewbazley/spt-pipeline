#!/usr/bin/env python3
"""
SPT Unified Pipeline
====================
Consolidated pipeline for Single-Particle Tracking analysis.
All original script functionalities are preserved as subcommands.

Usage:
    spt-pipeline <subcommand> [options]
    # or, when running from a checkout:
    python -m spt_pipeline.pipeline <subcommand> [options]

Subcommands:
    prep        - Data preparation (TrackMate, ND2 conversion, file finding)
    track       - Tracking operations (trackpy)
    analyze     - Analysis operations (compare-conditions)
    draw        - Movie/track overlays
    utils       - Utility operations (split metrics, find files)

Notes:
    `analyze compare-conditions` supports cross-experiment pooling of flat
    *_diffusion_coefficients.csv files plus per-condition track filtering
    (--min-nspots, --min-alpha), per-condition localization-error
    subtraction (--subtract-loc-error, --loc-error-alpha-max,
    --loc-error-min-nspots), and per-pattern MPP rescaling (--mpp-rescale).
"""

import argparse
import colorsys
import glob
import json
import logging
import math
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, ListedColormap
from matplotlib.collections import LineCollection
from scipy import stats
from scipy.stats import gaussian_kde

# scikit-learn is an optional extra (only the GMM paths need it). Importing it
# at module scope would make the whole legacy dispatcher -- and therefore the
# CLI's fallback for prep/draw/utils -- fail on a core-only install.
try:
    from sklearn.mixture import GaussianMixture
    HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    GaussianMixture = None
    HAVE_SKLEARN = False

# Try optional imports
try:
    import nd2
    HAVE_ND2 = True
except ImportError:
    HAVE_ND2 = False

try:
    import pims
    HAVE_PIMS = True
except ImportError:
    HAVE_PIMS = False

try:
    import trackpy as tp
    import numba
    HAVE_TRACKPY = True
except ImportError:
    HAVE_TRACKPY = False

try:
    import tifffile
    HAVE_TIFFFILE = True
except ImportError:
    HAVE_TIFFFILE = False

try:
    import statsmodels.api as sm
    HAVE_SM = True
except ImportError:
    HAVE_SM = False

# ============================================================================
# SHARED UTILITIES
# ============================================================================

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find column by name (case-insensitive, partial match)."""
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in low:
            return low[cand.lower()]
        for c in cols:
            if cand.lower() in c.lower():
                return c
    return None

def simple_clean(s: str) -> str:
    """Clean string for filenames."""
    return s.strip().replace(" ", "").lower()

PIPELINE_CONTEXT_FILENAME = "spt_pipeline_context.json"

def _should_ignore_path(path_like) -> bool:
    """Ignore hidden/AppleDouble files when recursively scanning inputs."""
    p = Path(path_like)
    return any(part.startswith(".") or part.startswith("._") or part == "__MACOSX" for part in p.parts)

def _rglob_filtered(root: Path, pattern: str) -> List[Path]:
    """Recursive glob that skips hidden/AppleDouble files."""
    return sorted(
        (p for p in root.rglob(pattern) if p.is_file() and not _should_ignore_path(p)),
        key=lambda p: str(p),
    )

def _discover_movie_inputs(root: Path) -> List[Path]:
    files = []
    for pattern in ("*.nd2", "*.tif", "*.tiff"):
        files.extend(_rglob_filtered(root, pattern))
    return sorted({p for p in files}, key=lambda p: str(p))

def _normalize_track_output_name(name: str) -> str:
    base = str(name).strip()
    if base.endswith("_tracks"):
        base = base[:-7]
    return base

def _load_groupnames_condition_map(root: Path) -> Dict[str, str]:
    group_file = root / "groupnames.csv"
    if not group_file.exists() or _should_ignore_path(group_file):
        return {}
    try:
        group_df = pd.read_csv(group_file, low_memory=False)
    except Exception:
        return {}
    if not all(col in group_df.columns for col in ("folder", "condition")):
        return {}
    out = {}
    for _, row in group_df.iterrows():
        folder = _normalize_track_output_name(str(row["folder"]).strip())
        condition = str(row["condition"]).strip()
        if folder and condition:
            out[folder] = condition
    return out

def _nanmean_series(s: pd.Series) -> float:
    return float(np.nanmean(s.to_numpy(dtype=float)))

def _nanmedian_series(s: pd.Series) -> float:
    return float(np.nanmedian(s.to_numpy(dtype=float)))

def _argv_has_flag(*flags: str) -> bool:
    argv = sys.argv[1:]
    for token in argv:
        for flag in flags:
            if token == flag or token.startswith(flag + "="):
                return True
    return False

def _context_file(root: Path) -> Path:
    return root / PIPELINE_CONTEXT_FILENAME

_pipeline_context_cache: Dict[str, Any] = {}

def _load_pipeline_context(root: Path) -> Dict[str, Any]:
    key = str(root)
    if key in _pipeline_context_cache:
        return _pipeline_context_cache[key]
    path = _context_file(root)
    if not path.exists():
        return {}
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
        result = data if isinstance(data, dict) else {}
        _pipeline_context_cache[key] = result
        return result
    except Exception:
        return {}

def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value

def _save_pipeline_context(root: Path, section: List[str], params: Dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    ctx = _load_pipeline_context(root)
    cur = ctx
    for key in section[:-1]:
        cur = cur.setdefault(key, {})
    cur[section[-1]] = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "params": _to_jsonable(params),
    }
    with open(_context_file(root), "w") as fh:
        json.dump(ctx, fh, indent=2)
    _pipeline_context_cache[str(root)] = ctx

def _context_params(root: Path, section: List[str]) -> Dict[str, Any]:
    cur: Any = _load_pipeline_context(root)
    for key in section:
        if not isinstance(cur, dict) or key not in cur:
            return {}
        cur = cur[key]
    if isinstance(cur, dict) and isinstance(cur.get("params"), dict):
        return cur["params"]
    return {}

def _resolve_param(explicit: bool, current_value: Any, context_value: Any, default_value: Any) -> Any:
    if explicit:
        return current_value
    if context_value is not None:
        return context_value
    return default_value

def _add_bool_flag(parser, flag: str, dest: str, default: bool, help_text: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(flag, dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no-{flag[2:]}", dest=dest, action="store_false", help=f"Disable: {help_text}")
    parser.set_defaults(**{dest: default})

@contextmanager
def _trackpy_quiet():
    """Reduce Trackpy's per-frame progress logging during batch runs."""
    logger_names = [
        "trackpy",
        "trackpy.feature",
        "trackpy.feature.batch",
        "trackpy.linking",
        "trackpy.linking.linking",
        "trackpy.linking.linking.link_iter",
    ]
    saved = []
    for name in logger_names:
        logger = logging.getLogger(name)
        saved.append((logger, logger.level))
        logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for logger, old_level in saved:
            logger.setLevel(old_level)


def _select_channel(arr: np.ndarray, channel: int = 0) -> np.ndarray:
    """Select a single channel from a multichannel movie array.

    Handles the common layouts produced by ``nd2`` and ``tifffile``:
      - 2D ``(Y, X)``: returned unchanged (single frame).
      - 3D ``(T, Y, X)``: returned unchanged (single-channel time-series, keeps
        existing single-channel behaviour for backward compatibility).
      - 4D ``(T, C, Y, X)`` or ``(T, Y, X, C)``: the channel axis is inferred
        heuristically from axis size and the requested channel is returned,
        yielding a ``(T, Y, X)`` array.

    ``channel`` is 0-indexed and clamped to the available range so an
    out-of-range request falls back to the closest valid channel rather than
    raising. Defaults to the first channel to preserve prior behaviour.
    """
    if arr.ndim <= 3:
        return arr
    channel = max(int(channel), 0)
    if arr.ndim == 4:
        shape = arr.shape
        # Prefer channel-first layout (T, C, Y, X); this is the nd2 default.
        if shape[1] <= 8 and (shape[-1] > 8 or shape[1] <= shape[-1]):
            idx = min(channel, shape[1] - 1)
            return arr[:, idx, :, :]
        if shape[-1] <= 8:
            idx = min(channel, shape[-1] - 1)
            return arr[..., idx]
        idx = min(channel, shape[1] - 1)
        return arr[:, idx, :, :]
    return arr


# ============================================================================
# WORKFLOW HELPERS - High-level chained operations
# ============================================================================

def cmd_prep_trackmate(args):
    """TrackMate preprocessing - from find_trackmate_prep.py"""
    from pathlib import Path as PPath
    
    PREFS = [
        "Spots in tracks statistics.csv",
        "spots in tracks statistics.csv",
        "All spots statistics.csv",
        "all spots statistics.csv",
    ]
    
    DATE_DIR_RE = re.compile(r"^(?P<yyyy>\d{4})(?P<mm>\d{2})(?P<dd>\d{2})$")
    DATE_YYMMDD_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
    DATE_YYYYMMDD_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")
    SPOOL_RE = re.compile(r"spool[_-](\d+)", re.I)
    
    def read_any(path: str) -> pd.DataFrame:
        return pd.read_csv(path, sep=None, engine="python")
    
    def find_trackmate_spots_csv(path: str) -> Optional[str]:
        p = PPath(path)
        if p.is_file() and p.suffix.lower() == ".csv":
            return str(p)
        if p.is_dir():
            for name in PREFS:
                cand = p / name
                if cand.exists():
                    return str(cand)
            for cand in p.glob("*.csv"):
                try:
                    df = read_any(str(cand))
                except Exception:
                    continue
                low = [c.lower().strip() for c in df.columns]
                if any(c in low for c in ("position_x","x","x [µm]","x [um]","x (pixel)","x (µm)","x (um)")) and \
                   any(c in low for c in ("position_y","y","y [µm]","y [um]","y (pixel)","y (µm)","y (um)")) and \
                   any(c in low for c in ("frame","spot_frame","t","frame index","frame_index","frame id","frameid")) and \
                   any(c in low for c in ("track_id","track id","trajectory","trackindex","track_index","id_track")):
                    return str(cand)
        return None
    
    def _col(df, *cands):
        low = {c.lower(): c for c in df.columns}
        for c in cands:
            if c in low:
                return low[c]
        return None
    
    def clean_trackmate_csv(in_csv: str) -> pd.DataFrame:
        df = read_any(in_csv)
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        
        cx = _col(df, "position_x","x","x [µm]","x [um]","x (pixel)","x (µm)","x (um)")
        cy = _col(df, "position_y","y","y [µm]","y [um]","y (pixel)","y (µm)","y (um)")
        cf = _col(df, "frame","spot_frame","t","frame index","frame_index","frame id","frameid")
        ct = _col(df, "track_id","track id","trajectory","trackindex","track_index","id_track")
        
        if not all([cx,cy,cf,ct]):
            missing = [n for n,v in zip(["x","y","frame","track_id"],[cx,cy,cf,ct]) if v is None]
            raise RuntimeError(f"Missing required columns: {missing} in {in_csv}")
        
        out = df.copy()
        out["x"] = pd.to_numeric(out[cx], errors="coerce")
        out["y"] = pd.to_numeric(out[cy], errors="coerce")
        out["frame"] = pd.to_numeric(out[cf], errors="coerce")
        out["track_id"] = pd.to_numeric(out[ct], errors="coerce")
        out = out.dropna(subset=["x","y","frame","track_id"]).copy()
        out["track_id"] = out["track_id"].astype("int64")
        return out
    
    def _nearest_date_code(p: PPath) -> Optional[str]:
        for part in p.parts[::-1]:
            if re.fullmatch(r"\d{8}", part or ""):
                return part
        return None
    
    def _normalize_datecode(raw: Optional[str], name: str, want_format: str) -> Optional[str]:
        if raw:
            if want_format.upper() == "YYMMDD" and len(raw) == 8:
                return raw[2:]
            if want_format.upper() == "YYYYMMDD" and len(raw) == 6:
                return "20" + raw
            return raw
        m8 = DATE_YYYYMMDD_RE.search(name)
        if m8:
            raw = m8.group(1)
            return raw[2:] if want_format.upper() == "YYMMDD" else raw
        m6 = DATE_YYMMDD_RE.search(name)
        if m6:
            raw = m6.group(1)
            return raw if want_format.upper() == "YYMMDD" else ("20" + raw)
        return None
    
    def infer_condition_and_replicate(p: PPath) -> Tuple[str, str]:
        spool_dir = None
        for anc in [p.parent] + list(p.parents):
            if SPOOL_RE.search(anc.name):
                spool_dir = anc
                break
        
        if spool_dir is not None:
            m = SPOOL_RE.search(spool_dir.name)
            rep = m.group(1) if m else None
            cond = spool_dir.parent.name if spool_dir.parent else "UNKNOWN"
        else:
            cond = p.parent.name
            rep = None
        
        if rep is None:
            m = re.search(r"(\d+)", p.stem)
            rep = m.group(1) if m else "01"
        
        return cond, rep
    
    def standardize_to_traj(in_csv: str, out_dir: str, include_date: bool = False, date_format: str = "YYMMDD") -> str:
        in_path = PPath(in_csv).resolve()
        out_base = PPath(out_dir).resolve()
        out_base.mkdir(parents=True, exist_ok=True)
        
        condition, rep = infer_condition_and_replicate(in_path)
        raw_date = _nearest_date_code(in_path)
        datecode = _normalize_datecode(raw_date, in_path.name, date_format)
        
        clean = clean_trackmate_csv(str(in_path))
        clean["condition"] = condition
        clean["replicate"] = int(rep) if rep.isdigit() else rep
        
        if include_date and datecode:
            out_csv = out_base / f"Traj_{condition}-{datecode}_{rep}.csv"
        else:
            out_csv = out_base / f"Traj_{condition}_{rep}.csv"
        
        clean.to_csv(out_csv, index=False)
        return str(out_csv)
    
    root = PPath(args.root_dir).resolve()
    out_base = PPath(args.out_dir).resolve() if args.out_dir else (root / "raw")
    out_base.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for cand in root.glob("*.csv"):
        try:
            out = standardize_to_traj(str(cand), str(out_base), args.include_date, args.date_format)
            print("wrote", out)
            count += 1
        except Exception as e:
            print("[prep] skip", cand, "->", e)
    
    for d in root.glob("**/*"):
        if not d.is_dir():
            continue
        spots = find_trackmate_spots_csv(str(d))
        if not spots:
            continue
        try:
            out = standardize_to_traj(spots, str(out_base), args.include_date, args.date_format)
            print("wrote", out)
            count += 1
        except Exception as e:
            print("[prep] skip", d, "->", e)
    
    if count == 0:
        print("No TrackMate tables found under", args.root_dir)
    else:
        print(f"Converted {count} file(s). Outputs at:", out_base)

def cmd_prep_nd2_to_tiff(args):
    """ND2 to TIFF conversion - from splitsave.py"""
    if not HAVE_ND2 or not HAVE_TIFFFILE:
        raise SystemExit("nd2 and tifffile required for ND2 conversion")
    
    def extract_first_channel(arr):
        if arr.ndim == 3:
            if arr.shape[0] < 10:
                return arr[0, :, :]
            elif arr.shape[-1] < 10:
                return arr[..., 0]
        return arr
    
    channel = int(getattr(args, "channel", 0) or 0)

    def _read_nd2_frames(file_path, channel: int = 0):
        """Read frames from an ND2 file using the nd2 library, selecting one channel."""
        with nd2.ND2File(file_path) as f:
            data = f.asarray()
        return _select_channel(data, channel)
    
    def process_movie(file_path, output_dir):
        base = os.path.basename(file_path)
        newimage = base.replace(".nd2", ".tif")
        try:
            output_array = _read_nd2_frames(file_path, channel=channel)
            output_path = os.path.join(output_dir, newimage)
            tifffile.imwrite(output_path, output_array)
            print(f"Processed movie: {base} -> {newimage} (channel {channel})")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    def process_image(file_path, output_dir):
        base = os.path.basename(file_path)
        base_no_ext, _ = os.path.splitext(base)
        modified_name = base_no_ext.replace("_BF_", "_")
        if not modified_name.startswith("BF_"):
            modified_name = "BF_" + modified_name
        newimage = modified_name + ".tif"
        
        try:
            output_array = _read_nd2_frames(file_path, channel=channel)
            output_path = os.path.join(output_dir, newimage)
            tifffile.imwrite(output_path, output_array)
            print(f"Processed BF image: {base} -> {newimage} (channel {channel})")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    output_dir = os.path.join(args.directory, "tif-files")
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(args.directory) if f.lower().endswith(".nd2")]
    file_paths = [os.path.join(args.directory, f) for f in files]
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for file_path in file_paths:
            base = os.path.basename(file_path)
            if "BF" in base:
                futures.append(executor.submit(process_image, file_path, output_dir))
            else:
                futures.append(executor.submit(process_movie, file_path, output_dir))
        for future in futures:
            future.result()

def cmd_prep_stack_channels(args):
    """Stack two-channel images - from spoolpool.py"""
    if not HAVE_TIFFFILE:
        raise SystemExit("tifffile required for stacking")
    
    SLICE_RE = re.compile(r".*?_(\d+)\.tif$", re.IGNORECASE)
    
    def find_slice_number(path):
        m = SLICE_RE.match(os.path.basename(path))
        if not m:
            return None
        return int(m.group(1))
    
    def sorted_slices(folder):
        files = glob.glob(os.path.join(folder, "*.tif"))
        pairs = []
        for f in files:
            n = find_slice_number(f)
            if n is not None:
                pairs.append((n, f))
        pairs.sort(key=lambda x: x[0])
        return pairs
    
    if not os.path.isdir(args.spool_dir):
        raise SystemExit(f"Input spool directory not found: {args.spool_dir}")
    left_dir = os.path.join(args.spool_dir, "lefts")
    right_dir = os.path.join(args.spool_dir, "rights")
    if not os.path.isdir(left_dir) or not os.path.isdir(right_dir):
        raise SystemExit("Expected 'lefts' and 'rights' subfolders inside the spool directory.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    left_slices = dict(sorted_slices(left_dir))
    right_slices = dict(sorted_slices(right_dir))
    
    common_idxs = sorted(set(left_slices.keys()) & set(right_slices.keys()))
    if not common_idxs:
        raise SystemExit("No matching slice numbers found between lefts and rights.")
    
    first_left = tifffile.imread(left_slices[common_idxs[0]])
    first_right = tifffile.imread(right_slices[common_idxs[0]])
    
    if first_left.shape != first_right.shape:
        raise SystemExit(f"Left/right image shapes differ: {first_left.shape} vs {first_right.shape}")
    
    y, x = first_left.shape
    dtype = first_left.dtype
    
    Z = len(common_idxs)
    stack = np.empty((Z, 2, y, x), dtype=dtype)
    
    for zi, idx in enumerate(common_idxs):
        L = tifffile.imread(left_slices[idx])
        R = tifffile.imread(right_slices[idx])
        if L.shape != (y, x) or R.shape != (y, x):
            raise SystemExit(f"Slice {idx} size mismatch from first slice.")
        stack[zi, 0] = L
        stack[zi, 1] = R
        if (zi + 1) % 50 == 0 or zi == Z - 1:
            print(f"Loaded {zi + 1}/{Z} slices", flush=True)
    
    spool_name = os.path.basename(os.path.normpath(args.spool_dir))
    out_path = os.path.join(args.output_dir, f"{spool_name}_2ch.ome.tif")
    
    print(f"Writing OME-TIFF: {out_path}")
    tifffile.imwrite(
        out_path,
        stack,
        photometric="minisblack",
        metadata={"axes": "ZCYX", "Channel": {"Name": ["left", "right"]}},
        ome=True,
        compression="zlib",
    )
    print("Done.")

def cmd_prep_find_ome_tif(args):
    """Find OME-TIFF files - from find_ome-tif.py"""
    import csv
    
    matching_files = []
    for root, _, files in os.walk(args.input_folder):
        for file in files:
            if "_MMStack_Pos0.ome.tif" in file:
                full_path = os.path.join(root, file)
                matching_files.append([full_path])
    
    folder_name = os.path.basename(os.path.normpath(args.input_folder))
    output_csv = os.path.join(args.output_dir, f"{folder_name}.csv")
    
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["File Path"])
        writer.writerows(matching_files)
    
    print(f"Found {len(matching_files)} matching files.")
    print(f"CSV written to: {output_csv}")

# ============================================================================
# TRACK SUBCOMMAND - Tracking Operations
# ============================================================================

def cmd_track_trackpy(args):
    """Trackpy-based tracking - from tp_dwell.5.5.py"""
    if not HAVE_TRACKPY or not HAVE_ND2:
        raise SystemExit("trackpy and nd2 required")
    
    input_root = Path(args.input_directory)
    out_dir_arg = getattr(args, "out_dir", None)
    output_root = Path(out_dir_arg).resolve() if out_dir_arg else input_root
    output_root.mkdir(parents=True, exist_ok=True)
    # Context lives alongside the outputs so downstream commands find it there.
    context_root = output_root
    # Fall back to any context saved at the input directory (back-compat for
    # pre-existing runs that predate --out-dir).
    track_ctx = _context_params(context_root, ["track", "trackpy"])
    if not track_ctx and output_root != input_root:
        track_ctx = _context_params(input_root, ["track", "trackpy"])
    # Keep argparse defaults authoritative when no explicit flag or saved context exists.
    args.mpp = float(_resolve_param(_argv_has_flag("--mpp"), args.mpp, track_ctx.get("mpp"), args.mpp))
    args.fps = float(_resolve_param(_argv_has_flag("--fps"), args.fps, track_ctx.get("fps"), args.fps))
    args.diam = int(_resolve_param(_argv_has_flag("--diam"), args.diam, track_ctx.get("diam"), args.diam))
    args.range = int(_resolve_param(_argv_has_flag("--range"), args.range, track_ctx.get("range"), args.range))
    args.memory = int(_resolve_param(_argv_has_flag("--memory"), args.memory, track_ctx.get("memory"), args.memory))
    args.minmass = float(_resolve_param(_argv_has_flag("--minmass"), args.minmass, track_ctx.get("minmass"), args.minmass))
    args.percentile = float(_resolve_param(_argv_has_flag("--percentile"), args.percentile, track_ctx.get("percentile"), args.percentile))
    args.diffusion_tau = int(_resolve_param(
        _argv_has_flag("--diffusion_tau"),
        args.diffusion_tau,
        track_ctx.get("diffusion_tau"),
        args.diffusion_tau,
    ))
    args.min_separation = int(_resolve_param(
        _argv_has_flag("--min-separation", "--min_separation"),
        args.min_separation,
        track_ctx.get("min_separation"),
        args.min_separation,
    ))
    args.min_track_length = int(_resolve_param(
        _argv_has_flag("--min-track-length"),
        args.min_track_length,
        track_ctx.get("min_track_length"),
        args.min_track_length,
    ))
    args.channel = int(_resolve_param(
        _argv_has_flag("--channel"),
        args.channel,
        track_ctx.get("channel"),
        args.channel,
    ))
    ctx_cb = track_ctx.get("correct_brightness")
    if not _argv_has_flag("--correct-brightness") and not _argv_has_flag("--no-correct-brightness") and ctx_cb is not None:
        args.correct_brightness = bool(ctx_cb)
    
    def compute_track_alpha(track_df, fps, mpp):
        if "frame" in track_df.index.names:
            track_df = track_df.reset_index()
        track_df = track_df.sort_values("frame")
        positions = track_df[["x", "y"]].to_numpy() * mpp
        n = len(positions)
        if n < 2:
            return np.nan
        
        lags = np.arange(1, n)
        msd_values = [np.mean(np.sum((positions[lag:] - positions[:-lag]) ** 2, axis=1))
                      for lag in lags]
        lag_times = lags / fps
        if len(lag_times) < 2:
            return np.nan
        
        log_lag = np.log(lag_times)
        log_msd = np.log(msd_values)
        slope = np.polyfit(log_lag, log_msd, 1)[0]
        if slope < 0:
            slope = 0
        return slope
    
    def normalize_track_table(track_df: pd.DataFrame) -> pd.DataFrame:
        out = track_df.reset_index(drop=True) if any(name is not None for name in track_df.index.names) else track_df
        for col in ("frame", "particle", "x", "y"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out
    
    def compute_alpha_from_msd_curve(tau_s: np.ndarray, msd_vals: np.ndarray) -> float:
        finite = np.isfinite(tau_s) & np.isfinite(msd_vals) & (tau_s > 0) & (msd_vals > 0)
        if finite.sum() < 2:
            return np.nan
        slope = float(np.polyfit(np.log(tau_s[finite]), np.log(msd_vals[finite]), 1)[0])
        return max(slope, 0.0)
    
    def compute_alpha_map_from_imsd(imsd: pd.DataFrame) -> Dict[Any, float]:
        if imsd is None or imsd.empty:
            return {}
        tau_s = pd.to_numeric(pd.Index(imsd.index), errors="coerce").to_numpy(dtype=float)
        tau_finite = np.isfinite(tau_s) & (tau_s > 0)
        if tau_finite.sum() < 2:
            return {col: np.nan for col in imsd.columns}
        vals_mat = imsd.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        log_tau = np.log(tau_s[tau_finite])
        alpha_map: Dict[Any, float] = {}
        for j, col in enumerate(imsd.columns):
            col_vals = vals_mat[tau_finite, j]
            ok = np.isfinite(col_vals) & (col_vals > 0)
            if ok.sum() < 2:
                alpha_map[col] = np.nan
            else:
                slope = float(np.polyfit(log_tau[ok], np.log(col_vals[ok]), 1)[0])
                alpha_map[col] = max(slope, 0.0)
        return alpha_map
    
    def attach_alpha_from_imsd(filtered_tracks: pd.DataFrame, imsd: pd.DataFrame) -> pd.DataFrame:
        out = normalize_track_table(filtered_tracks)
        alpha_map = compute_alpha_map_from_imsd(imsd)
        lookup = {str(k): v for k, v in alpha_map.items()}
        def _alpha_for_particle(particle):
            if particle in alpha_map:
                return alpha_map[particle]
            return lookup.get(str(particle), np.nan)
        out["alpha"] = out["particle"].map(_alpha_for_particle)
        return out
    
    def _save_fig(fig, path: str):
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
    
    def _to_series(df_or_series):
        if isinstance(df_or_series, pd.Series):
            return df_or_series
        if isinstance(df_or_series, pd.DataFrame):
            if df_or_series.shape[1] == 0:
                return pd.Series(dtype=float)
            return df_or_series.iloc[:, 0]
        return pd.Series(dtype=float)
    
    def _fit_flatfield_poly(filtered_tracks: pd.DataFrame, diff_df: pd.DataFrame):
        """Fit 2D polynomial illumination surface from particle detections.
        Returns (coeffs, mean_x, mean_y, fit_x, fit_y, fit_m) or None on failure."""
        if filtered_tracks is not None and not filtered_tracks.empty and "mass" in filtered_tracks.columns:
            fit_x = filtered_tracks["x"].values
            fit_y = filtered_tracks["y"].values
            fit_m = filtered_tracks["mass"].values
        else:
            fit_x = diff_df["mean_x_px"].values
            fit_y = diff_df["mean_y_px"].values
            fit_m = diff_df["mass_per_frame"].values
        valid = np.isfinite(fit_x) & np.isfinite(fit_y) & np.isfinite(fit_m) & (fit_m > 0)
        if valid.sum() < 20:
            return None
        fit_x, fit_y, fit_m = fit_x[valid], fit_y[valid], fit_m[valid]
        try:
            mean_x, mean_y = fit_x.mean(), fit_y.mean()
            cx = fit_x - mean_x
            cy = fit_y - mean_y
            A = np.column_stack([np.ones(len(cx)), cx, cy, cx**2, cx*cy, cy**2])
            coeffs, _, _, _ = np.linalg.lstsq(A, fit_m, rcond=None)
            return coeffs, mean_x, mean_y, fit_x, fit_y, fit_m
        except Exception:
            return None

    def flatfield_correct_brightness(diff_df: pd.DataFrame, filtered_tracks: pd.DataFrame,
                                      poly_degree: int = 2) -> pd.DataFrame:
        """Fit 2D polynomial illumination profile from per-detection brightness
        and correct each track's mass_per_frame. Adds 'mass_per_frame_ff' column.
        Falls back gracefully if fit fails or insufficient data."""
        if diff_df.empty or "mean_x_px" not in diff_df.columns:
            diff_df["mass_per_frame_ff"] = diff_df.get("mass_per_frame", np.nan)
            diff_df["flatfield_factor"] = 1.0
            return diff_df
        fit_result = _fit_flatfield_poly(filtered_tracks, diff_df)
        if fit_result is None:
            diff_df["mass_per_frame_ff"] = diff_df.get("mass_per_frame", np.nan)
            diff_df["flatfield_factor"] = 1.0
            return diff_df
        coeffs, mean_x, mean_y, _, _, _ = fit_result
        try:
            tx = diff_df["mean_x_px"].values - mean_x
            ty = diff_df["mean_y_px"].values - mean_y
            A_track = np.column_stack([np.ones(len(tx)), tx, ty, tx**2, tx*ty, ty**2])
            predicted = A_track @ coeffs
            median_pred = np.nanmedian(predicted[predicted > 0])
            if median_pred > 0:
                flatfield_factor = predicted / median_pred
            else:
                flatfield_factor = np.ones(len(predicted))
            flatfield_factor = np.clip(flatfield_factor, 0.1, 10.0)
            diff_df["flatfield_factor"] = flatfield_factor
            diff_df["mass_per_frame_ff"] = diff_df["mass_per_frame"] / flatfield_factor
        except Exception:
            diff_df["mass_per_frame_ff"] = diff_df.get("mass_per_frame", np.nan)
            diff_df["flatfield_factor"] = 1.0
        return diff_df

    def save_flatfield_diagnostic(first_frame: Optional[np.ndarray],
                                   filtered_tracks: pd.DataFrame,
                                   diff_df: pd.DataFrame,
                                   movie_output_dir: str, movie_name: str):
        """Save before/after flatfield correction images and illumination surface."""
        if first_frame is None:
            return
        frame = np.array(first_frame, dtype=np.float32)
        if frame.ndim > 2:
            frame = frame[0]

        fit_result = _fit_flatfield_poly(filtered_tracks, diff_df)
        if fit_result is None:
            return
        coeffs, mean_x, mean_y, fit_x, fit_y, fit_m = fit_result

        H, W = frame.shape
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        px = xx - mean_x
        py = yy - mean_y
        surface = (coeffs[0] + coeffs[1]*px + coeffs[2]*py
                   + coeffs[3]*px**2 + coeffs[4]*px*py + coeffs[5]*py**2)
        median_s = np.nanmedian(surface[surface > 0])
        if median_s > 0:
            surface /= median_s
        surface = np.clip(surface, 0.1, 10.0)
        corrected = frame / surface

        vmin, vmax = np.percentile(frame, [1, 99.5])

        fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
        im0 = axes[0].imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0].set_title("Original Frame 0", fontsize=13)
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Intensity")

        im1 = axes[1].imshow(surface, cmap="inferno")
        axes[1].set_title("Fitted Illumination Surface\n(2D polynomial from particle masses)", fontsize=13)
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Flatfield Factor")

        im2 = axes[2].imshow(corrected, cmap="gray", vmin=vmin, vmax=vmax)
        axes[2].set_title("Corrected Frame 0\n(Original / Illumination Surface)", fontsize=13)
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Intensity")

        fig.suptitle(f"Flatfield Brightness Correction — {movie_name}", fontsize=15, y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(movie_output_dir, f"{movie_name}_flatfield_correction.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(7, 6.5))
        im = ax2.imshow(surface, cmap="inferno")
        ax2.scatter(fit_x, fit_y, c=fit_m, cmap="viridis", s=4, alpha=0.3, edgecolors="none")
        ax2.set_title(f"Illumination Surface with Particle Detections\n(color = raw mass) — {movie_name}", fontsize=12)
        ax2.axis("off")
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Flatfield Factor")
        plt.savefig(os.path.join(movie_output_dir, f"{movie_name}_flatfield_surface.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig2)

    def add_brightness_zscore(diff_df: pd.DataFrame) -> pd.DataFrame:
        """Add per-movie z-score normalization of mass_per_frame. Operates on a
        single movie's diffusion table (no movie_name grouping needed)."""
        col = "mass_per_frame_ff" if "mass_per_frame_ff" in diff_df.columns else "mass_per_frame"
        if col not in diff_df.columns or diff_df.empty:
            diff_df["mass_per_frame_zscore"] = np.nan
            return diff_df
        vals = diff_df[col]
        mu = vals.mean()
        sigma = vals.std()
        if sigma > 0:
            diff_df["mass_per_frame_zscore"] = (vals - mu) / sigma
        else:
            diff_df["mass_per_frame_zscore"] = 0.0
        return diff_df

    def build_step_size_table(filtered_tracks: pd.DataFrame, fps: float, mpp: float) -> pd.DataFrame:
        if filtered_tracks.empty:
            return pd.DataFrame(columns=["particle", "frame_prev", "frame", "frame_delta", "dx_px", "dy_px", "step_px", "step_um", "tau_s"])
        ft = filtered_tracks.reset_index(drop=True) if "frame" in filtered_tracks.index.names else filtered_tracks
        ft = ft.sort_values(["particle", "frame"]).reset_index(drop=True)
        ft["frame_prev"] = ft.groupby("particle")["frame"].shift(1)
        ft["x_prev"] = ft.groupby("particle")["x"].shift(1)
        ft["y_prev"] = ft.groupby("particle")["y"].shift(1)
        steps = ft.dropna(subset=["frame_prev", "x_prev", "y_prev"]).copy()
        if steps.empty:
            return pd.DataFrame(columns=["particle", "frame_prev", "frame", "frame_delta", "dx_px", "dy_px", "step_px", "step_um", "tau_s"])
        steps["frame_prev"] = steps["frame_prev"].astype(int)
        steps["frame"] = steps["frame"].astype(int)
        steps["frame_delta"] = steps["frame"] - steps["frame_prev"]
        steps["dx_px"] = steps["x"] - steps["x_prev"]
        steps["dy_px"] = steps["y"] - steps["y_prev"]
        steps["step_px"] = np.sqrt(steps["dx_px"] ** 2 + steps["dy_px"] ** 2)
        steps["step_um"] = steps["step_px"] * float(mpp)
        steps["tau_s"] = steps["frame_delta"] / float(fps)
        return steps[["particle", "frame_prev", "frame", "frame_delta", "dx_px", "dy_px", "step_px", "step_um", "tau_s"]]
    
    def build_diffusion_table(filtered_tracks: pd.DataFrame, imsd: pd.DataFrame, fps: float, mpp: float,
                              diffusion_tau: int) -> pd.DataFrame:
        if filtered_tracks.empty or imsd is None or imsd.empty:
            return pd.DataFrame(columns=[
                "particle", "n_spots", "dwell_time", "alpha", "total_mass",
                "mean_x_px", "mean_y_px", "mass_per_frame",
                "tau_fit_frames", "tau_fit_s", "msd_fit_um2", "D_fit_um2_s", "D_fit_px2_frame",
                "tau_last_frames", "tau_last_s", "msd_last_um2", "D_last_um2_s", "D_last_px2_frame",
            ])
        filtered_tracks = normalize_track_table(filtered_tracks)
        alpha_map = compute_alpha_map_from_imsd(imsd)
        meta = (
            filtered_tracks.groupby("particle", dropna=True)
            .agg(
                n_spots=("frame", "size"),
                dwell_time=("dwell_time", "max"),
                alpha=("alpha", "first") if "alpha" in filtered_tracks.columns else ("frame", lambda x: np.nan),
                total_mass=("total_mass", "first"),
                mean_x_px=("x", "mean"),
                mean_y_px=("y", "mean"),
                mean_mass_per_frame=("mass", "mean"),
            )
        )
        if meta["alpha"].isna().all():
            meta["alpha"] = [alpha_map.get(idx, alpha_map.get(str(idx), np.nan)) for idx in meta.index]
        tau = pd.to_numeric(pd.Index(imsd.index), errors="coerce").to_numpy(dtype=float)
        tau_frames = np.full_like(tau, np.nan, dtype=float)
        finite_tau = np.isfinite(tau)
        tau_frames[finite_tau] = np.rint(tau[finite_tau] * float(fps))
        target_tau_frames = int(max(1, diffusion_tau))
        target_tau_s = target_tau_frames / float(fps)
        meta_lookup = {str(k): k for k in meta.index.tolist()}
        imsd_numeric = imsd.apply(pd.to_numeric, errors="coerce")
        vals_mat = imsd_numeric.to_numpy(dtype=float)
        tau_pos = np.isfinite(tau) & (tau > 0)
        fit_candidate = tau_pos & np.isfinite(tau_frames) & (tau_frames == target_tau_frames)
        px2_frame_divisor = float(mpp) ** 2 * float(fps)
        rows = []
        for j, particle_col in enumerate(imsd_numeric.columns):
            vals = vals_mat[:, j]
            finite = tau_pos & np.isfinite(vals)
            tau_last = msd_last = D_last_um2_s = np.nan
            tau_last_frames = np.nan
            if np.any(finite):
                last_idx = np.where(finite)[0][-1]
                tau_last = float(tau[last_idx])
                tau_last_frames = float(tau_frames[last_idx]) if np.isfinite(tau_frames[last_idx]) else np.nan
                msd_last = float(vals[last_idx])
                D_last_um2_s = float(msd_last / (4.0 * tau_last)) if tau_last > 0 else np.nan
            fit_mask = fit_candidate & np.isfinite(vals)
            msd_fit = D_fit_um2_s = np.nan
            if np.any(fit_mask):
                fit_idx = np.where(fit_mask)[0][0]
                msd_fit = float(vals[fit_idx])
                D_fit_um2_s = float(msd_fit / (4.0 * target_tau_s)) if target_tau_s > 0 else np.nan
            particle_key = particle_col if particle_col in meta.index else meta_lookup.get(str(particle_col))
            row = {
                "particle": particle_col,
                "n_spots": np.nan,
                "dwell_time": np.nan,
                "alpha": np.nan,
                "total_mass": np.nan,
                "mean_x_px": np.nan,
                "mean_y_px": np.nan,
                "mean_mass_per_frame": np.nan,
                "tau_fit_frames": target_tau_frames,
                "tau_fit_s": target_tau_s,
                "msd_fit_um2": msd_fit,
                "D_fit_um2_s": D_fit_um2_s,
                "D_fit_px2_frame": np.nan if not np.isfinite(D_fit_um2_s) else float(D_fit_um2_s / px2_frame_divisor),
                "tau_last_frames": tau_last_frames,
                "tau_last_s": tau_last,
                "msd_last_um2": msd_last,
                "D_last_um2_s": D_last_um2_s,
                "D_last_px2_frame": np.nan if not np.isfinite(D_last_um2_s) else float(D_last_um2_s / px2_frame_divisor),
            }
            if particle_key is not None:
                row.update(meta.loc[particle_key].to_dict())
                row["particle"] = particle_key
            rows.append(row)
        result = pd.DataFrame(rows)
        if not result.empty:
            result["mass_per_frame"] = result["mean_mass_per_frame"]
            result.drop(columns=["mean_mass_per_frame"], inplace=True, errors="ignore")
        return result
    
    def build_msd_summary_table(imsd: pd.DataFrame, emsd) -> pd.DataFrame:
        tau_imsd = pd.to_numeric(pd.Index(imsd.index), errors="coerce").to_numpy(dtype=float) if imsd is not None and not imsd.empty else np.array([], dtype=float)
        emsd_series = _to_series(emsd)
        tau_emsd = pd.to_numeric(pd.Index(emsd_series.index), errors="coerce").to_numpy(dtype=float) if not emsd_series.empty else np.array([], dtype=float)
        tau = np.union1d(tau_imsd[np.isfinite(tau_imsd)], tau_emsd[np.isfinite(tau_emsd)])
        if tau.size == 0:
            return pd.DataFrame(columns=["tau_s", "emsd_um2", "imsd_mean_um2", "imsd_median_um2", "imsd_n_tracks"])
        out = pd.DataFrame({"tau_s": tau})
        if not emsd_series.empty:
            emsd_clean = pd.Series(pd.to_numeric(emsd_series, errors="coerce").to_numpy(dtype=float), index=tau_emsd)
            emsd_clean = emsd_clean[~emsd_clean.index.duplicated(keep="first")]
            out["emsd_um2"] = out["tau_s"].map(emsd_clean.to_dict())
        else:
            out["emsd_um2"] = np.nan
        if imsd is not None and not imsd.empty:
            imsd_numeric = imsd.apply(pd.to_numeric, errors="coerce")
            imsd_numeric.index = tau_imsd
            imsd_numeric = imsd_numeric[~imsd_numeric.index.duplicated(keep="first")]
            out["imsd_mean_um2"] = out["tau_s"].map(imsd_numeric.mean(axis=1).to_dict())
            out["imsd_median_um2"] = out["tau_s"].map(imsd_numeric.median(axis=1).to_dict())
            out["imsd_n_tracks"] = out["tau_s"].map(imsd_numeric.notna().sum(axis=1).to_dict())
        else:
            out["imsd_mean_um2"] = np.nan
            out["imsd_median_um2"] = np.nan
            out["imsd_n_tracks"] = 0
        return out
    
    def plot_histogram(values: np.ndarray, xlabel: str, title: str, out_png: str):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        fig = plt.figure(figsize=(6, 4))
        plt.hist(values, bins=50)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        _save_fig(fig, out_png)
    
    def plot_msd_vs_tau(msd_df: pd.DataFrame, title: str, out_png: str):
        if msd_df.empty:
            return
        fig = plt.figure(figsize=(6, 4.5))
        plotted = False
        for col, label in [
            ("emsd_um2", "EMSD"),
            ("imsd_mean_um2", "Mean IMSD"),
            ("imsd_median_um2", "Median IMSD"),
        ]:
            if col not in msd_df.columns:
                continue
            sub = msd_df[["tau_s", col]].dropna()
            sub = sub[(sub["tau_s"] > 0) & (sub[col] > 0)]
            if sub.empty:
                continue
            plt.loglog(sub["tau_s"], sub[col], marker="o", markersize=3, linewidth=1.2, label=label)
            plotted = True
        if not plotted:
            plt.close(fig)
            return
        plt.xlabel("Tau (s)")
        plt.ylabel("MSD (um^2)")
        plt.title(title)
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        _save_fig(fig, out_png)
    
    def write_movie_summary_outputs(movie_output_dir: str, movie_name: str, filtered_tracks: pd.DataFrame,
                                    imsd: pd.DataFrame, emsd, fps: float, mpp: float,
                                    diffusion_tau: int,
                                    correct_brightness: bool = True,
                                    first_frame: Optional[np.ndarray] = None):
        fixed_tau_s = float(diffusion_tau) / float(fps)
        step_df = build_step_size_table(filtered_tracks, fps=fps, mpp=mpp)
        step_csv = os.path.join(movie_output_dir, f"{movie_name}_step_sizes.csv")
        step_df.to_csv(step_csv, index=False)
        plot_histogram(
            step_df["step_um"].to_numpy(dtype=float) if not step_df.empty else np.array([]),
            xlabel="Step size (um)",
            title=f"Step Size Histogram: {movie_name}",
            out_png=os.path.join(movie_output_dir, f"{movie_name}_step_size_hist.png"),
        )
        
        diff_df = build_diffusion_table(filtered_tracks, imsd, fps=fps, mpp=mpp, diffusion_tau=diffusion_tau)
        if correct_brightness:
            diff_df = flatfield_correct_brightness(diff_df, filtered_tracks)
            save_flatfield_diagnostic(first_frame, filtered_tracks, diff_df,
                                      movie_output_dir, movie_name)
        else:
            diff_df["mass_per_frame_ff"] = diff_df.get("mass_per_frame", np.nan)
            diff_df["flatfield_factor"] = 1.0
        diff_df = add_brightness_zscore(diff_df)
        diff_csv = os.path.join(movie_output_dir, f"{movie_name}_diffusion_coefficients.csv")
        diff_df.to_csv(diff_csv, index=False)
        plot_histogram(
            diff_df["D_fit_um2_s"].to_numpy(dtype=float) if not diff_df.empty else np.array([]),
            xlabel=f"Diffusion coefficient at tau={fixed_tau_s:.6g} s ({diffusion_tau} frame) (um^2/s)",
            title=f"Fixed-Tau Diffusion Histogram: {movie_name}",
            out_png=os.path.join(movie_output_dir, f"{movie_name}_diffusion_hist_fixed_tau.png"),
        )
        plot_histogram(
            diff_df["D_last_um2_s"].to_numpy(dtype=float) if not diff_df.empty else np.array([]),
            xlabel="Diffusion coefficient at last finite tau (um^2/s)",
            title=f"Last-Tau Diffusion Histogram: {movie_name}",
            out_png=os.path.join(movie_output_dir, f"{movie_name}_diffusion_hist_last_tau.png"),
        )
        
        msd_df = build_msd_summary_table(imsd, emsd)
        msd_csv = os.path.join(movie_output_dir, f"{movie_name}_MSD_vs_tau.csv")
        msd_df.to_csv(msd_csv, index=False)
        plot_msd_vs_tau(
            msd_df,
            title=f"MSD vs Tau: {movie_name}",
            out_png=os.path.join(movie_output_dir, f"{movie_name}_MSD_vs_tau.png"),
        )
    
    def write_condition_summary_outputs(input_directory: str, folder_conditions: Dict[str, str],
                                        diffusion_tau: int):
        grouped_movies: Dict[str, List[str]] = {}
        for movie_name, condition in folder_conditions.items():
            grouped_movies.setdefault(condition, []).append(movie_name)
        if not grouped_movies:
            return
        root_out = os.path.join(input_directory, "condition_summaries")
        os.makedirs(root_out, exist_ok=True)
        for condition, movie_names in grouped_movies.items():
            cond_out = os.path.join(root_out, simple_clean(condition))
            os.makedirs(cond_out, exist_ok=True)
            
            step_parts = []
            diff_parts = []
            msd_parts = []
            for movie_name in sorted(movie_names):
                movie_dir = os.path.join(input_directory, movie_name)
                step_path = os.path.join(movie_dir, f"{movie_name}_step_sizes.csv")
                diff_path = os.path.join(movie_dir, f"{movie_name}_diffusion_coefficients.csv")
                msd_path = os.path.join(movie_dir, f"{movie_name}_MSD_vs_tau.csv")
                if os.path.exists(step_path):
                    step_df = pd.read_csv(step_path, low_memory=False)
                    if not step_df.empty:
                        step_df.insert(0, "movie_name", movie_name)
                        step_parts.append(step_df)
                if os.path.exists(diff_path):
                    diff_df = pd.read_csv(diff_path, low_memory=False)
                    if not diff_df.empty:
                        diff_df.insert(0, "movie_name", movie_name)
                        diff_parts.append(diff_df)
                if os.path.exists(msd_path):
                    msd_df = pd.read_csv(msd_path, low_memory=False)
                    if not msd_df.empty:
                        msd_df.insert(0, "movie_name", movie_name)
                        msd_parts.append(msd_df)
            
            if step_parts:
                cond_steps = pd.concat(step_parts, ignore_index=True)
                cond_steps.to_csv(os.path.join(cond_out, f"{simple_clean(condition)}_step_sizes.csv"), index=False)
                plot_histogram(
                    cond_steps["step_um"].to_numpy(dtype=float),
                    xlabel="Step size (um)",
                    title=f"Step Size Histogram: {condition}",
                    out_png=os.path.join(cond_out, f"{simple_clean(condition)}_step_size_hist.png"),
                )
            if diff_parts:
                cond_diff = pd.concat(diff_parts, ignore_index=True)
                fixed_tau_s = float(diffusion_tau) / float(args.fps)
                cond_diff.to_csv(os.path.join(cond_out, f"{simple_clean(condition)}_diffusion_coefficients.csv"), index=False)
                plot_histogram(
                    cond_diff["D_fit_um2_s"].to_numpy(dtype=float),
                    xlabel=f"Diffusion coefficient at tau={fixed_tau_s:.6g} s ({diffusion_tau} frame) (um^2/s)",
                    title=f"Fixed-Tau Diffusion Histogram: {condition}",
                    out_png=os.path.join(cond_out, f"{simple_clean(condition)}_diffusion_hist_fixed_tau.png"),
                )
                plot_histogram(
                    cond_diff["D_last_um2_s"].to_numpy(dtype=float),
                    xlabel="Diffusion coefficient at last finite tau (um^2/s)",
                    title=f"Last-Tau Diffusion Histogram: {condition}",
                    out_png=os.path.join(cond_out, f"{simple_clean(condition)}_diffusion_hist_last_tau.png"),
                )
            if msd_parts:
                cond_msd_all = pd.concat(msd_parts, ignore_index=True)
                cond_msd_all.to_csv(os.path.join(cond_out, f"{simple_clean(condition)}_MSD_vs_tau_all_movies.csv"), index=False)
                summary = (
                    cond_msd_all.groupby("tau_s", dropna=True)
                    .agg(
                        emsd_mean_um2=("emsd_um2", _nanmean_series),
                        emsd_median_um2=("emsd_um2", _nanmedian_series),
                        imsd_mean_um2=("imsd_mean_um2", _nanmean_series),
                        imsd_median_um2=("imsd_median_um2", _nanmedian_series),
                        n_movies=("movie_name", pd.Series.nunique),
                    )
                    .reset_index()
                    .sort_values("tau_s")
                )
                summary.to_csv(os.path.join(cond_out, f"{simple_clean(condition)}_MSD_vs_tau_summary.csv"), index=False)
                plot_msd_vs_tau(
                    summary.rename(columns={"emsd_mean_um2": "emsd_um2", "imsd_mean_um2": "imsd_mean_um2", "imsd_median_um2": "imsd_median_um2"}),
                    title=f"MSD vs Tau: {condition}",
                    out_png=os.path.join(cond_out, f"{simple_clean(condition)}_MSD_vs_tau.png"),
                )
    
    def process_file(file_path):
        movie_name = os.path.splitext(os.path.basename(file_path))[0]
        
        if "BF_" in movie_name:
            print(f"Skipping file {file_path} because it contains 'BF_' in the title.")
            return (None, [], None)
        
        file_extension = os.path.splitext(file_path)[1].lower()
        condition_name = os.path.basename(os.path.dirname(file_path))
        print(f"Processing {file_path} (Condition: {condition_name})...")
        
        frames = None
        channel = int(getattr(args, "channel", 0) or 0)
        if file_extension == '.nd2':
            with nd2.ND2File(file_path) as f:
                data = f.asarray()
            data = _select_channel(data, channel)
            frames = [data[i].astype(np.float32) for i in range(data.shape[0])]
            del data
        elif file_extension in ['.tif', '.tiff']:
            if HAVE_PIMS and channel == 0:
                # Preserve lazy-loading default behaviour for single-channel TIFFs.
                frames = pims.open(file_path)
            else:
                raw = tifffile.imread(file_path)
                if raw.ndim >= 4:
                    raw = _select_channel(raw, channel)
                frames = raw
        else:
            print(f"Unsupported file format: {file_extension}. Skipping.")
            return (condition_name, [], movie_name)
        
        try:
            # Output goes to the resolved output root, with movie-named subdirectory.
            movie_output_dir = os.path.join(str(output_root), movie_name)
            os.makedirs(movie_output_dir, exist_ok=True)
            
            particles_file = os.path.join(movie_output_dir, f"{movie_name}_particles.csv")
            tracks_file = os.path.join(movie_output_dir, f"{movie_name}_tracks.csv")
            imsd_file = os.path.join(movie_output_dir, f"{movie_name}_IMSD.csv")
            emsd_file = os.path.join(movie_output_dir, f"{movie_name}_EMSD.csv")
            
            if os.path.exists(particles_file) and os.path.exists(tracks_file):
                print(f"Reusing existing tracks for {movie_name}; recomputing MSD/alpha summaries with current fps/mpp.")
                try:
                    filtered_tracks = normalize_track_table(pd.read_csv(tracks_file, low_memory=False))
                    if filtered_tracks.empty:
                        return (condition_name, [0], movie_name)
                    imsd = tp.imsd(filtered_tracks, args.mpp, args.fps, max_lagtime=25)
                    emsd = tp.emsd(filtered_tracks, args.mpp, args.fps, max_lagtime=25)
                    filtered_tracks = attach_alpha_from_imsd(filtered_tracks, imsd)
                    filtered_tracks.to_csv(tracks_file, index=False)
                    imsd.to_csv(imsd_file)
                    emsd.to_csv(emsd_file)
                    _first = frames[0] if frames is not None and len(frames) > 0 else None
                    write_movie_summary_outputs(
                        movie_output_dir, movie_name, filtered_tracks, imsd, emsd,
                        fps=args.fps, mpp=args.mpp, diffusion_tau=args.diffusion_tau,
                        correct_brightness=args.correct_brightness,
                        first_frame=_first,
                    )
                    msd_vals = imsd.values.flatten().tolist()
                except Exception:
                    msd_vals = [0]
                return (condition_name, msd_vals, movie_name)
            
            with _trackpy_quiet():
                particles = tp.batch(frames, diameter=args.diam, minmass=args.minmass,
                                     separation=args.min_separation, noise_size=1, percentile=args.percentile, engine='numba')
            particles.to_csv(particles_file, index=False)
            
            try:
                with _trackpy_quiet():
                    tracks = tp.link(particles, search_range=args.range, memory=args.memory)
            except Exception as e:
                print(f"Warning: {e} for movie {movie_name}. Skipping linking and setting MSD=0.")
                return (condition_name, [0], movie_name)
            
            filtered_tracks = tp.filter_stubs(tracks, args.min_track_length)
            if filtered_tracks.empty:
                return (condition_name, [0], movie_name)
            
            dwell_times = filtered_tracks.groupby("particle")["frame"].count()
            filtered_tracks["dwell_time"] = filtered_tracks["particle"].map(dwell_times)
            filtered_tracks["total_mass"] = filtered_tracks.groupby("particle")["mass"].transform("sum")
            
            if len(frames) > 0:
                first_frame = np.array(frames[0], dtype=np.float32)
                if first_frame.ndim > 2:
                    first_frame = first_frame[0]
                fig, ax = plt.subplots(figsize=(20, 16))
                ax.imshow(first_frame, cmap='gray')
                for particle, group in filtered_tracks.groupby("particle"):
                    group = group.reset_index(drop=True).sort_values("frame")
                    ax.plot(group["x"], group["y"], marker="o", markersize=0.1, linewidth=1.5, alpha=0.9)
                ax.set_title(f"Filtered Tracks Overlay: {movie_name}")
                ax.axis("off")
                overlay_path = os.path.join(movie_output_dir, f"{movie_name}_tracks_overlay.png")
                plt.savefig(overlay_path, bbox_inches='tight', dpi=300)
                plt.close()
            
            imsd = tp.imsd(filtered_tracks, args.mpp, args.fps, max_lagtime=25)
            emsd = tp.emsd(filtered_tracks, args.mpp, args.fps, max_lagtime=25)
            filtered_tracks = attach_alpha_from_imsd(filtered_tracks, imsd)
            
            filtered_tracks.to_csv(tracks_file, index=False)
            imsd.to_csv(imsd_file)
            emsd.to_csv(emsd_file)
            _first = frames[0] if frames is not None and len(frames) > 0 else None
            write_movie_summary_outputs(
                movie_output_dir, movie_name, filtered_tracks, imsd, emsd,
                fps=args.fps, mpp=args.mpp, diffusion_tau=args.diffusion_tau,
                correct_brightness=args.correct_brightness,
                first_frame=_first,
            )
            
            msd_vals = imsd.values.flatten().tolist()
            return (condition_name, msd_vals, movie_name)
        except Exception as e:
            print(f"Error processing {movie_name}: {e}")
            return (condition_name, [0], movie_name)
    
    if args.notrack:
        print("Trackpy analysis skipped. Generating MSD plots only.")
        return
    
    file_paths = [str(p) for p in _discover_movie_inputs(Path(args.input_directory))]
    if not file_paths:
        raise SystemExit(f"No supported movie files found in {args.input_directory}")
    results = []
    for fp in file_paths:
        result = process_file(fp)
        results.append(result)
    
    condition_msd_values = {}
    folder_conditions = {}
    for condition, msd_vals, movie_name in results:
        if condition is None:
            continue
        if condition not in condition_msd_values:
            condition_msd_values[condition] = []
        condition_msd_values[condition].extend(msd_vals)
        if movie_name:
            folder_conditions[movie_name] = condition
    
    if condition_msd_values:
        plt.figure(figsize=(10, 6))
        for condition, msd_values in condition_msd_values.items():
            if msd_values:
                msd_values = np.array(msd_values)
                msd_values = msd_values[np.isfinite(msd_values)]
                msd_values[msd_values <= 0] = 1
                log_msd = np.log10(msd_values)
                log_msd = log_msd[np.isfinite(log_msd)]
                plt.hist(log_msd, bins=100, alpha=0.5,
                         label=condition, histtype='stepfilled')
        plt.xlabel("log10(MSD)")
        plt.ylabel("Frequency")
        plt.title("Histogram of MSD values (log10 scale) by Experimental Condition")
        plt.grid(True)
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(str(output_root), "all_msds_histogram.png"), dpi=300)
        plt.close()
        
        if folder_conditions:
            folder_list = sorted(folder_conditions.keys())
            groupnames_df = pd.DataFrame({
                "folder": folder_list,
                "condition": [folder_conditions.get(f, '') for f in folder_list]
            })
            groupnames_csv_path = os.path.join(str(output_root), "groupnames.csv")
            groupnames_df.to_csv(groupnames_csv_path, index=False)
            print(f"Saved groupnames CSV at {groupnames_csv_path} with {len(folder_list)} folders")
            write_condition_summary_outputs(str(output_root), folder_conditions, args.diffusion_tau)
    
    _save_pipeline_context(context_root, ["track", "trackpy"], {
        "input_directory": str(input_root),
        "out_dir": str(output_root),
        "mpp": float(args.mpp),
        "fps": float(args.fps),
        "dt": float(1.0 / args.fps) if float(args.fps) != 0 else None,
        "diam": int(args.diam),
        "range": int(args.range),
        "memory": int(args.memory),
        "minmass": float(args.minmass),
        "percentile": float(args.percentile),
        "diffusion_tau": int(args.diffusion_tau),
        "min_separation": int(args.min_separation),
        "min_track_length": int(args.min_track_length),
        "correct_brightness": bool(args.correct_brightness),
        "channel": int(args.channel),
    })

# ============================================================================
# ANALYZE SUBCOMMAND - Analysis Operations
# ============================================================================

def cmd_analyze_compare_conditions(args):
    """Compare diffusion coefficients, step sizes, and brightness across
    experimental conditions.  Reads per-movie CSVs from the directory tree
    produced by ``track trackpy`` (each movie in its own subfolder) and
    pools them by condition using ``groupnames.csv``.

    Outputs a ``compare_conditions/`` folder with publication-quality
    figures, GMM subpopulation tables, brightness-diffusivity cross-
    tabulations, and statistical test results.
    """
    import warnings
    warnings.filterwarnings("ignore")

    input_dir = Path(args.input_dir)
    results_dir = input_dir / "compare_conditions"
    results_dir.mkdir(parents=True, exist_ok=True)

    NT_COLOR = "#2166AC"
    TG_COLOR = "#B2182B"
    COMP_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#CC79A7"]

    # ------------------------------------------------------------------
    # Discover conditions: prefer groupnames.csv, else use --conditions
    # ------------------------------------------------------------------
    groupnames_path = input_dir / "groupnames.csv"
    if groupnames_path.exists():
        gn = pd.read_csv(groupnames_path)
        folder_to_cond = dict(zip(gn["folder"], gn["condition"]))
    else:
        folder_to_cond = {}

    # Collect per-movie diff and step CSVs
    diff_parts, step_parts = [], []
    for movie_dir in sorted(input_dir.iterdir()):
        if not movie_dir.is_dir():
            continue
        mname = movie_dir.name
        diff_csv = movie_dir / f"{mname}_diffusion_coefficients.csv"
        step_csv = movie_dir / f"{mname}_step_sizes.csv"
        cond = folder_to_cond.get(mname)
        if cond is None:
            # Try to infer from name
            for token in (args.conditions or []):
                if token.lower() in mname.lower():
                    cond = token
                    break
        if cond is None:
            # Fallback: use prefix before first underscore-digit
            cond = mname.split("_")[0] if "_" in mname else mname
        if diff_csv.exists():
            df = pd.read_csv(diff_csv, low_memory=False)
            if not df.empty:
                df.insert(0, "movie_name", mname)
                df["condition"] = cond
                diff_parts.append(df)
        if step_csv.exists():
            sf = pd.read_csv(step_csv, low_memory=False)
            if not sf.empty:
                sf.insert(0, "movie_name", mname)
                sf["condition"] = cond
                step_parts.append(sf)

    # Also discover flat *_diffusion_coefficients.csv files directly in input_dir.
    # These are pre-pooled condition CSVs (e.g. from multiple experiment dates).
    # Expected naming: {date}_{condition}_diffusion_coefficients.csv
    # or {anything}_{condition}_diffusion_coefficients.csv
    # Each must have a movie_name column already.
    for csv_path in sorted(input_dir.glob("*_diffusion_coefficients.csv")):
        if csv_path.is_dir():
            continue
        if csv_path.parent != input_dir:
            continue
        stem = csv_path.stem.replace("_diffusion_coefficients", "")
        df = pd.read_csv(csv_path, low_memory=False)
        if df.empty or "movie_name" not in df.columns:
            continue
        cond = None
        for token in (args.conditions or []):
            if token.lower() in stem.lower():
                cond = token
                break
        if cond is None:
            parts = stem.split("_")
            for p in parts:
                if p.lower() in ("nt", "tg", "tg30", "dmso", "thapsigargin"):
                    cond = p
                    break
            if cond is None:
                cond = parts[-1] if parts else stem
        # Prefix movie_name with stem to avoid collisions across experiments
        df["movie_name"] = stem + "/" + df["movie_name"].astype(str)
        df["condition"] = cond
        df["_source_file"] = csv_path.name
        diff_parts.append(df)
        print(f"  Loaded flat CSV: {csv_path.name} -> condition '{cond}', "
              f"{len(df)} tracks, {df['movie_name'].nunique()} movies")

    if not diff_parts:
        raise SystemExit("No diffusion coefficient CSVs found.")
    diff_all = pd.concat(diff_parts, ignore_index=True)
    step_all = pd.concat(step_parts, ignore_index=True) if step_parts else pd.DataFrame()

    # ------------------------------------------------------------------
    # MPP rescaling (correct pixel size mismatch for specific experiments)
    # ------------------------------------------------------------------
    if getattr(args, 'mpp_rescale', None):
        d_cols_to_scale = [c for c in ["D_fit_um2_s", "D_last_um2_s", "D_um2_s",
                                        "msd_fit_um2", "msd_last_um2"]
                           if c in diff_all.columns]
        for pattern, factor_str in args.mpp_rescale:
            factor = float(factor_str)
            mask = diff_all["movie_name"].str.contains(pattern, case=False, na=False)
            n_matched = mask.sum()
            if n_matched > 0:
                for col in d_cols_to_scale:
                    diff_all.loc[mask, col] = diff_all.loc[mask, col] * factor
                print(f"  MPP rescale: multiplied {d_cols_to_scale} by {factor} "
                      f"for {n_matched} tracks matching '{pattern}'")
            else:
                print(f"  MPP rescale WARNING: pattern '{pattern}' matched 0 tracks")

    # ------------------------------------------------------------------
    # Track-level filtering
    # ------------------------------------------------------------------
    n_before = len(diff_all)
    filter_desc = []
    if getattr(args, 'min_nspots', None) is not None:
        diff_all = diff_all[diff_all["n_spots"] >= args.min_nspots]
        filter_desc.append(f"n_spots >= {args.min_nspots}")
    if getattr(args, 'min_alpha', None) is not None and "alpha" in diff_all.columns:
        diff_all = diff_all[diff_all["alpha"] >= args.min_alpha]
        filter_desc.append(f"alpha >= {args.min_alpha}")
    n_after = len(diff_all)
    if filter_desc:
        diff_all = diff_all.reset_index(drop=True)
        print(f"Track filtering ({', '.join(filter_desc)}): "
              f"{n_before:,} -> {n_after:,} tracks ({n_after/n_before:.1%} retained)")

    # ------------------------------------------------------------------
    # Per-condition localization error subtraction
    # ------------------------------------------------------------------

    # Pick D column early so loc error subtraction targets the right one.
    # Precedence prefers the MSD-fit value over last-lag and the legacy column.
    d_col = "D_fit_um2_s" if "D_fit_um2_s" in diff_all.columns else (
        "D_last_um2_s" if "D_last_um2_s" in diff_all.columns else
        "D_um2_s"
    )

    if getattr(args, 'subtract_loc_error', False):
        loc_alpha_max = getattr(args, 'loc_error_alpha_max', 0.2)
        loc_min_n = getattr(args, 'loc_error_min_nspots', 6)
        if d_col in diff_all.columns and "alpha" in diff_all.columns:
            diff_all[d_col + "_raw"] = diff_all[d_col].copy()
            # Group by (source_file, condition) when source info is available so
            # each experiment gets its own error floor estimate. Falls back to
            # condition-only grouping for directory-loaded data.
            has_source = "_source_file" in diff_all.columns
            if has_source:
                group_cols = ["_source_file", "condition"]
                print("Localization error subtraction (per-experiment, per-condition):")
            else:
                group_cols = ["condition"]
                print("Localization error subtraction (per-condition):")
            for group_vals, grp_idx in diff_all.groupby(group_cols).groups.items():
                if has_source:
                    src, cond = group_vals
                    label = f"{src} / {cond}"
                else:
                    cond = group_vals
                    label = cond
                grp_mask = diff_all.index.isin(grp_idx)
                immobile = diff_all[grp_mask &
                                    (diff_all["alpha"] < loc_alpha_max) &
                                    (diff_all["n_spots"] >= loc_min_n)]
                floor = immobile[d_col + "_raw"].dropna().median()
                if pd.isna(floor) or floor <= 0:
                    print(f"  {label}: could not estimate error floor "
                          f"(n_immobile={len(immobile)}), skipping")
                    continue
                diff_all.loc[grp_mask, d_col] = (
                    diff_all.loc[grp_mask, d_col] - floor
                )
                n_neg = (diff_all.loc[grp_mask, d_col] < 0).sum()
                n_grp = grp_mask.sum()
                print(f"  {label}: floor = {floor:.4f} um^2/s, "
                      f"{n_neg}/{n_grp} tracks -> negative D")
        else:
            print("WARNING: loc error subtraction requested but required columns "
                  f"({d_col}, alpha) not found. Skipping.")

    conditions = sorted(diff_all["condition"].unique())
    if len(conditions) < 2:
        print(f"Warning: only {len(conditions)} condition(s) found: {conditions}. Comparison plots will still be generated.")

    # Assign colors
    cond_colors = {}
    palette = [NT_COLOR, TG_COLOR, "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
    for i, c in enumerate(conditions):
        cond_colors[c] = palette[i % len(palette)]

    # d_col was already selected above, before loc error subtraction.
    print(f"Using diffusion column: {d_col}")

    # Brightness column
    bright_col = "mass_per_frame_ff" if "mass_per_frame_ff" in diff_all.columns else "mass_per_frame"
    has_brightness = bright_col in diff_all.columns
    if has_brightness:
        print(f"Using brightness column: {bright_col}")
    else:
        print("No mass_per_frame column found; skipping brightness analysis.")

    # Z-score column
    zscore_col = "mass_per_frame_zscore"
    has_zscore = zscore_col in diff_all.columns

    # ------------------------------------------------------------------
    # Helper: dual histogram (linear + log Y)
    # ------------------------------------------------------------------
    def dual_hist(data_dict, bins, xlabel, title, outpath, xlim=None):
        fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        for ax, yscale in zip(axes, ["linear", "log"]):
            for cond in conditions:
                if cond not in data_dict:
                    continue
                vals = data_dict[cond]
                ax.hist(vals, bins=bins, alpha=0.6, color=cond_colors[cond],
                        label=f"{cond} (n={len(vals):,})", density=True)
            ax.set_yscale(yscale)
            ax.set_ylabel("Density")
            ax.legend(frameon=False)
            if yscale == "log":
                ax.set_ylim(bottom=1e-3)
        axes[1].set_xlabel(xlabel)
        axes[0].set_title(title)
        if xlim:
            axes[0].set_xlim(xlim)
        plt.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  Saved {outpath}")

    def gmm_bic_select(values, max_k=5):
        vals = values.reshape(-1, 1)
        bics, models = [], []
        for k in range(1, max_k + 1):
            g = GaussianMixture(k, random_state=42, n_init=5).fit(vals)
            bics.append(g.bic(vals))
            models.append(g)
        best = int(np.argmin(bics))
        return models[best], bics

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    print("Computing summary statistics...")
    summary_rows = []
    for metric, data, col in [("D", diff_all, d_col), ("Step", step_all, "step_um")]:
        if data.empty or col not in data.columns:
            continue
        for cond in conditions:
            vals = data.loc[data["condition"] == cond, col].dropna()
            summary_rows.append({
                "Metric": metric, "Condition": cond, "N": len(vals),
                "Mean": vals.mean(), "Median": vals.median(), "Std": vals.std(),
                "Q25": vals.quantile(0.25), "Q75": vals.quantile(0.75),
            })
    if has_brightness:
        for cond in conditions:
            vals = diff_all.loc[diff_all["condition"] == cond, bright_col].dropna()
            summary_rows.append({
                "Metric": "Brightness", "Condition": cond, "N": len(vals),
                "Mean": vals.mean(), "Median": vals.median(), "Std": vals.std(),
                "Q25": vals.quantile(0.25), "Q75": vals.quantile(0.75),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "summary_statistics.csv", index=False)

    # ------------------------------------------------------------------
    # Figure 1: D histograms
    # ------------------------------------------------------------------
    print("Plotting D histograms...")
    d_by_cond = {c: diff_all.loc[diff_all["condition"] == c, d_col].dropna().values for c in conditions}
    d_xlim = args.d_xlim
    dual_hist(d_by_cond, np.linspace(0, d_xlim, 80),
              r"Diffusion Coefficient ($\mu m^2/s$)",
              "Diffusion Coefficient Distributions",
              str(results_dir / "fig1_diff_coeff_hist.png"), xlim=(0, d_xlim))

    # ------------------------------------------------------------------
    # Figure 2: Step size histograms
    # ------------------------------------------------------------------
    if not step_all.empty and "step_um" in step_all.columns:
        print("Plotting step size histograms...")
        s_by_cond = {c: step_all.loc[step_all["condition"] == c, "step_um"].dropna().values for c in conditions}
        step_xlim = args.step_xlim
        dual_hist(s_by_cond, np.linspace(0, step_xlim, 80),
                  r"Step Size ($\mu m$)", "Step Size Distributions",
                  str(results_dir / "fig2_step_size_hist.png"), xlim=(0, step_xlim))

    # ------------------------------------------------------------------
    # Figure 3: log10(D) histograms
    # ------------------------------------------------------------------
    print("Plotting log10(D) histograms...")
    logd_by_cond = {c: np.log10(np.clip(d_by_cond[c], 1e-6, None)) for c in conditions}
    dual_hist(logd_by_cond, np.linspace(-4, 1.5, 100),
              r"$\log_{10}(D)$ ($\mu m^2/s$)",
              r"$\log_{10}$ Diffusion Coefficient Distributions",
              str(results_dir / "fig3_logD_hist.png"))

    # ------------------------------------------------------------------
    # GMM on log10(D)
    # ------------------------------------------------------------------
    print("Fitting GMMs on log10(D)...")
    gmm_D = {}
    gmm_rows = []
    for cond in conditions:
        vals = logd_by_cond[cond]
        model, bics = gmm_bic_select(vals, max_k=args.max_gmm_k)
        k = model.n_components
        gmm_D[cond] = {"model": model, "means": model.means_.flatten(),
                        "stds": np.sqrt(model.covariances_.flatten()),
                        "weights": model.weights_, "best_k": k, "bics": bics}
        order = np.argsort(model.means_.flatten())
        interps = []
        for rank in range(len(order)):
            if rank == 0:
                interps.append("Immobile/bound")
            elif rank == len(order) - 1:
                interps.append("Freely diffusing")
            else:
                interps.append("Confined/substrate-engaged")
        for rank, i in enumerate(order):
            n_total = len(vals)
            gmm_rows.append({
                "Condition": cond, "Population": f"Pop {rank+1}",
                "Interpretation": interps[rank],
                "D_um2_s": round(10**model.means_[i, 0], 4),
                "log10_D_mean": round(model.means_[i, 0], 3),
                "log10_D_std": round(np.sqrt(model.covariances_[i, 0, 0]), 3),
                "Weight": round(model.weights_[i], 3),
                "Fraction_pct": round(model.weights_[i] * 100, 1),
                "N_approx": int(round(model.weights_[i] * n_total)),
            })
        print(f"  {cond}: K={k}, BICs={[f'{b:.0f}' for b in bics]}")
    pd.DataFrame(gmm_rows).to_csv(results_dir / "gmm_subpopulations.csv", index=False)

    # Figure 4: GMM panels
    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5))
    if len(conditions) == 1:
        axes = [axes]
    for ax, cond in zip(axes, conditions):
        vals = logd_by_cond[cond]
        ax.hist(vals, bins=100, density=True, alpha=0.4, color=cond_colors[cond], label="Data")
        x = np.linspace(-5, 2, 500)
        r = gmm_D[cond]
        total = np.zeros_like(x)
        order = np.argsort(r["means"])
        for rank, i in enumerate(order):
            comp = r["weights"][i] * stats.norm.pdf(x, r["means"][i], r["stds"][i])
            total += comp
            ax.plot(x, comp, "--", color=COMP_COLORS[rank % len(COMP_COLORS)],
                    label=f"Pop {rank+1}: D={10**r['means'][i]:.3f}, w={r['weights'][i]:.2f}")
        ax.plot(x, total, "k-", lw=2, label="GMM fit")
        ax.set_xlabel(r"$\log_{10}(D)$ ($\mu m^2/s$)")
        ax.set_ylabel("Density")
        ax.set_title(f"{cond} (K={r['best_k']})")
        ax.legend(fontsize=7, frameon=False)
        ax.set_xlim(-4, 1.5)
    fig.suptitle("GMM Decomposition of Diffusion Coefficients", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(results_dir / "fig4_gmm_D.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved fig4_gmm_D.png")

    # ------------------------------------------------------------------
    # GMM on log10(step) -- subsampled
    # ------------------------------------------------------------------
    if not step_all.empty and "step_um" in step_all.columns:
        print("Fitting GMMs on log10(step size)...")
        rng = np.random.default_rng(42)
        gmm_step = {}
        logstep_by_cond = {}
        for cond in conditions:
            full = np.log10(np.clip(s_by_cond[cond], 1e-6, None))
            logstep_by_cond[cond] = full
            sub = rng.choice(full, size=min(args.step_subsample, len(full)), replace=False)
            model, bics = gmm_bic_select(sub, max_k=args.max_gmm_k)
            gmm_step[cond] = {"model": model, "means": model.means_.flatten(),
                              "stds": np.sqrt(model.covariances_.flatten()),
                              "weights": model.weights_, "best_k": model.n_components, "bics": bics}
            print(f"  {cond} steps: K={model.n_components}")
        # Figure 5
        fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5))
        if len(conditions) == 1:
            axes = [axes]
        for ax, cond in zip(axes, conditions):
            vals = logstep_by_cond[cond]
            ax.hist(vals, bins=100, density=True, alpha=0.4, color=cond_colors[cond], label="Data")
            x = np.linspace(-4, 0.5, 500)
            r = gmm_step[cond]
            total = np.zeros_like(x)
            order = np.argsort(r["means"])
            for rank, i in enumerate(order):
                comp = r["weights"][i] * stats.norm.pdf(x, r["means"][i], r["stds"][i])
                total += comp
                ax.plot(x, comp, "--", color=COMP_COLORS[rank % len(COMP_COLORS)],
                        label=f"Pop {rank+1}: step={10**r['means'][i]:.3f}, w={r['weights'][i]:.2f}")
            ax.plot(x, total, "k-", lw=2, label="GMM fit")
            ax.set_xlabel(r"$\log_{10}$(Step Size) ($\mu m$)")
            ax.set_ylabel("Density")
            ax.set_title(f"{cond} (K={r['best_k']})")
            ax.legend(fontsize=7, frameon=False)
        fig.suptitle("GMM Decomposition of Step Sizes", fontsize=13)
        plt.tight_layout()
        fig.savefig(str(results_dir / "fig5_gmm_step.png"), bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  Saved fig5_gmm_step.png")

    # ------------------------------------------------------------------
    # Figure 6: Per-movie median D boxplot
    # ------------------------------------------------------------------
    print("Plotting per-movie median D comparison...")
    movie_medians = diff_all.groupby(["condition", "movie_name"])[d_col].median().reset_index()
    fig, ax = plt.subplots(figsize=(max(4, 2 * len(conditions)), 5))
    positions = list(range(1, len(conditions) + 1))
    bp_data = [movie_medians.loc[movie_medians["condition"] == c, d_col].values for c in conditions]
    bp = ax.boxplot(bp_data, positions=positions, widths=0.4, patch_artist=True, showfliers=False)
    for i, c in enumerate(conditions):
        bp["boxes"][i].set(facecolor=cond_colors[c] + "40", edgecolor=cond_colors[c])
        for el in ["whiskers", "caps", "medians"]:
            for item in bp[el][2*i:2*i+2]:
                item.set_color(cond_colors[c])
        jitter = 0.08
        rng_j = np.random.default_rng(i + 1)
        vals = bp_data[i]
        ax.scatter(positions[i] + rng_j.uniform(-jitter, jitter, len(vals)),
                   vals, color=cond_colors[c], alpha=0.6, s=20, zorder=5)
    # Stat test between first two conditions if applicable
    if len(conditions) >= 2:
        stat_res = stats.mannwhitneyu(bp_data[0], bp_data[1], alternative="two-sided")
        pval = stat_res.pvalue
        stars = "****" if pval < 1e-4 else "***" if pval < 1e-3 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        y_max = max(v.max() for v in bp_data if len(v) > 0) * 1.1
        ax.plot([1, 2], [y_max, y_max], "k-", lw=1)
        ax.text(1.5, y_max * 1.02, f"{stars}\np={pval:.2e}", ha="center", fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels(conditions)
    ax.set_ylabel(r"Median $D$ ($\mu m^2/s$)")
    ax.set_title("Per-Movie Median Diffusion Coefficient")
    plt.tight_layout()
    fig.savefig(str(results_dir / "fig6_boxplot_D.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved fig6_boxplot_D.png")

    # ------------------------------------------------------------------
    # Brightness analysis (figs 7-11) -- only if brightness data exists
    # ------------------------------------------------------------------
    if has_brightness:
        print("Running brightness analysis...")

        # Figure 7: Brightness GMM
        bright_gmm_rows = []
        fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5))
        if len(conditions) == 1:
            axes = [axes]
        for ax, cond in zip(axes, conditions):
            vals = diff_all.loc[diff_all["condition"] == cond, bright_col].dropna().values
            ax.hist(vals, bins=150, range=(vals.min(), np.percentile(vals, 99)),
                    density=True, alpha=0.4, color=cond_colors[cond], label="Data")
            gmm_b = GaussianMixture(5, random_state=42, n_init=5).fit(vals.reshape(-1, 1))
            x_b = np.linspace(vals.min(), np.percentile(vals, 99), 500)
            order_b = np.argsort(gmm_b.means_.flatten())
            base_mean = gmm_b.means_.flatten()[order_b[0]]
            total_b = np.zeros_like(x_b)
            for rank, i_b in enumerate(order_b):
                mu_b = gmm_b.means_[i_b, 0]
                sig_b = np.sqrt(gmm_b.covariances_[i_b, 0, 0])
                w_b = gmm_b.weights_[i_b]
                comp_b = w_b * stats.norm.pdf(x_b, mu_b, sig_b)
                total_b += comp_b
                ratio = mu_b / base_mean if base_mean > 0 else np.nan
                ax.plot(x_b, comp_b, "--", color=COMP_COLORS[rank % len(COMP_COLORS)],
                        label=f"Pop {rank+1}: {mu_b:.0f} ({ratio:.1f}x), {w_b*100:.0f}%")
                bright_gmm_rows.append({
                    "Condition": cond, "Population": f"Pop {rank+1}",
                    "Mean_brightness": round(mu_b, 1), "Std": round(sig_b, 1),
                    "Weight": round(w_b, 3), "Fraction_pct": round(w_b * 100, 1),
                    "Ratio_to_base": round(ratio, 2),
                })
            ax.plot(x_b, total_b, "k-", lw=2, label="GMM fit")
            ax.set_xlabel("Brightness (a.u.)")
            ax.set_ylabel("Density")
            ax.set_title(cond)
            ax.legend(fontsize=7, frameon=False)
        fig.suptitle("Per-Frame Brightness Distribution (GMM K=5)", fontsize=13)
        plt.tight_layout()
        fig.savefig(str(results_dir / "fig7_brightness_gmm.png"), bbox_inches="tight", dpi=300)
        plt.close(fig)
        pd.DataFrame(bright_gmm_rows).to_csv(results_dir / "gmm_brightness.csv", index=False)
        print(f"  Saved fig7_brightness_gmm.png")

        # Figure 8: 2D hexbin D vs brightness
        fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5))
        if len(conditions) == 1:
            axes = [axes]
        from matplotlib.colors import LogNorm as _LogNorm
        for ax, cond in zip(axes, conditions):
            sub = diff_all[diff_all["condition"] == cond]
            logd = np.log10(sub[d_col].clip(lower=1e-6))
            mass = sub[bright_col]
            hb = ax.hexbin(logd, mass, gridsize=60, cmap="inferno", norm=_LogNorm(),
                           mincnt=1)
            ax.set_xlabel(r"$\log_{10}(D)$")
            ax.set_ylabel("Brightness (a.u.)")
            ax.set_title(cond)
            plt.colorbar(hb, ax=ax, label="Count")
        fig.suptitle("Diffusion Coefficient vs Brightness", fontsize=13)
        plt.tight_layout()
        fig.savefig(str(results_dir / "fig8_hexbin_D_vs_brightness.png"), bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  Saved fig8_hexbin_D_vs_brightness.png")

        # Figure 9: Median D vs brightness decile
        fig, ax = plt.subplots(figsize=(7, 5))
        for cond in conditions:
            sub = diff_all[diff_all["condition"] == cond].copy()
            sub = sub.dropna(subset=[bright_col, d_col])
            if len(sub) < 20:
                continue
            sub["bq"] = pd.qcut(sub[bright_col], 10, labels=False, duplicates="drop")
            grouped = sub.groupby("bq").agg(
                bmed=(bright_col, "median"), dmed=(d_col, "median"),
                dq25=(d_col, lambda x: x.quantile(0.25)),
                dq75=(d_col, lambda x: x.quantile(0.75)),
            )
            ax.plot(grouped["bmed"], grouped["dmed"], "o-", color=cond_colors[cond], label=cond, lw=2)
            ax.fill_between(grouped["bmed"], grouped["dq25"], grouped["dq75"],
                            color=cond_colors[cond], alpha=0.15)
        ax.set_xlabel("Brightness (a.u.)")
        ax.set_ylabel(r"Median $D$ ($\mu m^2/s$)")
        ax.set_title("Diffusion Coefficient vs Brightness (Decile)")
        ax.legend(frameon=False)
        plt.tight_layout()
        fig.savefig(str(results_dir / "fig9_D_vs_brightness_decile.png"), bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  Saved fig9_D_vs_brightness_decile.png")

        # Pre-fit GMM for D subpopulations (reused by figures 10 and 11)
        _gmm_cache: Dict[str, Any] = {}
        for cond in conditions:
            sub = diff_all[diff_all["condition"] == cond].dropna(subset=[d_col, bright_col])
            logd = np.log10(sub[d_col].clip(lower=1e-6)).values.reshape(-1, 1)
            if len(logd) < 10:
                continue
            gmm3 = GaussianMixture(3, random_state=42, n_init=5).fit(logd)
            order = np.argsort(gmm3.means_.flatten())
            _gmm_cache[cond] = (gmm3, order)

        # Figure 10: Brightness by D subpopulation
        fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5))
        if len(conditions) == 1:
            axes = [axes]
        for ax, cond in zip(axes, conditions):
            if cond not in _gmm_cache:
                continue
            gmm3, order = _gmm_cache[cond]
            sub = diff_all[diff_all["condition"] == cond].dropna(subset=[d_col, bright_col]).copy()
            logd = np.log10(sub[d_col].clip(lower=1e-6)).values.reshape(-1, 1)
            labels = gmm3.predict(logd)
            pop_names = {order[0]: "Immobile", order[1]: "Confined", order[2]: "Free"}
            sub["d_pop"] = labels
            brange = (sub[bright_col].quantile(0.01), sub[bright_col].quantile(0.99))
            for i_o in order:
                vals = sub.loc[sub["d_pop"] == i_o, bright_col]
                ax.hist(vals, bins=80, range=brange, density=True, alpha=0.5,
                        color=COMP_COLORS[list(order).index(i_o) % len(COMP_COLORS)],
                        label=f"{pop_names[i_o]} (med={vals.median():.0f})")
            ax.set_xlabel("Brightness (a.u.)")
            ax.set_ylabel("Density")
            ax.set_title(cond)
            ax.legend(fontsize=9, frameon=False)
        fig.suptitle("Brightness by Diffusive Subpopulation", fontsize=13)
        plt.tight_layout()
        fig.savefig(str(results_dir / "fig10_brightness_by_dpop.png"), bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  Saved fig10_brightness_by_dpop.png")

        # Figure 11: Stacked bar - D subpop fraction by brightness quintile
        ct_rows = []
        fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5))
        if len(conditions) == 1:
            axes = [axes]
        for ax, cond in zip(axes, conditions):
            if cond not in _gmm_cache:
                continue
            gmm3, order = _gmm_cache[cond]
            sub = diff_all[diff_all["condition"] == cond].dropna(subset=[d_col, bright_col]).copy()
            logd = np.log10(sub[d_col].clip(lower=1e-6)).values.reshape(-1, 1)
            labels = gmm3.predict(logd)
            pop_names = {order[0]: "Immobile", order[1]: "Confined", order[2]: "Free"}
            # Use z-score quintiles if available, else raw brightness
            qcol = zscore_col if has_zscore else bright_col
            sub["d_pop"] = [pop_names[l] for l in labels]
            sub["bq"] = pd.qcut(sub[qcol], 5,
                                labels=["Q1\n(dim)", "Q2", "Q3", "Q4", "Q5\n(bright)"],
                                duplicates="drop")
            ct = pd.crosstab(sub["bq"], sub["d_pop"], normalize="index")
            for pop in ["Immobile", "Confined", "Free"]:
                if pop not in ct.columns:
                    ct[pop] = 0.0
            ct = ct[["Immobile", "Confined", "Free"]]
            ct.plot(kind="bar", stacked=True, ax=ax,
                    color=[COMP_COLORS[0], COMP_COLORS[1], COMP_COLORS[2]])
            ax.set_ylabel("Fraction")
            ax.set_xlabel("Brightness Quintile" + (" (z-score)" if has_zscore else ""))
            ax.set_title(cond)
            ax.legend(fontsize=8, frameon=False)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            for q_label in ct.index:
                q_str = str(q_label).replace("\n", " ")
                for pop in ["Immobile", "Confined", "Free"]:
                    ct_rows.append({"Condition": cond, "Quintile": q_str,
                                    "D_subpopulation": pop,
                                    "Fraction_pct": round(ct.loc[q_label, pop] * 100, 1)})
        fig.suptitle("D Subpopulation Fraction by Brightness Quintile", fontsize=13)
        plt.tight_layout()
        fig.savefig(str(results_dir / "fig11_dpop_by_brightness_quintile.png"), bbox_inches="tight", dpi=300)
        plt.close(fig)
        pd.DataFrame(ct_rows).to_csv(results_dir / "brightness_x_diffusion.csv", index=False)
        print(f"  Saved fig11_dpop_by_brightness_quintile.png")

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------
    print("Running statistical tests...")
    test_rows = []
    if len(conditions) >= 2:
        c1, c2 = conditions[0], conditions[1]
        # Per-track D
        u, p = stats.mannwhitneyu(d_by_cond[c1], d_by_cond[c2], alternative="two-sided")
        test_rows.append({"Test": f"Mann-Whitney U (per-track D, {c1} vs {c2})",
                          "Statistic": f"{u:.4e}", "p_value": f"{p:.4e}",
                          "Note": f"{c1} n={len(d_by_cond[c1]):,}, {c2} n={len(d_by_cond[c2]):,}"})
        # KS
        ks, pk = stats.ks_2samp(d_by_cond[c1], d_by_cond[c2])
        test_rows.append({"Test": f"KS test (D, {c1} vs {c2})",
                          "Statistic": f"{ks:.4f}", "p_value": f"{pk:.4e}",
                          "Note": "Two-sample Kolmogorov-Smirnov"})
        # Per-movie
        m1 = movie_medians.loc[movie_medians["condition"] == c1, d_col].values
        m2 = movie_medians.loc[movie_medians["condition"] == c2, d_col].values
        if len(m1) > 0 and len(m2) > 0:
            um, pm = stats.mannwhitneyu(m1, m2, alternative="two-sided")
            test_rows.append({"Test": f"Mann-Whitney U (per-movie median D, {c1} vs {c2})",
                              "Statistic": f"{um:.0f}", "p_value": f"{pm:.4e}",
                              "Note": f"{c1}: {len(m1)} movies (med={np.median(m1):.4f}), "
                                      f"{c2}: {len(m2)} movies (med={np.median(m2):.4f})"})
        # Step sizes
        if not step_all.empty and "step_um" in step_all.columns:
            rng_s = np.random.default_rng(42)
            s1 = rng_s.choice(s_by_cond[c1], min(50000, len(s_by_cond[c1])), replace=False)
            s2 = rng_s.choice(s_by_cond[c2], min(50000, len(s_by_cond[c2])), replace=False)
            us, ps = stats.mannwhitneyu(s1, s2, alternative="two-sided")
            test_rows.append({"Test": f"Mann-Whitney U (steps, {c1} vs {c2}, subsampled 50k)",
                              "Statistic": f"{us:.4e}", "p_value": f"{ps:.4e}", "Note": "Subsampled"})
        # Brightness correlations
        if has_brightness:
            for cond in conditions:
                sub = diff_all[diff_all["condition"] == cond].dropna(subset=[bright_col, d_col])
                rho, pr = stats.spearmanr(sub[bright_col], sub[d_col])
                test_rows.append({"Test": f"Spearman (brightness vs D, {cond})",
                                  "Statistic": f"{rho:.4f}", "p_value": f"{pr:.4e}",
                                  "Note": f"Using {bright_col}"})
                logm = np.log10(sub[bright_col].clip(lower=1e-6))
                logd_v = np.log10(sub[d_col].clip(lower=1e-6))
                slope = np.polyfit(logm, logd_v, 1)[0]
                test_rows.append({"Test": f"Log-log slope (brightness vs D, {cond})",
                                  "Statistic": f"{slope:.3f}", "p_value": "N/A",
                                  "Note": "Stokes-Einstein predicts -0.33"})
    tests_df = pd.DataFrame(test_rows)
    tests_df.to_csv(results_dir / "statistical_tests.csv", index=False)
    print(f"Statistical tests:\n{tests_df.to_string(index=False)}")

    print(f"\nAll outputs saved to: {results_dir}")


# ============================================================================
# PLOT SUBCOMMAND - Plotting Operations  
# ============================================================================

def cmd_utils_split_metrics(args):
    """Split metrics by condition - from split_metrics_by_condition.py"""
    DEFAULT_PARAMS = ["D_fit", "Alpha", "r2_fit", "vacf_lag1", "confinement_idx", "hull_area_um2", "tortuosity", "Frames"]
    ALIASES = {
        "frames": ["n_frames","frames","num_frames"],
        "alpha": ["alpha","alpha_fit","alpha_hat"],
        "d_fit": ["d_fit","d","d_um2_per_s","diffusion"],
        "r2_fit": ["r2_fit","r2","r_squared"],
        "confinement_idx": ["confinement_idx","confinement_index"],
        "hull_area_um2": ["hull_area_um2","hull_area"],
        "tortuosity": ["tortuosity","tort"],
        "vacf_lag1": ["vacf_lag1","vacf_lag_1"],
    }
    
    def infer_condition_column(df: pd.DataFrame, override: Optional[str] = None) -> str:
        if override:
            if override in df.columns:
                return override
            raise ValueError(f"--cond-col '{override}' not found in columns: {list(df.columns)}")
        for c in df.columns:
            if "condition" in c.lower():
                return c
        raise ValueError("Could not infer the condition column. Use --cond-col to specify it.")
    
    def find_actual_col(requested: str, columns: List[str]) -> Optional[str]:
        req_lower = requested.lower()
        for c in columns:
            if c.lower() == req_lower:
                return c
        if req_lower in ALIASES:
            for alias in ALIASES[req_lower]:
                for c in columns:
                    if c.lower() == alias.lower():
                        return c
        for c in columns:
            if req_lower in c.lower():
                return c
        return None
    
    def build_out_table(df: pd.DataFrame, cond_col: str, value_col: str) -> pd.DataFrame:
        series = df[[cond_col, value_col]].dropna()
        grouped = series.groupby(cond_col)[value_col].apply(list)
        max_len = max((len(v) for v in grouped.values), default=0)
        data = {str(cond): (vals + [np.nan]*(max_len - len(vals))) for cond, vals in grouped.items()}
        out_df = pd.DataFrame(data)
        cols = list(out_df.columns)
        if any(c.lower() == "vector" for c in cols):
            vector = [c for c in cols if c.lower() == "vector"][0]
            others = sorted([c for c in cols if c != vector], key=lambda x: x.lower())
            out_df = out_df[[vector] + others]
        else:
            out_df = out_df[sorted(cols, key=lambda x: x.lower())]
        return out_df
    
    df = pd.read_csv(args.input_csv)
    cond = infer_condition_column(df, args.cond_col)
    df[cond] = df[cond].astype(str)
    
    columns = df.columns.tolist()
    params = args.params if args.params else DEFAULT_PARAMS
    actual_map = {p: find_actual_col(p, columns) for p in params}
    
    out_dir = Path(args.input_csv).parent / "split metrics by condition"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    for req, actual in actual_map.items():
        if actual is None:
            results[req] = "(not created)"
            continue
        if not pd.api.types.is_numeric_dtype(df[actual]):
            df[actual] = pd.to_numeric(df[actual], errors="coerce")
        if df[actual].notna().sum() == 0:
            results[req] = "(not created)"
            continue
        out_df = build_out_table(df, cond, actual)
        out_path = out_dir / f"{req.replace('/', '-').replace(' ', '_')}.csv"
        out_df.to_csv(out_path, index=False)
        results[req] = str(out_path)
    
    print("Saved files:")
    for k, v in results.items():
        print(f"  {k}: {v}")

# ============================================================================
# DRAW SUBCOMMAND - Movie rendering with track overlays
# ============================================================================

def cmd_draw_movie_tracks(args):
    """Render track-overlay movies as TIFF stacks.

    For each movie that has a *_tracks.csv from a previous ``track trackpy``
    run, composite coloured trajectories onto an inverted-greyscale copy of
    the original frames and write out an RGB TIFF stack.
    """
    from PIL import Image, ImageDraw

    if not HAVE_ND2 and not HAVE_TIFFFILE:
        raise SystemExit("nd2 or tifffile required to read source movies")
    if not HAVE_TIFFFILE:
        raise SystemExit("tifffile required to write TIFF stacks")

    input_dir = Path(args.input_dir)
    output_dir = input_dir / "track movies"
    output_dir.mkdir(parents=True, exist_ok=True)
    tail_length = int(args.tail_length)
    line_width = int(args.line_width)

    # Easy-to-tune rendering controls for movie-tracks output.
    movie_brightness_scale = 0.88
    track_alpha_max = 255
    track_alpha_min_alive = 90

    # ---- locate *_tracks.csv files produced by ``track trackpy`` ----------
    track_csvs: List[Path] = sorted(input_dir.rglob("*_tracks.csv"))
    if not track_csvs:
        raise SystemExit(f"No *_tracks.csv files found under {input_dir}")

    # ---- locate source movies (ND2 / TIF) --------------------------------
    source_movies: Dict[str, Path] = {}
    for p in _discover_movie_inputs(input_dir):
        source_movies[p.stem] = p

    # ---- assign deterministic per-track gradients for clear separation -----
    def _track_gradient_colors(track_key: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        seed = sum((i + 1) * ord(ch) for i, ch in enumerate(track_key)) % (2**32)
        rng = np.random.default_rng(seed)
        hue = float(rng.uniform(0.0, 1.0))
        sat = float(rng.uniform(0.7, 0.95))
        dark_val = float(rng.uniform(0.45, 0.6))
        light_val = float(rng.uniform(0.85, 1.0))
        dark_rgb = colorsys.hsv_to_rgb(hue, sat, dark_val)
        light_rgb = colorsys.hsv_to_rgb(hue, sat * 0.75, light_val)
        dark = tuple(int(255 * c) for c in dark_rgb)
        light = tuple(int(255 * c) for c in light_rgb)
        return dark, light

    def _gradient_color(
        start_col: Tuple[int, int, int],
        end_col: Tuple[int, int, int],
        frac: float,
    ) -> Tuple[int, int, int]:
        frac = max(0.0, min(1.0, float(frac)))
        return tuple(
            int(round((1.0 - frac) * start_col[i] + frac * end_col[i]))
            for i in range(3)
        )

    channel = int(getattr(args, "channel", 0) or 0)

    def _load_frames(path: Path) -> List[np.ndarray]:
        ext = path.suffix.lower()
        if ext == ".nd2":
            with nd2.ND2File(path) as f:
                data = f.asarray()
            data = _select_channel(data, channel)
            frames = [data[i].astype(np.float32) for i in range(data.shape[0])]
            del data
            return frames
        else:
            raw = tifffile.imread(str(path))
            if raw.ndim == 2:
                return [raw.astype(np.float32)]
            if raw.ndim >= 4:
                raw = _select_channel(raw, channel)
            return [raw[i].astype(np.float32) for i in range(raw.shape[0])]

    for track_csv in track_csvs:
        movie_name = track_csv.stem.replace("_tracks", "")
        if movie_name not in source_movies:
            print(f"[draw] Skipping {movie_name}: source movie not found")
            continue

        print(f"[draw] Rendering {movie_name} ...")
        tracks_df = pd.read_csv(track_csv, low_memory=False)
        for col in ("x", "y", "frame", "particle"):
            if col not in tracks_df.columns:
                print(f"[draw] Skipping {movie_name}: missing column '{col}'")
                break
        else:
            pass  # all columns present, proceed
        if any(col not in tracks_df.columns for col in ("x", "y", "frame", "particle")):
            continue

        tracks_df = tracks_df.sort_values(["particle", "frame"])
        # pre-index: for each particle, store sorted arrays of (frame, x, y)
        track_data: Dict[Any, Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int], Tuple[int, int, int]]] = {}
        frames = _load_frames(source_movies[movie_name])
        n_frames_total = len(frames)
        n_frames_render = min(n_frames_total, 100)
        for pid, grp in tracks_df.groupby("particle"):
            g = grp.sort_values("frame")
            dark_col, light_col = _track_gradient_colors(f"{movie_name}:{pid}")
            track_data[pid] = (
                g["frame"].values.astype(int),
                g["x"].values.astype(float),
                g["y"].values.astype(float),
                dark_col,
                light_col,
            )

        # global intensity range for consistent contrast across frames
        all_px = np.concatenate([f.ravel() for f in frames[:min(10, n_frames_total)]])
        vmin, vmax = float(np.percentile(all_px, 0.5)), float(np.percentile(all_px, 99.5))
        del all_px

        rendered: List[np.ndarray] = []
        for t in range(n_frames_render):
            # invert greyscale: clip, normalise 0-1, invert so particles are dark
            f = frames[t]
            if f.ndim > 2:
                f = f[0]
            normed = np.clip((f - vmin) / (vmax - vmin + 1e-8), 0, 1)
            inv = (1.0 - normed)
            base_uint8 = np.clip(inv * 255 * movie_brightness_scale, 0, 255).astype(np.uint8)

            # RGB base
            base_rgb = np.stack([base_uint8, base_uint8, base_uint8], axis=-1)

            # draw tracks using PIL for proper alpha compositing
            base_img = Image.fromarray(base_rgb, mode="RGB").convert("RGBA")
            overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            for pid, (tframes, txs, tys, dark_col, light_col) in track_data.items():
                first_f, last_f = int(tframes[0]), int(tframes[-1])
                if t < first_f:
                    continue

                # mask: points up to current frame, optionally within tail window
                if tail_length > 0:
                    window_start = max(first_f, t - tail_length)
                else:
                    window_start = first_f

                mask = (tframes >= window_start) & (tframes <= t)
                if mask.sum() < 2:
                    continue

                seg_frames = tframes[mask]
                seg_xs = txs[mask]
                seg_ys = tys[mask]
                track_span = max(1, last_f - first_f)

                for i in range(1, len(seg_frames)):
                    seg_age = t - int(seg_frames[i])
                    seg_progress = (float(seg_frames[i]) - float(first_f)) / float(track_span)
                    # fade after track ends
                    fade_after = t - last_f if t > last_f else 0
                    if tail_length > 0 and fade_after > tail_length:
                        break

                    # alpha: full while track alive, fading tail when over
                    if t <= last_f:
                        if tail_length > 0:
                            frac = 1.0 - (seg_age / tail_length)
                            alpha = max(track_alpha_min_alive, int(track_alpha_max * frac))
                        else:
                            alpha = track_alpha_max
                    else:
                        base_alpha = track_alpha_max
                        if tail_length > 0:
                            frac_seg = 1.0 - (seg_age / tail_length) if tail_length > 0 else 1.0
                            frac_end = 1.0 - (fade_after / tail_length)
                            base_alpha = max(0, int(track_alpha_max * min(frac_seg, frac_end)))
                        alpha = base_alpha

                    if alpha <= 0:
                        continue

                    col = _gradient_color(dark_col, light_col, seg_progress)

                    draw.line(
                        [(float(seg_xs[i - 1]), float(seg_ys[i - 1])),
                         (float(seg_xs[i]), float(seg_ys[i]))],
                        fill=col + (alpha,),
                        width=line_width,
                    )

            composited = Image.alpha_composite(base_img, overlay).convert("RGB")
            rendered.append(np.array(composited))

            if (t + 1) % 50 == 0 or t == n_frames_render - 1:
                print(f"  frame {t + 1}/{n_frames_render}")

        stack = np.array(rendered, dtype=np.uint8)
        out_path = output_dir / f"{movie_name}_track_movie.tif"
        tifffile.imwrite(str(out_path), stack, photometric="rgb")
        print(
            f"  -> {out_path}  ({stack.shape[0]} frames, {stack.shape[1]}x{stack.shape[2]} RGB"
            f"; source had {n_frames_total} frames)"
        )

    print(f"\n[draw] All track movies saved to: {output_dir}")


# ============================================================================
# MAIN ARGUMENT PARSER
# ============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SPT Unified Pipeline - Consolidated Single-Particle Tracking analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare TrackMate data
  spt-pipeline prep trackmate /path/to/data --out-dir raw

  # Convert ND2 to TIFF
  spt-pipeline prep nd2-to-tiff /path/to/nd2/files

  # Track with trackpy
  spt-pipeline track trackpy /path/to/images --diam 11 --minmass 70

  # Compare conditions (with optional loc-error subtraction and filters)
  spt-pipeline analyze compare-conditions /path/to/pooled \\
      --conditions NT Tg30 --min-nspots 20 --subtract-loc-error

  # Split metrics by condition
  spt-pipeline utils split-metrics metrics.csv --params D_fit Alpha
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Subcommands')
    
    # PREP subcommands
    prep_parser = subparsers.add_parser('prep', help='Data preparation')
    prep_sub = prep_parser.add_subparsers(dest='prep_cmd')
    
    prep_trackmate = prep_sub.add_parser('trackmate', help='TrackMate preprocessing')
    prep_trackmate.add_argument('root_dir', help='Root directory to scan')
    prep_trackmate.add_argument('--out-dir', help='Output directory (default: <root>/raw)')
    prep_trackmate.add_argument('--include-date', action='store_true', help='Include date in filename')
    prep_trackmate.add_argument('--date-format', choices=['YYMMDD','YYYYMMDD'], default='YYMMDD')
    
    prep_nd2 = prep_sub.add_parser('nd2-to-tiff', help='ND2 to TIFF conversion')
    prep_nd2.add_argument('directory', help='Directory with ND2 files')
    prep_nd2.add_argument('--channel', type=int, default=0,
        help='0-indexed channel to extract from multichannel movies (default: 0)')
    
    prep_stack = prep_sub.add_parser('stack-channels', help='Stack two-channel images')
    prep_stack.add_argument('spool_dir', help='Spool directory with lefts/rights')
    prep_stack.add_argument('output_dir', help='Output directory')
    
    prep_find = prep_sub.add_parser('find-ome-tif', help='Find OME-TIFF files')
    prep_find.add_argument('input_folder', help='Input folder to search')
    prep_find.add_argument('--output-dir', default=os.path.expanduser('~/Desktop'), help='Output directory')
    
    # TRACK subcommands
    track_parser = subparsers.add_parser('track', help='Tracking operations')
    track_sub = track_parser.add_subparsers(dest='track_cmd')
    
    
    track_trackpy = track_sub.add_parser('trackpy', help='Trackpy-based tracking')
    track_trackpy.add_argument('input_directory', help='Input directory')
    track_trackpy.add_argument('--notrack', action='store_true', help='Skip tracking')
    track_trackpy.add_argument('--mpp', type=float, default=0.122, help='Microns per pixel')
    track_trackpy.add_argument('--fps', type=float, default=50, help='Frames per second')
    track_trackpy.add_argument('--diam', type=int, default=7, help='Particle diameter')
    track_trackpy.add_argument('--range', type=int, default=10, help='Search range')
    track_trackpy.add_argument('--memory', type=int, default=0, help='Memory (frames)')
    track_trackpy.add_argument('--minmass', type=float, default=25, help='Minimum mass')
    track_trackpy.add_argument('--percentile', type=float, default=90, help='Percentile threshold')
    track_trackpy.add_argument(
        '--diffusion_tau', type=int, default=5,
        help='Fixed lag (in frames) used for same-tau diffusion estimates (default: 5 frames)'
    )
    track_trackpy.add_argument(
        '--min-separation', '--min_separation', type=int, default=3,
        help='Minimum separation distance between detected particles in pixels (default: 3)'
    )
    track_trackpy.add_argument(
        '--min-track-length', type=int, default=4,
        help='Minimum track length in frames; shorter tracks are discarded (default: 4)'
    )
    track_trackpy.add_argument(
        '--channel', type=int, default=0,
        help='0-indexed channel to analyze for multichannel ND2/TIF movies (default: 0)'
    )
    track_trackpy.add_argument(
        '-o', '--out-dir', '--output-dir', dest='out_dir', default=None,
        help='Output directory for per-movie folders, condition summaries, '
             'groupnames.csv, and the saved pipeline context. '
             'Defaults to input_directory (backward-compatible).'
    )
    _add_bool_flag(track_trackpy, '--correct-brightness', 'correct_brightness', True,
                   'Apply flatfield brightness correction based on 2D polynomial illumination fit')
    
    # ANALYZE subcommands
    analyze_parser = subparsers.add_parser('analyze', help='Analysis operations')
    analyze_sub = analyze_parser.add_subparsers(dest='analyze_cmd')
    

    analyze_cc = analyze_sub.add_parser('compare-conditions',
        help='Compare D, step sizes, and brightness across conditions')
    analyze_cc.add_argument('input_dir', help='Root directory with per-movie subfolders and groupnames.csv')
    analyze_cc.add_argument('--conditions', nargs='*', default=None,
        help='Condition tokens to match in movie names (fallback if no groupnames.csv)')
    analyze_cc.add_argument('--max-gmm-k', type=int, default=5,
        help='Max GMM components to test (default: 5)')
    analyze_cc.add_argument('--step-subsample', type=int, default=100000,
        help='Subsample size for step-size GMM (default: 100000)')
    analyze_cc.add_argument('--d-xlim', type=float, default=2.0,
        help='X-axis limit for linear D histograms (default: 2.0)')
    analyze_cc.add_argument('--step-xlim', type=float, default=0.5,
        help='X-axis limit for step histograms (default: 0.5)')
    analyze_cc.add_argument('--min-nspots', type=int, default=None,
        help='Minimum track length (n_spots) to include (e.g. 20)')
    analyze_cc.add_argument('--min-alpha', type=float, default=None,
        help='Minimum anomalous exponent to include (e.g. 0.3)')
    analyze_cc.add_argument('--subtract-loc-error', action='store_true', default=False,
        help='Subtract per-condition localization error floor from D values')
    analyze_cc.add_argument('--loc-error-alpha-max', type=float, default=0.2,
        help='Max alpha for immobile proxy used in loc error estimation (default: 0.2)')
    analyze_cc.add_argument('--loc-error-min-nspots', type=int, default=6,
        help='Min n_spots for immobile proxy in loc error estimation (default: 6)')
    analyze_cc.add_argument('--mpp-rescale', nargs=2, action='append', metavar=('PATTERN', 'FACTOR'),
        help='Rescale D_fit_um2_s for movies matching PATTERN by FACTOR. '
             'E.g. --mpp-rescale NT_0 2.25 --mpp-rescale Tg30_0 2.25 '
             'to correct for pixel size mismatch. Can be specified multiple times.')

    # DRAW subcommands
    draw_parser = subparsers.add_parser('draw', help='Rendering / movie generation')
    draw_sub = draw_parser.add_subparsers(dest='draw_cmd')

    draw_mt = draw_sub.add_parser('movie-tracks',
        help='Render TIFF stacks of movies with coloured track overlays')
    draw_mt.add_argument('input_dir',
        help='Root directory from a previous "track trackpy" run (contains *_tracks.csv)')
    draw_mt.add_argument('--tail-length', type=int, default=30,
        help='Number of past frames to show per track tail; 0 = full history (default: 30)')
    draw_mt.add_argument('--line-width', type=int, default=2,
        help='Track line width in pixels (default: 2)')
    draw_mt.add_argument('--channel', type=int, default=0,
        help='0-indexed channel to render as the background for multichannel movies (default: 0)')


    # NOTE: the brightness-mobility / steps-brightness / dwell arm, and the
    # 'workflow' chain that ran them, were retired. Their code is preserved
    # under retired/ at the repository root. 'workflow auto' now names the v2
    # core path and is handled in cli.py.

    # UTILS subcommands
    utils_parser = subparsers.add_parser('utils', help='Utility operations')
    utils_sub = utils_parser.add_subparsers(dest='utils_cmd')
    
    utils_split = utils_sub.add_parser('split-metrics', help='Split metrics by condition')
    utils_split.add_argument('input_csv', type=Path, help='Input CSV')
    utils_split.add_argument('--cond-col', help='Condition column name')
    utils_split.add_argument('--params', nargs='+', help='Parameters to export')
    
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return
    
    # Dispatch to appropriate command
    if args.command == 'prep':
        if args.prep_cmd == 'trackmate':
            cmd_prep_trackmate(args)
        elif args.prep_cmd == 'nd2-to-tiff':
            cmd_prep_nd2_to_tiff(args)
        elif args.prep_cmd == 'stack-channels':
            cmd_prep_stack_channels(args)
        elif args.prep_cmd == 'find-ome-tif':
            cmd_prep_find_ome_tif(args)
        else:
            prep_parser.print_help()
    elif args.command == 'track':
        if args.track_cmd == 'trackpy':
            cmd_track_trackpy(args)
        else:
            track_parser.print_help()
    elif args.command == 'analyze':
        if args.analyze_cmd == 'compare-conditions':
            cmd_analyze_compare_conditions(args)
        else:
            analyze_parser.print_help()
    elif args.command == 'draw':
        if args.draw_cmd == 'movie-tracks':
            cmd_draw_movie_tracks(args)
        else:
            draw_parser.print_help()
    elif args.command == 'utils':
        if args.utils_cmd == 'split-metrics':
            cmd_utils_split_metrics(args)
        else:
            utils_parser.print_help()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

