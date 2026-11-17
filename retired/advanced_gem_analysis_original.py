"""
Advanced GEM Analysis Library
=============================
Consolidated analyses beyond SPT-pipeline scope:

  * VACF   -- velocity autocorrelation (per-track & ensemble, multi-delta)
  * CFBM   -- fractional-Brownian-motion fit to collapsed VACF  (alpha', H)
  * Angles -- turning-angle distributions, <cos(theta)> vs tau, KS tests
  * EB     -- ergodicity-breaking parameter  Var(MSD) / <MSD>^2
  * NGP    -- non-Gaussian parameter (1-D, combined x & y displacements)
  * G'/G'' -- complex shear modulus via Mason (generalized Stokes-Einstein)

Input: GEMspa trajectory CSVs  (Trajectory, Frame, x, y)
       or SPT-pipeline per-particle CSVs  (auto-detected).
All spatial data converted to physical units (microns) on load.

Dependencies: numpy, pandas, matplotlib, scipy
"""

from __future__ import annotations

import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette (global default)
_CB_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=_CB_PALETTE)

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import gamma as gamma_func
from scipy.stats import kurtosis, ks_2samp

# ======================================================================
#  CONSTANTS  &  DEFAULTS
# ======================================================================
K_B = 1.380649e-23  # Boltzmann constant  [J/K]
N_JOBS = min(8, os.cpu_count() or 1)


# ======================================================================
#  TRACK  INDEXING  (O(1) per-track lookups)
# ======================================================================

def _build_track_index(
    track_data: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]], np.ndarray]:
    """Sort track_data by track_id and build an O(1) lookup index.

    Returns
    -------
    sorted_data : ndarray, same shape as *track_data*, sorted by column 0
    index : {int(track_id): (start_row, end_row)} into *sorted_data*
    track_lengths : (M, 2) array  [track_id, length]
    """
    order = np.argsort(track_data[:, 0], kind="mergesort")
    sd = track_data[order]
    ids, counts = np.unique(sd[:, 0], return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts
    idx = {int(t): (int(s), int(e)) for t, s, e in zip(ids, starts, ends)}
    tl = np.column_stack([ids, counts])
    return sd, idx, tl


def _get_track(
    sorted_data: np.ndarray,
    track_index: Dict[int, Tuple[int, int]],
    tid: int,
) -> np.ndarray:
    """Return a *view* (no copy) of one track from pre-indexed data."""
    s, e = track_index[tid]
    return sorted_data[s:e]


# ======================================================================
#  I/O  &  UTILITIES
# ======================================================================

def load_track_data(
    filepath: Union[str, Path],
    fmt: str = "auto",
    micron_per_px: float = 1.0,
    time_step: float = 1.0,
) -> Tuple[np.ndarray, float, float]:
    """Load trajectory data and return (track_array, micron_per_px, time_step).

    Parameters
    ----------
    filepath : path to CSV / TSV
    fmt : 'gemspa', 'spt', or 'auto'
    micron_per_px : spatial calibration  (applied only for gemspa raw-pixel data)
    time_step : frame interval in seconds

    Returns
    -------
    track_data : ndarray, shape (N, 4) -- columns [track_id, frame, x_um, y_um]
    micron_per_px : echoed back (useful when auto-detected)
    time_step : echoed back
    """
    filepath = Path(filepath)
    sep = "\t" if filepath.suffix in (".txt", ".tsv") else ","
    df = pd.read_csv(filepath, sep=sep)

    # --- auto-detect format ---
    cols_lower = {c.lower(): c for c in df.columns}
    if fmt == "auto":
        if "trajectory" in cols_lower:
            fmt = "gemspa"
        elif "particle" in cols_lower:
            fmt = "spt"
        else:
            raise ValueError(
                f"Cannot auto-detect format.  Columns: {list(df.columns)}"
            )

    if fmt == "gemspa":
        id_col = cols_lower.get("trajectory", "Trajectory")
        fr_col = cols_lower.get("frame", "Frame")
        x_col = cols_lower.get("x", "x")
        y_col = cols_lower.get("y", "y")
        track_data = df[[id_col, fr_col, x_col, y_col]].to_numpy(dtype=float)
        track_data[:, 2] *= micron_per_px
        track_data[:, 3] *= micron_per_px

    elif fmt == "spt":
        # SPT-pipeline linked CSV: particle, frame, x, y  (already in um)
        id_col = cols_lower.get("particle", "particle")
        fr_col = cols_lower.get("frame", "frame")
        x_col = cols_lower.get("x", "x")
        y_col = cols_lower.get("y", "y")
        track_data = df[[id_col, fr_col, x_col, y_col]].to_numpy(dtype=float)
        # positions assumed already in microns
    else:
        raise ValueError(f"Unknown format: {fmt}")

    return track_data, micron_per_px, time_step


def load_multiple_files(
    file_list: Sequence[Union[str, Path]],
    **load_kwargs,
) -> Tuple[np.ndarray, float, float]:
    """Load and concatenate multiple trajectory files, re-indexing track IDs."""
    all_tracks = []
    mpp = ts = None
    next_id = 0
    for fp in file_list:
        td, mpp, ts = load_track_data(fp, **load_kwargs)
        td[:, 0] += next_id
        next_id = td[:, 0].max() + 1
        all_tracks.append(td)
    return np.vstack(all_tracks), mpp, ts


def compute_track_lengths(track_data: np.ndarray) -> np.ndarray:
    """Return array (N_tracks, 2) with columns [track_id, length]."""
    ids, counts = np.unique(track_data[:, 0], return_counts=True)
    return np.column_stack([ids, counts])


# ======================================================================
#  VACF  (Velocity Autocorrelation Function)
# ======================================================================

def vacf_per_track(
    track_data: np.ndarray,
    track_lengths: np.ndarray,
    min_len: int = 3,
    delta: int = 1,
    max_tlag: Optional[int] = None,
    _track_index: Optional[Dict[int, Tuple[int, int]]] = None,
) -> np.ndarray:
    """Compute the normalized VACF for every track at a given delta.

    Parameters
    ----------
    track_data : (N, 4) array  [id, frame, x, y]  in physical units.
        When *_track_index* is supplied this must be the **sorted** array
        returned by :func:`_build_track_index`.
    track_lengths : (M, 2) array  [id, length]
    min_len : minimum track length to include
    delta : displacement lag (number of frames between positions for velocity)
    max_tlag : maximum tau in frames to store (default None = keep all).
        Only autocorrelation lags with tau <= *max_tlag* are retained.
    _track_index : pre-built index from :func:`_build_track_index` (O(1) lookups)

    Returns
    -------
    vacf_array : (K, 7) array with columns
        0 = track_id
        1 = tau  (in frames, multiples of delta)
        2 = raw autocorrelation x
        3 = raw autocorrelation y
        4 = normalized autocorrelation x  (divided by zero-lag)
        5 = normalized autocorrelation y
        6 = number of velocity values for this track
    """
    min_len = max(min_len, delta + 2)  # need >= delta+2 points for >= 2 velocities

    filt = track_lengths[track_lengths[:, 1] >= min_len]
    ids = filt[:, 0].astype(int)
    if len(ids) == 0:
        return np.empty((0, 7))

    # pre-allocate (over-estimate)
    max_rows = int(np.sum(filt[:, 1]) - len(ids) * delta)
    vacf = np.zeros((max_rows, 7))

    ptr = 0
    for tid in ids:
        trk = (_get_track(track_data, _track_index, tid)
               if _track_index is not None
               else track_data[track_data[:, 0] == tid])
        vx = (trk[delta:, 2] - trk[:-delta, 2]) / delta
        vy = (trk[delta:, 3] - trk[:-delta, 3]) / delta
        n_v = len(vx)
        if n_v < 2:
            continue

        cx = np.correlate(vx, vx, mode="full")
        cy = np.correlate(vy, vy, mode="full")
        half = cx.size // 2  # zero-lag index

        # Truncate to max_tlag if requested
        length = n_v
        if max_tlag is not None:
            length = min(length, max_tlag // delta + 1)
        vacf[ptr:ptr + length, 0] = tid
        vacf[ptr:ptr + length, 1] = delta * np.arange(length)
        vacf[ptr:ptr + length, 2] = cx[half:half + length]
        vacf[ptr:ptr + length, 3] = cy[half:half + length]
        vacf[ptr:ptr + length, 4] = cx[half:half + length] / cx[half] if cx[half] != 0 else 0
        vacf[ptr:ptr + length, 5] = cy[half:half + length] / cy[half] if cy[half] != 0 else 0
        vacf[ptr:ptr + length, 6] = n_v
        ptr += length

    return vacf[:ptr]


def vacf_ensemble_average(
    per_track_vacf: np.ndarray,
    track_lengths: np.ndarray,
    min_len: int = 3,
    delta: int = 1,
) -> np.ndarray:
    """Ensemble-average the normalized VACF over all qualifying tracks.

    Returns
    -------
    result : (T, 7) array with columns
        0 = tau (frames)
        1 = mean normalized VACF_x
        2 = std  VACF_x
        3 = SEM  VACF_x
        4 = mean normalized VACF_y
        5 = std  VACF_y
        6 = SEM  VACF_y
    """
    min_len = max(min_len, delta + 2)
    valid = per_track_vacf[per_track_vacf[:, 6] >= (min_len - delta)]
    if len(valid) == 0:
        return np.empty((0, 7))

    max_tau = int(np.max(valid[:, 1]))
    result = []
    for tau in range(0, max_tau + delta, delta):
        cx = valid[valid[:, 1] == tau][:, 4]
        cy = valid[valid[:, 1] == tau][:, 5]
        if len(cx) == 0:
            continue
        result.append([
            tau,
            np.nanmean(cx), np.nanstd(cx), np.nanstd(cx) / np.sqrt(len(cx)),
            np.nanmean(cy), np.nanstd(cy), np.nanstd(cy) / np.sqrt(len(cy)),
        ])
    return np.asarray(result) if result else np.empty((0, 7))


def vacf_multi_delta(
    track_data: np.ndarray,
    track_lengths: np.ndarray,
    min_len: int = 11,
    deltas: Sequence[int] = range(1, 11),
    max_tlag: Optional[int] = None,
    n_jobs: int = N_JOBS,
    _track_index: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Compute per-track and ensemble VACF for multiple deltas.

    Each delta is independent; when *n_jobs* > 1 they execute in
    parallel using threads (numpy releases the GIL for array ops).

    Parameters
    ----------
    max_tlag : maximum tau in frames to retain (default None = all).

    Returns dict  {delta: (per_track_vacf, ensemble_avg)}
    """
    deltas = list(deltas)

    def _one_delta(d: int) -> Tuple[int, Tuple[np.ndarray, np.ndarray]]:
        ptv = vacf_per_track(track_data, track_lengths, min_len, d,
                             max_tlag=max_tlag, _track_index=_track_index)
        avg = vacf_ensemble_average(ptv, track_lengths, min_len, d)
        return d, (ptv, avg)

    results: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    if n_jobs > 1 and len(deltas) > 1:
        workers = min(n_jobs, len(deltas))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_one_delta, d): d for d in deltas}
            for fut in as_completed(futs):
                d, pair = fut.result()
                results[d] = pair
    else:
        for d in deltas:
            _, pair = _one_delta(d)
            results[d] = pair

    return results


# ======================================================================
#  CFBM  FIT  (alpha' / Hurst parameter from collapsed VACF)
# ======================================================================

def cfbm_model(xi: np.ndarray, a: float) -> np.ndarray:
    """Fractional Brownian Motion VACF correlation function.

    C(xi) = 0.5 * [ (xi+1)^a  +  |xi-1|^a  -  2*xi^a ]
    """
    return 0.5 * ((xi + 1) ** a + np.abs(xi - 1) ** a - 2 * xi ** a)


def fit_cfbm(
    vacf_results: Dict[int, Tuple[np.ndarray, np.ndarray]],
    xi_max: float = 3.0,
) -> Dict[str, float]:
    """Fit CFBM model to collapsed (normalized, rescaled) multi-delta VACF.

    The VACF for each delta is normalized by Cv(0) and rescaled so the
    minimum (dip) is at xi = 1.  All deltas are pooled for a single
    global fit of the exponent *a*.

    Returns
    -------
    dict with keys: 'a', 'a_err', 'H' (Hurst = a/2)
    """
    xi_all, cv_all = [], []

    for delta, (_, avg) in vacf_results.items():
        if len(avg) < 3:
            continue
        tau = avg[:, 0]
        cv = (avg[:, 1] + avg[:, 4]) / 2  # average of x and y
        if cv[0] == 0:
            continue
        cv = cv / cv[0]  # normalize

        idx_min = np.argmin(cv)
        if tau[idx_min] == 0:
            continue
        xi = tau / tau[idx_min]

        mask = (xi > 0) & (xi < xi_max)
        mask[0] = False  # skip xi = 0
        if np.sum(mask) < 2:
            continue

        xi_all.append(xi[mask])
        cv_all.append(cv[mask])

    if not xi_all:
        warnings.warn("CFBM fit: insufficient data")
        return {"a": np.nan, "a_err": np.nan, "H": np.nan}

    xi_all = np.concatenate(xi_all)
    cv_all = np.concatenate(cv_all)

    try:
        popt, pcov = curve_fit(
            cfbm_model, xi_all, cv_all,
            p0=[0.6], bounds=([0.01], [2.0]), maxfev=30000,
        )
        a_fit = popt[0]
        a_err = np.sqrt(pcov[0, 0])
    except RuntimeError:
        warnings.warn("CFBM curve_fit failed to converge")
        a_fit = a_err = np.nan

    return {"a": a_fit, "a_err": a_err, "H": a_fit / 2}


# ======================================================================
#  ANGLE  CORRELATION
# ======================================================================

def compute_angles(
    track_data: np.ndarray,
    track_lengths: np.ndarray,
    min_len: int = 3,
    max_tlag: int = 10,
    _track_index: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Dict[int, np.ndarray]:
    """Compute turning angles (degrees, 0-180) for each tlag.

    The turning angle at time t for tlag *tau* is the angle between
    displacement vectors  d1 = r(t+tau) - r(t)  and  d2 = r(t+2*tau) - r(t+tau).

    Returns dict {tlag: 1-D array of angles in degrees}.
    """
    angles_by_tlag: Dict[int, list] = {t: [] for t in range(1, max_tlag + 1)}

    filt = track_lengths[track_lengths[:, 1] >= min_len]
    ids = filt[:, 0].astype(int)

    for tid in ids:
        trk = (_get_track(track_data, _track_index, tid)
               if _track_index is not None
               else track_data[track_data[:, 0] == tid])
        x, y = trk[:, 2], trk[:, 3]
        n = len(trk)

        for tlag in range(1, max_tlag + 1):
            if n < 2 * tlag + 1:
                continue
            # displacement vectors
            dx1 = x[tlag : n - tlag] - x[: n - 2 * tlag]
            dy1 = y[tlag : n - tlag] - y[: n - 2 * tlag]
            dx2 = x[2 * tlag :] - x[tlag : n - tlag]
            dy2 = y[2 * tlag :] - y[tlag : n - tlag]

            dot = dx1 * dx2 + dy1 * dy2
            mag1 = np.sqrt(dx1**2 + dy1**2)
            mag2 = np.sqrt(dx2**2 + dy2**2)
            denom = mag1 * mag2

            valid = denom > 0
            if np.any(valid):
                cos_th = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
                angles_by_tlag[tlag].extend(np.degrees(np.arccos(cos_th)))

    return {t: np.asarray(v) for t, v in angles_by_tlag.items() if len(v) > 0}


def cos_theta_vs_tau(
    angles_by_tlag: Dict[int, np.ndarray],
    time_step: float = 1.0,
    include_random_baseline: bool = True,
) -> pd.DataFrame:
    """Mean <cos(theta)> vs tau.

    Returns DataFrame with columns: tlag, tau_s, mean_cos, std_cos, sem_cos,
    n_angles, [random_mean_cos].
    """
    rows = []
    for tlag in sorted(angles_by_tlag.keys()):
        ang = angles_by_tlag[tlag]
        cos_vals = np.cos(np.deg2rad(ang))
        row = {
            "tlag": tlag,
            "tau_s": tlag * time_step,
            "mean_cos": np.mean(cos_vals),
            "std_cos": np.std(cos_vals),
            "sem_cos": np.std(cos_vals) / np.sqrt(len(cos_vals)),
            "n_angles": len(ang),
        }
        if include_random_baseline:
            rand = np.random.uniform(0, 180, size=len(ang))
            row["random_mean_cos"] = np.mean(np.cos(np.deg2rad(rand)))
        rows.append(row)
    return pd.DataFrame(rows)


def angle_distribution(
    angles: np.ndarray,
    bins: np.ndarray = np.arange(0, 190, 10),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Histogram of angles.  Returns (bin_centers, counts, proportions)."""
    counts, edges = np.histogram(angles, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    total = np.sum(counts)
    props = counts / total if total > 0 else counts.astype(float)
    return centers, counts, props


def angle_ks_tests_consecutive(
    angles_by_tlag: Dict[int, np.ndarray],
) -> pd.DataFrame:
    """KS test between angle distributions at consecutive tlags.

    Returns DataFrame: tlag_a, tlag_b, ks_stat, p_value.
    """
    tlags = sorted(angles_by_tlag.keys())
    rows = []
    for i in range(len(tlags) - 1):
        a, b = tlags[i], tlags[i + 1]
        stat, pval = ks_2samp(angles_by_tlag[a], angles_by_tlag[b])
        rows.append({"tlag_a": a, "tlag_b": b, "ks_stat": stat, "p_value": pval})
    return pd.DataFrame(rows)


def angle_ks_tests_between_conditions(
    angles_a: Dict[int, np.ndarray],
    angles_b: Dict[int, np.ndarray],
) -> pd.DataFrame:
    """KS test between two conditions at each tlag."""
    common_tlags = sorted(set(angles_a) & set(angles_b))
    rows = []
    for t in common_tlags:
        stat, pval = ks_2samp(angles_a[t], angles_b[t])
        rows.append({"tlag": t, "ks_stat": stat, "p_value": pval})
    return pd.DataFrame(rows)


# ======================================================================
#  ERGODICITY  BREAKING  PARAMETER
# ======================================================================

def compute_msd_per_track(
    track_data: np.ndarray,
    track_lengths: np.ndarray,
    min_len: int = 3,
    max_tlag: int = 10,
    _track_index: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Dict[int, np.ndarray]:
    """Time-averaged MSD for each track at each tlag.

    Returns dict {tlag: 1-D array of per-track MSD values (um^2)}.
    """
    filt = track_lengths[track_lengths[:, 1] >= min_len]
    ids = filt[:, 0].astype(int)
    msd_by_tlag: Dict[int, list] = {t: [] for t in range(1, max_tlag + 1)}

    for tid in ids:
        trk = (_get_track(track_data, _track_index, tid)
               if _track_index is not None
               else track_data[track_data[:, 0] == tid])
        x, y = trk[:, 2], trk[:, 3]
        n = len(trk)
        for tlag in range(1, min(max_tlag + 1, n)):
            sq_disp = (x[tlag:] - x[:-tlag]) ** 2 + (y[tlag:] - y[:-tlag]) ** 2
            msd_by_tlag[tlag].append(np.mean(sq_disp))

    return {t: np.asarray(v) for t, v in msd_by_tlag.items() if len(v) > 0}


def ergodicity_breaking_parameter(
    msd_per_track: Dict[int, np.ndarray],
    time_step: float = 1.0,
) -> pd.DataFrame:
    """EB(tau) = Var(MSD_i) / <MSD_i>^2.

    Returns DataFrame: tlag, tau_s, eb, mean_msd, var_msd, n_tracks.
    """
    rows = []
    for tlag in sorted(msd_per_track.keys()):
        vals = msd_per_track[tlag]
        if len(vals) < 2:
            continue
        mean_msd = np.mean(vals)
        var_msd = np.var(vals, ddof=0)
        eb = var_msd / mean_msd**2 if mean_msd != 0 else np.nan
        rows.append({
            "tlag": tlag,
            "tau_s": tlag * time_step,
            "eb": eb,
            "mean_msd": mean_msd,
            "var_msd": var_msd,
            "n_tracks": len(vals),
        })
    return pd.DataFrame(rows)


# ======================================================================
#  NON-GAUSSIAN  PARAMETER  (NGP)
# ======================================================================

def compute_displacements(
    track_data: np.ndarray,
    track_lengths: np.ndarray,
    min_len: int = 3,
    max_tlag: int = 10,
    _track_index: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Compute x and y displacements at each tlag.

    Returns (deltaX, deltaY) dicts  {tlag: 1-D array of displacements (um)}.
    """
    filt = track_lengths[track_lengths[:, 1] >= min_len]
    ids = filt[:, 0].astype(int)
    dx_by_tlag: Dict[int, list] = {t: [] for t in range(1, max_tlag + 1)}
    dy_by_tlag: Dict[int, list] = {t: [] for t in range(1, max_tlag + 1)}

    for tid in ids:
        trk = (_get_track(track_data, _track_index, tid)
               if _track_index is not None
               else track_data[track_data[:, 0] == tid])
        x, y = trk[:, 2], trk[:, 3]
        n = len(trk)
        for tlag in range(1, min(max_tlag + 1, n)):
            dx_by_tlag[tlag].extend(x[tlag:] - x[:-tlag])
            dy_by_tlag[tlag].extend(y[tlag:] - y[:-tlag])

    dx = {t: np.asarray(v) for t, v in dx_by_tlag.items() if len(v) > 0}
    dy = {t: np.asarray(v) for t, v in dy_by_tlag.items() if len(v) > 0}
    return dx, dy


def non_gaussian_parameter(
    deltaX: Dict[int, np.ndarray],
    deltaY: Dict[int, np.ndarray],
) -> pd.DataFrame:
    """Non-Gaussian parameter (1-D, x and y combined).

    NGP = <dr^4> / (3 * <dr^2>^2) - 1
    For a Gaussian distribution, NGP = 0.

    Returns DataFrame: tlag, ngp, kurtosis_excess, n_displacements.
    """
    rows = []
    for tlag in sorted(set(deltaX) & set(deltaY)):
        arr = np.concatenate([deltaX[tlag], deltaY[tlag]])
        if len(arr) < 4:
            continue
        m2 = np.mean(arr**2)
        m4 = np.mean(arr**4)
        ngp = (m4 / (3 * m2**2)) - 1 if m2 != 0 else np.nan
        kurt = kurtosis(arr, fisher=True)  # excess kurtosis
        rows.append({
            "tlag": tlag,
            "ngp": ngp,
            "kurtosis_excess": kurt,
            "n_displacements": len(arr),
        })
    return pd.DataFrame(rows)


# ======================================================================
#  RHEOLOGY  --  G'(omega), G''(omega) via Mason approximation
# ======================================================================

def compute_ensemble_msd(
    track_data: np.ndarray,
    track_lengths: np.ndarray,
    min_len: int = 11,
    max_tlag: int = 100,
    _track_index: Optional[Dict[int, Tuple[int, int]]] = None,
) -> pd.DataFrame:
    """Compute ensemble-average MSD vs tau.

    Uses a single pass over tracks (instead of tlag x tracks) and
    running-sum accumulators to avoid storing all displacements.

    Returns DataFrame: tlag, msd_um2, std_um2, sem_um2, n.
    """
    filt = track_lengths[track_lengths[:, 1] >= min_len]

    # Running accumulators  (index = tlag - 1)
    sums = np.zeros(max_tlag, dtype=np.float64)
    sq_sums = np.zeros(max_tlag, dtype=np.float64)
    counts = np.zeros(max_tlag, dtype=np.int64)

    for row in filt:
        tid = int(row[0])
        n = int(row[1])
        trk = (_get_track(track_data, _track_index, tid)
               if _track_index is not None
               else track_data[track_data[:, 0] == tid])
        x, y = trk[:, 2], trk[:, 3]

        for tlag in range(1, min(max_tlag + 1, n)):
            sd = (x[tlag:] - x[:-tlag]) ** 2 + (y[tlag:] - y[:-tlag]) ** 2
            idx = tlag - 1
            sums[idx] += sd.sum()
            sq_sums[idx] += (sd * sd).sum()
            counts[idx] += len(sd)

    rows = []
    for tlag in range(1, max_tlag + 1):
        idx = tlag - 1
        if counts[idx] == 0:
            continue
        mean_v = sums[idx] / counts[idx]
        var_v = sq_sums[idx] / counts[idx] - mean_v ** 2
        std_v = np.sqrt(max(var_v, 0.0))
        rows.append({
            "tlag": tlag,
            "msd_um2": mean_v,
            "std_um2": std_v,
            "sem_um2": std_v / np.sqrt(counts[idx]),
            "n": int(counts[idx]),
        })
    return pd.DataFrame(rows)


def compute_complex_modulus(
    msd_df: pd.DataFrame,
    time_step: float,
    particle_radius_um: float,
    temperature_K: float = 298.15,
) -> pd.DataFrame:
    """Compute G'(omega) and G''(omega) via the Mason approximation.

    Uses the generalized Stokes-Einstein relation for a 2-D projected MSD
    of a sphere of radius *a* in a viscoelastic medium:

        |G*(omega)| = 2 k_B T / [ 3 pi a  MSD(1/omega)  Gamma(1 + alpha) ]

    where alpha = d ln(MSD) / d ln(tau)  evaluated at tau = 1/omega, and
    the factor 2/3 converts the 2-D MSD to 3-D  (MSD_3D = 3/2 MSD_2D).

    Parameters
    ----------
    msd_df : DataFrame with columns 'tlag', 'msd_um2'
    time_step : frame interval (seconds)
    particle_radius_um : probe radius in microns
    temperature_K : absolute temperature

    Returns
    -------
    DataFrame with columns:
        omega, tau_s, alpha_local, G_abs, G_prime, G_double_prime
    All moduli in Pascals.
    """
    tau_s = msd_df["tlag"].values * time_step
    msd = msd_df["msd_um2"].values

    if len(tau_s) < 3:
        warnings.warn("Need >= 3 MSD points for rheology calculation")
        return pd.DataFrame()

    ln_tau = np.log(tau_s)
    ln_msd = np.log(msd)

    # local log-log slope  (central difference, one-sided at boundaries)
    alpha = np.gradient(ln_msd, ln_tau)

    # clamp alpha to (0, 2) to keep Gamma well-behaved
    alpha = np.clip(alpha, 0.01, 1.99)

    # physical constants
    kbt = K_B * temperature_K                 # Joules
    a_m = particle_radius_um * 1e-6           # metres
    msd_m2 = msd * 1e-12                      # m^2

    # Mason formula
    g_abs = (2 * kbt) / (3 * np.pi * a_m * msd_m2 * gamma_func(1 + alpha))

    omega = 1.0 / tau_s

    g_prime = g_abs * np.cos(np.pi * alpha / 2)
    g_double_prime = g_abs * np.sin(np.pi * alpha / 2)

    return pd.DataFrame({
        "omega": omega,
        "tau_s": tau_s,
        "alpha_local": alpha,
        "G_abs_Pa": g_abs,
        "G_prime_Pa": g_prime,
        "G_double_prime_Pa": g_double_prime,
    })


# ======================================================================
#  PER-REPLICATE  ANALYSIS  &  AGGREGATION
# ======================================================================

def run_single_file(
    filepath: Union[str, Path],
    time_step: float = 0.01,
    micron_per_px: float = 1.0,
    fmt: str = "auto",
    particle_radius_um: float = 0.020,
    temperature_K: float = 298.15,
    min_track_len: int = 10,
    max_tlag_angles: int = 10,
    max_tlag_msd: int = 10,
    max_tlag_vacf: int = 10,
    deltas: Sequence[int] = range(1, 11),
) -> dict:
    """Run all analyses on a single trajectory file (one replicate).

    Returns the same dict structure as :func:`run_all` but without
    saving any files.  Intended for per-replicate analysis that feeds
    into :func:`aggregate_replicates`.
    """
    td, _, _ = load_track_data(filepath, fmt=fmt,
                               micron_per_px=micron_per_px,
                               time_step=time_step)
    sorted_td, tindex, tl = _build_track_index(td)

    # VACF + CFBM
    vacf_res = vacf_multi_delta(sorted_td, tl, min_track_len, deltas,
                                max_tlag=max_tlag_vacf, n_jobs=1,
                                _track_index=tindex)
    cfbm = fit_cfbm(vacf_res)

    # Angles
    angles = compute_angles(sorted_td, tl, min_track_len, max_tlag_angles,
                            _track_index=tindex)
    cos_df = cos_theta_vs_tau(angles, time_step)

    # Ergodicity
    msd_pt = compute_msd_per_track(sorted_td, tl, min_track_len, max_tlag_msd,
                                   _track_index=tindex)
    eb_df = ergodicity_breaking_parameter(msd_pt, time_step)

    # NGP
    dx, dy = compute_displacements(sorted_td, tl, min_track_len, max_tlag_angles,
                                   _track_index=tindex)
    ngp_df = non_gaussian_parameter(dx, dy)

    # MSD + Rheology
    msd_ens = compute_ensemble_msd(sorted_td, tl, min_track_len, max_tlag_msd,
                                   _track_index=tindex)
    rheo_df = compute_complex_modulus(msd_ens, time_step, particle_radius_um,
                                     temperature_K)

    return {
        "filepath": str(filepath),
        "track_lengths": tl,
        "vacf_results": vacf_res,
        "cfbm_params": cfbm,
        "angles": angles,
        "cos_theta_df": cos_df,
        "eb_df": eb_df,
        "ngp_df": ngp_df,
        "msd_df": msd_ens,
        "rheology_df": rheo_df,
    }


def _interp_series(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    x_common: np.ndarray,
) -> np.ndarray:
    """Interpolate (x, y) onto x_common; NaN outside bounds."""
    if len(x_vals) < 2:
        return np.full_like(x_common, np.nan)
    f = interp1d(x_vals, y_vals, bounds_error=False, fill_value=np.nan)
    return f(x_common)


def aggregate_replicates(
    replicate_results: List[dict],
    time_step: float = 0.01,
    n_omega_points: int = 150,
    use_sem: bool = True,
) -> dict:
    """Aggregate per-replicate results into mean +/- SEM statistics.

    Parameters
    ----------
    replicate_results : list of dicts from :func:`run_single_file`
    time_step : frame interval (seconds)
    n_omega_points : number of points for interpolated omega grid
    use_sem : if True, report SEM; if False, report SD

    Returns
    -------
    dict with keys:
        'n_replicates'       : int
        'vacf_agg'           : dict with 'tau_s', 'mean', 'err', 'all'
        'cos_theta_agg'      : DataFrame (tau_s, mean_cos, err_cos, n_reps)
        'eb_agg'             : DataFrame (tau_s, mean_eb, err_eb, n_reps)
        'ngp_agg'            : DataFrame (tlag, tau_s, mean_ngp, err_ngp, n_reps)
        'msd_agg'            : DataFrame (tlag, tau_s, mean_msd, err_msd, n_reps)
        'rheology_agg'       : dict with omega, Gp_mean, Gp_err, Gpp_mean, Gpp_err, ...
        'cfbm_agg'           : dict with a_mean, a_err, a_values
        'per_replicate'      : the original replicate_results list
    """
    n_reps = len(replicate_results)
    if n_reps == 0:
        warnings.warn("aggregate_replicates: no replicates to aggregate")
        return {}

    err_divisor = np.sqrt(n_reps) if use_sem else 1.0

    # ------------------------------------------------------------------
    # VACF (delta=1, normalized, average of x and y)
    # ------------------------------------------------------------------
    vacf_per_rep = []
    for r in replicate_results:
        vr = r["vacf_results"]
        if 1 not in vr:
            continue
        _, avg = vr[1]
        if len(avg) == 0:
            continue
        cv = (avg[:, 1] + avg[:, 4]) / 2  # mean of norm_x, norm_y
        tau_frames = avg[:, 0]
        vacf_per_rep.append((tau_frames, cv))

    vacf_agg = {}
    if vacf_per_rep:
        max_len = max(len(v[1]) for v in vacf_per_rep)
        padded = np.full((len(vacf_per_rep), max_len), np.nan)
        for i, (tau_f, cv) in enumerate(vacf_per_rep):
            padded[i, :len(cv)] = cv
        tau_s = np.arange(max_len) * time_step  # delta=1 → tau = frame * dt
        # But tau_frames from the ensemble may not start at 0; use first rep's
        tau_s = vacf_per_rep[0][0] * time_step
        if len(tau_s) < max_len:
            tau_s = np.arange(max_len) * time_step
        vacf_agg = {
            "tau_s": tau_s[:max_len],
            "mean": np.nanmean(padded, axis=0),
            "err": np.nanstd(padded, axis=0, ddof=1) / err_divisor,
            "all": padded,
        }

    # ------------------------------------------------------------------
    # cos(theta) vs tau
    # ------------------------------------------------------------------
    cos_per_rep = []
    for r in replicate_results:
        ct = r["cos_theta_df"]
        if not ct.empty:
            cos_per_rep.append(ct.set_index("tlag")["mean_cos"])

    cos_theta_agg = pd.DataFrame()
    if cos_per_rep:
        combined = pd.DataFrame(cos_per_rep)
        cos_theta_agg = pd.DataFrame({
            "tlag": combined.columns,
            "tau_s": combined.columns * time_step,
            "mean_cos": combined.mean(axis=0).values,
            "err_cos": (combined.std(axis=0, ddof=1) / err_divisor).values,
            "n_reps": combined.count(axis=0).values.astype(int),
        })

    # ------------------------------------------------------------------
    # EB vs tau
    # ------------------------------------------------------------------
    eb_per_rep = []
    for r in replicate_results:
        eb = r["eb_df"]
        if not eb.empty:
            eb_per_rep.append(eb.set_index("tlag")["eb"])

    eb_agg = pd.DataFrame()
    if eb_per_rep:
        combined = pd.DataFrame(eb_per_rep)
        eb_agg = pd.DataFrame({
            "tlag": combined.columns,
            "tau_s": combined.columns * time_step,
            "mean_eb": combined.mean(axis=0).values,
            "err_eb": (combined.std(axis=0, ddof=1) / err_divisor).values,
            "n_reps": combined.count(axis=0).values.astype(int),
        })

    # ------------------------------------------------------------------
    # NGP vs tlag
    # ------------------------------------------------------------------
    ngp_per_rep = []
    for r in replicate_results:
        ngp = r["ngp_df"]
        if not ngp.empty:
            ngp_per_rep.append(ngp.set_index("tlag")["ngp"])

    ngp_agg = pd.DataFrame()
    if ngp_per_rep:
        combined = pd.DataFrame(ngp_per_rep)
        ngp_agg = pd.DataFrame({
            "tlag": combined.columns,
            "tau_s": combined.columns * time_step,
            "mean_ngp": combined.mean(axis=0).values,
            "err_ngp": (combined.std(axis=0, ddof=1) / err_divisor).values,
            "n_reps": combined.count(axis=0).values.astype(int),
        })

    # ------------------------------------------------------------------
    # MSD vs tau (ensemble MSD from each replicate, then aggregate)
    # ------------------------------------------------------------------
    msd_per_rep = []
    for r in replicate_results:
        msd = r["msd_df"]
        if not msd.empty:
            msd_per_rep.append(msd.set_index("tlag")["msd_um2"])

    msd_agg = pd.DataFrame()
    if msd_per_rep:
        combined = pd.DataFrame(msd_per_rep)
        msd_agg = pd.DataFrame({
            "tlag": combined.columns,
            "tau_s": combined.columns * time_step,
            "mean_msd": combined.mean(axis=0).values,
            "err_msd": (combined.std(axis=0, ddof=1) / err_divisor).values,
            "n_reps": combined.count(axis=0).values.astype(int),
        })

    # ------------------------------------------------------------------
    # Rheology: interpolate onto common omega grid, then aggregate
    # ------------------------------------------------------------------
    rheo_reps = [r["rheology_df"] for r in replicate_results
                 if not r["rheology_df"].empty]
    rheology_agg = {}
    if rheo_reps:
        rheology_agg = _aggregate_rheology(rheo_reps, n_omega_points, err_divisor)

    # ------------------------------------------------------------------
    # CFBM: collect per-replicate 'a' values
    # ------------------------------------------------------------------
    a_vals = np.array([r["cfbm_params"]["a"] for r in replicate_results])
    a_vals = a_vals[np.isfinite(a_vals)]
    cfbm_agg = {
        "a_mean": np.mean(a_vals) if len(a_vals) > 0 else np.nan,
        "a_std": np.std(a_vals, ddof=1) if len(a_vals) > 1 else np.nan,
        "a_err": (np.std(a_vals, ddof=1) / err_divisor) if len(a_vals) > 1 else np.nan,
        "a_values": a_vals,
        "H_mean": np.mean(a_vals) / 2 if len(a_vals) > 0 else np.nan,
    }

    return {
        "n_replicates": n_reps,
        "vacf_agg": vacf_agg,
        "cos_theta_agg": cos_theta_agg,
        "eb_agg": eb_agg,
        "ngp_agg": ngp_agg,
        "msd_agg": msd_agg,
        "rheology_agg": rheology_agg,
        "cfbm_agg": cfbm_agg,
        "per_replicate": replicate_results,
    }


def _aggregate_rheology(
    rheo_dfs: List[pd.DataFrame],
    n_omega_points: int = 150,
    err_divisor: float = 1.0,
) -> dict:
    """Interpolate per-replicate G'/G'' onto a common omega grid and aggregate.

    Returns dict with keys: omega, Gp_mean, Gp_err, Gpp_mean, Gpp_err,
    ratio_mean, ratio_err, Gp_all, Gpp_all, alpha_mean, alpha_err.
    """
    # Find shared omega range across all replicates
    omega_min = max(df["omega"].min() for df in rheo_dfs)
    omega_max = min(df["omega"].max() for df in rheo_dfs)

    if omega_min >= omega_max:
        warnings.warn("Rheology aggregation: no overlapping omega range")
        return {}

    omega_common = np.logspace(np.log10(omega_min), np.log10(omega_max),
                               n_omega_points)

    Gp_matrix = []
    Gpp_matrix = []
    alpha_matrix = []

    for df in rheo_dfs:
        omega_r = df["omega"].values
        idx = np.argsort(omega_r)
        omega_r = omega_r[idx]

        Gp_matrix.append(_interp_series(omega_r, df["G_prime_Pa"].values[idx],
                                        omega_common))
        Gpp_matrix.append(_interp_series(omega_r,
                                         df["G_double_prime_Pa"].values[idx],
                                         omega_common))
        if "alpha_local" in df.columns:
            alpha_matrix.append(_interp_series(omega_r,
                                               df["alpha_local"].values[idx],
                                               omega_common))

    Gp_matrix = np.array(Gp_matrix)
    Gpp_matrix = np.array(Gpp_matrix)

    Gp_mean = np.nanmean(Gp_matrix, axis=0)
    Gpp_mean = np.nanmean(Gpp_matrix, axis=0)
    Gp_err = np.nanstd(Gp_matrix, axis=0, ddof=1) / err_divisor
    Gpp_err = np.nanstd(Gpp_matrix, axis=0, ddof=1) / err_divisor

    # G'/G'' ratio (per-replicate, then aggregate)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_matrix = Gp_matrix / Gpp_matrix
    ratio_mean = np.nanmean(ratio_matrix, axis=0)
    ratio_err = np.nanstd(ratio_matrix, axis=0, ddof=1) / err_divisor

    result = {
        "omega": omega_common,
        "Gp_mean": Gp_mean,
        "Gp_err": Gp_err,
        "Gpp_mean": Gpp_mean,
        "Gpp_err": Gpp_err,
        "ratio_mean": ratio_mean,
        "ratio_err": ratio_err,
        "Gp_all": Gp_matrix,
        "Gpp_all": Gpp_matrix,
    }

    if alpha_matrix:
        alpha_matrix = np.array(alpha_matrix)
        result["alpha_mean"] = np.nanmean(alpha_matrix, axis=0)
        result["alpha_err"] = np.nanstd(alpha_matrix, axis=0, ddof=1) / err_divisor

    return result


def run_replicate_analysis(
    file_list: Sequence[Union[str, Path]],
    time_step: float = 0.01,
    micron_per_px: float = 1.0,
    fmt: str = "auto",
    n_omega_points: int = 150,
    use_sem: bool = True,
    **analysis_kwargs,
) -> dict:
    """Run per-replicate analysis on a list of files and aggregate.

    This is the main entry point for replicate-level analysis.  Each file
    is treated as an independent biological replicate.

    Parameters
    ----------
    file_list : trajectory CSV paths (one per replicate)
    time_step : seconds per frame
    **analysis_kwargs : forwarded to :func:`run_single_file`

    Returns
    -------
    dict from :func:`aggregate_replicates`
    """
    rep_results = []
    for fp in file_list:
        print(f"  Replicate: {Path(fp).name}")
        try:
            res = run_single_file(fp, time_step=time_step,
                                  micron_per_px=micron_per_px,
                                  fmt=fmt, **analysis_kwargs)
            rep_results.append(res)
        except Exception as e:
            warnings.warn(f"Skipping {Path(fp).name}: {e}")
    if not rep_results:
        warnings.warn("No replicates produced results")
        return {}
    return aggregate_replicates(rep_results, time_step, n_omega_points, use_sem)


# ======================================================================
#  PUBLICATION-QUALITY  RHEOLOGY  PLOTS
# ======================================================================

def plot_frequency_sweep(
    rheology_agg_by_condition: Dict[str, dict],
    colors: Optional[Dict[str, str]] = None,
    omega_lim: Optional[Tuple[float, float]] = None,
    modulus_lim: Optional[Tuple[float, float]] = None,
    ratio_lim: Optional[Tuple[float, float]] = (0.05, 20),
    ratio_scale: str = "log",
    figsize: Tuple[float, float] = (6.5, 7.5),
    title: str = "",
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Publication-quality two-panel frequency sweep.

    Top panel : G'(omega) and G''(omega) with SEM bands per condition.
    Bottom panel : G'/G'' ratio vs omega.

    Parameters
    ----------
    rheology_agg_by_condition : {condition_label: dict from _aggregate_rheology}
        Each dict must contain 'omega', 'Gp_mean', 'Gp_err', 'Gpp_mean',
        'Gpp_err', 'ratio_mean', 'ratio_err'.
    colors : {condition_label: hex color}  (defaults to Okabe-Ito palette)
    omega_lim : (lo, hi) for x-axis; auto if None
    modulus_lim : (lo, hi) for modulus y-axis; auto if None
    ratio_lim : (lo, hi) for ratio y-axis
    ratio_scale : 'log' or 'linear' for ratio panel
    """
    conds = list(rheology_agg_by_condition.keys())
    if colors is None:
        colors = {c: _CB_PALETTE[i % len(_CB_PALETTE)] for i, c in enumerate(conds)}

    fig, (ax_mod, ax_rat) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.06},
    )

    from matplotlib.lines import Line2D

    for cond in conds:
        ra = rheology_agg_by_condition[cond]
        if not ra:
            continue
        omega = ra["omega"]
        col = colors.get(cond, "#333333")

        # --- Top panel: G' and G'' ---
        ax_mod.loglog(omega, ra["Gp_mean"], color=col, lw=2.0, ls="-",
                      zorder=3)
        ax_mod.fill_between(omega,
                            ra["Gp_mean"] - ra["Gp_err"],
                            ra["Gp_mean"] + ra["Gp_err"],
                            color=col, alpha=0.18, lw=0, zorder=2)

        ax_mod.loglog(omega, ra["Gpp_mean"], color=col, lw=2.0, ls="--",
                      zorder=3)
        ax_mod.fill_between(omega,
                            ra["Gpp_mean"] - ra["Gpp_err"],
                            ra["Gpp_mean"] + ra["Gpp_err"],
                            color=col, alpha=0.10, lw=0, zorder=2)

        # --- Bottom panel: ratio ---
        ax_rat.plot(omega, ra["ratio_mean"], color=col, lw=2.0, zorder=3)
        ax_rat.fill_between(omega,
                            ra["ratio_mean"] - ra["ratio_err"],
                            ra["ratio_mean"] + ra["ratio_err"],
                            color=col, alpha=0.18, lw=0, zorder=2)

    # -- Reference lines --
    ax_rat.axhline(1, color="#555555", lw=1.0, ls=":", alpha=0.7, zorder=1)
    ax_rat.text(ax_rat.get_xlim()[0] if omega_lim is None else omega_lim[0],
                1.08, "G' = G''", fontsize=7.5, color="#555555", alpha=0.7,
                va="bottom")

    # -- Legends --
    # Condition legend (colors)
    cond_handles = [Line2D([], [], color=colors.get(c, "#333"), lw=2.5, label=c)
                    for c in conds]
    leg_cond = ax_mod.legend(handles=cond_handles, loc="upper left",
                             frameon=True, framealpha=0.92, edgecolor="none",
                             fontsize=9, title="Condition", title_fontsize=9)
    ax_mod.add_artist(leg_cond)

    # Style legend (G' vs G'')
    style_handles = [
        Line2D([], [], color="#333", lw=2.0, ls="-",
               label=r"$G'$ (storage)"),
        Line2D([], [], color="#333", lw=2.0, ls="--",
               label=r"$G''$ (loss)"),
    ]
    ax_mod.legend(handles=style_handles, loc="upper right",
                  frameon=True, framealpha=0.92, edgecolor="none", fontsize=9)

    # -- Axis formatting --
    ax_mod.set_ylabel("Modulus (Pa)", fontsize=12)
    ax_rat.set_ylabel(r"$G' / G''$", fontsize=12)
    ax_rat.set_xlabel(r"Frequency $\omega$ (s$^{-1}$)", fontsize=12)

    ax_rat.set_xscale("log")
    if ratio_scale == "log":
        ax_rat.set_yscale("log")

    if omega_lim:
        ax_mod.set_xlim(omega_lim)
    if modulus_lim:
        ax_mod.set_ylim(modulus_lim)
    if ratio_lim:
        ax_rat.set_ylim(ratio_lim)

    # -- Spine styling --
    for ax in (ax_mod, ax_rat):
        ax.tick_params(axis="both", which="major", labelsize=11, width=1.4,
                       length=5, direction="in", top=True, right=True)
        ax.tick_params(axis="both", which="minor", width=1.0, length=3,
                       direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)
        ax.grid(True, which="major", alpha=0.08, lw=0.5)

    plt.setp(ax_mod.get_xticklabels(), visible=False)

    if title:
        fig.suptitle(title, fontsize=13, y=0.98)

    fig.subplots_adjust(hspace=0.06)
    if title:
        fig.subplots_adjust(top=0.94)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        # Also save PDF if path ends with .png
        if save_path.endswith(".png"):
            fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")

    return fig, (ax_mod, ax_rat)


def plot_replicate_rheology_overlay(
    rheology_agg_by_condition: Dict[str, dict],
    colors: Optional[Dict[str, str]] = None,
    omega_lim: Optional[Tuple[float, float]] = None,
    modulus_lim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (6.5, 5),
    title: str = "",
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay individual replicate G'/G'' traces behind the condition mean.

    Thin lines = individual replicates, thick line + band = mean +/- SEM.
    """
    conds = list(rheology_agg_by_condition.keys())
    if colors is None:
        colors = {c: _CB_PALETTE[i % len(_CB_PALETTE)] for i, c in enumerate(conds)}

    fig, ax = plt.subplots(figsize=figsize)

    for cond in conds:
        ra = rheology_agg_by_condition[cond]
        if not ra:
            continue
        omega = ra["omega"]
        col = colors.get(cond, "#333")

        # Individual replicates (thin, faint)
        if "Gp_all" in ra:
            for i in range(ra["Gp_all"].shape[0]):
                ax.loglog(omega, ra["Gp_all"][i], color=col, lw=0.6,
                          alpha=0.30, zorder=1)
                ax.loglog(omega, ra["Gpp_all"][i], color=col, lw=0.6,
                          alpha=0.20, ls="--", zorder=1)

        # Mean
        ax.loglog(omega, ra["Gp_mean"], color=col, lw=2.2, ls="-",
                  label=f"{cond} G'", zorder=3)
        ax.loglog(omega, ra["Gpp_mean"], color=col, lw=2.2, ls="--",
                  zorder=3)

    ax.set_xlabel(r"Frequency $\omega$ (s$^{-1}$)", fontsize=12)
    ax.set_ylabel("Modulus (Pa)", fontsize=12)
    if omega_lim:
        ax.set_xlim(omega_lim)
    if modulus_lim:
        ax.set_ylim(modulus_lim)

    ax.tick_params(axis="both", which="major", labelsize=11, width=1.4,
                   length=5, direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="minor", width=1.0, length=3,
                   direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    ax.grid(True, which="major", alpha=0.08, lw=0.5)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor="none")

    if title:
        ax.set_title(title, fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if save_path.endswith(".png"):
            fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")

    return fig, ax


# ======================================================================
#  MULTI-CSV  DISCOVERY  &  CONDITION  POOLING
# ======================================================================

# Regex: trailing _NNN (1-4 digits) right before extension-related junk
_REPLICATE_SUFFIX = re.compile(
    r"[_\-](\d{1,4})"          # _001  or  -03
    r"(\.nd2_crop|_crop)?"     # optional .nd2_crop / _crop
    r"$"
)

# Optional leading Traj_ prefix
_TRAJ_PREFIX = re.compile(r"^Traj_", re.IGNORECASE)


def discover_csvs(
    folder: Union[str, Path],
    pattern: str = "*.csv",
    recursive: bool = True,
) -> List[Path]:
    """Find trajectory CSV files in *folder*, sorted alphabetically.

    Skips hidden files, __MACOSX artefacts, and files whose names start
    with '.' or '._'.
    """
    folder = Path(folder)
    glob_fn = folder.rglob if recursive else folder.glob
    files = sorted(
        p for p in glob_fn(pattern)
        if p.is_file()
        and not any(part.startswith(".") or part == "__MACOSX" for part in p.parts)
    )
    return files


def infer_condition(filepath: Union[str, Path]) -> str:
    """Derive a condition label from a trajectory filename.

    Strategy (mirrors SPT-pipeline's convention):
      1. Take the file stem  (``Traj_WT_cytGEM_DMSO_04.nd2_crop`` from
         ``Traj_WT_cytGEM_DMSO_04.nd2_crop.csv``).
      2. Strip a leading ``Traj_`` prefix if present.
      3. Strip a trailing replicate number  ``_NNN``  (1-4 digits), plus any
         intermediate ``_crop`` / ``.nd2_crop`` tag.

    What remains is the condition string.
    """
    stem = Path(filepath).stem
    # Some GEMspa files have double extensions: file.nd2_crop.csv
    # Path.stem only strips the last .csv, leaving "file.nd2_crop"
    # We treat ".nd2_crop" as part of the stem for stripping purposes.

    # Strip Traj_ prefix
    stem = _TRAJ_PREFIX.sub("", stem)

    # Strip trailing replicate number (and optional _crop / .nd2_crop)
    stem = _REPLICATE_SUFFIX.sub("", stem)

    # Clean up trailing underscores / hyphens
    stem = stem.rstrip("_- ")

    return stem if stem else Path(filepath).stem


def pool_by_condition(
    file_list: Sequence[Union[str, Path]],
    manifest_path: Optional[Union[str, Path]] = None,
) -> Dict[str, List[Path]]:
    """Group trajectory files by condition.

    If *manifest_path* is given, it should be a CSV with columns
    ``filename`` and ``condition``.  Files listed there use the manually
    assigned condition; the rest fall through to :func:`infer_condition`.

    Returns
    -------
    dict  {condition_label: [Path, Path, ...]}  sorted by condition name.
    """
    manual_map: Dict[str, str] = {}
    if manifest_path is not None:
        mf = pd.read_csv(manifest_path)
        for _, row in mf.iterrows():
            fname = str(row["filename"]).strip()
            cond = str(row["condition"]).strip()
            if fname and cond:
                manual_map[fname] = cond

    groups: Dict[str, List[Path]] = {}
    for fp in file_list:
        fp = Path(fp)
        cond = manual_map.get(fp.name) or infer_condition(fp)
        groups.setdefault(cond, []).append(fp)

    return dict(sorted(groups.items()))


def load_condition(
    files: Sequence[Path],
    fmt: str = "auto",
    micron_per_px: float = 1.0,
    time_step: float = 1.0,
) -> np.ndarray:
    """Load and concatenate all trajectory files for one condition."""
    td, _, _ = load_multiple_files(files, fmt=fmt,
                                   micron_per_px=micron_per_px,
                                   time_step=time_step)
    return td


# ======================================================================
#  BETWEEN-CONDITION  COMPARISONS
# ======================================================================

def compare_conditions(
    results_by_condition: Dict[str, dict],
) -> Dict[str, pd.DataFrame]:
    """Run pairwise comparisons between conditions.

    Parameters
    ----------
    results_by_condition : {condition_label: results_dict} where each
        results_dict is the output of :func:`run_all`.

    Returns
    -------
    dict with keys:
        'summary'        -- per-condition summary metrics
        'angle_ks'       -- pairwise KS tests on angle distributions
        'ngp_comparison' -- side-by-side NGP values
        'eb_comparison'  -- side-by-side EB values
        'cfbm_comparison'-- side-by-side CFBM parameters
    """
    conds = sorted(results_by_condition.keys())
    out: Dict[str, pd.DataFrame] = {}

    # ---- per-condition summary table ----
    summary_rows = []
    for cond in conds:
        r = results_by_condition[cond]
        tl = r["track_lengths"]
        cfbm = r["cfbm_params"]
        row = {
            "condition": cond,
            "n_tracks": len(tl),
            "median_track_len": np.median(tl[:, 1]),
            "cfbm_a": cfbm["a"],
            "cfbm_a_err": cfbm["a_err"],
            "cfbm_H": cfbm["H"],
        }
        # cos theta at tlag=1
        ct = r["cos_theta_df"]
        if not ct.empty:
            row["cos_theta_tlag1"] = ct.iloc[0]["mean_cos"]
        # EB at tlag=1
        eb = r["eb_df"]
        if not eb.empty:
            row["eb_tlag1"] = eb.iloc[0]["eb"]
        # NGP at tlag=1
        ngp = r["ngp_df"]
        if not ngp.empty:
            row["ngp_tlag1"] = ngp.iloc[0]["ngp"]
        # median G' from rheology (lowest omega = longest tau)
        rheo = r.get("rheology_df")
        if rheo is not None and not rheo.empty:
            row["G_prime_median_Pa"] = np.median(rheo["G_prime_Pa"])
            row["G_double_prime_median_Pa"] = np.median(rheo["G_double_prime_Pa"])
        summary_rows.append(row)
    out["summary"] = pd.DataFrame(summary_rows)

    # ---- pairwise angle KS tests ----
    ks_rows = []
    for ca, cb in combinations(conds, 2):
        ang_a = results_by_condition[ca]["angles"]
        ang_b = results_by_condition[cb]["angles"]
        common_tlags = sorted(set(ang_a) & set(ang_b))
        for t in common_tlags:
            stat, pval = ks_2samp(ang_a[t], ang_b[t])
            ks_rows.append({
                "condition_a": ca, "condition_b": cb,
                "tlag": t, "ks_stat": stat, "p_value": pval,
            })
    out["angle_ks"] = pd.DataFrame(ks_rows)

    # ---- side-by-side NGP ----
    ngp_frames = []
    for cond in conds:
        df = results_by_condition[cond]["ngp_df"].copy()
        df.insert(0, "condition", cond)
        ngp_frames.append(df)
    if ngp_frames:
        out["ngp_comparison"] = pd.concat(ngp_frames, ignore_index=True)

    # ---- side-by-side EB ----
    eb_frames = []
    for cond in conds:
        df = results_by_condition[cond]["eb_df"].copy()
        df.insert(0, "condition", cond)
        eb_frames.append(df)
    if eb_frames:
        out["eb_comparison"] = pd.concat(eb_frames, ignore_index=True)

    # ---- side-by-side CFBM ----
    cfbm_rows = []
    for cond in conds:
        p = results_by_condition[cond]["cfbm_params"]
        cfbm_rows.append({"condition": cond, **p})
    out["cfbm_comparison"] = pd.DataFrame(cfbm_rows)

    return out


def plot_condition_overlay(
    results_by_condition: Dict[str, dict],
    time_step: float = 1.0,
    output_dir: Optional[Union[str, Path]] = None,
    select: Optional[Sequence[str]] = None,
):
    """Overlay key metrics for all conditions on shared axes.

    ``select`` restricts which overlays are drawn (same keywords as
    :func:`run_all`); ``None`` draws all of them.
    """
    conds = sorted(results_by_condition.keys())
    colors = [_CB_PALETTE[i % len(_CB_PALETTE)] for i in range(len(conds))]
    ALL = {"vacf", "cfbm", "angles", "eb", "ngp", "rheology"}
    want = ALL if select is None else {str(s).lower() for s in select} & ALL

    # ---- VACF overlay (delta=1 only) ----
    if "vacf" in want:
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, cond in enumerate(conds):
            vr = results_by_condition[cond]["vacf_results"]
            if 1 in vr:
                _, avg = vr[1]
                if len(avg) > 0:
                    cv = (avg[:, 1] + avg[:, 4]) / 2
                    ax.plot(avg[:, 0] * time_step, cv, color=colors[i],
                            alpha=0.8, label=cond)
        ax.axhline(0, color="k", alpha=0.2)
        ax.set_xlabel("tau (s)")
        ax.set_ylabel("Normalized VACF (delta=1)")
        ax.legend(fontsize=8)
        ax.set_title("VACF comparison")
        plt.tight_layout()
        if output_dir:
            fig.savefig(Path(output_dir) / "comparison_vacf.png", dpi=150)

    # ---- cos(theta) overlay ----
    if "angles" in want:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, cond in enumerate(conds):
            ct = results_by_condition[cond]["cos_theta_df"]
            if not ct.empty:
                ax.errorbar(ct["tau_s"], ct["mean_cos"], yerr=ct["sem_cos"],
                            fmt="o-", capsize=3, color=colors[i], label=cond)
        ax.axhline(0, color="grey", alpha=0.3)
        ax.set_xlabel("tau (s)")
        ax.set_ylabel("<cos(theta)>")
        ax.legend(fontsize=8)
        ax.set_title("Angle correlation comparison")
        plt.tight_layout()
        if output_dir:
            fig.savefig(Path(output_dir) / "comparison_cos_theta.png", dpi=150)

    # ---- EB overlay ----
    if "eb" in want:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, cond in enumerate(conds):
            eb = results_by_condition[cond]["eb_df"]
            if not eb.empty:
                ax.plot(eb["tau_s"], eb["eb"], "o-", color=colors[i],
                        markersize=4, label=cond)
        ax.set_xlabel("tau (s)")
        ax.set_ylabel("EB parameter")
        ax.legend(fontsize=8)
        ax.set_title("Ergodicity breaking comparison")
        plt.tight_layout()
        if output_dir:
            fig.savefig(Path(output_dir) / "comparison_eb.png", dpi=150)

    # ---- NGP overlay ----
    if "ngp" in want:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, cond in enumerate(conds):
            ngp = results_by_condition[cond]["ngp_df"]
            if not ngp.empty:
                ax.plot(ngp["tlag"] * time_step, ngp["ngp"], "o-",
                        color=colors[i], markersize=4, label=cond)
        ax.axhline(0, color="grey", alpha=0.3)
        ax.set_xlabel("tau (s)")
        ax.set_ylabel("NGP")
        ax.legend(fontsize=8)
        ax.set_title("Non-Gaussian parameter comparison")
        plt.tight_layout()
        if output_dir:
            fig.savefig(Path(output_dir) / "comparison_ngp.png", dpi=150)

    # ---- Rheology overlays (separate G' and G'') ----
    has_rheo = "rheology" in want and any(
        not results_by_condition[c].get("rheology_df", pd.DataFrame()).empty
        for c in conds
    )
    if has_rheo:
        # G' (storage modulus)
        fig, ax = plt.subplots(figsize=(6, 5))
        for i, cond in enumerate(conds):
            rdf = results_by_condition[cond].get("rheology_df", pd.DataFrame())
            if not rdf.empty:
                ax.loglog(rdf["omega"], rdf["G_prime_Pa"], "o-",
                          color=colors[i], markersize=4, label=cond)
        ax.set_xlabel("omega (rad/s)")
        ax.set_ylabel("G' (Pa)")
        ax.legend(fontsize=8)
        ax.set_title("Storage modulus comparison")
        plt.tight_layout()
        if output_dir:
            fig.savefig(Path(output_dir) / "comparison_G_prime.png", dpi=150)

        # G'' (loss modulus)
        fig, ax = plt.subplots(figsize=(6, 5))
        for i, cond in enumerate(conds):
            rdf = results_by_condition[cond].get("rheology_df", pd.DataFrame())
            if not rdf.empty:
                ax.loglog(rdf["omega"], rdf["G_double_prime_Pa"], "s-",
                          color=colors[i], markersize=4, label=cond)
        ax.set_xlabel("omega (rad/s)")
        ax.set_ylabel("G'' (Pa)")
        ax.legend(fontsize=8)
        ax.set_title("Loss modulus comparison")
        plt.tight_layout()
        if output_dir:
            fig.savefig(Path(output_dir) / "comparison_G_double_prime.png", dpi=150)

    plt.close("all")


# ======================================================================
#  TOP-LEVEL  MULTI-CONDITION  ENTRY  POINT
# ======================================================================

def run_multi_condition(
    folder: Union[str, Path],
    output_dir: Union[str, Path] = "results",
    time_step: float = 0.01,
    micron_per_px: float = 1.0,
    fmt: str = "auto",
    manifest_path: Optional[Union[str, Path]] = None,
    csv_pattern: str = "*.csv",
    recursive: bool = True,
    n_jobs: int = N_JOBS,
    **analysis_kwargs,
) -> Tuple[Dict[str, dict], Dict[str, pd.DataFrame]]:
    """Discover CSVs in *folder*, pool by condition, analyse, and compare.

    Parameters
    ----------
    folder : directory containing trajectory CSVs
    output_dir : where to write results (per-condition subdirectories + comparisons)
    time_step : seconds per frame
    micron_per_px : spatial calibration (applied only if format is gemspa)
    fmt : 'auto', 'gemspa', or 'spt'
    manifest_path : optional CSV with columns (filename, condition) for manual grouping
    csv_pattern : glob pattern for trajectory files (default ``*.csv``)
    recursive : search subdirectories (default True)
    **analysis_kwargs : forwarded to :func:`run_all`  (particle_radius_um, etc.)

    Returns
    -------
    (results_by_condition, comparison_tables)
    """
    folder = Path(folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- discover & pool ----
    files = discover_csvs(folder, csv_pattern, recursive)
    if not files:
        raise FileNotFoundError(f"No CSV files matching '{csv_pattern}' in {folder}")

    groups = pool_by_condition(files, manifest_path)
    n_conds = len(groups)
    n_files = sum(len(v) for v in groups.values())

    print(f"Found {n_files} files -> {n_conds} condition(s):")
    for cond, flist in groups.items():
        print(f"  {cond}  ({len(flist)} files)")
        for f in flist:
            print(f"    {f.name}")

    # ---- save grouping for reproducibility ----
    grouping_rows = []
    for cond, flist in groups.items():
        for f in flist:
            grouping_rows.append({"filename": f.name, "path": str(f), "condition": cond})
    pd.DataFrame(grouping_rows).to_csv(output_dir / "condition_grouping.csv", index=False)
    print(f"\nCondition grouping saved to {output_dir / 'condition_grouping.csv'}")
    print("  (Edit this file and pass it back via --manifest to override auto-grouping)\n")

    # ---- per-condition analysis (pooled + per-replicate) ----
    results_by_condition: Dict[str, dict] = {}
    replicate_agg_by_condition: Dict[str, dict] = {}

    for cond, flist in groups.items():
        print(f"\n{'='*60}")
        print(f"  Condition: {cond}")
        print(f"{'='*60}")

        # --- Pooled analysis (existing behaviour) ---
        td = load_condition(flist, fmt=fmt, micron_per_px=micron_per_px,
                            time_step=time_step)
        cond_dir = output_dir / cond
        res = run_all(
            td,
            time_step=time_step,
            output_dir=cond_dir,
            label=cond,
            n_jobs=n_jobs,
            **analysis_kwargs,
        )
        results_by_condition[cond] = res

        # --- Per-replicate analysis ---
        if len(flist) >= 2:
            print(f"\n  Per-replicate analysis ({len(flist)} replicates)...")
            rep_kwargs = {k: v for k, v in analysis_kwargs.items()
                         if k in ("particle_radius_um", "temperature_K",
                                  "min_track_len", "max_tlag_angles",
                                  "max_tlag_msd", "max_tlag_vacf", "deltas")}
            rep_agg = run_replicate_analysis(
                flist, time_step=time_step, micron_per_px=micron_per_px,
                fmt=fmt, **rep_kwargs,
            )
            replicate_agg_by_condition[cond] = rep_agg

            # Save replicate-level aggregated CSVs
            rep_dir = cond_dir / "replicate_stats"
            rep_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"{cond}_rep_"

            if isinstance(rep_agg.get("cos_theta_agg"), pd.DataFrame) and \
               not rep_agg["cos_theta_agg"].empty:
                rep_agg["cos_theta_agg"].to_csv(
                    rep_dir / f"{prefix}cos_theta.csv", index=False)
            if isinstance(rep_agg.get("eb_agg"), pd.DataFrame) and \
               not rep_agg["eb_agg"].empty:
                rep_agg["eb_agg"].to_csv(
                    rep_dir / f"{prefix}eb.csv", index=False)
            if isinstance(rep_agg.get("ngp_agg"), pd.DataFrame) and \
               not rep_agg["ngp_agg"].empty:
                rep_agg["ngp_agg"].to_csv(
                    rep_dir / f"{prefix}ngp.csv", index=False)
            if isinstance(rep_agg.get("msd_agg"), pd.DataFrame) and \
               not rep_agg["msd_agg"].empty:
                rep_agg["msd_agg"].to_csv(
                    rep_dir / f"{prefix}msd.csv", index=False)

            ra = rep_agg.get("rheology_agg", {})
            if ra:
                rheo_df = pd.DataFrame({
                    "omega": ra["omega"],
                    "Gp_mean": ra["Gp_mean"], "Gp_err": ra["Gp_err"],
                    "Gpp_mean": ra["Gpp_mean"], "Gpp_err": ra["Gpp_err"],
                    "ratio_mean": ra["ratio_mean"], "ratio_err": ra["ratio_err"],
                })
                rheo_df.to_csv(rep_dir / f"{prefix}rheology.csv", index=False)

            cfbm_a = rep_agg.get("cfbm_agg", {})
            if cfbm_a:
                pd.Series(cfbm_a).to_csv(
                    rep_dir / f"{prefix}cfbm.csv", header=False)

            print(f"  Replicate stats saved to {rep_dir}/")
        else:
            print(f"  Skipping per-replicate analysis (only {len(flist)} file)")

    # ---- between-condition comparisons ----
    print(f"\n{'='*60}")
    print("  Between-condition comparisons")
    print(f"{'='*60}")
    comparisons = compare_conditions(results_by_condition)

    # save comparison tables
    for name, df in comparisons.items():
        if not df.empty:
            df.to_csv(output_dir / f"comparison_{name}.csv", index=False)

    # overlay plots (existing)
    plot_condition_overlay(results_by_condition, time_step, output_dir)

    # ---- Replicate-level frequency sweep plot ----
    if replicate_agg_by_condition:
        rheo_agg_for_plot = {
            c: ra.get("rheology_agg", {})
            for c, ra in replicate_agg_by_condition.items()
            if ra.get("rheology_agg")
        }
        if rheo_agg_for_plot:
            print("  Generating frequency sweep plots...")
            plot_frequency_sweep(
                rheo_agg_for_plot,
                title="Microrheology: Frequency Sweep",
                save_path=str(output_dir / "comparison_frequency_sweep.png"),
            )
            plot_replicate_rheology_overlay(
                rheo_agg_for_plot,
                title="Per-Replicate Rheology Overlay",
                save_path=str(output_dir / "comparison_rheology_replicates.png"),
            )

    plt.close("all")
    print(f"\nAll results saved to {output_dir}/")
    return results_by_condition, comparisons


# ======================================================================
#  PLOTTING  HELPERS  (basic, publication tweaking left to user)
# ======================================================================

def plot_vacf_multi_delta(
    vacf_results: Dict[int, Tuple[np.ndarray, np.ndarray]],
    max_tau_plot: int = 10,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Plot ensemble-averaged VACF for multiple deltas."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for delta, (_, avg) in sorted(vacf_results.items()):
        if len(avg) == 0:
            continue
        cv_mean = (avg[:, 1] + avg[:, 4]) / 2
        ax.plot(avg[:, 0], cv_mean, alpha=0.7, label=f"delta={delta}")
    ax.axhline(0, color="k", alpha=0.2)
    ax.set_xlim(0, max_tau_plot)
    ax.set_xlabel("tau (frames)")
    ax.set_ylabel("Normalized VACF  (avg x,y)")
    ax.legend(fontsize=8, ncol=2)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, ax


def plot_cfbm_collapse(
    vacf_results: Dict[int, Tuple[np.ndarray, np.ndarray]],
    cfbm_params: Dict[str, float],
    xi_max: float = 3.0,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Plot the collapsed VACF with the CFBM fit."""
    fig, ax = plt.subplots(figsize=(6, 5))
    markers = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "*"]

    for i, (delta, (_, avg)) in enumerate(sorted(vacf_results.items())):
        if len(avg) < 3:
            continue
        tau = avg[:, 0]
        cv = (avg[:, 1] + avg[:, 4]) / 2
        if cv[0] == 0:
            continue
        cv = cv / cv[0]
        idx_min = np.argmin(cv)
        if tau[idx_min] == 0:
            continue
        xi = tau / tau[idx_min]
        ax.scatter(
            xi[1:], cv[1:],
            marker=markers[i % len(markers)],
            facecolors="none", edgecolors="steelblue",
            linewidths=1.2, alpha=0.7, s=35,
        )

    # fit curve
    a = cfbm_params.get("a", np.nan)
    if not np.isnan(a):
        xi_dense = np.linspace(0.01, xi_max, 300)
        ax.plot(xi_dense, cfbm_model(xi_dense, a), "--k", lw=2,
                label=f"CFBM  a={a:.3f}")
        ax.legend(fontsize=10, frameon=False)

    ax.axhline(0, color="grey", alpha=0.3)
    ax.set_xlim(0, xi_max)
    ax.set_ylim(-0.3, 1.05)
    ax.set_xlabel("xi")
    ax.set_ylabel("C(xi)")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, ax


def plot_cos_theta(
    cos_df: pd.DataFrame,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Plot <cos(theta)> vs tau."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(cos_df["tau_s"], cos_df["mean_cos"], yerr=cos_df["sem_cos"],
                fmt="o-", capsize=3, label="data")
    if "random_mean_cos" in cos_df.columns:
        ax.plot(cos_df["tau_s"], cos_df["random_mean_cos"],
                "k--", alpha=0.5, label="random")
    ax.axhline(0, color="grey", alpha=0.3)
    ax.set_xlabel("tau (s)")
    ax.set_ylabel("<cos(theta)>")
    ax.legend(fontsize=9)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, ax


def plot_ergodicity(
    eb_df: pd.DataFrame,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Plot ergodicity-breaking parameter vs tau."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(eb_df["tau_s"], eb_df["eb"], s=30)
    ax.set_xlabel("tau (s)")
    ax.set_ylabel("EB = Var(MSD) / <MSD>^2")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, ax


def plot_ngp(
    ngp_df: pd.DataFrame,
    time_step: float = 1.0,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Plot non-Gaussian parameter vs tlag."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ngp_df["tlag"] * time_step, ngp_df["ngp"], s=30)
    ax.axhline(0, color="grey", alpha=0.3, label="Gaussian")
    ax.set_xlabel("tau (s)")
    ax.set_ylabel("Non-Gaussian Parameter")
    ax.legend(fontsize=9)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, ax


def plot_rheology(
    rheo_df: pd.DataFrame,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Log-log plot of G'(omega) and G''(omega)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(rheo_df["omega"], rheo_df["G_prime_Pa"], "o-",
              markersize=4, label="G' (storage)")
    ax.loglog(rheo_df["omega"], rheo_df["G_double_prime_Pa"], "s-",
              markersize=4, label="G'' (loss)")
    ax.set_xlabel("omega (rad/s)")
    ax.set_ylabel("Modulus (Pa)")
    ax.legend(fontsize=9)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, ax


# ======================================================================
#  CONVENIENCE:  run all analyses on one dataset
# ======================================================================

def run_all(
    track_data: np.ndarray,
    time_step: float = 0.01,
    micron_per_px: float = 1.0,
    particle_radius_um: float = 0.020,
    temperature_K: float = 298.15,
    min_track_len: int = 10,
    max_tlag_angles: int = 10,
    max_tlag_msd: int = 10,
    max_tlag_vacf: int = 10,
    deltas: Sequence[int] = range(1, 11),
    output_dir: Optional[Union[str, Path]] = None,
    label: str = "",
    n_jobs: int = N_JOBS,
    select: Optional[Sequence[str]] = None,
) -> dict:
    """Run every analysis and optionally save results + plots.

    ``select`` restricts the work to a subset of
    ``{'vacf','cfbm','angles','eb','ngp','rheology'}``. Dependencies are
    honoured automatically (asking for ``cfbm`` computes the VACF it is fitted
    to). ``None`` means everything.

    Parameters
    ----------
    track_data : (N, 4) array  [id, frame, x_um, y_um]
    time_step : seconds per frame
    particle_radius_um : probe radius for rheology
    temperature_K : temperature for rheology
    min_track_len : minimum track length for all analyses
    max_tlag_angles : max tlag for angle computation
    max_tlag_msd : max tlag for MSD / ergodicity / rheology
    max_tlag_vacf : max tau in frames for VACF (default 10)
    deltas : sequence of delta values for multi-delta VACF
    output_dir : if given, save CSVs and PNGs here
    label : prefix for file names
    n_jobs : number of parallel threads (default ``N_JOBS``)

    Returns
    -------
    dict with keys:
        track_lengths, vacf_results, cfbm_params,
        angles, cos_theta_df, angle_ks_df,
        eb_df, ngp_df, msd_df, rheology_df
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{label}_" if label else ""

    ALL = {"vacf", "cfbm", "angles", "eb", "ngp", "rheology"}
    want = ALL if select is None else {str(s).lower() for s in select} & ALL
    # Dependency: the CFBM collapse is a fit *to* the VACF, so it drags it in.
    need_vacf = bool(want & {"vacf", "cfbm"})

    # ---- Build track index once, pass to every function ----
    sorted_td, tindex, tl = _build_track_index(track_data)
    print(f"Loaded {len(tl)} tracks, lengths {int(tl[:,1].min())}-{int(tl[:,1].max())}")

    vacf_res: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    cfbm = {"a": np.nan, "a_err": np.nan, "H": np.nan}
    angles: Dict[int, np.ndarray] = {}
    cos_df = pd.DataFrame()
    ks_df = pd.DataFrame()
    eb_df = pd.DataFrame()
    ngp_df = pd.DataFrame()
    msd_ens = pd.DataFrame()
    rheo_df = pd.DataFrame()

    # ---- VACF + CFBM  (deltas run in parallel) ----
    if need_vacf:
        print("Computing VACF (multi-delta)...")
        vacf_res = vacf_multi_delta(sorted_td, tl, min_track_len, deltas,
                                    max_tlag=max_tlag_vacf,
                                    n_jobs=n_jobs, _track_index=tindex)
        cfbm = fit_cfbm(vacf_res)
        print(f"  CFBM fit: a = {cfbm['a']:.4f} +/- {cfbm['a_err']:.4f},  H = {cfbm['H']:.4f}")

    # ---- Angles ----
    if "angles" in want:
        print("Computing angle correlations...")
        angles = compute_angles(sorted_td, tl, min_track_len, max_tlag_angles,
                                _track_index=tindex)
        cos_df = cos_theta_vs_tau(angles, time_step)
        ks_df = angle_ks_tests_consecutive(angles)

    # ---- Ergodicity ----
    if "eb" in want:
        print("Computing ergodicity-breaking parameter...")
        msd_pt = compute_msd_per_track(sorted_td, tl, min_track_len, max_tlag_msd,
                                       _track_index=tindex)
        eb_df = ergodicity_breaking_parameter(msd_pt, time_step)

    # ---- NGP (alpha_2) ----
    if "ngp" in want:
        print("Computing non-Gaussian parameter...")
        dx, dy = compute_displacements(sorted_td, tl, min_track_len, max_tlag_angles,
                                       _track_index=tindex)
        ngp_df = non_gaussian_parameter(dx, dy)

    # ---- Rheology ----
    if "rheology" in want:
        print("Computing G'/G'' (Mason approximation)...")
        msd_ens = compute_ensemble_msd(sorted_td, tl, min_track_len, max_tlag_msd,
                                       _track_index=tindex)
        rheo_df = compute_complex_modulus(msd_ens, time_step, particle_radius_um,
                                          temperature_K)

    # ---- Save ----
    if output_dir is not None:
        print(f"Saving results to {output_dir} ...")

        # CSVs
        if "angles" in want:
            cos_df.to_csv(output_dir / f"{prefix}cos_theta.csv", index=False)
            ks_df.to_csv(output_dir / f"{prefix}angle_ks_tests.csv", index=False)
        if "eb" in want:
            eb_df.to_csv(output_dir / f"{prefix}ergodicity.csv", index=False)
        if "ngp" in want:
            ngp_df.to_csv(output_dir / f"{prefix}ngp.csv", index=False)
        if "rheology" in want:
            msd_ens.to_csv(output_dir / f"{prefix}ensemble_msd.csv", index=False)
            if len(rheo_df) > 0:
                rheo_df.to_csv(output_dir / f"{prefix}rheology.csv", index=False)

        # VACF ensemble averages
        if need_vacf:
            for delta, (_, avg) in vacf_res.items():
                if len(avg) > 0:
                    vdf = pd.DataFrame(avg, columns=[
                        "tau", "mean_cx", "std_cx", "sem_cx",
                        "mean_cy", "std_cy", "sem_cy",
                    ])
                    vdf.to_csv(output_dir / f"{prefix}vacf_delta{delta}.csv", index=False)
            pd.Series(cfbm).to_csv(output_dir / f"{prefix}cfbm_fit.csv", header=False)

        # Plots
        if "vacf" in want:
            plot_vacf_multi_delta(vacf_res, title=f"{label} VACF",
                                  save_path=str(output_dir / f"{prefix}vacf.png"))
        if "cfbm" in want:
            plot_cfbm_collapse(vacf_res, cfbm, title=f"{label} CFBM collapse",
                               save_path=str(output_dir / f"{prefix}cfbm_collapse.png"))
        if "angles" in want:
            plot_cos_theta(cos_df, title=f"{label} Angle correlation",
                           save_path=str(output_dir / f"{prefix}cos_theta.png"))
        if "eb" in want:
            plot_ergodicity(eb_df, title=f"{label} Ergodicity breaking",
                            save_path=str(output_dir / f"{prefix}ergodicity.png"))
        if "ngp" in want:
            plot_ngp(ngp_df, time_step, title=f"{label} NGP",
                     save_path=str(output_dir / f"{prefix}ngp.png"))
        if "rheology" in want and len(rheo_df) > 0:
            plot_rheology(rheo_df, title=f"{label} Rheology",
                          save_path=str(output_dir / f"{prefix}rheology.png"))
        plt.close("all")

    return {
        "track_lengths": tl,
        "vacf_results": vacf_res,
        "cfbm_params": cfbm,
        "angles": angles,
        "cos_theta_df": cos_df,
        "angle_ks_df": ks_df,
        "eb_df": eb_df,
        "ngp_df": ngp_df,
        "msd_df": msd_ens,
        "rheology_df": rheo_df,
    }
