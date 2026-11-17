"""Numerical kernels for the advanced trajectory analyses.

Adapted from the standalone GEM analysis library. Only the *computations* were
kept: velocity autocorrelation and its fractional-Brownian-motion collapse,
turning-angle statistics, the ergodicity-breaking parameter, the non-Gaussian
parameter, ensemble MSD, and the Mason/GSER complex shear modulus.

Everything around them -- CSV discovery, filename-based condition inference,
replicate pooling by directory layout, and the ``run_all`` /
``run_multi_condition`` wrappers -- was discarded. That machinery duplicated the
manifest, which is the pipeline's single source of truth for what a movie is and
which condition and experiment it belongs to. Orchestration now lives in
:mod:`spt_pipeline.advanced`, and plotting in :mod:`spt_pipeline.plotting`
house style.

Every function here is pure: arrays in, arrays or DataFrames out. No file I/O,
no matplotlib, no globals.
"""
from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import gamma as gamma_func
from scipy.stats import kurtosis, ks_2samp

K_B = 1.380649e-23  # Boltzmann constant [J/K]
N_JOBS = min(8, os.cpu_count() or 1)

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
