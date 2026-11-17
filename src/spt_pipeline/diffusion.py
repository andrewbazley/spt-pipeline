"""MSD / IMSD / EMSD and effective diffusion coefficients.

Unifies the ``build_diffusion_table`` and ``build_msd_summary_table`` logic that
was duplicated (and drifting) across the three legacy scripts. Diffusivity is
scale-dependent in cytoplasm, so D is always reported *at a specific tau*
(Deff-at-tau), following Holt-lab convention.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from . import utils

DIFFUSION_COLUMNS = [
    "particle", "n_spots", "dwell_time", "alpha", "total_mass",
    "mean_x_px", "mean_y_px", "mass_per_frame",
    "tau_fit_frames", "tau_fit_s", "msd_fit_um2", "D_fit_um2_s", "D_fit_px2_frame",
    "tau_last_frames", "tau_last_s", "msd_last_um2", "D_last_um2_s", "D_last_px2_frame",
]


def _to_series(obj) -> pd.Series:
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            return pd.Series(dtype=float)
        return obj.iloc[:, 0]
    return pd.Series(obj)


def alpha_map_from_imsd(imsd: pd.DataFrame) -> Dict[Any, float]:
    """Anomalous exponent alpha per particle from the log-log IMSD slope."""
    if imsd is None or imsd.empty:
        return {}
    tau_s = pd.to_numeric(pd.Index(imsd.index), errors="coerce").to_numpy(dtype=float)
    ok_tau = np.isfinite(tau_s) & (tau_s > 0)
    if ok_tau.sum() < 2:
        return {col: np.nan for col in imsd.columns}
    vals = imsd.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    log_tau = np.log(tau_s[ok_tau])
    out: Dict[Any, float] = {}
    for j, col in enumerate(imsd.columns):
        col_vals = vals[ok_tau, j]
        good = np.isfinite(col_vals) & (col_vals > 0)
        if good.sum() < 2:
            out[col] = np.nan
        else:
            slope = float(np.polyfit(log_tau[good], np.log(col_vals[good]), 1)[0])
            out[col] = max(slope, 0.0)
    return out


def build_diffusion_table(
    tracks: pd.DataFrame, imsd: pd.DataFrame, fps: float, mpp: float, diffusion_tau: int,
) -> pd.DataFrame:
    """Per-particle diffusion summary with D at a fixed target tau and at the
    last available lag."""
    if tracks is None or tracks.empty or imsd is None or imsd.empty:
        return pd.DataFrame(columns=DIFFUSION_COLUMNS)

    tracks = utils.normalize_track_table(tracks)
    amap = alpha_map_from_imsd(imsd)
    agg = {
        "n_spots": ("frame", "size"),
        "mean_x_px": ("x", "mean"),
        "mean_y_px": ("y", "mean"),
    }
    if "dwell_time" in tracks.columns:
        agg["dwell_time"] = ("dwell_time", "max")
    if "total_mass" in tracks.columns:
        agg["total_mass"] = ("total_mass", "first")
    if "mass" in tracks.columns:
        agg["mean_mass_per_frame"] = ("mass", "mean")
    if "alpha" in tracks.columns:
        agg["alpha"] = ("alpha", "first")
    meta = tracks.groupby("particle", dropna=True).agg(**agg)
    if "alpha" not in meta.columns or meta["alpha"].isna().all():
        meta["alpha"] = [amap.get(idx, amap.get(str(idx), np.nan)) for idx in meta.index]

    tau = pd.to_numeric(pd.Index(imsd.index), errors="coerce").to_numpy(dtype=float)
    tau_frames = np.full_like(tau, np.nan, dtype=float)
    finite = np.isfinite(tau)
    tau_frames[finite] = np.rint(tau[finite] * float(fps))
    target = int(max(1, diffusion_tau))
    target_s = target / float(fps)
    px2_frame = float(mpp) ** 2 * float(fps)

    imsd_num = imsd.apply(pd.to_numeric, errors="coerce")
    vals = imsd_num.to_numpy(dtype=float)
    tau_pos = np.isfinite(tau) & (tau > 0)
    fit_cand = tau_pos & np.isfinite(tau_frames) & (tau_frames == target)
    meta_lookup = {str(k): k for k in meta.index.tolist()}

    rows = []
    for j, pcol in enumerate(imsd_num.columns):
        col = vals[:, j]
        finite_col = tau_pos & np.isfinite(col)
        tau_last = msd_last = D_last = np.nan
        tau_last_fr = np.nan
        if finite_col.any():
            k = np.where(finite_col)[0][-1]
            tau_last = float(tau[k])
            tau_last_fr = float(tau_frames[k]) if np.isfinite(tau_frames[k]) else np.nan
            msd_last = float(col[k])
            D_last = float(msd_last / (4.0 * tau_last)) if tau_last > 0 else np.nan
        fmask = fit_cand & np.isfinite(col)
        msd_fit = D_fit = np.nan
        if fmask.any():
            k = np.where(fmask)[0][0]
            msd_fit = float(col[k])
            D_fit = float(msd_fit / (4.0 * target_s)) if target_s > 0 else np.nan
        key = pcol if pcol in meta.index else meta_lookup.get(str(pcol))
        row = {c: np.nan for c in DIFFUSION_COLUMNS}
        row.update({
            "particle": pcol,
            "tau_fit_frames": target, "tau_fit_s": target_s,
            "msd_fit_um2": msd_fit, "D_fit_um2_s": D_fit,
            "D_fit_px2_frame": np.nan if not np.isfinite(D_fit) else D_fit / px2_frame,
            "tau_last_frames": tau_last_fr, "tau_last_s": tau_last,
            "msd_last_um2": msd_last, "D_last_um2_s": D_last,
            "D_last_px2_frame": np.nan if not np.isfinite(D_last) else D_last / px2_frame,
        })
        if key is not None:
            m = meta.loc[key].to_dict()
            if "mean_mass_per_frame" in m:
                m["mass_per_frame"] = m.pop("mean_mass_per_frame")
            row.update(m)
            row["particle"] = key
        rows.append(row)

    return pd.DataFrame(rows, columns=DIFFUSION_COLUMNS)


def build_msd_summary(imsd: pd.DataFrame, emsd) -> pd.DataFrame:
    """Ensemble + individual MSD vs tau summary table."""
    tau_i = (pd.to_numeric(pd.Index(imsd.index), errors="coerce").to_numpy(dtype=float)
             if imsd is not None and not imsd.empty else np.array([], dtype=float))
    es = _to_series(emsd)
    tau_e = (pd.to_numeric(pd.Index(es.index), errors="coerce").to_numpy(dtype=float)
             if not es.empty else np.array([], dtype=float))
    tau = np.union1d(tau_i[np.isfinite(tau_i)], tau_e[np.isfinite(tau_e)])
    cols = ["tau_s", "emsd_um2", "imsd_mean_um2", "imsd_median_um2", "imsd_n_tracks"]
    if tau.size == 0:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame({"tau_s": tau})
    if not es.empty:
        clean = pd.Series(pd.to_numeric(es, errors="coerce").to_numpy(dtype=float), index=tau_e)
        clean = clean[~clean.index.duplicated(keep="first")]
        out["emsd_um2"] = out["tau_s"].map(clean.to_dict())
    else:
        out["emsd_um2"] = np.nan
    if imsd is not None and not imsd.empty:
        num = imsd.apply(pd.to_numeric, errors="coerce")
        num.index = tau_i
        num = num[~num.index.duplicated(keep="first")]
        out["imsd_mean_um2"] = out["tau_s"].map(num.mean(axis=1).to_dict())
        out["imsd_median_um2"] = out["tau_s"].map(num.median(axis=1).to_dict())
        out["imsd_n_tracks"] = out["tau_s"].map(num.notna().sum(axis=1).to_dict())
    else:
        out["imsd_mean_um2"] = np.nan
        out["imsd_median_um2"] = np.nan
        out["imsd_n_tracks"] = 0
    return out


def estimate_loc_error_floor(
    diff_df: pd.DataFrame, alpha_max: float = 0.3, min_nspots: int = 10,
) -> float:
    """Estimate localization-error MSD floor (um^2) from quasi-immobile particles.

    Particles with low alpha and enough spots are treated as effectively
    immobile; their median short-lag MSD approximates ``4*sigma_loc^2`` and is
    subtracted from D at fit tau elsewhere.
    """
    if diff_df is None or diff_df.empty:
        return 0.0
    d = diff_df.copy()
    for c in ("alpha", "n_spots", "msd_fit_um2", "tau_fit_s"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    mask = pd.Series(True, index=d.index)
    if "alpha" in d.columns:
        mask &= d["alpha"] <= alpha_max
    if "n_spots" in d.columns:
        mask &= d["n_spots"] >= min_nspots
    sub = d[mask]
    if sub.empty or "msd_fit_um2" not in sub.columns:
        return 0.0
    floor = float(np.nanmedian(sub["msd_fit_um2"].to_numpy(dtype=float)))
    return floor if np.isfinite(floor) and floor > 0 else 0.0


def apply_loc_error_subtraction(diff_df: pd.DataFrame, msd_floor_um2: float) -> pd.DataFrame:
    """Subtract an MSD floor and recompute corrected D at fit tau (>=0)."""
    if diff_df is None or diff_df.empty or msd_floor_um2 <= 0:
        d = diff_df.copy() if diff_df is not None else pd.DataFrame()
        if not d.empty:
            d["D_fit_um2_s_corr"] = pd.to_numeric(d.get("D_fit_um2_s"), errors="coerce")
        return d
    d = diff_df.copy()
    msd = pd.to_numeric(d["msd_fit_um2"], errors="coerce").to_numpy(dtype=float)
    tau_s = pd.to_numeric(d["tau_fit_s"], errors="coerce").to_numpy(dtype=float)
    corrected_msd = np.clip(msd - msd_floor_um2, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        d["msd_fit_um2_corr"] = corrected_msd
        d["D_fit_um2_s_corr"] = np.where(tau_s > 0, corrected_msd / (4.0 * tau_s), np.nan)
    return d
