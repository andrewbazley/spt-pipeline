"""Brightness: flatfield correction, z-scoring, and brightness-mobility links.

Ports the illumination-correction and brightness-normalization logic from the
legacy scripts into one place. Flatfield correction fits a 2D quadratic
illumination surface from per-detection masses and divides it out, so that
apparent brightness differences reflect the molecule rather than its position
in an unevenly illuminated field.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def fit_flatfield_poly(tracks: Optional[pd.DataFrame], diff_df: pd.DataFrame):
    """Fit a 2nd-order 2D polynomial illumination surface from detection masses.

    Returns ``(coeffs, mean_x, mean_y, fit_x, fit_y, fit_m)`` or ``None``.
    """
    if tracks is not None and not tracks.empty and "mass" in tracks.columns:
        fx = pd.to_numeric(tracks["x"], errors="coerce").to_numpy()
        fy = pd.to_numeric(tracks["y"], errors="coerce").to_numpy()
        fm = pd.to_numeric(tracks["mass"], errors="coerce").to_numpy()
    elif {"mean_x_px", "mean_y_px", "mass_per_frame"}.issubset(diff_df.columns):
        fx = pd.to_numeric(diff_df["mean_x_px"], errors="coerce").to_numpy()
        fy = pd.to_numeric(diff_df["mean_y_px"], errors="coerce").to_numpy()
        fm = pd.to_numeric(diff_df["mass_per_frame"], errors="coerce").to_numpy()
    else:
        return None
    valid = np.isfinite(fx) & np.isfinite(fy) & np.isfinite(fm) & (fm > 0)
    if valid.sum() < 20:
        return None
    fx, fy, fm = fx[valid], fy[valid], fm[valid]
    mx, my = fx.mean(), fy.mean()
    cx, cy = fx - mx, fy - my
    A = np.column_stack([np.ones(len(cx)), cx, cy, cx ** 2, cx * cy, cy ** 2])
    try:
        coeffs, *_ = np.linalg.lstsq(A, fm, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return coeffs, mx, my, fx, fy, fm


def flatfield_correct(diff_df: pd.DataFrame, tracks: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Add ``mass_per_frame_ff`` and ``flatfield_factor`` columns.

    Falls back to identity correction when the fit is unavailable.
    """
    d = diff_df.copy()
    if d.empty or "mean_x_px" not in d.columns:
        d["mass_per_frame_ff"] = d.get("mass_per_frame", np.nan)
        d["flatfield_factor"] = 1.0
        return d
    fit = fit_flatfield_poly(tracks, d)
    if fit is None:
        d["mass_per_frame_ff"] = d.get("mass_per_frame", np.nan)
        d["flatfield_factor"] = 1.0
        return d
    coeffs, mx, my, *_ = fit
    tx = pd.to_numeric(d["mean_x_px"], errors="coerce").to_numpy() - mx
    ty = pd.to_numeric(d["mean_y_px"], errors="coerce").to_numpy() - my
    A = np.column_stack([np.ones(len(tx)), tx, ty, tx ** 2, tx * ty, ty ** 2])
    predicted = A @ coeffs
    median_pred = np.nanmedian(predicted[predicted > 0]) if np.any(predicted > 0) else 0
    factor = predicted / median_pred if median_pred > 0 else np.ones(len(predicted))
    factor = np.clip(factor, 0.1, 10.0)
    d["flatfield_factor"] = factor
    d["mass_per_frame_ff"] = pd.to_numeric(d["mass_per_frame"], errors="coerce") / factor
    return d


def add_brightness_zscore(diff_df: pd.DataFrame) -> pd.DataFrame:
    """Per-table z-score of (flatfielded) mass_per_frame."""
    d = diff_df.copy()
    col = "mass_per_frame_ff" if "mass_per_frame_ff" in d.columns else "mass_per_frame"
    if col not in d.columns or d.empty:
        d["mass_per_frame_zscore"] = np.nan
        return d
    vals = pd.to_numeric(d[col], errors="coerce")
    mu, sigma = vals.mean(), vals.std()
    d["mass_per_frame_zscore"] = (vals - mu) / sigma if sigma and sigma > 0 else 0.0
    return d


def brightness_diffusion_correlation(diff_df: pd.DataFrame,
                                     brightness_col: str = "mass_per_frame_ff",
                                     d_col: str = "D_fit_um2_s") -> dict:
    """Spearman correlation and log-log (Stokes-Einstein) slope of D vs brightness."""
    out = {"spearman_rho": np.nan, "spearman_p": np.nan,
           "loglog_slope": np.nan, "n": 0}
    if diff_df is None or diff_df.empty:
        return out
    bcol = brightness_col if brightness_col in diff_df.columns else "mass_per_frame"
    if bcol not in diff_df.columns or d_col not in diff_df.columns:
        return out
    b = pd.to_numeric(diff_df[bcol], errors="coerce").to_numpy()
    dd = pd.to_numeric(diff_df[d_col], errors="coerce").to_numpy()
    ok = np.isfinite(b) & np.isfinite(dd) & (b > 0) & (dd > 0)
    if ok.sum() < 5:
        return out
    b, dd = b[ok], dd[ok]
    rho, p = stats.spearmanr(b, dd)
    slope = float(np.polyfit(np.log(b), np.log(dd), 1)[0])
    out.update({"spearman_rho": float(rho), "spearman_p": float(p),
                "loglog_slope": slope, "n": int(ok.sum())})
    return out
