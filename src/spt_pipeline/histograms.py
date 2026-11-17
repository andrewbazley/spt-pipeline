"""Pure histogram computation -- four representations of a fixed-tau step
distribution.

Each function takes arrays/dataframes of ``step_um`` values plus bin edges and
returns a tidy DataFrame suitable for both CSV export and plotting. No
matplotlib here -- plotting lives in :mod:`spt_pipeline.plotting`.

The four representations (from the design doc) trade off track-length bias:

* **raw pooled** -- every displacement, pooled; primary published-style figure
  (long tracks contribute more).
* **track-weighted** -- each track's own normalized histogram, averaged; one
  track, one vote.
* **track-mean** -- one mean step per track.
* **track-length-split** -- raw pooled, stratified by track-length bins, to make
  the bias visible.

When raw-pooled and track-weighted agree, track-length bias is not distorting
the result.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .bins import bin_centers


def raw_pooled_histogram(step_um: np.ndarray, bins: np.ndarray, condition: str) -> pd.DataFrame:
    """Fraction-per-bin over all pooled displacements in a condition."""
    vals = np.asarray(step_um, dtype=float)
    vals = vals[np.isfinite(vals)]
    counts, edges = np.histogram(vals, bins=bins)
    total = counts.sum()
    frac = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
    return pd.DataFrame({
        "condition": condition,
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "bin_center": bin_centers(edges),
        "fraction": frac,
        "count": counts,
        "n_total": int(total),
    })


def track_weighted_histogram(
    step_df: pd.DataFrame, bins: np.ndarray, condition: str,
    track_keys: Sequence[str] = ("movie_id", "particle"),
) -> pd.DataFrame:
    """Average of per-track normalized histograms (+ SEM across tracks)."""
    centers = bin_centers(bins)
    keys = [k for k in track_keys if k in step_df.columns]
    if step_df.empty or not keys:
        return pd.DataFrame({
            "condition": condition, "bin_center": centers,
            "mean_fraction_per_track": np.zeros_like(centers),
            "sem_fraction_per_track": np.zeros_like(centers), "n_tracks": 0,
        })
    # Vectorized: assign every displacement to a bin, count per (track, bin),
    # normalize within track, then take the mean/SEM across tracks per bin. This
    # replaces a Python loop over every track (millions at full-dataset scale).
    nbins = len(bins) - 1
    su = pd.to_numeric(step_df["step_um"], errors="coerce").to_numpy()
    idx = np.searchsorted(bins, su, side="right") - 1
    valid = np.isfinite(su) & (idx >= 0) & (idx < nbins)
    if not valid.any():
        return pd.DataFrame({
            "condition": condition, "bin_center": centers,
            "mean_fraction_per_track": np.zeros_like(centers),
            "sem_fraction_per_track": np.zeros_like(centers), "n_tracks": 0,
        })
    d = step_df.loc[valid, keys].copy()
    d["_bin"] = idx[valid]
    counts = d.groupby(keys + ["_bin"], observed=True).size().rename("c").reset_index()
    totals = d.groupby(keys, observed=True).size().rename("tot").reset_index()
    n = len(totals)  # number of contributing tracks
    counts = counts.merge(totals, on=keys, copy=False)
    counts["frac"] = counts["c"] / counts["tot"]
    counts["frac2"] = counts["frac"] ** 2
    agg = counts.groupby("_bin")[["frac", "frac2"]].sum()
    s1 = np.zeros(nbins); s2 = np.zeros(nbins)
    s1[agg.index.to_numpy()] = agg["frac"].to_numpy()
    s2[agg.index.to_numpy()] = agg["frac2"].to_numpy()
    mean = s1 / n
    if n > 1:
        var = np.clip((s2 - s1 ** 2 / n) / (n - 1), 0.0, None)
        sem = np.sqrt(var / n)
    else:
        sem = np.zeros(nbins)
    return pd.DataFrame({
        "condition": condition, "bin_center": centers,
        "mean_fraction_per_track": mean, "sem_fraction_per_track": sem, "n_tracks": n,
    })


def track_mean_histogram(
    step_df: pd.DataFrame, bins: np.ndarray, condition: str,
    track_keys: Sequence[str] = ("movie_id", "particle"),
) -> pd.DataFrame:
    """Histogram of per-track mean step sizes (QC/secondary summary)."""
    keys = [k for k in track_keys if k in step_df.columns]
    centers = bin_centers(bins)
    if step_df.empty or not keys:
        return pd.DataFrame({
            "condition": condition, "bin_center": centers,
            "fraction": np.zeros_like(centers), "count": np.zeros_like(centers, dtype=int),
            "n_tracks": 0,
        })
    means = step_df.groupby(keys)["step_um"].mean().to_numpy()
    means = means[np.isfinite(means)]
    counts, edges = np.histogram(means, bins=bins)
    total = counts.sum()
    frac = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
    return pd.DataFrame({
        "condition": condition, "bin_center": bin_centers(edges),
        "fraction": frac, "count": counts, "n_tracks": int(len(means)),
    })


def track_length_split_histogram(
    step_df: pd.DataFrame, bins: np.ndarray, condition: str,
    length_bins: Optional[Sequence[int]] = None,
    track_keys: Sequence[str] = ("movie_id", "particle"),
) -> pd.DataFrame:
    """Raw pooled histogram stratified by per-track displacement count.

    ``length_bins`` are edges on the number of displacements at this tau;
    quartile-based when omitted.
    """
    keys = [k for k in track_keys if k in step_df.columns]
    if step_df.empty or not keys:
        return pd.DataFrame(columns=[
            "condition", "length_bin", "bin_center", "fraction", "n_tracks", "n_displacements"])
    counts_per_track = step_df.groupby(keys)["step_um"].size()
    if length_bins:
        edges = np.asarray(sorted(set([0] + list(length_bins) + [int(counts_per_track.max()) + 1])), dtype=float)
        labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges) - 1)]
        cats = pd.cut(counts_per_track, bins=edges, labels=labels, include_lowest=True)
    else:
        try:
            cats = pd.qcut(counts_per_track, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        except ValueError:
            cats = pd.Series(["all"] * len(counts_per_track), index=counts_per_track.index)
    # Map each track's length category back onto its displacements with a
    # vectorized merge (never a row-wise apply, which is catastrophic at
    # full-dataset scale).
    cat_df = cats.rename("_len_cat").reset_index()
    tmp = step_df[keys + ["step_um"]].merge(cat_df, on=keys, how="left", copy=False)
    rows: List[pd.DataFrame] = []
    for cat, grp in tmp.groupby("_len_cat", observed=True):
        vals = pd.to_numeric(grp["step_um"], errors="coerce").to_numpy()
        vals = vals[np.isfinite(vals)]
        cnt, edges2 = np.histogram(vals, bins=bins)
        total = cnt.sum()
        frac = cnt / total if total > 0 else np.zeros_like(cnt, dtype=float)
        n_tracks = grp.groupby(keys).ngroups
        rows.append(pd.DataFrame({
            "condition": condition, "length_bin": str(cat),
            "bin_center": bin_centers(edges2), "fraction": frac,
            "n_tracks": n_tracks, "n_displacements": int(total),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[
        "condition", "length_bin", "bin_center", "fraction", "n_tracks", "n_displacements"])


def attrition_row(step_df: pd.DataFrame, n_tracks_total: int, tau_frames: int, tau_s: float,
                  condition: str, track_keys: Sequence[str] = ("movie_id", "particle")) -> Dict:
    """One row of the cross-tau attrition summary."""
    keys = [k for k in track_keys if k in step_df.columns]
    if step_df.empty or not keys:
        n_contrib = 0
        med_len = np.nan
        n_disp = 0
    else:
        sizes = step_df.groupby(keys)["step_um"].size()
        n_contrib = int((sizes > 0).sum())
        n_disp = int(sizes.sum())
        med_len = float(np.nanmedian(sizes.to_numpy())) if len(sizes) else np.nan
    frac_excl = 1.0 - (n_contrib / n_tracks_total) if n_tracks_total else np.nan
    return {
        "condition": condition, "tau_frames": tau_frames, "tau_s": tau_s,
        "n_tracks_total": n_tracks_total, "n_tracks_contributing": n_contrib,
        "n_displacements": n_disp, "median_displacements_per_track": med_len,
        "fraction_tracks_excluded": frac_excl,
    }
