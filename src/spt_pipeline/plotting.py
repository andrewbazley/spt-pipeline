"""All matplotlib code: one palette, one house style, one lock.

Centralizing plotting guarantees a consistent, colorblind-safe look across
every figure and keeps analysis modules free of drawing code. matplotlib's
pyplot state machine is not thread-safe, so all figure work is serialized
through ``_PLOT_LOCK``.

Palette: the Okabe-Ito colorblind-safe qualitative set, plus perceptually
uniform sequential maps (viridis/cividis) for continuous data.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PLOT_LOCK = threading.Lock()

# Okabe-Ito colorblind-safe qualitative palette (8 colors).
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]
SEQUENTIAL_CMAP = "viridis"
SEQUENTIAL_CMAP_ALT = "cividis"


def condition_colors(conditions: Sequence[str]) -> Dict[str, str]:
    """Deterministic condition -> color mapping, identical on every figure.

    Sorted before assignment so a condition keeps its colour regardless of the
    order it happens to arrive in. Without this, the same condition can be blue
    on one panel and orange on the next depending on dict iteration order, which
    is the single most common way a multi-panel figure misleads a reader.
    """
    conds = sorted(dict.fromkeys(conditions), key=str)
    return {c: OKABE_ITO[i % len(OKABE_ITO)] for i, c in enumerate(conds)}


def apply_house_style() -> None:
    """Publication-quality rcParams: legible fonts, clean spines, tight fits."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.autolayout": False,
    })


def _save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step-size histograms
# ---------------------------------------------------------------------------

def p_to_stars(p: Optional[float]) -> str:
    """Conventional significance notation. Blank when the test did not run.

    Thresholds follow the usual convention: * 0.05, ** 0.01, *** 0.001,
    **** 0.0001. Anything above 0.05 is ``n.s.``.
    """
    if p is None:
        return ""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def significance_brackets(
    ax, comparisons: Sequence[tuple], positions: Dict[str, float], *,
    y_start: Optional[float] = None, tier_gap: float = 0.075,
    color: str = "0.25", fontsize: int = 11, max_brackets: int = 8,
) -> None:
    """Draw comparison bars with asterisks over a categorical axis.

    ``comparisons`` is a sequence of ``(condition_a, condition_b, p_value)``.
    Bars are stacked in tiers from the shortest span upward so they never cross
    each other, and the y-limit is extended to make room rather than letting a
    bar collide with the data.

    Only the marker is drawn -- no test name, no numeric p. The marker is the
    conventional reading; the exact statistics live in the CSVs, where they can
    be read precisely instead of squinted at.
    """
    pairs = [(a, b, p) for a, b, p in comparisons
             if a in positions and b in positions and p_to_stars(p)]
    if not pairs:
        return
    # With many conditions the number of pairs grows quadratically and a full
    # ladder of bars swamps the data. Shed the least informative first: drop
    # 'n.s.' bars, then keep the shortest spans (adjacent comparisons, which are
    # the ones a reader looks for). The complete set of tests is in the CSV.
    if len(pairs) > max_brackets:
        sig = [t for t in pairs if p_to_stars(t[2]) != "n.s."]
        pairs = sig if sig else pairs
    if len(pairs) > max_brackets:
        pairs = sorted(pairs, key=lambda t: abs(positions[t[1]] - positions[t[0]]))[:max_brackets]
    # Shortest spans first -> low tiers, so long bars sit above short ones.
    pairs.sort(key=lambda t: abs(positions[t[1]] - positions[t[0]]))

    lo, hi = ax.get_ylim()
    span = hi - lo
    if span <= 0:
        return
    base = (y_start if y_start is not None else hi - span * 0.02)
    step = span * tier_gap
    tick = span * 0.014

    occupied: List[tuple] = []      # (xmin, xmax, tier) already drawn
    top = base
    for a, b, p in pairs:
        x1, x2 = sorted((positions[a], positions[b]))
        tier = 0
        while any(t == tier and not (x2 < xmin - 0.05 or x1 > xmax + 0.05)
                  for xmin, xmax, t in occupied):
            tier += 1
        occupied.append((x1, x2, tier))
        y = base + tier * step
        ax.plot([x1, x1, x2, x2], [y, y + tick, y + tick, y], linewidth=1.1,
                color=color, clip_on=False, solid_joinstyle="miter")
        ax.text((x1 + x2) / 2.0, y + tick * 1.15, p_to_stars(p), ha="center",
                va="bottom", fontsize=fontsize if p_to_stars(p) != "n.s." else fontsize - 2,
                color=color, clip_on=False)
        top = max(top, y + tick + step * 0.45)
    ax.set_ylim(lo, max(hi, top + span * 0.04))


def comparisons_from_stats(stat_df: pd.DataFrame, p_col: str = "p_value") -> List[tuple]:
    """Extract ``(condition_a, condition_b, p)`` triples from a results table.

    Prefers the BH-corrected column when several conditions were compared, since
    that is the value the asterisks should reflect.
    """
    if stat_df is None or getattr(stat_df, "empty", True):
        return []
    col = "p_value_bh" if "p_value_bh" in stat_df.columns else p_col
    if col not in stat_df.columns:
        return []
    return [(str(r["condition_a"]), str(r["condition_b"]), r[col])
            for _, r in stat_df.iterrows()]



#: Bins with no counts cannot be drawn on a log axis. They are masked out
#: (leaving a visible gap) rather than floored to a small number, because a
#: floor invents a value the data does not contain.
def _log_y_setup(ax, series_list: Sequence[np.ndarray]) -> bool:
    """Switch an axis to log-y and set a lower limit from the data.

    The bottom is half the smallest positive value plotted, so the axis spans
    exactly the range that exists. Note that on a log axis a single-count bin
    is visually as prominent as a well-populated one, so the extreme tail --
    where mislinks live -- looks more substantial than it is.
    """
    vals = np.concatenate([np.asarray(v, dtype=float).ravel() for v in series_list
                           if v is not None and len(v)]) if series_list else np.array([])
    pos = vals[np.isfinite(vals) & (vals > 0)]
    if pos.size == 0:
        return False
    ax.set_yscale("log")
    ax.set_ylim(bottom=float(pos.min()) * 0.5, top=float(pos.max()) * 1.6)
    return True


def _mask_nonpositive(y: np.ndarray) -> np.ndarray:
    """Replace non-positive values with NaN so a log plot leaves a gap."""
    y = np.asarray(y, dtype=float).copy()
    y[~(y > 0)] = np.nan
    return y


def overlay_step_histograms(
    hist_by_condition: Dict[str, pd.DataFrame], out_png: Path, *,
    title: str, xlabel: str = "Step size (µm)", ylabel: str = "Fraction per bin",
    legend_meta: Optional[Dict[str, str]] = None, xlim: Optional[float] = None,
    ylim: Optional[float] = None, log_y: bool = False,
) -> None:
    """Overlay per-condition raw-pooled (or track-mean) histograms as step curves.

    ``xlim``/``ylim`` are enforced identically across conditions (and, when the
    caller passes the same values, across the four representations at a given
    tau) so the plots are directly comparable. They are set to span the full
    data so no tail is clipped.

    ``log_y`` puts the fraction axis on a log scale. Step-size distributions are
    heavy-tailed and the mobile population lives in the tail, where linear
    fractions of ~1e-3 are indistinguishable from zero; on a log axis that tail
    is legible, and a two-population mixture shows as a change of slope rather
    than an invisible shoulder. The shading is dropped, since filling to zero is
    meaningless on a log axis.
    """
    with _PLOT_LOCK:
        apply_house_style()
        colors = condition_colors(list(hist_by_condition.keys()))
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        drawn = []
        for cond, df in hist_by_condition.items():
            if df is None or df.empty:
                continue
            label = cond if not legend_meta else f"{cond} ({legend_meta.get(cond, '')})"
            y = df["fraction"].to_numpy(dtype=float)
            drawn.append(y)
            ax.step(df["bin_center"], _mask_nonpositive(y) if log_y else y,
                    where="mid", color=colors[cond], linewidth=1.8, label=label)
            if not log_y:
                ax.fill_between(df["bin_center"], y, step="mid",
                                color=colors[cond], alpha=0.12)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        if xlim:
            ax.set_xlim(0, xlim)
        if log_y:
            _log_y_setup(ax, drawn)
        elif ylim:
            ax.set_ylim(0, ylim)
        ax.legend(loc="upper right")
        _save(fig, out_png)


def overlay_track_weighted(
    hist_by_condition: Dict[str, pd.DataFrame], out_png: Path, *,
    title: str, xlabel: str = "Step size (µm)", xlim: Optional[float] = None,
    ylim: Optional[float] = None, log_y: bool = False,
) -> None:
    """Track-weighted mean histograms with SEM ribbons.

    On a log axis the lower SEM edge is clipped to the axis floor rather than
    dropped: ``mean - SEM`` goes negative wherever the SEM is comparable to the
    mean, which is most of the tail, and discarding those bins would silently
    hide the least certain part of the curve.
    """
    with _PLOT_LOCK:
        apply_house_style()
        colors = condition_colors(list(hist_by_condition.keys()))
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        drawn = []
        for cond, df in hist_by_condition.items():
            if df is None or df.empty:
                continue
            c = colors[cond]
            m = df["mean_fraction_per_track"].to_numpy(dtype=float)
            s = df["sem_fraction_per_track"].to_numpy(dtype=float)
            x = df["bin_center"].to_numpy()
            drawn.append(m)
            ax.plot(x, _mask_nonpositive(m) if log_y else m, color=c, linewidth=1.8,
                    label=f"{cond} (n={int(df['n_tracks'].iloc[0])} tracks)")
            if log_y:
                pos = m[m > 0]
                floor = float(pos.min()) * 0.5 if pos.size else 1e-12
                ax.fill_between(x, np.clip(m - s, floor, None),
                                np.clip(m + s, floor, None), color=c, alpha=0.2)
            else:
                ax.fill_between(x, m - s, m + s, color=c, alpha=0.2)
        ax.set_xlabel(xlabel); ax.set_ylabel("Mean fraction per track")
        ax.set_title(title)
        if xlim:
            ax.set_xlim(0, xlim)
        if log_y:
            _log_y_setup(ax, drawn)
        elif ylim:
            ax.set_ylim(0, ylim)
        ax.legend(loc="upper right")
        _save(fig, out_png)


def track_length_split_panels(
    split_by_condition: Dict[str, pd.DataFrame], out_png: Path, *,
    title: str, xlabel: str = "Step size (µm)", xlim: Optional[float] = None,
    ylim: Optional[float] = None, log_y: bool = False,
) -> None:
    """One panel per condition; curves colored by track-length stratum.

    All panels share identical x and y axes (``sharex``/``sharey`` plus explicit
    limits) so conditions are directly comparable and no tail is clipped.
    """
    with _PLOT_LOCK:
        apply_house_style()
        conds = [c for c, d in split_by_condition.items() if d is not None and not d.empty]
        if not conds:
            return
        n = len(conds)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False,
                                 sharey=True, sharex=True)
        split_vals: List[np.ndarray] = []
        for ax, cond in zip(axes[0], conds):
            df = split_by_condition[cond]
            strata = list(dict.fromkeys(df["length_bin"]))
            cmap = plt.get_cmap(SEQUENTIAL_CMAP)
            for i, st in enumerate(strata):
                sub = df[df["length_bin"] == st]
                color = cmap(i / max(1, len(strata) - 1))
                yv = sub["fraction"].to_numpy(dtype=float)
                split_vals.append(yv)
                ax.step(sub["bin_center"], _mask_nonpositive(yv) if log_y else yv,
                        where="mid", color=color, linewidth=1.5,
                        label=f"{st} (n={int(sub['n_tracks'].iloc[0])})")
            ax.set_title(cond); ax.set_xlabel(xlabel)
            if xlim:
                ax.set_xlim(0, xlim)
            if log_y:
                _log_y_setup(ax, split_vals)
            elif ylim:
                ax.set_ylim(0, ylim)
            ax.legend(loc="upper right", fontsize=7)
        axes[0][0].set_ylabel("Fraction per bin")
        fig.suptitle(title, y=1.02)
        _save(fig, out_png)


def cross_tau_overlay(
    hist_by_tau: Dict[int, pd.DataFrame], out_png: Path, *,
    condition: str, tau_seconds: Dict[int, float], xlim: Optional[float] = None,
    log_y: bool = False,
) -> None:
    """One condition, tau-colored raw-pooled curves."""
    with _PLOT_LOCK:
        apply_house_style()
        taus = sorted(hist_by_tau.keys())
        cmap = plt.get_cmap(SEQUENTIAL_CMAP)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        drawn = []
        for i, tau in enumerate(taus):
            df = hist_by_tau[tau]
            if df is None or df.empty:
                continue
            color = cmap(i / max(1, len(taus) - 1))
            secs = tau_seconds.get(tau, np.nan)
            yv = df["fraction"].to_numpy(dtype=float)
            drawn.append(yv)
            ax.step(df["bin_center"], _mask_nonpositive(yv) if log_y else yv,
                    where="mid", color=color, linewidth=1.7,
                    label=f"τ = {tau} fr ({secs:.3g} s)")
        ax.set_xlabel("Step size (µm)"); ax.set_ylabel("Fraction per bin")
        ax.set_title(str(condition))
        if xlim:
            ax.set_xlim(0, xlim)
        if log_y:
            _log_y_setup(ax, drawn)
        ax.legend(loc="upper right", title="lag")
        _save(fig, out_png)


# ---------------------------------------------------------------------------
# Diffusion / MSD
# ---------------------------------------------------------------------------

def overlay_d_histograms(
    d_by_condition: Dict[str, np.ndarray], bins: np.ndarray, out_png: Path, *,
    log_scale: bool = False, title: str = "Diffusion coefficient distribution",
    log_y: bool = False,
) -> None:
    """Per-condition D histograms. ``log_scale`` is the x-axis (log10 D);
    ``log_y`` is the fraction axis, which makes the mobile tail legible."""
    with _PLOT_LOCK:
        apply_house_style()
        colors = condition_colors(list(d_by_condition.keys()))
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        drawn = []
        for cond, vals in d_by_condition.items():
            v = np.asarray(vals, dtype=float)
            v = v[np.isfinite(v) & (v > 0)]
            if log_scale:
                v = np.log10(v)
            if v.size == 0:
                continue
            counts, edges = np.histogram(v, bins=bins)
            total = counts.sum()
            frac = counts / total if total else counts
            centers = 0.5 * (edges[:-1] + edges[1:])
            drawn.append(np.asarray(frac, dtype=float))
            ax.step(centers, _mask_nonpositive(frac) if log_y else frac, where="mid",
                    color=colors[cond], linewidth=1.8, label=f"{cond} (n={v.size})")
            if not log_y:
                ax.fill_between(centers, frac, step="mid", color=colors[cond], alpha=0.12)
        ax.set_xlabel("log₁₀ D (µm²/s)" if log_scale else "D (µm²/s)")
        ax.set_ylabel("Fraction per bin"); ax.set_title(title)
        if log_y:
            _log_y_setup(ax, drawn)
        ax.legend(loc="upper right")
        _save(fig, out_png)


def overlay_value_histograms(
    by_condition: Dict[str, np.ndarray], bins: np.ndarray, out_png: Path, *,
    xlabel: str, ylabel: str = "Fraction per bin", log_x: bool = False,
    title: str = "", log_y: bool = False,
) -> None:
    """Pooled per-condition histogram overlay for an arbitrary quantity."""
    import matplotlib.pyplot as plt
    with _PLOT_LOCK:
        apply_house_style()
        colors = condition_colors(list(by_condition.keys()))
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        centres = 0.5 * (np.asarray(bins)[:-1] + np.asarray(bins)[1:])
        drawn = []
        for cond in sorted(by_condition):
            v = np.asarray(by_condition[cond], dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            counts, _ = np.histogram(v, bins=bins)
            total = counts.sum()
            frac = counts / total if total else counts
            drawn.append(np.asarray(frac, dtype=float))
            ax.step(centres, _mask_nonpositive(frac) if log_y else frac, where="mid",
                    color=colors[cond], linewidth=1.8, label=f"{cond} (n={v.size})")
            if not log_y:
                ax.fill_between(centres, frac, step="mid", color=colors[cond], alpha=0.12)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            _log_y_setup(ax, drawn)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.legend(loc="best")
        _save(fig, out_png)


def per_movie_distribution_overlay(
    df: pd.DataFrame, out_png: Path, *, value_col: str, group_col: str,
    movie_col: str = "movie_id", xlabel: str, bins: int = 60,
    log_x: bool = False, title: str = "", xlim: Optional[tuple] = None,
) -> Optional[pd.DataFrame]:
    """One faint curve per movie, one bold curve per condition, per panel.

    The point is heterogeneity: a condition whose movies all lie on top of each
    other means something different from one whose median is the average of two
    populations. Individual movies are drawn unlabelled -- with fifty per
    condition a legend is unreadable -- and identified instead through the
    accompanying per-movie summary table, which is the right medium for
    picking out an outlier by name.

    Returns the per-movie summary, or None if there was nothing to draw.
    """
    import matplotlib.pyplot as plt
    if df is None or df.empty or value_col not in df.columns:
        return None
    d = df.copy()
    d["_v"] = pd.to_numeric(d[value_col], errors="coerce")
    d = d[np.isfinite(d["_v"])]
    if log_x:
        d = d[d["_v"] > 0]
    if d.empty:
        return None

    groups = sorted(d[group_col].astype(str).unique())
    colors = condition_colors(groups)
    lo, hi = (xlim if xlim else (float(np.nanpercentile(d["_v"], 0.2)),
                                 float(np.nanpercentile(d["_v"], 99.8))))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    edges = (np.logspace(np.log10(max(lo, 1e-12)), np.log10(hi), bins + 1)
             if log_x else np.linspace(lo, hi, bins + 1))
    centres = 0.5 * (edges[:-1] + edges[1:])

    with _PLOT_LOCK:
        apply_house_style()
        n = len(groups)
        fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 3.8), squeeze=False,
                                 sharex=True, sharey=True)
        for ax, g in zip(axes[0], groups):
            sub = d[d[group_col].astype(str) == g]
            for _mid, m in sub.groupby(movie_col):
                c, _ = np.histogram(m["_v"], bins=edges)
                if c.sum():
                    ax.step(centres, c / c.sum(), where="mid",
                            color=colors[g], alpha=0.28, linewidth=0.8)
            c, _ = np.histogram(sub["_v"], bins=edges)
            if c.sum():
                ax.step(centres, c / c.sum(), where="mid",
                        color=colors[g], linewidth=2.4)
            ax.set_title(f"{g}  (n = {sub[movie_col].nunique()} movies)")
            ax.set_xlabel(xlabel)
            if log_x:
                ax.set_xscale("log")
        axes[0][0].set_ylabel("Fraction per bin")
        if title:
            fig.suptitle(title, fontsize=12)
        _save(fig, out_png)

    keys = [k for k in (group_col, "experiment", movie_col) if k in d.columns]
    summary = (d.groupby(keys)["_v"]
               .agg(n="size", median="median", mean="mean",
                    p10=lambda v: float(np.percentile(v, 10)),
                    p90=lambda v: float(np.percentile(v, 90)))
               .reset_index().rename(columns={"median": f"median_{value_col}",
                                              "mean": f"mean_{value_col}"}))
    return summary


def origin_aligned_tracks(
    aligned: pd.DataFrame, out_png: Path, *, row_col: Optional[str] = None,
    col_col: str = "condition", max_tracks_per_panel: int = 250,
    lim_um: Optional[float] = None, title: str = "", seed: int = 0,
    row_order: Optional[Sequence] = None,
) -> None:
    """Every track translated to a common origin -- a 'flower plot'.

    With the starting position removed, the only thing left on the page is how
    far and how tortuously each particle wandered. Two populations can then be
    compared directly by eye, with no summary statistic standing in between,
    which is exactly what you want when the summary statistic is the thing in
    question.

    Faceting by a density stratum (``row_col``) against condition (``col_col``)
    puts matched-crowding populations side by side: if the conditions differ in
    the same direction in every row, crowding is not what produced the
    difference.

    All panels share one axis range, because a flower plot with per-panel limits
    compares nothing at all.
    """
    import matplotlib.pyplot as plt
    if aligned is None or aligned.empty:
        return
    need = {"particle", "dx_um", "dy_um", col_col}
    if not need.issubset(aligned.columns):
        return
    d = aligned.dropna(subset=["dx_um", "dy_um"])
    if d.empty:
        return

    cols = sorted(d[col_col].astype(str).unique())
    if row_col is None or row_col not in d.columns:
        rows = [None]
    elif row_order:
        present = set(d[row_col].astype(str))
        rows = [str(r) for r in row_order if str(r) in present]
    else:
        rows = sorted(d[row_col].astype(str).unique())
    colors = condition_colors(cols)
    if lim_um is None:
        lim_um = float(np.nanpercentile(np.hypot(d["dx_um"], d["dy_um"]), 99.0))
    lim_um = max(lim_um, 1e-3)
    rng = np.random.default_rng(seed)

    with _PLOT_LOCK:
        apply_house_style()
        fig, axes = plt.subplots(len(rows), len(cols),
                                 figsize=(2.9 * len(cols) + 0.8, 2.9 * len(rows) + 0.8),
                                 squeeze=False, sharex=True, sharey=True)
        for ri, rv in enumerate(rows):
            for ci, cv in enumerate(cols):
                ax = axes[ri][ci]
                sub = d[d[col_col].astype(str) == cv]
                if rv is not None:
                    sub = sub[sub[row_col].astype(str) == rv]
                ids = sub["particle"].unique()
                if len(ids) > max_tracks_per_panel:
                    ids = rng.choice(ids, max_tracks_per_panel, replace=False)
                    sub = sub[sub["particle"].isin(ids)]
                for _pid, g in sub.groupby("particle"):
                    ax.plot(g["dx_um"], g["dy_um"], linewidth=0.5, alpha=0.45,
                            color=colors[cv], solid_capstyle="round")
                ax.set_xlim(-lim_um, lim_um); ax.set_ylim(-lim_um, lim_um)
                ax.set_aspect("equal")
                ax.grid(False)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True); spine.set_color("0.8")
                if ri == 0:
                    ax.set_title(str(cv), fontsize=11)
                if ci == 0 and rv is not None:
                    ax.set_ylabel(str(rv), fontsize=9)
                # Per-panel count: a spaghetti plot cannot be read without
                # knowing how many strands are in it, and the counts differ
                # panel to panel.
                ax.text(0.03, 0.97, f"n={len(ids)}", transform=ax.transAxes,
                        ha="left", va="top", fontsize=7.5, color="0.35")
        # One scale bar, bottom-left panel, instead of tick labels on 12 axes.
        ax = axes[-1][0]
        bar = _nice_scale(lim_um)
        ax.plot([-lim_um * 0.9, -lim_um * 0.9 + bar], [-lim_um * 0.85] * 2,
                color="0.2", linewidth=2.2, solid_capstyle="butt")
        ax.text(-lim_um * 0.9 + bar / 2, -lim_um * 0.78, f"{bar:g} µm",
                ha="center", va="bottom", fontsize=8, color="0.2")
        if title:
            fig.suptitle(title, fontsize=12)
        _save(fig, out_png)


def _nice_scale(lim: float) -> float:
    """A round scale-bar length that is a sensible fraction of the axis."""
    target = lim * 0.5
    for step in (0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10):
        if step >= target:
            return step
    return float(np.round(target, 1))


def cell_map(labels: np.ndarray, out_png: Path, *, bin_px: float,
             tracks: Optional[pd.DataFrame] = None, note: str = "",
             title: str = "") -> None:
    """The segmented cells, with tracks drawn on top.

    A QC figure: segmentation you cannot see is segmentation you cannot trust,
    and the failure modes (one cell split in two, two cells merged, debris
    called a cell) are obvious at a glance and invisible in a table.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    if labels is None or labels.size == 0:
        return
    n = int(labels.max())
    with _PLOT_LOCK:
        apply_house_style()
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        palette = ["#FFFFFF"] + [OKABE_ITO[i % len(OKABE_ITO)] for i in range(max(n, 1))]
        ax.imshow(labels, cmap=ListedColormap(palette), interpolation="nearest",
                  origin="upper", vmin=0, vmax=max(n, 1),
                  extent=(0, labels.shape[1] * bin_px, labels.shape[0] * bin_px, 0),
                  alpha=0.55)
        if tracks is not None and not tracks.empty and \
                {"particle", "x", "y"}.issubset(tracks.columns):
            for _pid, g in tracks.groupby("particle"):
                ax.plot(g["x"], g["y"], linewidth=0.4, alpha=0.6, color="0.15")
        for cid in range(1, n + 1):
            yy, xx = np.where(labels == cid)
            if yy.size:
                ax.text(xx.mean() * bin_px, yy.mean() * bin_px, str(cid),
                        ha="center", va="center", fontsize=13, color="0.1",
                        fontweight="bold")
        ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
        ax.set_title(title or (note if note else f"{n} cell(s)"), fontsize=10)
        ax.grid(False)
        _save(fig, out_png)


def condition_boxplot(
    per_movie: pd.DataFrame, out_png: Path, *, value_col: str, ylabel: str,
    title: str = "", comparisons: Optional[Sequence[tuple]] = None,
    ylim: Optional[tuple] = None, log_scale: bool = False,
    show_experiment_markers: bool = True,
) -> None:
    """Per-condition box plot of a per-movie summary value.

    Box = IQR, whiskers = 10th-90th percentile across movies. Each movie is a
    dot coloured by experiment; each experiment's median is a diamond. Pairwise
    significance, when supplied, is drawn as comparison bars with asterisks --
    no test names, no numeric p-values.

    ``per_movie`` needs ``condition``, ``movie_id``, ``value_col``, and
    optionally ``experiment``.
    """
    import matplotlib.lines as mlines
    with _PLOT_LOCK:
        apply_house_style()
        if per_movie is None or per_movie.empty or value_col not in per_movie.columns:
            return
        d = per_movie.copy()
        d["_v"] = pd.to_numeric(d[value_col], errors="coerce")
        d = d[np.isfinite(d["_v"])]
        if log_scale:
            d = d[d["_v"] > 0]
            d["_v"] = np.log10(d["_v"])
        if d.empty:
            return

        conds = sorted(d["condition"].astype(str).unique())
        has_exp = "experiment" in d.columns
        experiments = sorted(d["experiment"].astype(str).unique()) if has_exp else []
        exp_colors = {e: OKABE_ITO[i % len(OKABE_ITO)] for i, e in enumerate(experiments)}
        positions = np.arange(1, len(conds) + 1)

        bstats = []
        for c in conds:
            v = d.loc[d["condition"].astype(str) == c, "_v"].to_numpy(dtype=float)
            if v.size:
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                wlo, whi = np.percentile(v, [10, 90])
            else:
                q1 = med = q3 = wlo = whi = np.nan
            bstats.append({"med": med, "q1": q1, "q3": q3,
                           "whislo": wlo, "whishi": whi, "fliers": []})

        fig, ax = plt.subplots(figsize=(1.75 * len(conds) + 2.4, 5.0))
        ax.bxp(bstats, positions=positions, widths=0.5, showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor="white", edgecolor="0.35", linewidth=1.4),
               medianprops=dict(color="0.10", linewidth=1.6),
               whiskerprops=dict(color="0.35", linewidth=1.2),
               capprops=dict(color="0.35", linewidth=1.2))

        rng = np.random.default_rng(0)
        for i, c in enumerate(conds):
            csub = d[d["condition"].astype(str) == c]
            for _, row in csub.iterrows():
                exp = str(row["experiment"]) if has_exp else ""
                ax.scatter([positions[i] + rng.uniform(-0.16, 0.16)], [row["_v"]],
                           s=32, alpha=0.55, zorder=3,
                           color=exp_colors.get(exp, "0.35"), edgecolors="none")
            if has_exp and show_experiment_markers:
                groups = list(csub.groupby("experiment"))
                for j, (exp, esub) in enumerate(groups):
                    x = positions[i] + (j - len(groups) / 2 + 0.5) * 0.10
                    ax.scatter([x], [float(np.median(esub["_v"]))], marker="D", s=90,
                               zorder=5, color=exp_colors.get(str(exp), "0.35"),
                               edgecolors="black", linewidths=1.0)

        ax.set_xticks(positions)
        ax.set_xticklabels(conds, rotation=0 if max(len(c) for c in conds) <= 8 else 20,
                           ha="center" if max(len(c) for c in conds) <= 8 else "right")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.4, len(conds) + 0.6)
        if ylim is not None and not log_scale:
            ax.set_ylim(*ylim)
        if title:
            ax.set_title(title)
        ax.grid(axis="x", visible=False)

        if comparisons:
            significance_brackets(ax, comparisons,
                                  {c: float(positions[i]) for i, c in enumerate(conds)})

        handles = []
        if has_exp and len(experiments) > 1:
            handles += [mlines.Line2D([], [], marker="o", linestyle="none",
                                      color=exp_colors[e], markersize=7, label=str(e))
                        for e in experiments]
        if has_exp and show_experiment_markers:
            handles.append(mlines.Line2D([], [], marker="D", linestyle="none", color="0.45",
                                         markersize=8, markeredgecolor="black",
                                         label="experiment median"))
        if handles:
            ax.legend(handles=handles, loc="best", fontsize=8)
        _save(fig, out_png)


def grouped_boxplot(
    df: pd.DataFrame, out_png: Path, *, x_col: str, group_col: str, value_col: str,
    xlabel: str, ylabel: str, x_order: Optional[Sequence] = None,
    group_order: Optional[Sequence] = None, log_scale: bool = False,
    comparisons_within_x: Optional[Dict[Any, Sequence[tuple]]] = None,
    title: str = "",
) -> None:
    """Boxes of ``value_col`` at each ``x_col`` level, side by side per group.

    Used for "D across expression bins, grouped by condition". Significance is
    drawn *within* each x level, comparing groups at matched expression, which
    is the comparison that controls for the covariate rather than confounding
    with it.
    """
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    with _PLOT_LOCK:
        apply_house_style()
        if df is None or df.empty:
            return
        d = df.copy()
        d["_v"] = pd.to_numeric(d[value_col], errors="coerce")
        d = d[np.isfinite(d["_v"])]
        if log_scale:
            d = d[d["_v"] > 0]
            d["_v"] = np.log10(d["_v"])
        if d.empty:
            return
        xs = list(x_order) if x_order is not None else sorted(d[x_col].dropna().unique())
        gs = list(group_order) if group_order is not None else sorted(
            d[group_col].astype(str).unique())
        colors = condition_colors(gs)

        width = 0.8 / max(len(gs), 1)
        # Size by total box count, not the product of the two dimensions: the
        # latter grows the canvas far faster than the content and leaves the
        # axis labels microscopic on an otherwise empty figure.
        n_boxes = len(xs) * max(len(gs), 1)
        fig, ax = plt.subplots(figsize=(min(14.0, max(5.5, 0.72 * n_boxes + 2.2)), 4.8))
        positions: Dict[Any, Dict[str, float]] = {}
        for gi, g in enumerate(gs):
            offs = (gi - len(gs) / 2 + 0.5) * width
            stats_list, pos = [], []
            for xi, xv in enumerate(xs):
                v = d.loc[(d[x_col] == xv) & (d[group_col].astype(str) == g), "_v"].to_numpy()
                if v.size:
                    q1, med, q3 = np.percentile(v, [25, 50, 75])
                    wlo, whi = np.percentile(v, [10, 90])
                else:
                    q1 = med = q3 = wlo = whi = np.nan
                stats_list.append({"med": med, "q1": q1, "q3": q3,
                                   "whislo": wlo, "whishi": whi, "fliers": []})
                p = xi + 1 + offs
                pos.append(p)
                positions.setdefault(xv, {})[g] = p
            ax.bxp(stats_list, positions=pos, widths=width * 0.82, showfliers=False,
                   patch_artist=True,
                   boxprops=dict(facecolor=colors[g], edgecolor="0.3",
                                 linewidth=1.0, alpha=0.75),
                   medianprops=dict(color="0.10", linewidth=1.4),
                   whiskerprops=dict(color="0.35", linewidth=1.0),
                   capprops=dict(color="0.35", linewidth=1.0))

        ax.set_xticks(np.arange(1, len(xs) + 1))
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlim(0.4, len(xs) + 0.6)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.grid(axis="x", visible=False)

        if comparisons_within_x:
            flat, pos_map = [], {}
            for xv, comps in comparisons_within_x.items():
                if xv not in positions:
                    continue
                for a, b, p in comps:
                    if a in positions[xv] and b in positions[xv]:
                        ka, kb = f"{xv}|{a}", f"{xv}|{b}"
                        pos_map[ka] = positions[xv][a]
                        pos_map[kb] = positions[xv][b]
                        flat.append((ka, kb, p))
            significance_brackets(ax, flat, pos_map, max_brackets=len(xs) * 3)

        ax.legend(handles=[mlines.Line2D([], [], marker="s", linestyle="none",
                                         color=colors[g], markersize=9, label=str(g))
                           for g in gs], loc="best")
        _save(fig, out_png)


def scatter_with_trend(
    df: pd.DataFrame, out_png: Path, *, x_col: str, y_col: str, group_col: str,
    xlabel: str, ylabel: str, log_y: bool = False,
    annotations: Optional[Dict[str, str]] = None, title: str = "",
) -> None:
    """Per-track scatter with a per-group monotone trend line.

    ``annotations`` maps group -> a short suffix appended to that group's legend
    entry (used to carry the correlation coefficient and its significance marker
    in the legend, where it belongs, rather than as free text on the axes).

    The trend is a LOWESS-free running median over x-quantiles: robust to the
    heavy tail of D, and it makes no linearity assumption that the underlying
    relationship has not earned.
    """
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    with _PLOT_LOCK:
        apply_house_style()
        if df is None or df.empty:
            return
        d = df.copy()
        d["_x"] = pd.to_numeric(d[x_col], errors="coerce")
        d["_y"] = pd.to_numeric(d[y_col], errors="coerce")
        d = d[np.isfinite(d["_x"]) & np.isfinite(d["_y"])]
        if log_y:
            d = d[d["_y"] > 0]
        if d.empty:
            return
        groups = sorted(d[group_col].astype(str).unique())
        colors = condition_colors(groups)
        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        handles = []
        for g in groups:
            sub = d[d[group_col].astype(str) == g]
            if sub.empty:
                continue
            alpha = min(0.45, max(0.04, 1500.0 / len(sub)))
            ax.scatter(sub["_x"], sub["_y"], s=9, alpha=alpha, color=colors[g],
                       edgecolors="none", zorder=2)
            # running median over deciles of x
            if len(sub) >= 20:
                q = np.linspace(0, 1, 11)
                edges = np.unique(np.quantile(sub["_x"], q))
                if edges.size >= 3:
                    idx = np.digitize(sub["_x"], edges[1:-1])
                    cx, cy = [], []
                    for b in range(edges.size - 1):
                        m = idx == b
                        if m.sum() >= 5:
                            cx.append(float(np.median(sub["_x"][m])))
                            cy.append(float(np.median(sub["_y"][m])))
                    if len(cx) >= 2:
                        ax.plot(cx, cy, "-", color=colors[g], linewidth=2.2, zorder=3)
            lab = g + (f" ({annotations[g]})" if annotations and g in annotations else "")
            handles.append(mlines.Line2D([], [], marker="o", linestyle="-",
                                         color=colors[g], markersize=6, label=lab))
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if handles:
            ax.legend(handles=handles, loc="best")
        _save(fig, out_png)


def d_boxplot(diff_pool: pd.DataFrame, out_png: Path, *, d_col: str = "D_fit_um2_s",
              log_scale: bool = False, title: str = "",
              comparisons: Optional[Sequence[tuple]] = None,
              ylim: Optional[tuple] = None) -> None:
    """Per-condition diffusion-coefficient box plot over per-movie medians."""
    if diff_pool is None or diff_pool.empty or d_col not in diff_pool.columns:
        return
    d = diff_pool.copy()
    d["_v"] = pd.to_numeric(d[d_col], errors="coerce")
    d = d[np.isfinite(d["_v"]) & (d["_v"] > 0)]
    if d.empty:
        return
    keys = [k for k in ("condition", "experiment", "movie_id") if k in d.columns]
    per_movie = d.groupby(keys)["_v"].median().reset_index().rename(columns={"_v": "median_D"})
    condition_boxplot(
        per_movie, out_png, value_col="median_D",
        ylabel="log\u2081\u2080 D (\u00b5m\u00b2/s)" if log_scale else "D (\u00b5m\u00b2/s)",
        title=title, comparisons=comparisons, ylim=ylim, log_scale=log_scale)


def msd_vs_tau_by_condition(
    msd_by_condition: Dict[str, pd.DataFrame], out_png: Path, *, loglog: bool = False,
    title: str = "MSD vs τ by condition",
    n_movies: Optional[Dict[str, int]] = None,
) -> None:
    """Overlay ensemble MSD-vs-τ curves per condition, SEM across movies."""
    with _PLOT_LOCK:
        apply_house_style()
        colors = condition_colors(list(msd_by_condition.keys()))
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        for cond, df in msd_by_condition.items():
            if df is None or df.empty:
                continue
            tau = pd.to_numeric(df["tau_s"], errors="coerce").to_numpy()
            mean = pd.to_numeric(df["msd_mean_um2"], errors="coerce").to_numpy()
            sem = pd.to_numeric(df.get("msd_sem_um2"), errors="coerce").to_numpy()
            ok = np.isfinite(tau) & np.isfinite(mean)
            nmov = (n_movies or {}).get(cond)
            label = f"{cond}" + (f" (n={nmov} movies)" if nmov else "")
            ax.plot(tau[ok], mean[ok], "o-", color=colors[cond], markersize=4, label=label)
            if np.isfinite(sem).any():
                ax.fill_between(tau[ok], (mean - sem)[ok], (mean + sem)[ok],
                                color=colors[cond], alpha=0.2)
        if loglog:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("τ (s)"); ax.set_ylabel("MSD (µm²)"); ax.set_title(title)
        ax.legend(loc="upper left")
        _save(fig, out_png)


def msd_vs_tau(msd_summary: pd.DataFrame, out_png: Path, *, loglog: bool = False,
               title: str = "Ensemble MSD vs τ") -> None:
    with _PLOT_LOCK:
        apply_house_style()
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        tau = pd.to_numeric(msd_summary["tau_s"], errors="coerce").to_numpy()
        emsd = pd.to_numeric(msd_summary.get("emsd_um2"), errors="coerce").to_numpy()
        mean = pd.to_numeric(msd_summary.get("imsd_mean_um2"), errors="coerce").to_numpy()
        ok = np.isfinite(tau)
        ax.plot(tau[ok], emsd[ok], "o-", color=OKABE_ITO[0], label="EMSD", markersize=4)
        ax.plot(tau[ok], mean[ok], "s--", color=OKABE_ITO[1], label="Mean IMSD", markersize=4)
        if loglog:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("τ (s)"); ax.set_ylabel("MSD (µm²)"); ax.set_title(title)
        ax.legend(loc="upper left")
        _save(fig, out_png)


def gmm_diagnostic(values: np.ndarray, model, bics: List[float], out_png: Path, *,
                   xlabel: str = "log₁₀ D (µm²/s)") -> None:
    with _PLOT_LOCK:
        apply_house_style()
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].hist(vals, bins=60, density=True, color="#BBBBBB", alpha=0.7)
        if model is not None:
            xs = np.linspace(vals.min(), vals.max(), 400).reshape(-1, 1)
            means = model.means_.flatten(); stds = np.sqrt(model.covariances_.flatten())
            weights = model.weights_.flatten()
            total = np.zeros_like(xs.flatten())
            for k in range(len(means)):
                comp = weights[k] * np.exp(-0.5 * ((xs.flatten() - means[k]) / stds[k]) ** 2) / (stds[k] * np.sqrt(2 * np.pi))
                total += comp
                axes[0].plot(xs.flatten(), comp, "--", color=OKABE_ITO[k % len(OKABE_ITO)],
                             label=f"comp {k}")
            axes[0].plot(xs.flatten(), total, "-", color="black", linewidth=1.6, label="mixture")
            axes[0].legend(fontsize=8)
        axes[0].set_xlabel(xlabel); axes[0].set_ylabel("Density"); axes[0].set_title("GMM fit")
        axes[1].plot(range(1, len(bics) + 1), bics, "o-", color=OKABE_ITO[3])
        axes[1].set_xlabel("n components"); axes[1].set_ylabel("BIC")
        axes[1].set_title("Model selection (min BIC)")
        _save(fig, out_png)


def histogram(values: np.ndarray, out_png: Path, *, xlabel: str, title: str,
              bins: int = 60, color: str = OKABE_ITO[0]) -> None:
    """Generic single-distribution histogram (per-movie diagnostics)."""
    with _PLOT_LOCK:
        apply_house_style()
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        ax.hist(vals, bins=bins, color=color, alpha=0.8)
        ax.set_xlabel(xlabel); ax.set_ylabel("Count"); ax.set_title(title)
        _save(fig, out_png)
