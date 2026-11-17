"""Publication-style ER-GEM figures, paneled by genotype group.

Panels: FICD-WT / FICD-E234G / WT-untransfected. Within a panel, DMSO (black) vs
Tg (magenta).

* :func:`alpha_vs_logd_joint` -- alpha vs log10(D at tau) joint plot (scatter +
  top/right marginal KDEs + per-treatment median guides + 2D KDE contours).
* :func:`d_vs_ficd_joint` -- the same joint layout with FICD expression (linear
  x) vs log10(D at tau) (y), complemented panels only.
* :func:`d_boxplot` -- box (IQR) / whisker (10-90 pct) of per-movie D, points
  coloured by replicate, per-replicate median diamonds, DMSO-vs-Tg significance.

D is taken "at tau" -- run ingest with ``--tau-ms 100`` so ``D_fit_um2_s`` is
D_100ms.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import gaussian_kde, spearmanr, mannwhitneyu  # noqa: E402

from .crosscorr import load_diffusion_per_track, FICD_VALUE_COL

DMSO_COLOR = "#000000"
TG_COLOR = "#E6007E"
_PANEL_ORDER = ["FICD-WT", "E234G", "WT-untransfected"]
_ANNOT_BOX = dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75",
                  linewidth=0.6, alpha=0.75)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _panel_of(compl: str) -> str:
    c = str(compl).strip()
    return c if c else "WT-untransfected"


def _treat_norm(t: str) -> str:
    u = str(t).strip().upper()
    if u.startswith("TG"):
        return "Tg"
    if "DMSO" in u or "NT" in u:
        return "DMSO"
    return str(t)


def _color(treat: str) -> str:
    return TG_COLOR if treat == "Tg" else DMSO_COLOR


def assemble(manifest_df: pd.DataFrame, d_col: str = "D_fit_um2_s", n_jobs: int = 8) -> pd.DataFrame:
    df = load_diffusion_per_track(manifest_df, d_col=d_col, extra_cols=["alpha"], n_jobs=n_jobs)
    if df.empty:
        return df
    df["panel"] = [_panel_of(c) for c in df.get("complementation", "")]
    df["treatment"] = df["treatment_code"].map(_treat_norm)
    d = pd.to_numeric(df[d_col], errors="coerce")
    df["D_lin"] = d
    df["log10_D"] = np.log10(d.where(d > 0))
    if "alpha" in df.columns:
        df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    return df


def _panels_present(df: pd.DataFrame) -> List[str]:
    present = [p for p in _PANEL_ORDER if p in set(df["panel"])]
    present += [p for p in sorted(set(df["panel"])) if p not in present]
    return present


# ---------------------------------------------------------------------------
# Shared styling helpers
# ---------------------------------------------------------------------------

def _styled_legend(ax, treatments: List[str], counts: Dict[str, int], loc="upper left"):
    """Legend with large coloured markers, coloured label text, boxed background."""
    handles, labels = [], []
    for t in treatments:
        handles.append(Line2D([0], [0], marker="o", linestyle="None", markersize=11,
                              markerfacecolor=_color(t), markeredgecolor=_color(t)))
        n = counts.get(t)
        labels.append(f"{t}  (n={n:,})" if n is not None else t)
    leg = ax.legend(handles, labels, loc=loc, frameon=True, fontsize=9,
                    handletextpad=0.4, borderpad=0.5, labelspacing=0.35)
    fr = leg.get_frame()
    fr.set_facecolor("white"); fr.set_alpha(0.75); fr.set_edgecolor("0.75"); fr.set_linewidth(0.6)
    for txt, t in zip(leg.get_texts(), treatments):
        txt.set_color(_color(t))
    return leg


def _boxed_text(ax, x, y, text, color, ha="left"):
    ax.text(x, y, text, transform=ax.transAxes, color=color, fontsize=11,
            ha=ha, va="center", bbox=_ANNOT_BOX)


# ---------------------------------------------------------------------------
# Generic joint panel figure (used by both joint plots)
# ---------------------------------------------------------------------------

def _joint_figure(df: pd.DataFrame, out_path: Path, xcol: str, ycol: str,
                  xlabel: str, ylabel: str, panels: List[str],
                  annotate: Callable, xlim, ylim,
                  max_points: int = 3000, seed: int = 0) -> Path:
    n = len(panels)
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(6.3 * n, 5.6))
    gs = fig.add_gridspec(2, 2 * n, width_ratios=[4, 1] * n, height_ratios=[1, 4],
                          wspace=0.05, hspace=0.05)
    xs = np.linspace(xlim[0], xlim[1], 200)
    ys = np.linspace(ylim[0], ylim[1], 200)
    for i, panel in enumerate(panels):
        ax = fig.add_subplot(gs[1, 2 * i])
        axt = fig.add_subplot(gs[0, 2 * i], sharex=ax)
        axr = fig.add_subplot(gs[1, 2 * i + 1], sharey=ax)
        sub = df[df["panel"] == panel]
        counts = {}
        for treat in ["DMSO", "Tg"]:
            t = sub[sub["treatment"] == treat]
            xv = pd.to_numeric(t[xcol], errors="coerce").to_numpy()
            yv = pd.to_numeric(t[ycol], errors="coerce").to_numpy()
            ok = np.isfinite(xv) & np.isfinite(yv)
            xv, yv = xv[ok], yv[ok]
            counts[treat] = xv.size
            if xv.size == 0:
                continue
            col = _color(treat)
            if xv.size > max_points:
                idx = rng.choice(xv.size, max_points, replace=False)
                sx, sy = xv[idx], yv[idx]
            else:
                sx, sy = xv, yv
            ax.scatter(sx, sy, s=6, c=col, alpha=0.35, linewidths=0)
            try:
                if xv.size > 50:
                    k = gaussian_kde(np.vstack([xv, yv]))
                    XX, YY = np.meshgrid(xs, ys)
                    ZZ = k(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
                    ax.contour(XX, YY, ZZ, levels=5, colors=col, linewidths=0.6, alpha=0.8)
            except Exception:
                pass
            ax.axvline(np.median(xv), color=col, ls="--", lw=1.4)
            ax.axhline(np.median(yv), color=col, ls="--", lw=1.4)
            try:
                axt.plot(xs, gaussian_kde(xv)(xs), color=col, lw=1.5)
            except Exception:
                pass
            try:
                axr.plot(gaussian_kde(yv)(ys), ys, color=col, lw=1.5)
            except Exception:
                pass
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(ylabel)
        _styled_legend(ax, ["DMSO", "Tg"], counts, loc="upper left")
        annotate(ax, sub)
        axt.set_title(panel, fontsize=13)
        axt.axis("off"); axr.axis("off")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(Path(out_path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def alpha_vs_logd_joint(df: pd.DataFrame, out_path: Path, tau_ms: float = 100.0,
                        xlim=(-3, 2), ylim=(0, 2)) -> Path:
    panels = _panels_present(df)

    def annotate(ax, sub):
        med = sub.groupby("treatment").agg(a=("alpha", "median"), d=("D_lin", "median"))
        y0 = 0.66
        for treat in ["DMSO", "Tg"]:
            if treat in med.index:
                _boxed_text(ax, 0.03, y0, rf"$\alpha$ = {med.loc[treat,'a']:.2f}", _color(treat))
                y0 -= 0.085
        yd = 0.155
        for treat in ["DMSO", "Tg"]:
            if treat in med.index:
                _boxed_text(ax, 0.97, yd, f"D = {med.loc[treat,'d']:.2f}", _color(treat), ha="right")
                yd -= 0.085

    return _joint_figure(
        df, out_path, xcol="log10_D", ycol="alpha",
        xlabel=r"$\log_{10}(D_{%d\,ms})$   [$\mu m^2\,s^{-1}$]" % int(round(tau_ms)),
        ylabel=r"$\alpha$", panels=panels, annotate=annotate, xlim=xlim, ylim=ylim)


def d_vs_ficd_joint(cross_per_track: pd.DataFrame, out_path: Path, tau_ms: float = 100.0,
                    d_col: str = "D_fit_um2_s", ylim=(-3, 2)) -> Optional[Path]:
    if cross_per_track is None or cross_per_track.empty or FICD_VALUE_COL not in cross_per_track.columns:
        return None
    df = cross_per_track.copy()
    df["panel"] = [_panel_of(c) for c in df.get("complementation", "")]
    df["treatment"] = df["treatment_code"].map(_treat_norm)
    d = pd.to_numeric(df[d_col], errors="coerce")
    df["log10_D"] = np.log10(d.where(d > 0))
    df["ficd"] = pd.to_numeric(df[FICD_VALUE_COL], errors="coerce")
    panels = [p for p in _panels_present(df) if p != "WT-untransfected"]
    if not panels:
        return None
    fv = df["ficd"].to_numpy()
    fv = fv[np.isfinite(fv)]
    if fv.size == 0:
        return None
    xlim = (max(0.0, float(np.nanpercentile(fv, 0.5))), float(np.nanpercentile(fv, 99.5)))

    def annotate(ax, sub):
        yd = 0.155
        for treat in ["DMSO", "Tg"]:
            t = sub[sub["treatment"] == treat].dropna(subset=["ficd", "log10_D"])
            if len(t) >= 3:
                rho, p = spearmanr(t["ficd"], t["log10_D"])
                _boxed_text(ax, 0.97, yd, rf"$\rho$ = {rho:.2f}", _color(treat), ha="right")
                yd -= 0.085

    return _joint_figure(
        df, out_path, xcol="ficd", ycol="log10_D",
        xlabel="FICD expression  [a.u.]",
        ylabel=r"$\log_{10}(D_{%d\,ms})$   [$\mu m^2\,s^{-1}$]" % int(round(tau_ms)),
        panels=panels, annotate=annotate, xlim=xlim, ylim=ylim)


# ---------------------------------------------------------------------------
# D box/whisker plot (points coloured by replicate)
# ---------------------------------------------------------------------------

def _sig_stars(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "ns"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def d_boxplot(df: pd.DataFrame, out_path: Path, tau_ms: float = 100.0) -> Path:
    panels = _panels_present(df)
    # per-movie D (median log10 D within a movie) and per-experiment medians
    per_movie = (df.groupby(["panel", "treatment", "experiment", "movie_id"])["log10_D"]
                 .median().reset_index())
    per_exp = (df.groupby(["panel", "treatment", "experiment"])["log10_D"]
               .median().reset_index())
    experiments = sorted(df["experiment"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    exp_color = {e: cmap(i % 10) for i, e in enumerate(experiments)}

    fig, ax = plt.subplots(figsize=(2.2 * len(panels) + 2.5, 5.2))
    box_w = 0.5
    gap = 0.9
    xtick_pos, xtick_lab, panel_centers = [], [], []
    rngj = np.random.default_rng(1)
    for i, panel in enumerate(panels):
        base = i * (2 * gap + 1.0)
        pos = {"DMSO": base, "Tg": base + gap}
        panel_centers.append((panel, (pos["DMSO"] + pos["Tg"]) / 2))
        stat_vals = {}
        top_here = -np.inf
        for treat in ["DMSO", "Tg"]:
            pm = per_movie[(per_movie["panel"] == panel) & (per_movie["treatment"] == treat)]
            pm = pm.dropna(subset=["log10_D"])   # align points, colours, and box stats
            vals = pm["log10_D"].to_numpy()
            if vals.size == 0:
                continue
            col = _color(treat)
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            wlo, whi = np.percentile(vals, [10, 90])
            bx = ax.bxp([{"med": med, "q1": q1, "q3": q3, "whislo": wlo, "whishi": whi,
                          "fliers": []}], positions=[pos[treat]], widths=box_w,
                        showfliers=False, patch_artist=True)
            for patch in bx["boxes"]:
                patch.set_facecolor("white"); patch.set_edgecolor(col); patch.set_linewidth(1.6)
            for key in ("whiskers", "caps", "medians"):
                for ln in bx[key]:
                    ln.set_color(col); ln.set_linewidth(1.6)
            # individual per-movie points, coloured by replicate
            jit = rngj.normal(0, 0.055, vals.size)
            ax.scatter(np.full(vals.size, pos[treat]) + jit, vals, s=16,
                       c=[exp_color.get(e) for e in pm["experiment"]],
                       alpha=0.8, linewidths=0, zorder=3)
            # per-replicate median diamonds
            pe = per_exp[(per_exp["panel"] == panel) & (per_exp["treatment"] == treat)]
            evals = pe["log10_D"].dropna().to_numpy()
            ax.scatter(np.full(evals.size, pos[treat]), evals, marker="D", s=55,
                       facecolor="white", edgecolor="black", linewidth=1.1, zorder=4)
            stat_vals[treat] = pe["log10_D"].dropna().to_numpy()
            top_here = max(top_here, whi, evals.max() if evals.size else whi)
            xtick_pos.append(pos[treat]); xtick_lab.append(treat)
        # DMSO vs Tg significance (Mann-Whitney on per-experiment medians)
        if "DMSO" in stat_vals and "Tg" in stat_vals \
                and stat_vals["DMSO"].size >= 1 and stat_vals["Tg"].size >= 1 \
                and (stat_vals["DMSO"].size + stat_vals["Tg"].size) >= 3:
            try:
                _, p = mannwhitneyu(stat_vals["DMSO"], stat_vals["Tg"], alternative="two-sided")
            except ValueError:
                p = np.nan
            y = top_here + 0.18
            ax.plot([pos["DMSO"], pos["DMSO"], pos["Tg"], pos["Tg"]],
                    [y, y + 0.08, y + 0.08, y], color="0.2", lw=1.2)
            ax.text((pos["DMSO"] + pos["Tg"]) / 2, y + 0.10, _sig_stars(p),
                    ha="center", va="bottom", fontsize=12)
        if i < len(panels) - 1:
            ax.axvline(base + 2 * gap - 0.15, color="0.85", ls="--", lw=1)
    ax.set_xticks(xtick_pos); ax.set_xticklabels(xtick_lab)
    for panel, cx in panel_centers:
        ax.text(cx, 1.02, panel, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"$\log_{10}(D_{%d\,ms})$   [$\mu m^2\,s^{-1}$]" % int(round(tau_ms)))
    # replicate colour legend + marker key
    rep_handles = [Line2D([0], [0], marker="o", linestyle="None", markersize=7,
                          markerfacecolor=exp_color[e], markeredgecolor="none", label=str(e))
                   for e in experiments]
    rep_handles.append(Line2D([0], [0], marker="D", linestyle="None", markersize=8,
                              markerfacecolor="white", markeredgecolor="black",
                              label="replicate median"))
    leg = ax.legend(handles=rep_handles, title="replicate", frameon=True, fontsize=8,
                    loc="upper left", bbox_to_anchor=(1.01, 1.0))
    leg.get_frame().set_alpha(0.85)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(Path(out_path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(manifest_df: pd.DataFrame, out_dir: Path, d_col: str = "D_fit_um2_s",
        tau_ms: float = 100.0, crosscorr_csv: Optional[Path] = None, n_jobs: int = 8) -> Path:
    out_dir = Path(out_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = assemble(manifest_df, d_col=d_col, n_jobs=n_jobs)
    if not df.empty:
        alpha_vs_logd_joint(df, out_dir / "alpha_vs_logD_joint.png", tau_ms=tau_ms)
        d_boxplot(df, out_dir / "D_boxplot_by_genotype.png", tau_ms=tau_ms)
    made = []
    if crosscorr_csv is not None and Path(crosscorr_csv).exists():
        ct = pd.read_csv(crosscorr_csv)
        if not ct.empty:
            r = d_vs_ficd_joint(ct, out_dir / "D_vs_ficd_expression.png", tau_ms=tau_ms, d_col=d_col)
            if r is not None:
                made.append(r.name)
        # per-cell figure (pseudoreplication-safe): D_cell_median vs FICD
        cell_csv = Path(crosscorr_csv).with_name("Dcell_vs_ficd_expression.csv")
        if cell_csv.exists():
            cc = pd.read_csv(cell_csv)
            if not cc.empty and "D_cell_median_um2_s" in cc.columns:
                r = d_vs_ficd_joint(cc, out_dir / "Dcell_vs_ficd_expression.png",
                                    tau_ms=tau_ms, d_col="D_cell_median_um2_s")
                if r is not None:
                    made.append(r.name)
    return out_dir
