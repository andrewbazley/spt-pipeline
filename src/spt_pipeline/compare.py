"""Phase C: condition-level comparison (lightweight, fast re-run).

Reads only the cached Phase B tau-step CSVs and Phase A diffusion CSVs -- no
tracks-file I/O -- so re-running with different bin widths, condition
groupings, or QC thresholds is fast. This function replaces the ~1500-line
``cmd_analyze_compare_conditions`` monolith.

For each requested tau it produces the four histogram representations, the
cross-tau overlay, per-experiment summaries and between-condition statistics,
plus the diffusion and MSD analyses, all under a single output tree.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import bins as bins_mod
from . import (diffusion, filters as filters_mod, gmm, histograms, plotting,
               statistics, track_metrics, utils)
from .tau_steps import tau_filename

_TAU_RE = re.compile(r"_tau_steps_(\d+)\.csv$")

#: Independently runnable arms of the comparison. Each reads only cached
#: per-movie outputs, so any subset can be re-run on its own in seconds.
ARMS = ("steps", "diffusion", "msd", "marker", "density", "advanced")

#: Spellings accepted for --run-only.
_ARM_ALIASES = {
    "steps": "steps", "step": "steps", "step-size": "steps", "step_size": "steps",
    "stepsize": "steps", "tau": "steps", "tau-steps": "steps",
    "diffusion": "diffusion", "d": "diffusion", "diff": "diffusion",
    "msd": "msd",
    "marker": "marker", "expression": "marker", "expression-marker": "marker",
    "ficd": "marker", "intensity": "marker",
    "density": "density", "crowding": "density", "qc-density": "density",
    "cells": "density",
    "advanced": "advanced", "gem": "advanced",
    "all": "__all__",
}


class UnknownArmError(ValueError):
    """Raised for an unrecognized --run-only name."""


def resolve_arms(names) -> frozenset:
    """Map --run-only spellings to canonical arm names; None -> everything.

    An unrecognized name raises rather than being dropped: silently running
    four arms when five were asked for is the kind of thing you notice a week
    later, from a missing figure.
    """
    if not names:
        return frozenset(ARMS)
    out, unknown = set(), []
    for raw in names:
        key = str(raw).strip().lower().replace(" ", "-").replace("_", "-")
        canon = _ARM_ALIASES.get(key) or _ARM_ALIASES.get(key.replace("-", "_"))
        if canon == "__all__":
            return frozenset(ARMS)
        if canon is None:
            unknown.append(raw)
        else:
            out.add(canon)
    if unknown:
        raise UnknownArmError(
            f"Unknown analysis arm(s): {', '.join(map(str, unknown))}. "
            f"Choices (aliases accepted): {', '.join(ARMS)}.")
    return frozenset(out or ARMS)


@dataclass
class CompareConfig:
    taus: Optional[List[int]] = None            # None -> auto-discover
    mpp: float = 0.183
    fps: float = 50.0
    min_nspots: int = 0
    min_alpha: Optional[float] = None
    subtract_loc_error: bool = False
    loc_error_alpha_max: float = 0.3
    loc_error_min_nspots: int = 10
    confinement_threshold_um: Optional[float] = None
    track_length_bins: Optional[List[int]] = None
    step_bin_width_um: object = "auto"
    diffusion_bin_width: object = "auto"
    logd_bin_width: object = "auto"
    #: Fraction-axis scale for the step-size and D histograms: "linear",
    #: "log", or "both" (emits a *_logy.png companion). Log makes the
    #: heavy tail -- where the mobile population is -- legible.
    hist_yscale: str = "linear"
    skip_gmm: bool = True          # GMM fitting is opt-in
    max_gmm_k: int = 5
    percentile_xlim: float = 99.5
    msd_max_lag: Optional[int] = None       # frames; None -> auto by support
    msd_min_tracks: int = 20                # min supporting tracks/condition for auto max-lag
    # Advanced suite (VACF / CFBM / angles / EB / alpha_2 / rheology), opt-in.
    advanced: bool = False
    advanced_select: Optional[frozenset] = None   # None -> all
    # None -> inherit from the tracking context (min_track_length / max_lagtime).
    advanced_min_track_len: Optional[int] = None
    advanced_max_tlag: Optional[int] = None
    advanced_particle_radius_um: float = 0.020
    advanced_particle_radius_explicit: bool = True
    advanced_temperature_k: float = 298.15
    advanced_n_jobs: int = 4
    # Expression-marker (FICD) analysis. Runs automatically when per-track
    # marker intensities exist; these only tune it.
    marker_intensity_bins: int = 4
    marker_min_disk_coverage: float = 0.0    # drop tracks whose disk overhung the frame
    marker_edge_margin_px: float = 0.0       # drop tracks within N px of any edge
    #: Absolute brightness bin edges for matched comparison (None -> no
    #: brightness stratification). Absolute, not quantile: matching requires
    #: the same boundaries in every condition.
    brightness_bin_edges: Optional[List[float]] = None
    brightness_bins: Optional[int] = None    # quantile fallback, if no edges
    #: Track filtering, applied once to the shared per-track population.
    track_filter: Optional[object] = None
    #: Names a filtered pass so successive settings sit side by side rather
    #: than overwriting one another.
    run_label: str = ""
    #: Crowding / density QC. Runs automatically when density data exists.
    density_strata: int = 4                  # quantile strata for link safety
    density_match: bool = False              # also produce one balanced subset
    density_match_column: str = "track_median_field_localizations_n"
    density_match_length_bins: Optional[List[int]] = None
    density_match_seed: int = 1729
    #: Which arms to run. None -> all of them (advanced still gated by
    #: ``advanced``). Lets the marker analysis be re-run without redoing the
    #: step-size, diffusion and MSD work.
    arms: Optional[frozenset] = None
    # movie_ids excluded upstream because tracking found nothing (for the log).
    excluded_movies: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Plot helpers (shared axes + stat annotations)
# ---------------------------------------------------------------------------

def _yscales(cfg) -> List[tuple]:
    """(log_y, filename suffix) pairs to emit for a histogram."""
    mode = str(getattr(cfg, "hist_yscale", "linear")).lower()
    if mode == "log":
        return [(True, "")]
    if mode == "both":
        return [(False, ""), (True, "_logy")]
    return [(False, "")]


def _hist_ymax(hist_by_cond: Dict[str, pd.DataFrame], col: str) -> Optional[float]:
    vals = [float(d[col].max()) for d in hist_by_cond.values()
            if d is not None and not d.empty and col in d.columns]
    return (max(vals) * 1.08) if vals else None


def _tw_ymax(tw_by_cond: Dict[str, pd.DataFrame]) -> Optional[float]:
    vals = []
    for d in tw_by_cond.values():
        if d is None or d.empty:
            continue
        vals.append(float((d["mean_fraction_per_track"] + d["sem_fraction_per_track"]).max()))
    return (max(vals) * 1.08) if vals else None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_tau_files(out_dir: Path) -> Dict[int, List[Path]]:
    """Map tau_frames -> list of tau-step CSV paths found under out_dir."""
    found: Dict[int, List[Path]] = {}
    for p in utils.rglob_filtered(out_dir, "*_tau_steps_*.csv"):
        m = _TAU_RE.search(p.name)
        if m:
            found.setdefault(int(m.group(1)), []).append(p)
    return found


def _movie_id_from_tau_file(path: Path) -> str:
    return _TAU_RE.sub("", path.name)


def _label_map(manifest_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    out = {}
    for _, row in manifest_df.iterrows():
        out[str(row["movie_id"])] = {"condition": str(row["condition"]),
                                     "experiment": str(row["experiment"])}
    return out


def _load_tau_pool(files: Sequence[Path], labels: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Concatenate tau-step CSVs, attaching movie_id/condition/experiment.

    Only the columns the comparison actually uses (``particle`` and ``step_um``)
    are read, so the pooled table stays small even when tens of millions of
    displacements are combined across hundreds of movies.
    """
    usecols = ["particle", "step_um"]
    frames = []
    for p in files:
        mid = _movie_id_from_tau_file(p)
        meta = labels.get(mid)
        if meta is None:
            continue
        try:
            df = pd.read_csv(p, usecols=lambda c: c in usecols)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        if df.empty or "step_um" not in df.columns:
            continue
        df["movie_id"] = mid
        df["condition"] = meta["condition"]
        df["experiment"] = meta["experiment"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_diffusion_pool(manifest_df: pd.DataFrame, cfg: CompareConfig) -> pd.DataFrame:
    """The shared per-track population every analysis works from.

    Built from the cached ``*_track_metrics.csv`` tables, so it already carries
    mobility (D, path speed, median step per tau), quality (alpha, n_spots),
    brightness, and marker intensity together. Because one table feeds every
    stage, a filter applied here applies everywhere, and no two figures in a
    results folder can silently describe different tracks.
    """
    pool = track_metrics.load_pool(manifest_df, default_fps=cfg.fps)
    if pool.empty:
        return pool
    # Legacy scalar filters, kept so existing commands behave unchanged.
    if cfg.min_nspots > 0 and "n_spots" in pool.columns:
        pool = pool[pd.to_numeric(pool["n_spots"], errors="coerce").fillna(0) >= cfg.min_nspots]
    if cfg.min_alpha is not None and "alpha" in pool.columns:
        pool = pool[pd.to_numeric(pool["alpha"], errors="coerce").fillna(-np.inf) >= cfg.min_alpha]
    if cfg.subtract_loc_error:
        pool = _apply_loc_error(pool, cfg)
    return pool


def _apply_loc_error(pool: pd.DataFrame, cfg: CompareConfig) -> pd.DataFrame:
    """Subtract the localization-error floor, per movie, and say what happened.

    The floor is estimated from quasi-immobile particles (low alpha, enough
    localizations) whose short-lag MSD is essentially ``4*sigma_loc^2``. It is
    estimated per movie because the localization floor is an optical property
    of that acquisition, not a global constant.

    This needs ``msd_fit_um2`` and ``tau_fit_s`` from the diffusion table.
    Without them the correction silently degrades to a no-op copy, which is
    worse than not offering it -- so the absence is reported as an error rather
    than left for the user to infer from an unchanged figure.
    """
    need = {"msd_fit_um2", "tau_fit_s"}
    missing = sorted(need - set(pool.columns))
    if missing:
        print(f"  --subtract-loc-error: NOT APPLIED. The diffusion tables lack "
              f"{', '.join(missing)}, which the floor estimate needs. Re-run "
              f"'track trackpy' (or delete the *_track_metrics.csv caches and "
              f"re-run) to regenerate them.")
        return pool
    parts, rows = [], []
    for mid, g in pool.groupby("movie_id", sort=False):
        n_used = int(((pd.to_numeric(g.get("alpha"), errors="coerce") <= cfg.loc_error_alpha_max)
                      & (pd.to_numeric(g.get("n_spots"), errors="coerce")
                         >= cfg.loc_error_min_nspots)).sum())
        floor = diffusion.estimate_loc_error_floor(
            g, cfg.loc_error_alpha_max, cfg.loc_error_min_nspots)
        parts.append(diffusion.apply_loc_error_subtraction(g, floor))
        rows.append({"movie_id": mid, "msd_floor_um2": floor,
                     "sigma_loc_nm": (1000.0 * np.sqrt(floor / 4.0)) if floor > 0 else np.nan,
                     "n_immobile_used": n_used, "n_tracks": int(len(g))})
    out = pd.concat(parts, ignore_index=True)
    rep = pd.DataFrame(rows)
    out.attrs["loc_error_report"] = rep
    est = rep[rep["msd_floor_um2"] > 0]
    if est.empty:
        print(f"  --subtract-loc-error: no movie yielded a floor "
              f"(no tracks with alpha <= {cfg.loc_error_alpha_max} and "
              f">= {cfg.loc_error_min_nspots} spots). D is unchanged.")
    else:
        d0 = pd.to_numeric(out["D_fit_um2_s"], errors="coerce")
        d1 = pd.to_numeric(out["D_fit_um2_s_corr"], errors="coerce")
        print(f"  --subtract-loc-error: floor estimated for {len(est)}/{len(rep)} movies, "
              f"median sigma_loc = {est['sigma_loc_nm'].median():.0f} nm; "
              f"median D {d0.median():.4g} -> {d1.median():.4g} um^2/s")
        # The floor is only meaningful if the population it came from really was
        # immobile. If it was not, the subtraction eats the whole distribution
        # and every slow track collapses to exactly zero -- which looks like a
        # result and is an artifact.
        zeroed = float((d1 <= 0).mean())
        if zeroed > 0.4:
            print(f"  WARNING: {zeroed:.0%} of tracks were driven to D = 0. The "
                  f"alpha <= {cfg.loc_error_alpha_max} population used to estimate the "
                  f"floor is probably not immobile, so the floor is too high. "
                  f"Lower --loc-error-alpha-max, raise --loc-error-min-nspots, or "
                  f"check loc_error_floor_by_movie.csv against your known "
                  f"localization precision.")
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compare_conditions(manifest_df: pd.DataFrame, out_dir: Path, cfg: CompareConfig) -> Path:
    """Run the full Phase C comparison; returns the compare_conditions dir."""
    out_dir = Path(out_dir)
    # A labelled pass gets its own subtree, so you can filter, look, re-filter
    # and still have the earlier result to compare against.
    root = out_dir / "compare_conditions"
    if cfg.run_label:
        root = root / utils_safe(cfg.run_label)
    root.mkdir(parents=True, exist_ok=True)
    labels = _label_map(manifest_df)

    # Choose the statistical unit from the data, once, up front: every test,
    # figure annotation and CSV below reads from this single decision.
    design = statistics.infer_design(manifest_df)
    print(f"Statistical design: {design.short_label} — {design.reason}")
    if design.is_within_experiment:
        print("  Between-condition tests use per-movie medians (technical replicates). "
              "Track-level distributional tests are written to track_level_ks_tests.csv.")
    utils.dump_json(design.to_dict(), root / "statistical_design.json")

    binset = bins_mod.BinSettings(
        mpp=cfg.mpp, fps=cfg.fps, step_bin_width_um=cfg.step_bin_width_um,
        diffusion_bin_width=cfg.diffusion_bin_width, logd_bin_width=cfg.logd_bin_width)
    utils.dump_json(binset.to_dict(), root / "histogram_bin_settings.json")
    utils.dump_json(_config_dict(cfg), root / "run_parameters.json")

    # ---- one filtered track population, shared by every analysis below ----
    diff_pool = _load_diffusion_pool(manifest_df, cfg)
    keep_particles: Optional[set] = None
    tf = cfg.track_filter
    if tf is not None and bool(tf) and not diff_pool.empty:
        before = diff_pool
        diff_pool, report = tf.apply(diff_pool)
        report.to_csv(root / "filter_report.csv", index=False)
        attr = filters_mod.per_condition_attrition(before, diff_pool)
        if not attr.empty:
            attr.to_csv(root / "filter_attrition_by_condition.csv", index=False)
        print(f"Track filter: {len(before)} -> {len(diff_pool)} tracks "
              f"({len(diff_pool) / max(len(before), 1):.1%} kept)")
        for name, win in tf.describe().items():
            print(f"    {name} {win}")
        if not attr.empty:
            spread = attr["frac_kept"].max() - attr["frac_kept"].min()
            if spread > 0.25:
                # Uneven attrition turns a treatment comparison into a
                # selection-effect comparison; say so rather than bury it.
                print(f"  WARNING: attrition differs by {spread:.0%} across "
                      f"conditions -- see filter_attrition_by_condition.csv")
        # Downstream step-size tables are per displacement, not per track, so
        # they are restricted by (movie_id, particle) membership.
        keep_particles = set(zip(diff_pool["movie_id"].astype(str),
                                 diff_pool["particle"].map(_pkey)))
        if diff_pool.empty:
            print("  No tracks survived the filter; nothing to compare.")
            return root

    arms = cfg.arms if cfg.arms else frozenset(ARMS)
    if cfg.arms:
        print(f"Running arms: {', '.join(sorted(arms))} "
              f"(skipping {', '.join(sorted(set(ARMS) - set(arms))) or 'nothing'})")

    available = _discover_tau_files(out_dir) if "steps" in arms else {}
    taus = sorted(cfg.taus) if cfg.taus else sorted(available.keys())
    taus = [t for t in taus if t in available]

    # ---- step-size analysis per tau ----
    step_root = root / "step_size_analysis"
    cross_pooled: Dict[str, Dict[int, pd.DataFrame]] = {}
    attrition_rows: List[dict] = []
    all_stat_frames: List[pd.DataFrame] = []
    track_stat_frames: List[pd.DataFrame] = []

    for tau in taus:
        pool = _load_tau_pool(available[tau], labels)
        if pool.empty:
            continue
        tau_dir = step_root / f"tau_{tau:03d}"
        tau_dir.mkdir(parents=True, exist_ok=True)
        if keep_particles is not None:
            mask = [ (m, _pkey(p)) in keep_particles
                     for m, p in zip(pool["movie_id"].astype(str), pool["particle"]) ]
            pool = pool[np.asarray(mask)]
            if pool.empty:
                continue

        # Shared x-limit = the longest tail across all conditions at this tau, so
        # no displacement data is dropped from the histograms (np.histogram would
        # otherwise silently discard values beyond the last bin edge) and all
        # conditions/representations share an identical, comparable x-axis.
        all_steps = pool["step_um"].to_numpy(dtype=float)
        all_steps = all_steps[np.isfinite(all_steps)]
        xmax = float(all_steps.max()) if all_steps.size else binset.step_width
        # small margin so the final bar/tail is fully visible
        xlim = xmax * 1.02
        step_bins = bins_mod.make_bins(0.0, xlim, binset.step_width)

        raw_by_cond: Dict[str, pd.DataFrame] = {}
        tw_by_cond: Dict[str, pd.DataFrame] = {}
        tmean_by_cond: Dict[str, pd.DataFrame] = {}
        split_by_cond: Dict[str, pd.DataFrame] = {}
        legend_meta: Dict[str, str] = {}

        raw_rows, tw_rows, tmean_rows, split_rows = [], [], [], []
        for cond, sub in pool.groupby("condition"):
            raw = histograms.raw_pooled_histogram(sub["step_um"].to_numpy(), step_bins, cond)
            tw = histograms.track_weighted_histogram(sub, step_bins, cond)
            tmean = histograms.track_mean_histogram(sub, step_bins, cond)
            split = histograms.track_length_split_histogram(
                sub, step_bins, cond, length_bins=cfg.track_length_bins)
            raw_by_cond[cond] = raw; tw_by_cond[cond] = tw
            tmean_by_cond[cond] = tmean; split_by_cond[cond] = split
            raw_rows.append(raw); tw_rows.append(tw); tmean_rows.append(tmean); split_rows.append(split)

            n_tracks = sub.groupby(["movie_id", "particle"]).ngroups
            n_disp = len(sub)
            med = float(np.nanmedian(sub["step_um"].to_numpy(dtype=float)))
            legend_meta[cond] = f"n={n_tracks}"

            # attrition: total tracks in condition come from diffusion/track presence;
            # approximate with distinct particles observed at tau=1 if available.
            attrition_rows.append(histograms.attrition_row(
                sub, n_tracks_total=n_tracks, tau_frames=tau,
                tau_s=tau / cfg.fps, condition=cond))
            cross_pooled.setdefault(cond, {})[tau] = raw

        # write CSVs
        pd.concat(raw_rows, ignore_index=True).to_csv(tau_dir / "raw_pooled_step_hist_bins.csv", index=False)
        pd.concat(tw_rows, ignore_index=True).to_csv(tau_dir / "track_weighted_step_hist_bins.csv", index=False)
        pd.concat(tmean_rows, ignore_index=True).to_csv(tau_dir / "track_mean_step_hist_bins.csv", index=False)
        pd.concat(split_rows, ignore_index=True).to_csv(tau_dir / "track_length_split_step_hist_bins.csv", index=False)

        # per-experiment summary + between-condition statistics (computed first so
        # they can be annotated on the plots below)
        per_unit = statistics.per_unit_step_summary(
            pool, design.unit_column, confinement_threshold_um=cfg.confinement_threshold_um)
        per_unit.to_csv(tau_dir / f"per_{design.unit}_summary.csv", index=False)
        # The per-experiment table is always written too: even when the
        # experiment is not the unit of inference, it is the right table for
        # eyeballing day-to-day variation.
        if design.unit_column != "experiment":
            statistics.per_unit_step_summary(
                pool, "experiment", confinement_threshold_um=cfg.confinement_threshold_um
            ).to_csv(tau_dir / "per_experiment_summary.csv", index=False)
        pooled_arrays = {c: pd.to_numeric(s["step_um"], errors="coerce").dropna().to_numpy()
                         for c, s in pool.groupby("condition")}
        stat_df = statistics.between_condition_tests(
            per_unit, "median_step_um", pooled=pooled_arrays,
            label=f"step_um@tau{tau}", design=design)
        if not stat_df.empty:
            stat_df["tau_frames"] = tau
            all_stat_frames.append(stat_df)
        # Track-level distributional tests: computed always, plotted never.
        track_med = (pool.groupby(["condition", "movie_id", "particle"])["step_um"]
                     .median().reset_index())
        track_pooled = {c: pd.to_numeric(s["step_um"], errors="coerce").dropna().to_numpy()
                        for c, s in track_med.groupby("condition")}
        tks = statistics.track_level_tests(track_pooled, label=f"median_step_um@tau{tau}")
        if not tks.empty:
            tks["tau_frames"] = tau
            track_stat_frames.append(tks)
        step_comparisons = plotting.comparisons_from_stats(stat_df)

        # shared y-limits so conditions/representations are directly comparable
        raw_ymax = _hist_ymax(raw_by_cond, "fraction")
        tmean_ymax = _hist_ymax(tmean_by_cond, "fraction")
        tw_ymax = _tw_ymax(tw_by_cond)
        split_ymax = max((float(d["fraction"].max()) for d in split_by_cond.values()
                          if d is not None and not d.empty), default=None)
        split_ymax = split_ymax * 1.08 if split_ymax else None

        # titles carry tau in frames AND seconds
        sec = tau / cfg.fps
        tau_lab = f"τ = {tau} frames ({sec:.3g} s)"
        for log_y, sfx in _yscales(cfg):
            plotting.overlay_step_histograms(
                raw_by_cond, tau_dir / f"raw_pooled_step_hist{sfx}.png", title=tau_lab,
                legend_meta=legend_meta, xlim=xlim, ylim=raw_ymax, log_y=log_y)
            plotting.overlay_track_weighted(
                tw_by_cond, tau_dir / f"track_weighted_step_hist{sfx}.png",
                title=tau_lab, xlim=xlim, ylim=tw_ymax, log_y=log_y)
            plotting.overlay_step_histograms(
                tmean_by_cond, tau_dir / f"track_mean_step_hist{sfx}.png",
                title=tau_lab, xlim=xlim, ylim=tmean_ymax, log_y=log_y)
            plotting.track_length_split_panels(
                split_by_cond, tau_dir / f"track_length_split_step_hist{sfx}.png",
                title=tau_lab, xlim=xlim, ylim=split_ymax, log_y=log_y)
        # Categorical companion: the distribution overlays cannot carry pairwise
        # comparison bars sensibly, so the significance lives here instead.
        per_movie_step = statistics.per_unit_step_summary(pool, "movie_id")
        plotting.condition_boxplot(
            per_movie_step, tau_dir / "median_step_boxplot.png",
            value_col="median_step_um", ylabel="Median step (µm)", title=tau_lab,
            comparisons=step_comparisons)

    # cross-tau overlays. Only when the step arm actually produced something --
    # an empty step_size_analysis/ folder in a --run-only marker run reads as a
    # failed analysis rather than a skipped one.
    if cross_pooled:
        cross_dir = step_root / "cross_tau"
        cross_dir.mkdir(parents=True, exist_ok=True)
        tau_seconds = {t: t / cfg.fps for t in taus}
        for cond, per_tau in cross_pooled.items():
            for log_y, sfx in _yscales(cfg):
                plotting.cross_tau_overlay(
                    per_tau, cross_dir / f"{utils_safe(cond)}_multi_tau_overlay{sfx}.png",
                    condition=cond, tau_seconds=tau_seconds, log_y=log_y)
        if attrition_rows:
            pd.DataFrame(attrition_rows).to_csv(cross_dir / "cross_tau_summary.csv",
                                                index=False)

    # ---- diffusion / MSD ----
    if "diffusion" in arms:
        _diffusion_outputs(diff_pool, root, cfg, binset, design)
    if "msd" in arms:
        _msd_outputs(manifest_df, root, cfg, design)

    # ---- top-level statistical tests ----
    d_col = "D_fit_um2_s_corr" if cfg.subtract_loc_error and "D_fit_um2_s_corr" in diff_pool.columns else "D_fit_um2_s"
    if isinstance(diff_pool.attrs.get("loc_error_report"), pd.DataFrame):
        diff_pool.attrs["loc_error_report"].to_csv(
            root / "loc_error_floor_by_movie.csv", index=False)
    if "diffusion" in arms and not diff_pool.empty:
        d_per_unit = statistics.per_unit_d_summary(diff_pool, design.unit_column, d_col=d_col)
        ddir = root / "diffusion_analysis"; ddir.mkdir(parents=True, exist_ok=True)
        d_per_unit.to_csv(ddir / f"D_per_{design.unit}_summary.csv", index=False)
        if design.unit_column != "experiment":
            statistics.per_unit_d_summary(diff_pool, "experiment", d_col=d_col).to_csv(
                ddir / "D_per_experiment_summary.csv", index=False)
        pooled_d = {c: pd.to_numeric(s[d_col], errors="coerce").dropna().to_numpy()
                    for c, s in diff_pool.groupby("condition")} if d_col in diff_pool.columns else None
        d_stats = statistics.between_condition_tests(
            d_per_unit, "median_D", pooled=pooled_d, label="D", design=design)
        if not d_stats.empty:
            all_stat_frames.append(d_stats)
        if pooled_d:
            dtk = statistics.track_level_tests(pooled_d, label="D_per_track")
            if not dtk.empty:
                track_stat_frames.append(dtk)

    # ---- crowding / density QC (automatic when density data is present) ----
    if "density" in arms:
        try:
            all_stat_frames.extend(
                _density_outputs(manifest_df, diff_pool, root, cfg, design, d_col) or [])
        except Exception as exc:
            print(f"  density analysis failed: {type(exc).__name__}: {exc}")

    # ---- expression-marker analysis (automatic when marker data is present) ----
    if "marker" in arms:
        try:
            all_stat_frames.extend(
                _marker_outputs(manifest_df, diff_pool, root, cfg, design, d_col) or [])
        except Exception as exc:
            print(f"  marker analysis failed: {type(exc).__name__}: {exc}")

    # A partial run must not overwrite the full statistical table with a
    # subset of it, so the filename records that it is incomplete.
    partial = set(arms) != set(ARMS)
    stats_name = "statistical_tests_partial.csv" if partial else "statistical_tests.csv"
    if all_stat_frames:
        pd.concat(all_stat_frames, ignore_index=True).to_csv(root / stats_name, index=False)
    if track_stat_frames:
        # Kept in its own file, never annotated on a figure: n = tracks here, so
        # p-values are anti-conservative and describe distributions only.
        pd.concat(track_stat_frames, ignore_index=True).to_csv(
            root / ("track_level_ks_tests_partial.csv" if partial
                    else "track_level_ks_tests.csv"), index=False)

    _condition_summary(manifest_df, diff_pool, root, d_col, design)

    # ---- advanced trajectory analyses (opt-in) ----
    if cfg.advanced and "advanced" in arms:
        from . import advanced as advanced_mod
        sel = sorted(cfg.advanced_select) if cfg.advanced_select else None
        # Inherit the tracking parameters unless the user overrode them, so the
        # advanced analyses operate on the same track population as everything
        # upstream instead of quietly applying a second, different cutoff.
        ctx = _tracking_context(manifest_df)
        min_len = (cfg.advanced_min_track_len if cfg.advanced_min_track_len is not None
                   else (ctx.get("min_track_length") or 10))
        max_tlag = (cfg.advanced_max_tlag if cfg.advanced_max_tlag is not None
                    else (ctx.get("max_lagtime") or ctx.get("diffusion_tau") or 10))
        # A lag cannot exceed what the shortest surviving track supports.
        max_tlag = int(max(2, min(int(max_tlag), max(2, int(min_len) - 1))))
        print(f"Advanced analyses: {', '.join(sel) if sel else 'all'} "
              f"(min_track_len={min_len}, max_tlag={max_tlag} frames)")
        advanced_mod.run_advanced(
            manifest_df, root / "advanced_analysis", select=sel,
            mpp=cfg.mpp, fps=cfg.fps,
            min_track_len=int(min_len), max_tlag=max_tlag,
            particle_radius_um=cfg.advanced_particle_radius_um,
            particle_radius_explicit=cfg.advanced_particle_radius_explicit,
            temperature_k=cfg.advanced_temperature_k,
            n_jobs=cfg.advanced_n_jobs)

    return root


def utils_safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def _export_d_medians(diff_pool, ddir, d_col):
    """Export the numbers behind the D boxplot for GraphPad.

    Writes tidy long tables (one row per movie / per experiment) plus wide
    tables (one column per condition, ragged) that paste straight into a
    GraphPad Column/grouped table. Values are linear D (um^2/s), median over
    finite positive tracks -- exactly the dots (per movie) and diamonds
    (per experiment) drawn on ``D_boxplot_by_condition.png``. The filenames
    carry the ``D_boxplot_`` prefix so the correspondence between figure and
    source numbers is visible in a directory listing.
    """
    if diff_pool is None or diff_pool.empty or d_col not in diff_pool.columns:
        return
    d = diff_pool.copy()
    d["_v"] = pd.to_numeric(d[d_col], errors="coerce")
    d = d[np.isfinite(d["_v"]) & (d["_v"] > 0)]
    if d.empty:
        return
    keys_m = [k for k in ("condition", "experiment", "movie_id") if k in d.columns]
    mm = (d.groupby(keys_m)["_v"].agg(median_D_um2_s="median", n_tracks="size")
          .reset_index())
    mm.to_csv(ddir / "D_boxplot_movie_medians.csv", index=False)
    keys_e = [k for k in ("condition", "experiment") if k in d.columns]
    em = (d.groupby(keys_e)["_v"].agg(median_D_um2_s="median", n_tracks="size")
          .reset_index())
    em.to_csv(ddir / "D_boxplot_experiment_medians.csv", index=False)

    def _wide(df):
        cols = {str(c): sub["median_D_um2_s"].reset_index(drop=True)
                for c, sub in df.groupby("condition")}
        return pd.DataFrame(cols)  # unequal lengths -> NaN-padded (blank cells in GraphPad)

    _wide(mm).to_csv(ddir / "D_boxplot_movie_medians_wide.csv", index=False)
    _wide(em).to_csv(ddir / "D_boxplot_experiment_medians_wide.csv", index=False)


def _diffusion_outputs(diff_pool, root, cfg, binset, design=None):
    ddir = root / "diffusion_analysis"; ddir.mkdir(parents=True, exist_ok=True)
    if diff_pool.empty:
        return
    d_col = "D_fit_um2_s_corr" if cfg.subtract_loc_error and "D_fit_um2_s_corr" in diff_pool.columns else "D_fit_um2_s"
    if d_col not in diff_pool.columns:
        return
    d_by_cond = {c: pd.to_numeric(s[d_col], errors="coerce").to_numpy()
                 for c, s in diff_pool.groupby("condition")}
    all_d = pd.to_numeric(diff_pool[d_col], errors="coerce").to_numpy()
    all_d = all_d[np.isfinite(all_d) & (all_d > 0)]
    if all_d.size == 0:
        return
    d_xlim = float(np.nanpercentile(all_d, 99.0))
    lin_bins = bins_mod.make_bins(0.0, d_xlim, binset.diffusion_width)
    logd = np.log10(all_d)
    log_bins = bins_mod.make_bins(float(logd.min()), float(logd.max()), binset.logd_width)
    # between-condition stats at the design's unit, for annotation
    if design is None:
        design = statistics.infer_design(diff_pool)
    d_per_unit = statistics.per_unit_d_summary(diff_pool, design.unit_column, d_col=d_col)
    pooled_d = {c: pd.to_numeric(s[d_col], errors="coerce").dropna().to_numpy()
                for c, s in diff_pool.groupby("condition")}
    d_stat_df = statistics.between_condition_tests(
        d_per_unit, "median_D", pooled=pooled_d, label="D", design=design)
    d_comparisons = plotting.comparisons_from_stats(d_stat_df)

    for log_y, sfx in _yscales(cfg):
        plotting.overlay_d_histograms(
            d_by_cond, lin_bins, ddir / f"fig1_D_histogram{sfx}.png",
            log_scale=False, title="", log_y=log_y)
        plotting.overlay_d_histograms(
            d_by_cond, log_bins, ddir / f"fig1_logD_histogram{sfx}.png",
            log_scale=True, title="", log_y=log_y)
    _export_d_medians(diff_pool, ddir, d_col)
    # No fixed y-window: a hard 0-0.3 axis squashes typical BiP diffusivities
    # into the bottom sixth of the plot and pushes the significance bars far
    # from the data they describe.
    plotting.d_boxplot(diff_pool, ddir / "D_boxplot_by_condition.png", d_col=d_col,
                       log_scale=False, title="", comparisons=d_comparisons)

    # bins CSV
    rows = []
    for cond, vals in d_by_cond.items():
        v = np.asarray(vals, float); v = v[np.isfinite(v) & (v > 0)]
        counts, edges = np.histogram(v, bins=lin_bins)
        total = counts.sum()
        for i in range(len(counts)):
            rows.append({"condition": cond, "bin_left": edges[i], "bin_right": edges[i+1],
                         "bin_center": 0.5*(edges[i]+edges[i+1]),
                         "fraction": counts[i]/total if total else 0.0, "count": int(counts[i])})
    pd.DataFrame(rows).to_csv(ddir / "fig1_D_histogram_bins.csv", index=False)

    if not cfg.skip_gmm:
        gmm_rows = []
        for cond, vals in d_by_cond.items():
            v = np.asarray(vals, float); v = v[np.isfinite(v) & (v > 0)]
            if v.size < 5:
                continue
            model, bic, table = gmm.fit_log_d(pd.DataFrame({d_col: v}), d_col=d_col, max_k=cfg.max_gmm_k)
            if model is None:
                continue
            table["condition"] = cond
            gmm_rows.append(table)
            plotting.gmm_diagnostic(np.log10(v), model, bic,
                                    ddir / f"gmm_fit_{utils_safe(cond)}.png")
        if gmm_rows:
            pd.concat(gmm_rows, ignore_index=True).to_csv(ddir / "gmm_subpopulations.csv", index=False)



def _imsd_path_for(tracks_csv: str):
    from pathlib import Path as _P
    tc = str(tracks_csv).strip()
    if not tc:
        return None
    return _P(tc).with_name(_P(tc).name.replace("_tracks.csv", "_IMSD.csv"))


def _tracking_context(manifest_df):
    """Read min_track_length / diffusion_tau actually used at tracking, from the
    per-movie ``*_context.json`` written by ``track trackpy``. Returns the modal
    value across movies (with a 'mixed' flag if a run combined different params).
    """
    import json as _json
    from collections import Counter
    from pathlib import Path as _P
    mtl, dtau, mlag, diam = Counter(), Counter(), Counter(), Counter()
    for _, row in manifest_df.iterrows():
        tc = str(row.get("tracks_csv", "")).strip()
        if not tc:
            continue
        cj = _P(tc).with_name(_P(tc).name.replace("_tracks.csv", "_context.json"))
        if not cj.exists():
            continue
        try:
            d = _json.load(open(cj))
        except (OSError, ValueError):
            continue
        prm = d.get("params") or {}
        if "min_track_length" in prm:
            try: mtl[int(prm["min_track_length"])] += 1
            except (TypeError, ValueError): pass
        if "diffusion_tau" in prm:
            try: dtau[int(prm["diffusion_tau"])] += 1
            except (TypeError, ValueError): pass
        if "max_lagtime" in prm:
            try: mlag[int(prm["max_lagtime"])] += 1
            except (TypeError, ValueError): pass
        if "diameter" in prm:
            try: diam[int(prm["diameter"])] += 1
            except (TypeError, ValueError): pass
    def modal(c): return c.most_common(1)[0][0] if c else None
    return {"min_track_length": modal(mtl), "diffusion_tau": modal(dtau),
            "max_lagtime": modal(mlag), "diameter": modal(diam),
            "mixed": (len(mtl) > 1 or len(dtau) > 1)}


def _fixed_support_msd(manifest_df, cfg):
    """Ensemble MSD-vs-tau over a length-consistent track population.

    Reads each movie's per-track IMSD.csv and keeps only tracks that still have a
    finite MSD at the chosen maximum lag, so the SAME tracks contribute at every
    displayed lag. This removes the survivorship kink that appears at
    tau = min_track_length when averaging all tracks regardless of length.

    Returns (msd_by_condition, n_movies, n_tracks_by_cond, max_lag_frames) or None
    if no IMSD.csv files are available (caller then falls back to the old path).
    """
    fps = float(cfg.fps)
    movies = []  # (condition, experiment, movie_id, frame_index(np int), imsd(np 2d: lag x particle))
    for _, row in manifest_df.iterrows():
        ip = _imsd_path_for(row.get("tracks_csv", ""))
        if ip is None or not ip.exists():
            continue
        try:
            im = pd.read_csv(ip, index_col=0)
        except (OSError, pd.errors.ParserError, ValueError):
            continue
        if im.empty:
            continue
        lag_s = pd.to_numeric(pd.Index(im.index), errors="coerce").to_numpy(float)
        ok = np.isfinite(lag_s) & (lag_s > 0)
        im = im.loc[ok]
        frames = np.rint(lag_s[ok] * fps).astype(int)
        vals = im.apply(pd.to_numeric, errors="coerce").to_numpy(float)  # lag x particle
        movies.append((str(row["condition"]), str(row["experiment"]), str(row["movie_id"]),
                       frames, vals))
    if not movies:
        return None

    max_avail = int(max(fr.max() for _, _, _, fr, _ in movies if fr.size))

    def support_at(L):
        """per-condition number of tracks with a finite MSD at frame L."""
        counts = {}
        for cond, _, _, frames, vals in movies:
            hit = np.where(frames == L)[0]
            if hit.size == 0:
                continue
            counts[cond] = counts.get(cond, 0) + int(np.isfinite(vals[hit[0]]).sum())
        return counts

    ctx = _tracking_context(manifest_df)
    min_len = ctx.get("min_track_length")
    diff_tau = ctx.get("diffusion_tau")
    if cfg.msd_max_lag:
        max_lag = int(min(cfg.msd_max_lag, max_avail))      # explicit override
    elif min_len:
        # Rely on the tracking min_track_length: every surviving track (length
        # >= min_len) contributes at all lags <= min_len - 1, so the displayed
        # curve is over one consistent, maximal-N population -> no kink.
        max_lag = int(min(max(1, int(min_len) - 1), max_avail))
    else:
        # No tracking context available: fall back to auto-by-support.
        max_lag = 1
        for L in range(1, max_avail + 1):
            c = support_at(L)
            if c and min(c.values()) >= cfg.msd_min_tracks:
                max_lag = L
            else:
                break
    # Experiment-level MSD test at the tracking D-fit lag (diffusion_tau), over
    # tracks that span it -- same lag as build_diffusion_table's D_fit.
    fit_lag = int(min(diff_tau, max_avail)) if diff_tau else max_lag

    # per-movie fixed-support curve over frames 1..max_lag
    per_cond_curves = {}   # cond -> list of (frame_array, mean_msd_array)
    n_tracks_by_cond = {}
    per_exp_at_target = []  # rows for the experiment-level MSD test (at max_lag)
    for cond, exp, mid, frames, vals in movies:
        hit = np.where(frames == max_lag)[0]
        if hit.size == 0:
            continue
        qual = np.isfinite(vals[hit[0]])           # tracks spanning max_lag
        if not qual.any():
            continue
        keep = np.where(frames <= max_lag)[0]
        fk = frames[keep]
        sub = vals[np.ix_(keep, np.where(qual)[0])]
        curve = np.nanmean(sub, axis=1)            # mean over qualifying tracks per lag
        per_cond_curves.setdefault(cond, []).append((fk, curve))
        n_tracks_by_cond[cond] = n_tracks_by_cond.get(cond, 0) + int(qual.sum())
        hf = np.where(frames == fit_lag)[0]
        if hf.size:
            qf = np.isfinite(vals[hf[0]])
            if qf.any():
                # Keep movie_id as well as experiment so the caller can
                # aggregate at whichever unit the statistical design selected.
                per_exp_at_target.append({"condition": cond, "experiment": exp,
                                          "movie_id": mid,
                                          "msd_at_fit": float(np.nanmean(vals[hf[0]][qf]))})

    if not per_cond_curves:
        return None

    msd_by_condition = {}
    n_movies = {}
    for cond, curves in per_cond_curves.items():
        # align on frame grid 1..max_lag
        grid = np.arange(1, max_lag + 1)
        mat = np.full((len(curves), grid.size), np.nan)
        for i, (fk, curve) in enumerate(curves):
            idx = {int(f): v for f, v in zip(fk, curve)}
            mat[i] = [idx.get(int(g), np.nan) for g in grid]
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(mat, axis=0)
            nvec = np.sum(np.isfinite(mat), axis=0)
            sd = np.nanstd(mat, axis=0, ddof=1)
            sem = np.where(nvec > 1, sd / np.sqrt(nvec), np.nan)
        agg = pd.DataFrame({"tau_s": grid / fps, "msd_mean_um2": mean,
                            "msd_sem_um2": sem, "n_movies": nvec,
                            "n_tracks": n_tracks_by_cond.get(cond, 0)})
        agg = agg.dropna(subset=["msd_mean_um2"])
        msd_by_condition[cond] = agg
        n_movies[cond] = len(curves)
    # One row per movie; the caller collapses to the design's unit.
    pe = pd.DataFrame(per_exp_at_target)
    if not pe.empty:
        pe = (pe.groupby(["condition", "experiment", "movie_id"])["msd_at_fit"]
              .median().reset_index())
    return msd_by_condition, n_movies, n_tracks_by_cond, max_lag, pe, {
        "min_track_length": min_len, "diffusion_tau": diff_tau, "fit_lag_frames": fit_lag,
        "mixed_tracking_params": ctx.get("mixed", False)}


def _msd_outputs(manifest_df, root, cfg, design=None):
    """Ensemble MSD-vs-τ compared between conditions.

    Each movie's per-movie mean IMSD curve is one observation; per condition we
    average those curves and show the SEM across movies. The legend states how
    many movies contribute to each condition, so it is explicit what the curves
    represent. Because the statistical unit is the experiment, MSD values at the
    fit lag are also compared between conditions with an experiment-level test.
    """
    mdir = root / "msd_analysis"; mdir.mkdir(parents=True, exist_ok=True)

    # Preferred: fixed-support ensemble MSD (length-consistent track population),
    # which removes the survivorship kink at tau = min_track_length. Falls back to
    # the legacy all-tracks aggregation if per-movie IMSD.csv files are absent.
    fs = _fixed_support_msd(manifest_df, cfg)
    if fs is not None:
        msd_by_condition, n_movies, n_tracks_by_cond, max_lag, per_exp, ctx = fs
        tidy = []
        for cond, agg in msd_by_condition.items():
            a = agg.copy(); a["condition"] = cond; tidy.append(a)
        if tidy:
            pd.concat(tidy, ignore_index=True).to_csv(mdir / "msd_vs_tau_by_condition.csv", index=False)
        fit_s = ctx.get("fit_lag_frames", max_lag) / cfg.fps
        msd_comparisons = []
        try:
            if per_exp is not None and not per_exp.empty:
                if design is None:
                    design = statistics.infer_design(manifest_df)
                # Collapse per-movie MSDs to the design's statistical unit.
                unit_col = design.unit_column
                per_unit = (per_exp.groupby(["condition", unit_col])["msd_at_fit"]
                            .median().reset_index()) if unit_col in per_exp.columns else per_exp
                pooled = {c: g["msd_at_fit"].dropna().to_numpy()
                          for c, g in per_unit.groupby("condition")}
                sdf = statistics.between_condition_tests(
                    per_unit, "msd_at_fit", pooled=pooled, label=f"MSD@{fit_s:.3g}s",
                    design=design)
                per_unit.to_csv(mdir / f"msd_at_fit_per_{design.unit}.csv", index=False)
                msd_comparisons = plotting.comparisons_from_stats(sdf)
                # Categorical companion carrying the significance bars; the
                # MSD curves themselves stay unannotated.
                pm = (per_exp.groupby(["condition", "experiment", "movie_id"])["msd_at_fit"]
                      .median().reset_index())
                plotting.condition_boxplot(
                    pm, mdir / "msd_at_fit_boxplot.png", value_col="msd_at_fit",
                    ylabel=f"MSD at τ = {fit_s:.3g} s (µm²)",
                    comparisons=msd_comparisons)
        except Exception:
            pass
        mtl = ctx.get("min_track_length")
        plotting.msd_vs_tau_by_condition(
            msd_by_condition, mdir / "msd_vs_tau_by_condition.png", loglog=False,
            title="", n_movies=n_movies)
        plotting.msd_vs_tau_by_condition(
            msd_by_condition, mdir / "msd_vs_tau_by_condition_loglog.png", loglog=True,
            title="", n_movies=n_movies)
        utils.dump_json({"estimator": "fixed_support",
                         "source_of_params": "track_trackpy context (*_context.json)",
                         "min_track_length": (int(mtl) if mtl else None),
                         "diffusion_tau": (int(ctx["diffusion_tau"]) if ctx.get("diffusion_tau") else None),
                         "msd_max_lag_frames": int(max_lag),
                         "msd_fit_lag_frames": int(ctx.get("fit_lag_frames", max_lag)),
                         "msd_max_lag_override": cfg.msd_max_lag,
                         "mixed_tracking_params": ctx.get("mixed_tracking_params", False),
                         "n_tracks_by_condition": {k: int(v) for k, v in n_tracks_by_cond.items()}},
                        mdir / "msd_support.json")
        return

    frames = []
    for _, row in manifest_df.iterrows():
        tc = str(row.get("tracks_csv", "")).strip()
        if not tc:
            continue
        msd_path = Path(tc).with_name(Path(tc).name.replace("_tracks.csv", "_MSD_vs_tau.csv"))
        if not msd_path.exists():
            continue
        try:
            df = pd.read_csv(msd_path)
        except (OSError, pd.errors.ParserError):
            continue
        df["condition"] = str(row["condition"])
        df["experiment"] = str(row["experiment"])
        df["movie_id"] = str(row["movie_id"])
        frames.append(df)
    if not frames:
        return
    allmsd = pd.concat(frames, ignore_index=True)
    # Per-movie curve = mean IMSD on its own lag grid. We deliberately do NOT
    # fall back to EMSD here: the two quantities live on floating-point-duplicated
    # tau grids, and interleaving them produces a spurious sawtooth. Keep one
    # consistent quantity, dropping lags where it is undefined.
    allmsd["msd_movie"] = pd.to_numeric(allmsd.get("imsd_mean_um2"), errors="coerce")
    if allmsd["msd_movie"].notna().sum() == 0:  # fallback only if IMSD absent entirely
        allmsd["msd_movie"] = pd.to_numeric(allmsd.get("emsd_um2"), errors="coerce")
    allmsd["tau_s"] = pd.to_numeric(allmsd["tau_s"], errors="coerce").round(6)
    allmsd = allmsd.dropna(subset=["msd_movie", "tau_s"])

    msd_by_condition: Dict[str, pd.DataFrame] = {}
    n_movies: Dict[str, int] = {}
    tidy_rows = []
    for cond, csub in allmsd.groupby("condition"):
        agg = (csub.groupby("tau_s")["msd_movie"]
               .agg(msd_mean_um2="mean",
                    msd_sem_um2=lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum()) if x.notna().sum() > 1 else np.nan,
                    n_movies="count")
               .reset_index())
        msd_by_condition[cond] = agg
        n_movies[cond] = int(csub["movie_id"].nunique())
        agg2 = agg.copy(); agg2["condition"] = cond
        tidy_rows.append(agg2)
    pd.concat(tidy_rows, ignore_index=True).to_csv(mdir / "msd_vs_tau_by_condition.csv", index=False)

    # experiment-level test of MSD at the fit lag (diffusion_tau frames)
    try:
        if design is None:
            design = statistics.infer_design(manifest_df)
        target_s = _closest_tau_s(allmsd["tau_s"], 5 / cfg.fps)
        per_unit = (allmsd[np.isclose(allmsd["tau_s"], target_s)]
                    .groupby(["condition", design.unit_column])["msd_movie"]
                    .median().reset_index()
                    .rename(columns={"msd_movie": "msd_at_fit"}))
        pooled = {c: s["msd_at_fit"].dropna().to_numpy() for c, s in per_unit.groupby("condition")}
        statistics.between_condition_tests(
            per_unit, "msd_at_fit", pooled=pooled, label=f"MSD@{target_s:.3g}s", design=design)
    except Exception:
        pass

    plotting.msd_vs_tau_by_condition(
        msd_by_condition, mdir / "msd_vs_tau_by_condition.png", loglog=False,
        title="", n_movies=n_movies)
    plotting.msd_vs_tau_by_condition(
        msd_by_condition, mdir / "msd_vs_tau_by_condition_loglog.png", loglog=True,
        title="", n_movies=n_movies)


def _closest_tau_s(series, target):
    vals = pd.to_numeric(series, errors="coerce").dropna().unique()
    if len(vals) == 0:
        raise ValueError("no tau_s values")
    return float(vals[np.argmin(np.abs(vals - target))])


#: The two crowding axes, kept separate on purpose. Field density is the
#: biology-facing question (how crowded was the ER); link safety is the
#: QC-facing one (how much room the linker had). They answer different
#: objections and can disagree.
_DENSITY_AXES = (
    ("cell_median_field_localizations_n", "cell_density",
     "Cell density (localizations in field)", "log2"),
    ("track_median_field_localizations_n", "density",
     "Local density (localizations in field)", "log2"),
    ("link_safety_ratio", "link_safety",
     "Link safety (neighbour spacing / step)", "quantile"),
)


def _density_outputs(manifest_df, diff_pool, root, cfg, design, d_col):
    """Compare conditions at matched crowding.

    Two cells under the same condition can carry very different GEM densities,
    and density biases mobility twice over: a crowded ER is a different
    environment, and crowding also makes linking ambiguous, which truncates
    tracks and depresses apparent mobility in exactly the cells that express
    most. A condition difference can therefore be manufactured entirely by a
    density difference.

    Outputs, in the order you would read them:

    1. **Is there a density difference to worry about?** Per-condition density
       and link-safety distributions, and how many cells each field held.
    2. **Does the mobility difference survive at matched density?** Conditions
       compared within each density stratum, keeping every track.
    3. **Optionally, one balanced subset.** Coarsened exact matching equalizes
       the density distribution by discarding, with a before/after balance plot.
    4. **What does it look like?** Origin-aligned tracks faceted by stratum.
    """
    from . import density as density_mod
    from . import density_runner

    pool = density_runner.load_pool(manifest_df)
    if diff_pool is None or diff_pool.empty:
        return []
    ddir = root / "density_analysis"

    have_inline = ("track_median_field_localizations_n" in diff_pool.columns
                   and diff_pool["track_median_field_localizations_n"].notna().any())
    if have_inline:
        merged = diff_pool.copy()
    elif not pool.empty:
        want = [c for c in density_mod.TRACK_DENSITY_COLUMNS if c in pool.columns]
        d = diff_pool.drop(columns=[c for c in want if c in diff_pool.columns]).copy()
        d["_pk"] = d["particle"].map(_pkey)
        p2 = pool.copy(); p2["_pk"] = p2["particle"].map(_pkey)
        merged = d.merge(p2[["movie_id", "_pk"] + want], on=["movie_id", "_pk"],
                         how="inner").drop(columns=["_pk"])
    else:
        return []
    if merged.empty or "track_median_field_localizations_n" not in merged.columns:
        return []
    ddir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(ddir / "per_track_density.csv", index=False)

    unit_col = design.unit_column
    frames: List[pd.DataFrame] = []

    # ---- 0. provenance and honesty about where density came from ----
    sources = (merged["density_source"].value_counts().to_dict()
               if "density_source" in merged.columns else {})
    if sources.get("post_link"):
        print(f"  density: {sources['post_link']} track(s) measured post-link. "
              f"Linking and stub filtering already removed detections, most of "
              f"them in crowded fields, so this understates crowding.")

    cells = density_runner.load_cell_pool(manifest_df)
    if not cells.empty:
        cells.to_csv(ddir / "per_cell_summary.csv", index=False)
        unresolved = cells[cells["n_cells_in_field"] == 1]
        n_conf = int(unresolved["cell_segmentation_note"]
                     .astype(str).str.contains("confluent").sum())
        if n_conf:
            print(f"  density: {n_conf} field(s) could not be split into separate "
                  f"cells (touching or confluent) and are treated as one field.")

    # ---- 1. is there a density difference between conditions at all? ----
    for col, slug, label, _mode in _DENSITY_AXES:
        if col not in merged.columns or not merged[col].notna().any():
            continue
        per_unit = (merged.groupby(["condition", unit_col])[col]
                    .median().reset_index())
        if per_unit["condition"].nunique() >= 2:
            st = statistics.between_condition_tests(
                per_unit.rename(columns={col: "value"}), "value",
                label=f"{slug} (crowding)", design=design)
            if not st.empty:
                st.to_csv(ddir / f"{slug}_between_conditions.csv", index=False)
                frames.append(st)
            plotting.condition_boxplot(
                merged.groupby(["condition", "experiment", "movie_id"])[col]
                .median().reset_index().rename(columns={col: "value"}),
                ddir / f"{slug}_by_condition.png", value_col="value", ylabel=label,
                comparisons=plotting.comparisons_from_stats(st))
        by_cond = {c: pd.to_numeric(g[col], errors="coerce").dropna().to_numpy()
                   for c, g in merged.groupby("condition")}
        by_cond = {c: v for c, v in by_cond.items() if v.size}
        if by_cond:
            allv = np.concatenate(list(by_cond.values()))
            lo, hi = np.nanpercentile(allv, [0.5, 99.5])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                for log_y, sfx in _yscales(cfg):
                    plotting.overlay_value_histograms(
                        by_cond, np.linspace(lo, hi, 51),
                        ddir / f"{slug}_distribution{sfx}.png",
                        xlabel=label, log_y=log_y)

    # Link safety is a QC number with an absolute meaning; say when it is bad.
    if "link_safety_ratio" in merged.columns:
        ls = pd.to_numeric(merged["link_safety_ratio"], errors="coerce")
        bad = float((ls < density_mod.LINK_SAFETY_WARN).mean())
        if np.isfinite(bad) and bad > 0.2:
            print(f"  WARNING: {bad:.0%} of tracks have link safety below "
                  f"{density_mod.LINK_SAFETY_WARN:g} -- neighbours are closer than a "
                  f"few typical steps, so some links are guesses. Consider "
                  f"--filter-linksafety {density_mod.LINK_SAFETY_WARN:g} -")

    # ---- 2. mobility at matched crowding, keeping every track ----
    metrics = [(d_col, "D (µm²/s)", "D", True)]
    for tau in track_metrics.available_taus(merged)[:1]:
        metrics.append((f"step_median_um_tau{tau}", f"Median step at τ={tau} (µm)",
                        f"step_tau{tau}", False))

    for col, slug, label, mode in _DENSITY_AXES:
        if col not in merged.columns or not merged[col].notna().any():
            continue
        if mode == "log2":
            strata = density_mod.density_strata(merged[col])
            order = sorted([s for s in strata.unique() if s != "invalid"],
                           key=lambda s: float(s.strip("[)").split(",")[0]))
        else:
            strata, table = filters_mod.absolute_bins(
                merged[col], n_quantile=cfg.density_strata, prefix="S")
            order = list(table["bin"]) if not table.empty else []
            if not table.empty:
                table.to_csv(ddir / f"{slug}_stratum_edges.csv", index=False)
        s = merged.copy()
        s["_stratum"] = strata
        s = s[s["_stratum"].astype(str) != "invalid"].dropna(subset=["_stratum"])
        if s.empty or not order:
            continue

        # Matching only means anything where the conditions actually overlap.
        # If every condition sits in its own range -- transfection so uneven that
        # no crowded control cell exists to pair with a crowded treated one --
        # then no correction can rescue the comparison, and saying so is far more
        # use than an empty table.
        shared = [b for b, g in s.groupby("_stratum", observed=True)
                  if g["condition"].nunique() == s["condition"].nunique()]
        frac = float(s["_stratum"].isin(shared).mean()) if shared else 0.0
        ov = pd.DataFrame([{"axis": slug, "stratum": str(b),
                            "shared_by_all_conditions": b in shared,
                            **{f"n_{c}": int(n) for c, n in
                               s[s["_stratum"] == b]["condition"].value_counts().items()}}
                           for b in order if b in set(s["_stratum"])])
        if not ov.empty:
            ov.to_csv(ddir / f"{slug}_stratum_overlap.csv", index=False)
        if not shared:
            print(f"  {slug}: the conditions share NO stratum -- their crowding "
                  f"distributions do not overlap at all, so no density-matched "
                  f"comparison exists. Any difference here is inseparable from "
                  f"the density difference. See {slug}_stratum_overlap.csv.")
            continue
        if frac < 0.5:
            print(f"  {slug}: only {frac:.0%} of tracks lie in strata shared by every "
                  f"condition; the matched comparison rests on that minority.")

        for mcol, mlabel, mslug, log_y in metrics:
            if mcol not in s.columns or not s[mcol].notna().any():
                continue
            sub = s[np.isfinite(pd.to_numeric(s[mcol], errors="coerce"))]
            if log_y:
                sub = sub[pd.to_numeric(sub[mcol], errors="coerce") > 0]
            if len(sub) < 50:
                continue
            per_bin = (sub.groupby(["_stratum", "condition", unit_col], observed=True)[mcol]
                       .median().reset_index())
            per_bin.to_csv(ddir / f"{mslug}_by_{slug}_stratum_per_{design.unit}.csv",
                           index=False)
            comps, out = {}, []
            for b, g in per_bin.groupby("_stratum", observed=True):
                if g["condition"].nunique() < 2:
                    continue
                st = statistics.between_condition_tests(
                    g.rename(columns={mcol: "value"}), "value",
                    label=f"{mslug} | {slug} {b}", design=design)
                if not st.empty:
                    st[f"{slug}_stratum"] = str(b)
                    out.append(st)
                    comps[b] = plotting.comparisons_from_stats(st)
            if out:
                allb = pd.concat(out, ignore_index=True)
                allb.to_csv(ddir / f"{mslug}_between_conditions_by_{slug}.csv", index=False)
                frames.append(allb)
            plotting.grouped_boxplot(
                sub, ddir / f"{mslug}_by_{slug}_stratum.png", x_col="_stratum",
                group_col="condition", value_col=mcol,
                xlabel=label, ylabel=mlabel, x_order=order,
                comparisons_within_x=comps)

        # Step-size distribution and MSD-style read at each stratum.
        _density_stratified_steps(manifest_df, s, order, slug, label, ddir, cfg)

    # ---- 3. optional balanced subset ----
    if cfg.density_match:
        try:
            res = density_mod.density_match_tracks(
                merged, density_column=cfg.density_match_column,
                length_bin_edges=cfg.density_match_length_bins,
                seed=cfg.density_match_seed)
            res.assignments.to_csv(ddir / "density_match_assignments.csv", index=False)
            res.balance.to_csv(ddir / "density_match_balance.csv", index=False)
            kept = density_mod.apply_density_match(merged, res.assignments)
            n_sel = int(res.assignments["density_match_selected"].sum())
            print(f"  density matching: {n_sel}/{len(res.assignments)} tracks retained "
                  f"({n_sel / max(len(res.assignments), 1):.0%})")
            if n_sel == 0:
                why = (res.assignments["density_match_reason"].value_counts()
                       .to_dict() if not res.assignments.empty else {})
                print(f"    Nothing matched on '{cfg.density_match_column}'. Reasons: "
                      f"{why}. If every track is 'no_shared_stratum', the conditions "
                      f"occupy separate density ranges and cannot be compared at "
                      f"matched density -- that is a limit of the data, not a setting "
                      f"to tune.")
            _density_balance_plot(res.assignments, ddir / "density_match_balance.png",
                                  cfg.density_match_column)
            if not kept.empty:
                kept.to_csv(ddir / "per_track_density_matched.csv", index=False)
                for mcol, mlabel, mslug, _ in metrics:
                    if mcol not in kept.columns:
                        continue
                    pu = (kept.groupby(["condition", unit_col])[mcol]
                          .median().reset_index())
                    if pu["condition"].nunique() < 2:
                        continue
                    st = statistics.between_condition_tests(
                        pu.rename(columns={mcol: "value"}), "value",
                        label=f"{mslug} (density-matched)", design=design)
                    if not st.empty:
                        frames.append(st)
                    plotting.condition_boxplot(
                        kept.groupby(["condition", "experiment", "movie_id"])[mcol]
                        .median().reset_index().rename(columns={mcol: "value"}),
                        ddir / f"{mslug}_density_matched.png", value_col="value",
                        ylabel=mlabel, comparisons=plotting.comparisons_from_stats(st))
        except Exception as exc:
            print(f"  density matching failed: {type(exc).__name__}: {exc}")

    # ---- 4. origin-aligned tracks, faceted by stratum ----
    try:
        _origin_aligned_by_stratum(manifest_df, merged, ddir, cfg)
    except Exception as exc:
        print(f"  origin-aligned track plot failed: {type(exc).__name__}: {exc}")

    utils.dump_json(
        {"n_tracks": int(len(merged)),
         "density_source_counts": sources,
         "n_cells_total": (int(cells["cell_id"].count()) if not cells.empty else 0),
         "fields_unresolved": (int((cells["n_cells_in_field"] == 1).sum())
                               if not cells.empty else 0),
         "density_strata": cfg.density_strata,
         "matched": bool(cfg.density_match),
         "match_column": cfg.density_match_column},
        ddir / "density_analysis_summary.json")
    return frames


def _density_stratified_steps(manifest_df, stratified, order, slug, label, ddir, cfg):
    """Step-size histograms per density stratum, overlaid by condition.

    The distribution is where mis-linking shows itself: a truncated or wrongly
    joined track changes the shape of the tail, not just the median. Comparing
    shapes within a stratum asks whether the conditions differ once both are
    looking at fields of equal crowding.
    """
    available = _discover_tau_files(Path(ddir).parent.parent)
    if not available:
        return
    tau = sorted(available)[0] if 1 not in available else 1
    labels = _label_map(manifest_df)
    pool = _load_tau_pool(available[tau], labels)
    if pool.empty:
        return
    key = stratified[["movie_id", "particle", "_stratum"]].copy()
    key["_pk"] = key["particle"].map(_pkey)
    pool["_pk"] = pool["particle"].map(_pkey)
    pool = pool.merge(key[["movie_id", "_pk", "_stratum"]], on=["movie_id", "_pk"],
                      how="inner")
    if pool.empty:
        return
    binset = bins_mod.BinSettings(mpp=cfg.mpp, fps=cfg.fps,
                                 step_bin_width_um=cfg.step_bin_width_um)
    allv = pool["step_um"].to_numpy(float)
    allv = allv[np.isfinite(allv)]
    if allv.size == 0:
        return
    xlim = float(allv.max()) * 1.02
    edges = bins_mod.make_bins(0.0, xlim, binset.step_width)
    sdir = ddir / f"step_size_by_{slug}"
    sdir.mkdir(parents=True, exist_ok=True)
    for stratum in order:
        sub = pool[pool["_stratum"].astype(str) == str(stratum)]
        if len(sub) < 200:
            continue
        hists = {c: histograms.raw_pooled_histogram(g["step_um"].to_numpy(), edges, c)
                 for c, g in sub.groupby("condition")}
        if len(hists) < 2:
            continue
        for log_y, sfx in _yscales(cfg):
            plotting.overlay_step_histograms(
                hists, sdir / f"step_hist_{utils_safe(str(stratum))}{sfx}.png",
                title=f"{label}: {stratum}   (τ = {tau})",
                legend_meta={c: f"n={sub[sub.condition == c].shape[0]}" for c in hists},
                xlim=xlim, log_y=log_y)


def _origin_aligned_by_stratum(manifest_df, merged, ddir, cfg):
    """Flower plots: tracks translated to a common origin, faceted by crowding."""
    from . import density as density_mod
    col = "track_median_field_localizations_n"
    if col not in merged.columns:
        return
    strata = density_mod.density_strata(merged[col])
    merged = merged.assign(_stratum=strata)
    merged = merged[merged["_stratum"] != "invalid"]
    if merged.empty:
        return
    order = sorted(merged["_stratum"].unique(),
                   key=lambda s: float(s.strip("[)").split(",")[0]))
    keep = {(str(m), _pkey(p)): str(s) for m, p, s in
            zip(merged["movie_id"], merged["particle"], merged["_stratum"])}
    cond_of = {str(m): str(c) for m, c in zip(merged["movie_id"], merged["condition"])}

    frames = []
    for _, row in manifest_df.iterrows():
        tc = str(row.get("tracks_csv", "")).strip()
        if not tc or not Path(tc).exists():
            continue
        mid = str(row["movie_id"])
        try:
            t = pd.read_csv(tc, usecols=lambda c: c in ("particle", "frame", "x", "y"))
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        if t.empty:
            continue
        al = density_mod.origin_aligned(t, cfg.mpp, max_tracks=250)
        if al.empty:
            continue
        al["movie_id"] = mid
        al["condition"] = cond_of.get(mid, str(row["condition"]))
        al["_stratum"] = [keep.get((mid, _pkey(p))) for p in al["particle"]]
        al["particle"] = [f"{mid}::{_pkey(p)}" for p in al["particle"]]
        frames.append(al.dropna(subset=["_stratum"]))
    if not frames:
        return
    aligned = pd.concat(frames, ignore_index=True)
    if aligned.empty:
        return
    aligned.to_csv(ddir / "origin_aligned_tracks.csv.gz", index=False,
                   compression="gzip")
    present = [s for s in order if s in set(aligned["_stratum"])]
    plotting.origin_aligned_tracks(
        aligned, ddir / "origin_aligned_tracks_by_density.png",
        row_col="_stratum", col_col="condition", row_order=present)
    plotting.origin_aligned_tracks(
        aligned, ddir / "origin_aligned_tracks.png", row_col=None,
        col_col="condition")


def _density_balance_plot(assignments, out_png, density_column):
    """Density distributions before and after matching, in house style."""
    import matplotlib.pyplot as plt
    if assignments is None or assignments.empty or density_column not in assignments:
        return
    conds = sorted(assignments["condition"].astype(str).unique())
    colors = plotting.condition_colors(conds)
    v_all = pd.to_numeric(assignments[density_column], errors="coerce")
    v_all = v_all[np.isfinite(v_all) & (v_all > 0)]
    if v_all.empty:
        return
    edges = np.linspace(np.log2(v_all.min()), np.log2(v_all.max()), 31)
    with plotting._PLOT_LOCK:
        plotting.apply_house_style()
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True, sharey=True)
        for ax, sel, ttl in zip(axes, [None, True], ["Before matching", "After matching"]):
            d = (assignments if sel is None else
                 assignments[assignments["density_match_selected"].astype(bool)])
            for c in conds:
                v = pd.to_numeric(d.loc[d["condition"].astype(str) == c, density_column],
                                  errors="coerce")
                v = v[np.isfinite(v) & (v > 0)]
                if not len(v):
                    continue
                counts, _ = np.histogram(np.log2(v), bins=edges)
                tot = counts.sum()
                ax.step(0.5 * (edges[:-1] + edges[1:]), counts / tot if tot else counts,
                        where="mid", color=colors[c], linewidth=1.8, label=c)
            ax.set_title(ttl, fontsize=11)
            ax.set_xlabel("log₂ local density")
        axes[0].set_ylabel("Fraction per bin")
        axes[0].legend(loc="best")
        plotting._save(fig, out_png)


def _marker_metric_axes(pool, cfg, d_col):
    """Mobility metrics to plot expression against, longest-lag first.

    D and median step disagree on subdiffusive data -- D at a single lag mixes
    the MSD amplitude with the anomalous exponent, while the step median does
    not -- so both are emitted. Agreement between the two panels is evidence;
    disagreement localizes the problem to the metric rather than the biology.
    """
    axes = [(d_col, "D (µm²/s)", "D", True)]
    for tau in track_metrics.available_taus(pool)[:2]:
        col = f"step_median_um_tau{tau}"
        axes.append((col, f"Median step at τ={tau} (µm)", f"step_tau{tau}", False))
    if "v_path_um_s" in pool.columns:
        axes.append(("v_path_um_s", "Mean path speed (µm/s)", "path_speed", True))
    return [a for a in axes if a[0] in pool.columns]


def _marker_outputs(manifest_df, diff_pool, root, cfg, design, d_col):
    """Mobility versus expression-marker intensity.

    Four questions, each with its own output:

    * **Do high-expressing cells behave differently?** A correlation across
      cells, using each cell's median. This is the biological question, and it
      needs the *between*-cell variance -- which in a typical dataset is most
      of the intensity variance, since expression level is a property of a cell
      rather than of a position within one.
    * **Does mobility track expression inside a cell?** The within-cell
      correlation, computed per unit and tested for reproducibility. A
      different, weaker question: within one cell, marker intensity mostly
      reports local ER density, not expression.
    * **Does either relationship differ between conditions?**
    * **Do conditions differ at matched expression, and at matched
      brightness?** Which separates a real mobility effect from a difference in
      what got detected.
    """
    from . import marker as marker_mod
    pool = marker_mod.load_marker_pool(manifest_df)
    if diff_pool is None or diff_pool.empty:
        return []
    mdir = root / "marker_analysis"

    # The metrics table already carries marker_intensity; fall back to joining
    # the standalone table for manifests measured before that existed.
    if "marker_intensity" in diff_pool.columns and diff_pool["marker_intensity"].notna().any():
        merged = diff_pool.copy()
    elif not pool.empty:
        want = ["marker_intensity", "marker_intensity_sd", "x_range_px", "y_range_px",
                "dist_to_edge_px", "frac_disk_in_frame", "n_localizations_measured"]
        have = [c for c in want if c in pool.columns]
        d = diff_pool.drop(columns=[c for c in want if c in diff_pool.columns]).copy()
        d["_pk"] = d["particle"].map(_pkey)
        pool = pool.copy(); pool["_pk"] = pool["particle"].map(_pkey)
        merged = d.merge(pool[["movie_id", "_pk"] + have], on=["movie_id", "_pk"],
                         how="inner", validate="one_to_one").drop(columns=["_pk"])
    else:
        return []
    if "marker_intensity" not in merged.columns:
        return []
    merged = merged[np.isfinite(pd.to_numeric(merged["marker_intensity"], errors="coerce"))]
    if merged.empty:
        return []
    mdir.mkdir(parents=True, exist_ok=True)

    # Marker-specific QC gates (position / disk coverage).
    if cfg.marker_min_disk_coverage > 0 and "frac_disk_in_frame" in merged.columns:
        merged = merged[pd.to_numeric(merged["frac_disk_in_frame"], errors="coerce")
                        .fillna(0) >= cfg.marker_min_disk_coverage]
    if cfg.marker_edge_margin_px > 0 and "dist_to_edge_px" in merged.columns:
        merged = merged[pd.to_numeric(merged["dist_to_edge_px"], errors="coerce")
                        .fillna(-1) >= cfg.marker_edge_margin_px]
    if merged.empty:
        return []
    merged["_I"] = pd.to_numeric(merged["marker_intensity"], errors="coerce")
    merged.to_csv(mdir / "per_track_mobility_vs_marker.csv", index=False)

    _brightness_distributions(merged, mdir, design, cfg)

    unit_col = design.unit_column
    axes = _marker_metric_axes(merged, cfg, d_col)
    frames: List[pd.DataFrame] = []
    within_all, percell_all = [], []

    for col, ylabel, slug, log_y in axes:
        sub = merged[np.isfinite(pd.to_numeric(merged[col], errors="coerce"))].copy()
        if log_y:
            sub = sub[pd.to_numeric(sub[col], errors="coerce") > 0]
        if len(sub) < 50:
            continue
        adir = mdir / slug
        adir.mkdir(parents=True, exist_ok=True)

        # --- across cells: the biological question ---
        cell = (sub.groupby(["condition", "experiment", "movie_id"])
                .agg(median_intensity=("_I", "median"),
                     median_metric=(col, "median"), n_tracks=(col, "size")).reset_index())
        cell.to_csv(adir / "per_cell_medians.csv", index=False)
        pc_rows = []
        for cond, g in cell.groupby("condition"):
            if len(g) < 6:
                continue
            r, pv = stats_spearman(g["median_intensity"], g["median_metric"])
            pc_rows.append({"metric": slug, "condition": cond, "level": "across cells",
                            "rho": r, "p_value": pv, "n_cells": int(len(g))})
        if len(cell) >= 6:
            r, pv = stats_spearman(cell["median_intensity"], cell["median_metric"])
            pc_rows.append({"metric": slug, "condition": "ALL", "level": "across cells",
                            "rho": r, "p_value": pv, "n_cells": int(len(cell))})
        pc = pd.DataFrame(pc_rows)
        if not pc.empty:
            pc.to_csv(adir / "correlation_across_cells.csv", index=False)
            percell_all.append(pc)
        plotting.scatter_with_trend(
            cell, adir / "per_cell_metric_vs_marker.png",
            x_col="median_intensity", y_col="median_metric", group_col="condition",
            xlabel="Cell median marker intensity (a.u.)",
            ylabel=f"Cell median {ylabel}", log_y=log_y,
            annotations={str(r["condition"]): _rho_label(r)
                         for _, r in pc.iterrows() if r["condition"] != "ALL"})

        # --- within cells ---
        per_unit = statistics.correlation_by_unit(
            sub.rename(columns={"_I": "_x", col: "_y"}), "_x", "_y", unit_col=unit_col)
        per_unit.to_csv(adir / f"correlation_per_{design.unit}.csv", index=False)
        within = statistics.correlation_significance(per_unit, design=design)
        if not within.empty:
            within.insert(0, "metric", slug)
            within.to_csv(adir / "correlation_within_condition.csv", index=False)
            within_all.append(within)
        if not per_unit.empty and per_unit["condition"].nunique() >= 2:
            rs = statistics.between_condition_tests(
                per_unit, "rho", label=f"{slug}~marker rho", design=design)
            if not rs.empty:
                rs.to_csv(adir / "correlation_between_conditions.csv", index=False)
                frames.append(rs)

        # --- track scatter ---
        plotting.scatter_with_trend(
            sub, adir / "track_metric_vs_marker.png", x_col="marker_intensity",
            y_col=col, group_col="condition",
            xlabel="Marker intensity (a.u.)", ylabel=ylabel, log_y=log_y,
            annotations={str(r["condition"]): _rho_label(r)
                         for _, r in within.iterrows()} if not within.empty else None)

        # --- binned by expression, and by brightness ---
        frames += _binned_compare(sub, col, ylabel, adir, cfg, design, unit_col,
                                  by="_I", edges=None,
                                  n_q=cfg.marker_intensity_bins,
                                  bin_name="intensity",
                                  xlabel="Marker intensity bin", slug=slug)
        if "mass_per_frame" in sub.columns:
            frames += _binned_compare(sub, col, ylabel, adir, cfg, design, unit_col,
                                      by="mass_per_frame",
                                      edges=cfg.brightness_bin_edges,
                                      n_q=cfg.brightness_bins,
                                      bin_name="brightness",
                                      xlabel="Track brightness bin", slug=slug)

    if within_all:
        pd.concat(within_all, ignore_index=True).to_csv(
            mdir / "correlation_within_condition_all_metrics.csv", index=False)
    if percell_all:
        pd.concat(percell_all, ignore_index=True).to_csv(
            mdir / "correlation_across_cells_all_metrics.csv", index=False)

    # --- is expression itself different between conditions? ---
    per_unit_i = (merged.groupby(["condition", unit_col])["_I"].median()
                  .reset_index().rename(columns={"_I": "median_intensity"}))
    if per_unit_i["condition"].nunique() >= 2:
        i_stats = statistics.between_condition_tests(
            per_unit_i, "median_intensity", label="marker intensity", design=design)
        if not i_stats.empty:
            i_stats.to_csv(mdir / "marker_intensity_between_conditions.csv", index=False)
            frames.append(i_stats)
        plotting.condition_boxplot(
            merged.groupby(["condition", "experiment", "movie_id"])["_I"].median()
            .reset_index().rename(columns={"_I": "median_intensity"}),
            mdir / "marker_intensity_by_condition.png", value_col="median_intensity",
            ylabel="Marker intensity (a.u.)",
            comparisons=plotting.comparisons_from_stats(i_stats))

    utils.dump_json(
        {"n_tracks": int(len(merged)),
         "conditions": sorted(merged["condition"].unique()),
         "unit": design.unit,
         "metrics": [a[2] for a in axes],
         "tau_fit_s": (float(merged["tau_fit_s"].dropna().iloc[0])
                       if "tau_fit_s" in merged.columns
                       and merged["tau_fit_s"].notna().any() else None),
         "intensity_bins": cfg.marker_intensity_bins,
         "brightness_bin_edges": cfg.brightness_bin_edges,
         "min_disk_coverage": cfg.marker_min_disk_coverage,
         "edge_margin_px": cfg.marker_edge_margin_px},
        mdir / "marker_analysis_summary.json")
    return frames


def _brightness_distributions(merged, mdir, design, cfg=None):
    """Brightness distributions: GEM track brightness, and marker disk intensity.

    Two different quantities that are easy to conflate:

    * ``mass_per_frame`` -- how bright the tracked GEM particle is. A proxy for
      how many fluorophores are in the spot and how well it was detected, and a
      known confounder of D, so its distribution is worth inspecting before any
      brightness-matched comparison is trusted.
    * ``marker_intensity`` -- the expression-reporter signal in the disk around
      each track. This is the FICD readout.

    Both get a per-movie overlay inside each condition, because the shape of
    the between-movie spread is the thing a pooled histogram hides: a bimodal
    condition and a heterogeneous-across-cells condition look identical once
    pooled, and they mean entirely different things.
    """
    bdir = mdir / "brightness"; bdir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("mass_per_frame", "track_brightness",
         "Track brightness (mass/frame, a.u.)", True),
        ("marker_intensity", "marker_disk_intensity",
         "Marker disk intensity (a.u.)", False),
    ]
    for col, slug, xlabel, log_x in specs:
        if col not in merged.columns:
            continue
        v = pd.to_numeric(merged[col], errors="coerce")
        if not np.isfinite(v).any():
            continue
        summary = plotting.per_movie_distribution_overlay(
            merged, bdir / f"{slug}_per_movie.png", value_col=col,
            group_col="condition", xlabel=xlabel, log_x=log_x)
        if summary is not None:
            summary.to_csv(bdir / f"{slug}_per_movie_summary.csv", index=False)

        # Pooled across all movies of a condition.
        by_cond = {c: pd.to_numeric(g[col], errors="coerce").dropna().to_numpy()
                   for c, g in merged.groupby("condition")}
        by_cond = {c: a[a > 0] if log_x else a for c, a in by_cond.items() if a.size}
        if not by_cond:
            continue
        allv = np.concatenate(list(by_cond.values()))
        lo, hi = np.nanpercentile(allv, [0.2, 99.8])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        edges = (np.logspace(np.log10(max(lo, 1e-12)), np.log10(hi), 61)
                 if log_x else np.linspace(lo, hi, 61))
        for log_y, sfx in _yscales(cfg):
            plotting.overlay_value_histograms(
                by_cond, edges, bdir / f"{slug}_by_condition{sfx}.png",
                xlabel=xlabel, log_x=log_x, log_y=log_y)
        rows = []
        for c, a in by_cond.items():
            counts, _ = np.histogram(a, bins=edges)
            tot = counts.sum()
            for i in range(len(counts)):
                rows.append({"condition": c, "bin_left": edges[i], "bin_right": edges[i + 1],
                             "count": int(counts[i]),
                             "fraction": counts[i] / tot if tot else 0.0})
        pd.DataFrame(rows).to_csv(bdir / f"{slug}_by_condition_bins.csv", index=False)

        # Between-condition test on the per-unit medians.
        per_unit = (merged.groupby(["condition", design.unit_column])[col]
                    .median().reset_index())
        if per_unit["condition"].nunique() >= 2:
            st = statistics.between_condition_tests(
                per_unit.rename(columns={col: "value"}), "value",
                label=slug, design=design)
            if not st.empty:
                st.to_csv(bdir / f"{slug}_between_conditions.csv", index=False)
            plotting.condition_boxplot(
                merged.groupby(["condition", "experiment", "movie_id"])[col]
                .median().reset_index().rename(columns={col: "value"}),
                bdir / f"{slug}_by_condition_boxplot.png", value_col="value",
                ylabel=xlabel, comparisons=plotting.comparisons_from_stats(st))


def _rho_label(row) -> str:
    """Legend suffix carrying rho and its significance marker, nothing else."""
    rho = row.get("median_rho", row.get("rho"))
    if rho is None or not np.isfinite(rho):
        return ""
    star = plotting.p_to_stars(row.get("p_value_bh", row.get("p_value")))
    return f"ρ = {rho:+.2f}" + ("" if star in ("", "n.s.") else star)


def stats_spearman(x, y):
    from scipy import stats as _st
    x = pd.to_numeric(x, errors="coerce").to_numpy()
    y = pd.to_numeric(y, errors="coerce").to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan, np.nan
    r, p = _st.spearmanr(x[ok], y[ok])
    return float(r), float(p)


def _binned_compare(sub, col, ylabel, adir, cfg, design, unit_col, *, by, edges,
                    n_q, bin_name, xlabel, slug):
    """Compare conditions within bins of ``by`` -- matched-covariate comparison.

    Absolute edges when supplied: matching requires identical boundaries in
    every condition, which quantile bins would not give.
    """
    if by not in sub.columns:
        return []
    labels, table = filters_mod.absolute_bins(
        sub[by], edges=edges, n_quantile=n_q, prefix="Q" if edges is None else "B")
    if table.empty:
        return []
    s = sub.copy()
    s["_bin"] = labels
    s = s.dropna(subset=["_bin"])
    if s.empty:
        return []
    table.to_csv(adir / f"{bin_name}_bin_edges.csv", index=False)

    per_bin = (s.groupby(["_bin", "condition", unit_col], observed=True)[col]
               .median().reset_index())
    per_bin.to_csv(adir / f"{bin_name}_bins_per_{design.unit}.csv", index=False)

    comps, out = {}, []
    for b, g in per_bin.groupby("_bin", observed=True):
        if g["condition"].nunique() < 2:
            continue
        st = statistics.between_condition_tests(
            g.rename(columns={col: "value"}), "value",
            label=f"{slug} | {bin_name} {b}", design=design)
        if not st.empty:
            st[f"{bin_name}_bin"] = str(b)
            out.append(st)
            comps[b] = plotting.comparisons_from_stats(st)
    if out:
        pd.concat(out, ignore_index=True).to_csv(
            adir / f"between_conditions_by_{bin_name}_bin.csv", index=False)
    plotting.grouped_boxplot(
        s, adir / f"{slug}_by_{bin_name}_bin.png", x_col="_bin",
        group_col="condition", value_col=col, xlabel=xlabel, ylabel=ylabel,
        x_order=list(table["bin"]), comparisons_within_x=comps)
    return out


def _pkey(value):
    """Particle id normalized to text, so merges survive int/float round trips."""
    try:
        f = float(value)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(value)


def _condition_summary(manifest_df, diff_pool, root, d_col, design=None):
    rows = []
    for cond, sub in manifest_df.groupby("condition"):
        n_exp = sub["experiment"].nunique()
        n_movies = len(sub)
        med_d = np.nan
        if not diff_pool.empty and d_col in diff_pool.columns:
            dvals = pd.to_numeric(diff_pool.loc[diff_pool["condition"] == cond, d_col],
                                  errors="coerce").to_numpy()
            dvals = dvals[np.isfinite(dvals)]
            med_d = float(np.median(dvals)) if dvals.size else np.nan
        row = {"condition": cond, "n_experiments": n_exp, "n_movies": n_movies,
               "median_D_um2_s": med_d}
        if design is not None:
            row["statistical_unit"] = design.unit
            row["n_statistical_units"] = (n_movies if design.is_within_experiment else n_exp)
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / "condition_summary.csv", index=False)


def _config_dict(cfg: CompareConfig) -> dict:
    """Config as JSON, with the filter expanded into its windows.

    The exact selection has to survive into ``run_parameters.json`` -- a
    filtered figure whose criteria are not recorded beside it cannot be
    reproduced or defended six months later.
    """
    out = {}
    for k, v in cfg.__dict__.items():
        if k == "arms":
            out["arms"] = sorted(v) if v else sorted(ARMS)
        elif k == "track_filter":
            out["track_filter"] = (v.describe() if v is not None and bool(v) else None)
            out["track_filter_drop_missing"] = bool(getattr(v, "drop_missing", False))
        else:
            out[k] = v
    return out
