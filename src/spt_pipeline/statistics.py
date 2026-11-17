"""Statistical framework with an automatically chosen unit of inference.

The hierarchy: a *track* is the unit of measurement, an *experiment* (imaging
day / biological replicate) is the unit of statistical inference, and a
*condition* is the unit of comparison. Per-experiment summaries feed the
hypothesis tests; we deliberately do **not** treat individual steps or movies
as independent biological replicates.

Choosing the design
-------------------
That hierarchy only holds when the dataset actually *contains* replicate
experiments. Run a single imaging day through an experiment-level test and
every condition contributes n = 1, which yields p-values that are either
undefined or meaningless -- printed on a figure they are worse than nothing,
because they look authoritative.

:func:`infer_design` reads the manifest's ``experiment`` column and picks:

* **between-experiment** -- every condition has >= 2 experiments. The unit is
  the experiment (biological replicate). This is the design that supports
  claims about biology.
* **within-experiment** -- at least one condition has a single experiment. The
  unit falls back to the *movie* (technical replicate), and every figure and
  table says so in as many words. A significant result here means the
  conditions differ *within this one experiment*; whether the effect
  reproduces across biological replicates is simply not addressed.

Both designs run the same Mann-Whitney machinery, so the only thing that
changes is the grouping key and -- crucially -- the label.

Track-level distributional tests (KS on the pooled per-track values) are always
computed and written to their own CSV. They are anti-conservative by
construction: n is the number of tracks, which is enormous, so p-values collapse
toward zero and describe the *distributions*, not the replicates. They are kept
out of the figures on purpose and live in
``track_level_ks_tests.csv`` for when a distributional claim is what's wanted.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

#: Column used as the statistical unit under each design.
UNIT_COLUMN = {"between-experiment": "experiment", "within-experiment": "movie_id"}


@dataclass(frozen=True)
class StatDesign:
    """Which statistical unit applies, and the language to describe it."""
    design: str                     # 'between-experiment' | 'within-experiment'
    unit: str                       # 'experiment' | 'movie'
    unit_column: str                # 'experiment' | 'movie_id'
    n_experiments: int
    n_experiments_min_per_condition: int
    reason: str

    @property
    def is_within_experiment(self) -> bool:
        return self.design == "within-experiment"

    @property
    def unit_noun(self) -> str:
        return "movies" if self.is_within_experiment else "experiments"

    @property
    def short_label(self) -> str:
        return ("within-experiment (technical replicates)"
                if self.is_within_experiment else "experiment-level (biological replicates)")

    @property
    def caveat(self) -> str:
        if self.is_within_experiment:
            return ("Single experiment: unit = movie (technical replicate). "
                    "Not a biological-replicate test.")
        return f"Unit = experiment (biological replicate); n = {self.n_experiments}."

    def to_dict(self) -> Dict[str, object]:
        return {"design": self.design, "unit": self.unit, "unit_column": self.unit_column,
                "n_experiments": self.n_experiments,
                "n_experiments_min_per_condition": self.n_experiments_min_per_condition,
                "reason": self.reason, "caveat": self.caveat}


def infer_design(df: pd.DataFrame, min_experiments: int = 2) -> StatDesign:
    """Pick the statistical design from the manifest's ``experiment`` column.

    The gate is the *minimum* number of experiments across conditions, not the
    total. Two experiments split as 2-and-1 across conditions still cannot
    support an experiment-level comparison -- one arm has n = 1 -- so that case
    correctly falls back to within-experiment.
    """
    if df is None or df.empty or "experiment" not in df.columns:
        return StatDesign("within-experiment", "movie", "movie_id", 0, 0,
                          "no experiment column available")
    n_total = int(df["experiment"].nunique())
    if "condition" in df.columns and df["condition"].nunique() > 0:
        per_cond = df.groupby("condition")["experiment"].nunique()
        n_min = int(per_cond.min()) if len(per_cond) else 0
    else:
        n_min = n_total
    if n_min >= min_experiments:
        return StatDesign("between-experiment", "experiment", "experiment", n_total, n_min,
                          f"every condition has >= {min_experiments} experiments")
    if n_total <= 1:
        reason = "manifest contains a single experiment"
    else:
        reason = (f"{n_total} experiments overall, but at least one condition has only "
                  f"{n_min}; an experiment-level test would compare n=1 against n>1")
    return StatDesign("within-experiment", "movie", "movie_id", n_total, n_min, reason)


# ---------------------------------------------------------------------------
# Per-experiment summaries
# ---------------------------------------------------------------------------

def per_unit_step_summary(
    step_df: pd.DataFrame, unit_col: str = "experiment",
    confinement_threshold_um: Optional[float] = None,
    track_keys: Sequence[str] = ("movie_id", "particle"),
) -> pd.DataFrame:
    """Summarize step sizes per (condition, unit), where unit is the design's.

    ``unit_col`` is ``experiment`` for a replicated dataset and ``movie_id``
    when only one experiment exists. The output always carries an ``experiment``
    column as well, so figures can still colour points by imaging day even when
    the movie is the statistical unit.
    """
    cols = ["condition", unit_col, "median_step_um", "mean_step_um", "iqr_step_um",
            "frac_tracks_below_threshold", "n_tracks", "n_displacements"]
    if unit_col != "experiment":
        cols.insert(2, "experiment")
    if step_df.empty or not {"condition", unit_col, "step_um"}.issubset(step_df.columns):
        return pd.DataFrame(columns=cols)
    keys = [k for k in track_keys if k in step_df.columns]
    rows = []
    for (cond, unit), grp in step_df.groupby(["condition", unit_col]):
        vals = pd.to_numeric(grp["step_um"], errors="coerce").to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        frac_below = np.nan
        n_tracks = grp.groupby(keys).ngroups if keys else np.nan
        if confinement_threshold_um is not None and keys:
            track_med = grp.groupby(keys)["step_um"].median()
            frac_below = float((track_med < confinement_threshold_um).mean())
        row = {
            "condition": cond, unit_col: unit,
            "median_step_um": float(np.median(vals)),
            "mean_step_um": float(np.mean(vals)),
            "iqr_step_um": float(q3 - q1),
            "frac_tracks_below_threshold": frac_below,
            "n_tracks": n_tracks, "n_displacements": int(vals.size),
        }
        if unit_col != "experiment" and "experiment" in grp.columns:
            row["experiment"] = str(grp["experiment"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def per_unit_d_summary(diff_df: pd.DataFrame, unit_col: str = "experiment",
                       d_col: str = "D_fit_um2_s") -> pd.DataFrame:
    """Median/mean D per (condition, unit)."""
    cols = ["condition", unit_col, "median_D", "mean_D", "n_tracks"]
    if unit_col != "experiment":
        cols.insert(2, "experiment")
    if diff_df.empty or not {"condition", unit_col}.issubset(diff_df.columns) \
            or d_col not in diff_df.columns:
        return pd.DataFrame(columns=cols)
    rows = []
    for (cond, unit), grp in diff_df.groupby(["condition", unit_col]):
        vals = pd.to_numeric(grp[d_col], errors="coerce").to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        row = {"condition": cond, unit_col: unit,
               "median_D": float(np.median(vals)), "mean_D": float(np.mean(vals)),
               "n_tracks": int(vals.size)}
        if unit_col != "experiment" and "experiment" in grp.columns:
            row["experiment"] = str(grp["experiment"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


# Backwards-compatible aliases: the experiment-unit special case.
def per_experiment_step_summary(
    step_df: pd.DataFrame, confinement_threshold_um: Optional[float] = None,
    track_keys: Sequence[str] = ("movie_id", "particle"),
) -> pd.DataFrame:
    """Summarize step sizes per (condition, experiment)."""
    return per_unit_step_summary(step_df, "experiment",
                                 confinement_threshold_um=confinement_threshold_um,
                                 track_keys=track_keys)


def per_experiment_d_summary(diff_df: pd.DataFrame, d_col: str = "D_fit_um2_s") -> pd.DataFrame:
    """Median/mean D per (condition, experiment)."""
    return per_unit_d_summary(diff_df, "experiment", d_col=d_col)


# ---------------------------------------------------------------------------
# Effect size / bootstrap
# ---------------------------------------------------------------------------

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size in [-1, 1] (nonparametric, rank-based)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    # rank-based O(n log n) computation
    all_vals = np.concatenate([a, b])
    ranks = stats.rankdata(all_vals)
    ra = ranks[:a.size].sum()
    u = ra - a.size * (a.size + 1) / 2.0
    return float(2.0 * u / (a.size * b.size) - 1.0)


def bootstrap_median_ci(values: Sequence[float], n_boot: int = 5000, ci: float = 0.95,
                        random_state: int = 0) -> Dict[str, float]:
    """Bootstrap CI for a median (resampling the given values, e.g. per-experiment)."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"median": np.nan, "lo": np.nan, "hi": np.nan}
    rng = np.random.default_rng(random_state)
    boots = np.median(rng.choice(vals, size=(n_boot, vals.size), replace=True), axis=1)
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"median": float(np.median(vals)), "lo": float(lo), "hi": float(hi)}


# ---------------------------------------------------------------------------
# Between-condition tests
# ---------------------------------------------------------------------------

def _bh_correct(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out.tolist()


def between_condition_tests(
    per_unit: pd.DataFrame, value_col: str, pooled: Optional[Dict[str, np.ndarray]] = None,
    label: str = "", design: Optional[StatDesign] = None,
) -> pd.DataFrame:
    """Pairwise Mann-Whitney on per-unit summaries, optional KS on pooled values.

    ``per_unit`` must have ``condition`` and ``value_col``, one row per
    statistical unit -- an experiment under the between-experiment design, a
    movie under the within-experiment design. ``design`` supplies the labelling
    so a reader of the CSV can never mistake a technical comparison for a
    biological one; omitting it keeps the historical experiment-unit wording.

    ``pooled`` maps condition -> pooled per-track/per-step array for the
    (anti-conservative) KS distributional test carried alongside.
    """
    unit = design.unit if design else "experiment"
    design_name = design.design if design else "between-experiment"
    rows = []
    conditions = sorted(per_unit["condition"].unique()) if not per_unit.empty else []
    raw_p = []
    for c1, c2 in combinations(conditions, 2):
        a = pd.to_numeric(per_unit.loc[per_unit["condition"] == c1, value_col],
                          errors="coerce").dropna().to_numpy()
        b = pd.to_numeric(per_unit.loc[per_unit["condition"] == c2, value_col],
                          errors="coerce").dropna().to_numpy()
        row = {"metric": label or value_col, "condition_a": c1, "condition_b": c2,
               # 'n_exp_a/b' keeps its historical name (many downstream readers
               # and figure annotations use it) but now counts whichever unit
               # the design selected; 'unit' says which that is.
               "n_exp_a": int(a.size), "n_exp_b": int(b.size),
               "unit": unit, "design": design_name, "test": "Mann-Whitney U",
               "statistic": np.nan, "p_value": np.nan, "cliffs_delta": np.nan,
               "ks_stat": np.nan, "ks_p": np.nan,
               "interpretation": (design.caveat if design else "")}
        if a.size >= 1 and b.size >= 1 and (a.size + b.size) >= 3:
            try:
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                row["statistic"], row["p_value"] = float(u), float(p)
                raw_p.append(float(p))
            except ValueError:
                raw_p.append(np.nan)
            row["cliffs_delta"] = cliffs_delta(a, b)
        else:
            raw_p.append(np.nan)
        if pooled and c1 in pooled and c2 in pooled:
            pa, pb = pooled[c1], pooled[c2]
            if len(pa) and len(pb):
                ks, ksp = stats.ks_2samp(pa, pb)
                row["ks_stat"], row["ks_p"] = float(ks), float(ksp)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and len(conditions) > 2:
        df["p_value_bh"] = _bh_correct(raw_p)
    elif not df.empty:
        df["p_value_bh"] = df["p_value"]
    return df


def correlation_by_unit(
    df: pd.DataFrame, x_col: str, y_col: str, unit_col: str = "movie_id",
    min_n: int = 8,
) -> pd.DataFrame:
    """Spearman correlation of ``x`` vs ``y`` computed *within* each unit.

    Computing one correlation over all pooled tracks would treat every track as
    independent, which they are not -- tracks from one movie share a cell, a
    field of view, and an illumination profile. Correlating within a unit and
    then treating the resulting coefficients as the observations respects that
    hierarchy: the question becomes "does this relationship hold repeatedly?"
    rather than "is there any structure in this giant pile of points?".
    """
    cols = ["condition", unit_col, "rho", "p_value", "n_tracks"]
    need = {"condition", unit_col, x_col, y_col}
    if df is None or df.empty or not need.issubset(df.columns):
        return pd.DataFrame(columns=cols)
    rows = []
    for (cond, unit), g in df.groupby(["condition", unit_col]):
        x = pd.to_numeric(g[x_col], errors="coerce").to_numpy()
        y = pd.to_numeric(g[y_col], errors="coerce").to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < min_n:
            continue
        try:
            rho, p = stats.spearmanr(x[ok], y[ok])
        except ValueError:
            continue
        if not np.isfinite(rho):
            continue
        rows.append({"condition": cond, unit_col: unit, "rho": float(rho),
                     "p_value": float(p), "n_tracks": int(ok.sum())})
    return pd.DataFrame(rows, columns=cols)


def correlation_significance(
    per_unit: pd.DataFrame, design: Optional[StatDesign] = None,
) -> pd.DataFrame:
    """Is the within-unit correlation reproducibly non-zero, per condition?

    A one-sample Wilcoxon signed-rank test on the per-unit coefficients. The
    null is that the relationship is absent -- coefficients scattered around
    zero -- so a significant result means the association recurs across units
    rather than appearing once and being averaged in.

    With fewer than six units Wilcoxon cannot reach p < 0.05 at all; the row is
    still emitted with the coefficients and a NaN p, so the reader sees the
    effect and the reason it is untested rather than an absent row.
    """
    unit = design.unit if design else "unit"
    cols = ["condition", "median_rho", "n_units", "unit", "test", "p_value", "note"]
    if per_unit is None or per_unit.empty:
        return pd.DataFrame(columns=cols)
    rows, raw_p = [], []
    for cond, g in per_unit.groupby("condition"):
        r = pd.to_numeric(g["rho"], errors="coerce").dropna().to_numpy()
        row = {"condition": cond, "median_rho": float(np.median(r)) if r.size else np.nan,
               "n_units": int(r.size), "unit": unit,
               "test": "Wilcoxon signed-rank vs 0", "p_value": np.nan, "note": ""}
        if r.size >= 6 and np.any(r != 0):
            try:
                _, p = stats.wilcoxon(r)
                row["p_value"] = float(p)
            except ValueError as exc:
                row["note"] = str(exc)
        else:
            row["note"] = f"only {r.size} {unit}(s); too few for a signed-rank test"
        raw_p.append(row["p_value"])
        rows.append(row)
    out = pd.DataFrame(rows, columns=cols)
    if len(out) > 1:
        out["p_value_bh"] = _bh_correct([p if p == p else 1.0 for p in raw_p])
    else:
        out["p_value_bh"] = out["p_value"]
    return out


def track_level_tests(
    pooled: Dict[str, np.ndarray], label: str = "", value_name: str = "value",
) -> pd.DataFrame:
    """Pairwise KS + Mann-Whitney + Cliff's delta on pooled per-track values.

    Deliberately separate from :func:`between_condition_tests` and deliberately
    kept off the figures. Here n is the number of tracks, so essentially any
    difference reaches significance; the p-value is a statement about the two
    *distributions*, not about whether the effect reproduces. Cliff's delta is
    the column to read -- it is an effect size and does not inflate with n.
    """
    conditions = sorted(pooled.keys())
    rows, raw_p = [], []
    for c1, c2 in combinations(conditions, 2):
        a = np.asarray(pooled[c1], dtype=float); a = a[np.isfinite(a)]
        b = np.asarray(pooled[c2], dtype=float); b = b[np.isfinite(b)]
        row = {"metric": label or value_name, "condition_a": c1, "condition_b": c2,
               "n_tracks_a": int(a.size), "n_tracks_b": int(b.size),
               "unit": "track", "design": "pooled-tracks",
               "ks_stat": np.nan, "ks_p": np.nan,
               "mannwhitney_u": np.nan, "mannwhitney_p": np.nan,
               "cliffs_delta": np.nan,
               "median_a": float(np.median(a)) if a.size else np.nan,
               "median_b": float(np.median(b)) if b.size else np.nan,
               "interpretation": ("Pooled across tracks: anti-conservative, treats each "
                                  "track as independent. Read cliffs_delta, not p.")}
        if a.size and b.size:
            try:
                ks, ksp = stats.ks_2samp(a, b)
                row["ks_stat"], row["ks_p"] = float(ks), float(ksp)
            except ValueError:
                pass
            try:
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                row["mannwhitney_u"], row["mannwhitney_p"] = float(u), float(p)
                raw_p.append(float(p))
            except ValueError:
                raw_p.append(np.nan)
            row["cliffs_delta"] = cliffs_delta(a, b)
        else:
            raw_p.append(np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["mannwhitney_p_bh"] = (_bh_correct(raw_p) if len(conditions) > 2
                                  else df["mannwhitney_p"])
    return df
