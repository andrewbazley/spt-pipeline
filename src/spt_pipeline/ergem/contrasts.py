"""Factored WT-vs-E234G x DMSO-vs-Tg contrasts for the FICD experiment.

The core :mod:`spt_pipeline.compare` runs *all* pairwise condition tests, which
for six conditions yields many biologically meaningless pairs. Here we compute
only the contrasts that answer the question -- does inactivating BiP (E234G)
blunt the Tg-induced rise in ER-GEM diffusion -- keeping the experiment (imaging
day) as the unit of statistical inference, exactly as the core does.

Families of contrasts (each with Mann-Whitney on per-experiment median D,
Cliff's delta effect size, and BH correction within the family):

  1. complementation within treatment  : WT vs E234G under DMSO; under Tg
  2. treatment within complementation  : Tg vs DMSO in WT; in E234G
  3. interaction (difference-in-diffs)  : per-experiment (Tg - DMSO) median D,
     WT vs E234G  -> the direct test of the blunting hypothesis
  4. baseline vs untransfected WT       : optional reference contrasts

Also provided: clone-resolved versions (A8/D2 powered; C11 flagged N=2) and an
optional statsmodels mixed-effects model with experiment and clone as random
effects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .. import statistics as core_stats
from .crosscorr import load_diffusion_per_track


# ---------------------------------------------------------------------------
# Label detection (robust to exact spelling in the map)
# ---------------------------------------------------------------------------

def detect_labels(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    compl = [c for c in df["complementation"].dropna().unique() if str(c).strip()]
    wt = next((c for c in compl if "WT" in str(c).upper()), None)
    eg = next((c for c in compl if "E234G" in str(c).upper() or c.upper().endswith("EG")), None)
    treat = [t for t in df["treatment_code"].dropna().unique() if str(t).strip()]
    dmso = next((t for t in treat if "DMSO" in str(t).upper() or "NT" in str(t).upper()), None)
    tg = next((t for t in treat if str(t).upper().startswith("TG")), None)
    return {"WT": wt, "EG": eg, "DMSO": dmso, "Tg": tg}


# ---------------------------------------------------------------------------
# Per-experiment medians
# ---------------------------------------------------------------------------

def per_experiment_median(df: pd.DataFrame, group_cols: List[str], d_col: str) -> pd.DataFrame:
    keys = group_cols + ["experiment"]
    rows = []
    for vals, grp in df.groupby(keys):
        vals = vals if isinstance(vals, tuple) else (vals,)
        d = pd.to_numeric(grp[d_col], errors="coerce").to_numpy()
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        row = dict(zip(keys, vals))
        row["median_D"] = float(np.median(d))
        row["n_tracks"] = int(d.size)
        rows.append(row)
    return pd.DataFrame(rows)


def _mw(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size < 1 or b.size < 1 or (a.size + b.size) < 3:
        return np.nan, np.nan, core_stats.cliffs_delta(a, b)
    try:
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        return np.nan, np.nan, core_stats.cliffs_delta(a, b)
    return float(u), float(p), core_stats.cliffs_delta(a, b)


def _contrast_row(family: str, name: str, per_exp: pd.DataFrame,
                  col: str, va, vb, value_col: str = "median_D") -> dict:
    a = pd.to_numeric(per_exp.loc[per_exp[col] == va, value_col], errors="coerce").to_numpy()
    b = pd.to_numeric(per_exp.loc[per_exp[col] == vb, value_col], errors="coerce").to_numpy()
    u, p, delta = _mw(a, b)
    return {"family": family, "contrast": name, "group_a": va, "group_b": vb,
            "n_exp_a": int(np.isfinite(a).sum()), "n_exp_b": int(np.isfinite(b).sum()),
            "median_a": float(np.nanmedian(a)) if a.size else np.nan,
            "median_b": float(np.nanmedian(b)) if b.size else np.nan,
            "unit": "experiment", "test": "Mann-Whitney U",
            "U": u, "p_value": p, "cliffs_delta": delta}


def _bh_within_family(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["p_value_bh"] = np.nan
    for fam, idx in df.groupby("family").groups.items():
        sub = df.loc[idx, "p_value"].tolist()
        df.loc[idx, "p_value_bh"] = core_stats._bh_correct(sub)
    return df


# ---------------------------------------------------------------------------
# Factored contrasts
# ---------------------------------------------------------------------------

def factored_contrasts(df: pd.DataFrame, d_col: str, include_baseline: bool = True) -> pd.DataFrame:
    lab = detect_labels(df)
    WT, EG, DMSO, Tg = lab["WT"], lab["EG"], lab["DMSO"], lab["Tg"]
    rows: List[dict] = []

    # Family 1: complementation within treatment
    pe_ct = per_experiment_median(df, ["complementation", "treatment_code"], d_col)
    for tr in [DMSO, Tg]:
        if tr is None:
            continue
        sub = pe_ct[pe_ct["treatment_code"] == tr]
        rows.append(_contrast_row("complementation_within_treatment",
                                  f"WT_vs_E234G@{tr}", sub, "complementation", WT, EG))

    # Family 2: treatment within complementation
    for cp in [WT, EG]:
        if cp is None:
            continue
        sub = pe_ct[pe_ct["complementation"] == cp]
        rows.append(_contrast_row("treatment_within_complementation",
                                  f"Tg_vs_DMSO@{cp}", sub, "treatment_code", Tg, DMSO))

    # Family 3: interaction (difference-in-differences) on per-experiment deltas
    if all(v is not None for v in (WT, EG, DMSO, Tg)):
        deltas = []
        for cp in (WT, EG):
            sub = pe_ct[pe_ct["complementation"] == cp]
            piv = sub.pivot_table(index="experiment", columns="treatment_code",
                                  values="median_D", aggfunc="first")
            if Tg in piv.columns and DMSO in piv.columns:
                d = (piv[Tg] - piv[DMSO]).dropna()
                for exp, val in d.items():
                    deltas.append({"complementation": cp, "experiment": exp, "delta_D": float(val)})
        dd = pd.DataFrame(deltas)
        if not dd.empty:
            a = dd.loc[dd["complementation"] == WT, "delta_D"].to_numpy()
            b = dd.loc[dd["complementation"] == EG, "delta_D"].to_numpy()
            u, p, delta = _mw(a, b)
            rows.append({"family": "interaction", "contrast": "deltaTg_WT_vs_E234G",
                         "group_a": f"{WT} (Tg-DMSO)", "group_b": f"{EG} (Tg-DMSO)",
                         "n_exp_a": int(np.isfinite(a).sum()), "n_exp_b": int(np.isfinite(b).sum()),
                         "median_a": float(np.nanmedian(a)) if a.size else np.nan,
                         "median_b": float(np.nanmedian(b)) if b.size else np.nan,
                         "unit": "experiment (paired delta)", "test": "Mann-Whitney U",
                         "U": u, "p_value": p, "cliffs_delta": delta})

    # Family 4: baseline vs untransfected WT (per treatment)
    if include_baseline:
        pe_cond = per_experiment_median(df, ["condition"], d_col)
        untrans = [c for c in df["condition"].unique() if "untrans" in str(c).lower()]
        complemented = [c for c in df["condition"].unique() if "untrans" not in str(c).lower()]
        for u_cond in untrans:
            u_treat = "Tg" if "tg" in str(u_cond).lower() else ("DMSO" if "dmso" in str(u_cond).lower() else "")
            for c_cond in complemented:
                if u_treat and u_treat.lower() not in str(c_cond).lower():
                    continue
                rows.append(_contrast_row("baseline_vs_untransfected",
                                          f"{c_cond}_vs_{u_cond}", pe_cond, "condition", c_cond, u_cond))

    out = pd.DataFrame(rows)
    return _bh_within_family(out)


def clone_resolved_contrasts(df: pd.DataFrame, d_col: str) -> pd.DataFrame:
    frames = []
    for clone, sub in df.groupby("clone"):
        if not str(clone).strip():
            continue
        fc = factored_contrasts(sub, d_col, include_baseline=False)
        if fc.empty:
            continue
        fc.insert(0, "clone", clone)
        n_exp = sub["experiment"].nunique()
        fc["low_power"] = n_exp < 3
        fc["n_experiments_in_clone"] = n_exp
        frames.append(fc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def mixed_effects(df: pd.DataFrame, d_col: str) -> pd.DataFrame:
    """Optional MixedLM: log10 D ~ complementation*treatment, RE=experiment(+clone)."""
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:  # pragma: no cover
        return pd.DataFrame([{"note": f"statsmodels unavailable: {exc}"}])
    lab = detect_labels(df)
    WT, EG, DMSO = lab["WT"], lab["EG"], lab["DMSO"]
    d = df.copy()
    d = d[d["complementation"].astype(str).str.strip() != ""]  # drop untransfected
    d["Dv"] = pd.to_numeric(d[d_col], errors="coerce")
    d = d[np.isfinite(d["Dv"]) & (d["Dv"] > 0)]
    if d.empty or WT is None or EG is None:
        return pd.DataFrame([{"note": "insufficient data for mixed model"}])
    d["log10_D"] = np.log10(d["Dv"])
    d["compl"] = pd.Categorical(d["complementation"],
                                categories=[c for c in (WT, EG) if c in set(d["complementation"])])
    d["treat"] = d["treatment_code"].astype(str)
    if DMSO is not None:
        cats = [DMSO] + [t for t in d["treat"].unique() if t != DMSO]
        d["treat"] = pd.Categorical(d["treat"], categories=cats)
    import warnings
    try:
        vc = {"clone": "0 + C(clone)"} if d["clone"].astype(str).str.strip().nunique() > 1 else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm("log10_D ~ C(compl) * C(treat)", data=d,
                                groups=d["experiment"], vc_formula=vc)
            fit = model.fit(reml=True, method="lbfgs")
        res = pd.DataFrame({"term": fit.params.index, "coef": fit.params.values,
                            "std_err": fit.bse.reindex(fit.params.index).values,
                            "p_value": fit.pvalues.reindex(fit.params.index).values})
        res["note"] = "log10 D; RE=experiment; VC=clone" if vc else "log10 D; RE=experiment"
        return res
    except Exception as exc:
        return pd.DataFrame([{"note": f"mixed model failed: {exc}"}])


def run(manifest_df: pd.DataFrame, out_dir: Path, d_col: str = "D_fit_um2_s",
        do_mixed: bool = False, n_jobs: int = 8) -> Path:
    out_dir = Path(out_dir) / "ficd_contrasts"
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = load_diffusion_per_track(manifest_df, d_col=d_col, n_jobs=n_jobs)
    if pool.empty:
        pd.DataFrame([{"note": "no diffusion data found; run ingest first"}]).to_csv(
            out_dir / "factored_contrasts.csv", index=False)
        return out_dir
    per_exp = per_experiment_median(pool, ["condition", "complementation", "treatment_code", "clone"], d_col)
    per_exp.to_csv(out_dir / "per_experiment_D.csv", index=False)
    factored_contrasts(pool, d_col).to_csv(out_dir / "factored_contrasts.csv", index=False)
    clone_resolved_contrasts(pool, d_col).to_csv(out_dir / "clone_resolved_contrasts.csv", index=False)
    if do_mixed:
        mixed_effects(pool, d_col).to_csv(out_dir / "mixed_effects_summary.csv", index=False)
    return out_dir
