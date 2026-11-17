"""Diffusion-versus-FICD expression joins and correlation summaries.

The per-track table is preserved for visualization and track-level modeling.
Additive per-cell outputs aggregate diffusion within each cell before testing,
so one cellular expression measurement is not pseudoreplicated across tracks.
Valid ``marker_status=no_signal`` rows remain eligible when ``qc_pass=True``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

FICD_VALUE_COL = "ficd_bgsub_intensity"
LEGACY_FICD_VALUE_COL = "ficd_bgsub_mean"
ANALYSIS_VERSION = "3.1.0-adaptive-track-territory"
FACTOR_COLS = ["condition", "experiment", "complementation", "treatment_code", "clone"]


def _read_one_diff(job):
    diff_csv, d_col, extra_cols, movie_id, factors = job
    if not diff_csv or not Path(diff_csv).exists():
        return None
    try:
        table = pd.read_csv(diff_csv, low_memory=False)
    except (OSError, pd.errors.ParserError):
        return None
    if table.empty or "particle" not in table.columns or d_col not in table.columns:
        return None
    columns = ["particle", d_col] + [c for c in extra_cols if c in table.columns]
    kept = table[columns].copy()
    kept["movie_id"] = str(movie_id)
    for column, value in factors.items():
        kept[column] = value
    return kept


def load_diffusion_per_track(
    manifest_df: pd.DataFrame,
    d_col: str = "D_fit_um2_s",
    extra_cols=None,
    n_jobs: int = 8,
    progress: bool = True,
) -> pd.DataFrame:
    from . import _util

    extra_cols = list(extra_cols or [])
    jobs = [
        (
            str(row.get("diff_csv", "")).strip(),
            d_col,
            extra_cols,
            row["movie_id"],
            {column: row.get(column, "") for column in FACTOR_COLS},
        )
        for _, row in manifest_df.iterrows()
    ]
    parts = _util.pmap(
        _read_one_diff,
        jobs,
        n_jobs=n_jobs,
        desc="read diffusion",
        mode="thread",
        progress=progress,
    )
    frames = [part for part in parts if part is not None]
    if not frames:
        return pd.DataFrame()
    output = pd.concat(frames, ignore_index=True)
    output["particle"] = pd.to_numeric(output["particle"], errors="coerce")
    return output


def load_expression(
    expression_csv: Path,
    marker_to_movie: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    try:
        table = pd.read_csv(expression_csv, low_memory=False)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()
    if table.empty:
        return table
    if "movie_id" not in table.columns:
        if "image" not in table.columns:
            raise ValueError("trajectory_expression.csv lacks movie_id and image")

        def movie_id(image: str) -> str:
            basename = str(image)
            if marker_to_movie and basename in marker_to_movie:
                return marker_to_movie[basename]
            stem = Path(basename).stem
            return stem.removeprefix("Red_").removeprefix("Traj_")

        table["movie_id"] = table["image"].map(movie_id)
    table["Trajectory"] = pd.to_numeric(table.get("Trajectory"), errors="coerce")
    if FICD_VALUE_COL not in table.columns and LEGACY_FICD_VALUE_COL in table.columns:
        table[FICD_VALUE_COL] = pd.to_numeric(
            table[LEGACY_FICD_VALUE_COL], errors="coerce"
        ).clip(lower=0)
    return table


def _correlation_row(table: pd.DataFrame, d_col: str) -> Dict[str, float]:
    from scipy import stats

    diffusion = pd.to_numeric(table[d_col], errors="coerce")
    expression = pd.to_numeric(table[FICD_VALUE_COL], errors="coerce")
    valid = np.isfinite(diffusion) & np.isfinite(expression)
    n = int(valid.sum())
    if n < 3 or diffusion[valid].nunique() < 2 or expression[valid].nunique() < 2:
        return {
            "n": n,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "n_positive_log": 0,
            "pearson_logD_logFICD_r": np.nan,
            "pearson_p": np.nan,
        }

    rho, spearman_p = stats.spearmanr(diffusion[valid], expression[valid])
    log_d = np.log10(diffusion[valid].where(lambda values: values > 0))
    log_f = np.log10(expression[valid].where(lambda values: values > 0))
    positive = np.isfinite(log_d) & np.isfinite(log_f)
    n_positive = int(positive.sum())
    if n_positive >= 3 and log_d[positive].nunique() > 1 and log_f[positive].nunique() > 1:
        pearson_r, pearson_p = stats.pearsonr(log_d[positive], log_f[positive])
    else:
        pearson_r, pearson_p = np.nan, np.nan
    return {
        "n": n,
        "spearman_rho": float(rho),
        "spearman_p": float(spearman_p),
        "n_positive_log": n_positive,
        "pearson_logD_logFICD_r": float(pearson_r),
        "pearson_p": float(pearson_p),
    }


def _group_correlations(
    table: pd.DataFrame,
    group_columns: list[str],
    d_col: str,
    n_name: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    grouper = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, subset in table.groupby(grouper, dropna=False):
        keys = (keys,) if len(group_columns) == 1 else tuple(keys)
        row = dict(zip(group_columns, keys))
        row.update(_correlation_row(subset, d_col))
        row[n_name] = row.pop("n")
        rows.append(row)
    return pd.DataFrame(rows)


def cross_correlate(
    diffusion_per_track: pd.DataFrame,
    expression: pd.DataFrame,
    d_col: str = "D_fit_um2_s",
    require_qc_pass: bool = True,
    require_assigned: bool = True,
) -> Dict[str, pd.DataFrame]:
    empty = {
        "per_track": pd.DataFrame(),
        "per_cell": pd.DataFrame(),
        "per_condition": pd.DataFrame(),
        "per_experiment": pd.DataFrame(),
        "per_condition_cells": pd.DataFrame(),
        "per_experiment_cells": pd.DataFrame(),
    }
    if diffusion_per_track.empty or expression.empty:
        return empty

    expression = expression.copy()
    if FICD_VALUE_COL not in expression.columns and LEGACY_FICD_VALUE_COL in expression.columns:
        expression[FICD_VALUE_COL] = pd.to_numeric(
            expression[LEGACY_FICD_VALUE_COL], errors="coerce"
        ).clip(lower=0)
    if require_qc_pass and "qc_pass" in expression.columns:
        expression = expression[
            expression["qc_pass"].astype(str).str.lower().isin(["true", "1", "1.0"])
        ]
    if require_assigned and "assignment_qc" in expression.columns:
        expression = expression[
            expression["assignment_qc"].astype(str).str.lower() == "assigned"
        ]
    if FICD_VALUE_COL not in expression.columns:
        return empty
    expression = expression.dropna(subset=["movie_id", "Trajectory", FICD_VALUE_COL])
    if expression.empty:
        return empty

    optional = [
        column
        for column in (
            "cell_id",
            "analysis_version",
            "assignment_fraction",
            "marker_detected",
            "marker_status",
            "perimeter_found",
            "background_peak",
            "background_sigma",
            "signal_estimator",
            "growth_stop_reason",
            "warning_flags",
            "phenotype_flags",
        )
        if column in expression.columns
    ]
    expression = expression.drop_duplicates(["movie_id", "Trajectory"])
    merged = diffusion_per_track.merge(
        expression[["movie_id", "Trajectory", FICD_VALUE_COL] + optional],
        left_on=["movie_id", "particle"],
        right_on=["movie_id", "Trajectory"],
        how="inner",
    )
    if merged.empty:
        return empty
    merged["log10_D"] = np.log10(
        pd.to_numeric(merged[d_col], errors="coerce").where(lambda values: values > 0)
    )
    merged["log10_ficd"] = np.log10(
        pd.to_numeric(merged[FICD_VALUE_COL], errors="coerce").where(
            lambda values: values > 0
        )
    )

    per_condition = _group_correlations(
        merged, ["condition"], d_col, "n_tracks"
    )
    per_experiment = _group_correlations(
        merged, ["condition", "experiment"], d_col, "n_tracks"
    )

    if "cell_id" in merged.columns:
        group_columns = [
            column
            for column in ["movie_id", "cell_id"] + FACTOR_COLS
            if column in merged.columns
        ]
        aggregations = {
            "D_cell_median_um2_s": (d_col, "median"),
            "D_cell_mean_um2_s": (d_col, "mean"),
            "n_tracks_in_cell": (d_col, "count"),
            FICD_VALUE_COL: (FICD_VALUE_COL, "first"),
        }
        for column in (
            "analysis_version",
            "marker_detected",
            "marker_status",
            "perimeter_found",
            "background_peak",
            "background_sigma",
            "signal_estimator",
            "growth_stop_reason",
            "warning_flags",
            "phenotype_flags",
        ):
            if column in merged.columns:
                aggregations[column] = (column, "first")
        per_cell = merged.groupby(group_columns, dropna=False).agg(**aggregations).reset_index()
        per_condition_cells = _group_correlations(
            per_cell, ["condition"], "D_cell_median_um2_s", "n_cells"
        )
        per_experiment_cells = _group_correlations(
            per_cell,
            ["condition", "experiment"],
            "D_cell_median_um2_s",
            "n_cells",
        )
    else:
        per_cell = pd.DataFrame()
        per_condition_cells = pd.DataFrame()
        per_experiment_cells = pd.DataFrame()

    return {
        "per_track": merged,
        "per_cell": per_cell,
        "per_condition": per_condition,
        "per_experiment": per_experiment,
        "per_condition_cells": per_condition_cells,
        "per_experiment_cells": per_experiment_cells,
    }


def run(
    manifest_df: pd.DataFrame,
    expression_csv: Path,
    out_dir: Path,
    d_col: str = "D_fit_um2_s",
    marker_to_movie: Optional[Dict[str, str]] = None,
    n_jobs: int = 8,
) -> Path:
    out_dir = Path(out_dir) / "cross_correlation"
    out_dir.mkdir(parents=True, exist_ok=True)
    diffusion = load_diffusion_per_track(manifest_df, d_col=d_col, n_jobs=n_jobs)
    expression = load_expression(Path(expression_csv), marker_to_movie=marker_to_movie)
    results = cross_correlate(diffusion, expression, d_col=d_col)
    output_names = {
        "per_track": "D_vs_ficd_expression.csv",
        "per_cell": "Dcell_vs_ficd_expression.csv",
        "per_condition": "correlation_by_condition.csv",
        "per_experiment": "correlation_by_experiment.csv",
        "per_condition_cells": "correlation_by_condition_cells.csv",
        "per_experiment_cells": "correlation_by_experiment_cells.csv",
    }
    for key, filename in output_names.items():
        results[key].to_csv(out_dir / filename, index=False)
    if expression.empty:
        print(
            f"  cross-correlation skipped: no expression rows in {expression_csv}",
            flush=True,
        )
    return out_dir
