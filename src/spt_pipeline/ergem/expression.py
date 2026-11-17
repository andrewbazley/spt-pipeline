"""Run track-seeded FICD marker measurement from the ER-GEM manifest.

Each explicit marker/trajectory pair is staged as ``Red_<movie_id>.nd2`` and
``Traj_<movie_id>.csv``. The staged basename is the pairing key, so both the
staging layer and the measurement script preserve it instead of resolving the
red symlink back to an unrelated original marker filename.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

_SCRIPT = Path(__file__).with_name("_ficd_cell_intensity.py")
ANALYSIS_VERSION = "3.1.0-adaptive-track-territory"


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "1.0", "yes")
    return bool(value) if not pd.isna(value) else False


def _path_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _replace_link(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    try:
        link.symlink_to(target)
    except OSError:
        shutil.copy2(target, link)


def _stage_pairs(
    manifest_df: pd.DataFrame,
    cmap: dict,
    staging_root: Path,
) -> Tuple[Dict[str, dict], List[dict]]:
    """Stage explicit pairs and return ``(experiment_groups, failures)``."""
    marker_by_movie: Dict[str, str] = {}
    if "ficd_marker_nd2" not in manifest_df.columns:
        for condition in cmap.get("conditions", []):
            for replicate in condition.get("replicates", []):
                for movie in replicate.get("movies", []):
                    marker = _path_text(movie.get("ficd_marker_nd2"))
                    if marker:
                        marker_by_movie[str(movie["movie_id"])] = marker

    groups: Dict[str, dict] = {}
    failures: List[dict] = []
    for _, row in manifest_df.iterrows():
        if not _truthy(row.get("has_paired_red")):
            continue
        movie_id = str(row["movie_id"])
        marker = _path_text(row.get("ficd_marker_nd2")) or marker_by_movie.get(movie_id, "")
        trajectory = _path_text(row.get("tracks_csv"))
        experiment = str(row.get("experiment", "unknown"))
        if not marker or not trajectory:
            failures.append(
                {
                    "experiment": experiment,
                    "movie_id": movie_id,
                    "stage": "pairing",
                    "error": "missing ficd_marker_nd2 or tracks_csv in manifest",
                }
            )
            continue

        marker_path = Path(marker).expanduser().absolute()
        trajectory_path = Path(trajectory).expanduser().absolute()
        missing = [str(path) for path in (marker_path, trajectory_path) if not path.is_file()]
        if missing:
            failures.append(
                {
                    "experiment": experiment,
                    "movie_id": movie_id,
                    "stage": "pairing",
                    "error": "missing input file(s): " + "; ".join(missing),
                }
            )
            continue

        staging_dir = staging_root / experiment
        staging_dir.mkdir(parents=True, exist_ok=True)
        red_link = staging_dir / f"Red_{movie_id}.nd2"
        trajectory_link = staging_dir / f"Traj_{movie_id}.csv"
        try:
            _replace_link(red_link, marker_path)
            _replace_link(trajectory_link, trajectory_path)
        except OSError as exc:
            failures.append(
                {
                    "experiment": experiment,
                    "movie_id": movie_id,
                    "stage": "staging",
                    "error": str(exc),
                }
            )
            continue

        group = groups.setdefault(
            experiment,
            {"dir": str(staging_dir), "marker_to_movie": {}, "movie_ids": []},
        )
        group["marker_to_movie"][red_link.name] = movie_id
        group["movie_ids"].append(movie_id)
    return groups, failures


def run(
    manifest_df: pd.DataFrame,
    cmap: dict,
    out_dir: Path,
    mask_dir: Optional[Path] = None,
    extra_args: Optional[List[str]] = None,
    python_exe: Optional[str] = None,
    n_jobs: int = 8,
) -> Dict[str, object]:
    """Run measurement per experiment and pool cell/trajectory outputs."""
    out_dir = Path(out_dir)
    staging_root = out_dir / "_staging"
    results_root = out_dir / "ficd_expression"
    results_root.mkdir(parents=True, exist_ok=True)
    groups, stage_failures = _stage_pairs(manifest_df, cmap, staging_root)

    from . import _util

    python = python_exe or sys.executable
    marker_to_movie: Dict[str, str] = {}
    for info in groups.values():
        marker_to_movie.update(info["marker_to_movie"])

    def _run_experiment(item):
        experiment, info = item
        staging_dir = info["dir"]
        experiment_output = results_root / experiment
        experiment_output.mkdir(parents=True, exist_ok=True)
        command = [
            python,
            str(_SCRIPT),
            "--red",
            staging_dir,
            "--trajectories",
            staging_dir,
            "--output-dir",
            str(experiment_output),
        ]
        if mask_dir is not None:
            command += ["--mask-dir", str(mask_dir)]
        if extra_args:
            command += list(extra_args)
        process = subprocess.run(command, capture_output=True, text=True)
        return {
            "experiment": experiment,
            "returncode": process.returncode,
            "stdout_tail": process.stdout[-500:],
            "stderr_tail": process.stderr[-1000:],
            "exp_out": str(experiment_output),
            "n_staged_pairs": len(info["movie_ids"]),
        }

    run_log = _util.pmap(
        _run_experiment,
        list(groups.items()),
        n_jobs=n_jobs,
        desc="ficd expression",
        mode="thread",
        progress=True,
    )

    all_expression: List[pd.DataFrame] = []
    all_cells: List[pd.DataFrame] = []
    all_growth_history: List[pd.DataFrame] = []
    for result in run_log:
        experiment = result["experiment"]
        experiment_output = Path(result["exp_out"])
        expression_csv = experiment_output / "trajectory_expression.csv"
        metadata_csv = experiment_output / "run_metadata.csv"
        if result["returncode"] == 0 and metadata_csv.exists():
            metadata = pd.read_csv(metadata_csv)
            paired = metadata.get("trajectory_file", pd.Series(dtype=str)).fillna("").ne("")
            n_total = int(len(metadata))
            n_paired = int(paired.sum())
            result["n_paired_images"] = n_paired
            result["n_unpaired_images"] = n_total - n_paired
            # Per-image validation: keep the images that paired. Unpaired images
            # contribute no trajectory rows, so one bad field no longer discards
            # the whole experiment; only a fully-unpaired experiment is an error.
            if n_paired == 0:
                result["returncode"] = 3
                result["stderr_tail"] = (
                    result.get("stderr_tail", "")
                    + f"\nno images paired ({n_total} staged)"
                )[-1000:]
                continue
            if n_paired < n_total:
                names = (
                    metadata.loc[~paired, "image"].astype(str).tolist()
                    if "image" in metadata.columns else []
                )
                result["stderr_tail"] = (
                    result.get("stderr_tail", "")
                    + f"\n{n_total - n_paired}/{n_total} image(s) unpaired; kept {n_paired}"
                    + (f"; e.g. {names[:5]}" if names else "")
                )[-1000:]
            if expression_csv.exists():
                expression = pd.read_csv(expression_csv)
                expression["experiment"] = experiment
                all_expression.append(expression)
            cells_csv = experiment_output / "cells.csv"
            if cells_csv.exists():
                cells = pd.read_csv(cells_csv)
                if "image" in cells.columns:
                    cells["movie_id"] = cells["image"].map(
                        lambda basename: marker_to_movie.get(
                            str(basename), Path(str(basename)).stem.removeprefix("Red_")
                        )
                    )
                cells["experiment"] = experiment
                all_cells.append(cells)
            growth_csv = experiment_output / "dilation_history.csv"
            if growth_csv.exists():
                growth = pd.read_csv(growth_csv)
                if "image" in growth.columns:
                    growth["movie_id"] = growth["image"].map(
                        lambda basename: marker_to_movie.get(
                            str(basename), Path(str(basename)).stem.removeprefix("Red_")
                        )
                    )
                growth["experiment"] = experiment
                all_growth_history.append(growth)

    pooled_expression = (
        pd.concat(all_expression, ignore_index=True) if all_expression else pd.DataFrame()
    )
    if not pooled_expression.empty and "image" in pooled_expression.columns:
        pooled_expression["movie_id"] = pooled_expression["image"].map(
            lambda basename: marker_to_movie.get(
                str(basename), Path(str(basename)).stem.removeprefix("Red_")
            )
        )
    pooled_path = results_root / "trajectory_expression_pooled.csv"
    pooled_expression.to_csv(pooled_path, index=False)
    if all_cells:
        pd.concat(all_cells, ignore_index=True).to_csv(
            results_root / "cells_pooled.csv", index=False
        )
    if all_growth_history:
        pd.concat(all_growth_history, ignore_index=True).to_csv(
            results_root / "dilation_history_pooled.csv", index=False
        )

    log_rows = run_log + [
        {
            "experiment": failure["experiment"],
            "returncode": 2,
            "stderr_tail": failure["error"],
            "stage": failure["stage"],
            "movie_id": failure["movie_id"],
        }
        for failure in stage_failures
    ]
    log_path = results_root / "ficd_run_log.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False)

    n_failed = sum(int(row.get("returncode", 1) != 0) for row in log_rows)
    if n_failed:
        print(
            f"  WARNING: {n_failed} FICD staging/measurement failure(s). See {log_path}",
            flush=True,
        )
    n_rows = len(pooled_expression)
    if n_rows == 0:
        print(
            "  WARNING: no paired per-trajectory expression rows were produced; "
            "cross-correlation will be skipped.",
            flush=True,
        )
    return {
        "analysis_version": ANALYSIS_VERSION,
        "pooled_expression_csv": str(pooled_path),
        "marker_to_movie": marker_to_movie,
        "n_experiments": len(groups),
        "n_failed": n_failed,
        "n_rows": n_rows,
        "dilation_history_csv": str(results_root / "dilation_history_pooled.csv"),
        "run_log": log_rows,
    }
