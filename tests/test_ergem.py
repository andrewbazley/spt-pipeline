"""Lightweight tests for the FICD ER-GEM subpackage (no trackpy/nd2 needed)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spt_pipeline import manifest as core_manifest
from spt_pipeline import utils
from spt_pipeline.ergem import condition_map as cm
from spt_pipeline.ergem import ingest, crosscorr, contrasts


def test_mosaic_column_aliasing_and_trackpy_passthrough():
    mos = pd.DataFrame({"Trajectory": [1, 1, 2], "Frame": [0, 1, 0],
                        "x": [1.0, 1.5, 3.0], "y": [2.0, 2.2, 4.0], "m0": [10, 11, 12]})
    n = utils.normalize_track_table(mos)
    assert {"particle", "frame", "x", "y", "mass"}.issubset(n.columns)
    tp = pd.DataFrame({"x": [1.0], "y": [2.0], "mass": [9.0], "frame": [0],
                       "particle": [5], "mean_x_px": [1.0]})
    n2 = utils.normalize_track_table(tp)
    assert "mean_x_px" in n2.columns and list(n2.columns).count("x") == 1


def _make_traj(path, n_tracks, sigma, rng):
    rows = []
    for tid in range(1, n_tracks + 1):
        L = int(rng.integers(10, 18))
        x = 50 + np.cumsum(rng.normal(0, sigma, L))
        y = 50 + np.cumsum(rng.normal(0, sigma, L))
        for f in range(L):
            rows.append((tid, f, x[f], y[f], 100.0))
    pd.DataFrame(rows, columns=["Trajectory", "Frame", "x", "y", "m0"]).to_csv(path, index=False)


@pytest.fixture
def synthetic(tmp_path):
    rng = np.random.default_rng(0)
    sig = {("FICD-WT", "DMSO"): 0.8, ("FICD-WT", "Tg"): 1.3,
           ("E234G", "DMSO"): 0.8, ("E234G", "Tg"): 0.9}
    conditions = []
    for compl in ["FICD-WT", "E234G"]:
        for treat in ["DMSO", "Tg"]:
            reps = []
            for exp in ["2024-05-16", "2024-05-29", "2024-06-07"]:
                movies = []
                for clone in ["A8", "D2"]:
                    mid = f"{clone.lower()}_{compl.lower().replace('-','')}_{treat.lower()}_{exp.replace('-','')}"
                    tp = tmp_path / f"Traj_{mid}.csv"
                    _make_traj(tp, 20, sig[(compl, treat)], rng)
                    movies.append({"movie_id": mid, "clone": clone, "raw_nd2": str(tp),
                                   "trajectory_csv": str(tp),
                                   "ficd_marker_nd2": str(tmp_path / f"Red_{mid}.nd2"),
                                   "has_paired_red_image": True})
                reps.append({"replicate_id": exp, "experiment_date": exp,
                             "analysis_provenance": {"ficd_expression_measurement_table": ""},
                             "movies": movies})
            conditions.append({"condition_id": f"KO_{compl}__{treat}", "genotype": "FICD-KO",
                               "complementation": compl, "treatment_code": treat,
                               "expresses_ficd_marker": True, "replicates": reps})
    jpath = tmp_path / "cmap.json"
    json.dump({"conditions": conditions}, open(jpath, "w"))
    return tmp_path, jpath


def test_from_json_ingest_and_contrasts(synthetic):
    tmp, jpath = synthetic
    mpath = tmp / "manifest.csv"
    df = cm.from_json(jpath, out_manifest=mpath, default_mpp=0.11, default_fps=100.0)
    assert len(df) == 24
    assert {"complementation", "treatment_code", "clone"}.issubset(df.columns)
    assert (df["mpp"].astype(float) == 0.11).all()

    # core validate accepts it and preserves extras
    loaded = core_manifest.load(mpath)
    core_manifest.validate(loaded, default_mpp=0.11, default_fps=100.0, require_tracks=True)
    assert "genotype" in loaded.columns

    res = ingest.ingest_all(df, tmp / "out", 0.11, 100.0, diffusion_tau=5, manifest_path=mpath, n_jobs=1)
    assert all(r["status"] in ("ok", "skipped") for r in res)

    pool = crosscorr.load_diffusion_per_track(df, n_jobs=1)
    med = pool.groupby(["complementation", "treatment_code"])["D_fit_um2_s"].median()
    wt_delta = med[("FICD-WT", "Tg")] - med[("FICD-WT", "DMSO")]
    eg_delta = med[("E234G", "Tg")] - med[("E234G", "DMSO")]
    assert wt_delta > eg_delta  # blunting recovered

    cdir = contrasts.run(df, tmp / "out", do_mixed=False, n_jobs=1)
    fc = pd.read_csv(cdir / "factored_contrasts.csv")
    assert (fc["contrast"] == "deltaTg_WT_vs_E234G").any()
    assert set(fc["family"]) >= {"complementation_within_treatment",
                                 "treatment_within_complementation", "interaction"}
