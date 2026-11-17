"""Synthetic-data unit tests for the SPT pipeline v2 core.

These tests do not require trackpy or imaging libraries. They validate the
numeric engines that carry correctness risk: the fixed-tau merge-join (against a
brute-force reference), the histogram representations, the statistical
framework, the bin infrastructure, and manifest scan/validate.

Run with:  pytest -q   (or)   python -m pytest tests/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spt_pipeline import (bins, diffusion, histograms, manifest, statistics,
                          tau_steps)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_track(particle, frames, x0=0.0, y0=0.0, vx=1.0, vy=0.0):
    """A track with detections on given frames, moving linearly in pixels."""
    frames = np.asarray(frames, dtype=int)
    return pd.DataFrame({
        "particle": particle,
        "frame": frames,
        "x": x0 + vx * frames,
        "y": y0 + vy * frames,
        "mass": 100.0,
    })


def brute_force_tau(tracks, tau, mpp):
    """Reference implementation: explicit loop over frame pairs."""
    rows = []
    for pid, g in tracks.groupby("particle"):
        by_frame = {int(f): (x, y) for f, x, y in zip(g["frame"], g["x"], g["y"])}
        for f, (x, y) in by_frame.items():
            tgt = f + tau
            if tgt in by_frame:
                x2, y2 = by_frame[tgt]
                dx, dy = x2 - x, y2 - y
                rows.append(np.sqrt(dx * dx + dy * dy) * mpp)
    return np.sort(np.array(rows))


# ---------------------------------------------------------------------------
# Tau-step engine
# ---------------------------------------------------------------------------

def test_tau_merge_join_matches_brute_force():
    rng = np.random.default_rng(0)
    tracks = []
    for pid in range(20):
        n = rng.integers(5, 60)
        # gappy frames so f+tau isn't always present
        frames = np.sort(rng.choice(np.arange(0, 100), size=n, replace=False))
        tracks.append(make_track(pid, frames, x0=rng.normal(), vx=rng.normal(0, 0.5),
                                 vy=rng.normal(0, 0.5)))
    tracks = pd.concat(tracks, ignore_index=True)
    mpp = 0.183
    for tau in (1, 2, 5, 10):
        out = tau_steps.compute_tau_displacements(tracks, tau, mpp=mpp, fps=50.0)
        got = np.sort(out["step_um"].to_numpy())
        ref = brute_force_tau(tracks, tau, mpp)
        assert got.size == ref.size, f"count mismatch at tau={tau}"
        assert np.allclose(got, ref, atol=1e-9), f"value mismatch at tau={tau}"


def test_tau_count_formula_dense_track():
    # A dense track of length L has exactly L - tau displacements at that tau.
    tracks = make_track(0, np.arange(0, 200))
    for tau in (1, 10, 50):
        out = tau_steps.compute_tau_displacements(tracks, tau, mpp=1.0, fps=1.0)
        assert len(out) == 200 - tau


def test_tau_linear_motion_value():
    # Constant velocity 3 px/frame -> displacement at tau is 3*tau px.
    tracks = make_track(0, np.arange(0, 50), vx=3.0, vy=0.0)
    out = tau_steps.compute_tau_displacements(tracks, 4, mpp=0.5, fps=10.0)
    assert np.allclose(out["step_um"], 3.0 * 4 * 0.5)
    assert np.allclose(out["tau_s"], 4 / 10.0)


def test_non_overlapping_reduces_yield():
    tracks = make_track(0, np.arange(0, 100))
    full = tau_steps.compute_tau_displacements(tracks, 10, mpp=1.0, fps=1.0)
    nonov = tau_steps.compute_tau_displacements(tracks, 10, mpp=1.0, fps=1.0,
                                                non_overlapping=True)
    assert len(nonov) < len(full)
    # Independent grid: frames 0,10,20,... paired with +10 -> 9 disjoint steps
    assert len(nonov) == 9


def test_particle_filter():
    t = pd.concat([make_track(0, np.arange(10)), make_track(1, np.arange(10))],
                  ignore_index=True)
    out = tau_steps.compute_tau_displacements(t, 1, mpp=1.0, fps=1.0, particles={0})
    assert set(out["particle"].unique()) == {0}


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------

def test_raw_pooled_fractions_sum_to_one():
    vals = np.random.default_rng(1).uniform(0, 1, 1000)
    b = bins.make_bins(0, 1, 0.05)
    h = histograms.raw_pooled_histogram(vals, b, "A")
    assert abs(h["fraction"].sum() - 1.0) < 1e-9
    assert h["count"].sum() == 1000


def test_track_weighted_equal_vote():
    # One long track + one short track. Track-weighted gives each equal weight,
    # so a value only in the short track still contributes 0.5 of a bin.
    long_t = pd.DataFrame({"movie_id": "m", "particle": 0,
                           "step_um": np.zeros(100)})  # all in bin 0
    short_t = pd.DataFrame({"movie_id": "m", "particle": 1,
                            "step_um": np.full(2, 0.5)})  # all in a later bin
    df = pd.concat([long_t, short_t], ignore_index=True)
    b = bins.make_bins(0, 1, 0.1)
    tw = histograms.track_weighted_histogram(df, b, "A")
    # bin containing 0.5 should have mean fraction ~0.5 (one of two tracks fully there)
    idx = np.argmin(np.abs(tw["bin_center"].to_numpy() - 0.55))
    assert abs(tw["mean_fraction_per_track"].iloc[idx] - 0.5) < 1e-9
    assert tw["n_tracks"].iloc[0] == 2


def test_track_length_split_runs():
    rng = np.random.default_rng(2)
    rows = []
    for pid in range(40):
        n = rng.integers(2, 50)
        rows.append(pd.DataFrame({"movie_id": "m", "particle": pid,
                                  "step_um": rng.uniform(0, 1, n)}))
    df = pd.concat(rows, ignore_index=True)
    b = bins.make_bins(0, 1, 0.1)
    split = histograms.track_length_split_histogram(df, b, "A")
    assert not split.empty
    assert set(["condition", "length_bin", "bin_center", "fraction"]).issubset(split.columns)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_cliffs_delta_bounds_and_sign():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([6, 7, 8, 9, 10])
    d = statistics.cliffs_delta(a, b)
    assert d == pytest.approx(-1.0)
    assert statistics.cliffs_delta(b, a) == pytest.approx(1.0)
    assert abs(statistics.cliffs_delta(a, a)) < 1e-9


def test_bootstrap_ci_contains_median():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci = statistics.bootstrap_median_ci(vals, n_boot=2000, random_state=0)
    assert ci["lo"] <= ci["median"] <= ci["hi"]


def test_between_condition_bh_columns():
    per_exp = pd.DataFrame({
        "condition": ["NT", "NT", "NT", "Tg", "Tg", "Tg", "X", "X", "X"],
        "median_step_um": [0.1, 0.11, 0.12, 0.3, 0.31, 0.29, 0.2, 0.21, 0.19],
    })
    out = statistics.between_condition_tests(per_exp, "median_step_um", label="s")
    assert "p_value_bh" in out.columns
    assert len(out) == 3  # 3 pairwise comparisons among 3 conditions


# ---------------------------------------------------------------------------
# Bins
# ---------------------------------------------------------------------------

def test_bin_width_floor():
    # step floor is mpp/10; requesting a finer width should be raised to floor.
    bs = bins.BinSettings(mpp=0.183, fps=50.0, step_bin_width_um=0.001)
    assert bs.step_width == pytest.approx(0.0183)
    assert "floor" in bs.step_source


def test_make_bins_spans_range():
    b = bins.make_bins(0, 1.0, 0.1)
    assert b[0] == 0 and b[-1] >= 1.0


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------

def test_diffusion_D_from_imsd():
    # IMSD linear in tau: MSD = 4*D*tau with D=0.25 -> slope gives D back.
    D_true = 0.25
    tau_s = np.array([0.02, 0.04, 0.06, 0.08, 0.10])
    imsd = pd.DataFrame({0: 4 * D_true * tau_s, 1: 4 * D_true * tau_s}, index=tau_s)
    tracks = pd.concat([make_track(0, np.arange(6)), make_track(1, np.arange(6))],
                       ignore_index=True)
    dt = diffusion.build_diffusion_table(tracks, imsd, fps=50.0, mpp=0.183, diffusion_tau=1)
    # tau=1 frame at fps=50 -> 0.02 s row
    assert np.allclose(dt["D_fit_um2_s"].dropna().to_numpy(), D_true, atol=1e-9)


def test_loc_error_subtraction_nonnegative():
    diff_df = pd.DataFrame({
        "particle": [0, 1], "alpha": [0.1, 1.0], "n_spots": [50, 50],
        "msd_fit_um2": [0.001, 0.02], "tau_fit_s": [0.02, 0.02],
        "D_fit_um2_s": [0.0125, 0.25],
    })
    floor = diffusion.estimate_loc_error_floor(diff_df, alpha_max=0.3, min_nspots=10)
    assert floor == pytest.approx(0.001)
    corr = diffusion.apply_loc_error_subtraction(diff_df, floor)
    assert (corr["D_fit_um2_s_corr"] >= 0).all()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def test_manifest_scan_from_tracks(tmp_path):
    # create fake tracks CSVs in two "experiment" folders
    for day, cond in [("2025-04-08", "NT"), ("2025-04-15", "Tg")]:
        d = tmp_path / f"experiment_{day}"
        d.mkdir()
        t = make_track(0, np.arange(10))
        t.to_csv(d / f"{cond}_cell1_tracks.csv", index=False)
    res = manifest.scan([tmp_path], default_mpp=0.183, default_fps=50.0, from_tracks=True)
    df = res.manifest
    assert len(df) == 2
    assert set(df["condition"]) == {"NT", "Tg"}
    assert set(df["experiment"]) == {"2025-04-08", "2025-04-15"}
    warnings = manifest.validate(df, default_mpp=0.183, default_fps=50.0, require_tracks=True)
    assert isinstance(warnings, list)


def test_manifest_validate_flags_fill_in(tmp_path):
    df = pd.DataFrame({
        "movie_id": ["a"], "source_path": [""], "file_format": ["tracks"],
        "condition": ["FILL_IN"], "experiment": ["2025-01-01"],
        "tracks_csv": [""], "diff_csv": [""], "mpp": ["0.183"], "fps": ["50"],
        "n_frames": [""], "n_channels": [""], "notes": [""],
    })
    with pytest.raises(manifest.ManifestError):
        manifest.validate(df, default_mpp=0.183, default_fps=50.0)


# ---------------------------------------------------------------------------
# Manifest portability
# ---------------------------------------------------------------------------

def _manifest_row(tmp_path, movie_id="m1", tracks_csv="", condition="NT",
                  experiment="2025-01-01"):
    return {"movie_id": movie_id, "source_path": tracks_csv, "file_format": "tracks",
            "condition": condition, "experiment": experiment,
            "tracks_csv": tracks_csv, "diff_csv": "", "mpp": "0.183", "fps": "50",
            "n_frames": "", "n_channels": "", "notes": ""}


def test_manifest_paths_written_relative_and_resolve_back(tmp_path):
    """Absolute paths in, POSIX-relative paths on disk, absolute paths back out."""
    data = tmp_path / "data" / "m1"
    data.mkdir(parents=True)
    tcsv = data / "m1_tracks.csv"
    make_track(0, np.arange(10)).to_csv(tcsv, index=False)

    df = pd.DataFrame([_manifest_row(tmp_path, tracks_csv=str(tcsv))],
                      columns=manifest.COLUMNS)
    mpath = tmp_path / "manifest.csv"
    manifest.write(df, mpath)

    raw = pd.read_csv(mpath, dtype=str, keep_default_na=False)
    stored = raw["tracks_csv"].iloc[0]
    assert stored == "data/m1/m1_tracks.csv", stored
    assert not Path(stored).is_absolute()
    assert "\\" not in stored

    back = manifest.load(mpath)
    assert Path(back["tracks_csv"].iloc[0]).resolve() == tcsv.resolve()


def test_manifest_accepts_windows_separators(tmp_path):
    """A manifest authored on Windows resolves on POSIX without editing."""
    data = tmp_path / "data" / "m1"
    data.mkdir(parents=True)
    tcsv = data / "m1_tracks.csv"
    make_track(0, np.arange(10)).to_csv(tcsv, index=False)

    row = _manifest_row(tmp_path, tracks_csv=r"data\m1\m1_tracks.csv")
    pd.DataFrame([row], columns=manifest.COLUMNS).to_csv(tmp_path / "manifest.csv",
                                                         index=False)
    df = manifest.load(tmp_path / "manifest.csv")
    assert Path(df["tracks_csv"].iloc[0]).resolve() == tcsv.resolve()


def test_manifest_path_base_override(tmp_path):
    """--data-root anchors paths when the dataset is mounted away from the manifest."""
    mounted = tmp_path / "mnt" / "m1"
    mounted.mkdir(parents=True)
    tcsv = mounted / "m1_tracks.csv"
    make_track(0, np.arange(10)).to_csv(tcsv, index=False)

    row = _manifest_row(tmp_path, tracks_csv="m1/m1_tracks.csv")
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    pd.DataFrame([row], columns=manifest.COLUMNS).to_csv(analysis / "manifest.csv",
                                                         index=False)
    df = manifest.load(analysis / "manifest.csv", path_base=tmp_path / "mnt")
    assert Path(df["tracks_csv"].iloc[0]).resolve() == tcsv.resolve()


# ---------------------------------------------------------------------------
# No-track movies are skipped, not fatal
# ---------------------------------------------------------------------------

def test_blank_tracks_csv_warns_and_is_skipped(tmp_path):
    good = tmp_path / "good_tracks.csv"
    make_track(0, np.arange(10)).to_csv(good, index=False)
    df = pd.DataFrame([
        _manifest_row(tmp_path, "good", str(good)),
        _manifest_row(tmp_path, "dark", ""),          # tracking found nothing
    ], columns=manifest.COLUMNS)

    warnings = manifest.validate(df, default_mpp=0.183, default_fps=50.0,
                                 require_tracks=True)
    assert any("no tracks_csv" in w for w in warnings)
    assert manifest.skipped_movies(df) == ["dark"]
    assert list(manifest.rows_with_tracks(df)["movie_id"]) == ["good"]


def test_missing_tracks_file_is_still_an_error(tmp_path):
    """A path that is listed but absent is a real fault, unlike a blank cell."""
    df = pd.DataFrame([_manifest_row(tmp_path, "m", str(tmp_path / "nope_tracks.csv"))],
                      columns=manifest.COLUMNS)
    with pytest.raises(manifest.ManifestError, match="listed but not found"):
        manifest.validate(df, default_mpp=0.183, default_fps=50.0, require_tracks=True)


def test_all_blank_tracks_is_an_error(tmp_path):
    df = pd.DataFrame([_manifest_row(tmp_path, "a", ""), _manifest_row(tmp_path, "b", "")],
                      columns=manifest.COLUMNS)
    with pytest.raises(manifest.ManifestError, match="nothing to analyze"):
        manifest.validate(df, default_mpp=0.183, default_fps=50.0, require_tracks=True)


# ---------------------------------------------------------------------------
# Statistical design inference
# ---------------------------------------------------------------------------

def _design_frame(pairs):
    return pd.DataFrame([{"condition": c, "experiment": e, "movie_id": f"{c}_{e}_{i}"}
                         for i, (c, e) in enumerate(pairs)])


def test_design_between_experiment_when_replicated():
    df = _design_frame([("NT", "d1"), ("NT", "d2"), ("Tg", "d1"), ("Tg", "d2")])
    d = statistics.infer_design(df)
    assert d.design == "between-experiment"
    assert d.unit == "experiment" and d.unit_column == "experiment"
    assert not d.is_within_experiment


def test_design_within_experiment_for_single_day():
    df = _design_frame([("NT", "d1"), ("NT", "d1"), ("Tg", "d1"), ("Tg", "d1")])
    d = statistics.infer_design(df)
    assert d.design == "within-experiment"
    assert d.unit == "movie" and d.unit_column == "movie_id"
    assert "technical" in d.caveat.lower()


def test_design_unbalanced_falls_back_to_within_experiment():
    """2 experiments overall but one condition has 1 -> n=1 arm, so not biological."""
    df = _design_frame([("NT", "d1"), ("NT", "d2"), ("Tg", "d1")])
    d = statistics.infer_design(df)
    assert d.design == "within-experiment"
    assert d.n_experiments == 2 and d.n_experiments_min_per_condition == 1


def test_between_condition_tests_report_the_design_unit():
    df = _design_frame([("NT", "d1"), ("NT", "d1"), ("Tg", "d1"), ("Tg", "d1")])
    design = statistics.infer_design(df)
    per_unit = pd.DataFrame({"condition": ["NT", "NT", "NT", "Tg", "Tg", "Tg"],
                             "median_D": [0.05, 0.06, 0.055, 0.03, 0.031, 0.029]})
    out = statistics.between_condition_tests(per_unit, "median_D", design=design)
    assert set(out["unit"]) == {"movie"}
    assert set(out["design"]) == {"within-experiment"}
    assert "technical" in out["interpretation"].iloc[0].lower()


def test_track_level_tests_report_effect_size():
    rng = np.random.default_rng(1)
    pooled = {"NT": rng.normal(0.05, 0.01, 500), "Tg": rng.normal(0.03, 0.01, 500)}
    out = statistics.track_level_tests(pooled, label="D_per_track")
    assert len(out) == 1
    assert set(out["unit"]) == {"track"}
    assert out["cliffs_delta"].iloc[0] > 0.8      # NT clearly above Tg
    assert out["n_tracks_a"].iloc[0] == 500


# ---------------------------------------------------------------------------
# Advanced-analysis selection
# ---------------------------------------------------------------------------

def test_advanced_keyword_aliases():
    from spt_pipeline.advanced import ADVANCED_KEYWORDS, resolve_selection
    assert resolve_selection(["a2"]) == {"ngp"}
    assert resolve_selection(["VACF", "a2"]) == {"vacf", "ngp"}
    assert resolve_selection(["Non-Gaussian", "ergodicity"]) == {"ngp", "eb"}
    assert resolve_selection(None) == set(ADVANCED_KEYWORDS)
    assert resolve_selection(["all"]) == set(ADVANCED_KEYWORDS)


def test_advanced_unknown_keyword_raises():
    from spt_pipeline.advanced import UnknownAnalysisError, resolve_selection
    with pytest.raises(UnknownAnalysisError, match="bogus"):
        resolve_selection(["vacf", "bogus"])


# ---------------------------------------------------------------------------
# Tracking defaults
# ---------------------------------------------------------------------------

def test_tracking_defaults_include_tau_steps():
    from spt_pipeline import tracking
    p = tracking.TrackingParams()
    assert p.tau_steps == (1, 5, 10)
    assert p.qc_plots is True
    # argparse hands over a list, possibly unsorted/duplicated
    p2 = tracking.TrackingParams(tau_steps=[10, 1, 1, 5, 0, -3])
    assert p2.tau_steps == (1, 5, 10)
    assert tracking.TrackingParams(tau_steps=[]).tau_steps == ()
    assert p2.to_dict()["tau_steps"] == [1, 5, 10]     # picklable/JSON-able


# ---------------------------------------------------------------------------
# Localization QC
# ---------------------------------------------------------------------------

def test_qc_panels_and_bias_metric(tmp_path):
    from spt_pipeline import qc
    rng = np.random.default_rng(0)
    n = 2000
    parts = pd.DataFrame({
        "frame": rng.integers(0, 120, n),
        # centroids snapped to pixel centres: the classic pixel-biting signature
        "x": np.round(rng.uniform(0, 256, n)) + rng.normal(0, 0.03, n),
        "y": rng.uniform(0, 256, n),
        "mass": np.abs(rng.normal(150, 50, n)),
        "size": np.abs(rng.normal(1.4, 0.3, n)),
        "signal": np.abs(rng.normal(60, 20, n)),
        "ecc": np.clip(np.abs(rng.normal(0.15, 0.08, n)), 0, 1),
    })
    tracks = pd.DataFrame({"particle": rng.integers(0, 150, n),
                           "frame": parts["frame"], "x": parts["x"], "y": parts["y"]})
    frames = [rng.normal(100, 10, (48, 48)) for _ in range(120)]
    summary = qc.localization_qc("MOV", tmp_path, particles=parts, tracks=tracks,
                                 frames=frames, diameter=7, minmass=40.0,
                                 min_track_length=4, overlay_frames=[100])
    qc_dir = tmp_path / qc.QC_DIRNAME
    assert qc_dir.is_dir(), "QC figures must land in their own subfolder"
    for panel in ("feature_properties", "subpixel_bias", "ecc_vs_mass",
                  "features_per_frame", "track_lengths", "detections_frame100"):
        assert panel in summary["panels"], panel
    assert (qc_dir / "MOV_qc_feature_properties.png").exists()
    assert (qc_dir / "MOV_qc_detections_frame0100.png").exists()
    assert summary["subpixel_bias_max_dev"] > 1.0      # bias detected
    assert 0.0 <= summary["frac_detections_above_minmass"] <= 1.0


def test_qc_skips_out_of_range_overlay_frame(tmp_path):
    """A frame past the end of the movie is skipped, never silently swapped."""
    from spt_pipeline import qc
    rng = np.random.default_rng(1)
    n = 400
    parts = pd.DataFrame({"frame": rng.integers(0, 10, n),
                          "x": rng.uniform(0, 64, n), "y": rng.uniform(0, 64, n),
                          "mass": np.abs(rng.normal(150, 40, n))})
    frames = [rng.normal(100, 10, (32, 32)) for _ in range(10)]
    summary = qc.localization_qc("SHORT", tmp_path, particles=parts, tracks=None,
                                 frames=frames, overlay_frames=[100])
    assert not any("detections" in p for p in summary["panels"])


def test_qc_is_safe_on_empty_input(tmp_path):
    """QC must never be able to break a tracking run."""
    from spt_pipeline import qc
    summary = qc.localization_qc("EMPTY", tmp_path, particles=pd.DataFrame(),
                                 tracks=None, frames=None)
    assert summary["panels"] == []


# ---------------------------------------------------------------------------
# Figure conventions
# ---------------------------------------------------------------------------

def test_significance_notation():
    from spt_pipeline import plotting
    assert plotting.p_to_stars(1e-5) == "****"
    assert plotting.p_to_stars(5e-4) == "***"
    assert plotting.p_to_stars(0.004) == "**"
    assert plotting.p_to_stars(0.03) == "*"
    assert plotting.p_to_stars(0.4) == "n.s."
    assert plotting.p_to_stars(None) == ""
    assert plotting.p_to_stars(float("nan")) == ""


def test_condition_colors_are_order_independent():
    """A condition must keep its colour across panels, whatever the input order."""
    from spt_pipeline import plotting
    a = plotting.condition_colors(["Tg", "NT", "DMSO"])
    b = plotting.condition_colors(["NT", "DMSO", "Tg"])
    assert a == b
    assert len(set(a.values())) == 3


def test_comparisons_prefer_corrected_p():
    from spt_pipeline import plotting
    df = pd.DataFrame({"condition_a": ["A"], "condition_b": ["B"],
                       "p_value": [0.001], "p_value_bh": [0.20]})
    assert plotting.comparisons_from_stats(df) == [("A", "B", 0.20)]
    assert plotting.comparisons_from_stats(pd.DataFrame()) == []


def test_boxplot_draws_brackets_without_prose(tmp_path):
    """The only text added by significance is the marker itself."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from spt_pipeline import plotting
    rng = np.random.default_rng(0)
    rows = [{"condition": c, "experiment": "d1", "movie_id": f"{c}{k}",
             "v": rng.normal(mu, 0.004)}
            for c, mu in (("NT", 0.06), ("Tg", 0.03)) for k in range(6)]
    out = tmp_path / "box.png"
    plotting.condition_boxplot(pd.DataFrame(rows), out, value_col="v",
                               ylabel="D (µm²/s)", comparisons=[("NT", "Tg", 1e-6)])
    assert out.exists()

    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    plotting.significance_brackets(ax, [("A", "B", 1e-6)], {"A": 1.0, "B": 2.0})
    texts = [t.get_text() for t in ax.texts]
    assert texts == ["****"], texts          # no test name, no p-value
    plt.close(fig)


def test_brackets_thin_out_when_crowded():
    """Many conditions must not produce an unreadable ladder of bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from spt_pipeline import plotting
    conds = list("ABCDE")
    pos = {c: float(i + 1) for i, c in enumerate(conds)}
    comps = [(a, b, 1e-6) for i, a in enumerate(conds) for b in conds[i + 1:]]
    assert len(comps) == 10
    fig, ax = plt.subplots(); ax.set_ylim(0, 1)
    plotting.significance_brackets(ax, comps, pos, max_brackets=6)
    assert len(ax.texts) == 6
    plt.close(fig)


# ---------------------------------------------------------------------------
# Expression-marker stage
# ---------------------------------------------------------------------------

def _mrow(mid, src, notes="", **kw):
    base = {"movie_id": mid, "source_path": str(src), "file_format": "tif",
            "condition": "NT", "experiment": "2026-01-05", "tracks_csv": "",
            "diff_csv": "", "marker_image": "", "mpp": "0.183", "fps": "50",
            "n_frames": "400", "n_channels": "1", "notes": notes}
    base.update(kw)
    return base


def test_marker_and_single_frame_are_not_trackable():
    from spt_pipeline import manifest as m
    r = lambda **k: pd.Series({"movie_id": "x", "notes": "", "n_frames": "400", **k})
    assert m.is_trackable(r())
    assert not m.is_trackable(r(notes="marker"))
    assert not m.is_trackable(r(n_frames="1"))
    # A blank n_frames must not be guessed at.
    assert m.is_trackable(r(n_frames=""))
    # 'marker' only as a whole word.
    assert not m.is_marker_row(r(notes="no_marker_found"))
    assert m.is_marker_row(r(notes="marker, for=488__001"))
    assert m.skip_tracking_reason(r(notes="marker")) == "expression-marker image"


def test_marker_pairs_by_index_within_a_directory(tmp_path):
    """488__001 pairs with 560__001 in the same folder, and only there."""
    from spt_pipeline import manifest as m
    from spt_pipeline import marker
    rows = []
    for cell in ("cell1", "cell2"):
        d = tmp_path / cell
        d.mkdir()
        for name in ("488__001.tif", "560__001.tif"):
            (d / name).write_bytes(b"")
        rows.append(_mrow(f"{cell}_gem", d / "488__001.tif"))
        rows.append(_mrow(f"{cell}_mark", d / "560__001.tif", notes="marker",
                          n_frames="1"))
    df = pd.DataFrame(rows, columns=m.COLUMNS)
    pairs = {p.movie_id: p for p in marker.pair_markers(df)}
    assert set(pairs) == {"cell1_gem", "cell2_gem"}
    for cell in ("cell1", "cell2"):
        p = pairs[f"{cell}_gem"]
        assert p.ok and p.method == "trailing index in directory"
        # the critical property: never reach into the other directory
        assert Path(p.marker_path).parent.name == cell


def test_marker_pairs_by_stem_appendage(tmp_path):
    from spt_pipeline import manifest as m
    from spt_pipeline import marker
    d = tmp_path / "d"; d.mkdir()
    (d / "488__007.tif").write_bytes(b"")
    (d / "488__007_FICD.tif").write_bytes(b"")
    df = pd.DataFrame([
        _mrow("gem", d / "488__007.tif"),
        _mrow("mk", d / "488__007_FICD.tif", notes="marker", n_frames="1"),
    ], columns=m.COLUMNS)
    p = marker.pair_markers(df)[0]
    assert p.ok and p.method == "stem appendage"


def test_marker_image_column_overrides_matching(tmp_path):
    from spt_pipeline import manifest as m
    from spt_pipeline import marker
    d = tmp_path / "d"; d.mkdir()
    for n in ("488__001.tif", "560__001.tif", "chosen.tif"):
        (d / n).write_bytes(b"")
    df = pd.DataFrame([
        _mrow("gem", d / "488__001.tif", marker_image=str(d / "chosen.tif")),
        _mrow("mk", d / "560__001.tif", notes="marker", n_frames="1"),
    ], columns=m.COLUMNS)
    p = marker.pair_markers(df)[0]
    assert p.method == "marker_image column"
    assert Path(p.marker_path).name == "chosen.tif"


def test_marker_ambiguity_is_reported_not_guessed(tmp_path):
    from spt_pipeline import manifest as m
    from spt_pipeline import marker
    d = tmp_path / "d"; d.mkdir()
    for n in ("488__001.tif", "560__001.tif", "640__001.tif", "488__002.tif"):
        (d / n).write_bytes(b"")
    df = pd.DataFrame([
        _mrow("gem", d / "488__001.tif"),
        _mrow("gem2", d / "488__002.tif"),
        _mrow("m1", d / "560__001.tif", notes="marker", n_frames="1"),
        _mrow("m2", d / "640__001.tif", notes="marker", n_frames="1"),
    ], columns=m.COLUMNS)
    pairs = {p.movie_id: p for p in marker.pair_markers(df)}
    assert pairs["gem"].method == "ambiguous" and not pairs["gem"].ok
    assert not pairs["gem2"].ok          # index 002 matches no marker


def test_disk_mean_matches_brute_force():
    """The convolution shortcut must equal an explicit neighbourhood mean."""
    from spt_pipeline import marker
    rng = np.random.default_rng(0)
    img = rng.normal(500, 80, (48, 48))
    for diam in (5, 7, 11):
        mean_img, cov = marker.disk_mean_image(img, diam)
        k = marker.disk_kernel(diam)
        n = k.shape[0] // 2
        yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
        mask = (xx ** 2 + yy ** 2) <= (diam / 2.0) ** 2
        for (y, x) in ((20, 20), (24, 31), (13, 39)):
            ref = img[y - n:y + n + 1, x - n:x + n + 1][mask].mean()
            assert mean_img[y, x] == pytest.approx(ref, rel=1e-9)
        assert cov[24, 24] == pytest.approx(1.0)
        assert 0 < cov[0, 0] < 1          # partial disk at the corner


def test_disk_kernel_area_matches_trackpy_diameter():
    from spt_pipeline import marker
    assert int(np.count_nonzero(marker.disk_kernel(7))) == 37
    assert marker.disk_kernel(7).sum() == pytest.approx(1.0)


def test_sampling_rejects_out_of_frame():
    from spt_pipeline import marker
    img = np.ones((16, 16)) * 5.0
    mean_img, cov = marker.disk_mean_image(img, 3)
    v, _ = marker.sample_disk_means(mean_img, cov,
                                    np.array([8.0, -1.0, 99.0, np.nan]),
                                    np.array([8.0, 8.0, 8.0, 8.0]))
    assert v[0] == pytest.approx(5.0)
    assert np.isnan(v[1]) and np.isnan(v[2]) and np.isnan(v[3])


def test_one_intensity_per_track():
    """A track must never carry two intensity values into the analysis."""
    from spt_pipeline import marker
    rng = np.random.default_rng(0)
    n = 300
    tracks = pd.DataFrame({
        "particle": rng.integers(0, 25, n),
        "frame": rng.integers(0, 50, n),
        "x": rng.uniform(5, 60, n), "y": rng.uniform(5, 60, n),
        "marker_intensity_localization": rng.normal(120, 15, n),
        "marker_disk_coverage": 1.0,
    })
    per = marker.aggregate_per_track(tracks, (64, 64))
    assert per["particle"].is_unique
    assert len(per) == tracks["particle"].nunique()
    for c in ("marker_intensity", "mean_x_px", "mean_y_px", "x_range_px",
              "dist_to_edge_px", "frac_disk_in_frame", "n_localizations_measured"):
        assert c in per.columns
    assert (per["dist_to_edge_px"] >= 0).all()


def test_correlation_by_unit_respects_hierarchy():
    from spt_pipeline import statistics as st
    rng = np.random.default_rng(3)
    rows = []
    for cond, slope in (("NT", 1.0), ("Tg", 0.0)):
        for mv in range(4):
            x = rng.uniform(0, 100, 40)
            y = 0.05 + slope * 3e-4 * x + rng.normal(0, 0.004, 40)
            rows += [{"condition": cond, "movie_id": f"{cond}{mv}",
                      "I": xi, "D": yi} for xi, yi in zip(x, y)]
    df = pd.DataFrame(rows)
    per_unit = st.correlation_by_unit(df, "I", "D", unit_col="movie_id")
    assert len(per_unit) == 8
    nt = per_unit.loc[per_unit["condition"] == "NT", "rho"].median()
    tg = per_unit.loc[per_unit["condition"] == "Tg", "rho"].median()
    assert nt > 0.5 and abs(tg) < 0.35
    sig = st.correlation_significance(per_unit)
    assert set(sig["condition"]) == {"NT", "Tg"}
    assert "n_units" in sig.columns


# ---------------------------------------------------------------------------
# Per-track metrics
# ---------------------------------------------------------------------------

def test_path_speed_and_step_median_match_brute_force(tmp_path):
    """Mean path speed and step medians must equal explicit references."""
    from spt_pipeline import tau_steps as ts, track_metrics
    rng = np.random.default_rng(0)
    mpp, fps = 0.11, 100.0
    rows = []
    for pid in range(12):
        n = int(rng.integers(15, 40))
        pos = np.cumsum(rng.normal(0, 1.5, (n, 2)), axis=0) + 50
        for f in range(n):
            rows.append((pid, f, pos[f, 0], pos[f, 1]))
    tr = pd.DataFrame(rows, columns=["particle", "frame", "x", "y"])
    files = {}
    for tau in (1, 5):
        p = tmp_path / ts.tau_filename("m", tau)
        ts.compute_tau_displacements(tr, tau, mpp=mpp, fps=fps).to_csv(p, index=False)
        files[tau] = p
    got = track_metrics.path_speed_and_steps(files, fps).set_index("_pk")

    for pid, g in tr.groupby("particle"):
        g = g.sort_values("frame")
        st = np.hypot(np.diff(g["x"]), np.diff(g["y"])) * mpp
        row = got.loc[str(pid)]
        assert row["v_path_um_s"] == pytest.approx(st.sum() / (len(st) / fps), rel=1e-9)
        assert row["step_median_um_tau1"] == pytest.approx(np.median(st), rel=1e-9)
        assert row["path_length_um"] == pytest.approx(st.sum(), rel=1e-9)


def test_metrics_pool_has_one_row_per_track(tmp_path):
    from spt_pipeline import manifest as m
    from spt_pipeline import tau_steps as ts, track_metrics
    d = tmp_path / "mv"; d.mkdir()
    rng = np.random.default_rng(1)
    rows = [(pid, f, *(rng.normal(50, 3, 2)), rng.normal(300, 20))
            for pid in range(8) for f in range(20)]
    tr = pd.DataFrame(rows, columns=["particle", "frame", "x", "y", "mass"])
    tc = d / "mv_tracks.csv"; tr.to_csv(tc, index=False)
    dd = pd.DataFrame({"particle": range(8), "n_spots": 20, "alpha": 1.0,
                       "D_fit_um2_s": rng.uniform(.01, .2, 8),
                       "mass_per_frame": rng.uniform(100, 900, 8),
                       "tau_fit_s": 0.05})
    dc = d / "mv_diffusion_coefficients.csv"; dd.to_csv(dc, index=False)
    for tau in (1, 5):
        ts.compute_tau_displacements(tr, tau, mpp=0.11, fps=100.0).to_csv(
            d / ts.tau_filename("mv", tau), index=False)
    row = pd.Series({"movie_id": "mv", "tracks_csv": str(tc), "diff_csv": str(dc),
                     "condition": "NT", "experiment": "d1", "mpp": "0.11", "fps": "100"})
    track_metrics.build_movie(row, 100.0, force=True)
    pool = track_metrics.load_pool(pd.DataFrame([row]), default_fps=100.0)
    assert len(pool) == 8 and pool["particle"].is_unique
    for c in ("D_fit_um2_s", "tau_fit_s", "v_path_um_s", "step_median_um_tau1",
              "mass_per_frame", "alpha"):
        assert c in pool.columns, c
    assert track_metrics.available_taus(pool) == [1, 5]


# ---------------------------------------------------------------------------
# Track filters
# ---------------------------------------------------------------------------

def _filter_args(**kw):
    from spt_pipeline.filters import CRITERIA
    class A: pass
    a = A()
    for k in CRITERIA:
        setattr(a, f"filter_{k}", None)
    a.filter_drop_missing = kw.pop("drop_missing", False)
    for k, v in kw.items():
        setattr(a, f"filter_{k}", v)
    return a


def _pool(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "condition": np.where(rng.random(n) < .5, "NT", "Tg"),
        "movie_id": rng.integers(0, 6, n).astype(str),
        "particle": np.arange(n),
        "D_fit_um2_s": rng.uniform(0.001, 0.5, n),
        "v_path_um_s": rng.uniform(0.5, 40, n),
        "mass_per_frame": rng.uniform(50, 1000, n),
        "alpha": rng.uniform(0, 1.6, n),
        "n_spots": rng.integers(5, 120, n),
    })


def test_filter_windows_compose_as_and():
    from spt_pipeline import filters
    p = _pool()
    tf = filters.TrackFilter.from_args(
        _filter_args(d=("0.05", "0.3"), brightness=("400", "-"), nspots=("20", "-")))
    kept, rep = tf.apply(p)
    ref = p[(p.D_fit_um2_s >= .05) & (p.D_fit_um2_s <= .3) &
            (p.mass_per_frame >= 400) & (p.n_spots >= 20)]
    assert len(kept) == len(ref)
    assert set(kept["particle"]) == set(ref["particle"])
    assert rep.iloc[-1]["criterion"] == "TOTAL"
    assert rep.iloc[-1]["n_before"] == len(p) and rep.iloc[-1]["n_after"] == len(kept)
    # attrition per criterion sums to the total removed
    assert rep[rep.criterion != "TOTAL"]["n_removed"].sum() == len(p) - len(kept)


def test_open_bounds_and_no_op_filter():
    from spt_pipeline import filters
    p = _pool()
    assert not bool(filters.TrackFilter.from_args(_filter_args()))
    kept, _ = filters.TrackFilter.from_args(_filter_args()).apply(p)
    assert len(kept) == len(p)
    tf = filters.TrackFilter.from_args(_filter_args(velocity=("-", "10")))
    assert tf.describe() == {"velocity": "<= 10"}
    kept, _ = tf.apply(p)
    assert len(kept) == int((p.v_path_um_s <= 10).sum())


def test_missing_column_is_reported_not_silently_ignored():
    from spt_pipeline import filters
    p = _pool().drop(columns=["v_path_um_s"])
    kept, rep = filters.TrackFilter.from_args(_filter_args(velocity=("1", "5"))).apply(p)
    assert len(kept) == len(p)                       # nothing dropped
    assert "NOT APPLIED" in rep.iloc[0]["status"]    # but it is on the record


def test_missing_values_kept_unless_asked():
    from spt_pipeline import filters
    p = _pool(50)
    p.loc[:9, "mass_per_frame"] = np.nan
    keep_kept, _ = filters.TrackFilter.from_args(
        _filter_args(brightness=("0", "-"))).apply(p)
    keep_drop, _ = filters.TrackFilter.from_args(
        _filter_args(brightness=("0", "-"), drop_missing=True)).apply(p)
    assert len(keep_kept) == 50
    assert len(keep_drop) == 40


def test_attrition_exposes_uneven_selection():
    from spt_pipeline import filters
    rng = np.random.default_rng(2)
    p = pd.DataFrame({
        "condition": ["NT"] * 100 + ["Tg"] * 100,
        "movie_id": ["a"] * 100 + ["b"] * 100,
        "mass_per_frame": np.r_[rng.normal(800, 60, 100), rng.normal(200, 60, 100)]})
    kept, _ = filters.TrackFilter.from_args(_filter_args(brightness=("500", "-"))).apply(p)
    attr = filters.per_condition_attrition(p, kept)
    nt = attr.loc[attr.condition == "NT", "frac_kept"].iloc[0]
    tg = attr.loc[attr.condition == "Tg", "frac_kept"].iloc[0]
    assert nt > 0.9 and tg < 0.1          # the selection effect is visible
    assert set(attr.columns) >= {"condition", "n_before", "n_after", "frac_kept"}


def test_absolute_bins_use_identical_edges_across_groups():
    from spt_pipeline import filters
    v = pd.Series([50, 150, 250, 350, 450, 1200])
    lab, tbl = filters.absolute_bins(v, edges=[0, 200, 400, 600])
    assert list(tbl["lower"]) == [0.0, 200.0, 400.0]
    assert tbl["kind"].iloc[0] == "absolute"
    assert pd.isna(lab.iloc[-1])          # 1200 is outside the outermost edge
    assert list(lab[:5]) == ["B1", "B1", "B2", "B2", "B3"]
    lab_q, tbl_q = filters.absolute_bins(v, n_quantile=3)
    assert tbl_q["kind"].iloc[0] == "quantile"


# ---------------------------------------------------------------------------
# Analysis-arm selection
# ---------------------------------------------------------------------------

def test_arm_aliases_resolve():
    from spt_pipeline.compare import ARMS, resolve_arms
    assert resolve_arms(["marker"]) == {"marker"}
    assert resolve_arms(["expression"]) == {"marker"}
    assert resolve_arms(["ficd"]) == {"marker"}
    assert resolve_arms(["step-size", "MSD"]) == {"steps", "msd"}
    assert resolve_arms(None) == set(ARMS)
    assert resolve_arms(["all"]) == set(ARMS)


def test_unknown_arm_raises():
    from spt_pipeline.compare import UnknownArmError, resolve_arms
    with pytest.raises(UnknownArmError, match="bogus"):
        resolve_arms(["marker", "bogus"])


def test_partial_run_does_not_claim_completeness(tmp_path):
    """A subset run must not overwrite the full statistical table."""
    from spt_pipeline import compare, manifest as m
    from spt_pipeline import tau_steps as ts, utils
    rng = np.random.default_rng(0)
    rows, mrows = [], []
    for cond in ("NT", "Tg"):
        for k in range(3):
            mid = f"{cond}{k}"
            d = tmp_path / mid; d.mkdir()
            tr = pd.DataFrame([(p, f, *rng.normal(50, 2, 2), rng.normal(300, 30))
                               for p in range(12) for f in range(15)],
                              columns=["particle", "frame", "x", "y", "mass"])
            tr["alpha"] = 1.0
            tc = d / f"{mid}_tracks.csv"; tr.to_csv(tc, index=False)
            dd = pd.DataFrame({"particle": range(12), "n_spots": 15, "alpha": 1.0,
                               "D_fit_um2_s": rng.uniform(.02, .2, 12),
                               "mass_per_frame": rng.uniform(100, 700, 12),
                               "tau_fit_s": 0.05,
                               "msd_fit_um2": rng.uniform(.004, .04, 12)})
            dc = d / f"{mid}_diffusion_coefficients.csv"; dd.to_csv(dc, index=False)
            for tau in (1, 5):
                ts.compute_tau_displacements(tr, tau, mpp=.11, fps=100.).to_csv(
                    d / ts.tau_filename(mid, tau), index=False)
            utils.dump_json({"movie_id": mid, "params": {"min_track_length": 4,
                                                         "diffusion_tau": 5}},
                            d / f"{mid}_context.json")
            rows.append({"movie_id": mid, "source_path": str(tc), "file_format": "tracks",
                         "condition": cond, "experiment": f"d{k % 2}",
                         "tracks_csv": str(tc), "diff_csv": str(dc), "marker_image": "",
                         "mpp": "0.11", "fps": "100", "n_frames": "300",
                         "n_channels": "1", "notes": ""})
    df = pd.DataFrame(rows, columns=m.COLUMNS)

    root = compare.compare_conditions(
        df, tmp_path, compare.CompareConfig(
            mpp=.11, fps=100., arms=compare.resolve_arms(["diffusion"]),
            run_label="diff_only"))
    names = {p.name for p in root.iterdir()}
    assert "diffusion_analysis" in names
    assert "step_size_analysis" not in names, "skipped arm left an empty folder"
    assert "msd_analysis" not in names
    assert "statistical_tests_partial.csv" in names
    assert "statistical_tests.csv" not in names

    full = compare.compare_conditions(
        df, tmp_path, compare.CompareConfig(mpp=.11, fps=100., run_label="full"))
    fnames = {p.name for p in full.iterdir()}
    assert "statistical_tests.csv" in fnames
    assert root != full            # labelled passes stay separate


def test_loc_error_reports_instead_of_silently_no_opping(capsys):
    """Missing msd_fit_um2 must be announced, not quietly ignored."""
    from spt_pipeline import compare
    pool = pd.DataFrame({"movie_id": ["a"] * 20, "condition": "NT",
                         "alpha": 0.1, "n_spots": 30,
                         "D_fit_um2_s": np.linspace(.01, .2, 20)})
    cfg = compare.CompareConfig(subtract_loc_error=True)
    out = compare._apply_loc_error(pool, cfg)
    msg = capsys.readouterr().out
    assert "NOT APPLIED" in msg and "msd_fit_um2" in msg
    assert "D_fit_um2_s_corr" not in out.columns


def test_loc_error_subtracts_and_reports_sigma(capsys):
    from spt_pipeline import compare
    tau = 0.05
    # 10 immobile tracks at the noise floor, 10 genuinely mobile
    msd = np.r_[np.full(10, 0.004), np.linspace(.02, .08, 10)]
    pool = pd.DataFrame({"movie_id": ["a"] * 20, "condition": "NT",
                         "alpha": np.r_[np.full(10, .1), np.full(10, 1.0)],
                         "n_spots": 30, "tau_fit_s": tau, "msd_fit_um2": msd,
                         "D_fit_um2_s": msd / (4 * tau)})
    out = compare._apply_loc_error(pool, compare.CompareConfig(subtract_loc_error=True))
    msg = capsys.readouterr().out
    assert "sigma_loc" in msg
    rep = out.attrs["loc_error_report"]
    assert rep["msd_floor_um2"].iloc[0] == pytest.approx(0.004)
    assert rep["sigma_loc_nm"].iloc[0] == pytest.approx(1000 * np.sqrt(0.004 / 4))
    # corrected D is the uncorrected D minus the floor, never negative
    assert (out["D_fit_um2_s_corr"] >= 0).all()
    assert out["D_fit_um2_s_corr"].iloc[0] == pytest.approx(0.0)
    assert out["D_fit_um2_s_corr"].iloc[-1] == pytest.approx((0.08 - 0.004) / (4 * tau))


def test_loc_error_warns_when_it_eats_the_distribution(capsys):
    from spt_pipeline import compare
    tau = 0.05
    msd = np.linspace(.02, .04, 20)          # "immobile" set is not immobile
    pool = pd.DataFrame({"movie_id": ["a"] * 20, "condition": "NT", "alpha": 0.1,
                         "n_spots": 30, "tau_fit_s": tau, "msd_fit_um2": msd,
                         "D_fit_um2_s": msd / (4 * tau)})
    compare._apply_loc_error(pool, compare.CompareConfig(subtract_loc_error=True))
    assert "WARNING" in capsys.readouterr().out


def test_per_movie_overlay_returns_summary(tmp_path):
    from spt_pipeline import plotting
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "condition": ["NT"] * 300 + ["Tg"] * 300,
        "experiment": "d1",
        "movie_id": np.r_[rng.integers(0, 3, 300), rng.integers(3, 6, 300)].astype(str),
        "mass_per_frame": np.r_[rng.normal(600, 80, 300), rng.normal(300, 80, 300)]})
    out = tmp_path / "b.png"
    s = plotting.per_movie_distribution_overlay(
        df, out, value_col="mass_per_frame", group_col="condition",
        xlabel="brightness", log_x=True)
    assert out.exists()
    assert len(s) == 6 and "median_mass_per_frame" in s.columns
    assert s.loc[s.condition == "NT", "median_mass_per_frame"].mean() > \
           s.loc[s.condition == "Tg", "median_mass_per_frame"].mean()


# ---------------------------------------------------------------------------
# Log-scaled fraction axis
# ---------------------------------------------------------------------------

def _hist_frame(cond, frac):
    return pd.DataFrame({"condition": cond,
                         "bin_center": np.arange(len(frac), dtype=float),
                         "fraction": np.asarray(frac, dtype=float),
                         "n_tracks": 10})


def test_log_y_masks_empty_bins_rather_than_flooring():
    """A zero-count bin must leave a gap, not be invented as a small number."""
    from spt_pipeline import plotting
    y = np.array([0.5, 0.0, 0.25, 0.0, 0.001])
    m = plotting._mask_nonpositive(y)
    assert np.isnan(m[1]) and np.isnan(m[3])
    assert m[0] == 0.5 and m[4] == 0.001
    assert not np.isnan(m[[0, 2, 4]]).any()


def test_log_y_limits_span_only_the_data():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from spt_pipeline import plotting
    fig, ax = plt.subplots()
    ok = plotting._log_y_setup(ax, [np.array([0.0, 1e-4, 0.2])])
    assert ok and ax.get_yscale() == "log"
    lo, hi = ax.get_ylim()
    assert lo == pytest.approx(0.5e-4) and hi == pytest.approx(0.2 * 1.6)
    plt.close(fig)
    # all-zero input must not switch the axis at all
    fig, ax = plt.subplots()
    assert plotting._log_y_setup(ax, [np.zeros(5)]) is False
    assert ax.get_yscale() == "linear"
    plt.close(fig)


def test_step_histogram_log_y_renders_and_drops_fill(tmp_path):
    from spt_pipeline import plotting
    hb = {"NT": _hist_frame("NT", [0.4, 0.3, 0.05, 0.0, 1e-4]),
          "Tg": _hist_frame("Tg", [0.5, 0.3, 0.02, 0.0, 1e-5])}
    lin, log = tmp_path / "lin.png", tmp_path / "log.png"
    plotting.overlay_step_histograms(hb, lin, title="t", xlim=5, ylim=0.6)
    plotting.overlay_step_histograms(hb, log, title="t", xlim=5, ylim=0.6, log_y=True)
    assert lin.exists() and log.exists()
    assert lin.stat().st_size > 0 and log.stat().st_size > 0


def test_all_histogram_types_accept_log_y(tmp_path):
    """Every distribution figure the flag claims to cover must accept it."""
    from spt_pipeline import plotting
    rng = np.random.default_rng(0)
    hb = {"NT": _hist_frame("NT", [0.4, 0.3, 0.05, 1e-4]),
          "Tg": _hist_frame("Tg", [0.5, 0.3, 0.02, 1e-5])}
    tw = {c: d.assign(mean_fraction_per_track=d["fraction"],
                      sem_fraction_per_track=d["fraction"] * 0.5)
          for c, d in hb.items()}
    split = {c: d.assign(length_bin="Q1") for c, d in hb.items()}
    for fn, arg in (
        (lambda p: plotting.overlay_step_histograms(hb, p, title="t", log_y=True), "step"),
        (lambda p: plotting.overlay_track_weighted(tw, p, title="t", log_y=True), "tw"),
        (lambda p: plotting.track_length_split_panels(split, p, title="t", log_y=True), "split"),
        (lambda p: plotting.cross_tau_overlay({1: hb["NT"], 5: hb["Tg"]}, p,
                                              condition="NT", tau_seconds={1: .01, 5: .05},
                                              log_y=True), "cross"),
        (lambda p: plotting.overlay_d_histograms(
            {"NT": rng.lognormal(-2, 1, 500), "Tg": rng.lognormal(-2.4, 1, 500)},
            np.linspace(0, 1, 30), p, log_y=True), "dhist"),
        (lambda p: plotting.overlay_value_histograms(
            {"NT": rng.normal(500, 80, 400), "Tg": rng.normal(300, 80, 400)},
            np.linspace(100, 800, 30), p, xlabel="b", log_y=True), "vhist"),
    ):
        out = tmp_path / f"{arg}.png"
        fn(out)
        assert out.exists() and out.stat().st_size > 0, arg


def test_yscale_mode_selects_outputs():
    from spt_pipeline import compare
    assert compare._yscales(compare.CompareConfig(hist_yscale="linear")) == [(False, "")]
    assert compare._yscales(compare.CompareConfig(hist_yscale="log")) == [(True, "")]
    assert compare._yscales(compare.CompareConfig(hist_yscale="both")) == \
        [(False, ""), (True, "_logy")]


# ---------------------------------------------------------------------------
# Density, cells and mis-linking risk
# ---------------------------------------------------------------------------

def _cloud(spec, nframes=25, size=512, seed=0):
    """spec = [(cx, cy, n_per_frame)] -> a detection table."""
    rng = np.random.default_rng(seed)
    rows = [(f, *p) for f in range(nframes) for (cx, cy, n) in spec
            for p in rng.normal((cx, cy), 42, (n, 2))]
    d = pd.DataFrame(rows, columns=["frame", "x", "y"])
    return d[(d.x.between(1, size - 2)) & (d.y.between(1, size - 2))]


def test_density_counts_are_within_frame_only():
    """A localization in another frame is never a neighbour."""
    from spt_pipeline import density
    # Same coordinates, different frames: nobody has a neighbour.
    d = pd.DataFrame({"frame": [0, 1, 2, 3], "x": [10.0] * 4, "y": [10.0] * 4})
    out = density.measure_detection_density(d, search_range_px=5, mpp=0.11)
    assert (out["field_neighbors_n"] == 0).all()
    assert (out["field_localizations_n"] == 1).all()
    assert out["nn_distance_px"].isna().all()
    # Same frame, two points 3 px apart: each sees one neighbour.
    d2 = pd.DataFrame({"frame": [0, 0], "x": [10.0, 13.0], "y": [10.0, 10.0]})
    o2 = density.measure_detection_density(d2, search_range_px=5, mpp=0.11)
    assert (o2["neighbors_R_n"] == 1).all()
    assert o2["nn_distance_px"].tolist() == pytest.approx([3.0, 3.0])


def test_density_edge_truncation_flag():
    from spt_pipeline import density
    d = pd.DataFrame({"frame": [0, 0], "x": [2.0, 64.0], "y": [64.0, 64.0]})
    out = density.measure_detection_density(d, search_range_px=5, field_radius_px=10,
                                            mpp=0.11, frame_shape=(128, 128))
    assert bool(out["field_window_truncated"].iloc[0]) is True    # near the edge
    assert bool(out["field_window_truncated"].iloc[1]) is False   # in the middle


def test_segmentation_separates_a_sparse_and_a_crowded_cell():
    """The case that matters: unequal expression in one field of view."""
    from spt_pipeline import density
    det = _cloud([(140, 140, 60), (380, 380, 300)], size=512)
    seg = density.segment_cells(det, (512, 512), mpp=0.11, min_cell_area_um2=5.0)
    assert seg.n_cells == 2 and seg.resolved
    det = det.assign(cell_id=density.assign_cells(det, seg))
    dens = density.measure_detection_density(det, search_range_px=5, mpp=0.11)
    dens["cell_id"] = det["cell_id"].to_numpy()
    med = dens[dens.cell_id > 0].groupby("cell_id")["field_localizations_n"].median()
    assert med.max() > med.min(), "the crowded cell should read as denser"


def test_confluent_field_is_reported_not_forced_apart():
    """Touching cells become one field, and the note says why."""
    from spt_pipeline import density
    rng = np.random.default_rng(1)
    det = pd.DataFrame([(f, *p) for f in range(20)
                        for p in rng.uniform(10, 502, (900, 2))],
                       columns=["frame", "x", "y"])
    seg = density.segment_cells(det, (512, 512), mpp=0.11)
    assert seg.n_cells == 1
    assert seg.resolved is False
    assert "confluent" in seg.note or "touching" in seg.note


def test_link_safety_uses_a_movie_scale_not_the_track_speed():
    """A fast track must not be scored as risky just for being fast.

    Otherwise the QC metric would correlate with mobility by construction,
    which is precisely the confound it exists to detect.
    """
    from spt_pipeline import density
    rng = np.random.default_rng(0)
    rows = []
    for pid, sigma in enumerate([0.4, 0.4, 0.4, 4.0]):     # one much faster track
        pos = np.cumsum(rng.normal(0, sigma, (25, 2)), axis=0) + (60, 60)
        for f in range(25):
            rows.append((pid, f, pos[f, 0], pos[f, 1]))
    tr = pd.DataFrame(rows, columns=["particle", "frame", "x", "y"])
    scale = density.movie_step_scale_px(tr)
    assert np.isfinite(scale) and scale > 0
    d = density.measure_detection_density(tr, search_range_px=5, mpp=0.11)
    per = density.summarize_track_density(d, step_scale_px=scale)
    # Every track is divided by the same movie-level scale.
    ratio = per["link_safety_ratio"] / per["track_median_nn_distance_px"]
    assert np.allclose(ratio.dropna(), 1.0 / scale)


def test_p_link_ambiguous_is_poisson_in_the_competitor_count():
    from spt_pipeline import density
    d = pd.DataFrame({"particle": [0, 0, 1, 1], "frame": [0, 1, 0, 1],
                      "x": [1.0, 1.0, 2.0, 2.0], "y": [1.0, 1.0, 2.0, 2.0],
                      "field_localizations_n": [3, 3, 1, 1],
                      "neighbors_R_n": [2, 2, 0, 0],
                      "neighbors_2R_n": [2, 2, 0, 0],
                      "nn_distance_px": [1.4, 1.4, 1.4, 1.4]})
    per = density.summarize_track_density(d, step_scale_px=1.0).set_index("particle")
    assert per.loc[0, "p_link_ambiguous"] == pytest.approx(1 - np.exp(-2))
    assert per.loc[1, "p_link_ambiguous"] == pytest.approx(0.0)


def test_density_strata_are_absolute_and_shared():
    """Strata boundaries must not depend on which condition you are looking at."""
    from spt_pipeline import density
    a = density.density_strata(pd.Series([1.0, 3.0, 5.0, 20.0]))
    b = density.density_strata(pd.Series([5.0, 20.0]))
    assert list(a) == ["[1,2)", "[2,4)", "[4,8)", "[16,32)"]
    assert list(b) == ["[4,8)", "[16,32)"]      # same labels for the same values
    assert density.density_strata(pd.Series([0.0, -1.0])).tolist() == ["invalid"] * 2


def test_cem_matching_equalizes_counts_per_stratum():
    from spt_pipeline import density
    rng = np.random.default_rng(0)
    rows = []
    for cond, n_sparse, n_dense in (("NT", 40, 5), ("Tg", 5, 40)):
        for i in range(n_sparse):
            rows.append({"movie_id": f"{cond}_a", "particle": i, "condition": cond,
                         "experiment": "d1", "track_median_field_localizations_n": 2.5})
        for i in range(n_dense):
            rows.append({"movie_id": f"{cond}_b", "particle": i, "condition": cond,
                         "experiment": "d1", "track_median_field_localizations_n": 9.0})
    res = density.density_match_tracks(pd.DataFrame(rows), seed=0)
    sel = res.assignments[res.assignments.density_match_selected]
    counts = sel.groupby(["density_match_stratum", "condition"]).size().unstack()
    # Both conditions contribute equally in every stratum that was used.
    assert (counts["NT"] == counts["Tg"]).all()
    assert set(counts.sum(axis=1)) == {10, 10}          # 2*min(40,5) per stratum


def test_cem_reports_when_conditions_never_overlap():
    """Disjoint density ranges mean no matched comparison exists."""
    from spt_pipeline import density
    rows = ([{"movie_id": "a", "particle": i, "condition": "NT", "experiment": "d1",
              "track_median_field_localizations_n": 2.0} for i in range(20)]
            + [{"movie_id": "b", "particle": i, "condition": "Tg", "experiment": "d1",
                "track_median_field_localizations_n": 40.0} for i in range(20)])
    res = density.density_match_tracks(pd.DataFrame(rows), seed=0)
    assert int(res.assignments.density_match_selected.sum()) == 0
    assert set(res.assignments.density_match_reason) == {"no_shared_stratum"}


def test_origin_alignment_moves_every_track_to_zero():
    from spt_pipeline import density
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(6):
        pos = np.cumsum(rng.normal(0, 2, (15, 2)), axis=0) + rng.uniform(10, 90, 2)
        for f in range(15):
            rows.append((pid, f, pos[f, 0], pos[f, 1]))
    tr = pd.DataFrame(rows, columns=["particle", "frame", "x", "y"])
    al = density.origin_aligned(tr, mpp=0.11)
    firsts = al.sort_values("frame").groupby("particle").first()
    assert np.allclose(firsts["dx_um"], 0.0) and np.allclose(firsts["dy_um"], 0.0)
    assert len(al) == len(tr)


def test_density_arm_is_selectable():
    from spt_pipeline.compare import ARMS, resolve_arms
    assert "density" in ARMS
    assert resolve_arms(["density"]) == {"density"}
    assert resolve_arms(["crowding"]) == {"density"}
