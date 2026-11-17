"""Command-line interface (thin orchestration layer).

The v2 CLI owns the new modular architecture:

* ``prep scan``               -> build/refresh the manifest
* ``track trackpy``           -> Phase A tracking (+ fixed-tau steps + QC)
* ``compute tau-steps``       -> Phase B, for adding further lags later
* ``analyze compare-conditions`` -> Phase C comparison (+ optional advanced)
* ``workflow auto``           -> the four above, end to end

Peripheral subcommands (``prep nd2-to-tiff``/``stack-channels``/
``find-ome-tif``/``trackmate``, ``draw``, ``utils``) are dispatched unchanged to
:mod:`spt_pipeline.legacy`.

The brightness-mobility / step-brightness / dwell analysis arm has been retired;
its code is preserved under ``retired/`` at the repository root.

Argparse namespaces are unpacked into explicit keyword arguments before calling
module functions, keeping the modules independently testable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__

# Commands handled by the new modular code (everything else -> legacy).
_NEW_ROUTES = {
    ("prep", "scan"),
    ("track", "trackpy"),
    ("analyze", "compare-conditions"),
    ("analyze", "marker-intensity"),
    ("analyze", "density"),
    ("workflow", "auto"),
}


#: Groups whose help must come from the v2 parser. Asking for
#: ``analyze --help`` and being shown the legacy parser -- which does not list
#: ``marker-intensity`` -- makes a command effectively undiscoverable.
_V2_HELP_GROUPS = ("prep", "track", "analyze", "compute", "workflow")


def _is_new_route(argv: List[str]) -> bool:
    if not argv:
        return False
    if argv[0] == "compute":
        return True
    # A group named with no subcommand, or followed only by a flag, is a
    # request for help: answer it from the v2 parser.
    if argv[0] in _V2_HELP_GROUPS and (len(argv) == 1 or argv[1].startswith("-")):
        return True
    if len(argv) >= 2 and (argv[0], argv[1]) in _NEW_ROUTES:
        return True
    return False


def _data_root(a) -> Optional[Path]:
    """The portable-path anchor, if the user overrode it."""
    dr = getattr(a, "data_root", None)
    return Path(dr) if dr else None


def _add_manifest_path_args(p: argparse.ArgumentParser) -> None:
    """``--data-root``: anchor for the manifest's OS-portable relative paths.

    Manifest paths are stored relative to the manifest file, so a manifest and
    its dataset move together between Windows, macOS, and Linux untouched. Pass
    ``--data-root`` when the dataset is mounted somewhere other than beside the
    manifest (a different drive letter, a network share, a container mount).
    """
    p.add_argument("--data-root", default=None, metavar="DIR",
                   help="Anchor for relative manifest paths (default: the manifest's "
                        "own directory). Use when the dataset is mounted elsewhere.")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spt-pipeline",
        description="SPT unified pipeline (v2). Modular phases: prep scan -> "
                    "track trackpy -> compute tau-steps -> analyze compare-conditions.",
    )
    p.add_argument("--version", action="version", version=f"spt-pipeline {__version__}")
    sub = p.add_subparsers(dest="command")

    # ---- prep scan ----
    prep = sub.add_parser(
        "prep", help="Data preparation (scan; + legacy conversions)",
        epilog="Format conversions (trackmate, nd2-to-tiff, stack-channels, "
               "find-ome-tif) are also available under this group.")
    prep_sub = prep.add_subparsers(dest="prep_cmd")
    scan = prep_sub.add_parser("scan", help="Scan input dirs -> draft manifest CSV")
    scan.add_argument("input_dirs", nargs="+", help="Directories to scan")
    scan.add_argument("--output", "-o", default="manifest.csv", help="Manifest CSV path")
    scan.add_argument("--mpp", type=float, default=None, help="Default microns-per-pixel")
    scan.add_argument("--fps", type=float, default=None, help="Default frames-per-second")
    scan.add_argument("--exclude-pattern", action="append", default=[],
                      help="Filename substring to exclude (repeatable)")
    scan.add_argument("--condition-tokens", default=None,
                      help="Comma-separated condition tokens for inference")
    scan.add_argument("--from-tracks", action="store_true",
                      help="Discover existing *_tracks.csv instead of raw images")
    _add_manifest_path_args(scan)

    # ---- track trackpy ----
    track = sub.add_parser("track", help="Tracking (trackpy)")
    track_sub = track.add_subparsers(dest="track_cmd")
    tp = track_sub.add_parser("trackpy", help="Phase A: trackpy tracking from manifest")
    tp.add_argument("--manifest", "-m", required=True)
    tp.add_argument("--out-dir", "-o", required=True)
    tp.add_argument("--mpp", type=float, default=0.183)
    tp.add_argument("--fps", type=float, default=50.0)
    tp.add_argument("--diam", type=int, default=7)
    tp.add_argument("--minmass", type=float, default=40.0)
    tp.add_argument("--range", type=int, default=5, dest="search_range")
    tp.add_argument("--memory", type=int, default=0)
    tp.add_argument("--min-separation", type=int, default=3)
    tp.add_argument("--percentile", type=float, default=90.0)
    tp.add_argument("--min-track-length", type=int, default=4)
    tp.add_argument("--diffusion-tau", type=int, default=5)
    tp.add_argument("--channel", type=int, default=0)
    tp.add_argument("--tau", type=int, nargs="+", default=list(_default_taus()),
                    help="Fixed-tau lags (frames) to compute inline, as part of tracking "
                         "(default: 1 5 10). Further lags can be added later with "
                         "'compute tau-steps' without re-tracking.")
    tp.add_argument("--no-tau-steps", action="store_true",
                    help="Skip the inline fixed-tau displacement computation")
    tp.add_argument("--no-qc-plots", action="store_true",
                    help="Skip the localization/linking QC figures")
    tp.add_argument("--qc-detections-overlay-frames", type=int, nargs="+", default=[100],
                    metavar="N",
                    help="Frame number(s) to draw the QC detections overlay on "
                         "(default: 100). Frames past the end of a movie are skipped.")
    tp.add_argument("--no-density", action="store_true",
                    help="Skip the local-density / cell-segmentation measurement")
    tp.add_argument("--density-field-radius-px", type=float, default=None,
                    metavar="PX",
                    help="Crowding neighbourhood radius in pixels (default: 10, which "
                         "is ~1.1 um at 0.11 um/px)")
    tp.add_argument("--no-save-detections", action="store_true",
                    help="Do not keep the pre-link detection table. Keeping it is what "
                         "lets density be re-measured later without re-tracking")
    tp.add_argument("--n-jobs", type=int, default=8)
    tp.add_argument("--force", action="store_true")
    _add_manifest_path_args(tp)

    # ---- compute tau-steps ----
    comp = sub.add_parser("compute", help="Phase B: fixed-tau displacement computation")
    comp_sub = comp.add_subparsers(dest="compute_cmd")
    ts = comp_sub.add_parser("tau-steps", help="Compute per-movie fixed-tau displacements")
    ts.add_argument("--manifest", "-m", required=True)
    ts.add_argument("--out-dir", "-o", required=True)
    ts.add_argument("--tau", type=int, nargs="+", default=[1, 3, 5, 8],
                    help="Tau values in frames (default: 1 3 5 8)")
    ts.add_argument("--mpp", type=float, default=0.183)
    ts.add_argument("--fps", type=float, default=50.0)
    ts.add_argument("--min-nspots", type=int, default=0)
    ts.add_argument("--min-alpha", type=float, default=None)
    ts.add_argument("--non-overlapping", action="store_true")
    ts.add_argument("--n-jobs", type=int, default=8)
    ts.add_argument("--force", action="store_true")
    _add_manifest_path_args(ts)

    # ---- analyze compare-conditions ----
    an = sub.add_parser("analyze", help="Analysis (compare-conditions)")
    an_sub = an.add_subparsers(dest="analyze_cmd")
    cc = an_sub.add_parser("compare-conditions", help="Phase C: condition-level comparison")
    cc.add_argument("--manifest", "-m", required=True)
    cc.add_argument("--out-dir", "-o", required=True)
    cc.add_argument("--tau", type=int, nargs="*", default=None,
                    help="Tau values to include (default: auto-discover cached taus)")
    cc.add_argument("--mpp", type=float, default=0.183)
    cc.add_argument("--fps", type=float, default=50.0)
    cc.add_argument("--min-nspots", type=int, default=0)
    cc.add_argument("--min-alpha", type=float, default=None)
    cc.add_argument("--subtract-loc-error", action="store_true")
    cc.add_argument("--loc-error-alpha-max", type=float, default=0.3)
    cc.add_argument("--loc-error-min-nspots", type=int, default=10)
    cc.add_argument("--confinement-threshold-um", type=float, default=None)
    cc.add_argument("--track-length-bins", type=int, nargs="*", default=None)
    cc.add_argument("--step-bin-width-um", default="auto")
    cc.add_argument("--diffusion-bin-width", default="auto")
    cc.add_argument("--logd-bin-width", default="auto")
    # GMM subpopulation fitting is opt-in (off by default).
    cc.add_argument("--gmm", action="store_true",
                    help="Enable BIC-selected GMM subpopulation fitting of D (off by default)")
    cc.add_argument("--max-gmm-k", type=int, default=5)
    cc.add_argument("--msd-max-lag", type=int, default=None,
                    help="Max lag (frames) for the fixed-support MSD; default = (min_track_length - 1) read from the track-trackpy context, i.e. full support over all surviving tracks")
    cc.add_argument("--msd-min-tracks", type=int, default=20,
                    help="Min supporting tracks/condition, only used as a fallback when the tracking context (min_track_length) is unavailable")
    cc.add_argument("--marker-intensity-bins", type=int, default=4,
                    help="Number of expression quantile bins for the marker analysis "
                         "(default: 4)")
    cc.add_argument("--marker-min-disk-coverage", type=float, default=0.0,
                    help="Drop tracks whose sampling disk overhung the frame edge by "
                         "more than this fraction (0-1; default: 0, keep all)")
    cc.add_argument("--marker-edge-margin-px", type=float, default=0.0,
                    help="Drop tracks coming within this many pixels of any frame edge "
                         "(default: 0, keep all)")
    cc.add_argument("--brightness-bin-edges", type=float, nargs="+", default=None,
                    metavar="EDGE",
                    help="Absolute track-brightness bin edges for matched comparison, "
                         "e.g. --brightness-bin-edges 0 200 400 800. Absolute, not "
                         "quantile: matching requires the same boundaries in every "
                         "condition. Tracks outside the outermost edges are excluded "
                         "from the binned figure.")
    cc.add_argument("--brightness-bins", type=int, default=None, metavar="N",
                    help="Quantile brightness bins, used only if --brightness-bin-edges "
                         "is not given")
    cc.add_argument("--density-strata", type=int, default=4, metavar="N",
                    help="Number of link-safety strata for the crowding comparison "
                         "(density itself uses power-of-two bins; default: 4)")
    cc.add_argument("--density-match", action="store_true",
                    help="Also produce one density-balanced subset by coarsened exact "
                         "matching, with a before/after balance plot. Discards tracks; "
                         "the stratified comparison keeps everything and runs anyway")
    cc.add_argument("--density-match-column",
                    default="track_median_field_localizations_n",
                    help="Track column matched on (default: median local density)")
    cc.add_argument("--density-match-length-bins", type=int, nargs="+", default=None,
                    metavar="EDGE",
                    help="Also match jointly on track length, using these edges")
    cc.add_argument("--density-match-seed", type=int, default=1729)
    cc.add_argument("--hist-yscale", choices=("linear", "log", "both"),
                    default="linear",
                    help="Fraction-axis scale for the step-size, D and brightness "
                         "histograms. 'log' makes the heavy tail legible -- the mobile "
                         "population lives there, at fractions too small to see on a "
                         "linear axis. 'both' also writes a *_logy.png companion "
                         "(default: linear)")
    cc.add_argument("--run-only", nargs="+", default=None, metavar="ARM",
                    help="Run only these analysis arms: steps, diffusion, msd, "
                         "marker, advanced. Each reads cached per-movie outputs, so "
                         "e.g. --run-only marker re-does the expression analysis "
                         "without redoing step-size, diffusion or MSD. Aliases "
                         "accepted (expression, ficd, step-size, ...). A partial run "
                         "writes statistical_tests_partial.csv so it cannot overwrite "
                         "the complete table.")
    cc.add_argument("--run-label", default="", metavar="NAME",
                    help="Write this pass to compare_conditions/<NAME>/ so successive "
                         "filter settings sit side by side instead of overwriting")
    _add_filter_args(cc)
    _add_advanced_args(cc)
    _add_manifest_path_args(cc)

    # ---- analyze marker-intensity ----
    mi = an_sub.add_parser(
        "marker-intensity",
        help="Measure expression-marker intensity per track from paired marker images",
        description="Pairs each movie with its expression-marker image, measures the "
                    "mean marker signal over the detection disk at every localization, "
                    "and writes one intensity per track. Runs independently of tracking, "
                    "so it can be repeated with a different disk size without re-tracking.")
    mi.add_argument("--manifest", "-m", required=True)
    mi.add_argument("--out-dir", "-o", default=None,
                    help="Run-log directory (default: beside the manifest)")
    mi.add_argument("--diameter", type=int, default=None,
                    help="Sampling disk diameter in pixels (default: the --diam used "
                         "at tracking, read from the per-movie context)")
    mi.add_argument("--marker-channel", type=int, default=0,
                    help="Zero-based channel of a multichannel marker image (default: 0)")
    mi.add_argument("--no-append", action="store_true",
                    help="Write the per-track CSV only; leave tracks/diffusion CSVs alone")
    mi.add_argument("--force", action="store_true",
                    help="Re-measure movies that already have a marker-intensity CSV")
    _add_manifest_path_args(mi)

    # ---- analyze density ----
    dn = an_sub.add_parser(
        "density",
        help="Measure local particle density, segment cells, and score linking risk",
        description="Measures how crowded each track's neighbourhood was, labels the "
                    "cells in each field of view, and scores how much room the linker "
                    "had. Uses the saved pre-link detections when tracking wrote them; "
                    "otherwise falls back to the linked tracks and flags every row "
                    "density_source=post_link, which understates crowding.")
    dn.add_argument("--manifest", "-m", required=True)
    dn.add_argument("--out-dir", "-o", default=None,
                    help="Run-log directory (default: beside the manifest)")
    dn.add_argument("--field-radius-px", type=float, default=None, metavar="PX",
                    help="Crowding neighbourhood radius (default: 10 px)")
    dn.add_argument("--search-range-px", type=float, default=None, metavar="PX",
                    help="Linking search range used for the risk metrics "
                         "(default: read from each movie's tracking context)")
    dn.add_argument("--no-segment", action="store_true",
                    help="Skip cell segmentation; measure per-track density only")
    dn.add_argument("--force", action="store_true",
                    help="Re-measure movies that already have a density table")
    _add_manifest_path_args(dn)

    # ---- workflow auto (v2 end-to-end) ----
    wf = sub.add_parser("workflow", help="End-to-end workflows")
    wf_sub = wf.add_subparsers(dest="workflow_cmd")
    wa = wf_sub.add_parser(
        "auto",
        help="Full core path: prep scan -> track trackpy -> analyze compare-conditions",
        description="Run the whole v2 core path on one or more input directories. "
                    "Stops after 'prep scan' if any condition/experiment is still "
                    "FILL_IN, so you can correct the manifest before spending compute.")
    wa.add_argument("input_dirs", nargs="+", help="Directories of raw movies to scan")
    wa.add_argument("--out-dir", "-o", required=True, help="Output root")
    wa.add_argument("--manifest", "-m", default=None,
                    help="Manifest path (default: <out-dir>/manifest.csv). If it already "
                         "exists, the scan step is skipped and the existing file is used.")
    wa.add_argument("--mpp", type=float, default=0.183)
    wa.add_argument("--fps", type=float, default=50.0)
    wa.add_argument("--tau", type=int, nargs="+", default=list(_default_taus()),
                    help="Fixed-tau lags in frames (default: 1 5 10)")
    wa.add_argument("--condition-tokens", default=None,
                    help="Comma-separated condition tokens for inference")
    wa.add_argument("--exclude-pattern", action="append", default=[],
                    help="Filename substring to exclude (repeatable)")
    # tracking knobs (mirror 'track trackpy' defaults)
    wa.add_argument("--diam", type=int, default=7)
    wa.add_argument("--minmass", type=float, default=40.0)
    wa.add_argument("--range", type=int, default=5, dest="search_range")
    wa.add_argument("--memory", type=int, default=0)
    wa.add_argument("--min-separation", type=int, default=3)
    wa.add_argument("--percentile", type=float, default=90.0)
    wa.add_argument("--min-track-length", type=int, default=4)
    wa.add_argument("--diffusion-tau", type=int, default=5)
    wa.add_argument("--channel", type=int, default=0)
    wa.add_argument("--no-qc-plots", action="store_true")
    wa.add_argument("--qc-detections-overlay-frames", type=int, nargs="+", default=[100],
                    metavar="N")
    wa.add_argument("--n-jobs", type=int, default=8)
    wa.add_argument("--force", action="store_true")
    # comparison knobs
    wa.add_argument("--gmm", action="store_true")
    wa.add_argument("--subtract-loc-error", action="store_true")
    wa.add_argument("--confinement-threshold-um", type=float, default=None)
    wa.add_argument("--skip-compare", action="store_true",
                    help="Stop after tracking; do not run compare-conditions")
    _add_advanced_args(wa)
    _add_manifest_path_args(wa)
    return p


def _filter_names():
    from .filters import CRITERIA
    return list(CRITERIA)


def _default_taus():
    from .tracking import DEFAULT_TAUS
    return DEFAULT_TAUS


def _add_filter_args(p: argparse.ArgumentParser) -> None:
    """Track-filter windows, applied to one shared population.

    Each takes MIN and MAX; use ``-`` for an open end, so
    ``--filter-d 0.05 -`` reads "D of at least 0.05". Filters apply to every
    analysis in the run, and the attrition is written to filter_report.csv.
    """
    from .filters import CRITERIA
    g = p.add_argument_group(
        "track filters",
        "Applied to one shared track population, so every figure in the run "
        "describes the same tracks. Use '-' for an open bound.")
    for name, (col, label) in CRITERIA.items():
        g.add_argument(f"--filter-{name}", nargs=2, metavar=("MIN", "MAX"),
                       default=None, help=f"Keep tracks whose {label} is in range")
    g.add_argument("--filter-drop-missing", action="store_true",
                   help="Drop tracks with no value for an active filter "
                        "(default: keep them, since missing means unmeasured)")


def _add_advanced_args(p: argparse.ArgumentParser) -> None:
    """Flags for the advanced (VACF / CFBM / angles / EB / α₂ / rheology) suite."""
    from .advanced import ADVANCED_KEYWORDS, DEFAULT_PARTICLE_RADIUS_UM
    keys = ", ".join(sorted(ADVANCED_KEYWORDS))
    p.add_argument("--advanced", action="store_true",
                   help="Run the advanced trajectory analyses (all of them by default)")
    p.add_argument("--advanced-only", nargs="+", default=None, metavar="NAME",
                   help=f"Run only these advanced analyses (implies --advanced). "
                        f"Case-insensitive; aliases accepted. Choices: {keys}. "
                        f"Example: --advanced-only VACF a2")
    p.add_argument("--advanced-min-track-len", type=int, default=None,
                   help="Minimum track length for the advanced analyses "
                        "(default: the --min-track-length used at tracking)")
    p.add_argument("--advanced-max-tlag", type=int, default=None,
                   help="Maximum lag in frames for VACF/MSD/angle analyses "
                        "(default: the max lag used at tracking, capped by track length)")
    p.add_argument("--advanced-particle-radius", type=float, default=None,
                   metavar="UM",
                   help=f"Probe radius in microns for the rheology G'/G'' "
                        f"(default: {DEFAULT_PARTICLE_RADIUS_UM} um = "
                        f"{DEFAULT_PARTICLE_RADIUS_UM * 1000:.0f} nm)")
    p.add_argument("--advanced-temperature", type=float, default=298.15,
                   help="Temperature in Kelvin, for the rheology G'/G'' (default: 298.15)")
    p.add_argument("--advanced-n-jobs", type=int, default=4)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_prep_scan(a) -> int:
    from . import manifest
    from .utils import dump_json
    tokens = [t.strip() for t in a.condition_tokens.split(",")] if a.condition_tokens else None
    res = manifest.scan(
        [Path(d) for d in a.input_dirs], default_mpp=a.mpp, default_fps=a.fps,
        exclude_patterns=a.exclude_pattern, condition_tokens=tokens, from_tracks=a.from_tracks)
    manifest.write(res.manifest, Path(a.output), path_base=_data_root(a))
    dump_json(res.log, Path(a.output).with_suffix(".scan_log.json"))
    log = res.log
    print(f"Found {log['n_movies']} movies across {len(log['input_dirs'])} directories. "
          f"{log['n_resolved']} fully resolved, {log['n_need_attention']} need attention.")
    print(f"Draft manifest: {a.output}")
    print("  Paths are stored relative to the manifest, so the manifest and its data "
          "move together across operating systems.")
    if log["n_need_attention"]:
        print("  Review FILL_IN values in the 'condition'/'experiment' columns before proceeding.")
    return 0


def _handle_track_trackpy(a) -> int:
    from . import manifest, tracking
    from .run_log import RunLog
    mpath = Path(a.manifest)
    base = _data_root(a)
    df = manifest.load(mpath, path_base=base)
    warnings = manifest.validate(df, default_mpp=a.mpp, default_fps=a.fps)
    for w in warnings:
        print(f"  warning: {w}")
    # Marker images and single-frame acquisitions have no time axis; they are
    # excluded here rather than failing one by one inside the tracker.
    non_track = [(str(r["movie_id"]), manifest.skip_tracking_reason(r))
                 for _, r in df.iterrows() if not manifest.is_trackable(r)]
    if non_track:
        print(f"  Not tracking {len(non_track)} row(s):")
        for mid, why in non_track[:5]:
            print(f"    {mid}: {why}")
        if len(non_track) > 5:
            print(f"    ... and {len(non_track) - 5} more")
    df = manifest.movie_rows(df)
    if df.empty:
        print("No trackable movies in the manifest.")
        return 0

    out_dir = Path(a.out_dir)
    rl = RunLog("track trackpy", vars(a), out_dir, mpath)
    taus = () if getattr(a, "no_tau_steps", False) else tuple(a.tau or ())
    defaults = tracking.TrackingParams(
        diameter=a.diam, minmass=a.minmass, search_range=a.search_range, memory=a.memory,
        min_separation=a.min_separation, percentile=a.percentile,
        min_track_length=a.min_track_length, diffusion_tau=a.diffusion_tau,
        channel=a.channel, mpp=a.mpp, fps=a.fps,
        tau_steps=taus, qc_plots=not getattr(a, "no_qc_plots", False),
        qc_detections_overlay_frames=tuple(
            getattr(a, "qc_detections_overlay_frames", None) or (100,)),
        density=not getattr(a, "no_density", False),
        density_field_radius_px=(getattr(a, "density_field_radius_px", None)
                                 or tracking.DEFAULT_FIELD_RADIUS_PX),
        save_detections=not getattr(a, "no_save_detections", False))
    results = tracking.track_all(df, out_dir, defaults, n_jobs=a.n_jobs, force=a.force)
    n_ok = n_skip = n_fail = n_none = 0
    no_track_ids = []
    for r in results:
        if r.status == "ok":
            n_ok += 1
            manifest.update_row_outputs(mpath, r.movie_id, r.tracks_csv, r.diff_csv,
                                        note="")
        elif r.status == "skipped":
            n_skip += 1
        elif r.status == "no_tracks":
            # Not a failure: an empty field of view is data. Leave tracks_csv
            # blank so every downstream stage skips this movie, and record why.
            n_none += 1
            no_track_ids.append(r.movie_id)
            manifest.update_row_outputs(mpath, r.movie_id,
                                        note=f"{manifest.NO_TRACKS_NOTE}: {r.error}")
            rl.note(f"{r.movie_id}: no tracks ({r.error})")
        else:
            n_fail += 1
            rl.note(f"{r.movie_id}: {r.error}")
    print(f"Tracking complete: {n_ok} tracked, {n_skip} already done, "
          f"{n_none} with no tracks, {n_fail} failed.")
    if no_track_ids:
        shown = ", ".join(no_track_ids[:5]) + ("..." if len(no_track_ids) > 5 else "")
        print(f"  No qualifying tracks (skipped downstream, not an error): {shown}")
        print("  See each movie's *_NO_TRACKS.json for the thresholds in force.")
    if taus:
        print(f"  Fixed-tau displacements computed at tau = {', '.join(map(str, taus))} frames.")

    # Expression-marker intensities, automatically, when marker rows exist.
    full = manifest.load(mpath, path_base=base)
    if not manifest.marker_rows(full).empty:
        print("\nMarker images found; measuring expression intensity per track.")
        _handle_marker_intensity(argparse.Namespace(
            manifest=str(mpath), out_dir=str(out_dir), diameter=a.diam,
            marker_channel=a.channel, no_append=False, force=a.force,
            data_root=a.data_root))

    rl.finish(n_processed=n_ok, n_skipped=n_skip + n_none,
              extra={"n_failed": n_fail, "n_no_tracks": n_none,
                     "no_track_movies": no_track_ids,
                     "movies": [r.__dict__ for r in results]})
    return 0 if n_fail == 0 else 1


def _handle_compute_tau_steps(a) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from . import manifest, tau_steps
    from .run_log import RunLog
    from .utils import resolve_n_jobs
    mpath = Path(a.manifest)
    df = manifest.load(mpath, path_base=_data_root(a))
    for w in manifest.validate(df, default_mpp=a.mpp, default_fps=a.fps, require_tracks=True):
        print(f"  warning: {w}")
    # Movies with no tracks contribute no displacements; skip rather than fail.
    df = manifest.rows_with_tracks(df)
    out_dir = Path(a.out_dir)
    rl = RunLog("compute tau-steps", vars(a), out_dir, mpath)

    jobs = []
    for _, row in df.iterrows():
        mpp, fps = manifest.row_mpp_fps(row, a.mpp, a.fps)
        tracks_csv = Path(row["tracks_csv"])
        movie_dir = tracks_csv.parent
        diff_csv = row.get("diff_csv", "") or None
        jobs.append(dict(movie_id=row["movie_id"], tracks_csv=tracks_csv, tau_list=a.tau,
                         mpp=mpp, fps=fps, out_dir=movie_dir,
                         diff_csv=Path(diff_csv) if diff_csv else None,
                         min_nspots=a.min_nspots, min_alpha=a.min_alpha,
                         non_overlapping=a.non_overlapping, force=a.force))
    n_jobs = resolve_n_jobs(a.n_jobs)
    n_done = n_skip = 0
    if n_jobs == 1:
        for j in jobs:
            r = tau_steps.process_movie(**j)
            n_done += len(r["computed"]); n_skip += len(r["skipped"])
            print(f"  {r['movie_id']}: computed {r['computed']}, skipped {r['skipped']}")
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = {ex.submit(tau_steps.process_movie, **j): j["movie_id"] for j in jobs}
            for fut in as_completed(futs):
                r = fut.result()
                n_done += len(r["computed"]); n_skip += len(r["skipped"])
                print(f"  {r['movie_id']}: computed {r['computed']}, skipped {r['skipped']}")
    print(f"Tau-step computation complete: {n_done} (movie,tau) computed, {n_skip} skipped.")
    rl.finish(n_processed=n_done, n_skipped=n_skip)
    return 0


def _advanced_selection(a):
    """Resolve --advanced / --advanced-only into (enabled, selected_keywords)."""
    from .advanced import resolve_selection
    only = getattr(a, "advanced_only", None)
    if only:
        return True, resolve_selection(only)
    if getattr(a, "advanced", False):
        return True, resolve_selection(None)
    return False, frozenset()


def _handle_compare_conditions(a) -> int:
    from . import compare, filters, manifest
    from .advanced import DEFAULT_PARTICLE_RADIUS_UM
    from .run_log import RunLog
    mpath = Path(a.manifest)
    df = manifest.load(mpath, path_base=_data_root(a))
    for w in manifest.validate(df, default_mpp=a.mpp, default_fps=a.fps, require_tracks=True):
        print(f"  warning: {w}")
    skipped = manifest.skipped_movies(df)
    if skipped:
        shown = ", ".join(skipped[:5]) + ("..." if len(skipped) > 5 else "")
        print(f"  Excluding {len(skipped)} movie(s) with no tracks: {shown}")
    df = manifest.rows_with_tracks(df)
    out_dir = Path(a.out_dir)
    rl = RunLog("analyze compare-conditions", vars(a), out_dir, mpath)
    adv_on, adv_sel = _advanced_selection(a)
    cfg = compare.CompareConfig(
        taus=a.tau, mpp=a.mpp, fps=a.fps, min_nspots=a.min_nspots, min_alpha=a.min_alpha,
        subtract_loc_error=a.subtract_loc_error, loc_error_alpha_max=a.loc_error_alpha_max,
        loc_error_min_nspots=a.loc_error_min_nspots,
        confinement_threshold_um=a.confinement_threshold_um,
        track_length_bins=a.track_length_bins, step_bin_width_um=a.step_bin_width_um,
        diffusion_bin_width=a.diffusion_bin_width, logd_bin_width=a.logd_bin_width,
        skip_gmm=not a.gmm, max_gmm_k=a.max_gmm_k,
        msd_max_lag=a.msd_max_lag, msd_min_tracks=a.msd_min_tracks,
        advanced=adv_on, advanced_select=adv_sel,
        advanced_min_track_len=a.advanced_min_track_len,
        advanced_max_tlag=a.advanced_max_tlag,
        advanced_particle_radius_um=(a.advanced_particle_radius
                                     if a.advanced_particle_radius is not None
                                     else DEFAULT_PARTICLE_RADIUS_UM),
        advanced_particle_radius_explicit=a.advanced_particle_radius is not None,
        advanced_temperature_k=a.advanced_temperature,
        advanced_n_jobs=a.advanced_n_jobs,
        marker_intensity_bins=a.marker_intensity_bins,
        marker_min_disk_coverage=a.marker_min_disk_coverage,
        marker_edge_margin_px=a.marker_edge_margin_px,
        hist_yscale=a.hist_yscale,
        density_strata=a.density_strata,
        density_match=a.density_match,
        density_match_column=a.density_match_column,
        density_match_length_bins=a.density_match_length_bins,
        density_match_seed=a.density_match_seed,
        brightness_bin_edges=a.brightness_bin_edges,
        brightness_bins=a.brightness_bins,
        track_filter=filters.TrackFilter.from_args(a),
        run_label=a.run_label,
        arms=compare.resolve_arms(a.run_only),
        excluded_movies=skipped)
    root = compare.compare_conditions(df, out_dir, cfg)
    print(f"Comparison complete. Outputs under: {root}")
    rl.finish(n_processed=1)
    return 0


def _marker_diameter(manifest_df, explicit):
    """Sampling disk diameter: explicit, else the tracking `--diam`, else 7."""
    if explicit:
        return int(explicit)
    from .compare import _tracking_context
    ctx = _tracking_context(manifest_df)
    return int(ctx.get("diameter") or 7)


def _handle_marker_intensity(a) -> int:
    import pandas as pd

    from . import manifest, marker
    from .run_log import RunLog
    mpath = Path(a.manifest)
    df = manifest.load(mpath, path_base=_data_root(a))
    out_dir = Path(a.out_dir) if a.out_dir else mpath.parent
    rl = RunLog("analyze marker-intensity", vars(a), out_dir, mpath)

    markers = manifest.marker_rows(df)
    if markers.empty:
        print("No marker rows in the manifest (nothing has 'marker' in notes); nothing to do.")
        rl.finish(n_processed=0, n_skipped=0)
        return 0

    diameter = _marker_diameter(df, a.diameter)
    print(f"Marker intensity: {len(markers)} marker image(s), sampling disk "
          f"diameter = {diameter} px "
          f"({'explicit' if a.diameter else 'inherited from tracking'})")

    work = manifest.rows_with_tracks(manifest.movie_rows(df))
    if not a.force:
        keep = []
        for _, row in work.iterrows():
            tc = Path(str(row["tracks_csv"]))
            done = tc.with_name(f"{row['movie_id']}_marker_intensity.csv")
            if done.exists():
                continue
            keep.append(row)
        n_cached = len(work) - len(keep)
        if n_cached:
            print(f"  {n_cached} movie(s) already measured; use --force to redo them.")
        work = pd.DataFrame(keep, columns=work.columns) if keep else work.iloc[0:0]

    # pair_markers() derives its marker pool from the frame it is handed, but
    # movie_rows()/rows_with_tracks() above strip the marker rows out. Add them
    # back so pairing can find them; marker rows are not trackable, so they are
    # used only for pairing and never measured.
    to_measure = pd.concat([work, markers], ignore_index=True)
    results, report = marker.measure_all(
        to_measure, diameter=diameter, marker_channel=a.marker_channel,
        append=not a.no_append, manifest_path=mpath)

    rpath = out_dir / "marker_intensity_report.csv"
    rpath.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(rpath, index=False)

    n_ok = sum(1 for r in results if r.status == "ok")
    n_un = sum(1 for r in results if r.status == "unmatched")
    n_sk = sum(1 for r in results if r.status == "skipped")
    n_fa = sum(1 for r in results if r.status == "failed")
    n_tracks = sum(r.n_tracks for r in results)
    print(f"Measured {n_tracks} track(s) across {n_ok} movie(s); "
          f"{n_un} unmatched, {n_sk} skipped, {n_fa} failed.")
    methods = {}
    for r in results:
        if r.status == "ok":
            methods[r.method] = methods.get(r.method, 0) + 1
    for m, n in sorted(methods.items()):
        print(f"  paired by {m}: {n}")
    for r in results:
        if r.status in ("unmatched", "failed"):
            print(f"  [{r.status}] {r.movie_id}: {r.error}")
            rl.note(f"{r.movie_id}: {r.status}: {r.error}")
    print(f"Pairing report: {rpath}")
    rl.finish(n_processed=n_ok, n_skipped=n_sk + n_un,
              extra={"n_failed": n_fa, "n_tracks": n_tracks, "diameter": diameter})
    return 0 if n_fa == 0 else 1


def _handle_density(a) -> int:
    import pandas as pd

    from . import density as density_mod
    from . import density_runner, manifest
    from .run_log import RunLog
    mpath = Path(a.manifest)
    df = manifest.load(mpath, path_base=_data_root(a))
    out_dir = Path(a.out_dir) if a.out_dir else mpath.parent
    rl = RunLog("analyze density", vars(a), out_dir, mpath)

    work = manifest.rows_with_tracks(manifest.movie_rows(df))
    if work.empty:
        print("No movies with tracks; nothing to measure.")
        rl.finish(n_processed=0, n_skipped=0)
        return 0
    radius = a.field_radius_px or density_mod.DEFAULT_FIELD_RADIUS_PX
    print(f"Density: {len(work)} movie(s), field radius {radius:g} px")

    res = density_runner.measure_manifest(
        work, field_radius_px=radius, default_mpp=0.183,
        default_search_range_px=(a.search_range_px or 5.0),
        segment=not a.no_segment, force=a.force)

    rep = pd.DataFrame(res["results"])
    rpath = out_dir / "density_report.csv"
    rpath.parent.mkdir(parents=True, exist_ok=True)
    if not rep.empty:
        rep.to_csv(rpath, index=False)

    print(f"\nMeasured {res['n_pre_link']} movie(s) from pre-link detections, "
          f"{res['n_post_link']} from linked tracks, {res['n_cached']} cached, "
          f"{res['n_failed']} failed.")
    if res["n_post_link"]:
        print("  Post-link density understates crowding: linking and stub filtering "
              "already removed detections, most of them in the crowded fields. "
              "Re-track to measure it properly; every row is flagged either way.")
    if not rep.empty and "n_cells" in rep.columns:
        ok = rep[rep["status"] == "ok"]
        if len(ok):
            multi = int((ok["n_cells"] > 1).sum())
            print(f"  Cells: {int(ok['n_cells'].sum())} total; {multi} field(s) "
                  f"resolved into more than one cell.")
            conf = ok[ok["segmentation_note"].astype(str).str.contains("confluent")]
            if len(conf):
                print(f"  {len(conf)} field(s) were confluent or touching and are "
                      f"treated as a single density field.")
            safe = pd.to_numeric(ok.get("median_link_safety_ratio"), errors="coerce")
            if safe.notna().any():
                low = int((safe < density_mod.LINK_SAFETY_WARN).sum())
                print(f"  Median link safety across movies: {safe.median():.1f}"
                      + (f"; {low} movie(s) below {density_mod.LINK_SAFETY_WARN:g}, "
                         f"where neighbours sit within a few typical steps and some "
                         f"links are guesses." if low else ""))
    print(f"Report: {rpath}")
    rl.finish(n_processed=res["n_pre_link"] + res["n_post_link"],
              n_skipped=res["n_cached"], extra={"n_failed": res["n_failed"]})
    return 0 if res["n_failed"] == 0 else 1


def _handle_workflow_auto(a) -> int:
    """The whole core path in one command: scan -> track (+tau, +QC) -> compare.

    Deliberately halts after the scan when any condition/experiment is still
    ``FILL_IN``. Tracking is the expensive step, and tracking a mislabelled
    dataset produces figures that look fine and mean nothing.
    """
    from . import manifest, tracking
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = Path(a.manifest) if a.manifest else out_dir / "manifest.csv"
    base = _data_root(a)

    # ---- 1/3  scan ----
    if mpath.exists() and not a.force:
        print(f"[workflow] 1/3 prep scan: using existing manifest {mpath}")
    else:
        print(f"[workflow] 1/3 prep scan: {len(a.input_dirs)} directory(ies)")
        scan_args = argparse.Namespace(
            input_dirs=a.input_dirs, output=str(mpath), mpp=a.mpp, fps=a.fps,
            exclude_pattern=a.exclude_pattern, condition_tokens=a.condition_tokens,
            from_tracks=False, data_root=a.data_root)
        rc = _handle_prep_scan(scan_args)
        if rc != 0:
            return rc

    df = manifest.load(mpath, path_base=base)
    unresolved = int(
        df["condition"].astype(str).str.strip().eq(manifest.FILL).sum()
        + df["experiment"].astype(str).str.strip().eq(manifest.FILL).sum()
    )
    if unresolved:
        print(f"\n[workflow] STOP: {unresolved} FILL_IN value(s) in condition/experiment.")
        print(f"  Edit {mpath}, then re-run the same command to continue.")
        return 2

    # ---- 2/3  track ----
    print(f"[workflow] 2/3 track trackpy ({len(df)} movies)")
    track_args = argparse.Namespace(
        manifest=str(mpath), out_dir=str(out_dir), mpp=a.mpp, fps=a.fps, diam=a.diam,
        minmass=a.minmass, search_range=a.search_range, memory=a.memory,
        min_separation=a.min_separation, percentile=a.percentile,
        min_track_length=a.min_track_length, diffusion_tau=a.diffusion_tau,
        channel=a.channel, tau=a.tau,
        no_tau_steps=False, no_qc_plots=a.no_qc_plots, n_jobs=a.n_jobs, force=a.force,
        no_density=False, density_field_radius_px=None, no_save_detections=False,
        qc_detections_overlay_frames=a.qc_detections_overlay_frames,
        data_root=a.data_root)
    rc = _handle_track_trackpy(track_args)
    if rc != 0:
        print("[workflow] tracking reported failures; continuing with the movies that "
              "did track (re-run with --force after fixing the failures).")

    if a.skip_compare:
        print("[workflow] Stopping before compare-conditions (--skip-compare).")
        return 0

    # ---- 3/3  compare ----
    print("[workflow] 3/3 analyze compare-conditions")
    cmp_args = argparse.Namespace(
        manifest=str(mpath), out_dir=str(out_dir), tau=None, mpp=a.mpp, fps=a.fps,
        min_nspots=0, min_alpha=None, subtract_loc_error=a.subtract_loc_error,
        loc_error_alpha_max=0.3, loc_error_min_nspots=10,
        confinement_threshold_um=a.confinement_threshold_um, track_length_bins=None,
        step_bin_width_um="auto", diffusion_bin_width="auto", logd_bin_width="auto",
        gmm=a.gmm, max_gmm_k=5, msd_max_lag=None, msd_min_tracks=20,
        marker_intensity_bins=4, marker_min_disk_coverage=0.0, marker_edge_margin_px=0.0,
        brightness_bin_edges=None, brightness_bins=None, run_label="", run_only=None,
        hist_yscale="linear", density_strata=4, density_match=False,
        density_match_column="track_median_field_localizations_n",
        density_match_length_bins=None, density_match_seed=1729,
        filter_drop_missing=False,
        **{f"filter_{k}": None for k in _filter_names()},
        advanced=a.advanced, advanced_only=a.advanced_only,
        advanced_min_track_len=a.advanced_min_track_len,
        advanced_max_tlag=a.advanced_max_tlag,
        advanced_particle_radius=a.advanced_particle_radius,
        advanced_temperature=a.advanced_temperature,
        advanced_n_jobs=a.advanced_n_jobs, data_root=a.data_root)
    rc = _handle_compare_conditions(cmp_args)
    print("\n[workflow] Completed: prep scan -> track trackpy -> analyze compare-conditions")
    return rc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Delegate peripheral subcommands to the legacy dispatcher untouched.
    if argv and not _is_new_route(argv) and argv[0] not in ("--version", "-h", "--help"):
        if argv[0] in ("prep", "track", "analyze", "draw", "utils"):
            from . import legacy
            return legacy.main(argv) or 0

    parser = build_parser()
    a = parser.parse_args(argv)

    if a.command == "prep" and getattr(a, "prep_cmd", None) == "scan":
        return _handle_prep_scan(a)
    if a.command == "track" and getattr(a, "track_cmd", None) == "trackpy":
        return _handle_track_trackpy(a)
    if a.command == "compute" and getattr(a, "compute_cmd", None) == "tau-steps":
        return _handle_compute_tau_steps(a)
    if a.command == "analyze" and getattr(a, "analyze_cmd", None) == "compare-conditions":
        return _handle_compare_conditions(a)
    if a.command == "analyze" and getattr(a, "analyze_cmd", None) == "marker-intensity":
        return _handle_marker_intensity(a)
    if a.command == "analyze" and getattr(a, "analyze_cmd", None) == "density":
        return _handle_density(a)
    if a.command == "workflow" and getattr(a, "workflow_cmd", None) == "auto":
        return _handle_workflow_auto(a)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
