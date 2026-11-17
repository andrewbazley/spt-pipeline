"""FICD ER-GEM CLI:  python -m spt_pipeline.ergem <command>

Commands
--------
from-json   condition_map.json  -> manifest.csv (direct JSON reader)
ingest      manifest            -> per-movie diffusion/MSD products (MOSAIC)
expression  manifest + json     -> per-cell FICD measurement (trajectory_expression)
crosscorr   manifest + expr csv -> D-vs-FICD cross-correlation table
contrasts   manifest            -> factored WT/E234G x DMSO/Tg tests (+clone,+mixed)
run         json                -> from-json -> ingest -> [expression -> crosscorr] -> contrasts

Step-size histograms and the standard comparison figures are produced by the
core pipeline, unchanged:
    spt-pipeline compute tau-steps -m <manifest> -o <out> --mpp 0.11 --fps 100
    spt-pipeline analyze compare-conditions -m <manifest> -o <out> --mpp 0.11 --fps 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import DEFAULT_MPP, DEFAULT_FPS
from . import condition_map as cm


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spt-pipeline-ergem",
                                description="FICD complementation ER-GEM workflow.")
    sub = p.add_subparsers(dest="command")

    fj = sub.add_parser("from-json", help="condition map JSON -> manifest CSV")
    fj.add_argument("json", help="Path to *_condition_map.json")
    fj.add_argument("-o", "--output", default="manifest.csv")
    fj.add_argument("--mpp", type=float, default=DEFAULT_MPP)
    fj.add_argument("--fps", type=float, default=DEFAULT_FPS)
    fj.add_argument("--no-metadata", action="store_true",
                    help="Do not read mpp/fps from ND2/TIF metadata")
    fj.add_argument("--correlation-only", action="store_true",
                    help="Keep only movies with a paired red image")

    ing = sub.add_parser("ingest", help="MOSAIC Traj -> diffusion/MSD products")
    ing.add_argument("-m", "--manifest", required=True)
    ing.add_argument("-o", "--out-dir", required=True)
    ing.add_argument("--mpp", type=float, default=DEFAULT_MPP)
    ing.add_argument("--fps", type=float, default=DEFAULT_FPS)
    ing.add_argument("--tau-ms", type=float, default=100.0,
                     help="D fit lag in milliseconds (default 100 ms; converted via fps)")
    ing.add_argument("--tau-fit", type=int, default=None,
                     help="D fit lag in FRAMES; overrides --tau-ms when set")
    ing.add_argument("--max-lag", type=int, default=25)
    ing.add_argument("--force", action="store_true")
    ing.add_argument("--n-jobs", type=int, default=8, help="parallel workers (default 8)")

    ex = sub.add_parser("expression", help="run per-cell FICD measurement")
    ex.add_argument("-m", "--manifest", required=True)
    ex.add_argument("--condition-map", required=True)
    ex.add_argument("-o", "--out-dir", required=True)
    ex.add_argument("--mask-dir", default=None)
    ex.add_argument("--n-jobs", type=int, default=8, help="experiments run in parallel (default 8)")
    ex.add_argument("ficd_args", nargs=argparse.REMAINDER,
                    help="Extra args passed through to ficd_cell_intensity (after --)")

    cc = sub.add_parser("crosscorr", help="join D to FICD expression")
    cc.add_argument("-m", "--manifest", required=True)
    cc.add_argument("--expression", required=True, help="pooled trajectory_expression CSV")
    cc.add_argument("-o", "--out-dir", required=True)
    cc.add_argument("--d-col", default="D_fit_um2_s")
    cc.add_argument("--n-jobs", type=int, default=8)

    ct = sub.add_parser("contrasts", help="factored WT/E234G x DMSO/Tg tests")
    ct.add_argument("-m", "--manifest", required=True)
    ct.add_argument("-o", "--out-dir", required=True)
    ct.add_argument("--d-col", default="D_fit_um2_s")
    ct.add_argument("--mixed", action="store_true", help="also fit a mixed-effects model")
    ct.add_argument("--n-jobs", type=int, default=8)

    fg = sub.add_parser("figures", help="ER-GEM figures (alpha-vs-logD, D violins, D-vs-FICD)")
    fg.add_argument("-m", "--manifest", required=True)
    fg.add_argument("-o", "--out-dir", required=True)
    fg.add_argument("--d-col", default="D_fit_um2_s")
    fg.add_argument("--tau-ms", type=float, default=100.0)
    fg.add_argument("--crosscorr", default=None,
                    help="D_vs_ficd_expression.csv (enables the D-vs-FICD figure)")
    fg.add_argument("--n-jobs", type=int, default=8)

    rn = sub.add_parser("run", help="end-to-end (no step-size figures)")
    rn.add_argument("--condition-map", required=True)
    rn.add_argument("-o", "--out-dir", required=True)
    rn.add_argument("--mpp", type=float, default=DEFAULT_MPP)
    rn.add_argument("--fps", type=float, default=DEFAULT_FPS)
    rn.add_argument("--tau-ms", type=float, default=100.0)
    rn.add_argument("--tau-fit", type=int, default=None)
    rn.add_argument("--with-expression", action="store_true",
                    help="also run the FICD measurement + cross-correlation")
    rn.add_argument("--mixed", action="store_true")
    rn.add_argument("--force", action="store_true")
    rn.add_argument("--n-jobs", type=int, default=8, help="parallel workers (default 8)")
    return p


def _fit_frames(tau_fit, tau_ms, fps):
    if tau_fit is not None:
        return int(tau_fit)
    return max(1, int(round(float(tau_ms) / 1000.0 * float(fps))))


def _load_manifest(path):
    from .. import manifest as core_manifest
    return core_manifest.load(Path(path))


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    from .. import manifest as core_manifest

    if args.command == "from-json":
        df = cm.from_json(Path(args.json), out_manifest=Path(args.output),
                          default_mpp=args.mpp, default_fps=args.fps,
                          read_metadata=not args.no_metadata,
                          correlation_only=args.correlation_only)
        s = cm.summarize(df)
        print(f"Wrote manifest with {len(df)} movies -> {args.output}")
        if not s.empty:
            print(s.to_string(index=False))
        return 0

    if args.command == "ingest":
        from . import ingest
        df = _load_manifest(args.manifest)
        tau_fr = _fit_frames(args.tau_fit, args.tau_ms, args.fps)
        res = ingest.ingest_all(df, Path(args.out_dir), args.mpp, args.fps,
                                diffusion_tau=tau_fr, max_lagtime=args.max_lag,
                                force=args.force, manifest_path=Path(args.manifest),
                                n_jobs=args.n_jobs)
        ok = sum(1 for r in res if r["status"] == "ok")
        sk = sum(1 for r in res if r["status"] == "skipped")
        fa = sum(1 for r in res if r["status"] == "failed")
        print(f"Ingest (D at {tau_fr} frames = {tau_fr/args.fps*1000:.0f} ms): "
              f"{ok} ingested, {sk} skipped, {fa} failed. Manifest updated: {args.manifest}")
        for r in res:
            if r["status"] == "failed":
                print(f"  FAILED {r['movie_id']}: {r.get('error')}")
        return 0 if fa == 0 else 1

    if args.command == "expression":
        from . import expression
        df = _load_manifest(args.manifest)
        cmap = cm.load_condition_map(Path(args.condition_map))
        extra = [a for a in (args.ficd_args or []) if a != "--"]
        out = expression.run(df, cmap, Path(args.out_dir),
                             mask_dir=Path(args.mask_dir) if args.mask_dir else None,
                             extra_args=extra, n_jobs=args.n_jobs)
        print(f"FICD expression over {out['n_experiments']} experiment(s) -> {out['pooled_expression_csv']}")
        return 0

    if args.command == "crosscorr":
        from . import crosscorr
        df = _load_manifest(args.manifest)
        out = crosscorr.run(df, Path(args.expression), Path(args.out_dir), d_col=args.d_col, n_jobs=args.n_jobs)
        print(f"Cross-correlation tables -> {out}")
        return 0

    if args.command == "contrasts":
        from . import contrasts
        df = _load_manifest(args.manifest)
        out = contrasts.run(df, Path(args.out_dir), d_col=args.d_col, do_mixed=args.mixed, n_jobs=args.n_jobs)
        print(f"Contrasts -> {out}")
        return 0

    if args.command == "figures":
        from . import figures
        df = _load_manifest(args.manifest)
        out = figures.run(df, Path(args.out_dir), d_col=args.d_col, tau_ms=args.tau_ms,
                          crosscorr_csv=Path(args.crosscorr) if args.crosscorr else None, n_jobs=args.n_jobs)
        print(f"Figures -> {out}")
        return 0

    if args.command == "run":
        from . import ingest, contrasts, figures
        nj = args.n_jobs
        total = 5 if args.with_expression else 4
        out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.csv"
        df = cm.from_json(Path(args.condition_map), out_manifest=manifest_path,
                          default_mpp=args.mpp, default_fps=args.fps)
        tau_fr = _fit_frames(args.tau_fit, args.tau_ms, args.fps)
        print(f"[1/{total}] manifest: {len(df)} movies -> {manifest_path}")
        print(f"[2/{total}] ingest: {len(df)} movies on {nj} workers "
              f"(D at {tau_fr} frames = {tau_fr/args.fps*1000:.0f} ms)")
        ingest.ingest_all(df, out_dir, args.mpp, args.fps, diffusion_tau=tau_fr,
                          force=args.force, manifest_path=manifest_path, n_jobs=nj)
        df = core_manifest.load(manifest_path)
        step = 3
        if args.with_expression:
            from . import expression, crosscorr
            print(f"[{step}/{total}] FICD expression + cross-correlation")
            cmap = cm.load_condition_map(Path(args.condition_map))
            ex = expression.run(df, cmap, out_dir, n_jobs=nj)
            # Always run cross-correlation so cross_correlation/ is populated
            # (it writes the CSVs even when empty and is a no-op-safe pass).
            ccdir = crosscorr.run(df, Path(ex["pooled_expression_csv"]), out_dir, n_jobs=nj)
            try:
                import pandas as _pd
                n_ct = len(_pd.read_csv(ccdir / "D_vs_ficd_expression.csv"))
                n_cell = len(_pd.read_csv(ccdir / "Dcell_vs_ficd_expression.csv"))
            except Exception:
                n_ct = n_cell = 0
            print(f"      expression rows: {ex.get('n_rows', 0)} | correlated tracks: {n_ct} | cells: {n_cell}")
            if ex.get("n_rows", 0) == 0:
                print("      NOTE: expression produced 0 per-trajectory rows, so the "
                      "correlation tables are empty and no D-vs-FICD plot is drawn.\n"
                      f"      Diagnose via {out_dir}/ficd_expression/ficd_run_log.csv "
                      "(returncodes) and trajectory_expression_pooled.csv (row count).")
            step += 1
        print(f"[{step}/{total}] contrasts (factored + clone-resolved"
              f"{' + mixed-effects' if args.mixed else ''})")
        contrasts.run(df, out_dir, do_mixed=args.mixed, n_jobs=nj)
        step += 1
        print(f"[{step}/{total}] figures")
        cc = out_dir / "cross_correlation" / "D_vs_ficd_expression.csv"
        figures.run(df, out_dir, d_col="D_fit_um2_s", tau_ms=args.tau_ms,
                    crosscorr_csv=cc if cc.exists() else None, n_jobs=nj)
        print("\nFor step-size histograms and comparison figures, run the core pipeline:\n"
              f'  python -m spt_pipeline.cli compute tau-steps -m "{manifest_path}" -o "{out_dir}" --mpp {args.mpp} --fps {args.fps}\n'
              f'  python -m spt_pipeline.cli analyze compare-conditions -m "{manifest_path}" -o "{out_dir}" --mpp {args.mpp} --fps {args.fps}')
        return 0

    build_parser().print_help()
    return 1
