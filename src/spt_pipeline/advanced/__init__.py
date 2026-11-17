"""Advanced trajectory analyses, native to the pipeline.

Six quantities that go beyond step sizes and D:

===========  ============================================================
``vacf``     Velocity autocorrelation, multi-δ
``cfbm``     Fractional-Brownian-motion fit to the collapsed VACF (α′, H)
``angles``   Turning-angle statistics, ⟨cos θ⟩ vs τ
``eb``       Ergodicity-breaking parameter, Var(MSD)/⟨MSD⟩²
``ngp``      Non-Gaussian parameter α₂
``rheology`` Complex shear modulus G′/G″ via Mason/GSER
===========  ============================================================

The numerics live in :mod:`._kernels`, adapted from the standalone GEM library.
Everything else is native: conditions and experiments come from the manifest,
tracks come from the paths the manifest already records, pixel size comes from
each movie's own ``mpp``, and figures use the pipeline's house style and
condition colours. The original script's parallel universe -- filename-based
condition inference, directory-layout replicate pooling, its own CSV discovery
and its own ``run_all``/``run_multi_condition`` drivers -- is gone, because a
second source of truth about what a movie is was the thing most likely to make
two figures in the same folder disagree.

Selection keywords are case-insensitive and aliased, so ``--advanced-only VACF
a2`` and ``--advanced-only vacf non-gaussian`` mean the same thing.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .. import plotting, utils
from . import _kernels as K

#: Canonical analysis names.
ADVANCED_KEYWORDS: FrozenSet[str] = frozenset(
    {"vacf", "cfbm", "angles", "eb", "ngp", "rheology"})

#: Accepted spellings -> canonical name. Aliases exist because these quantities
#: genuinely go by several names (α₂ / NGP, EB / ergodicity, G′G″ /
#: microrheology) and forcing one spelling just causes typos.
_ALIASES: Dict[str, str] = {
    "vacf": "vacf", "velocity": "vacf", "velocity_autocorrelation": "vacf",
    "cfbm": "cfbm", "fbm": "cfbm", "collapse": "cfbm", "hurst": "cfbm",
    "angles": "angles", "angle": "angles", "cos_theta": "angles",
    "costheta": "angles", "turning": "angles", "turning_angles": "angles",
    "eb": "eb", "ergodicity": "eb", "ergodicity_breaking": "eb",
    "ngp": "ngp", "a2": "ngp", "alpha2": "ngp", "alpha_2": "ngp",
    "nongaussian": "ngp", "non_gaussian": "ngp",
    "rheology": "rheology", "rheo": "rheology", "gmodulus": "rheology",
    "g_modulus": "rheology", "microrheology": "rheology", "gser": "rheology",
    "all": "__all__",
}

#: Default probe radius (microns). Reported in nm when the user does not set it,
#: because G′/G″ scale with it and a silently-assumed radius is a silently-wrong
#: modulus.
DEFAULT_PARTICLE_RADIUS_UM = 0.020


class UnknownAnalysisError(ValueError):
    """Raised for an unrecognized ``--advanced-only`` keyword."""


def resolve_selection(names: Optional[Iterable[str]]) -> FrozenSet[str]:
    """Map user keywords to canonical analysis names.

    ``None`` (or ``all``) selects everything. An unrecognized keyword raises
    rather than being dropped: quietly producing five plots when six were
    requested is the kind of failure nobody notices until the figure is in a
    draft.
    """
    if names is None:
        return frozenset(ADVANCED_KEYWORDS)
    out, unknown = set(), []
    for raw in names:
        key = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
        canon = _ALIASES.get(key)
        if canon == "__all__":
            return frozenset(ADVANCED_KEYWORDS)
        if canon is None:
            unknown.append(raw)
        else:
            out.add(canon)
    if unknown:
        raise UnknownAnalysisError(
            f"Unknown advanced analysis name(s): {', '.join(map(str, unknown))}. "
            f"Choices (aliases accepted): {', '.join(sorted(ADVANCED_KEYWORDS))}."
        )
    return frozenset(out or ADVANCED_KEYWORDS)


def safe_name(name: str) -> str:
    """Filesystem-safe label."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


# ---------------------------------------------------------------------------
# Manifest -> track arrays
# ---------------------------------------------------------------------------

def _movie_track_array(tracks_csv: Path, mpp: float, offset: int) -> Optional[np.ndarray]:
    """One movie's tracks as ``[track_id, frame, x_um, y_um]``.

    ``offset`` shifts particle ids into a globally unique range so pooling
    movies cannot merge two different particles that share a trackpy id.
    """
    try:
        df = pd.read_csv(tracks_csv, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    df = utils.normalize_track_table(df)
    if df.empty or not {"particle", "frame", "x", "y"}.issubset(df.columns):
        return None
    df = df.dropna(subset=["particle", "frame", "x", "y"])
    if df.empty:
        return None
    arr = np.empty((len(df), 4), dtype=float)
    arr[:, 0] = df["particle"].to_numpy(dtype=float) + float(offset)
    arr[:, 1] = df["frame"].to_numpy(dtype=float)
    arr[:, 2] = df["x"].to_numpy(dtype=float) * float(mpp)   # px -> um, per movie
    arr[:, 3] = df["y"].to_numpy(dtype=float) * float(mpp)
    return arr


def build_condition_tracks(manifest_df: pd.DataFrame, default_mpp: float) -> Dict[str, dict]:
    """Pool per-movie tracks into micron-unit arrays, per condition.

    Also keeps each movie's array separately under ``by_experiment`` so
    replicate-level aggregation uses the manifest's ``experiment`` column rather
    than inferring replicates from where files happen to sit.
    """
    from .. import manifest as manifest_mod
    out: Dict[str, dict] = {}
    offset = 0
    for _, row in manifest_df.iterrows():
        if not manifest_mod.has_tracks(row):
            continue
        mpp, _fps = manifest_mod.row_mpp_fps(row, default_mpp, 1.0)
        arr = _movie_track_array(Path(str(row["tracks_csv"])), mpp, offset)
        if arr is None or arr.size == 0:
            continue
        offset = int(arr[:, 0].max()) + 1
        cond, exp = str(row["condition"]), str(row.get("experiment", ""))
        slot = out.setdefault(cond, {"chunks": [], "movies": [], "by_experiment": {}})
        slot["chunks"].append(arr)
        slot["movies"].append(str(row["movie_id"]))
        slot["by_experiment"].setdefault(exp, []).append(arr)
    for slot in out.values():
        slot["tracks"] = np.vstack(slot.pop("chunks"))
        slot["by_experiment"] = {e: np.vstack(v) for e, v in slot["by_experiment"].items()}
    return out


# ---------------------------------------------------------------------------
# Per-condition analysis
# ---------------------------------------------------------------------------

def analyse_tracks(
    track_data: np.ndarray, *, time_step: float, select: FrozenSet[str],
    min_track_len: int, max_tlag: int, deltas: Sequence[int],
    particle_radius_um: float, temperature_k: float, n_jobs: int,
) -> Dict[str, Any]:
    """Run the selected analyses on one pooled track array.

    Dependencies are honoured: asking for ``cfbm`` computes the VACF it is
    fitted to. Nothing is computed that was not asked for.
    """
    need_vacf = bool(select & {"vacf", "cfbm"})
    sorted_td, tindex, tl = K._build_track_index(track_data)

    res: Dict[str, Any] = {"n_tracks": int(len(tl)), "track_lengths": tl}
    if need_vacf:
        res["vacf"] = K.vacf_multi_delta(sorted_td, tl, min_track_len, deltas,
                                         max_tlag=max_tlag, n_jobs=n_jobs,
                                         _track_index=tindex)
        res["cfbm"] = K.fit_cfbm(res["vacf"])
    if "angles" in select:
        angles = K.compute_angles(sorted_td, tl, min_track_len, max_tlag,
                                  _track_index=tindex)
        res["angles"] = angles
        res["cos_theta"] = K.cos_theta_vs_tau(angles, time_step)
        res["angle_ks"] = K.angle_ks_tests_consecutive(angles)
    if "eb" in select:
        msd_pt = K.compute_msd_per_track(sorted_td, tl, min_track_len, max_tlag,
                                         _track_index=tindex)
        res["eb"] = K.ergodicity_breaking_parameter(msd_pt, time_step)
    if "ngp" in select:
        dx, dy = K.compute_displacements(sorted_td, tl, min_track_len, max_tlag,
                                         _track_index=tindex)
        res["ngp"] = K.non_gaussian_parameter(dx, dy)
    if "rheology" in select:
        msd_ens = K.compute_ensemble_msd(sorted_td, tl, min_track_len, max_tlag,
                                         _track_index=tindex)
        res["ensemble_msd"] = msd_ens
        res["rheology"] = K.compute_complex_modulus(
            msd_ens, time_step, particle_radius_um, temperature_k)
    return res


def _write_condition_csvs(res: Dict[str, Any], out_dir: Path, label: str,
                          select: FrozenSet[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = f"{safe_name(label)}_"
    if "vacf" in res:
        for delta, (_, avg) in res["vacf"].items():
            if len(avg):
                pd.DataFrame(avg, columns=["tau", "mean_cx", "std_cx", "sem_cx",
                                           "mean_cy", "std_cy", "sem_cy"]
                             ).to_csv(out_dir / f"{pre}vacf_delta{delta}.csv", index=False)
        pd.Series(res["cfbm"]).to_frame("value").to_csv(out_dir / f"{pre}cfbm_fit.csv")
    for key, fname in (("cos_theta", "cos_theta"), ("angle_ks", "angle_ks_tests"),
                       ("eb", "ergodicity"), ("ngp", "ngp"),
                       ("ensemble_msd", "ensemble_msd"), ("rheology", "rheology")):
        df = res.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(out_dir / f"{pre}{fname}.csv", index=False)


# ---------------------------------------------------------------------------
# Between-condition figures (house style, minimal annotation)
# ---------------------------------------------------------------------------

def _overlay(results: Dict[str, dict], out_png: Path, *, xy, xlabel: str, ylabel: str,
             yerr=None, hline: Optional[float] = None, loglog: bool = False,
             marker: str = "o-") -> bool:
    """Generic per-condition overlay. ``xy(res)`` returns ``(x, y)`` or None."""
    import matplotlib.pyplot as plt
    conds = sorted(results)
    colors = plotting.condition_colors(conds)
    drew = False
    with plotting._PLOT_LOCK:
        plotting.apply_house_style()
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        for cond in conds:
            got = xy(results[cond])
            if got is None:
                continue
            x, y = got
            if x is None or len(x) == 0:
                continue
            e = yerr(results[cond]) if yerr else None
            if e is not None and np.isfinite(np.asarray(e, float)).any():
                ax.errorbar(x, y, yerr=e, fmt=marker, capsize=2.5, markersize=4,
                            color=colors[cond], label=cond)
            else:
                ax.plot(x, y, marker, markersize=4, color=colors[cond], label=cond)
            drew = True
        if not drew:
            plt.close(fig)
            return False
        if hline is not None:
            ax.axhline(hline, color="0.6", linewidth=0.9, zorder=0)
        if loglog:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        plotting._save(fig, out_png)
    return True


def _plot_vacf(results, out_png) -> bool:
    def xy(r):
        v = r.get("vacf") or {}
        if 1 not in v:
            return None
        _, avg = v[1]
        if not len(avg):
            return None
        return avg[:, 0] * r["_time_step"], (avg[:, 1] + avg[:, 4]) / 2.0
    return _overlay(results, out_png, xy=xy, xlabel="τ (s)",
                    ylabel="Normalized VACF", hline=0.0, marker="-")


def _plot_cfbm(results, out_png) -> bool:
    """VACF collapse: all δ curves on the scaling variable ξ = τ/δ."""
    import matplotlib.pyplot as plt
    conds = sorted(c for c in results if results[c].get("vacf"))
    if not conds:
        return False
    colors = plotting.condition_colors(conds)
    with plotting._PLOT_LOCK:
        plotting.apply_house_style()
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        drew = False
        for cond in conds:
            r = results[cond]
            for delta, (_, avg) in sorted(r["vacf"].items()):
                if not len(avg):
                    continue
                xi = avg[:, 0] / float(delta)
                cv = (avg[:, 1] + avg[:, 4]) / 2.0
                ax.plot(xi, cv, "-", linewidth=1.0, alpha=0.55, color=colors[cond],
                        label=cond if delta == min(r["vacf"]) else None)
                drew = True
        if not drew:
            plt.close(fig); return False
        ax.axhline(0.0, color="0.6", linewidth=0.9, zorder=0)
        ax.set_xlabel("ξ = τ / δ"); ax.set_ylabel("Normalized VACF")
        ax.set_xlim(0, 3)
        ax.legend(loc="best")
        plotting._save(fig, out_png)
    return True


def _plot_hurst(results, out_png) -> bool:
    """Hurst exponent per condition as a simple categorical panel."""
    import matplotlib.pyplot as plt
    rows = [(c, r["cfbm"].get("H"), r["cfbm"].get("a_err"))
            for c, r in sorted(results.items()) if r.get("cfbm")]
    rows = [(c, h, e) for c, h, e in rows if h is not None and np.isfinite(h)]
    if not rows:
        return False
    colors = plotting.condition_colors([c for c, _, _ in rows])
    with plotting._PLOT_LOCK:
        plotting.apply_house_style()
        fig, ax = plt.subplots(figsize=(1.5 * len(rows) + 2.0, 4.0))
        xs = np.arange(len(rows))
        ax.bar(xs, [h for _, h, _ in rows],
               yerr=[e if e and np.isfinite(e) else 0.0 for _, _, e in rows],
               color=[colors[c] for c, _, _ in rows], width=0.6, capsize=3)
        ax.axhline(0.5, color="0.6", linewidth=0.9, zorder=0)
        ax.set_xticks(xs); ax.set_xticklabels([c for c, _, _ in rows])
        ax.set_ylabel("Hurst exponent H")
        ax.grid(axis="x", visible=False)
        plotting._save(fig, out_png)
    return True


_OVERLAYS = {
    "vacf": ("comparison_vacf.png", _plot_vacf),
    "cfbm": ("comparison_cfbm_collapse.png", _plot_cfbm),
}


def _plot_simple_overlays(results: Dict[str, dict], out_dir: Path,
                          select: FrozenSet[str]) -> List[str]:
    made = []
    if "vacf" in select and _plot_vacf(results, out_dir / "comparison_vacf.png"):
        made.append("comparison_vacf.png")
    if "cfbm" in select:
        if _plot_cfbm(results, out_dir / "comparison_cfbm_collapse.png"):
            made.append("comparison_cfbm_collapse.png")
        if _plot_hurst(results, out_dir / "comparison_hurst.png"):
            made.append("comparison_hurst.png")
    if "angles" in select and _overlay(
            results, out_dir / "comparison_cos_theta.png",
            xy=lambda r: ((r["cos_theta"]["tau_s"], r["cos_theta"]["mean_cos"])
                          if isinstance(r.get("cos_theta"), pd.DataFrame)
                          and not r["cos_theta"].empty else None),
            yerr=lambda r: (r["cos_theta"]["sem_cos"]
                            if isinstance(r.get("cos_theta"), pd.DataFrame)
                            and not r["cos_theta"].empty else None),
            xlabel="τ (s)", ylabel="⟨cos θ⟩", hline=0.0):
        made.append("comparison_cos_theta.png")
    if "eb" in select and _overlay(
            results, out_dir / "comparison_ergodicity.png",
            xy=lambda r: ((r["eb"]["tau_s"], r["eb"]["eb"])
                          if isinstance(r.get("eb"), pd.DataFrame)
                          and not r["eb"].empty else None),
            xlabel="τ (s)", ylabel="EB parameter"):
        made.append("comparison_ergodicity.png")
    if "ngp" in select and _overlay(
            results, out_dir / "comparison_ngp.png",
            xy=lambda r: ((r["ngp"]["tlag"] * r["_time_step"], r["ngp"]["ngp"])
                          if isinstance(r.get("ngp"), pd.DataFrame)
                          and not r["ngp"].empty else None),
            xlabel="τ (s)", ylabel="α₂", hline=0.0):
        made.append("comparison_ngp.png")
    if "rheology" in select:
        for col, fname, lab in (("G_prime_Pa", "comparison_G_prime.png", "G′ (Pa)"),
                                ("G_double_prime_Pa", "comparison_G_double_prime.png", "G″ (Pa)")):
            if _overlay(results, out_dir / fname,
                        xy=(lambda c: lambda r: ((r["rheology"]["omega"], r["rheology"][c])
                                                 if isinstance(r.get("rheology"), pd.DataFrame)
                                                 and not r["rheology"].empty else None))(col),
                        xlabel="ω (rad/s)", ylabel=lab, loglog=True):
                made.append(fname)
    return made


def _comparison_tables(results: Dict[str, dict], out_dir: Path,
                       select: FrozenSet[str]) -> None:
    """Per-condition summary plus pairwise angle KS tests."""
    rows = []
    for cond, r in sorted(results.items()):
        tl = r.get("track_lengths")
        row = {"condition": cond, "n_tracks": r.get("n_tracks"),
               "median_track_len": float(np.median(tl[:, 1])) if tl is not None and len(tl) else np.nan}
        if r.get("cfbm"):
            row.update({"cfbm_a": r["cfbm"].get("a"), "cfbm_a_err": r["cfbm"].get("a_err"),
                        "cfbm_H": r["cfbm"].get("H")})
        for key, col, out_name in (("cos_theta", "mean_cos", "cos_theta_tlag1"),
                                   ("eb", "eb", "eb_tlag1"), ("ngp", "ngp", "ngp_tlag1")):
            df = r.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty and col in df.columns:
                row[out_name] = float(df.iloc[0][col])
        rheo = r.get("rheology")
        if isinstance(rheo, pd.DataFrame) and not rheo.empty:
            row["G_prime_median_Pa"] = float(np.median(rheo["G_prime_Pa"]))
            row["G_double_prime_median_Pa"] = float(np.median(rheo["G_double_prime_Pa"]))
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "comparison_summary.csv", index=False)

    if "angles" in select:
        angles_by_cond = {c: r["angles"] for c, r in results.items() if r.get("angles")}
        if len(angles_by_cond) >= 2:
            ks = K.angle_ks_tests_between_conditions(angles_by_cond)
            if isinstance(ks, pd.DataFrame) and not ks.empty:
                ks.to_csv(out_dir / "comparison_angle_ks.csv", index=False)

    for key, fname in (("eb", "comparison_eb.csv"), ("ngp", "comparison_ngp.csv"),
                       ("rheology", "comparison_rheology.csv")):
        frames = []
        for cond, r in sorted(results.items()):
            df = r.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                d = df.copy(); d.insert(0, "condition", cond); frames.append(d)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out_dir / fname, index=False)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_advanced(
    manifest_df: pd.DataFrame,
    out_dir: Path,
    *,
    select: Optional[Iterable[str]] = None,
    mpp: float = 0.183,
    fps: float = 50.0,
    min_track_len: int = 10,
    max_tlag: int = 10,
    particle_radius_um: float = DEFAULT_PARTICLE_RADIUS_UM,
    particle_radius_explicit: bool = True,
    temperature_k: float = 298.15,
    n_jobs: int = 4,
) -> Dict[str, Any]:
    """Run the selected analyses per condition, then compare conditions.

    Writes ``<out_dir>/<condition>/`` per-condition CSVs plus top-level
    ``comparison_*`` tables and overlay figures. Returns a summary for the run
    log.
    """
    sel = resolve_selection(select)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    time_step = 1.0 / float(fps)

    if "rheology" in sel and not particle_radius_explicit:
        # Printed once, before any work: G'/G'' scale with the probe radius, so
        # an unnoticed default is an unnoticed systematic error in the modulus.
        print(f"  [advanced] Using the default probe particle radius of "
              f"{particle_radius_um * 1000:.0f} nm for the G'/G'' calculation. "
              f"Set --advanced-particle-radius (in microns) to override.")

    grouped = build_condition_tracks(manifest_df, mpp)
    if not grouped:
        return {"status": "skipped", "reason": "no movies with usable tracks",
                "analyses": sorted(sel)}

    pd.DataFrame([{"condition": c, "movie_id": m}
                  for c, s in sorted(grouped.items()) for m in s["movies"]]
                 ).to_csv(out_dir / "condition_grouping.csv", index=False)

    deltas = list(range(1, max(2, int(max_tlag)) + 1))
    kw = dict(time_step=time_step, select=sel, min_track_len=min_track_len,
              max_tlag=max_tlag, deltas=deltas, particle_radius_um=particle_radius_um,
              temperature_k=temperature_k, n_jobs=n_jobs)

    results: Dict[str, dict] = {}
    for cond in sorted(grouped):
        slot = grouped[cond]
        n_tracks = int(np.unique(slot["tracks"][:, 0]).size)
        if n_tracks < 2:
            print(f"  [advanced] {cond}: only {n_tracks} track(s); skipped")
            continue
        print(f"  [advanced] {cond}: {n_tracks} tracks from {len(slot['movies'])} movie(s)")
        try:
            r = analyse_tracks(slot["tracks"], **kw)
        except Exception as exc:          # one bad condition must not sink the run
            print(f"  [advanced] {cond}: failed ({exc})")
            continue
        r["_time_step"] = time_step
        results[cond] = r
        _write_condition_csvs(r, out_dir / safe_name(cond), cond, sel)

    summary: Dict[str, Any] = {
        "status": "ok" if results else "skipped",
        "analyses": sorted(sel),
        "time_step_s": time_step,
        "min_track_len": min_track_len,
        "max_tlag_frames": max_tlag,
        "particle_radius_nm": (particle_radius_um * 1000.0) if "rheology" in sel else None,
        "particle_radius_was_default": (not particle_radius_explicit) if "rheology" in sel else None,
        "conditions": {c: {"n_tracks": results[c]["n_tracks"],
                           "n_movies": len(grouped[c]["movies"]),
                           "experiments": sorted(grouped[c]["by_experiment"])}
                       for c in results},
    }

    if len(results) >= 2:
        _comparison_tables(results, out_dir, sel)
        summary["figures"] = _plot_simple_overlays(results, out_dir, sel)
    elif results:
        summary["note"] = "Only one condition had usable tracks; no comparison drawn."

    utils.dump_json(summary, out_dir / "advanced_run_summary.json")
    return summary


__all__ = [
    "ADVANCED_KEYWORDS", "DEFAULT_PARTICLE_RADIUS_UM", "UnknownAnalysisError",
    "resolve_selection", "build_condition_tracks", "analyse_tracks", "run_advanced",
]
