"""Retired analysis arm: brightness-mobility, step-brightness, and dwell time.

These commands were removed from ``spt-pipeline`` because they form an analysis
arm that is no longer in use. They are preserved here verbatim -- not rewritten,
not made independently runnable -- so nothing is lost and they can be revived,
ported, or mined for reference later.

Retired from ``spt_pipeline.legacy``:

* ``cmd_workflow_auto``               (``workflow legacy-auto``)
* ``cmd_track_dwelltime``             (``track dwelltime``)
* ``cmd_analyze_brightness_mobility`` (``analyze brightness-mobility``)
* ``cmd_plot_steps_brightness``       (``plot steps-brightness``)
* ``cmd_plot_dwell``                  (``plot dwell``)
* ``cmd_plot_brightness_heatmap``     (``plot brightness-heatmap``)
* ``cmd_plot_brightness_map``         (``plot brightness-map``)

These reference module-level helpers that still live in ``spt_pipeline.legacy``
(``_find_col``, ``_rglob_filtered``, ``_load_groupnames_condition_map``,
``_argv_has_flag``, ``_context_params``, ``_trackpy_quiet``, ...) and are
therefore NOT importable as-is. That is deliberate: reviving any of this means
making a decision about where those helpers should live, and that decision is
better made then than pre-empted now.

The argparse definitions that used to register these subcommands are kept in
``cli_fragments.py`` beside this file.
"""

def cmd_workflow_auto(args):
    """
    Run a standard analysis workflow on a single input folder:
      1) analyze brightness-mobility
      2) plot steps-brightness
      3) plot dwell

    This assumes you have already run `track trackpy` or `track dwelltime`
    so that *_tracks.csv files and groupnames.csv exist under the input folder.
    """
    root = os.path.abspath(args.input_dir)
    print(f"[workflow] Using input directory: {root}")

    # ------------------------------------------------------------------
    # 1) Brightness–mobility analysis
    # ------------------------------------------------------------------
    print("[workflow] Step 1/3: analyze brightness-mobility")
    bm_args = argparse.Namespace(
        input=root,
        outdir=args.outdir,
        dt=args.dt if _argv_has_flag("--dt") else None,
        use_maxlag=True,
        immobile_threshold_px=args.immobile_threshold_px if _argv_has_flag("--immobile-threshold-px") else None,
        display_step_cap_px=args.display_step_cap_px if _argv_has_flag("--display-step-cap-px") else None,
    )
    cmd_analyze_brightness_mobility(bm_args)

    # ------------------------------------------------------------------
    # 2) Step–brightness heatmaps & overlays
    # ------------------------------------------------------------------
    print("[workflow] Step 2/3: plot steps-brightness")
    sb_args = argparse.Namespace(
        input_dir=root,
        # For workflow, focus on tracking outputs from trackpy / dwelltime
        pattern=args.pattern if _argv_has_flag("--pattern") else None,
        make=args.make if _argv_has_flag("--make") else None,
        stepsize_max=args.stepsize_max if _argv_has_flag("--stepsize-max") else None,
        bins_x=args.bins_x if _argv_has_flag("--bins-x") else None,
        bins_y=args.bins_y if _argv_has_flag("--bins-y") else None,
        count_cap=args.count_cap if _argv_has_flag("--count-cap") else None,
        units_label=None,
        line_width=args.line_width if _argv_has_flag("--line-width") else None,
        invert_lut_tracks=False,
        min_track_length=args.min_track_length if _argv_has_flag("--min-track-length") else None,
        brightness_col=args.brightness_col if _argv_has_flag("--brightness-col") else None,
        x_col=None,
        y_col=None,
        strip_datecodes=True,
        workers=None,
    )
    cmd_plot_steps_brightness(sb_args)

    # ------------------------------------------------------------------
    # 3) Dwell-time plotting
    # ------------------------------------------------------------------
    print("[workflow] Step 3/3: plot dwell")
    dwell_args = argparse.Namespace(
        input_folder=root,
        cutoff=args.dwell_cutoff if _argv_has_flag("--dwell-cutoff") else None,
        mass=None,
        size=None,
        ecc=None,
        signal=None,
        raw_mass=10000,
        ep=None,
        frame=None,
        particle=None,
        total_mass=None,
    )
    cmd_plot_dwell(dwell_args)

    print("[workflow] Completed: analyze brightness-mobility → plot steps-brightness → plot dwell")

# ============================================================================
# PREP SUBCOMMAND - Data Preparation
# ============================================================================


def cmd_track_dwelltime(args):
    """Dwelltime analysis from ND2 - from dwelltime.py"""
    if not HAVE_ND2 or not HAVE_TRACKPY:
        raise SystemExit("nd2 and trackpy required for dwelltime analysis")
    
    import seaborn as sns
    
    input_folder = Path(args.input_folder)
    # Output goes to input directory with movie-named subdirectories
    all_conditions = []
    
    # Handle both flat structure (movies directly in input_folder) and nested (condition folders)
    movie_files = list(input_folder.glob("*.nd2"))
    condition_folders = [d for d in input_folder.glob("*") if d.is_dir() and not d.name.startswith('.')]
    
    channel = int(getattr(args, "channel", 0) or 0)

    def process_movie(movie_path, output_dir):
        """Process a single movie file"""
        movie_name = movie_path.stem
        movie_output = output_dir / movie_name
        movie_output.mkdir(parents=True, exist_ok=True)
        
        with nd2.ND2File(movie_path) as f:
            data = f.asarray()
        data = _select_channel(data, channel)
        frames = [data[i].astype(np.float32) for i in range(data.shape[0])]
        del data
        
        filtered_frames = [tp.bandpass(frame, lshort=1, llong=99) for frame in frames]
        
        features = tp.batch(filtered_frames, diameter=args.diam, minmass=args.minmass, engine='numba')
        tp.subpx_bias(features)
        plt.savefig(movie_output / f"{movie_name}_sub-pixel-bias.png")
        plt.close()
        
        t = tp.link(features, 10, memory=5)
        
        features['mass'].hist(bins=50)
        plt.savefig(movie_output / f"{movie_name}_mass.png")
        plt.close()
        features['size'].hist(bins=50)
        plt.savefig(movie_output / f"{movie_name}_size.png")
        plt.close()
        features['signal'].hist(bins=50)
        plt.savefig(movie_output / f"{movie_name}_signal.png")
        plt.close()
        
        plt.figure()
        tp.mass_size(t.groupby('particle').mean())
        plt.savefig(movie_output / f"{movie_name}_size-mass.png")
        plt.close()
        
        plt.figure(figsize=(8, 6))
        plt.scatter(t['ecc'], t['mass'], alpha=0.5)
        plt.xlabel("Eccentricity")
        plt.ylabel("Mass")
        plt.title("Eccentricity vs. Mass")
        plt.grid(True)
        plt.savefig(movie_output / f"{movie_name}_mass-ecc.png")
        plt.close()
        
        t1 = tp.filter_stubs(t, 3)
        
        dwell_times = t1.groupby("particle")["frame"].count()
        t1 = t1.merge(dwell_times.rename("dwell_time"), on="particle")
        
        t1.to_csv(movie_output / f"{movie_name}_filtered_tracks.csv", index=False)
        
        sns.histplot(dwell_times, bins=10, kde=True)
        plt.xlabel("Dwell Time (Frames)")
        plt.ylabel("Count")
        plt.title("Dwell Time Distribution")
        plt.savefig(movie_output / f"{movie_name}_dwell_time_hist.png")
        plt.close()
        
        t1['total_mass'] = t1.groupby('particle')['mass'].transform('sum')
        
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=t1, x="dwell_time", y="total_mass", alpha=0.7)
        plt.xlabel("Dwell Time (frames)")
        plt.ylabel("Total Mass")
        plt.title(f"Total Mass vs. Dwell Time - {movie_name}")
        plt.savefig(movie_output / f"{movie_name}_total_mass_scatter.png")
        plt.close()
        
        t_filtered = t[(t['total_mass'] > 500)]
        dwell_times = t_filtered.groupby("particle")["frame"].count()
        frame_interval = 1.0 / args.fps
        dwell_times_sec = dwell_times * frame_interval
        
        sns.histplot(dwell_times_sec, bins=10, kde=False)
        plt.xlabel("Dwell Time (Seconds)")
        plt.ylabel("Count")
        plt.title("Dwell Time Distribution")
        plt.savefig(movie_output / f"{movie_name}_filtered_dwelltime.png")
        plt.close()
        
        return t1
    
    # Process flat structure: movies directly in input_folder
    for movie_path in movie_files:
        process_movie(movie_path, input_folder)
    
    # Process nested structure: condition folders
    for condition_folder in condition_folders:
        condition_name = condition_folder.stem
        condition_replicates = []
        
        for movie_path in condition_folder.glob("*.nd2"):
            t1 = process_movie(movie_path, condition_folder)
            if t1 is not None:
                condition_replicates.append(t1)
        
        if condition_replicates:
            condition_df = pd.concat(condition_replicates, ignore_index=True)
            condition_df.to_csv(condition_folder / f"{condition_name}_combined.csv", index=False)
            all_conditions.append(condition_df)
    
    if all_conditions:
        final_data = pd.concat(all_conditions, ignore_index=True)
        plt.figure(figsize=(10,6))
        import seaborn as sns
        sns.boxplot(data=final_data, x="condition", y="dwell_time")
        sns.stripplot(data=final_data, x="condition", y="dwell_time", jitter=True, color="black", alpha=0.5)
        plt.xlabel("Experimental Condition")
        plt.ylabel("Dwell Time (frames)")
        plt.title("Dwell Time Comparison Across Conditions")
        plt.savefig(input_folder / "dwell_time_comparison.png")
        plt.close()


def cmd_analyze_brightness_mobility(args):
    """Brightness-mobility analysis - from spt_brightness_mobility_analysis_v4.py"""
    in_path = Path(args.input)
    context_root = in_path if in_path.is_dir() else in_path.parent
    analyze_ctx = _context_params(context_root, ["analyze", "brightness_mobility"])
    track_ctx = _context_params(context_root, ["track", "trackpy"])
    track_dt = track_ctx.get("dt")
    if track_dt is None and track_ctx.get("fps") not in (None, 0):
        track_dt = 1.0 / float(track_ctx["fps"])
    args.dt = float(_resolve_param(_argv_has_flag("--dt"), args.dt, analyze_ctx.get("dt", track_dt), 1.0))
    args.immobile_threshold_px = float(_resolve_param(
        _argv_has_flag("--immobile-threshold-px"),
        args.immobile_threshold_px,
        analyze_ctx.get("immobile_threshold_px"),
        0.01,
    ))
    args.display_step_cap_px = float(_resolve_param(
        _argv_has_flag("--display-step-cap-px"),
        args.display_step_cap_px,
        analyze_ctx.get("display_step_cap_px"),
        3.0,
    ))
    args.use_maxlag = bool(_resolve_param(
        _argv_has_flag("--use-maxlag", "--no-use-maxlag"),
        args.use_maxlag,
        analyze_ctx.get("use_maxlag"),
        True,
    ))
    
    POSSIBLE_COLS = {
        "x": ["X", "POSITION_X", "x", "spot_x"],
        "y": ["Y", "POSITION_Y", "y", "spot_y"],
        # Support a wide variety of time / frame labels, including GEMspa-style "Frame"
        "frame": [
            "FRAME", "frame", "Frame",
            "FRAME_INDEX", "frame_index",
            "FRAME ID", "frame id", "frameid", "FrameID",
            "SPOT_FRAME", "spot_frame",
            "T", "POSITION_T"
        ],
        # Track identifiers from TrackMate, trackpy, and GEMspa ("Trajectory")
        "track_id": [
            "TRACK_ID", "track_id", "ID", "TRACK_INDEX",
            "particle", "PARTICLE",          # trackpy
            "trajectory", "Trajectory", "TRAJECTORY"
        ],
        "intensity": ["INTENSITY", "MEAN_INTENSITY", "MAX_INTENSITY", "mass", "MASS",  # trackpy uses "mass" for intensity
                      "Spot mean intensity", "spot_mean_intensity", "mean_intensity"],
        "sigma": ["POSITION_SIGMA", "SIGMA", "LOCALIZATION_ERROR", "spot_sigma", "UNCERTAINTY"],
        "quality": ["QUALITY", "QUALITY_SCORE", "spot_quality"]
    }
    
    def find_first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Case-insensitive and partial matching for flexible column detection."""
        cols = list(df.columns)
        low = {c.lower(): c for c in cols}
        for cand in candidates:
            # Exact match
            if cand in cols:
                return cand
            cl = cand.lower()
            # Case-insensitive exact match
            if cl in low:
                return low[cl]
            # Partial substring match (e.g. "frame" → "Frame", "frame_index")
            for c in cols:
                if cl in c.lower():
                    return c
        return None
    
    def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
        mapping = {}
        for k, v in POSSIBLE_COLS.items():
            col = find_first_present(df, v)
            if col is not None:
                mapping[k] = col
        required = ["x", "y", "frame", "track_id"]
        missing = [r for r in required if r not in mapping]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return mapping
    
    def estimate_sigma_if_missing(df: pd.DataFrame, mapping: Dict[str, str]) -> np.ndarray:
        if "sigma" in mapping:
            s = pd.to_numeric(df[mapping["sigma"]], errors="coerce").to_numpy(dtype=float)
            s = np.where(s <= 0, np.nan, s)
            med = np.nanmedian(s) if np.isfinite(s).any() else 0.3
            s = np.where(np.isfinite(s), s, med)
            return s.astype(float)
        
        if "quality" in mapping:
            q = pd.to_numeric(df[mapping["quality"]], errors="coerce").to_numpy(dtype=float)
            k = (np.nanmedian(q[q > 0]) ** 0.5) if np.isfinite(q).any() and (q > 0).any() else 10.0
            s = k / np.sqrt(np.clip(q, 0, None) + 1e-9)
            return np.clip(s, 0.05, 1.5).astype(float)
        
        return np.full(len(df), 0.3, dtype=float)
    
    def compute_steps(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        d = df.sort_values([mapping["track_id"], mapping["frame"]]).reset_index(drop=True)
        x = d[mapping["x"]].to_numpy(dtype=float)
        y = d[mapping["y"]].to_numpy(dtype=float)
        tid = d[mapping["track_id"]].to_numpy()
        frm = d[mapping["frame"]].to_numpy()
        same_track = tid[1:] == tid[:-1]
        consecutive = frm[1:] - frm[:-1] == 1
        valid = same_track & consecutive
        step = np.full(len(x), np.nan, dtype=float)
        step[:-1] = np.where(valid, np.hypot(x[1:] - x[:-1], y[1:] - y[:-1]), np.nan)
        d["step_px"] = step
        return d
    
    def msd_by_lag(track_xy: np.ndarray, max_lag: int) -> np.ndarray:
        N = track_xy.shape[0]
        msd = np.full(max_lag, np.nan, dtype=float)
        for lag in range(1, max_lag + 1):
            if N - lag < 1:
                break
            disp = track_xy[lag:] - track_xy[:-lag]
            msd[lag - 1] = np.mean(np.sum(disp**2, axis=1))
        return msd
    
    def diffusion_coefficients_for_track(track_df: pd.DataFrame, mapping: Dict[str, str],
                                         dt: float = 1.0, use_maxlag: bool = True,
                                         apply_sigma_correction: bool = False,
                                         sigma_values: Optional[np.ndarray] = None) -> float:
        xy = track_df[[mapping["x"], mapping["y"]]].to_numpy(dtype=float)
        N = xy.shape[0]
        if N < 3:
            return np.nan
        msd = msd_by_lag(xy, N - 1)
        if apply_sigma_correction:
            if sigma_values is None:
                sigma_values = estimate_sigma_if_missing(track_df, mapping)
            sig = float(np.nanmedian(sigma_values))
            msd = msd - 2.0 * (sig ** 2)
        if not np.isfinite(msd).any():
            return np.nan
        if use_maxlag:
            idx = np.where(np.isfinite(msd))[0]
            if idx.size == 0:
                return np.nan
            L = int(idx[-1] + 1)
            return float(msd[idx[-1]] / (4.0 * L * dt))
        else:
            k = int(min(5, np.isfinite(msd).sum()))
            if k < 2:
                return np.nan
            y = msd[:k]
            x = np.arange(1, k + 1, dtype=float) * dt
            A = np.vstack([4.0 * x, np.ones_like(x)]).T
            coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            return float(coef[0])
    
    def choose_brightness_column(df: pd.DataFrame, mapping: Dict[str, str]) -> Tuple[Optional[str], pd.Series]:
        candidates = []
        if "intensity" in mapping:
            candidates.append(mapping["intensity"])
        if "quality" in mapping:
            candidates.append(mapping["quality"])
        for c in candidates:
            series = pd.to_numeric(df[c], errors="coerce")
            if np.isfinite(series).mean() >= 0.5:
                return c, series
        if candidates:
            return candidates[0], pd.to_numeric(df[candidates[0]], errors="coerce")
        return None, pd.Series(np.ones(len(df), dtype=float))
    
    def analyze_file(path: Path, args, out_root: Path):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            print(f"[WARN] Could not read {path.name}: {e}")
            return
        if df.empty:
            print(f"[INFO] Skipping empty file {path.name}")
            return
        
        try:
            mapping = detect_columns(df)
        except Exception as e:
            # Skip files that do not look like trajectory tables
            print(f"[SKIP] {path.name}: {e}")
            return
        
        for key in ("x", "y", "frame"):
            col = mapping[key]
            df[col] = pd.to_numeric(df[col], errors="coerce")
        before = len(df)
        df = df.dropna(subset=[mapping["x"], mapping["y"], mapping["frame"]]).copy()
        dropped = before - len(df)
        if dropped:
            print(f"[INFO] {path.name}: dropped {dropped} rows with non-numeric x/y/frame")
        
        df["_sigma_px"] = estimate_sigma_if_missing(df, mapping)
        bname, bseries = choose_brightness_column(df, mapping)
        df["_brightness"] = bseries.to_numpy(dtype=float)
        
        df = compute_steps(df, mapping)
        
        display_cap = float(args.display_step_cap_px)
        df["_step_px_display"] = np.clip(df["step_px"], None, display_cap)
        df["_is_immobile"] = np.nan_to_num(df["step_px"]) <= args.immobile_threshold_px
        grp = df.groupby(mapping["track_id"], sort=False)
        per_track = grp.agg(
            mean_step_px=("step_px", _nanmean_series),
            mean_step_px_display=("_step_px_display", _nanmean_series),
            immobile_frac=("_is_immobile", "mean"),
            mean_brightness=("_brightness", _nanmean_series),
            median_sigma_px=("_sigma_px", _nanmedian_series),
            n_spots=(mapping["frame"], "size"),
        ).reset_index().rename(columns={mapping["track_id"]: "track_id"})
        
        D_unc, D_cor = [], []
        for _, g in grp:
            D_unc.append(diffusion_coefficients_for_track(
                g, mapping, dt=args.dt, use_maxlag=args.use_maxlag, apply_sigma_correction=False))
            D_cor.append(diffusion_coefficients_for_track(
                g, mapping, dt=args.dt, use_maxlag=args.use_maxlag, apply_sigma_correction=True,
                sigma_values=g["_sigma_px"].to_numpy(dtype=float)))
        per_track["D_uncorr"] = np.array(D_unc, dtype=float)
        per_track["D_corr"] = np.array(D_cor, dtype=float)
        
        outdir = out_root / f"{path.stem}_analysis"
        outdir.mkdir(parents=True, exist_ok=True)
        
        per_track.to_csv(outdir / "per_track_metrics.csv", index=False)
        
        def _save(fig, name):
            fig.tight_layout()
            fig.savefig(outdir / name, dpi=300)
            plt.close(fig)
        
        fig = plt.figure()
        plt.hist(per_track["mean_brightness"].to_numpy(dtype=float), bins=50)
        plt.xlabel(f"Brightness ({bname if bname else 'auto'})")
        plt.ylabel("Count")
        _save(fig, "hist_brightness.png")
        
        fig = plt.figure()
        plt.scatter(per_track["mean_brightness"].to_numpy(dtype=float),
                    per_track["mean_step_px"].to_numpy(dtype=float), s=6, alpha=0.5)
        plt.xlabel(f"Brightness ({bname if bname else 'auto'})")
        plt.ylabel("Mean step (px) [no cap]")
        _save(fig, "scatter_brightness_vs_meanstep_nocap.png")
        
        fig = plt.figure()
        plt.scatter(per_track["mean_brightness"].to_numpy(dtype=float),
                    per_track["mean_step_px_display"].to_numpy(dtype=float), s=6, alpha=0.5)
        plt.xlabel(f"Brightness ({bname if bname else 'auto'})")
        plt.ylabel(f"Mean step (px) [display cap {display_cap:g}]")
        _save(fig, "scatter_brightness_vs_meanstep_displaycap.png")
        
        fig = plt.figure()
        plt.violinplot([per_track["D_uncorr"].to_numpy(dtype=float),
                        per_track["D_corr"].to_numpy(dtype=float)], showmeans=True)
        plt.xticks([1, 2], ["D_uncorr", "D_corr"])
        plt.ylabel("D (px²/frame)")
        _save(fig, "violin_D_uncorr_vs_corr.png")
        
        df_valid = df[np.isfinite(df["step_px"]) & np.isfinite(df["_sigma_px"])].copy()
        if not df_valid.empty:
            q = np.quantile(df_valid["_sigma_px"].to_numpy(dtype=float), [0.33, 0.66])
            bins = [(-np.inf, q[0]), (q[0], q[1]), (q[1], np.inf)]
            labels = ["low σ", "mid σ", "high σ"]
            fig = plt.figure()
            for (lo, hi), lab in zip(bins, labels):
                sel = (df_valid["_sigma_px"] > lo) & (df_valid["_sigma_px"] <= hi)
                steps = df_valid.loc[sel, "step_px"].to_numpy(dtype=float)
                steps = steps[np.isfinite(steps)]
                if steps.size < 1:
                    continue
                s = np.sort(steps)
                c = np.arange(1, s.size + 1) / s.size
                plt.plot(s, c, label=lab)
            plt.xlabel("Step size (px)")
            plt.ylabel("CDF")
            plt.legend()
            _save(fig, "cdf_steps_by_sigma_bins.png")
        
        try:
            mask = np.isfinite(per_track["mean_brightness"]) & np.isfinite(per_track["median_sigma_px"]) & np.isfinite(per_track["mean_step_px"])
            if HAVE_SM and mask.sum() > 5:
                X = per_track.loc[mask, ["mean_brightness", "median_sigma_px"]].to_numpy(dtype=float)
                y = per_track.loc[mask, "mean_step_px"].to_numpy(dtype=float)
                Xd = sm.add_constant(X)
                model = sm.OLS(y, Xd).fit()
                with open(outdir / "partial_regression_summary.txt", "w") as fh:
                    fh.write(str(model.summary()))
            else:
                with open(outdir / "partial_regression_summary.txt", "w") as fh:
                    fh.write("Statsmodels unavailable or not enough points; skipped.\n")
        except Exception as e:
            with open(outdir / "partial_regression_summary.txt", "w") as fh:
                fh.write(f"Regression error: {e}\n")
        
        with open(outdir / "summary_metrics.json", "w") as fh:
            json.dump({
                "file": str(path),
                "output_dir": str(outdir),
                "brightness_column": bname if bname else "auto/constant",
                "immobile_threshold_px": float(args.immobile_threshold_px),
                "display_step_cap_px": float(args.display_step_cap_px),
                "dt_per_frame": float(args.dt),
                "use_maxlag_for_D": bool(args.use_maxlag),
                "n_tracks": int(per_track.shape[0])
            }, fh, indent=2)
        
        print(f"[OK] {path.name} → {outdir}")
    
    # Recursively search for CSV files if directory, or use single file
    if in_path.is_dir():
        # Prefer Trackpy trajectory outputs when scanning a top-level pipeline folder.
        csvs_tracks = _rglob_filtered(in_path, "*_tracks.csv")
        csvs = csvs_tracks if csvs_tracks else _rglob_filtered(in_path, "*.csv")
        if not csvs:
            print(f"[WARN] No CSV files found in {in_path} (searched recursively)")
            return
    else:
        csvs = [in_path]
    
    if args.outdir:
        out_root = Path(args.outdir)
    else:
        out_root = in_path if in_path.is_dir() else in_path.parent
    out_root.mkdir(parents=True, exist_ok=True)
    
    for p in csvs:
        analyze_file(p, args, out_root)
    
    _save_pipeline_context(context_root, ["analyze", "brightness_mobility"], {
        "input": str(in_path),
        "dt": float(args.dt),
        "use_maxlag": bool(args.use_maxlag),
        "immobile_threshold_px": float(args.immobile_threshold_px),
        "display_step_cap_px": float(args.display_step_cap_px),
    })
    
    print("All done.")


def cmd_plot_steps_brightness(args):
    """Step-brightness heatmaps and track overlays - from analyze_steps_and_tracks.py"""
    context_root = Path(args.input_dir)
    plot_ctx = _context_params(context_root, ["plot", "steps_brightness"])
    analyze_ctx = _context_params(context_root, ["analyze", "brightness_mobility"])
    args.pattern = _resolve_param(_argv_has_flag("--pattern"), args.pattern, plot_ctx.get("pattern"), "*.csv")
    args.make = _resolve_param(_argv_has_flag("--make"), args.make, plot_ctx.get("make"), "both")
    args.stepsize_max = float(_resolve_param(
        _argv_has_flag("--stepsize-max"),
        args.stepsize_max,
        plot_ctx.get("stepsize_max", analyze_ctx.get("display_step_cap_px")),
        3.0,
    ))
    args.bins_x = int(_resolve_param(_argv_has_flag("--bins-x"), args.bins_x, plot_ctx.get("bins_x"), 150))
    args.bins_y = int(_resolve_param(_argv_has_flag("--bins-y"), args.bins_y, plot_ctx.get("bins_y"), 150))
    args.count_cap = int(_resolve_param(_argv_has_flag("--count-cap"), args.count_cap, plot_ctx.get("count_cap"), 300))
    args.units_label = _resolve_param(_argv_has_flag("--units-label"), args.units_label, plot_ctx.get("units_label"), "pixels")
    args.line_width = float(_resolve_param(_argv_has_flag("--line-width"), args.line_width, plot_ctx.get("line_width"), 0.7))
    args.invert_lut_tracks = bool(_resolve_param(
        _argv_has_flag("--invert-lut-tracks", "--no-invert-lut-tracks"),
        args.invert_lut_tracks,
        plot_ctx.get("invert_lut_tracks"),
        False,
    ))
    args.min_track_length = int(_resolve_param(
        _argv_has_flag("--min-track-length"),
        args.min_track_length,
        plot_ctx.get("min_track_length"),
        10,
    ))
    args.brightness_col = _resolve_param(
        _argv_has_flag("--brightness-col"),
        args.brightness_col,
        plot_ctx.get("brightness_col"),
        None,
    )
    args.x_col = _resolve_param(_argv_has_flag("--x-col"), args.x_col, plot_ctx.get("x_col"), None)
    args.y_col = _resolve_param(_argv_has_flag("--y-col"), args.y_col, plot_ctx.get("y_col"), None)
    args.strip_datecodes = bool(_resolve_param(
        _argv_has_flag("--strip-datecodes", "--no-strip-datecodes"),
        args.strip_datecodes,
        plot_ctx.get("strip_datecodes"),
        True,
    ))
    args.workers = _resolve_param(_argv_has_flag("--workers"), args.workers, plot_ctx.get("workers"), None)
    
    def strip_datecode(name: str) -> str:
        return re.sub(r'-(\d{4,8})(?=(_|$))', '', name)
    
    def normalize_source_name(name: str) -> str:
        base = _normalize_track_output_name(os.path.splitext(os.path.basename(name))[0])
        return strip_datecode(base) if args.strip_datecodes else base
    
    def detect_mode(csv_path: str) -> str:
        df = pd.read_csv(csv_path, nrows=20, low_memory=False)
        x_col = _find_col(df, ["POSITION_X", "X", "POSITION X", "X_POS", "x"])
        y_col = _find_col(df, ["POSITION_Y", "Y", "POSITION Y", "Y_POS", "y"])
        track_col = _find_col(df, ["TRACK_ID", "TRACKID", "TRACK", "particle", "PARTICLE", "track_id"])
        if x_col and y_col and track_col:
            return "trajectory"
        if _find_col(df, ["step_size", "STEP_SIZE"]) and _find_col(df, ["brightness", "BRIGHTNESS"]):
            return "steps"
        raise ValueError(f"Could not detect data type for {os.path.basename(csv_path)}")
    
    def infer_condition_for_file(csv_path: str, condition_map: Dict[str, str], default_condition: str) -> str:
        source_name = normalize_source_name(csv_path)
        if source_name in condition_map:
            return condition_map[source_name]
        folder_name = _normalize_track_output_name(Path(csv_path).parent.name)
        if folder_name in condition_map:
            return condition_map[folder_name]
        return default_condition
    
    def discover_relevant_files(indir: str) -> Tuple[List[str], str, Dict[str, str]]:
        root = Path(indir)
        condition_map = _load_groupnames_condition_map(root)
        default_condition = root.name
        
        track_files = [str(p) for p in _rglob_filtered(root, "*_tracks.csv")]
        if track_files:
            print(f"[INFO] Auto-detected {len(track_files)} *_tracks.csv files")
            return track_files, "trajectory", condition_map
        
        candidates = [p for p in _rglob_filtered(root, args.pattern)]
        recognized: List[Tuple[str, str]] = []
        for p in candidates:
            try:
                recognized.append((str(p), detect_mode(str(p))))
            except Exception:
                continue
        
        if not recognized:
            raise SystemExit(f"No relevant CSV files found in {indir} (searched recursively with pattern {args.pattern!r})")
        
        trajectory_files = [path for path, mode in recognized if mode == "trajectory"]
        if trajectory_files:
            print(f"[INFO] Auto-detected {len(trajectory_files)} trajectory CSV files")
            return trajectory_files, "trajectory", condition_map
        
        step_files = [path for path, mode in recognized if mode == "steps"]
        print(f"[INFO] Auto-detected {len(step_files)} precomputed steps/brightness tables")
        return step_files, "steps", condition_map
    
    def compute_steps_from_trackmate(df: pd.DataFrame,
                                     brightness_override: Optional[str] = None,
                                     x_override: Optional[str] = None,
                                     y_override: Optional[str] = None) -> pd.DataFrame:
        x_col = x_override or _find_col(df, ["POSITION_X", "X", "POSITION X", "X_POS"])
        y_col = y_override or _find_col(df, ["POSITION_Y", "Y", "POSITION Y", "Y_POS"])
        track_col = _find_col(df, ["TRACK_ID", "TRACKID", "TRACK", "particle", "PARTICLE"])
        frame_col = _find_col(df, ["FRAME", "POSITION_T", "T", "TIME"])
        
        bright_col = None
        if brightness_override:
            if brightness_override in df.columns:
                bright_col = brightness_override
            else:
                bright_col = _find_col(df, [brightness_override])
        if bright_col is None:
            bright_col = _find_col(df, [
                "MEAN_INTENSITY_CH1", "MEAN_INTENSITY", "INTENSITY",
                "MEDIAN_INTENSITY", "QUALITY", "mass", "MASS",
                "raw_mass", "RAW_MASS", "signal", "SIGNAL"
            ])
        
        if not all([x_col, y_col, track_col, bright_col]):
            raise ValueError("Missing columns (need POSITION_X/Y, TRACK_ID, and a brightness column).")
        
        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
        df[bright_col] = pd.to_numeric(df[bright_col], errors="coerce")
        
        rows = []
        for tid, g in df.groupby(track_col, dropna=True):
            g = g.copy()
            if frame_col and frame_col in g.columns:
                g = g.sort_values(frame_col, kind="mergesort")
            if len(g) < 2:
                continue
            
            x = g[x_col].to_numpy(dtype=float)
            y = g[y_col].to_numpy(dtype=float)
            b = g[bright_col].to_numpy(dtype=float)
            
            x1000 = x * 1000.0
            y1000 = y * 1000.0
            dx = x1000[1:] - x1000[:-1]
            dy = y1000[1:] - y1000[:-1]
            step_px = np.sqrt(dx*dx + dy*dy) / 1000.0
            
            step_brightness = 0.5 * (b[1:] + b[:-1])
            rows.append(pd.DataFrame({"step_size": step_px, "brightness": step_brightness}))
        
        if not rows:
            return pd.DataFrame(columns=["step_size", "brightness"])
        return pd.concat(rows, ignore_index=True)
    
    def load_steps_table(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)
        step_col = _find_col(df, ["step_size", "STEP_SIZE"])
        bright_col = _find_col(df, ["brightness", "BRIGHTNESS"])
        if not step_col or not bright_col:
            raise ValueError(f"{os.path.basename(path)} is not a steps_vs_brightness table.")
        out = df[[step_col, bright_col]].copy()
        out.columns = ["step_size", "brightness"]
        return out
    
    def build_hist2d(df: pd.DataFrame, stepsize_max: float, ymin: float, ymax: float, bins_x: int, bins_y: int):
        x = df["step_size"].to_numpy()
        y = df["brightness"].to_numpy()
        H, _, _ = np.histogram2d(x, y, bins=[bins_x, bins_y], range=[[0, stepsize_max], [ymin, ymax]])
        return H
    
    def plot_heatmap_matrix(H: np.ndarray, stepsize_max: float, ymin: float, ymax: float, vmax_cap: int, title: str, out_png: str, units_label: str = "pixels"):
        Hc = np.clip(H, 0, vmax_cap)
        plt.figure(figsize=(6, 5))
        plt.imshow(Hc.T, origin="lower", aspect="auto",
                   extent=[0, stepsize_max, ymin, ymax],
                   cmap="viridis", norm=Normalize(vmin=0, vmax=vmax_cap))
        sat = Hc.T >= (vmax_cap - 1e-9)
        if np.any(sat):
            overlay = np.zeros((*sat.shape, 4))
            overlay[sat] = [1, 0, 0, 1]
            plt.imshow(overlay, origin="lower", aspect="auto",
                       extent=[0, stepsize_max, ymin, ymax])
        plt.colorbar(label=f"Counts (cap={vmax_cap}; red=saturated)")
        plt.xlabel(f"Step size ({units_label})")
        plt.ylabel("Brightness (a.u.)")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()
    
    def segments_from_trackmate(df: pd.DataFrame, min_track_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        x_col = _find_col(df, ["POSITION_X", "X", "POSITION X", "X_POS"])
        y_col = _find_col(df, ["POSITION_Y", "Y", "POSITION Y", "Y_POS"])
        track_col = _find_col(df, ["TRACK_ID", "TRACKID", "TRACK", "particle", "PARTICLE"])
        frame_col = _find_col(df, ["FRAME", "POSITION_T", "T", "TIME"])
        if not all([x_col, y_col, track_col]):
            raise ValueError("Missing columns for overlays (POSITION_X/Y, TRACK_ID).")
        
        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
        
        segs, vals = [], []
        for tid, g in df.groupby(track_col, dropna=True):
            g = g.copy()
            if frame_col and frame_col in g.columns:
                g = g.sort_values(frame_col, kind="mergesort")
            if len(g) < max(2, int(min_track_length)):
                continue
            
            x = g[x_col].to_numpy(dtype=float)
            y = g[y_col].to_numpy(dtype=float)
            
            x1000 = x * 1000.0
            y1000 = y * 1000.0
            dx = x1000[1:] - x1000[:-1]
            dy = y1000[1:] - y1000[:-1]
            step_px = np.sqrt(dx*dx + dy*dy) / 1000.0
            
            p0 = np.stack([x[:-1], y[:-1]], axis=1)
            p1 = np.stack([x[1:],  y[1:]],  axis=1)
            segs.extend(np.stack([p0, p1], axis=1))
            vals.extend(step_px)
        
        segs = np.asarray(segs) if segs else np.empty((0,2,2), dtype=float)
        vals = np.asarray(vals) if vals else np.empty((0,), dtype=float)
        return segs, vals
    
    indir = os.path.abspath(args.input_dir)
    if not os.path.isdir(indir):
        raise SystemExit(f"Input dir not found: {indir}")
    default_condition = Path(indir).name
    files, mode, condition_map = discover_relevant_files(indir)
    print(f"[INFO] Detected mode: {mode}")
    if condition_map:
        print(f"[INFO] Loaded {len(condition_map)} condition assignments from groupnames.csv")
    
    workers = args.workers or os.cpu_count() or 4
    workers = max(1, int(workers))
    
    if args.make in ("both", "heatmaps"):
        outdir_hm = os.path.join(indir, "brightness_stepsize")
        os.makedirs(outdir_hm, exist_ok=True)
        
        parts = []
        def _hm_worker(fpath: str):
            try:
                if mode == "trajectory":
                    df_raw = pd.read_csv(fpath, low_memory=False)
                    part = compute_steps_from_trackmate(df_raw,
                                                        brightness_override=args.brightness_col,
                                                        x_override=args.x_col,
                                                        y_override=args.y_col)
                else:
                    part = load_steps_table(fpath)
                if part is not None and not part.empty:
                    part["source_file"] = normalize_source_name(fpath)
                    part["condition"] = infer_condition_for_file(fpath, condition_map, default_condition)
                    return part
            except Exception as e:
                print(f"[WARN] Heatmap skip {os.path.basename(fpath)}: {e}")
            return None
        
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_hm_worker, f) for f in files]
            for fut in as_completed(futs):
                res = fut.result()
                if res is not None:
                    parts.append(res)
        
        if parts:
            combined = pd.concat(parts, ignore_index=True).dropna()
            combined = combined[combined["step_size"] < args.stepsize_max]
            all_csv = os.path.join(outdir_hm, "steps_vs_brightness_all.csv")
            combined.to_csv(all_csv, index=False)
            print(f"[OK] Wrote {all_csv} ({len(combined)} rows)")
            
            ymin, ymax = float(combined["brightness"].min()), float(combined["brightness"].max())
            global_vmax = 0
            file_groups = [(fname, df) for fname, df in combined.groupby("source_file", dropna=True)]
            condition_groups = [
                (cond, df) for cond, df in combined.groupby("condition", dropna=True)
                if str(cond).strip()
            ]
            for _, df in file_groups + condition_groups:
                H = build_hist2d(df, args.stepsize_max, ymin, ymax, args.bins_x, args.bins_y)
                global_vmax = max(global_vmax, int(H.max()))
            global_vmax = min(max(global_vmax, 1), int(args.count_cap))
            print(f"[INFO] Heatmap global vmax (capped): {global_vmax}")
            
            for fname, df in file_groups:
                safe = fname.replace(" ", "_")
                out_png = os.path.join(outdir_hm, f"heatmap_{safe}.png")
                H = build_hist2d(df, args.stepsize_max, ymin, ymax, args.bins_x, args.bins_y)
                plot_heatmap_matrix(H, args.stepsize_max, ymin, ymax, global_vmax, title=fname, out_png=out_png, units_label=args.units_label)
            
            for condition, df in condition_groups:
                safe = simple_clean(condition)
                cond_csv = os.path.join(outdir_hm, f"steps_vs_brightness_{safe}.csv")
                df.to_csv(cond_csv, index=False)
                out_png = os.path.join(outdir_hm, f"heatmap_condition_{safe}.png")
                H = build_hist2d(df, args.stepsize_max, ymin, ymax, args.bins_x, args.bins_y)
                plot_heatmap_matrix(
                    H,
                    args.stepsize_max,
                    ymin,
                    ymax,
                    global_vmax,
                    title=f"{condition} (pooled replicates)",
                    out_png=out_png,
                    units_label=args.units_label,
                )
    
    if args.make in ("both", "tracks") and mode == "trajectory":
        outdir_tr = os.path.join(indir, "tracks_stepsize_map")
        os.makedirs(outdir_tr, exist_ok=True)
        
        perfile = {}
        xmin = ymin = float("inf")
        xmax = ymax = float("-inf")
        
        def _track_worker(fpath: str):
            try:
                df_raw = pd.read_csv(fpath, low_memory=False)
                segs, vals = segments_from_trackmate(df_raw, min_track_length=args.min_track_length)
                if segs.size == 0:
                    return (fpath, None, None, None)
                x_all = segs[:, :, 0].ravel()
                y_all = segs[:, :, 1].ravel()
                bounds = (float(np.nanmin(x_all)), float(np.nanmax(x_all)),
                          float(np.nanmin(y_all)), float(np.nanmax(y_all)))
                return (fpath, segs, vals, bounds)
            except Exception as e:
                print(f"[WARN] Tracks skip {os.path.basename(fpath)}: {e}")
                return (fpath, None, None, None)
        
        results = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_track_worker, f) for f in files]
            for fut in as_completed(futs):
                results.append(fut.result())
        
        for f, segs, vals, bounds in results:
            if segs is None:
                continue
            perfile[f] = (segs, vals)
            xmin = min(xmin, bounds[0])
            xmax = max(xmax, bounds[1])
            ymin = min(ymin, bounds[2])
            ymax = max(ymax, bounds[3])
        
        if perfile:
            cmap_name = "viridis_r" if args.invert_lut_tracks else "viridis"
            norm = Normalize(vmin=0.0, vmax=max(args.stepsize_max, 1e-6))
            for f, (segs, vals) in perfile.items():
                base = normalize_source_name(f)
                safe = base.replace(" ", "_")
                out_png = os.path.join(outdir_tr, f"overlay_{safe}.png")
                
                plt.figure(figsize=(7, 7))
                ax = plt.gca()
                vals_clamped = np.clip(vals, norm.vmin, norm.vmax)
                lc = LineCollection(segs, array=vals_clamped, cmap=cmap_name, norm=norm, linewidths=args.line_width)
                ax.add_collection(lc)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_aspect("equal", adjustable="box")
                cbar = plt.colorbar(lc)
                cbar.set_label(f"Step size ({args.units_label})")
                ax.set_title(base)
                plt.tight_layout()
                plt.savefig(out_png, dpi=300)
                plt.close()
    
    _save_pipeline_context(context_root, ["plot", "steps_brightness"], {
        "input_dir": str(context_root),
        "pattern": args.pattern,
        "make": args.make,
        "stepsize_max": float(args.stepsize_max),
        "bins_x": int(args.bins_x),
        "bins_y": int(args.bins_y),
        "count_cap": int(args.count_cap),
        "units_label": args.units_label,
        "line_width": float(args.line_width),
        "invert_lut_tracks": bool(args.invert_lut_tracks),
        "min_track_length": int(args.min_track_length),
        "brightness_col": args.brightness_col,
        "x_col": args.x_col,
        "y_col": args.y_col,
        "strip_datecodes": bool(args.strip_datecodes),
        "workers": int(args.workers) if args.workers is not None else None,
    })


def cmd_plot_dwell(args):
    """Dwell time plotting - from plotDwell.5.3.py"""
    context_root = Path(args.input_folder)
    plot_ctx = _context_params(context_root, ["plot", "dwell"])
    args.cutoff = float(_resolve_param(_argv_has_flag("--cutoff"), args.cutoff, plot_ctx.get("cutoff"), 4.0))
    args.mass = _resolve_param(_argv_has_flag("--mass"), args.mass, plot_ctx.get("mass"), None)
    args.size = _resolve_param(_argv_has_flag("--size"), args.size, plot_ctx.get("size"), None)
    args.ecc = _resolve_param(_argv_has_flag("--ecc"), args.ecc, plot_ctx.get("ecc"), None)
    args.signal = _resolve_param(_argv_has_flag("--signal"), args.signal, plot_ctx.get("signal"), None)
    args.raw_mass = float(_resolve_param(_argv_has_flag("--raw_mass"), args.raw_mass, plot_ctx.get("raw_mass"), 10000))
    args.ep = _resolve_param(_argv_has_flag("--ep"), args.ep, plot_ctx.get("ep"), None)
    args.frame = _resolve_param(_argv_has_flag("--frame"), args.frame, plot_ctx.get("frame"), None)
    args.particle = _resolve_param(_argv_has_flag("--particle"), args.particle, plot_ctx.get("particle"), None)
    args.total_mass = _resolve_param(_argv_has_flag("--total_mass"), args.total_mass, plot_ctx.get("total_mass"), None)
    
    def fit_gmm(data, max_components=3):
        best_bic = np.inf
        best_model = None
        for n in range(1, max_components+1):
            try:
                model = GaussianMixture(n_components=n, random_state=0)
                model.fit(data.reshape(-1, 1))
                bic = model.bic(data.reshape(-1, 1))
                if bic < best_bic:
                    best_bic = bic
                    best_model = model
            except Exception:
                continue
        return best_model
    
    group_file = os.path.join(args.input_folder, "groupnames.csv")
    if not os.path.exists(group_file):
        print("groupnames.csv not found in the input folder.")
        return
    
    try:
        group_df = pd.read_csv(group_file)
    except Exception as e:
        print(f"Error reading groupnames.csv: {e}")
        return
    
    use_rntp = "rNTP" in group_df.columns
    if not all(col in group_df.columns for col in ["folder", "condition"]):
        print("groupnames.csv must contain at least 'folder' and 'condition' columns.")
        return
    
    parameters = ["dwell_time", "mass", "size", "ecc", "signal", "raw_mass", "ep", "frame", "particle", "total_mass"]
    conditions_data_raw = {}
    conditions_data_filtered = {}
    
    _all_tracks_cache = [
        f for f in glob.glob(os.path.join(args.input_folder, "**", "*_tracks.csv"), recursive=True)
        if not _should_ignore_path(f)
    ]
    
    for idx, row in group_df.iterrows():
        folder_name = str(row["folder"]).strip()
        condition = str(row["condition"]).strip()
        
        if use_rntp:
            rntp_str = str(row["rNTP"]).strip().lower()
            has_rntp = "yes" in rntp_str if rntp_str else True
        else:
            has_rntp = True
        
        for d in (conditions_data_raw, conditions_data_filtered):
            if condition not in d:
                d[condition] = {}
            if has_rntp not in d[condition]:
                d[condition][has_rntp] = {}
            for p in parameters:
                if p not in d[condition][has_rntp]:
                    d[condition][has_rntp][p] = []
        
        replicate_folder = os.path.join(args.input_folder, folder_name)
        if not os.path.isdir(replicate_folder):
            files = [f for f in _all_tracks_cache if folder_name in f]
            if not files:
                continue
        else:
            files = [f for f in _all_tracks_cache if f.startswith(replicate_folder + os.sep)]
            if not files:
                continue
        
        tracks_file = files[0]
        try:
            df_raw = pd.read_csv(tracks_file)
            if "frame" in df_raw.columns:
                df_raw = df_raw[df_raw["frame"] < 1000]
        except Exception as e:
            print(f"Error reading '{tracks_file}': {e}")
            continue
        
        for p in parameters:
            if p in df_raw.columns:
                conditions_data_raw[condition][has_rntp][p].extend(df_raw[p].tolist())
        
        df_filtered = df_raw.copy()
        df_filtered = df_filtered[df_filtered["dwell_time"] >= args.cutoff]
        optional_filters = {
            "mass": args.mass, "size": args.size, "ecc": args.ecc, "signal": args.signal,
            "raw_mass": args.raw_mass, "ep": args.ep, "frame": args.frame,
            "particle": args.particle, "total_mass": args.total_mass,
        }
        for col_name, min_value in optional_filters.items():
            if min_value is not None and col_name in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[col_name] >= min_value]
        
        if not df_filtered.empty:
            for p in parameters:
                if p in df_filtered.columns:
                    conditions_data_filtered[condition][has_rntp][p].extend(df_filtered[p].tolist())
    
    for data_label, data_dict in [("raw", conditions_data_raw), ("filtered", conditions_data_filtered)]:
        for cond, groups in data_dict.items():
            cond_folder = os.path.join(args.input_folder, "histograms", simple_clean(cond), "no_fits", data_label)
            os.makedirs(cond_folder, exist_ok=True)
            
            present_flags = list(groups.keys())
            colors = ["blue", "orange", "green", "purple", "red"]
            flag_labels = {True: "+ rNTP", False: "- rNTP"} if use_rntp else {True: cond}
            
            for p in parameters:
                flag_data = {flag: np.array(groups[flag].get(p, [])) for flag in present_flags}
                if all(data.size == 0 for data in flag_data.values()):
                    continue
                
                plt.figure(figsize=(8, 6))
                ax = plt.gca()
                
                for i, flag in enumerate(present_flags):
                    data = flag_data[flag]
                    if data.size == 0:
                        continue
                    data[data <= 0] = 1
                    label_suffix = flag_labels.get(flag, str(flag))
                    ax.hist(data, bins=30, density=True, alpha=0.5, color=colors[i % len(colors)],
                            label=f"{cond} {label_suffix}")
                    overall_mean = data.mean()
                    ax.axvline(overall_mean, color=colors[i % len(colors)], linestyle=":", linewidth=2,
                               label=f"Overall Mean {label_suffix} = {overall_mean:.2f}")
                    
                    if data_label == "filtered":
                        gmm = fit_gmm(data)
                        if gmm:
                            for m in gmm.means_.flatten():
                                ax.axvline(m, color=colors[i % len(colors)], linestyle="--", linewidth=1)
                
                ax.set_title(f"{cond} - {p} ({data_label})")
                ax.set_xlabel(p)
                ax.set_ylabel("Density")
                ax.legend()
                plt.tight_layout()
                plot_path = os.path.join(cond_folder, f"{simple_clean(p)}.png")
                plt.savefig(plot_path)
                plt.close()
    
    _save_pipeline_context(context_root, ["plot", "dwell"], {
        "input_folder": str(context_root),
        "cutoff": float(args.cutoff),
        "mass": args.mass,
        "size": args.size,
        "ecc": args.ecc,
        "signal": args.signal,
        "raw_mass": float(args.raw_mass),
        "ep": args.ep,
        "frame": args.frame,
        "particle": args.particle,
        "total_mass": args.total_mass,
    })


def cmd_plot_brightness_heatmap(args):
    """Condition-pooled brightness-step heatmaps - from brightness_step_heatmap.py"""
    DEFAULT_BRIGHTNESS_CANDIDATES = ["MEAN_INTENSITY", "MEAN_INTENSITY_CH1", "INTENSITY", "MEDIAN_INTENSITY", "QUALITY"]
    DEFAULT_CONDITIONS = ["vector", "HRAS", "HK1", "HKWT", "12V", "180S"]
    context_root = Path(args.input_dir)
    plot_ctx = _context_params(context_root, ["plot", "brightness_heatmap"])
    analyze_ctx = _context_params(context_root, ["analyze", "brightness_mobility"])
    args.pattern = _resolve_param(_argv_has_flag("--pattern"), args.pattern, plot_ctx.get("pattern"), "*.csv")
    args.track_col = _resolve_param(_argv_has_flag("--track-col"), args.track_col, plot_ctx.get("track_col"), "TRACK_ID")
    args.x_col = _resolve_param(_argv_has_flag("--x-col"), args.x_col, plot_ctx.get("x_col"), "POSITION_X")
    args.y_col = _resolve_param(_argv_has_flag("--y-col"), args.y_col, plot_ctx.get("y_col"), "POSITION_Y")
    args.frame_col = _resolve_param(
        _argv_has_flag("--frame-col"),
        args.frame_col,
        plot_ctx.get("frame_col"),
        ["FRAME", "POSITION_T", "T"],
    )
    args.brightness_column = _resolve_param(
        _argv_has_flag("--brightness-column"),
        args.brightness_column,
        plot_ctx.get("brightness_column"),
        None,
    )
    args.conditions = _resolve_param(
        _argv_has_flag("--conditions"),
        args.conditions,
        plot_ctx.get("conditions"),
        DEFAULT_CONDITIONS,
    )
    args.bins_x = int(_resolve_param(_argv_has_flag("--bins-x"), args.bins_x, plot_ctx.get("bins_x"), 80))
    args.bins_y = int(_resolve_param(_argv_has_flag("--bins-y"), args.bins_y, plot_ctx.get("bins_y"), 80))
    args.stepsize_max = float(_resolve_param(
        _argv_has_flag("--stepsize-max"),
        args.stepsize_max,
        plot_ctx.get("stepsize_max", analyze_ctx.get("display_step_cap_px")),
        5.0,
    ))
    args.count_cap = float(_resolve_param(_argv_has_flag("--count-cap"), args.count_cap, plot_ctx.get("count_cap"), 300.0))
    args.log_counts = bool(_resolve_param(
        _argv_has_flag("--log-counts", "--no-log-counts"),
        args.log_counts,
        plot_ctx.get("log_counts"),
        False,
    ))
    args.ymin = _resolve_param(_argv_has_flag("--ymin"), args.ymin, plot_ctx.get("ymin"), None)
    args.ymax = _resolve_param(_argv_has_flag("--ymax"), args.ymax, plot_ctx.get("ymax"), None)
    
    def find_column(candidates: List[str], columns: List[str]) -> Optional[str]:
        colset = {c.lower(): c for c in columns}
        for cand in candidates:
            if cand in columns:
                return cand
            if cand.lower() in colset:
                return colset[cand.lower()]
            for c in columns:
                if cand.lower() in c.lower():
                    return c
        return None
    
    def infer_condition_from_filename(filename: str, ordered_tokens: List[str]) -> str:
        base = os.path.basename(filename).lower()
        for tok in ordered_tokens:
            if tok.lower() in base:
                return tok
        return "unknown"
    
    def load_and_compute_steps(csv_path: str, track_col: str, x_col: str, y_col: str,
                                frame_col_candidates: List[str], brightness_col: Optional[str]) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        
        if track_col not in df.columns:
            maybe = find_column([track_col, "TRACK_ID", "TRACKID", "TRACK", "particle", "PARTICLE"], df.columns)
            if maybe is None:
                raise ValueError(f"Could not find a track ID column in {csv_path}.")
            track_col = maybe
        
        if x_col not in df.columns:
            maybe_x = find_column([x_col, "X", "POSITION X", "POSITIONX"], df.columns)
            if maybe_x is None:
                raise ValueError(f"Could not find X column in {csv_path}.")
            x_col = maybe_x
        
        if y_col not in df.columns:
            maybe_y = find_column([y_col, "Y", "POSITION Y", "POSITIONY"], df.columns)
            if maybe_y is None:
                raise ValueError(f"Could not find Y column in {csv_path}.")
            y_col = maybe_y
        
        frame_col = find_column(frame_col_candidates, df.columns)
        if frame_col is None:
            frame_col = find_column(["FRAME", "T", "POSITION_T"], df.columns)
        
        if brightness_col is None:
            brightness_col = find_column(DEFAULT_BRIGHTNESS_CANDIDATES, df.columns)
            if brightness_col is None:
                raise ValueError(f"Could not auto-detect a brightness/intensity column in {csv_path}.")
        else:
            if brightness_col not in df.columns:
                maybe_b = find_column([brightness_col], df.columns)
                if maybe_b is None:
                    raise ValueError(f"Brightness column '{brightness_col}' not found in {csv_path}.")
                brightness_col = maybe_b
        
        for col in [x_col, y_col, brightness_col]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        if frame_col and frame_col in df.columns:
            df = df.sort_values([track_col, frame_col], kind="mergesort").reset_index(drop=True)
        else:
            df = df.sort_values(track_col).reset_index(drop=True)
        
        tids = df[track_col].to_numpy()
        x_arr = df[x_col].to_numpy(dtype=float)
        y_arr = df[y_col].to_numpy(dtype=float)
        b_arr = df[brightness_col].to_numpy(dtype=float)
        
        same_track = tids[1:] == tids[:-1]
        dx = x_arr[1:] - x_arr[:-1]
        dy = y_arr[1:] - y_arr[:-1]
        step_size = np.sqrt(dx**2 + dy**2)
        step_brightness = 0.5 * (b_arr[:-1] + b_arr[1:])
        valid = same_track & np.isfinite(dx) & np.isfinite(dy) & np.isfinite(b_arr[:-1]) & np.isfinite(b_arr[1:])
        
        if not np.any(valid):
            return pd.DataFrame(columns=["source_file", "track_id", "step_index", "step_size", "brightness"])
        
        valid_idx = np.where(valid)[0]
        source = os.path.basename(csv_path)
        
        track_counts = pd.Series(tids).groupby(tids).cumcount().to_numpy()
        result = pd.DataFrame({
            "source_file": source,
            "track_id": tids[valid_idx],
            "step_index": track_counts[valid_idx].astype(int),
            "step_size": step_size[valid_idx],
            "brightness": step_brightness[valid_idx],
        })
        return result
    
    def compute_fixed_bins(xlim: Tuple[float, float], ylim: Tuple[float, float], bins_x: int, bins_y: int):
        x_edges = np.linspace(xlim[0], xlim[1], bins_x + 1)
        y_edges = np.linspace(ylim[0], ylim[1], bins_y + 1)
        return x_edges, y_edges
    
    def make_heatmap_with_cap(df: pd.DataFrame, x_col: str, y_col: str, x_edges: np.ndarray,
                               y_edges: np.ndarray, count_cap: float, log_counts: bool,
                               out_png: str, x_label: str = "Step size", y_label: str = "Brightness",
                               title: Optional[str] = None) -> None:
        if df.empty:
            raise ValueError("No data to plot for heatmap.")
        
        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        
        H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        
        if log_counts:
            norm = LogNorm(vmin=1.0, vmax=max(1.0, float(count_cap)))
            H_display = np.clip(H, 1.0, float(count_cap))
        else:
            norm = Normalize(vmin=0.0, vmax=float(count_cap))
            H_display = np.clip(H, 0.0, float(count_cap))
        
        plt.figure(figsize=(7, 6))
        mesh = plt.pcolormesh(x_edges, y_edges, H_display.T, shading="auto", norm=norm)
        
        over_mask = (H > float(count_cap))
        if np.any(over_mask):
            overlay = np.where(over_mask, 1.0, np.nan).T
            red_cmap = ListedColormap(["red"])
            plt.pcolormesh(x_edges, y_edges, overlay, shading="auto", cmap=red_cmap, vmin=0.0, vmax=1.0)
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if title:
            plt.title(title)
        
        cbar = plt.colorbar(mesh)
        cbar.set_label(f"Counts (capped at {int(count_cap)})" + (" [log]" if log_counts else ""))
        
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()
    
    outdir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(os.path.abspath(args.input_dir), args.output_dir)
    os.makedirs(outdir, exist_ok=True)
    
    # Search recursively for CSV files matching pattern
    csv_paths = [
        p for p in sorted(glob.glob(os.path.join(args.input_dir, "**", args.pattern), recursive=True))
        if not _should_ignore_path(p)
    ]
    if not csv_paths:
        raise SystemExit(f"No CSVs found in {args.input_dir!r} with pattern {args.pattern!r} (searched recursively).")
    
    all_rows = []
    for p in csv_paths:
        try:
            df_steps = load_and_compute_steps(
                csv_path=p,
                track_col=args.track_col,
                x_col=args.x_col,
                y_col=args.y_col,
                frame_col_candidates=args.frame_col,
                brightness_col=args.brightness_column,
            )
            if df_steps.empty:
                continue
            cond = infer_condition_from_filename(p, args.conditions)
            df_steps.insert(0, "condition", cond)
            all_rows.append(df_steps)
        except Exception as e:
            print(f"[WARN] Skipping {os.path.basename(p)} due to error: {e}")
    
    if not all_rows:
        raise SystemExit("No per-step data aggregated from the provided CSVs.")
    
    agg = pd.concat(all_rows, ignore_index=True)
    
    if args.stepsize_max is not None:
        agg = agg[agg["step_size"] < float(args.stepsize_max)].copy()
    
    all_csv = os.path.join(outdir, "steps_vs_brightness_all.csv")
    agg.to_csv(all_csv, index=False)
    print(f"[OK] Wrote {all_csv} with {len(agg)} rows from {len(csv_paths)} file(s).")
    
    xlim = (0.0, float(args.stepsize_max)) if args.stepsize_max is not None else (float(agg["step_size"].min()), float(agg["step_size"].max()))
    if args.ymin is not None and args.ymax is not None:
        ylim = (float(args.ymin), float(args.ymax))
    else:
        ylim = (float(agg["brightness"].min()), float(agg["brightness"].max()))
    
    x_edges, y_edges = compute_fixed_bins(xlim=xlim, ylim=ylim, bins_x=args.bins_x, bins_y=args.bins_y)
    
    for cond in args.conditions:
        sub = agg[agg["condition"].str.lower() == cond.lower()].copy()
        if sub.empty:
            print(f"[INFO] No rows for condition '{cond}', skipping.")
            continue
        cond_csv = os.path.join(outdir, f"steps_vs_brightness_{cond}.csv")
        sub.to_csv(cond_csv, index=False)
        print(f"[OK] Wrote {cond_csv} ({len(sub)} rows).")
        
        cond_png = os.path.join(outdir, f"heatmap_{cond}.png")
        try:
            make_heatmap_with_cap(
                sub,
                x_col="step_size",
                y_col="brightness",
                x_edges=x_edges,
                y_edges=y_edges,
                count_cap=args.count_cap,
                log_counts=args.log_counts,
                out_png=cond_png,
                title=f"{cond}: Brightness vs Step Size (< {args.stepsize_max})",
            )
            print(f"[OK] Wrote heatmap: {cond_png}")
        except Exception as e:
            print(f"[WARN] Could not generate heatmap for '{cond}': {e}")
    
    _save_pipeline_context(context_root, ["plot", "brightness_heatmap"], {
        "input_dir": str(context_root),
        "pattern": args.pattern,
        "track_col": args.track_col,
        "x_col": args.x_col,
        "y_col": args.y_col,
        "frame_col": list(args.frame_col) if args.frame_col is not None else None,
        "brightness_column": args.brightness_column,
        "conditions": list(args.conditions) if args.conditions is not None else None,
        "bins_x": int(args.bins_x),
        "bins_y": int(args.bins_y),
        "stepsize_max": float(args.stepsize_max),
        "count_cap": float(args.count_cap),
        "log_counts": bool(args.log_counts),
        "ymin": args.ymin,
        "ymax": args.ymax,
    })


def cmd_plot_brightness_map(args):
    """Brightness-colored track overlays - from tracks_brightness_map.py"""
    DEFAULT_BRIGHTNESS_CANDIDATES = ["MEAN_INTENSITY", "MEAN_INTENSITY_CH1", "INTENSITY", "MEDIAN_INTENSITY", "QUALITY"]
    context_root = Path(args.input_dir)
    plot_ctx = _context_params(context_root, ["plot", "brightness_map"])
    args.pattern = _resolve_param(_argv_has_flag("--pattern"), args.pattern, plot_ctx.get("pattern"), "*.csv")
    args.track_col = _resolve_param(_argv_has_flag("--track-col"), args.track_col, plot_ctx.get("track_col"), "TRACK_ID")
    args.x_col = _resolve_param(_argv_has_flag("--x-col"), args.x_col, plot_ctx.get("x_col"), "POSITION_X")
    args.y_col = _resolve_param(_argv_has_flag("--y-col"), args.y_col, plot_ctx.get("y_col"), "POSITION_Y")
    args.frame_col = _resolve_param(
        _argv_has_flag("--frame-col"),
        args.frame_col,
        plot_ctx.get("frame_col"),
        ["FRAME", "POSITION_T", "T"],
    )
    args.brightness_column = _resolve_param(
        _argv_has_flag("--brightness-column"),
        args.brightness_column,
        plot_ctx.get("brightness_column"),
        None,
    )
    args.dot_size = float(_resolve_param(_argv_has_flag("--dot-size"), args.dot_size, plot_ctx.get("dot_size"), 2.0))
    args.alpha = float(_resolve_param(_argv_has_flag("--alpha"), args.alpha, plot_ctx.get("alpha"), 0.7))
    args.draw_lines = bool(_resolve_param(
        _argv_has_flag("--draw-lines", "--no-draw-lines"),
        args.draw_lines,
        plot_ctx.get("draw_lines"),
        False,
    ))
    args.clip_low_quantile = float(_resolve_param(
        _argv_has_flag("--clip-low-quantile"),
        args.clip_low_quantile,
        plot_ctx.get("clip_low_quantile"),
        0.0,
    ))
    args.clip_high_quantile = float(_resolve_param(
        _argv_has_flag("--clip-high-quantile"),
        args.clip_high_quantile,
        plot_ctx.get("clip_high_quantile"),
        1.0,
    ))
    
    def find_column(candidates: List[str], columns: List[str]) -> Optional[str]:
        colset = {c.lower(): c for c in columns}
        for cand in candidates:
            if cand in columns:
                return cand
            if cand.lower() in colset:
                return colset[cand.lower()]
            for c in columns:
                if cand.lower() in c.lower():
                    return c
        return None
    
    def load_spots(csv_path: str, track_col: str, x_col: str, y_col: str,
                   frame_col_candidates: List[str], brightness_col: Optional[str]) -> pd.DataFrame:
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        
        if track_col not in df.columns:
            maybe = find_column([track_col, "TRACK_ID", "TRACKID", "TRACK", "particle", "PARTICLE"], df.columns)
            if maybe is None:
                raise ValueError(f"Could not find a track ID column in {csv_path}.")
            track_col = maybe
        
        if x_col not in df.columns:
            maybe_x = find_column([x_col, "X", "POSITION X", "POSITIONX"], df.columns)
            if maybe_x is None:
                raise ValueError(f"Could not find X column in {csv_path}.")
            x_col = maybe_x
        if y_col not in df.columns:
            maybe_y = find_column([y_col, "Y", "POSITION Y", "POSITIONY"], df.columns)
            if maybe_y is None:
                raise ValueError(f"Could not find Y column in {csv_path}.")
            y_col = maybe_y
        
        frame_col = find_column(frame_col_candidates, df.columns)
        if frame_col is None:
            frame_col = find_column(["FRAME", "POSITION_T", "T"], df.columns)
        
        if brightness_col is None:
            brightness_col = find_column(DEFAULT_BRIGHTNESS_CANDIDATES, df.columns)
            if brightness_col is None:
                raise ValueError(f"Could not auto-detect a brightness/intensity column in {csv_path}.")
        else:
            if brightness_col not in df.columns:
                maybe_b = find_column([brightness_col], df.columns)
                if maybe_b is None:
                    raise ValueError(f"Brightness column '{brightness_col}' not found in {csv_path}.")
                brightness_col = maybe_b
        
        for col in [x_col, y_col, brightness_col]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        cols = [track_col, x_col, y_col, brightness_col]
        if frame_col:
            cols.append(frame_col)
        
        out = df[cols].copy()
        new_names = ["track_id", "x", "y", "brightness"] + (["frame"] if frame_col else [])
        out.columns = new_names
        out.insert(0, "source_file", os.path.basename(csv_path))
        return out
    
    def plot_overlay(df: pd.DataFrame, out_png: str, draw_lines: bool, dot_size: float, alpha: float,
                     xlim: Tuple[float, float], ylim: Tuple[float, float], cmin: float, cmax: float,
                     title: Optional[str] = None):
        if df.empty:
            raise ValueError("No spots to plot.")
        
        xmin, xmax = xlim
        ymin, ymax = ylim
        
        plt.figure(figsize=(7, 7))
        ax = plt.gca()
        
        if draw_lines and "frame" in df.columns:
            segs = []
            for tid, d in df.groupby("track_id", dropna=True):
                d = d.dropna(subset=["x", "y"]).copy()
                if d.empty:
                    continue
                d = d.sort_values("frame")
                if len(d) < 2:
                    continue
                pts = d[["x", "y"]].to_numpy(dtype=float)
                segs.extend(np.stack([pts[:-1], pts[1:]], axis=1))
            if segs:
                lc = LineCollection(segs, linewidths=0.5, alpha=min(alpha, 0.3))
                ax.add_collection(lc)
        
        sc = ax.scatter(
            df["x"].to_numpy(),
            df["y"].to_numpy(),
            c=np.clip(df["brightness"].to_numpy(), cmin, cmax),
            s=float(dot_size),
            alpha=float(alpha),
        )
        
        cbar = plt.colorbar(sc)
        cbar.set_label(f"Brightness (clipped to [{cmin:.3g}, {cmax:.3g}])")
        
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()
    
    outdir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(os.path.abspath(args.input_dir), args.output_dir)
    os.makedirs(outdir, exist_ok=True)
    
    csv_paths = [
        p for p in sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
        if not _should_ignore_path(p)
    ]
    if not csv_paths:
        raise SystemExit(f"No CSVs found in {args.input_dir!r} matching {args.pattern!r}.")
    
    all_spots = []
    for p in csv_paths:
        try:
            spots = load_spots(
                csv_path=p,
                track_col=args.track_col,
                x_col=args.x_col,
                y_col=args.y_col,
                frame_col_candidates=args.frame_col,
                brightness_col=args.brightness_column,
            )
            all_spots.append(spots)
        except Exception as e:
            print(f"[WARN] Skipping {os.path.basename(p)}: {e}")
    
    if not all_spots:
        raise SystemExit("No valid spot rows aggregated.")
    
    df = pd.concat(all_spots, ignore_index=True)
    df = df.dropna(subset=["x", "y", "brightness"]).copy()
    
    xmin, xmax = float(df["x"].min()), float(df["x"].max())
    ymin, ymax = float(df["y"].min()), float(df["y"].max())
    
    lo_q = float(args.clip_low_quantile)
    hi_q = float(args.clip_high_quantile)
    if not (0.0 <= lo_q <= 1.0 and 0.0 <= hi_q <= 1.0 and lo_q < hi_q):
        raise SystemExit("clip quantiles must satisfy: 0 <= low < high <= 1")
    cmin = float(np.nanquantile(df["brightness"].to_numpy(), lo_q))
    cmax = float(np.nanquantile(df["brightness"].to_numpy(), hi_q))
    
    spots_csv = os.path.join(outdir, "spots_all.csv")
    df.to_csv(spots_csv, index=False)
    print(f"[OK] Wrote {spots_csv} with {len(df)} rows from {len(csv_paths)} file(s).")
    
    plot_overlay(
        df=df,
        out_png=os.path.join(outdir, "overlay_all.png"),
        draw_lines=args.draw_lines,
        dot_size=args.dot_size,
        alpha=args.alpha,
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
        cmin=cmin,
        cmax=cmax,
        title="All movies: brightness-colored spots",
    )
    print(f"[OK] Wrote {os.path.join(outdir, 'overlay_all.png')}")
    
    for fname, sub in df.groupby("source_file", dropna=True):
        safe = os.path.splitext(fname)[0].replace(" ", "_")
        out_png = os.path.join(outdir, f"overlay_{safe}.png")
        plot_overlay(
            df=sub,
            out_png=out_png,
            draw_lines=args.draw_lines,
            dot_size=args.dot_size,
            alpha=args.alpha,
            xlim=(xmin, xmax),
            ylim=(ymin, ymax),
            cmin=cmin,
            cmax=cmax,
            title=f"{fname}: brightness-colored spots",
        )
        print(f"[OK] Wrote {out_png}")
    
    _save_pipeline_context(context_root, ["plot", "brightness_map"], {
        "input_dir": str(context_root),
        "pattern": args.pattern,
        "track_col": args.track_col,
        "x_col": args.x_col,
        "y_col": args.y_col,
        "frame_col": list(args.frame_col) if args.frame_col is not None else None,
        "brightness_column": args.brightness_column,
        "dot_size": float(args.dot_size),
        "alpha": float(args.alpha),
        "draw_lines": bool(args.draw_lines),
        "clip_low_quantile": float(args.clip_low_quantile),
        "clip_high_quantile": float(args.clip_high_quantile),
    })

# ============================================================================
# UTILS SUBCOMMAND - Utility Operations
# ============================================================================
