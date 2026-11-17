"""Retired: brightness-vs-diffusion scatter, lifted from spt_pipeline.plotting."""

def brightness_d_scatter(diff_df: pd.DataFrame, out_png: Path, *,
                         brightness_col: str = "mass_per_frame_ff",
                         d_col: str = "D_fit_um2_s", stats_text: str = "") -> None:
    with _PLOT_LOCK:
        apply_house_style()
        bcol = brightness_col if brightness_col in diff_df.columns else "mass_per_frame"
        b = pd.to_numeric(diff_df.get(bcol), errors="coerce").to_numpy()
        d = pd.to_numeric(diff_df.get(d_col), errors="coerce").to_numpy()
        ok = np.isfinite(b) & np.isfinite(d) & (b > 0) & (d > 0)
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        ax.scatter(b[ok], d[ok], s=8, alpha=0.4, color=OKABE_ITO[0], edgecolors="none")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Brightness (mass/frame)"); ax.set_ylabel("D (µm²/s)")
        ax.set_title("Brightness vs diffusion")
        if stats_text:
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va="top", fontsize=8)
        _save(fig, out_png)


