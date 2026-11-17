"""Argparse definitions for the retired brightness/dwell commands.

Lifted verbatim out of ``spt_pipeline.legacy.main`` when the arm was retired.
Reference only -- these reference parser objects that no longer exist.
"""

def _retired_parser_fragments(subparsers, plot_sub, track_sub, analyze_sub, _add_bool_flag):
        track_dwell = track_sub.add_parser('dwelltime', help='Dwelltime analysis from ND2')
        track_dwell.add_argument('input_folder', help='Input folder')
        track_dwell.add_argument('--diam', type=int, default=7, help='Particle diameter')
        track_dwell.add_argument('--minmass', type=float, default=110, help='Minimum mass')
        track_dwell.add_argument('--fps', type=float, default=33.33, help='Frames per second (same as 1/frame_interval)')
        track_dwell.add_argument('--channel', type=int, default=0,
            help='0-indexed channel to analyze for multichannel movies (default: 0)')
        analyze_bm = analyze_sub.add_parser('brightness-mobility', help='Brightness-mobility analysis')
        analyze_bm.add_argument('input', help='CSV file or directory')
        analyze_bm.add_argument('-o', '--outdir', help='Output directory')
        analyze_bm.add_argument('--dt', type=float, default=1.0, help='Time per frame')
        analyze_bm.add_argument('--immobile-threshold-px', type=float, default=0.01)
        analyze_bm.add_argument('--display-step-cap-px', type=float, default=3.0)
        plot_sb = plot_sub.add_parser('steps-brightness', help='Step-brightness heatmaps and track overlays')
        plot_sb.add_argument('input_dir', help='Input directory')
        plot_sb.add_argument('--pattern', default='*.csv', help='File pattern')
        plot_sb.add_argument('--make', choices=['both','heatmaps','tracks'], default='both')
        plot_sb.add_argument('--stepsize-max', type=float, default=3.0)
        plot_sb.add_argument('--bins-x', type=int, default=150)
        plot_sb.add_argument('--bins-y', type=int, default=150)
        plot_sb.add_argument('--count-cap', type=int, default=300)
        plot_sb.add_argument('--units-label', default='pixels')
        plot_sb.add_argument('--line-width', type=float, default=0.7)
        plot_sb.add_argument('--min-track-length', type=int, default=10)
        plot_sb.add_argument('--brightness-col', default='MEAN_INTENSITY_CH1')
        plot_sb.add_argument('--x-col', default=None)
        plot_sb.add_argument('--y-col', default=None)
        plot_sb.add_argument('--workers', type=int, default=None)
        plot_dwell = plot_sub.add_parser('dwell', help='Dwell time plotting')
        plot_dwell.add_argument('input_folder', help='Input folder')
        plot_dwell.add_argument('--cutoff', type=float, default=4, help='Minimum dwell_time')
        plot_dwell.add_argument('--mass', type=float, default=None)
        plot_dwell.add_argument('--size', type=float, default=None)
        plot_dwell.add_argument('--ecc', type=float, default=None)
        plot_dwell.add_argument('--signal', type=float, default=None)
        plot_dwell.add_argument('--raw_mass', type=float, default=10000)
        plot_dwell.add_argument('--ep', type=float, default=None)
        plot_dwell.add_argument('--frame', type=float, default=None)
        plot_dwell.add_argument('--particle', type=float, default=None)
        plot_dwell.add_argument('--total_mass', type=float, default=None)
        plot_bh = plot_sub.add_parser('brightness-heatmap', help='Condition-pooled brightness-step heatmaps')
        plot_bh.add_argument('input_dir', help='Directory with TrackMate CSVs')
        plot_bh.add_argument('--pattern', default='*.csv', help='File pattern')
        plot_bh.add_argument('--output-dir', default='brightness_stepsize', help='Output directory')
        plot_bh.add_argument('--track-col', default='TRACK_ID', help='Track ID column')
        plot_bh.add_argument('--x-col', default='POSITION_X', help='X column')
        plot_bh.add_argument('--y-col', default='POSITION_Y', help='Y column')
        plot_bh.add_argument('--frame-col', nargs='*', default=['FRAME', 'POSITION_T', 'T'], help='Frame columns')
        plot_bh.add_argument('--brightness-column', default=None, help='Brightness column')
        plot_bh.add_argument('--conditions', nargs='*', default=['vector', 'HRAS', 'HK1', 'HKWT', '12V', '180S'], help='Condition tokens')
        plot_bh.add_argument('--bins-x', type=int, default=80, help='Bins along step axis')
        plot_bh.add_argument('--bins-y', type=int, default=80, help='Bins along brightness axis')
        plot_bh.add_argument('--stepsize-max', type=float, default=5.0, help='Max step size')
        plot_bh.add_argument('--count-cap', type=float, default=300.0, help='Count cap')
        plot_bh.add_argument('--ymin', type=float, default=None, help='Force Y min')
        plot_bh.add_argument('--ymax', type=float, default=None, help='Force Y max')
        plot_bmap = plot_sub.add_parser('brightness-map', help='Brightness-colored track overlays')
        plot_bmap.add_argument('input_dir', help='Directory with TrackMate CSVs')
        plot_bmap.add_argument('--pattern', default='*.csv', help='File pattern')
        plot_bmap.add_argument('--output-dir', default='tracks_brightness_map', help='Output directory')
        plot_bmap.add_argument('--track-col', default='TRACK_ID', help='Track ID column')
        plot_bmap.add_argument('--x-col', default='POSITION_X', help='X column')
        plot_bmap.add_argument('--y-col', default='POSITION_Y', help='Y column')
        plot_bmap.add_argument('--frame-col', nargs='*', default=['FRAME', 'POSITION_T', 'T'], help='Frame columns')
        plot_bmap.add_argument('--brightness-column', default=None, help='Brightness column')
        plot_bmap.add_argument('--dot-size', type=float, default=2.0, help='Scatter marker size')
        plot_bmap.add_argument('--alpha', type=float, default=0.7, help='Alpha')
        plot_bmap.add_argument('--clip-low-quantile', type=float, default=0.00, help='Lower quantile')
        plot_bmap.add_argument('--clip-high-quantile', type=float, default=1.00, help='Upper quantile')
        workflow_parser = subparsers.add_parser('workflow', help='High-level workflows')
        workflow_sub = workflow_parser.add_subparsers(dest='workflow_cmd')
        workflow_auto = workflow_sub.add_parser(
            'legacy-auto',
            aliases=['auto-legacy'],
            help='Legacy chain: analyze brightness-mobility, plot steps-brightness, plot dwell'
        )
        workflow_auto.add_argument('input_dir', help='Root input directory containing tracking results')
        workflow_auto.add_argument(
            '-o', '--outdir',
            help='Override output root for brightness-mobility (default: alongside input)'
        )
        workflow_auto.add_argument(
            '--dt', type=float, default=0.03,
            help='Time per frame (seconds) for brightness-mobility analysis'
        )
        workflow_auto.add_argument(
            '--immobile-threshold-px', type=float, default=0.01,
            help='Immobile step threshold (pixels) for brightness-mobility'
        )
        workflow_auto.add_argument(
            '--display-step-cap-px', type=float, default=3.0,
            help='Display cap for step sizes (pixels) in brightness-mobility outputs'
        )
        workflow_auto.add_argument('--pattern', default='*.csv', help='File pattern for CSVs')
        workflow_auto.add_argument(
            '--make', choices=['both', 'heatmaps', 'tracks'], default='both',
            help='What to generate for steps-brightness plots'
        )
        workflow_auto.add_argument(
            '--stepsize-max', type=float, default=3.0,
            help='Maximum step size (pixels) for heatmaps'
        )
        workflow_auto.add_argument(
            '--bins-x', type=int, default=150,
            help='Number of bins along step-size axis'
        )
        workflow_auto.add_argument(
            '--bins-y', type=int, default=150,
            help='Number of bins along brightness axis'
        )
        workflow_auto.add_argument(
            '--count-cap', type=int, default=300,
            help='Count cap for heatmap bins'
        )
        workflow_auto.add_argument(
            '--line-width', type=float, default=0.7,
            help='Line width for track overlays'
        )
        workflow_auto.add_argument(
            '--min-track-length', type=int, default=10,
            help='Minimum track length (frames) for overlays'
        )
        workflow_auto.add_argument(
            '--brightness-col', default='MEAN_INTENSITY_CH1',
            help='Brightness column for steps-brightness plots (TrackMate style)'
        )
        workflow_auto.add_argument(
            '--dwell-cutoff', type=float, default=4.0,
            help='Minimum dwell_time for dwell plots'
        )
                workflow_parser.print_help()


# Boolean-flag registrations for the retired parsers:
#    _add_bool_flag(analyze_bm, '--use-maxlag', 'use_maxlag', True, 'Use max-lag MSD')
#    _add_bool_flag(plot_sb, '--invert-lut-tracks', 'invert_lut_tracks', False, 'Invert LUT for track overlays')
#    _add_bool_flag(plot_sb, '--strip-datecodes', 'strip_datecodes', True, 'Strip date codes from inferred source names')
#    _add_bool_flag(plot_bh, '--log-counts', 'log_counts', False, 'Use log scaling')
#    _add_bool_flag(plot_bmap, '--draw-lines', 'draw_lines', False, 'Draw polylines')
