# Pipeline Compatibility Guide

This document explains how the different pipeline commands work together and how files are automatically detected.

## Workflow Overview

The pipeline is designed so you can use a **single input folder** throughout the entire workflow. Files are automatically detected from subdirectories.

### Typical Workflow

```
1. track trackpy <input_folder>     → Creates {movie_name}/{movie_name}_tracks.csv
2. analyze brightness-mobility <input_folder>  → Finds all *_tracks.csv files recursively
3. plot steps-brightness <input_folder>        → Finds all CSV files recursively
4. plot brightness-heatmap <input_folder>      → Finds all CSV files recursively
```

## File Detection

### Automatic Recursive Search

All commands now **automatically search recursively** through subdirectories:

- **`analyze brightness-mobility`**: Recursively finds all `*.csv` files, prioritizing `*_tracks.csv` files
- **`plot steps-brightness`**: Recursively finds all CSV files matching the pattern
- **`plot brightness-heatmap`**: Recursively finds all CSV files matching the pattern
- **`plot brightness-map`**: Recursively finds all CSV files matching the pattern
- **`plot dwell`**: Recursively finds `*_tracks.csv` files in subdirectories

### File Structure After `track trackpy`

When you run `track trackpy` on a folder, it creates:

```
input_folder/
├── movie1/
│   ├── movie1_tracks.csv          ← Automatically detected by analyze/plot commands
│   ├── movie1_particles.csv
│   ├── movie1_IMSD.csv
│   ├── movie1_EMSD.csv
│   └── movie1_tracks_overlay.png
├── movie2/
│   ├── movie2_tracks.csv          ← Automatically detected
│   └── ...
├── groupnames.csv                  ← Created automatically for plot dwell
└── all_msds_histogram.png
```

All downstream commands automatically find these files without needing to specify individual paths.

## Column Name Compatibility

The pipeline automatically handles different column naming conventions:

### Track ID Column
- Recognizes: `track_id`, `TRACK_ID`, `TRACKID`, `TRACK`, `particle`, `PARTICLE`
- **Trackpy outputs use `particle`** - automatically detected
- **TrackMate outputs use `track_id`** - automatically detected

### Position Columns
- X: `x`, `X`, `POSITION_X`, `POSITION X`, `spot_x`
- Y: `y`, `Y`, `POSITION_Y`, `POSITION Y`, `spot_y`
- Frame: `frame`, `FRAME`, `T`, `POSITION_T`, `TIME`

### Brightness/Intensity Column
- Recognizes: `INTENSITY`, `MEAN_INTENSITY`, `MAX_INTENSITY`, `mass`, `MASS`
- **Trackpy outputs use `mass`** - automatically detected
- **TrackMate outputs use `MEAN_INTENSITY_CH1`** - automatically detected

## Command Compatibility Matrix

| Command | Input | Output | Compatible With |
|---------|-------|--------|-----------------|
| `track trackpy` | ND2/TIFF files | `{movie_name}_tracks.csv` in subdirs | ✅ `analyze brightness-mobility`<br>✅ `plot steps-brightness`<br>✅ `plot brightness-heatmap`<br>✅ `plot brightness-map`<br>✅ `plot dwell` |
| `track dwelltime` | ND2 files | `{movie_name}_filtered_tracks.csv` in subdirs | ✅ `analyze brightness-mobility`<br>✅ `plot steps-brightness` |
| `prep trackmate` | TrackMate CSVs | `Traj_{condition}_{rep}.csv` | ✅ `analyze brightness-mobility`<br>✅ `plot steps-brightness`<br>✅ `plot brightness-heatmap` |
| `analyze brightness-mobility` | Any trajectory CSV | `per_track_metrics.csv` | ✅ `plot steps-brightness` (uses input CSVs)<br>✅ `utils split-metrics` |

## Example: Complete Workflow

```bash
# 1. Track particles (creates subdirectories with _tracks.csv files)
spt-pipeline track trackpy /data/my_experiment --diam 7 --minmass 110

# 2. Analyze brightness-mobility (automatically finds all _tracks.csv files)
spt-pipeline analyze brightness-mobility /data/my_experiment --dt 0.03

# 3. Create visualizations (automatically finds all CSV files)
spt-pipeline plot steps-brightness /data/my_experiment
spt-pipeline plot brightness-heatmap /data/my_experiment

# 4. Plot dwell times (uses groupnames.csv created by track trackpy)
spt-pipeline plot dwell /data/my_experiment
```

**Note**: You use the same input folder (`/data/my_experiment`) for all commands. Files are automatically detected from subdirectories.

## Output Locations

All outputs are saved in the **input directory** or its subdirectories:

- **Tracking outputs**: `{input_folder}/{movie_name}/`
- **Analysis outputs**: `{input_folder}/{movie_name}/{filename}_analysis/`
- **Plot outputs**: Same directory as input files or specified `--output-dir`

This keeps everything organized in one place and makes it easy to find related files.

## Troubleshooting

### Files Not Found

If a command says "No files found":
1. Check that you're pointing to the correct input folder
2. Verify that the expected files exist (use `find` command to search)
3. Check file patterns - some commands look for specific patterns like `*_tracks.csv`

### Column Detection Errors

If you get "Missing required columns" errors:
1. Check that your CSV files have the expected columns
2. The pipeline auto-detects common column names, but very non-standard names might not be recognized
3. You can specify custom column names using `--x-col`, `--y-col`, etc. where available

### Groupnames.csv Issues

If `plot dwell` can't find files:
1. Make sure `groupnames.csv` exists in the input folder (created by `track trackpy`)
2. Check that the "folder" column in `groupnames.csv` matches actual folder names
3. The pipeline now searches recursively, so files should be found even if folder structure is slightly different

