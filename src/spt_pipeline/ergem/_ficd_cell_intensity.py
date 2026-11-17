#!/usr/bin/env python3
"""Track-territory FICD expression measurement for ER-GEM experiments.

Trajectories with more than five steps are associated into candidate cellular
territories before the marker channel is consulted.  Each territory grows in
three-pixel rings while the new pixels remain intensity-compatible with the
accepted territory.  Compact puncta are excluded throughout.  The primary
per-cell value is the puncta-cleaned territory median minus the global
background mode, floored at zero.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nd2
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.signal import find_peaks, peak_prominences
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from skimage import filters, measure, morphology

ANALYSIS_VERSION = "3.1.0-adaptive-track-territory"


class TerritoryExclusion(RuntimeError):
    """Expected per-image exclusion, such as too few qualifying tracks."""


def um_to_px(value_um: float, pixel_um: float, minimum: int = 1) -> int:
    return max(minimum, int(round(value_um / pixel_um)))


def robust_mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    center = np.median(values)
    return float(1.4826 * np.median(np.abs(values - center)))


def remove_small(mask: np.ndarray, minimum_size: int) -> np.ndarray:
    mask = mask.astype(bool)
    if minimum_size <= 1:
        return mask
    labels = measure.label(mask)
    if labels.max() == 0:
        return mask
    counts = np.bincount(labels.ravel())
    keep = np.flatnonzero(counts >= int(minimum_size))
    keep = keep[keep != 0]
    return np.isin(labels, keep)


def metadata_value(text: str, pattern: str) -> str | float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return ""
    value = match.group(1).strip()
    try:
        return float(value)
    except ValueError:
        return value


def read_nd2_2d(path: Path) -> tuple[np.ndarray, float, str, dict]:
    with nd2.ND2File(path) as reader:
        image = np.asarray(reader.asarray())
        pixel_um = float(reader.voxel_size().x)
        sizes = dict(reader.sizes)
        description = reader.text_info.get("description", "")
        acquisition = {
            "acquisition_date": reader.text_info.get("date", ""),
            "camera_name": metadata_value(description, r"Camera Name:\s*([^\r\n]+)"),
            "exposure_ms": metadata_value(description, r"Exposure:\s*([\d.]+)\s*ms"),
            "laser_561_power_setting": metadata_value(
                description, r"ExW:561;\s*Power:\s*([\d.]+)"
            ),
            "conversion_gain": metadata_value(
                description, r"Conversion Gain:\s*([^\r\n]+)"
            ),
            "significant_bits": int(reader.attributes.bitsPerComponentSignificant),
        }

    projection = "none"
    if image.ndim > 2:
        leading_axes = tuple(range(image.ndim - 2))
        image = np.max(image, axis=leading_axes)
        projection = "max_over_" + "_".join(list(sizes)[:-2])
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image after projection, got {image.shape}")
    return image.astype(np.float64), pixel_um, projection, acquisition


def image_key(path: Path) -> str:
    return re.sub(r"^(Red_|Traj_)", "", path.stem, flags=re.IGNORECASE)


def load_trajectory(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    table = table.loc[:, ~table.columns.astype(str).str.startswith("Unnamed")]
    required = {"Trajectory", "Frame", "x", "y"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path.name} lacks columns: {sorted(missing)}")
    table = table.copy()
    for column in required:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    return table.dropna(subset=list(required))


def estimate_background_peak(
    image: np.ndarray,
    mode_hint: float | None,
    hint_window: float,
) -> dict:
    """Estimate the dominant camera/background peak from the full image.

    The lower side of the selected histogram peak estimates its spread without
    contamination from the positive fluorescence tail.
    """
    values = image[np.isfinite(image)].astype(float)
    if values.size < 100:
        raise ValueError("Not enough finite image pixels for background estimation")

    low = float(np.floor(np.percentile(values, 0.05)))
    high = float(np.ceil(np.percentile(values, 99.95)))
    if high <= low:
        high = low + 2.0
    bin_width = max(1.0, (high - low) / 1024.0)
    edges = np.arange(low - bin_width / 2, high + 1.5 * bin_width, bin_width)
    counts, edges = np.histogram(values, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    smooth_counts = ndi.gaussian_filter1d(counts.astype(float), sigma=1.5)

    peaks, props = find_peaks(
        smooth_counts,
        prominence=max(1.0, float(smooth_counts.max()) * 0.005),
    )
    if peaks.size == 0:
        peak_index = int(np.argmax(smooth_counts))
        prominence = float(smooth_counts[peak_index])
        selection = "global_mode"
    else:
        if mode_hint is not None:
            near = peaks[np.abs(centers[peaks] - mode_hint) <= hint_window]
        else:
            near = np.array([], dtype=int)
        if near.size:
            eligible = near[smooth_counts[near] >= 0.10 * smooth_counts.max()]
            if eligible.size == 0:
                eligible = near
            distance_penalty = 1.0 + np.abs(centers[eligible] - mode_hint)
            score = smooth_counts[eligible] / distance_penalty
            peak_index = int(eligible[np.argmax(score)])
            selection = "hinted_peak"
        else:
            cutoff = np.percentile(values, 60)
            eligible = peaks[centers[peaks] <= cutoff]
            if eligible.size == 0:
                eligible = peaks
            peak_index = int(eligible[np.argmax(smooth_counts[eligible])])
            selection = "dominant_lower_peak"
        prominence = float(peak_prominences(smooth_counts, [peak_index])[0][0])

    mode = float(centers[peak_index])
    lower_deviation = mode - values[values <= mode]
    lower_deviation = lower_deviation[
        (lower_deviation >= 0) & (lower_deviation <= max(20.0, hint_window))
    ]
    sigma = float(np.median(lower_deviation) / 0.67448975) if lower_deviation.size else float("nan")
    if not np.isfinite(sigma) or sigma < 0.5:
        local = values[np.abs(values - mode) <= max(5.0, hint_window / 3)]
        sigma = max(0.5, robust_mad(local))

    peak_fraction = float(np.mean(np.abs(values - mode) <= 2.0 * sigma))
    prominence_fraction = prominence / max(float(smooth_counts.max()), 1.0)
    qc = bool(
        np.isfinite(mode)
        and np.isfinite(sigma)
        and 0.5 <= sigma <= 30.0
        and peak_fraction >= 0.10
        and prominence_fraction >= 0.05
    )
    return {
        "mode": mode,
        "sigma": sigma,
        "peak_fraction": peak_fraction,
        "prominence_fraction": prominence_fraction,
        "selection": selection,
        "qc_pass": qc,
        "centers": centers,
        "counts": counts,
        "smooth_counts": smooth_counts,
    }


# Private v2 reference implementations remain below so historical behavior can
# be inspected during regression work.  The pipeline calls the v3 definitions
# later in this module; every retained v2 entry point is explicitly prefixed.
def _legacy_build_track_seeds(
    trajectories: pd.DataFrame,
    shape: tuple[int, int],
    pixel_um: float,
    minimum_steps_exclusive: int,
    density_sigma_um: float,
    seed_radius_um: float,
    bridge_radius_um: float,
    minimum_seed_area_um2: float,
    minimum_seed_tracks: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, float, dict[int, int]]:
    lengths = trajectories.groupby("Trajectory").size()
    steps = lengths - 1
    qualifying_ids = steps[steps > minimum_steps_exclusive].index
    qualifying = trajectories[trajectories["Trajectory"].isin(qualifying_ids)].copy()
    density = np.zeros(shape, dtype=float)
    if qualifying.empty:
        return np.zeros(shape, np.uint16), density, pd.DataFrame(), float("nan"), {}

    centroids = qualifying.groupby("Trajectory")[["x", "y"]].mean()
    for _, group in qualifying.groupby("Trajectory"):
        x = np.clip(np.rint(group["x"]).astype(int), 0, shape[1] - 1)
        y = np.clip(np.rint(group["y"]).astype(int), 0, shape[0] - 1)
        np.add.at(density, (y, x), 1.0 / len(group))

    density = ndi.gaussian_filter(
        density, sigma=um_to_px(density_sigma_um, pixel_um)
    )
    positive = density[density > 0]
    density_threshold = float(filters.threshold_otsu(positive))
    support = density > density_threshold
    support = morphology.dilation(
        support, morphology.disk(um_to_px(seed_radius_um, pixel_um))
    )
    support = morphology.closing(
        support, morphology.disk(um_to_px(bridge_radius_um, pixel_um))
    )
    support = ndi.binary_fill_holes(support)
    support = remove_small(
        support, int(round(minimum_seed_area_um2 / pixel_um**2))
    )

    initial = measure.label(support)
    counts: dict[int, set] = {}
    _, nearest_initial = ndi.distance_transform_edt(
        initial == 0, return_indices=True
    )
    for trajectory_id, row in centroids.iterrows():
        x = int(np.clip(round(row["x"]), 0, shape[1] - 1))
        y = int(np.clip(round(row["y"]), 0, shape[0] - 1))
        label_id = int(initial[y, x])
        if label_id == 0:
            label_id = int(
                initial[
                    nearest_initial[0, y, x],
                    nearest_initial[1, y, x],
                ]
            )
        if label_id > 0:
            counts.setdefault(label_id, set()).add(trajectory_id)

    keep_labels = [
        label_id for label_id, ids in counts.items() if len(ids) >= minimum_seed_tracks
    ]
    filtered = np.isin(initial, keep_labels)
    seed_labels = measure.label(filtered).astype(np.uint16)
    track_counts: dict[int, int] = {}
    _, nearest_seed = ndi.distance_transform_edt(
        seed_labels == 0, return_indices=True
    )
    for label_id in range(1, int(seed_labels.max()) + 1):
        n_tracks = 0
        for _, row in centroids.iterrows():
            x = int(np.clip(round(row["x"]), 0, shape[1] - 1))
            y = int(np.clip(round(row["y"]), 0, shape[0] - 1))
            assigned_label = int(seed_labels[y, x])
            if assigned_label == 0:
                assigned_label = int(
                    seed_labels[
                        nearest_seed[0, y, x],
                        nearest_seed[1, y, x],
                    ]
                )
            if assigned_label == label_id:
                n_tracks += 1
        track_counts[label_id] = n_tracks
    return seed_labels, density, centroids, density_threshold, track_counts


def _legacy_grow_track_seeds(
    image: np.ndarray,
    seed_labels: np.ndarray,
    pixel_um: float,
    background: dict,
    growth_z: float,
    detection_z: float,
    detection_fraction: float,
    probe_radius_um: float,
    signal_gap_radius_um: float,
    no_signal_radius_um: float,
    perimeter_radius_um: float,
    perimeter_z_max: float,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    mode = float(background["mode"])
    sigma = float(background["sigma"])
    clipped = np.minimum(image, np.percentile(image, 99.5))
    smooth = ndi.gaussian_filter(clipped, sigma=max(1.0, um_to_px(0.25, pixel_um)))
    signal_domain = smooth > mode + growth_z * sigma
    signal_domain = morphology.closing(
        signal_domain,
        morphology.disk(um_to_px(signal_gap_radius_um, pixel_um)),
    )
    signal_domain = remove_small(signal_domain, int(round(2.0 / pixel_um**2)))

    detected_union = np.zeros_like(seed_labels, dtype=bool)
    no_signal_union = np.zeros_like(seed_labels, dtype=bool)
    seed_status: list[dict] = []
    for seed_id in range(1, int(seed_labels.max()) + 1):
        seed = seed_labels == seed_id
        probe = morphology.dilation(
            seed, morphology.disk(um_to_px(probe_radius_um, pixel_um))
        )
        probe_values = smooth[probe]
        probe_median = float(np.median(probe_values))
        probe_z = (probe_median - mode) / sigma
        probe_fraction = float(np.mean(probe_values > mode + growth_z * sigma))
        detected = bool(
            probe_z >= detection_z and probe_fraction >= detection_fraction
        )

        if detected and np.any(seed & signal_domain):
            grown = ndi.binary_propagation(seed & signal_domain, mask=signal_domain)
            grown = morphology.closing(
                grown,
                morphology.disk(um_to_px(signal_gap_radius_um, pixel_um)),
            )
            grown = ndi.binary_fill_holes(grown)
            detected_union |= grown
            source_mask = grown
        else:
            detected = False
            source_mask = morphology.dilation(
                seed,
                morphology.disk(um_to_px(no_signal_radius_um, pixel_um)),
            )
            source_mask = morphology.closing(
                source_mask,
                morphology.disk(um_to_px(signal_gap_radius_um, pixel_um)),
            )
            source_mask = ndi.binary_fill_holes(source_mask)
            no_signal_union |= source_mask

        perimeter = morphology.dilation(
            source_mask,
            morphology.disk(um_to_px(perimeter_radius_um, pixel_um)),
        ) & ~source_mask
        perimeter_values = smooth[perimeter]
        perimeter_median = (
            float(np.median(perimeter_values)) if perimeter_values.size else float("nan")
        )
        perimeter_z = (perimeter_median - mode) / sigma
        perimeter_found = bool(
            detected
            and perimeter_values.size >= 100
            and np.isfinite(perimeter_z)
            and perimeter_z <= perimeter_z_max
        )
        seed_status.append(
            {
                "seed_id": seed_id,
                "marker_detected": detected,
                "probe_median": probe_median,
                "probe_z": probe_z,
                "probe_signal_fraction": probe_fraction,
                "perimeter_found": perimeter_found,
                "perimeter_median": perimeter_median,
                "perimeter_z": perimeter_z,
                "seed_area_px": int(np.count_nonzero(seed)),
                "source_area_px": int(np.count_nonzero(source_mask)),
            }
        )

    combined = detected_union | no_signal_union
    labels = measure.label(combined).astype(np.uint16)
    statuses: list[dict] = []
    seed_status_df = pd.DataFrame(seed_status)
    for cell_id in range(1, int(labels.max()) + 1):
        cell = labels == cell_id
        overlapping_seeds = np.unique(seed_labels[cell])
        overlapping_seeds = overlapping_seeds[overlapping_seeds > 0]
        sub = seed_status_df[seed_status_df["seed_id"].isin(overlapping_seeds)]
        detected = bool(sub["marker_detected"].any()) if not sub.empty else False
        perimeter_found = bool(sub["perimeter_found"].any()) if not sub.empty else False
        statuses.append(
            {
                "cell_id": cell_id,
                "marker_detected": detected,
                "marker_status": (
                    "detected_with_perimeter"
                    if detected and perimeter_found
                    else "detected_no_complete_perimeter"
                    if detected
                    else "no_signal"
                ),
                "perimeter_found": perimeter_found,
                "probe_z": float(sub["probe_z"].max()) if not sub.empty else float("nan"),
                "probe_signal_fraction": (
                    float(sub["probe_signal_fraction"].max())
                    if not sub.empty
                    else float("nan")
                ),
                "perimeter_median": (
                    float(sub["perimeter_median"].median())
                    if not sub.empty
                    else float("nan")
                ),
                "perimeter_z": (
                    float(sub["perimeter_z"].median())
                    if not sub.empty
                    else float("nan")
                ),
                "track_seed_area_px": (
                    int(sub["seed_area_px"].sum()) if not sub.empty else 0
                ),
            }
        )
    return labels, pd.DataFrame(statuses), smooth, signal_domain


def _legacy_detect_compact_puncta(
    image: np.ndarray, labels: np.ndarray, pixel_um: float
) -> np.ndarray:
    dog = ndi.gaussian_filter(image, 1.0) - ndi.gaussian_filter(image, 5.0)
    puncta = np.zeros_like(labels, dtype=bool)
    maximum_area_px = max(10, int(round(5.0 / pixel_um**2)))
    for cell_id in range(1, int(labels.max()) + 1):
        cell = labels == cell_id
        values = dog[cell]
        if values.size < 20:
            continue
        threshold = float(np.median(values) + 5.0 * max(robust_mad(values), 0.25))
        candidates = measure.label((dog > threshold) & cell)
        for region in measure.regionprops(candidates):
            if 3 <= region.area <= maximum_area_px:
                puncta[candidates == region.label] = True
    return morphology.dilation(puncta, morphology.disk(1))


def clipped_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    low, high = np.percentile(values, [1, 99])
    retained = values[(values >= low) & (values <= high)]
    return float(np.mean(retained)) if retained.size else float(np.mean(values))


def _legacy_measure_cells(
    image: np.ndarray,
    labels: np.ndarray,
    seed_labels: np.ndarray,
    statuses: pd.DataFrame,
    background: dict,
    pixel_um: float,
    image_name: str,
    projection: str,
    seed_track_counts: dict[int, int],
) -> tuple[pd.DataFrame, np.ndarray]:
    puncta = _legacy_detect_compact_puncta(image, labels, pixel_um)
    rows: list[dict] = []
    bg_mode = float(background["mode"])
    bg_sigma = float(background["sigma"])
    status_map = statuses.set_index("cell_id").to_dict("index") if not statuses.empty else {}

    for region in measure.regionprops(labels):
        cell_id = int(region.label)
        mask = labels == cell_id
        measurement_mask = mask & ~puncta
        if np.count_nonzero(measurement_mask) < 100:
            measurement_mask = mask.copy()
        values = image[measurement_mask]
        all_values = image[mask]
        status = status_map.get(cell_id, {})

        mean_value = float(np.mean(values))
        mean_all = float(np.mean(all_values))
        median_value = float(np.median(values))
        corrected_mean = mean_value - bg_mode
        corrected_all = mean_all - bg_mode
        corrected_median = median_value - bg_mode
        corrected_clipped = clipped_mean(values) - bg_mode
        puncta_fraction = float(np.count_nonzero(puncta & mask) / max(1, np.count_nonzero(mask)))
        minr, minc, maxr, maxc = region.bbox
        touches_border = bool(
            minr == 0
            or minc == 0
            or maxr == image.shape[0]
            or maxc == image.shape[1]
        )

        seed_overlap = seed_labels[mask]
        seed_ids = np.unique(seed_overlap[seed_overlap > 0])
        n_seed_tracks = int(sum(seed_track_counts.get(int(i), 0) for i in seed_ids))
        seed_area_px = int(status.get("track_seed_area_px", 0))
        expansion_ratio = float(region.area / max(seed_area_px, 1))
        marker_detected = bool(status.get("marker_detected", False))
        marker_status = str(status.get("marker_status", "no_signal"))

        technical_flags: list[str] = []
        phenotype_flags: list[str] = []
        if not background["qc_pass"]:
            technical_flags.append("uncertain_background_peak")
        if n_seed_tracks == 0:
            technical_flags.append("no_seed_tracks")
        if values.size < 100:
            technical_flags.append("insufficient_measurement_pixels")
        if puncta_fraction > 0.20:
            technical_flags.append("puncta_dominated")
        if region.area > 0.85 * image.size:
            technical_flags.append("growth_spans_image")
        if touches_border:
            phenotype_flags.append("touches_border")
        if not marker_detected:
            phenotype_flags.append("no_marker_signal")
        elif not bool(status.get("perimeter_found", False)):
            phenotype_flags.append("incomplete_marker_perimeter")

        rows.append(
            {
                "image": image_name,
                "analysis_version": ANALYSIS_VERSION,
                "cell_id": cell_id,
                "segmentation_source": "track_seeded_growth",
                "projection": projection,
                "pixel_size_um": pixel_um,
                "area_um2": float(region.area * pixel_um**2),
                "centroid_x_px": float(region.centroid[1]),
                "centroid_y_px": float(region.centroid[0]),
                "ficd_bgsub_mean": corrected_mean,
                "ficd_bgsub_mean_all_pixels": corrected_all,
                "ficd_bgsub_clipped_mean": corrected_clipped,
                "ficd_bgsub_median": corrected_median,
                "ficd_bgsub_mean_nonnegative": max(0.0, corrected_mean),
                "cell_raw_mean": mean_value,
                "cell_raw_median": median_value,
                "background_median": bg_mode,
                "background_mad": bg_sigma,
                "background_source": "global_histogram_peak",
                "background_n_pixels": int(image.size),
                "background_peak": bg_mode,
                "background_sigma": bg_sigma,
                "background_peak_fraction": float(background["peak_fraction"]),
                "background_peak_prominence": float(background["prominence_fraction"]),
                "background_peak_qc": bool(background["qc_pass"]),
                "measurement_n_pixels": int(values.size),
                "puncta_fraction": puncta_fraction,
                "bright_region_fraction": float(
                    np.mean(all_values > bg_mode + 3.0 * bg_sigma)
                ),
                "cell_p95": float(np.percentile(values, 95)),
                "cell_p99": float(np.percentile(values, 99)),
                "signal_to_background_noise": corrected_mean / bg_sigma,
                "robust_cv": robust_mad(values) / max(abs(corrected_median), bg_sigma),
                "marker_detected": marker_detected,
                "marker_status": marker_status,
                "probe_z": status.get("probe_z", float("nan")),
                "probe_signal_fraction": status.get(
                    "probe_signal_fraction", float("nan")
                ),
                "perimeter_found": bool(status.get("perimeter_found", False)),
                "perimeter_median": status.get("perimeter_median", float("nan")),
                "perimeter_z": status.get("perimeter_z", float("nan")),
                "track_seed_area_um2": float(seed_area_px * pixel_um**2),
                "growth_expansion_ratio": expansion_ratio,
                "n_seed_trajectories": n_seed_tracks,
                "touches_border": touches_border,
                "qc_pass": len(technical_flags) == 0,
                "qc_flags": ";".join(technical_flags),
                "phenotype_flags": ";".join(phenotype_flags),
            }
        )
    return pd.DataFrame(rows), puncta


def assign_trajectories(
    trajectories: pd.DataFrame,
    labels: np.ndarray,
    cells: pd.DataFrame,
    pixel_um: float,
    image_name: str,
    maximum_distance_um: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trajectories.empty:
        return pd.DataFrame(), trajectories.copy()

    work = trajectories.copy()
    x = np.clip(np.rint(work["x"]).astype(int), 0, labels.shape[1] - 1)
    y = np.clip(np.rint(work["y"]).astype(int), 0, labels.shape[0] - 1)
    work["localization_cell_id"] = labels[y, x].astype(int)

    if labels.max() > 0:
        distance, nearest = ndi.distance_transform_edt(labels == 0, return_indices=True)
        nearest_y = nearest[0, y, x]
        nearest_x = nearest[1, y, x]
        nearest_labels = labels[nearest_y, nearest_x].astype(int)
        nearest_distance_um = distance[y, x] * pixel_um
        close = (
            (work["localization_cell_id"].to_numpy() == 0)
            & (nearest_distance_um <= maximum_distance_um)
        )
        work.loc[close, "localization_cell_id"] = nearest_labels[close]

    summaries: list[dict] = []
    for trajectory_id, group in work.groupby("Trajectory", sort=True):
        assigned = group.loc[group["localization_cell_id"] > 0, "localization_cell_id"]
        if assigned.empty:
            cell_id = 0
            fraction = 0.0
        else:
            counts = assigned.value_counts()
            cell_id = int(counts.index[0])
            fraction = float(counts.iloc[0] / len(group))
        summaries.append(
            {
                "image": image_name,
                "Trajectory": trajectory_id,
                "n_localizations": int(len(group)),
                "n_steps": int(max(0, len(group) - 1)),
                "first_frame": float(group["Frame"].min()),
                "last_frame": float(group["Frame"].max()),
                "centroid_x_px": float(group["x"].mean()),
                "centroid_y_px": float(group["y"].mean()),
                "cell_id": cell_id,
                "assignment_fraction": fraction,
                "assignment_qc": (
                    "assigned"
                    if fraction >= 0.8
                    else "ambiguous"
                    if fraction > 0
                    else "unassigned"
                ),
            }
        )

    summary = pd.DataFrame(summaries)
    cell_columns = [
        "cell_id",
        "analysis_version",
        "ficd_bgsub_intensity",
        "ficd_bgsub_mean",
        "ficd_bgsub_mean_all_pixels",
        "ficd_bgsub_clipped_mean",
        "ficd_bgsub_median",
        "ficd_bgsub_mean_nonnegative",
        "marker_detected",
        "marker_status",
        "background_peak",
        "background_sigma",
        "perimeter_found",
        "signal_estimator",
        "growth_stop_reason",
        "warning_flags",
        "qc_pass",
        "qc_flags",
        "phenotype_flags",
    ]
    if not cells.empty:
        summary = summary.merge(cells[cell_columns], on="cell_id", how="left")
    for column in cell_columns[1:]:
        if column not in summary:
            summary[column] = np.nan

    appended = [
        "Trajectory",
        "cell_id",
        "assignment_fraction",
        "assignment_qc",
    ] + cell_columns[1:]
    work = work.merge(summary[appended], on="Trajectory", how="left")
    return summary, work


def load_manual_labels(mask_dir: Path | None, image_path: Path) -> np.ndarray | None:
    if mask_dir is None:
        return None
    for suffix in ("_labels.tif", "_labels.tiff", "_mask.tif"):
        path = mask_dir / f"{image_path.stem}{suffix}"
        if path.exists():
            labels = np.asarray(tifffile.imread(path))
            if labels.ndim != 2:
                raise ValueError(f"Manual mask must be 2D: {path}")
            if labels.dtype == bool or set(np.unique(labels)).issubset({0, 1}):
                labels = measure.label(labels > 0)
            return labels.astype(np.uint16)
    return None


def _legacy_save_qc_overlay(
    path: Path,
    image: np.ndarray,
    seed_labels: np.ndarray,
    labels: np.ndarray,
    puncta: np.ndarray,
    density: np.ndarray,
    background: dict,
    cells: pd.DataFrame,
) -> None:
    low, high = np.percentile(image, [1, 99.7])
    fig, axes = plt.subplots(1, 4, figsize=(19, 5), constrained_layout=True)
    axes[0].imshow(image, cmap="gray", vmin=low, vmax=high)
    axes[0].set_title("Raw marker")

    axes[1].imshow(image, cmap="gray", vmin=low, vmax=high)
    if seed_labels.max() > 0:
        axes[1].contour(seed_labels > 0, [0.5], colors="yellow", linewidths=0.8)
    if np.any(density):
        scale = max(float(np.percentile(density[density > 0], 99)), 1e-12)
        axes[1].imshow(density, cmap="magma", alpha=np.clip(density / scale, 0, 0.65))
    axes[1].set_title(">5-step density and seeds")

    axes[2].imshow(image, cmap="gray", vmin=low, vmax=high)
    if labels.max() > 0:
        axes[2].contour(labels > 0, [0.5], colors="cyan", linewidths=0.9)
    if np.any(puncta):
        axes[2].imshow(
            np.ma.masked_where(~puncta, puncta),
            cmap="autumn",
            alpha=0.6,
            interpolation="nearest",
        )
    for row in cells.itertuples():
        axes[2].text(
            row.centroid_x_px,
            row.centroid_y_px,
            f"{row.cell_id}\n{row.marker_status}",
            color="white",
            fontsize=7,
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1},
        )
    axes[2].set_title("Measurement masks; puncta red")

    centers = background["centers"]
    axes[3].plot(centers, background["counts"], color="0.65", lw=1, label="pixels")
    axes[3].plot(
        centers,
        background["smooth_counts"],
        color="black",
        lw=1.5,
        label="smoothed",
    )
    axes[3].axvline(background["mode"], color="royalblue", lw=1.5, label="background")
    axes[3].axvline(
        background["mode"] + background.get("growth_z", 1.5) * background["sigma"],
        color="darkorange",
        lw=1.5,
        label="growth threshold",
    )
    axes[3].set_xlim(
        max(centers.min(), background["mode"] - 8 * background["sigma"]),
        min(centers.max(), background["mode"] + 20 * background["sigma"]),
    )
    axes[3].set_xlabel("Pixel intensity")
    axes[3].set_ylabel("Count")
    axes[3].set_title("Global intensity histogram")
    axes[3].legend(fontsize=7)

    for axis in axes[:3]:
        axis.set_xlim(0, image.shape[1])
        axis.set_ylim(image.shape[0], 0)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(path.stem.replace("_qc", ""), fontsize=10)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def discover_files(args: argparse.Namespace) -> tuple[list[Path], dict[str, Path]]:
    red_paths: list[Path] = []
    for item in args.red:
        path = Path(item)
        if path.is_dir():
            red_paths.extend(sorted(path.glob("Red_*.nd2")))
        elif path.is_file():
            red_paths.append(path)
        else:
            raise FileNotFoundError(item)

    # Preserve staged symlink basenames. Resolving Red_<movie_id>.nd2 to its
    # original target name breaks pairing with Traj_<movie_id>.csv.
    red_paths = sorted(set(path.absolute() for path in red_paths))
    if not red_paths:
        raise FileNotFoundError("No Red_*.nd2 files found")

    trajectory_paths: list[Path] = []
    for item in args.trajectories:
        path = Path(item)
        if path.is_dir():
            trajectory_paths.extend(sorted(path.glob("Traj_*.csv")))
        elif path.is_file():
            trajectory_paths.append(path)
        else:
            raise FileNotFoundError(item)
    trajectory_map = {
        image_key(path): path.absolute() for path in trajectory_paths
    }
    return red_paths, trajectory_map


def build_track_seeds(
    trajectories: pd.DataFrame,
    shape: tuple[int, int],
    pixel_um: float,
    minimum_steps_exclusive: int,
    density_sigma_um: float,
    seed_radius_um: float,
    bridge_radius_um: float,
    minimum_seed_area_um2: float,
    minimum_seed_tracks: int,
    cluster_radius_um: float,
    minimum_track_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, float, dict[int, int]]:
    """Associate qualifying tracks first, then rasterize each cell territory.

    This is a compact DBSCAN-like association implemented with SciPy.  Track
    centroids without local support are discarded as outliers.  Disconnected
    ER subregions may retain one territory ID and subsequently grow toward one
    another without being filled by a convex hull.
    """
    lengths = trajectories.groupby("Trajectory").size()
    qualifying_ids = lengths.index[(lengths - 1) > minimum_steps_exclusive]
    qualifying = trajectories[trajectories["Trajectory"].isin(qualifying_ids)].copy()
    density = np.zeros(shape, dtype=float)
    if qualifying.empty:
        return np.zeros(shape, np.uint16), density, pd.DataFrame(), float("nan"), {}

    centroids = qualifying.groupby("Trajectory")[["x", "y"]].mean()
    coords_um = centroids[["x", "y"]].to_numpy(float) * pixel_um
    tree = cKDTree(coords_um)
    neighbor_lists = tree.query_ball_point(coords_um, r=cluster_radius_um)
    core = np.array(
        [len(neighbors) >= minimum_track_neighbors for neighbors in neighbor_lists],
        dtype=bool,
    )
    core_indices = np.flatnonzero(core)
    if core_indices.size == 0:
        return np.zeros(shape, np.uint16), density, pd.DataFrame(), float("nan"), {}

    core_tree = cKDTree(coords_um[core_indices])
    pairs = core_tree.query_pairs(cluster_radius_um, output_type="ndarray")
    if pairs.size:
        rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(core_indices.size)])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(core_indices.size)])
    else:
        rows = cols = np.arange(core_indices.size)
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(core_indices.size, core_indices.size),
    )
    _, core_components = connected_components(graph, directed=False)

    cluster = np.full(len(centroids), -1, dtype=int)
    cluster[core_indices] = core_components
    distances, nearest_core = core_tree.query(coords_um, k=1)
    border = (~core) & (distances <= cluster_radius_um)
    cluster[border] = core_components[nearest_core[border]]

    raw_counts = pd.Series(cluster[cluster >= 0]).value_counts()
    retained = sorted(raw_counts[raw_counts >= minimum_seed_tracks].index.tolist())
    remap = {old: new for new, old in enumerate(retained, start=1)}
    territory_ids = np.array([remap.get(value, 0) for value in cluster], dtype=int)
    centroids = centroids.copy()
    centroids["territory_id"] = territory_ids
    centroids = centroids[centroids["territory_id"] > 0]
    if centroids.empty:
        return np.zeros(shape, np.uint16), density, centroids, float("nan"), {}

    trajectory_to_territory = centroids["territory_id"].to_dict()
    qualifying["territory_id"] = qualifying["Trajectory"].map(trajectory_to_territory).fillna(0).astype(int)
    qualifying = qualifying[qualifying["territory_id"] > 0]
    sigma_px = um_to_px(density_sigma_um, pixel_um)
    seed_radius_px = um_to_px(seed_radius_um, pixel_um)
    bridge_radius_px = um_to_px(bridge_radius_um, pixel_um)
    minimum_area_px = max(1, int(round(minimum_seed_area_um2 / pixel_um**2)))

    masks: dict[int, np.ndarray] = {}
    raw_masks: dict[int, np.ndarray] = {}
    thresholds: list[float] = []
    for territory_id, territory_tracks in qualifying.groupby("territory_id"):
        local_density = np.zeros(shape, dtype=float)
        raw = np.zeros(shape, dtype=bool)
        for _, group in territory_tracks.groupby("Trajectory"):
            x = np.clip(np.rint(group["x"]).astype(int), 0, shape[1] - 1)
            y = np.clip(np.rint(group["y"]).astype(int), 0, shape[0] - 1)
            np.add.at(local_density, (y, x), 1.0 / len(group))
            raw[y, x] = True
        local_density = ndi.gaussian_filter(local_density, sigma=sigma_px)
        density += local_density
        positive = local_density[local_density > 0]
        threshold = (
            float(filters.threshold_otsu(positive))
            if positive.size > 1 and np.ptp(positive) > 0
            else float(positive.min()) if positive.size else 0.0
        )
        thresholds.append(threshold)
        support = local_density >= threshold
        support |= morphology.dilation(raw, morphology.disk(seed_radius_px))
        support = morphology.closing(support, morphology.disk(bridge_radius_px))
        support = ndi.binary_fill_holes(support)
        support = remove_small(support, minimum_area_px)
        masks[int(territory_id)] = support
        raw_masks[int(territory_id)] = raw

    # Seed masks from different candidate cells are never allowed to overlap.
    ids = sorted(masks)
    stack = np.stack([masks[i] for i in ids])
    overlap = stack.sum(axis=0) > 1
    if np.any(overlap):
        distances = np.stack([ndi.distance_transform_edt(~raw_masks[i]) for i in ids])
        winner = np.argmin(distances, axis=0)
        for index, territory_id in enumerate(ids):
            masks[territory_id][overlap] = winner[overlap] == index

    labels = np.zeros(shape, dtype=np.uint16)
    for territory_id in ids:
        labels[masks[territory_id]] = territory_id
    track_counts = {
        int(territory_id): int((centroids["territory_id"] == territory_id).sum())
        for territory_id in ids
    }
    return labels, density, centroids, float(np.median(thresholds)), track_counts


def detect_compact_puncta(
    image: np.ndarray,
    pixel_um: float,
    background: dict,
    maximum_area_um2: float,
) -> np.ndarray:
    """Detect compact locally bright foci without censoring diffuse signal."""
    dog = ndi.gaussian_filter(image, 1.0) - ndi.gaussian_filter(image, 5.0)
    finite = dog[np.isfinite(dog)]
    center = float(np.median(finite))
    spread = max(robust_mad(finite), 0.25)
    bright = image > float(background["mode"]) + 3.0 * float(background["sigma"])
    candidates = measure.label((dog > center + 5.0 * spread) & bright)
    maximum_area_px = max(10, int(round(maximum_area_um2 / pixel_um**2)))
    puncta = np.zeros_like(image, dtype=bool)
    for region in measure.regionprops(candidates):
        perimeter = max(float(region.perimeter), 1.0)
        circularity = float(4.0 * np.pi * region.area / perimeter**2)
        if 3 <= region.area <= maximum_area_px and circularity >= 0.12:
            puncta[candidates == region.label] = True
    return morphology.dilation(puncta, morphology.disk(1))


def grow_track_seeds(
    image: np.ndarray,
    seed_labels: np.ndarray,
    background: dict,
    puncta: np.ndarray,
    ring_width_px: int,
    ring_shift_rsd: float,
    ring_incompatible_fraction: float,
    maximum_area_ratio: float,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Grow all territories simultaneously using robust ring compatibility."""
    territory_ids = [int(i) for i in np.unique(seed_labels) if i > 0]
    masks = {i: (seed_labels == i) & ~puncta for i in territory_ids}
    initial_areas = {i: int(np.count_nonzero(masks[i])) for i in territory_ids}
    active = {i: True for i in territory_ids}
    stop_reason = {i: "" for i in territory_ids}
    accepted_steps = {i: 0 for i in territory_ids}
    collision_seen = {i: False for i in territory_ids}
    rejected_rings = np.zeros_like(seed_labels, dtype=bool)
    history: list[dict] = []
    footprint = morphology.disk(max(1, int(ring_width_px)))
    maximum_iterations = max(image.shape) // max(1, ring_width_px) + 2

    for iteration in range(1, maximum_iterations + 1):
        active_ids = [i for i in territory_ids if active[i]]
        if not active_ids:
            break
        occupied = np.zeros_like(seed_labels, dtype=bool)
        for mask in masks.values():
            occupied |= mask

        candidates: dict[int, np.ndarray] = {}
        for territory_id in active_ids:
            ring = morphology.dilation(masks[territory_id], footprint)
            ring &= ~occupied
            ring &= ~puncta
            candidates[territory_id] = ring

        candidate_stack = np.stack([candidates[i] for i in active_ids])
        contested = candidate_stack.sum(axis=0) > 1
        if np.any(contested):
            distances = np.stack(
                [ndi.distance_transform_edt(~masks[i]) for i in active_ids]
            )
            winner = np.argmin(distances, axis=0)
            for index, territory_id in enumerate(active_ids):
                if np.any(candidates[territory_id] & contested):
                    collision_seen[territory_id] = True
                candidates[territory_id][contested] = winner[contested] == index

        for territory_id in active_ids:
            ring = candidates[territory_id]
            ring_values = image[ring]
            current_values = image[masks[territory_id]]
            if ring_values.size < 20 or current_values.size < 20:
                active[territory_id] = False
                stop_reason[territory_id] = "border_or_blocked"
                history.append(
                    {
                        "cell_id": territory_id,
                        "iteration": iteration,
                        "accepted": False,
                        "reason": stop_reason[territory_id],
                        "ring_n_pixels": int(ring_values.size),
                    }
                )
                continue

            center = float(np.median(current_values))
            spread = max(
                robust_mad(current_values),
                float(background["sigma"]),
                0.5,
            )
            ring_median = float(np.median(ring_values))
            median_shift_rsd = abs(ring_median - center) / spread
            incompatible_fraction = float(
                np.mean(np.abs(ring_values - center) > spread)
            )
            proposed = masks[territory_id] | ring
            proposed_area = int(np.count_nonzero(proposed))
            area_ratio = proposed_area / max(initial_areas[territory_id], 1)
            components_before = int(measure.label(masks[territory_id]).max())
            components_after = int(measure.label(proposed).max())
            joins_same_territory = components_after < components_before

            intensity_reject = bool(
                median_shift_rsd > ring_shift_rsd
                and incompatible_fraction > ring_incompatible_fraction
                and not joins_same_territory
            )
            area_reject = bool(
                area_ratio > maximum_area_ratio and not joins_same_territory
            )
            accepted = not intensity_reject and not area_reject
            if accepted:
                masks[territory_id] = proposed
                accepted_steps[territory_id] += 1
                if area_ratio > maximum_area_ratio:
                    active[territory_id] = False
                    stop_reason[territory_id] = "connected_at_area_limit"
            else:
                rejected_rings |= ring
                active[territory_id] = False
                stop_reason[territory_id] = (
                    "intensity_shift" if intensity_reject else "area_limit"
                )

            history.append(
                {
                    "cell_id": territory_id,
                    "iteration": iteration,
                    "accepted": accepted,
                    "reason": stop_reason[territory_id] if not accepted else "accepted",
                    "ring_n_pixels": int(ring_values.size),
                    "ring_median": ring_median,
                    "territory_median_before": center,
                    "territory_robust_sd": spread,
                    "median_shift_rsd": median_shift_rsd,
                    "incompatible_fraction": incompatible_fraction,
                    "area_ratio_after": area_ratio,
                    "components_before": components_before,
                    "components_after": components_after,
                }
            )

    final_labels = np.zeros_like(seed_labels, dtype=np.uint16)
    statuses: list[dict] = []
    for territory_id in territory_ids:
        final_labels[masks[territory_id]] = territory_id
        final_area = int(np.count_nonzero(masks[territory_id]))
        statuses.append(
            {
                "cell_id": territory_id,
                "track_seed_area_px": initial_areas[territory_id],
                "source_area_px": final_area,
                "growth_expansion_ratio": final_area / max(initial_areas[territory_id], 1),
                "growth_iterations": accepted_steps[territory_id],
                "growth_stop_reason": stop_reason[territory_id] or "iteration_safety_limit",
                "growth_collision": collision_seen[territory_id],
                "final_components": int(measure.label(masks[territory_id]).max()),
            }
        )
    return final_labels, pd.DataFrame(statuses), pd.DataFrame(history), rejected_rings


def measure_cells(
    image: np.ndarray,
    labels: np.ndarray,
    seed_labels: np.ndarray,
    statuses: pd.DataFrame,
    background: dict,
    pixel_um: float,
    image_name: str,
    projection: str,
    seed_track_counts: dict[int, int],
    puncta: np.ndarray,
    minimum_seed_tracks: int,
    detection_z: float,
    detection_fraction: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    bg_mode = float(background["mode"])
    bg_sigma = float(background["sigma"])
    status_map = statuses.set_index("cell_id").to_dict("index") if not statuses.empty else {}

    for cell_id in sorted(int(i) for i in np.unique(labels) if i > 0):
        mask = labels == cell_id
        values = image[mask]
        owned_puncta = puncta & ndi.binary_fill_holes(mask)
        all_mask = mask | owned_puncta
        all_values = image[all_mask]
        if values.size == 0:
            continue
        status = status_map.get(cell_id, {})
        raw_mean = float(np.mean(values))
        raw_median = float(np.median(values))
        raw_trimmed_mean = clipped_mean(values)
        median_difference = raw_median - bg_mode
        signal_fraction = float(np.mean(values > bg_mode + detection_z * bg_sigma))
        marker_detected = bool(
            median_difference > detection_z * bg_sigma
            and signal_fraction >= detection_fraction
        )
        primary = max(0.0, median_difference) if marker_detected else 0.0
        marker_status = "detected" if marker_detected else "no_signal"
        n_seed_tracks = int(seed_track_counts.get(cell_id, 0))
        seed_area_px = int(status.get("track_seed_area_px", np.count_nonzero(seed_labels == cell_id)))
        puncta_fraction = float(
            np.count_nonzero(owned_puncta) / max(1, np.count_nonzero(all_mask))
        )
        ys, xs = np.nonzero(mask)
        touches_border = bool(
            np.any(ys == 0)
            or np.any(xs == 0)
            or np.any(ys == image.shape[0] - 1)
            or np.any(xs == image.shape[1] - 1)
        )

        technical_flags: list[str] = []
        warning_flags: list[str] = []
        phenotype_flags: list[str] = []
        if not background["qc_pass"]:
            technical_flags.append("uncertain_background_peak")
        if n_seed_tracks < minimum_seed_tracks:
            technical_flags.append("insufficient_seed_tracks")
        if values.size < 100:
            technical_flags.append("insufficient_measurement_pixels")
        if puncta_fraction > 0.20:
            technical_flags.append("puncta_dominated")
        stop = str(status.get("growth_stop_reason", ""))
        if stop:
            warning_flags.append(stop)
        if bool(status.get("growth_collision", False)):
            warning_flags.append("neighboring_territory_collision")
        if int(status.get("final_components", 1)) > 1:
            warning_flags.append("disconnected_subregions")
        if touches_border:
            warning_flags.append("touches_border")
        if not marker_detected:
            phenotype_flags.append("no_marker_signal")

        rows.append(
            {
                "image": image_name,
                "analysis_version": ANALYSIS_VERSION,
                "cell_id": cell_id,
                "segmentation_source": "track_territory_adaptive_dilation",
                "projection": projection,
                "pixel_size_um": pixel_um,
                "area_um2": float(np.count_nonzero(mask) * pixel_um**2),
                "centroid_x_px": float(np.mean(xs)),
                "centroid_y_px": float(np.mean(ys)),
                "ficd_bgsub_intensity": primary,
                # Compatibility aliases are deliberately nonnegative and use
                # the same canonical territory-median estimator.
                "ficd_bgsub_mean": primary,
                "ficd_bgsub_median": primary,
                "ficd_bgsub_mean_nonnegative": primary,
                "ficd_bgsub_mean_all_pixels": max(0.0, float(np.mean(all_values)) - bg_mode),
                "ficd_bgsub_clipped_mean": max(0.0, raw_trimmed_mean - bg_mode),
                "cell_raw_mean": raw_mean,
                "cell_raw_median": raw_median,
                "cell_raw_trimmed_mean": raw_trimmed_mean,
                "signal_estimator": "territory_median_minus_global_mode_floor_zero",
                "background_median": bg_mode,
                "background_mad": bg_sigma,
                "background_source": "global_histogram_peak",
                "background_n_pixels": int(image.size),
                "background_peak": bg_mode,
                "background_sigma": bg_sigma,
                "background_peak_fraction": float(background["peak_fraction"]),
                "background_peak_prominence": float(background["prominence_fraction"]),
                "background_peak_qc": bool(background["qc_pass"]),
                "measurement_n_pixels": int(values.size),
                "puncta_fraction": puncta_fraction,
                "bright_region_fraction": float(np.mean(values > bg_mode + 3.0 * bg_sigma)),
                "cell_p95": float(np.percentile(values, 95)),
                "cell_p99": float(np.percentile(values, 99)),
                "signal_to_background_noise": max(0.0, median_difference) / bg_sigma,
                "robust_cv": robust_mad(values) / max(abs(median_difference), bg_sigma),
                "marker_detected": marker_detected,
                "marker_status": marker_status,
                "signal_fraction": signal_fraction,
                "detection_z": detection_z,
                "detection_fraction_required": detection_fraction,
                "perimeter_found": False,
                "track_seed_area_um2": float(seed_area_px * pixel_um**2),
                "growth_expansion_ratio": float(status.get("growth_expansion_ratio", 1.0)),
                "growth_iterations": int(status.get("growth_iterations", 0)),
                "growth_stop_reason": stop,
                "growth_collision": bool(status.get("growth_collision", False)),
                "final_components": int(status.get("final_components", 1)),
                "n_seed_trajectories": n_seed_tracks,
                "touches_border": touches_border,
                "qc_pass": len(technical_flags) == 0,
                "qc_flags": ";".join(technical_flags),
                "warning_flags": ";".join(dict.fromkeys(warning_flags)),
                "phenotype_flags": ";".join(phenotype_flags),
            }
        )
    return pd.DataFrame(rows)


def save_qc_overlay(
    path: Path,
    image: np.ndarray,
    seed_labels: np.ndarray,
    labels: np.ndarray,
    puncta: np.ndarray,
    rejected_rings: np.ndarray,
    density: np.ndarray,
    background: dict,
    cells: pd.DataFrame,
) -> None:
    low, high = np.percentile(image, [1, 99.7])
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    axes[0].imshow(image, cmap="gray", vmin=low, vmax=high)
    axes[0].set_title("Raw marker")

    axes[1].imshow(image, cmap="gray", vmin=low, vmax=high)
    if np.any(density):
        scale = max(float(np.percentile(density[density > 0], 99)), 1e-12)
        axes[1].imshow(density, cmap="magma", alpha=np.clip(density / scale, 0, 0.65))
    for cell_id, color in zip(
        [int(i) for i in np.unique(seed_labels) if i > 0],
        plt.cm.tab10.colors,
    ):
        axes[1].contour(seed_labels == cell_id, [0.5], colors=[color], linewidths=0.9)
    axes[1].set_title(">5-step track territories")

    axes[2].imshow(image, cmap="gray", vmin=low, vmax=high)
    for cell_id, color in zip(
        [int(i) for i in np.unique(labels) if i > 0],
        plt.cm.tab10.colors,
    ):
        axes[2].contour(labels == cell_id, [0.5], colors=[color], linewidths=1.0)
    if np.any(rejected_rings):
        axes[2].contour(rejected_rings, [0.5], colors="magenta", linewidths=0.7)
    if np.any(puncta):
        axes[2].imshow(
            np.ma.masked_where(~puncta, puncta),
            cmap="autumn",
            alpha=0.65,
            interpolation="nearest",
        )
    for row in cells.itertuples():
        axes[2].text(
            row.centroid_x_px,
            row.centroid_y_px,
            f"{row.cell_id}: {row.ficd_bgsub_intensity:.1f}\n{row.marker_status}",
            color="white",
            fontsize=7,
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1},
        )
    axes[2].set_title("Final territories; puncta red; rejected ring magenta")

    centers = background["centers"]
    global_curve = background["smooth_counts"].astype(float)
    global_curve /= max(global_curve.max(), 1.0)
    axes[3].plot(centers, global_curve, color="black", lw=1.5, label="global pixels")
    axes[3].axvline(background["mode"], color="royalblue", lw=1.5, label="background mode")
    for row, color in zip(cells.itertuples(), plt.cm.tab10.colors):
        values = image[labels == row.cell_id]
        counts, _ = np.histogram(values, bins=np.r_[
            centers - (centers[1] - centers[0]) / 2,
            centers[-1] + (centers[1] - centers[0]) / 2,
        ])
        curve = ndi.gaussian_filter1d(counts.astype(float), 1.5)
        curve /= max(curve.max(), 1.0)
        axes[3].plot(centers, curve, color=color, lw=1.0, label=f"territory {row.cell_id}")
        axes[3].axvline(row.cell_raw_median, color=color, lw=0.8, ls="--")
    axes[3].set_xlim(
        max(centers.min(), background["mode"] - 8 * background["sigma"]),
        min(centers.max(), background["mode"] + 70 * background["sigma"]),
    )
    axes[3].set_xlabel("Pixel intensity")
    axes[3].set_ylabel("Normalized count")
    axes[3].set_title("Background and territory distributions")
    axes[3].legend(fontsize=6)

    for axis in axes[:3]:
        axis.set_xlim(0, image.shape[1])
        axis.set_ylim(image.shape[0], 0)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(path.stem.replace("_qc", ""), fontsize=10)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def process_one(
    red_path: Path,
    trajectory_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    image, pixel_um, projection, acquisition = read_nd2_2d(red_path)
    trajectories = load_trajectory(trajectory_path)
    qualifying_total = int(
        ((trajectories.groupby("Trajectory").size() - 1) > args.minimum_seed_steps_exclusive).sum()
    )
    background = estimate_background_peak(
        image, args.background_mode_hint, args.background_hint_window
    )
    seed_labels, density, centroids, density_threshold, seed_track_counts = build_track_seeds(
        trajectories,
        image.shape,
        pixel_um,
        args.minimum_seed_steps_exclusive,
        args.track_density_sigma_um,
        args.track_seed_radius_um,
        args.track_bridge_radius_um,
        args.minimum_seed_area_um2,
        args.minimum_seed_tracks,
        args.track_cluster_radius_um,
        args.minimum_track_neighbors,
    )

    if seed_labels.max() == 0:
        raise TerritoryExclusion(
            f"no territory with at least {args.minimum_seed_tracks} associated "
            f">{args.minimum_seed_steps_exclusive}-step trajectories "
            f"({qualifying_total} qualifying trajectories in movie)"
        )

    puncta = detect_compact_puncta(
        image,
        pixel_um,
        background,
        args.puncta_maximum_area_um2,
    )
    labels, statuses, growth_history, rejected_rings = grow_track_seeds(
        image,
        seed_labels,
        background,
        puncta,
        args.dilation_ring_px,
        args.ring_shift_rsd,
        args.ring_incompatible_fraction,
        args.maximum_area_ratio,
    )

    manual = load_manual_labels(args.mask_dir, red_path)
    if manual is not None:
        if manual.shape != image.shape:
            raise ValueError(
                f"Manual mask shape {manual.shape} does not match {image.shape}: {red_path.name}"
            )
        labels = manual.copy()
        labels[puncta] = 0
        manual_ids = [int(i) for i in np.unique(labels) if i > 0]
        statuses = pd.DataFrame(
            [
                {
                    "cell_id": cell_id,
                    "track_seed_area_px": int(np.count_nonzero(seed_labels == cell_id)),
                    "source_area_px": int(np.count_nonzero(labels == cell_id)),
                    "growth_expansion_ratio": float("nan"),
                    "growth_iterations": 0,
                    "growth_stop_reason": "manual_override",
                    "growth_collision": False,
                    "final_components": int(measure.label(labels == cell_id).max()),
                }
                for cell_id in manual_ids
            ]
        )
        growth_history = pd.DataFrame()
        rejected_rings = np.zeros_like(labels, dtype=bool)

    cells = measure_cells(
        image,
        labels,
        seed_labels,
        statuses,
        background,
        pixel_um,
        red_path.name,
        projection,
        seed_track_counts,
        puncta,
        args.minimum_seed_tracks,
        args.detection_z,
        args.detection_fraction,
    )
    tracks, localizations = assign_trajectories(
        trajectories,
        labels,
        cells,
        pixel_um,
        red_path.name,
        args.maximum_assignment_distance_um,
    )

    tifffile.imwrite(
        output_dir / "masks" / f"{red_path.stem}_labels.tif",
        labels.astype(np.uint16),
    )
    tifffile.imwrite(
        output_dir / "masks" / f"{red_path.stem}_track_seeds.tif",
        seed_labels.astype(np.uint16),
    )
    tifffile.imwrite(
        output_dir / "masks" / f"{red_path.stem}_excluded_puncta.tif",
        puncta.astype(np.uint8),
    )
    tifffile.imwrite(
        output_dir / "masks" / f"{red_path.stem}_rejected_ring.tif",
        rejected_rings.astype(np.uint8),
    )
    save_qc_overlay(
        output_dir / "qc" / f"{red_path.stem}_qc.png",
        image,
        seed_labels,
        labels,
        puncta,
        rejected_rings,
        density,
        background,
        cells,
    )

    metadata = {
        "image": red_path.name,
        "analysis_version": ANALYSIS_VERSION,
        "trajectory_file": trajectory_path.name,
        "processing_status": "ok",
        "pixel_size_um": pixel_um,
        "projection": projection,
        "red_threshold": "",
        "red_background_noise": background["sigma"],
        "seed_density_threshold": density_threshold,
        "n_seed_trajectories": int(len(centroids)),
        "n_qualifying_trajectories_total": qualifying_total,
        "n_qualifying_trajectory_outliers": qualifying_total - int(len(centroids)),
        "n_cells": int(len(cells)),
        "segmentation_source": "track_territory_adaptive_dilation",
        "background_peak": background["mode"],
        "background_sigma": background["sigma"],
        "background_peak_fraction": background["peak_fraction"],
        "background_peak_prominence": background["prominence_fraction"],
        "background_peak_selection": background["selection"],
        "background_peak_qc": background["qc_pass"],
        "dilation_ring_px": args.dilation_ring_px,
        "ring_shift_rsd": args.ring_shift_rsd,
        "ring_incompatible_fraction": args.ring_incompatible_fraction,
        "maximum_area_ratio": args.maximum_area_ratio,
        "minimum_seed_steps_exclusive": args.minimum_seed_steps_exclusive,
        "minimum_seed_tracks": args.minimum_seed_tracks,
        "track_cluster_radius_um": args.track_cluster_radius_um,
        "minimum_track_neighbors": args.minimum_track_neighbors,
        "n_marker_detected_cells": int(cells.get("marker_detected", pd.Series(dtype=bool)).sum()),
        "n_no_signal_cells": int((cells.get("marker_status", pd.Series(dtype=str)) == "no_signal").sum()),
        **acquisition,
    }
    if not growth_history.empty:
        growth_history.insert(0, "image", red_path.name)
    return cells, tracks, localizations, growth_history, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track-seeded per-cell FICD measurement and ER-GEM assignment"
    )
    parser.add_argument("--red", nargs="+", required=True)
    parser.add_argument("--trajectories", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--minimum-seed-steps-exclusive", type=int, default=5)
    parser.add_argument("--minimum-seed-tracks", type=int, default=100)
    parser.add_argument("--minimum-seed-area-um2", type=float, default=5.0)
    parser.add_argument("--track-density-sigma-um", type=float, default=0.45)
    parser.add_argument("--track-seed-radius-um", type=float, default=0.8)
    parser.add_argument("--track-bridge-radius-um", type=float, default=1.0)
    parser.add_argument("--track-cluster-radius-um", type=float, default=3.0)
    parser.add_argument("--minimum-track-neighbors", type=int, default=3)
    parser.add_argument("--dilation-ring-px", type=int, default=3)
    parser.add_argument("--ring-shift-rsd", type=float, default=1.0)
    parser.add_argument("--ring-incompatible-fraction", type=float, default=0.40)
    parser.add_argument("--maximum-area-ratio", type=float, default=3.0)
    parser.add_argument("--puncta-maximum-area-um2", type=float, default=2.0)
    parser.add_argument("--detection-z", type=float, default=1.0)
    parser.add_argument("--detection-fraction", type=float, default=0.50)
    parser.add_argument("--maximum-assignment-distance-um", type=float, default=2.0)
    parser.add_argument("--background-mode-hint", type=float, default=105.0)
    parser.add_argument("--background-hint-window", type=float, default=30.0)
    parser.add_argument("--version", action="version", version=ANALYSIS_VERSION)
    # Accepted but unused so older pipeline invocations do not break.
    parser.add_argument("--track-probe-radius-um", type=float, default=1.5, help=argparse.SUPPRESS)
    parser.add_argument("--no-signal-radius-um", type=float, default=1.5, help=argparse.SUPPRESS)
    parser.add_argument("--signal-gap-radius-um", type=float, default=0.55, help=argparse.SUPPRESS)
    parser.add_argument("--growth-z", type=float, default=1.5, help=argparse.SUPPRESS)
    parser.add_argument("--perimeter-radius-um", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--perimeter-z-max", type=float, default=1.5, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.absolute()
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    (output_dir / "qc").mkdir(parents=True, exist_ok=True)
    red_paths, trajectory_map = discover_files(args)

    missing = [path.name for path in red_paths if image_key(path) not in trajectory_map]
    if missing:
        raise FileNotFoundError(
            "Missing matching Traj_<key>.csv for: " + ", ".join(missing)
        )

    all_cells: list[pd.DataFrame] = []
    all_tracks: list[pd.DataFrame] = []
    all_localizations: list[pd.DataFrame] = []
    all_growth_history: list[pd.DataFrame] = []
    metadata_rows: list[dict] = []
    for red_path in red_paths:
        trajectory_path = trajectory_map[image_key(red_path)]
        try:
            cells, tracks, localizations, growth_history, metadata = process_one(
                red_path, trajectory_path, output_dir, args
            )
        except TerritoryExclusion as exc:
            metadata_rows.append(
                {
                    "image": red_path.name,
                    "trajectory_file": trajectory_path.name,
                    "processing_status": "excluded",
                    "exclusion_reason": str(exc),
                    "minimum_seed_steps_exclusive": args.minimum_seed_steps_exclusive,
                    "minimum_seed_tracks": args.minimum_seed_tracks,
                }
            )
            print(f"Excluded {red_path.name}: {exc}", flush=True)
            continue
        all_cells.append(cells)
        all_tracks.append(tracks)
        all_localizations.append(localizations)
        if not growth_history.empty:
            all_growth_history.append(growth_history)
        metadata_rows.append(metadata)

    cells_out = pd.concat(all_cells, ignore_index=True) if all_cells else pd.DataFrame(
        columns=["image", "cell_id", "ficd_bgsub_intensity", "ficd_bgsub_mean", "qc_pass"]
    )
    tracks_out = pd.concat(all_tracks, ignore_index=True) if all_tracks else pd.DataFrame(
        columns=["image", "Trajectory", "cell_id", "ficd_bgsub_intensity", "ficd_bgsub_mean"]
    )
    localizations_out = (
        pd.concat(all_localizations, ignore_index=True)
        if all_localizations
        else pd.DataFrame(columns=["Trajectory", "Frame", "x", "y", "cell_id"])
    )
    cells_out.to_csv(output_dir / "cells.csv", index=False)
    tracks_out.to_csv(
        output_dir / "trajectory_expression.csv", index=False
    )
    localizations_out.to_csv(
        output_dir / "localizations_with_expression.csv", index=False
    )
    growth_out = (
        pd.concat(all_growth_history, ignore_index=True)
        if all_growth_history
        else pd.DataFrame(columns=["image", "cell_id", "iteration", "accepted", "reason"])
    )
    growth_out.to_csv(output_dir / "dilation_history.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(output_dir / "run_metadata.csv", index=False)
    print(f"FICD analysis version: {ANALYSIS_VERSION}")
    print(f"Processed {len(red_paths)} paired marker/trajectory image(s)")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
