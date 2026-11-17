#!/usr/bin/env python3
"""
Summarize acquisition metadata from ND2/TIF/TIFF movies.

This is a lightweight prescreening utility, separate from the main
``spt-pipeline`` workflow. It recursively scans a directory for movie files
and writes one CSV row per movie, leaving cells blank when metadata cannot
be found.

Console-script entry point: ``spt-metadata``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import nd2
    HAVE_ND2 = True
except ImportError:
    HAVE_ND2 = False

try:
    import tifffile
    HAVE_TIFFFILE = True
except ImportError:
    HAVE_TIFFFILE = False


COLUMNS = [
    "movie_name",
    "extension",
    "relative_path",
    "absolute_path",
    "n_frames",
    "exposure_ms",
    "time_between_frames_ms",
    "pixel_size_um",
    "objective_name",
    "objective_magnification",
    "zoom_magnification",
    "relay_lens_magnification",
    "laser_wavelengths_active_nm",
    "laser_powers_active",
    "tirf_device",
    "tirf_angle_deg",
    "tirf_direction_deg",
    "tirf_focus",
    "tirf_focus_point",
    "tirf_x",
    "tirf_y",
    "tirf_alignment_name",
    "tirf_alignment_wavelength_nm",
    "metadata_source",
]


def _blank_row(path: Path, root: Path) -> Dict[str, Any]:
    return {
        "movie_name": path.stem,
        "extension": path.suffix.lower(),
        "relative_path": str(path.relative_to(root)),
        "absolute_path": str(path.resolve()),
        "n_frames": "",
        "exposure_ms": "",
        "time_between_frames_ms": "",
        "pixel_size_um": "",
        "objective_name": "",
        "objective_magnification": "",
        "zoom_magnification": "",
        "relay_lens_magnification": "",
        "laser_wavelengths_active_nm": "",
        "laser_powers_active": "",
        "tirf_device": "",
        "tirf_angle_deg": "",
        "tirf_direction_deg": "",
        "tirf_focus": "",
        "tirf_focus_point": "",
        "tirf_x": "",
        "tirf_y": "",
        "tirf_alignment_name": "",
        "tirf_alignment_wavelength_nm": "",
        "metadata_source": "",
    }


def _should_ignore_path(path_like: os.PathLike[str] | str) -> bool:
    p = Path(path_like)
    return any(part.startswith(".") or part.startswith("._") or part == "__MACOSX" for part in p.parts)


def _discover_movie_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for pattern in ("*.nd2", "*.tif", "*.tiff"):
        files.extend(
            p for p in root.rglob(pattern)
            if p.is_file() and not _should_ignore_path(p)
        )
    return sorted(set(files), key=lambda p: str(p))


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return round(value, 6)
    return value


def _flatten_scalars(obj: Any, prefix: str = "root", out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            _flatten_scalars(value, f"{prefix}.{key}", out)
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj[:100]):
            _flatten_scalars(value, f"{prefix}[{idx}]", out)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        out[prefix] = obj
    else:
        out[prefix] = str(obj)
    return out


def _first_numeric(flat: Dict[str, Any], key_patterns: Iterable[str], *, must_be_positive: bool = False) -> Any:
    pats = [p.lower() for p in key_patterns]
    for key, value in flat.items():
        key_low = key.lower()
        if not any(p in key_low for p in pats):
            continue
        if any(bad in key_low for bad in ("valid", "use", "exist", "size", "count")):
            continue
        if isinstance(value, (int, float)):
            if must_be_positive and value <= 0:
                continue
            return value
    return ""


def _first_string(flat: Dict[str, Any], key_patterns: Iterable[str]) -> str:
    pats = [p.lower() for p in key_patterns]
    for key, value in flat.items():
        key_low = key.lower()
        if not any(p in key_low for p in pats):
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _suffix_value(flat: Dict[str, Any], suffix: str) -> Any:
    suffix = suffix.lower()
    for key, value in flat.items():
        if key.lower().endswith(suffix):
            return value
    return ""


def _regex_first_number(text: str, patterns: Iterable[str], unit_scale: float = 1.0) -> Any:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1)) * unit_scale
            except Exception:
                continue
    return ""


def _format_active_lasers(entries: List[Tuple[Any, Any]]) -> Tuple[str, str]:
    wavelengths = []
    powers = []
    for wavelength, power in entries:
        if wavelength in ("", None) and power in ("", None):
            continue
        if wavelength not in ("", None):
            wavelengths.append(str(int(wavelength)) if float(wavelength).is_integer() else str(wavelength))
        if wavelength not in ("", None) and power not in ("", None):
            label = f"{int(wavelength)}nm:{power}"
        elif power not in ("", None):
            label = str(power)
        else:
            label = ""
        if label:
            powers.append(label)
    return "; ".join(wavelengths), "; ".join(powers)


def _parse_active_lasers_from_text(text: str) -> Tuple[str, str]:
    matches = re.findall(
        r"Line:\s*\d+\s*;\s*ExW:\s*([0-9.]+)\s*;\s*Power:\s*([0-9.]+)\s*;\s*(On|Off)",
        text,
        flags=re.IGNORECASE,
    )
    active = []
    for wavelength, power, state in matches:
        if state.lower() == "on":
            active.append((float(wavelength), float(power)))
    return _format_active_lasers(active)


def _safe_median(values: Iterable[float]) -> Any:
    vals = [float(v) for v in values if v is not None and pd.notna(v)]
    if not vals:
        return ""
    return float(median(vals))


def _summarize_nd2(path: Path, root: Path) -> Dict[str, Any]:
    row = _blank_row(path, root)

    if HAVE_ND2:
        with nd2.ND2File(path) as handle:
            row["metadata_source"] = "nd2"
            row["n_frames"] = getattr(handle.attributes, "sequenceCount", "") or handle.sizes.get("T", "")

            try:
                voxel = handle.voxel_size()
                row["pixel_size_um"] = getattr(voxel, "x", "") or ""
            except Exception:
                pass

            try:
                channel = handle.metadata.channels[0]
                row["objective_name"] = getattr(channel.microscope, "objectiveName", "") or ""
                row["objective_magnification"] = getattr(channel.microscope, "objectiveMagnification", "") or ""
                row["zoom_magnification"] = getattr(channel.microscope, "zoomMagnification", "") or ""
                if row["pixel_size_um"] == "":
                    axes_cal = getattr(channel.volume, "axesCalibration", ())
                    if axes_cal:
                        row["pixel_size_um"] = axes_cal[0]
            except Exception:
                pass

            text_info = getattr(handle, "text_info", {}) or {}
            text_blob = "\n".join(str(v) for v in text_info.values() if v is not None)

            try:
                events = handle.events()
            except Exception:
                events = []

            if isinstance(events, list) and events:
                exposure_vals = [ev.get("Exposure Time [ms]") for ev in events if isinstance(ev, dict)]
                row["exposure_ms"] = _safe_median(exposure_vals)
                time_vals = [ev.get("Time [s]") for ev in events if isinstance(ev, dict) and ev.get("Time [s]") is not None]
                if len(time_vals) >= 2:
                    deltas_ms = [(time_vals[i] - time_vals[i - 1]) * 1000.0 for i in range(1, len(time_vals))]
                    row["time_between_frames_ms"] = _safe_median(deltas_ms)

            flat = {}
            try:
                flat = _flatten_scalars(handle.unstructured_metadata())
            except Exception:
                flat = {}

            if row["exposure_ms"] == "":
                row["exposure_ms"] = _first_numeric(flat, ["exposure"], must_be_positive=True)
            if row["exposure_ms"] == "":
                row["exposure_ms"] = _regex_first_number(
                    text_blob,
                    [r"Exposure:\s*([0-9.]+)\s*ms", r"Exposure Time\s*\[ms\]\s*[:=]\s*([0-9.]+)"],
                )

            if row["time_between_frames_ms"] == "" and isinstance(events, list) and len(events) >= 2:
                times = [ev.get("Time [s]") for ev in events if isinstance(ev, dict) and ev.get("Time [s]") is not None]
                if len(times) >= 2:
                    row["time_between_frames_ms"] = _safe_median([(times[i] - times[i - 1]) * 1000.0 for i in range(1, len(times))])

            row["relay_lens_magnification"] = _suffix_value(flat, "relaylenszoom")
            if row["relay_lens_magnification"] in ("", None):
                projective_mag = _suffix_value(flat, "projectivemag")
                if isinstance(projective_mag, (int, float)) and projective_mag > 0:
                    row["relay_lens_magnification"] = projective_mag

            row["tirf_device"] = _first_string(flat, ["tirfpositionshortname", "tirfpositionfullname"])

            angle_valid = _suffix_value(flat, "tirfpositionanglevalid")
            direction_valid = _suffix_value(flat, "tirfpositiondirectionvalid")
            focus_valid = _suffix_value(flat, "tirfpositionfocusvalid")
            focus_point_valid = _suffix_value(flat, "tirfpositionfocuspointvalid")
            xy_valid = _suffix_value(flat, "tirfpositionxyvalid")

            if angle_valid not in (0, False):
                row["tirf_angle_deg"] = _suffix_value(flat, "tirfpositionangle")
            if direction_valid not in (0, False):
                row["tirf_direction_deg"] = _suffix_value(flat, "tirfpositiondirection")
            if focus_valid not in (0, False):
                row["tirf_focus"] = _suffix_value(flat, "tirfpositionfocus")
            if focus_point_valid not in (0, False):
                row["tirf_focus_point"] = _suffix_value(flat, "tirfpositionfocuspoint")
            if xy_valid not in (0, False):
                row["tirf_x"] = _suffix_value(flat, "tirfpositionx")
                row["tirf_y"] = _suffix_value(flat, "tirfpositiony")

            row["tirf_alignment_name"] = _suffix_value(flat, "tirfalignmentalignmentname")
            row["tirf_alignment_wavelength_nm"] = _suffix_value(flat, "tirfalignmentalignmentwavelength")

            wavelengths, powers = _parse_active_lasers_from_text(text_blob)
            row["laser_wavelengths_active_nm"] = wavelengths
            row["laser_powers_active"] = powers

            if row["pixel_size_um"] == "":
                row["pixel_size_um"] = _first_numeric(flat, ["pixel_microns", "axescalibration"], must_be_positive=True)

    else:
        raise RuntimeError("`nd2` is not installed. Install with: pip install nd2")

    return {k: _clean_scalar(v) for k, v in row.items()}


def _parse_ome_metadata(ome_xml: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        root = ET.fromstring(ome_xml)
    except Exception:
        return out

    ns_uri = ""
    if root.tag.startswith("{") and "}" in root.tag:
        ns_uri = root.tag[1:].split("}", 1)[0]
    ns = {"ome": ns_uri} if ns_uri else {}
    image = root.find(".//ome:Image", ns) if ns else root.find(".//Image")
    pixels = root.find(".//ome:Pixels", ns) if ns else root.find(".//Pixels")

    if pixels is not None:
        out["n_frames"] = pixels.attrib.get("SizeT", "")
        out["pixel_size_um"] = pixels.attrib.get("PhysicalSizeX", "")
        out["time_between_frames_ms"] = (
            float(pixels.attrib["TimeIncrement"]) * 1000.0
            if "TimeIncrement" in pixels.attrib else ""
        )

    planes = root.findall(".//ome:Plane", ns) if ns else root.findall(".//Plane")
    exposure_vals = []
    delta_vals = []
    for plane in planes:
        if "ExposureTime" in plane.attrib:
            try:
                exposure_vals.append(float(plane.attrib["ExposureTime"]) * 1000.0)
            except Exception:
                pass
        if "DeltaT" in plane.attrib:
            try:
                delta_vals.append(float(plane.attrib["DeltaT"]) * 1000.0)
            except Exception:
                pass
    out["exposure_ms"] = _safe_median(exposure_vals)
    if out.get("time_between_frames_ms", "") == "":
        out["time_between_frames_ms"] = _safe_median(delta_vals)

    if image is not None:
        out["objective_name"] = image.attrib.get("Name", "")
    return out


def _collect_tiff_text(tif: "tifffile.TiffFile") -> str:
    blocks: List[str] = []
    if getattr(tif, "ome_metadata", None):
        blocks.append(str(tif.ome_metadata))
    if getattr(tif, "imagej_metadata", None):
        try:
            blocks.append(json.dumps(tif.imagej_metadata, default=str))
        except Exception:
            blocks.append(str(tif.imagej_metadata))
    if tif.pages:
        first = tif.pages[0]
        for tag_name in ("ImageDescription", "Software", "DocumentName"):
            try:
                if tag_name in first.tags:
                    blocks.append(str(first.tags[tag_name].value))
            except Exception:
                continue
    return "\n".join(blocks)


def _summarize_tiff(path: Path, root: Path) -> Dict[str, Any]:
    if not HAVE_TIFFFILE:
        raise RuntimeError("`tifffile` is required for TIFF metadata extraction.")

    row = _blank_row(path, root)
    row["metadata_source"] = "tifffile"

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0] if tif.series else None
        if series is not None:
            shape = tuple(series.shape)
            if len(shape) >= 3:
                row["n_frames"] = shape[0]
            else:
                row["n_frames"] = len(tif.pages)
        else:
            row["n_frames"] = len(tif.pages)

        if getattr(tif, "ome_metadata", None):
            ome_fields = _parse_ome_metadata(tif.ome_metadata)
            for key, value in ome_fields.items():
                if key in row and row[key] == "":
                    row[key] = value

        text_blob = _collect_tiff_text(tif)

        if row["exposure_ms"] == "":
            row["exposure_ms"] = _regex_first_number(
                text_blob,
                [
                    r"Exposure:\s*([0-9.]+)\s*ms",
                    r"ExposureTime[^0-9]*([0-9.]+)",
                    r"Exposure Time[^0-9]*([0-9.]+)\s*ms",
                ],
            )
        if row["time_between_frames_ms"] == "":
            row["time_between_frames_ms"] = _regex_first_number(
                text_blob,
                [
                    r"TimeIncrement[^0-9]*([0-9.]+)",
                    r"DeltaT[^0-9]*([0-9.]+)",
                    r"time[_ ]between[_ ]frames[^0-9]*([0-9.]+)\s*ms",
                ],
                unit_scale=1000.0 if "TimeIncrement" in text_blob or "DeltaT" in text_blob else 1.0,
            )
        if row["pixel_size_um"] == "":
            row["pixel_size_um"] = _regex_first_number(
                text_blob,
                [
                    r'PhysicalSizeX="([0-9.]+)"',
                    r"pixel[_ ]size[^0-9]*([0-9.]+)\s*(?:um|µm)",
                ],
            )
        if row["relay_lens_magnification"] == "":
            relay = _regex_first_number(
                text_blob,
                [
                    r"relay[^0-9]*(1\.5|1\.0|1)",
                    r"projective[^0-9]*(1\.5|1\.0|1)",
                    r"zoom[^0-9]*(1\.5|1\.0|1)",
                ],
            )
            row["relay_lens_magnification"] = relay

        row["tirf_angle_deg"] = _regex_first_number(text_blob, [r"tirf[^\\n\\r]*angle[^0-9]*([0-9.]+)"])
        row["tirf_direction_deg"] = _regex_first_number(text_blob, [r"tirf[^\\n\\r]*direction[^0-9]*([0-9.]+)"])
        row["tirf_focus"] = _regex_first_number(text_blob, [r"tirf[^\\n\\r]*focus[^0-9]*([0-9.]+)"])
        row["tirf_x"] = _regex_first_number(text_blob, [r"tirf[^\\n\\r]*x[^0-9-]*([-+]?[0-9.]+)"])
        row["tirf_y"] = _regex_first_number(text_blob, [r"tirf[^\\n\\r]*y[^0-9-]*([-+]?[0-9.]+)"])

        wavelengths, powers = _parse_active_lasers_from_text(text_blob)
        row["laser_wavelengths_active_nm"] = wavelengths
        row["laser_powers_active"] = powers

    return {k: _clean_scalar(v) for k, v in row.items()}


def summarize_movie(path: Path, root: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".nd2":
            return _summarize_nd2(path, root)
        if suffix in (".tif", ".tiff"):
            return _summarize_tiff(path, root)
        row = _blank_row(path, root)
        row["metadata_source"] = "unsupported"
        return row
    except Exception as exc:
        row = _blank_row(path, root)
        row["metadata_source"] = f"error:{type(exc).__name__}"
        return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively scan ND2/TIF/TIFF movies and summarize acquisition metadata into a CSV."
    )
    parser.add_argument("input_dir", help="Top-level folder to scan recursively")
    parser.add_argument(
        "-o",
        "--output-csv",
        help="Output CSV path (default: <input_dir>/movie_metadata_summary.csv)",
    )
    args = parser.parse_args()

    root = Path(args.input_dir).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Input directory not found: {root}")

    movie_files = _discover_movie_files(root)
    if not movie_files:
        raise SystemExit(f"No ND2/TIF/TIFF movies found under: {root}")

    rows = [summarize_movie(path, root) for path in movie_files]
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.fillna("")

    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else (root / "movie_metadata_summary.csv")
    df.to_csv(output_csv, index=False)

    print(f"Scanned {len(movie_files)} movie file(s).")
    print(f"Wrote metadata summary CSV: {output_csv}")


if __name__ == "__main__":
    main()
