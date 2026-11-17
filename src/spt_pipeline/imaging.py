"""Image loading for ND2 and TIFF, with channel selection.

Isolates the optional heavy imaging dependencies (``nd2``, ``tifffile``,
``pims``) so the rest of the package imports cleanly without them. Format is
chosen from the file extension.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def select_channel(arr: np.ndarray, channel: int = 0) -> np.ndarray:
    """Reduce a multichannel movie array to ``(T, Y, X)`` for one channel.

    Handles ``(Y,X)``, ``(T,Y,X)``, ``(T,C,Y,X)`` and ``(T,Y,X,C)`` layouts.
    ``channel`` is clamped to the valid range.
    """
    if arr.ndim <= 3:
        return arr
    channel = max(int(channel), 0)
    if arr.ndim == 4:
        shape = arr.shape
        if shape[1] <= 8 and (shape[-1] > 8 or shape[1] <= shape[-1]):
            return arr[:, min(channel, shape[1] - 1), :, :]
        if shape[-1] <= 8:
            return arr[..., min(channel, shape[-1] - 1)]
        return arr[:, min(channel, shape[1] - 1), :, :]
    return arr


def load_movie(path: Path, file_format: Optional[str] = None, channel: int = 0) -> np.ndarray:
    """Load an image stack as a numpy array ``(T, Y, X)``.

    ``file_format`` overrides extension detection ('nd2', 'tif', 'tiff').
    """
    path = Path(path)
    fmt = (file_format or path.suffix.lstrip(".")).lower()
    if fmt == "nd2":
        import nd2  # type: ignore
        arr = nd2.imread(str(path))
        return select_channel(np.asarray(arr), channel)
    if fmt in ("tif", "tiff"):
        import tifffile  # type: ignore
        arr = tifffile.imread(str(path))
        arr = np.asarray(arr)
        if arr.ndim >= 4:
            arr = select_channel(arr, channel)
        return arr
    raise ValueError(f"Unsupported image format: {fmt} ({path})")
