"""Per-run JSON documentation.

Every pipeline invocation writes a sidecar JSON to ``{output_dir}/run_logs/``
capturing the full parameter set, environment, timing, and produced outputs.
This is the reproducibility backbone: given a run log you can recreate the
exact command and know which package version produced a figure.
"""
from __future__ import annotations

import hashlib
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from .utils import dump_json, to_jsonable


def _pkg_version(name: str) -> Optional[str]:
    try:
        module = __import__(name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


class RunLog:
    """Accumulates run metadata and writes it on ``finish``.

    Usage::

        rl = RunLog("track trackpy", args_dict, output_dir, manifest_path)
        ... do work, rl.add_output(path), rl.note("...") ...
        rl.finish(n_processed=10, n_skipped=2)
    """

    def __init__(
        self,
        command: str,
        args: Dict[str, Any],
        output_dir: Path,
        manifest_path: Optional[Path] = None,
    ) -> None:
        self.command = command
        self.args = dict(args)
        self.output_dir = Path(output_dir)
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.start = datetime.now()
        self.outputs: List[Dict[str, Any]] = []
        self.warnings: List[str] = []

    def note(self, message: str) -> None:
        self.warnings.append(str(message))

    def add_output(self, path: Path) -> None:
        p = Path(path)
        try:
            size = p.stat().st_size if p.exists() else None
        except OSError:
            size = None
        self.outputs.append({"path": str(p), "size_bytes": size})

    def finish(self, n_processed: int = 0, n_skipped: int = 0, extra: Optional[Dict[str, Any]] = None) -> Path:
        end = datetime.now()
        payload: Dict[str, Any] = {
            "command": self.command,
            "args": to_jsonable(self.args),
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "manifest_hash": sha256_file(self.manifest_path) if self.manifest_path else None,
            "script_version": __version__,
            "start_time": self.start.isoformat(timespec="seconds"),
            "end_time": end.isoformat(timespec="seconds"),
            "duration_s": round((end - self.start).total_seconds(), 3),
            "n_processed": int(n_processed),
            "n_skipped": int(n_skipped),
            "outputs": self.outputs,
            "warnings": self.warnings,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy_version": _pkg_version("numpy"),
            "pandas_version": _pkg_version("pandas"),
            "scipy_version": _pkg_version("scipy"),
            "trackpy_version": _pkg_version("trackpy"),
        }
        if extra:
            payload.update(to_jsonable(extra))
        stamp = self.start.strftime("%Y%m%d_%H%M%S")
        safe = self.command.replace(" ", "_").replace("-", "_")
        out_path = self.output_dir / "run_logs" / f"run_{stamp}_{safe}.json"
        dump_json(payload, out_path)
        return out_path
