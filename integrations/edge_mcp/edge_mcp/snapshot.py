"""Create a deterministic source archive, including working-tree changes."""

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

MAX_SOURCE_BYTES = 64 * 1024 * 1024
EXCLUDED = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_snapshot(directory: str | None) -> dict:
    if directory is None:
        files = {}
        git = None
        payloads = []
    else:
        root = Path(directory).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("source_path must be a directory")
        git = None
        probe = subprocess.run(["git", "--no-optional-locks", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
        if probe.returncode == 0:
            top = Path(probe.stdout.strip())
            prefix = root.relative_to(top).as_posix()
            result = subprocess.run(["git", "--no-optional-locks", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "."], capture_output=True, check=True)
            names = sorted(set(os.fsdecode(p) for p in result.stdout.split(b"\x00") if p))
            git = {
                "head": subprocess.check_output(["git", "--no-optional-locks", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
                "subdirectory": prefix,
                "dirty": bool(subprocess.check_output(["git", "--no-optional-locks", "-C", str(root), "status", "--porcelain", "--", "."])),
            }
        else:
            names = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() or p.is_symlink())
        files, payloads, size = {}, [], 0
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid source archive path")
            if any(part in EXCLUDED for part in relative.parts):
                continue
            path = root / relative
            # Parent symlinks must not turn a snapshot into an external-file read.
            if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
                raise ValueError(f"Source symlinks are not supported: {name}")
            if not path.exists():  # Tracked file deleted in the working tree.
                continue
            if not path.is_file():
                raise ValueError(f"Source entry is not a regular file: {name}")
            size += path.stat().st_size
            if size > MAX_SOURCE_BYTES:
                raise ValueError("Source snapshot exceeds 64 MiB; select a smaller source directory")
            data = path.read_bytes()
            mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
            files[relative.as_posix()] = {"sha256": hashlib.sha256(data).hexdigest(), "mode": mode, "size": len(data)}
            payloads.append((relative.as_posix(), data, mode))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data, mode in payloads:
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode, entry.mtime = len(data), mode, 0
            archive.addfile(entry, io.BytesIO(data))
    import gzip

    return {"sha256": digest_json(files), "files": files, "git": git,
            "archive": base64.b64encode(gzip.compress(buffer.getvalue(), mtime=0)).decode()}
