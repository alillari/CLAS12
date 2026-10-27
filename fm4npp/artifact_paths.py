"""Default research-product paths outside the source checkout."""

import os
from pathlib import Path


def artifact_path(*parts):
    """Honor an explicit artifact root, otherwise use a sibling storage folder."""
    configured = os.environ.get("CLAS12_ARTIFACT_ROOT")
    root = (Path(os.path.expandvars(configured)).expanduser() if configured
            else Path(__file__).resolve().parents[2] / "result_deep_storage")
    return str(root.joinpath(*parts).resolve())
