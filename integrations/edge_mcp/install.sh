#!/usr/bin/env bash
set -euo pipefail
edge_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 -m venv "$edge_root/.venv"
"$edge_root/.venv/bin/python" -m pip install -r "$edge_root/requirements.lock"
"$edge_root/.venv/bin/python" -m pip install --no-build-isolation --no-deps -e "$edge_root"
if [[ "${1:-}" == "--register" ]]; then
  codex mcp add clas12-edge -- "$edge_root/.venv/bin/edge-mcp"
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--register]" >&2
  exit 2
fi
printf 'Runner: %s\nMCP server: %s\n' "$edge_root/.venv/bin/edge-runner" "$edge_root/.venv/bin/edge-mcp"
