#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
#
# Build phase for vla_client_rbnx.
# Pure Python skill — no colcon, no vendored ROS packages.
# Only codegen + soft-check runtime deps.
set -euo pipefail
PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PKG"
CLEAN="${RBNX_BUILD_CLEAN:-}"

if [[ "$CLEAN" == "1" ]]; then
    echo "[vla_client/build] clean: removing rbnx-build/"
    rm -rf rbnx-build
fi
mkdir -p rbnx-build

# Codegen
if command -v rbnx &>/dev/null; then
    FLAGS=(--out-dir "$PKG/rbnx-build/codegen" --mcp)
    [[ "$CLEAN" == "1" ]] && FLAGS+=(--clean)
    echo "[vla_client/build] rbnx codegen ${FLAGS[*]}"
    rbnx codegen -p "$PKG" "${FLAGS[@]}" || true
fi

# Soft-check runtime deps
python3 -c "import requests, numpy, json_numpy" 2>/dev/null || \
    echo "[WARN] Missing: pip install requests numpy json-numpy"

touch "$PKG/rbnx-build/.rbnx-built"
echo "[vla_client/build] done."
