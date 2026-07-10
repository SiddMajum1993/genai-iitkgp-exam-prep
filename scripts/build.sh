#!/usr/bin/env bash
# Build the MkDocs site into ./site with strict mode (fails on broken links).
# Usage: ./scripts/build.sh
set -euo pipefail

cd "$(dirname "$0")/.."
mkdocs build --strict --verbose
echo ""
echo "Site built to: ./site"
echo "Preview locally with: python -m http.server 8000 --directory site"
