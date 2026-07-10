#!/usr/bin/env bash
# Serve the MkDocs site locally with hot reload.
# Usage: ./scripts/serve.sh
set -euo pipefail

cd "$(dirname "$0")/.."
mkdocs serve --dev-addr=127.0.0.1:8000
