#!/usr/bin/env bash
# Push code to a remote machine. Run from the project root on the Mac.
#
#   ./sync.sh              # -> the GPU box (gx10-3)
#   ./sync.sh cpu          # -> apa-dipsco
#   ./sync.sh gpu --dry    # show what would change, send nothing
#
# Code only. The corpus, indexes, venv and .env are deliberately excluded:
# they are large, machine-specific, or secret, and overwriting a remote .env
# with a local one has broken this setup before.

set -euo pipefail

GPU_HOST="ana@10.216.20.140"
CPU_HOST="ana@10.216.20.126"
REMOTE_DIR="~/RAG_pipeline"

target="${1:-gpu}"
shift || true

case "$target" in
  gpu) host="$GPU_HOST" ;;
  cpu) host="$CPU_HOST" ;;
  *)   echo "usage: ./sync.sh [gpu|cpu] [--dry]"; exit 1 ;;
esac

dry=""
[[ "${1:-}" == "--dry" ]] && dry="--dry-run"

echo "→ $host:$REMOTE_DIR ${dry:+(dry run)}"

rsync -avz --itemize-changes $dry \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude 'storage' \
  --exclude 'results' \
  --exclude '*.jsonl' \
  --exclude '.env' \
  --exclude '.DS_Store' \
  src scripts tests config.yaml requirements.txt \
  "$host:$REMOTE_DIR/"

echo
echo "done. On the remote:"
echo "    cd ~/RAG_pipeline && source ~/rag-venv/bin/activate"
