#!/bin/bash
# Copies this repo folder to a CHTC submit node, skipping the big tarballs, datasets and git history.
# usage: ./transfer_to_chtc.sh <user@submit-node> [remote dir, default: cs771-fall2026-hw2] [--dry-run]
#   e.g. ./transfer_to_chtc.sh asehgal9@ap2002.chtc.wisc.edu
set -e
HOST=$1
if [ -z "$HOST" ]; then echo "usage: $0 <user@submit-node> [remote dir] [--dry-run]" >&2; exit 1; fi
DEST=${2:-cs771-fall2026-hw2}
EXTRA=$3
[ "$DEST" = "--dry-run" ] && { EXTRA=--dry-run; DEST=cs771-fall2026-hw2; }

cd "$(dirname "$0")/../.."
# keeps results/*/model_final.pth.tar (needed by eval) but not logs tarballs or epoch checkpoints
# --checksum: compare file contents, not just size/mtime, so a stale or wrong checkpoint gets replaced
rsync -avh --checksum --progress $EXTRA \
    --exclude='.git/' \
    --exclude='*.tar.gz' \
    --exclude='/data/' \
    --exclude='/logs/' --exclude='/models/' \
    --exclude='epoch_*.pth.tar' \
    --exclude='__pycache__/' --exclude='.venv/' --exclude='.DS_Store' \
    ./ "$HOST:$DEST/"
