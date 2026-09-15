#!/usr/bin/env bash
# =============================================================================
# setup.sh
#
# Build a runnable checkout: clone upstream MedSAM2 and copy this repository's
# modified and added files over it.
#
#   bash setup.sh [target_dir]        # default: ./MedSAM2
#
# This repository holds only the files we changed or added. Everything else in
# MedSAM2 is used unmodified and is not redistributed here, so it is fetched
# from the upstream project.
#
# The script only writes inside the target directory. It refuses to overwrite an
# existing checkout unless you pass --force.
# =============================================================================

set -euo pipefail

UPSTREAM_URL="https://github.com/bowang-lab/MedSAM2.git"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET="./MedSAM2"
FORCE=0
for a in "$@"; do
    case "$a" in
        --force) FORCE=1 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) TARGET="$a" ;;
    esac
done

if [ -e "$TARGET" ] && [ "$FORCE" -eq 0 ]; then
    echo "Error: $TARGET already exists. Choose another path, or pass --force to"
    echo "copy this repository's files over the existing checkout."
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "Error: git is required."; exit 1
fi

if [ ! -d "$TARGET/.git" ]; then
    echo "Cloning upstream MedSAM2 into $TARGET ..."
    git clone --depth 1 "$UPSTREAM_URL" "$TARGET"
else
    echo "Reusing existing checkout at $TARGET"
fi

if [ ! -f "$TARGET/training/trainer.py" ]; then
    echo "Error: $TARGET does not look like a MedSAM2 checkout."; exit 1
fi

echo "Applying our files ..."
for d in training sam2 preprocessing scripts evaluation; do
    [ -d "$HERE/$d" ] || continue
    mkdir -p "$TARGET/$d"
    cp -r "$HERE/$d/." "$TARGET/$d/"
    echo "  $d/"
done

cat <<EOF

Done. Next steps:

  1. cd $TARGET
  2. Install MedSAM2 as described in its own README, then: pip install peft
  3. Download the SAM 2.1 tiny checkpoint into checkpoints/ as upstream instructs
  4. Preprocess your data:
       python preprocessing/preprocess_ct_npz.py --mode train ...
  5. Train:
       bash scripts/train_lora.sh 3

See our README for the three-stage fine-tuning chain and the evaluation protocol.
EOF
