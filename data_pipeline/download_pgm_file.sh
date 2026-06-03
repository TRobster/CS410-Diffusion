#!/usr/bin/env bash
# Downloads standard grayscale PGM test images used in image-processing benchmarks.
# Images come from John Burkardt's public PGM sample collection (FSU).
# Usage:
#   ./download_pgm_file.sh              # download all defaults
#   ./download_pgm_file.sh <url> [name] # download a single PGM from a custom URL
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/pgm_files"
mkdir -p "$OUT_DIR"

BASE_URL="https://people.sc.fsu.edu/~jburkardt/data/pgma/"

DEFAULT_IMAGES=(
    "mona_lisa.ascii.pgm"
    "brain_398.ascii.pgm"
    "fractal_tree.ascii.pgm"
    "galaxy.ascii.pgm"
    "apollonian_gasket.ascii.pgm"
)

download_one() {
    local url="$1"
    local dest="$2"
    if [[ -f "$dest" ]]; then
        echo "  already exists: $(basename "$dest")"
        return
    fi
    echo "  downloading $(basename "$dest") ..."
    if command -v wget &>/dev/null; then
        wget -q -O "$dest" "$url" || { echo "  warning: wget failed for $url" >&2; rm -f "$dest"; }
    elif command -v curl &>/dev/null; then
        curl -fsSL -o "$dest" "$url" || { echo "  warning: curl failed for $url" >&2; rm -f "$dest"; }
    else
        echo "error: neither wget nor curl found" >&2
        exit 1
    fi
}

if [[ $# -ge 1 ]]; then
    # Single custom URL mode
    url="$1"
    name="${2:-$(basename "$url")}"
    download_one "$url" "$OUT_DIR/$name"
else
    # Download all default test images
    echo "Downloading PGM test images to $OUT_DIR ..."
    for img in "${DEFAULT_IMAGES[@]}"; do
        download_one "$BASE_URL/$img" "$OUT_DIR/$img"
    done
fi

echo "Done."
