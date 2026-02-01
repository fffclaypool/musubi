#!/bin/bash
# Download ANN-Benchmarks datasets for evaluation
#
# Usage:
#   ./scripts/download_ann_data.sh           # Download glove-100-angular (default)
#   ./scripts/download_ann_data.sh sift      # Download sift-128-euclidean
#   ./scripts/download_ann_data.sh all       # Download all supported datasets

set -e

DATA_DIR="data/ann"
mkdir -p "$DATA_DIR"

download_dataset() {
    local name=$1
    local url=$2
    local file="$DATA_DIR/$name.hdf5"

    if [ -f "$file" ]; then
        echo "Dataset already exists: $file"
    else
        echo "Downloading $name..."
        wget -q --show-progress -O "$file" "$url"
        echo "Downloaded: $file"
    fi
}

case "${1:-glove}" in
    glove|glove-100-angular)
        download_dataset "glove-100-angular" "http://ann-benchmarks.com/glove-100-angular.hdf5"
        ;;
    sift|sift-128-euclidean)
        download_dataset "sift-128-euclidean" "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
        ;;
    fashion|fashion-mnist-784-euclidean)
        download_dataset "fashion-mnist-784-euclidean" "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5"
        ;;
    all)
        download_dataset "glove-100-angular" "http://ann-benchmarks.com/glove-100-angular.hdf5"
        download_dataset "sift-128-euclidean" "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
        download_dataset "fashion-mnist-784-euclidean" "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5"
        ;;
    *)
        echo "Unknown dataset: $1"
        echo "Supported: glove, sift, fashion, all"
        exit 1
        ;;
esac

echo "Done."
