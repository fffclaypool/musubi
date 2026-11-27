.PHONY: all build test lint fmt example example-docs install bench-download bench bench-full clean help

# Default target
all: build

# Build the library
build:
	cargo build --release

# Run tests
test:
	cargo test

# Format check + clippy (same as CI)
lint:
	cargo fmt --all --check
	cargo clippy --all-targets -- -D warnings

# Format code
fmt:
	cargo fmt --all

# Random vectors: build an index and compare with brute force
example:
	cargo run --release --example basic

# Install Python dependencies (only needed for example-docs)
install:
	uv sync

# Embed data/sample.csv and search similar documents
example-docs:
	uv run python/embed_csv.py --input data/sample.csv --output data/embeddings.jsonl
	cargo run --release --example similar_docs -- data/embeddings.jsonl

# Download benchmark data
bench-download:
	./scripts/download_ann_data.sh

# Run benchmark (10k subset). Requires libhdf5-dev.
bench:
	cargo run --release -p musubi-bench --bin bench_ann -- \
		--dataset data/ann/glove-100-angular.hdf5 \
		--train-limit 10000 \
		--test-limit 1000 \
		--save-index

# Run benchmark (full dataset). Requires libhdf5-dev.
bench-full:
	cargo run --release -p musubi-bench --bin bench_ann -- \
		--dataset data/ann/glove-100-angular.hdf5 \
		--save-index

# Clean build artifacts
clean:
	cargo clean

# Show help
help:
	@echo "Available targets:"
	@echo "  build          - Build the library (release)"
	@echo "  test           - Run tests"
	@echo "  lint           - Format check + clippy"
	@echo "  fmt            - Format code"
	@echo "  example        - Run the basic example (random vectors)"
	@echo "  install        - Install Python dependencies (for example-docs)"
	@echo "  example-docs   - Embed data/sample.csv and find similar documents"
	@echo "  bench-download - Download ANN-Benchmarks data"
	@echo "  bench          - Run benchmark (10k subset, needs libhdf5-dev)"
	@echo "  bench-full     - Run benchmark (full dataset, needs libhdf5-dev)"
	@echo "  clean          - Clean build artifacts"
