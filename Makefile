.PHONY: all build install run run-embed run-api dev clean test help

# Default target
all: build

# Build Rust binary
build:
	cargo build --release

# Install Python dependencies
install:
	uv sync

# Run embedding server (pre-loads model)
run-embed:
	uv run python/embedding_server.py

# Run API server (uses HTTP embedder)
run-api:
	MUSUBI_EMBED_URL=http://127.0.0.1:8081/embed cargo run --release

# Run API server with Python embedder (no pre-loading)
run-api-python:
	cargo run --release

# Development: run both servers (embedding first, then API)
# Usage: make dev
# Note: Run in separate terminals or use 'make run-embed' and 'make run-api' separately
dev:
	@echo "Start servers in separate terminals:"
	@echo "  Terminal 1: make run-embed"
	@echo "  Terminal 2: make run-api"

# Run tests
test:
	cargo test

# Clean build artifacts
clean:
	cargo clean

# Show help
help:
	@echo "Available targets:"
	@echo "  build         - Build Rust binary (release)"
	@echo "  install       - Install Python dependencies"
	@echo "  run-embed     - Start embedding server (port 8081)"
	@echo "  run-api       - Start API server with HTTP embedder (port 8080)"
	@echo "  run-api-python- Start API server with Python embedder (slower)"
	@echo "  dev           - Show instructions for development setup"
	@echo "  test          - Run tests"
	@echo "  clean         - Clean build artifacts"
	@echo ""
	@echo "Quick start:"
	@echo "  1. make install"
	@echo "  2. make run-embed  (terminal 1)"
	@echo "  3. make run-api    (terminal 2)"
