FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"
WORKDIR /app

# Build-time dependencies (minimal + Rust/reqwest build requirements)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    python3 \
    python3-venv \
    python3-pip \
    build-essential \
    pkg-config \
    libssl-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy manifests first for better layer caching
COPY pyproject.toml uv.lock ./
COPY Cargo.toml Cargo.lock ./

# Build Python env at image build time (required)
RUN uv sync --frozen

# Copy source and build Rust API binary at image build time (required)
COPY src ./src
COPY python ./python
RUN cargo build --release --bin musubi


FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:/app/.venv/bin:${PATH}"
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    python3 \
    python3-venv \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Runtime needs uv to support `uv run python/embedding_server.py`
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Python project files + locked environment
COPY --from=builder /app/pyproject.toml /app/uv.lock ./
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/python ./python

# Keep binary path compatible with `./target/release/musubi`
RUN mkdir -p /app/target/release
COPY --from=builder /app/target/release/musubi /app/target/release/musubi

EXPOSE 8080 8081

# Default runtime: Rust API server
CMD ["./target/release/musubi"]
