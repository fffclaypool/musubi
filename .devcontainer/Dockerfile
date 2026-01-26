FROM mcr.microsoft.com/devcontainers/rust:1-bookworm

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    cmake \
    pkg-config \
    libssl-dev \
    lldb \
    curl \
    gnupg \
    python3 \
    python3-venv \
    python3-pip \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

ENV UV_INSTALL_DIR="/usr/local/bin"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code
