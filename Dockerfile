# Copyright 2024 ZeroModel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# ZeroModel Full Dockerfile with GPU Support (using uv)
# =============================================================================
# This Dockerfile builds the complete ZeroModel environment including:
# - PyTorch with CUDA support
# - Flash Attention
# - vLLM for inference
# - Ray for distributed computing
# - All ablation study tools
#
# Usage:
#   docker build -t zeromodel:latest .
#   docker run --gpus all -it zeromodel:latest
# =============================================================================

# Use NVIDIA PyTorch base image with CUDA 12.1
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1

# Set working directory
WORKDIR /workspace/zeromodel

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files first for better caching
COPY pyproject.toml requirements.txt .python-version ./

# Install Python dependencies using uv
# Note: flash-attn requires special installation with CUDA
RUN uv pip install ninja packaging && \
    uv pip install flash-attn --no-build-isolation && \
    uv pip install -r requirements.txt

# Copy the entire project
COPY . .

# Install the package in development mode using uv
RUN uv pip install -e ".[test]"

# Set default command
CMD ["/bin/bash"]
