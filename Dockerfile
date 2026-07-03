# Multi-stage Dockerfile for MAFA
# Build: docker build -t mafa:latest .
# Run:   docker run -it --gpus all -p 7860:7860 mafa:latest

FROM nvidia/cuda:11.8.0-runtime-cudnn8-ubuntu22.04 as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git wget curl \
    libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install PyTorch with CUDA 11.8 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p data/chromadb logs models/hf_cache

# Build ChromaDB index
RUN python build_chromadb.py || true

# Set environment
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV CUDA_VISIBLE_DEVICES=0

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/api/predict || exit 1

# Run application
CMD ["python", "app.py"]
