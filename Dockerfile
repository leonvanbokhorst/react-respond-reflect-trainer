FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch and vLLM with compatible versions
RUN pip3 install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip3 install vllm==0.1.4

# Second stage for smaller image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Copy model and scripts
COPY ./rrr_model_vllm /app/rrr_model
COPY ./docker/serve.py /app/serve.py
COPY ./docker/chat_template.jinja /app/chat_template.jinja

# Expose port for API
EXPOSE 9999

# Run the vLLM server
CMD ["python3", "serve.py"] 