FROM rust:1.93-slim-bookworm AS builder

WORKDIR /usr/src/app
COPY . .

RUN sed -i 's/^Components: main$/Components: main contrib non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources

# install system dependencies needed to compile some crates (openssl, pkg-config, OpenCV, etc.)
# add CUDA toolkit for GPU support
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pkg-config \
        libssl-dev \
        clang \
        libclang-dev \
        llvm-dev \
        cmake \
        libopencv-dev \
        nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/src/app/target \
    cargo build --release

FROM debian:bookworm-slim
RUN sed -i 's/^Components: main$/Components: main contrib non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        python3 \
        python3-pip \
        python3-venv \
        curl \
        libopencv-dev \
        nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# install Hugging Face CLI (requires python)
RUN curl -LsSf https://hf.co/cli/install.sh | bash && \
    # make sure the CLI is on PATH immediately
    echo 'export PATH="/root/.local/bin:$PATH"' >> /etc/profile

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/huggingface \
    hf download lightonai/LightOnOCR-2-1B-bbox-soup tokenizer.json model.safetensors config.json --local-dir models/LightOnOCR
RUN --mount=type=cache,target=/root/.cache/huggingface \
    hf download vikhyatk/moondream1 tokenizer.json model.safetensors --local-dir models/moondream
RUN --mount=type=cache,target=/root/.cache/huggingface \
    hf download Riksarkivet/trocr-base-handwritten-swe config.json tokenizer.json model.safetensors --local-dir models/trocr
COPY --from=builder /usr/src/app/target/release/softwerk-ocr-hackathon /app/softwerk-ocr-hackathon


CMD ["./softwerk-ocr-hackathon"]

