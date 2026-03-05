ARG BUILD_TARGET=cuda

#--model downloader--
FROM debian:bookworm-slim AS model-downloader

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir huggingface_hub[cli] --break-system-packages
ARG MODEL_CACHE_VERSION=1

RUN hf download lightonai/LightOnOCR-2-1B-bbox-soup \
    tokenizer.json model.safetensors config.json \
    --local-dir /models/LightOnOCR

RUN hf download vikhyatk/moondream1 \
    tokenizer.json model.safetensors \
    --local-dir /models/moondream

RUN hf download Riksarkivet/trocr-base-handwritten-hist-swe-2 \
    config.json tokenizer.json model.safetensors \
    --local-dir /models/trocr


#--cuda builder--
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl pkg-config libssl-dev clang libclang-dev cmake \
    libopencv-dev python3 python3-pip poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

WORKDIR /app
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/src/app/target \
    cargo build --release --features opencv cuda && \
    cp target/release/softwerk-ocr-hackathon /usr/local/bin/softwerk-ocr-hackathon


#--cuda runtime image--
FROM nvidia/cuda:13.1.1-runtime-ubuntu22.04 AS runtime-cuda
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils python3 python3-pip libssl3 libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=model-downloader /models ./models
COPY --from=builder /usr/local/bin/softwerk-ocr-hackathon ./softwerk-ocr-hackathon
CMD ["./softwerk-ocr-hackathon"]

#--CPU builder--
FROM debian:bookworm-slim AS builder-cpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl pkg-config libssl-dev clang libclang-dev cmake \
    libopencv-dev python3 python3-pip poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

WORKDIR /app
COPY . .

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/src/app/target \
    cargo build --release --features opencv && \
    cp target/release/softwerk-ocr-hackathon /usr/local/bin/softwerk-ocr-hackathon

# -- CPU runtime --
FROM debian:bookworm-slim AS runtime-cpu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils python3 python3-pip libssl3 libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=model-downloader /models ./models
COPY --from=builder-cpu /usr/local/bin/softwerk-ocr-hackathon ./softwerk-ocr-hackathon
CMD ["./softwerk-ocr-hackathon"]

FROM runtime-${BUILD_TARGET}