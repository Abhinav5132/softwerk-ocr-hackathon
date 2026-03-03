FROM rust:1.93-slim-bookworm AS builder

WORKDIR /usr/src/app
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/src/app/target \
    cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && \
    apt-get install -y --no-install-recommends poppler-utils && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://hf.co/cli/install.sh | bash

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/huggingface \
    hf download lightonai/LightOnOCR-2-1B-bbox-soup tokenizer.json model.safetensors config.json --local-dir models/LightOnOCR
RUN --mount=type=cache,target=/root/.cache/huggingface \
    hf download vikhyatk/moondream2 tokenizer.json model.safetensors --local-dir models/moondream
COPY --from=builder /usr/src/app/target/release/softwerk-ocr-hackathon /app/softwerk-ocr-hackathon


CMD ["./softwerk-ocr-hackathon"]
