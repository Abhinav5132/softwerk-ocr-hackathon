# Softwerk-Ocr-Hackathon

## Build

Default build (CPU only, works on macOS without CUDA/OpenCV installed):

```bash
cargo build
```

Build with CUDA support (Linux/NVIDIA environment with CUDA toolkit installed):

```bash
cargo build --features cuda
```

Build with Metal support (Apple Silicon/macOS GPU):

```bash
cargo build --features metal
```

Build with OpenCV line segmentation support (requires system OpenCV):

```bash
cargo build --features opencv
```

You can combine features, for example:

```bash
cargo build --features "metal opencv"
```
