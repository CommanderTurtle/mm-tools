# Video Compact Architecture and Operations

## Architecture

```mermaid
flowchart LR
    Upload["Local video"] --> API["FastAPI workbench"]
    API --> Probe["ffprobe inspection"]
    Probe --> Plan["Compression settings"]
    Plan --> FFmpeg["Local FFmpeg encode"]
    FFmpeg --> Output["Local output"]
```

The browser streams media to a local workspace, probes it with the host FFmpeg toolchain, and exposes trim, scaling, frame-rate, codec, audio, metadata, and fast-start controls. No file is submitted to an external service.

## Setup and run

```bash
cd ~/multimedia/video-compact
./setupwithuv cpu
./startwithuv
```

The service listens on port `8240` unless `.env` overrides it. `VIDEO_COMPACT_MIN_FREE_GIB` reserves local disk space before accepting work.

## Runtime lanes

- Browser workbench: `./startwithuv`.
- CLI engine:

  ```bash
  uv run python engine.py input.mov output.mp4 \
    --codec av1 --engine nvenc --quality balanced --width 1920
  ```

Hardware and software encoder availability comes from the installed FFmpeg build.
