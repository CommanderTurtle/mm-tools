# img2svg Architecture and Operations

## Architecture

```mermaid
flowchart LR
    Input["PNG, JPEG, WebP, GIF, or BMP"] --> Decode["Rust image decoder"]
    Decode --> Cleanup["Background and alpha cleanup"]
    Cleanup --> Palette["Deterministic palette quantization"]
    Palette --> Trace["VTracer"]
    Trace --> SVG["Local SVG paths"]
    Browser["Embedded web studio"] --> Decode
    CLI["Native CLI"] --> Decode
```

One Rust binary owns decoding, tracing, the CLI, and the embedded browser interface. Inputs never leave the host.

## Build and run

```bash
cd ~/multimedia/img2svg
./setupwithuv cpu
./startwithuv
```

The setup creates an empty managed Python environment for portability and builds `target/release/img2svg`. The browser listens on port `4170` unless `IMG2SVG_HOST` or `IMG2SVG_PORT` overrides it.

## Runtime lanes

- Browser: `./startwithuv`
- One conversion:

  ```bash
  target/release/img2svg convert logo.png logo.svg --preset logo --colors 6
  ```

- Recursive conversion:

  ```bash
  target/release/img2svg convert rasters vectors \
    --recursive true --preset poster --colors 12
  ```

Web uploads are bounded before decoding. Output is plain SVG path markup with no scripts or remote assets.
