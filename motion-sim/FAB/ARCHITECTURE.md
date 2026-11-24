# FAB

Motion atlas studio: one take in, a tracked body back out — fitted mesh (SMPL
or SOMA), body parts, point contacts, a retargeted Unitree G1 reference at
50 Hz, and optional head-mounted FAH hands.

## Setup and start

```bash
./setupwithuv.sh
./startwithuv.sh
```

Choose `fab` in the model downloader. Setup installs the source committed in
this lane, `fab/` and `fah/`, into a CUDA 12.8 uv environment, plus a second
`.hands` environment for FAH because its package declares top-level
`bench`/`train` modules. `models/fab_assets.py` then builds the G1 rig and the
MANO pair from the donor studios, so nothing is downloaded for the runtime
itself. Open `http://127.0.0.1:8299`; `GRADIO_SERVER_PORT` changes the port.

## Checkpoints

Weights and body models live in `models/fab` (`FAB_HOME`), the SAM-3D-Body
checkpoint is shared with `models/flowhmr`, and SOMA assets come from
`motion-sim/nvidia-sim/SOMA-X`. `env.sh` resolves all of it and tolerates a
per-lane copy of either store.

## Lane contents

- `fab/` — Fast and Accurate Body Estimation, `ec65caf40632b4d36eb48f58ab45ea56d3cba21a`
- `fah/` — the companion hand estimator, `da03aabcd99cf61c7ea179c1ddca84f75be14962`
- `state.py` — records this lane's source; every job record carries that identity,
  so a result names the exact source that produced it.
