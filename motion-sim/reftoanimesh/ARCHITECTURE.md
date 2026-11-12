# FlowHMR studio

Reference video → SMPL-H motion, with optional PHC humanoid tracking. The inference sources
live in this folder as `flowhmr/`, `dinov3/` and the sibling trees setup links into
`flowhmr/local_app/`.

## Setup

Linux, an NVIDIA GPU, `uv`, Bun, FFmpeg, CMake ≥3.15 and a C++17 compiler. Choose `flowhmr` in
the root model picker and place the SMPL-H file listed in [the model
inventory](../../models/which-ones.txt); gated models require your own Hugging Face account.

```bash cd motion-sim/reftoanimesh ./setupwithuv.sh ./startwithuv.sh
# http://127.0.0.1:8281
```

Setup defaults to PyTorch 2.7.1 / CUDA 12.8 for Blackwell. Set `FLOWHMR_TORCH_PROFILE=upstream`
during setup for the original PyTorch 2.5.1 / CUDA 11.8 lane. Both retain native model precision
and sampling defaults. `FLOWHMR_PORT` changes the loopback port; `FLOWHMR_UPLOAD_MB` sets the
upload limit (4096 MB by default). Paths resolve from this checkout, including spaces.

## Workspace

Drop a video, choose reconstruction settings and select a person track when offered. Hover a
candidate to inspect its source frame, or click a tracked box. The suggested person is accepted
after ten minutes. Processing uses native 30 fps video, camera recovery, SAM features and
FlowHMR generation.

Compare reconstruction and physics side by side; scrub individual frames, change playback speed,
loop, orbit, follow the root, or toggle joints, wireframe and trails. NPZ and FBX downloads
retain native exports. Motion correction uses the released contact/foot-lock solver. **Record
WebM** captures a silent browser comparison of the source and both views; hiding the tab stops
recording.

Saved jobs retain sources, logs and intermediate files under `workspace/`. **Reuse source &
caches** creates another job and reuses only intermediates compatible with its settings. Enable
**Reselect on reuse** to choose another person. Jobs run one at a time in disposable processes.
Cancellation and closing the server release the job process group; interrupted jobs remain
reusable.

## Physics tracking

The teal body uses the released PHC controller's simulated poses and actual collision geometry.
PHC, SMPLSim, SMPLX and Chumpy are committed at this studio root and linked into
`flowhmr/local_app/`; the committed PHC files already carry this studio's four patches, whose
series stays in `patches/`. Its interpreter is separate from reconstruction and starts only
after reconstruction releases its models.

Select `flowhmr-physics` in the picker and supply the three SMPL files listed in the model
inventory. Prepare NVIDIA Isaac Gym Preview 4 in its supported Python 3.8 / PyTorch 1.13 (CUDA
11.7) environment, then:

```bash export FLOWHMR_PHC_PYTHON=/absolute/path/to/phc/bin/python ./setup-physics.sh
./startwithuv.sh ```

Keep that variable set when launching. The native simulator's GPU compatibility is separate from
the Blackwell reconstruction environment. Use the physical GPU index in the studio; keep
`CUDA_VISIBLE_DEVICES` unset for PHC. The optional lane preserves native tracking scores, mean
joint error and survival metrics, and exports body positions, rotations and references as
NPZ/JSON. If physics ends early, the reconstruction and its downloads remain available.
