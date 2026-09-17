# mm-tools studio runtime

This directory is the shared, model-agnostic browser runtime used by the new
mm-tools projects.  It supplies a single-GPU queue, resumable artifact ledger,
raw streaming uploads, model load/unload controls, server-sent progress events,
local API help, and one polished responsive interface.  Native model calls stay
inside each project's `local_app/adapter.py`; this layer never reimplements a
model or silently changes its settings.

The default listener is `127.0.0.1`, no CORS headers are emitted, no analytics
or remote assets exist, and uploaded/generated files stay beneath the selected
project's ignored `.runtime/studio` directory.  Set `MM_STUDIO_TOKEN` when the
listener is deliberately exposed on a private LAN.

Project manifests describe every real task and control.  The browser keeps
drafts and user presets locally, while the SQLite job ledger and output
manifests are durable on disk.

## Contract checks

The dependency-free suite verifies every studio manifest, adapter route,
field reference, browser control type, unique port, executable contract, and
offline launcher policy without loading a model:

```bash
python -m unittest studio.tests.test_contracts studio.tests.test_runtime
bun build studio/web/app.js --target=browser --outfile=/tmp/mm-tools-studio.js
```

These checks are intentionally separate from model generations.  Adapter and
artifact-specific interaction checks can therefore run cheaply on every
change, while full GPU generations remain deliberate acceptance tests.
