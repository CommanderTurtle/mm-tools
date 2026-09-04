# Runtime export policy

This repository is a pruned **source** distribution, not a full mirror of its
upstreams or an installed workstation. Static browser assets still need their
local inference servers, project environments, system tools, and model files.

The Git index is the existing-file allowlist. `.gitignore` denies new files by
default, then explicitly permits maintained entrypoints. An ignore rule does
not remove an already tracked file. Never prune by filename alone: dynamically
imported modules, tokenizers, package metadata and notices are dependencies too.

## Check what ships

```bash
cd ~/multimedia
python3 audit-export.py
python3 audit-export.py --export
```

The second command creates a NEW `/tmp/mm-tools-export-*` directory. It copies
only indexed paths, using current working-tree bytes and executable modes,
and checks each copy against its SHA256. It adds `EXPORT-MANIFEST.json` there.
It never copies `.git`, follows symlinks, starts a service, loads weights,
updates the index, modifies production files, or installs packages. Stage new
source files before running it. Unstaged edits to tracked files are included;
this is the same working-tree convention as `git ls-files -z | tar ...`, not a
snapshot of HEAD. Export directories are left for inspection/removal by you.

The audit flags missing build READMEs/licenses, tracked environments/caches,
private `.env` files, checkpoints, and non-regular paths. It is a packaging
check, not an automatic proof of every dynamic import or a secret scanner.

## September 2026 review

- Reviewed the tracked inventory across all 16 runtime projects and models
  tooling, startup/setup references, package metadata, largest assets, and
  MiniMax/Symphony inference imports. No checkpoints or symlinks were tracked.
- Excluded 16 unrelated Comfy tokenizer asset files (22,890,599 bytes). MiniMax
  Music 3 obtains `tokenizer_json` from its own checkpoint. The Qwen3-VL/Krea2
  writing route uses `qwen25_tokenizer`, which remains. The base SD tokenizer
  also remains. Shared Python modules and configuration JSON are retained:
  Comfy imports many model families even when the studio doesn't use them.
- Excluded six Symphony training/data-preparation entrypoint scripts (9,585
  bytes). Kept its RL constants, harmony filters and shared data/model code:
  the generation path really imports those. Kept other potential training
  helpers where removal offers little benefit or requires deeper changes.
- Restored upstream notices/licenses and the Muscriptor/Whisper READMEs that
  their packaging metadata expects. Upstream nested `.gitignore` files can
  still deny these; they were explicitly added to the root index. Their
  licenses remain authoritative; the root license does not replace them.
- Kept ACE-Step's genre vocabulary, model implementations, UI assets, the
  Muscriptor demo track, setup manifests, regression tests, and other shared
  dependencies. These are not disposable vendor bloat.

The 22 excluded files were removed **only from the index**, with their physical
contents hash-checked unchanged. Existing production installations are intact;
historical commits still contain the old files. The export shrank from about
49 MiB to about 27.3 MiB including the restored metadata. Git history does not
shrink unless separately rewritten, which this change does not do.

Validation used the isolated tracked export: real CPU-only MiniMax embedded
tokenizer, Krea2/Qwen and base-SD tokenization, Comfy node imports, Symphony
generation imports, and source syntax checks. No inference weights were loaded
and no production service was restarted. This does not claim a full fresh
install or GPU-generation test of every application.

## Maintaining the boundary

For additions, inspect `git status --short` and `git diff --cached --stat`;
allowlist only required source and assets. Preserve vendor notices. For
removals, establish the runtime dependency path, use exact `git rm --cached`
targets, verify the disk files remain, and test from the isolated export—not
from the production checkout where ignored files could hide missing inputs.
Repeat after upstream updates, especially when changing MiniMax's model types.
