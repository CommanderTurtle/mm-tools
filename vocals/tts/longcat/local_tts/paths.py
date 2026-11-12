#!/usr/bin/env python3
"""Locate the LongCat weights without assuming what the checkout is called.

The only fixed point is this project directory. The shared model store may be
named anything, may sit above or beside this project, and its Hugging Face
folders may be renamed or nested, so the store is inferred by walking up from
here and matching files instead of directory names. Setup links the store into
``<project>/models`` so every later run resolves in one hop.

Standard library only: setup and the start scripts call this before the model
dependencies are guaranteed to import.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORE_LINK = PROJECT_ROOT / "models"

ENV_STORE = "LONGCAT_MODELS_DIR"
ENV_MODEL = "LONGCAT_MODEL_PATH"
ENV_TOKENIZER = "LONGCAT_TOKENIZER_PATH"

# Preferred folder names from the model downloader; discovery never requires them.
MODEL_DIR_NAMES = ("meituan-longcat--audiodit-3.5b-tts-text-to-speech-sota",)
TOKENIZER_DIR_NAMES = ("google--umt5-base-tokenizer",)
MODEL_HINTS = ("longcat", "audiodit")
TOKENIZER_HINTS = ("umt5",)
STORE_DIR_NAMES = ("models", "model", "model-store", "model_store", "checkpoints", "weights")

MAX_WALK_DEPTH = 4
MAX_WALK_DIRS = 5000
MAX_ANCESTORS = 5
MAX_ANCESTOR_DEPTH = 3
SKIP_DIR_NAMES = frozenset({"__pycache__", "node_modules", "lost+found"})


@dataclass(frozen=True)
class ModelPaths:
    """Resolved checkpoint and tokenizer locations, plus how they were found."""

    model: Path | None
    tokenizer: Path | None
    store: Path | None
    source: str
    notes: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def explain(self) -> str:
        lines = []
        lines.append(f"checkpoint: {self.model or 'not found'}")
        lines.append(f"tokenizer:  {self.tokenizer or 'not found'}")
        lines.append(f"source:     {self.source}")
        if self.store is not None:
            lines.append(f"store:      {self.store}")
        lines.extend(self.notes)
        return "\n".join(lines)

    def require(self) -> tuple[Path, Path]:
        if self.ready:
            assert self.model is not None and self.tokenizer is not None
            return self.model, self.tokenizer
        missing = []
        if self.model is None:
            missing.append("LongCat checkpoint")
        if self.tokenizer is None:
            missing.append("UMT5 tokenizer")
        raise FileNotFoundError(
            f"{' and '.join(missing)} missing. {self.explain()}\n"
            f"Run ./setupwithuv.sh to link the shared model store into {STORE_LINK}, "
            f"or point {ENV_STORE} at it, or name the folders directly with "
            f"{ENV_MODEL}/{ENV_TOKENIZER}."
        )


def _expand(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _skipped(name: str) -> bool:
    return name.startswith(".") or name in SKIP_DIR_NAMES


def _is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:  # an unreadable neighbour cannot break startup
        return False


def _real(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _walk(root: Path, max_depth: int = MAX_WALK_DEPTH) -> Iterator[Path]:
    """Yield directories below root, without following links or searching forever."""
    visited = 0
    for current, subdirectories, _ in os.walk(root, followlinks=False):
        directory = Path(current)
        if visited >= MAX_WALK_DIRS:
            return
        depth = len(directory.relative_to(root).parts)
        subdirectories[:] = sorted(
            name for name in subdirectories if not _skipped(name) and depth < max_depth
        )
        visited += 1
        yield directory


def _has_checkpoint(directory: Path) -> bool:
    if not (directory / "config.json").is_file():
        return False
    return any(
        (directory / name).is_file()
        for name in ("model.safetensors", "model.safetensors.index.json")
    )


def _has_tokenizer(directory: Path) -> bool:
    if not (directory / "tokenizer_config.json").is_file():
        return False
    return any(
        (directory / name).is_file()
        for name in ("spiece.model", "tokenizer.json", "vocab.json")
    )


def _rank(relative: Path, names: tuple[str, ...], hints: tuple[str, ...]) -> int:
    """0 = the folder the downloader writes, 3 = structure alone proves it."""
    parts = [part.lower() for part in relative.parts]
    if not parts:
        return 0
    if parts[-1] in names or "/".join(parts) in names:
        return 0
    if any(hint in parts[-1] for hint in hints):
        return 1
    if any(hint in part for part in parts for hint in hints):
        return 2
    return 3


def _search(
    root: Path,
    names: tuple[str, ...],
    hints: tuple[str, ...],
    probe: Callable[[Path], bool],
    max_depth: int = MAX_WALK_DEPTH,
) -> tuple[int, Path] | None:
    """Best (rank, path) match inside root; ties break on shallower then lexical."""
    best: tuple[tuple[int, int, str], Path] | None = None
    for directory in _walk(root, max_depth):
        relative = directory.relative_to(root)
        if not probe(directory):
            continue
        key = (_rank(relative, names, hints), len(relative.parts), str(relative))
        if best is None or key < best[0]:
            best = (key, directory)
    return None if best is None else (best[0][0], best[1])


def _levels() -> tuple[Path, ...]:
    """Directories to search, nearest first: this project, then each ancestor."""
    levels: list[Path] = []
    for ancestor in (PROJECT_ROOT, *PROJECT_ROOT.parents):
        levels.append(ancestor)
        # Stop climbing at a top-level directory or after MAX_ANCESTORS levels;
        # sweeping every mount root cannot beat an explicit LONGCAT_MODELS_DIR.
        if len(ancestor.parents) <= 1 or len(levels) > MAX_ANCESTORS:
            break
    return tuple(levels)


def _level_candidates(ancestor: Path) -> Iterator[Path]:
    """Store-shaped folders at one level: the level itself, then its children."""
    for name in STORE_DIR_NAMES:
        yield ancestor / name
    try:
        children = sorted(ancestor.iterdir())
    except OSError:
        children = []
    for child in children:
        if _skipped(child.name) or not _is_dir(child):
            continue
        for name in STORE_DIR_NAMES:
            yield child / name


def _existing(ordered: Iterable[Path]) -> tuple[Path, ...]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in ordered:
        if not _is_dir(candidate):
            continue
        key = _real(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return tuple(unique)


def _override() -> Path | None:
    value = os.environ.get(ENV_STORE, "").strip()
    return _expand(value) if value else None


def store_candidates() -> tuple[Path, ...]:
    """Every model store worth searching, nearest to this project first."""
    ordered: list[Path] = []
    override = _override()
    if override is not None:
        ordered.append(override)
    for ancestor in _levels():
        ordered.extend(_level_candidates(ancestor))
    return _existing(ordered)


def _hinted_store(ancestor: Path) -> tuple[Path, Path] | None:
    """Guess an unusually named store at this level, but only on a real hint."""
    found = _search(
        ancestor, MODEL_DIR_NAMES, MODEL_HINTS, _has_checkpoint, max_depth=MAX_ANCESTOR_DEPTH
    )
    if found is None or found[0] > 2:
        return None
    checkpoint = found[1]
    relative = checkpoint.relative_to(ancestor)
    store = ancestor if len(relative.parts) <= 1 else ancestor / relative.parts[0]
    return store, checkpoint


def _checkpoint_in(root: Path) -> Path | None:
    if not _is_dir(root):
        return None
    found = _search(root, MODEL_DIR_NAMES, MODEL_HINTS, _has_checkpoint)
    return None if found is None else found[1]


def infer_store() -> tuple[Path, Path] | None:
    """Return (store, checkpoint) for the nearest store holding LongCat weights.

    One level is exhausted completely - store-named folders, then a hinted guess
    at an unusually named one - before climbing to the next directory. A sibling
    checkout can therefore never outrank this project's own model store.
    """
    override = _override()
    if override is not None:
        checkpoint = _checkpoint_in(override)
        if checkpoint is not None:
            return override, checkpoint
    for ancestor in _levels():
        for root in _level_candidates(ancestor):
            checkpoint = _checkpoint_in(root)
            if checkpoint is not None:
                return root, checkpoint
        hinted = _hinted_store(ancestor)
        if hinted is not None:
            return hinted
    return None


def _from_env(
    name: str, probe: Callable[[Path], bool], label: str, notes: list[str]
) -> Path | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    path = _expand(value)
    if _is_dir(path) and probe(path):
        return path
    notes.append(f"  ignored {name}={value}: no {label} there")
    return None


def resolve_model_paths() -> ModelPaths:
    """Checkpoint/tokenizer from the environment, else inferred from the store."""
    notes: list[str] = []
    model = _from_env(ENV_MODEL, _has_checkpoint, "checkpoint", notes)
    tokenizer = _from_env(ENV_TOKENIZER, _has_tokenizer, "tokenizer", notes)
    if model is not None and tokenizer is not None:
        store = next((root for root in store_candidates() if _inside(model, root)), None)
        return ModelPaths(model, tokenizer, store, "environment", tuple(notes))

    roots = store_candidates()
    store: Path | None = None
    if model is None:
        inferred = infer_store()
        if inferred is not None:
            store, model = inferred
    else:
        store = next((root for root in roots if _inside(model, root)), None)

    if tokenizer is None:
        ordered = ([store] if store is not None else []) + [
            root for root in roots if root != store
        ]
        for root in ordered:
            found = _search(root, TOKENIZER_DIR_NAMES, TOKENIZER_HINTS, _has_tokenizer)
            if found is None:
                continue
            rank, path = found
            if rank <= 2 or _belongs_to(path, model):
                tokenizer = path
                if store is None:
                    store = root
                break

    if store is None and model is not None:
        store = model.parent
    if model is None or tokenizer is None:
        tried = ", ".join(str(root) for root in roots) or "nothing"
        notes.append(f"  searched stores: {tried}")
        source = "unresolved"
    elif store is not None and _real(store) == _real(STORE_LINK):
        source = f"{STORE_LINK.name} link"
    else:
        source = f"inferred from {store}" if store is not None else "inferred"
    return ModelPaths(model, tokenizer, store, source, tuple(notes))


def _inside(path: Path, root: Path) -> bool:
    try:
        _real(path).relative_to(_real(root))
    except (ValueError, OSError):
        return False
    return True


def _belongs_to(tokenizer: Path, model: Path | None) -> bool:
    """A tokenizer counts unranked only if it travels with the checkpoint."""
    if model is None:
        return False
    return tokenizer == model or model in tokenizer.parents or tokenizer.parent == model.parent


def ensure_store_link(relink: bool = False) -> tuple[bool, str]:
    """Point ``<project>/models`` at the inferred store; never touch real folders."""
    inferred = infer_store()
    if inferred is None:
        return False, (
            f"No model store with LongCat weights found above {PROJECT_ROOT}. "
            f"Download it, then re-run setup or set {ENV_STORE}."
        )
    # Discovery may reach the store through an existing link; the link itself has
    # to be the real directory, or relinking would point it at itself.
    target = _real(inferred[0])
    checkpoint = inferred[1]
    if STORE_LINK.is_symlink():
        current = _real(STORE_LINK)
        if current == target and not relink:
            return True, f"{STORE_LINK} already links {target}"
        STORE_LINK.unlink(missing_ok=True)
        os.symlink(target, STORE_LINK)
        return True, f"{STORE_LINK} re-pointed from {current} to {target}"
    if STORE_LINK.exists():
        if _has_checkpoint(STORE_LINK) or _has_checkpoint(STORE_LINK / checkpoint.name):
            return True, f"{STORE_LINK} holds weights directly; left alone"
        return False, (
            f"{STORE_LINK} is a real directory without LongCat weights; "
            f"expected store: {target}"
        )
    os.symlink(target, STORE_LINK)
    return True, f"{STORE_LINK} -> {target}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve (and, with --link, create) the LongCat model store path."
    )
    parser.add_argument("--link", action="store_true", help="create or repair <project>/models")
    parser.add_argument("--relink", action="store_true", help="re-point an existing link")
    parser.add_argument("--check", action="store_true", help="exit non-zero if unresolved")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.link or args.relink:
        ok, message = ensure_store_link(relink=args.relink)
        print(message, file=None if ok else sys.stderr)
        if not ok:
            return 1
    paths = resolve_model_paths()
    if args.json:
        print(
            json.dumps(
                {
                    "project": str(PROJECT_ROOT),
                    "store": None if paths.store is None else str(paths.store),
                    "model": None if paths.model is None else str(paths.model),
                    "tokenizer": None if paths.tokenizer is None else str(paths.tokenizer),
                    "source": paths.source,
                    "ready": paths.ready,
                    "notes": list(paths.notes),
                },
                indent=2 if not args.check else None,
            )
        )
    elif args.check:
        if not paths.ready:
            print(paths.explain(), file=sys.stderr)
            return 1
        print(f"LongCat checkpoint: {paths.model}")
        print(f"LongCat tokenizer:  {paths.tokenizer} ({paths.source})")
    else:
        print(paths.explain())
    return 0 if paths.ready or not args.check else 1


if __name__ == "__main__":
    raise SystemExit(main())
