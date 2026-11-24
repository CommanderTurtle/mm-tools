"""Verify the standalone native payload and its pinned upstream dependency."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_files(root, rows):
    root = Path(root).resolve()
    for name, expected in rows.items():
        path = root / name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Missing or unsafe source file: {name}")
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError(f"Source differs from locked inventory: {name}")


def verify_runtime():
    manifest = json.loads((ROOT / 'source_manifest.json').read_text())
    verify_files(ROOT, manifest['files'])
    return len(manifest['files'])


def verify_upstream(source):
    source = Path(source).resolve()
    lock = json.loads((ROOT / 'llama.cpp.lock.json').read_text())
    revision = subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != lock['revision']:
        raise ValueError(f"llama.cpp HEAD must equal {lock['revision']}")
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', '--quiet', 'HEAD', '--'], check=True)
    verify_files(source, lock['files'])
    return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-source', type=Path)
    args = parser.parse_args()
    result = {'runtime_files_verified': verify_runtime()}
    if args.llama_source:
        result['llama_cpp_revision'] = verify_upstream(args.llama_source)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
