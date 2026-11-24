"""Fetch pinned llama.cpp, build the native engine and install into a prefix."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from verify import ROOT, verify_runtime, verify_upstream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--llama-source', type=Path, default=ROOT / 'vendor/llama.cpp')
    parser.add_argument('--build-dir', type=Path)
    parser.add_argument('--prefix', type=Path)
    parser.add_argument('--jobs', type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument('--cuda-architectures', help='CMake architectures, e.g. 89 for L20 or 90 for H20')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without fetching or building')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    for program in ('git', 'cmake'):
        if not shutil.which(program):
            parser.error(f'{program} is required')
    verify_runtime()
    lock = json.loads((ROOT / 'llama.cpp.lock.json').read_text())
    source = args.llama_source.absolute()
    build = (args.build_dir or ROOT / ('build-' + args.backend)).absolute()
    prefix = (args.prefix or ROOT / ('install-' + args.backend)).absolute()
    if source.is_symlink() or build.is_symlink() or prefix.is_symlink():
        parser.error('Use real source, build and install directories')
    commands = []
    if not source.exists():
        commands += [
            ['git', 'init', str(source)],
            ['git', '-C', str(source), 'remote', 'add', 'origin', lock['repository']],
            ['git', '-C', str(source), 'fetch', '--depth=1', 'origin', lock['revision']],
            ['git', '-C', str(source), 'checkout', '--detach', 'FETCH_HEAD'],
        ]
    configure = ['cmake', '-S', str(ROOT), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                 '-DLLAMA_CPP_DIR=' + str(source), '-DCMAKE_INSTALL_PREFIX=' + str(prefix),
                 '-DGGML_CUDA=' + ('ON' if args.backend == 'cuda' else 'OFF')]
    if args.cuda_architectures:
        configure.append('-DCMAKE_CUDA_ARCHITECTURES=' + args.cuda_architectures)
    build_commands = [configure, ['cmake', '--build', str(build), '--parallel', str(args.jobs)],
                      ['cmake', '--install', str(build)]]
    if args.dry_run:
        print(json.dumps({'commands': commands + build_commands}, indent=2))
        return
    for command in commands:
        subprocess.run(command, check=True)
    verify_upstream(source)
    for command in build_commands:
        subprocess.run(command, check=True)
    print(f'Installed: {prefix / "bin/wemm-native"}')


if __name__ == '__main__':
    main()
