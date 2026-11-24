"""Download one WeMM GGUF bundle from the Hub and invoke wemm-native."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def make_command(args, model_dir):
    root = Path(model_dir)
    command = [str(args.runner), '--model', str(root / 'language.gguf'),
               '--mmproj', str(root / 'mmproj.gguf'), '--audio-proj', str(root / 'audio.gguf'),
               '--audio-compute', 'bf16', '--prompt', args.prompt,
               '--output', str(args.output), '--gpu-layers', str(args.gpu_layers),
               '--context', str(args.context), '--batch', '512', '--threads', str(args.threads),
               '--fps', str(args.fps), '--max-frames', str(args.max_frames),
               '--total-pixels', str(args.total_pixels), '--max-new-tokens', str(args.max_new_tokens)]
    for path in args.image:
        command += ['--image', str(path)]
    for path in args.audio:
        command += ['--audio', str(path)]
    if args.video:
        command += ['--video', str(args.video)]
    if args.video_audio:
        command += ['--video-audio', str(args.video_audio)]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', required=True, help='Full Hub ID of one model size/format')
    parser.add_argument('--revision', default='main', help='Pin a Hub commit for reproducible use')
    parser.add_argument('--runner', default=shutil.which('wemm-native'))
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', action='append', type=Path, default=[])
    parser.add_argument('--audio', action='append', type=Path, default=[])
    video = parser.add_mutually_exclusive_group()
    video.add_argument('--video', type=Path, help='Video without audio')
    video.add_argument('--video-audio', type=Path, help='Video with its original audio')
    parser.add_argument('--gpu-layers', type=int, default=0)
    parser.add_argument('--context', type=int, default=8192)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--fps', type=float, default=1)
    parser.add_argument('--max-frames', type=int, default=4)
    parser.add_argument('--total-pixels', type=int, default=262144)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    args = parser.parse_args()
    if not args.runner or not Path(args.runner).is_file():
        parser.error('Install wemm-native and put its bin directory on PATH, or pass --runner')
    if args.output.exists() or Path(str(args.output) + '.evidence').exists():
        parser.error('Choose a new output filename; result/evidence paths already exist')
    if args.video_audio and args.audio:
        parser.error('--video-audio cannot be combined with separate --audio inputs')
    for path in args.image + args.audio + [p for p in (args.video, args.video_audio) if p]:
        if not path.is_file():
            parser.error(f'Missing media file: {path}')
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        parser.error('Install the optional huggingface_hub download helper; see README.md')
    files = ['language.gguf', 'mmproj.gguf', 'audio.gguf']
    model_dir = snapshot_download(repo_id=args.repo_id, revision=args.revision, allow_patterns=files)
    for name in files:
        if not (Path(model_dir) / name).is_file():
            parser.error(f'Incomplete WeMM bundle: missing {name}')
    command = make_command(args, model_dir)
    print(json.dumps({'repo_id': args.repo_id, 'revision': args.revision, 'command': command}), flush=True)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
