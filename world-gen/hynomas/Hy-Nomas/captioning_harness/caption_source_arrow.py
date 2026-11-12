#!/usr/bin/env python3
"""Caption original videos referenced by an Arrow file, without VAE decoding."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from urllib.parse import urlsplit, unquote

import pyarrow as pa
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from captioning_harness.prepare_sft import annotate, sample_video, timestamp


def run_row(index, row, args):
    directory = args.output_dir / f'{index:04d}'
    directory.mkdir(parents=True, exist_ok=True)
    video_path = row.get('video_path')
    if not isinstance(video_path, str) or not video_path.strip():
        raise ValueError(f'row {index}: missing video_path')
    parsed = urlsplit(video_path)
    remote = parsed.scheme in {'http', 'https'}
    if parsed.scheme and not remote and parsed.scheme != 'file':
        raise ValueError(f'row {index}: unsupported video_path scheme {parsed.scheme}')
    local = None
    if not remote:
        local = Path(unquote(parsed.path) if parsed.scheme == 'file' else video_path).expanduser()
        if not local.is_absolute():
            local = args.source_arrow.parent / local
        local = local.resolve()
        if not local.is_file():
            raise ValueError(f'row {index}: local video does not exist: {local}')
    sample_id = row.get('sample_id') or str(index)
    start = float(row.get('video_start_sec') or 0)
    duration = row.get('video_duration_sec')
    if not math.isfinite(start) or start < 0:
        raise ValueError(f'row {index}: invalid video_start_sec')
    downloaded = directory / 'source.mp4' if remote else local
    if remote and not downloaded.exists():
        temporary = downloaded.with_suffix('.download')
        print(f'row {index}: downloading original video', flush=True)
        try:
            with requests.get(video_path, stream=True, timeout=(15, 90)) as response:
                if response.status_code != 200:
                    raise ValueError(f'row {index}: video HTTP status {response.status_code}')
                with temporary.open('wb') as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
            os.replace(temporary, downloaded)
        except requests.RequestException as exc:
            temporary.unlink(missing_ok=True)
            raise ValueError(f'row {index}: download failed ({type(exc).__name__})') from None
    if duration is None:
        probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                '-of', 'json', str(downloaded)], capture_output=True, text=True, check=True)
        duration = float(json.loads(probe.stdout)['format']['duration']) - start
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f'row {index}: invalid video_duration_sec')
    config = {'row_index': index, 'sample_id': sample_id, 'source_arrow': str(args.source_arrow),
              'source_video_path': video_path,
              'clip_start_sec': start, 'clip_duration_sec': duration,
              'model': args.model, 'annotation_fps': args.fps,
              'max_frames_per_segment': args.max_frames_per_segment}
    config_path = directory / 'source.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(f'row {index}: changed input/settings; choose a new output directory')
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    clip = directory / 'clip.mp4'
    if not clip.exists():
        temporary_clip = directory / 'clip.tmp.mp4'
        result = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(start), '-i', str(downloaded),
                                 '-t', str(duration), '-an', '-c:v', 'libx264', '-preset', 'fast',
                                 '-crf', '18', str(temporary_clip)], capture_output=True, text=True)
        if result.returncode:
            raise ValueError(f'row {index}: ffmpeg failed: {result.stderr[-1000:]}')
        os.replace(temporary_clip, clip)
    if not (directory / 'frames.jpg').exists():
        from PIL import Image, ImageDraw
        frames = sample_video(clip, args.fps, 512, max(32.0, duration + 1))
        positions = sorted({round(i * (len(frames) - 1) / 11) for i in range(12)})
        sheet = Image.new('RGB', (1024, ((len(positions) + 3) // 4) * 280), 'white')
        draw = ImageDraw.Draw(sheet)
        for n, position in enumerate(positions):
            x, y = (n % 4) * 256, (n // 4) * 280
            sheet.paste(frames[position].resize((256, 256)), (x, y))
            draw.text((x + 4, y + 258), timestamp(position / args.fps), fill='black')
        sheet.save(directory / 'frames.jpg', quality=85)
        del frames
    annotations = directory / 'annotations.jsonl'
    settings = SimpleNamespace(input=str(clip), annotations=str(annotations),
        api_base=args.api_base, api_key=args.api_key, model=args.model, fps=args.fps,
        size=512, segment_seconds=4.0, max_seconds=max(32.0, duration + 1),
        max_segments=max(8, int(duration / 4) + 2), max_frames_per_segment=args.max_frames_per_segment,
        segment_max_tokens=8192, final_max_tokens=16384, timeout=1200,
        retries=2, thinking=False)
    annotate(settings)
    record = json.loads(annotations.read_text().splitlines()[-1])
    record.update(config)
    record['sample_id'] = sample_id
    record['source_kind'] = 'original_video'
    record['original_prompts'] = row.get('prompts')
    record_path = directory / 'caption.json'
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    lines = [f'Row {index} | {sample_id}', 'Visual input: original source video', '']
    for language in ('chinese_json', 'english_json'):
        caption = record['annotation'][language]
        lines.extend([language, f"{caption['perspective_view']} | {caption['Style']}"])
        for subject in caption['subject_list']:
            lines.append(f"{subject['subject_id']} {subject['display_name']}: {subject['subject_attribute']}")
        for segment in caption['segments']:
            lines.append(f"\n{segment['start_time']} - {segment['end_time']}")
            lines.append(segment['visual_description_long'] or segment['description_change'])
            for action in segment['contained_actions']:
                lines.append(f"{action['start_time']} - {action['end_time']}: {action['action_description']}")
        lines.append('')
    (directory / 'caption.txt').write_text('\n'.join(lines), encoding='utf-8')
    print(f'row {index}: ORIGINAL VIDEO CAPTION SAVED', flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-arrow', type=Path, default=os.getenv('SOURCE_ARROW', 'data/stage7_sft_aesthetic512_current_schema.arrow'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--rows', default='0', help='Comma-separated source row indices, or all')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--model', default=os.getenv('CAPTION_MODEL', 'qwen3.8-27b'))
    parser.add_argument('--api-base', default=os.getenv('CAPTION_API_BASE', 'http://127.0.0.1:30000/v1'))
    parser.add_argument('--api-key', default=os.getenv('CAPTION_API_KEY', ''))
    parser.add_argument('--fps', type=int, default=16)
    parser.add_argument('--max-frames-per-segment', type=int, default=12)
    args = parser.parse_args()
    if min(args.workers, args.fps, args.max_frames_per_segment) <= 0:
        parser.error('workers, fps and frame count must be positive')
    try:
        wanted = None if args.rows == 'all' else {int(x) for x in args.rows.split(',')}
    except ValueError:
        parser.error('--rows must be all or comma-separated integers')
    if wanted is not None and (not wanted or min(wanted) < 0):
        parser.error('row indices must be nonnegative')
    args.source_arrow = args.source_arrow.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    aggregate = args.output_dir / 'captions.jsonl'
    already_saved = {json.loads(line)['row_index'] for line in aggregate.read_text().splitlines()} if aggregate.exists() else set()
    tasks = []
    index = 0
    with pa.memory_map(str(args.source_arrow), 'r') as source:
        reader = pa.ipc.open_file(source)
        if 'video_path' not in reader.schema.names:
            parser.error('source Arrow is missing video_path (URL or local path)')
        columns = [name for name in reader.schema.names if name in {'video_path', 'video_start_sec', 'video_duration_sec', 'sample_id', 'prompts'}]
        for b in range(reader.num_record_batches):
            batch = reader.get_batch(b).select(columns)
            for row in batch.to_pylist():
                if wanted is None or index in wanted:
                    tasks.append((index, row))
                index += 1
    if wanted is not None and wanted - {i for i, _ in tasks}:
        parser.error('requested row index is outside the Arrow')
    succeeded, failed = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_row, i, row, args): i for i, row in tasks}
        for future in as_completed(futures):
            i = futures[future]
            try:
                record = future.result()
                if i not in already_saved:
                    with aggregate.open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                        handle.flush()
                        os.fsync(handle.fileno())
                    already_saved.add(i)
                succeeded += 1
            except Exception as exc:
                failed += 1
                with (args.output_dir / 'failures.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps({'row_index': i, 'error': str(exc)}, ensure_ascii=False) + '\n')
                print(f'row {i}: FAILED: {exc}', flush=True)
            (args.output_dir / 'status.json').write_text(json.dumps(
                {'requested': len(tasks), 'completed_this_run': succeeded, 'failed_this_run': failed,
                 'running': succeeded + failed < len(tasks)}, indent=2))
    print(f'DONE: {succeeded} captions, {failed} failures', flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
