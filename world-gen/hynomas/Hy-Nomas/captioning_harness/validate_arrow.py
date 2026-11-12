#!/usr/bin/env python3
"""Inspect source Arrow schemas and decode SFT rows without loading a model."""
import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hymm.dataset.single_file_sft import (
    OPTIONAL_COLUMNS, REQUIRED_COLUMNS, SingleFileSFTDataset,
)


def inspect(path, rows, require_sft):
    with pa.memory_map(str(path), 'r') as source:
        try:
            reader = pa.ipc.open_file(source)
        except pa.ArrowInvalid:
            source.seek(0)
            reader = pa.ipc.open_stream(source)
        names = set(reader.schema.names)
        missing = sorted(REQUIRED_COLUMNS - names)
        extra = sorted(names - REQUIRED_COLUMNS - OPTIONAL_COLUMNS)
        result = {'path': str(path), 'columns': reader.schema.names,
                  'missing_sft_columns': missing, 'extra_sft_columns': extra}
    compatible = not missing and not extra
    result['sft_column_match'] = compatible
    if compatible:
        count = 0
        with pa.memory_map(str(path), 'r') as source:
            reader = pa.ipc.open_file(source)
            for batch_index in range(reader.num_record_batches):
                batch = reader.get_batch(batch_index)
                limit = batch.num_rows if rows == 0 else min(batch.num_rows, rows - count)
                for row in batch.slice(0, limit).to_pylist():
                    sample = SingleFileSFTDataset._decode_row(row, count)
                    for prompt in sample['prompts']:
                        parsed = json.loads(prompt)
                        if (not isinstance(parsed, dict)
                            or not isinstance(parsed.get('segments'), list)
                            or len(parsed['segments']) != 1
                            or not isinstance(parsed['segments'][0], dict)):
                            raise ValueError(f'{path} row {count}: unexpected segment prompt structure')
                    count += 1
                if rows and count >= rows:
                    break
        if count == 0:
            raise ValueError(f'{path}: no rows')
        result.update(validated_rows=count, all_rows_checked=rows == 0)
    else:
        result['note'] = ('Production metadata schema; requires explicit video/latent/camera '
                          'resolution and conversion before SFT loading. annotate accepts local videos only.')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return compatible or not require_sft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-arrow', default=os.getenv('SOURCE_ARROW'),
                        help='Arrow file/directory; overrides default data/ via SOURCE_ARROW')
    parser.add_argument('--rows', type=int, default=3, help='Rows per SFT file; 0 checks all')
    parser.add_argument('--require-sft', action='store_true', help='Fail on incompatible source schemas')
    args = parser.parse_args()
    if args.rows < 0:
        parser.error('--rows must be nonnegative')
    path = Path(args.source_arrow) if args.source_arrow else Path(__file__).resolve().parents[1] / 'data'
    files = sorted(path.glob('*.arrow')) if path.is_dir() else [path]
    if not files:
        parser.error(f'No Arrow files under {path}')
    valid = True
    for file in files:
        valid = inspect(file, args.rows, args.require_sft) and valid
    sys.exit(0 if valid else 1)


if __name__ == '__main__':
    main()
