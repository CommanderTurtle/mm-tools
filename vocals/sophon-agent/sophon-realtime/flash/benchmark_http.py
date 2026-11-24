"""Benchmark Flash HTTP streaming audio; not a Live/Realtime benchmark."""
import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import random
import time
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf


async def sse_events(stream, max_event_bytes=16 * 1024 * 1024):
    """Parse LF/CRLF SSE incrementally without StreamReader's line limit.

    Limit raw event bytes (including comments/fields) and incomplete lines.
    UTF-8 is decoded only after a complete line, so transport splits are safe.
    """
    if max_event_bytes < 1:
        raise ValueError('max_event_bytes must be positive')
    pending = bytearray()
    data = []
    event_bytes = 0
    async for chunk in stream.iter_chunked(16 * 1024):
        pending.extend(chunk)
        while True:
            boundary = pending.find(b'\n')
            if boundary < 0:
                if event_bytes + len(pending) > max_event_bytes:
                    raise ValueError('SSE event exceeds max_event_bytes')
                break
            size = boundary + 1
            if event_bytes + size > max_event_bytes:
                raise ValueError('SSE event exceeds max_event_bytes')
            raw = bytes(pending[:boundary])
            del pending[:size]
            line = raw.removesuffix(b'\r').decode('utf-8')
            event_bytes += size
            if not line:
                if data:
                    yield '\n'.join(data)
                data = []
                event_bytes = 0
            elif line.startswith('data:'):
                value = line[5:]
                data.append(value[1:] if value.startswith(' ') else value)
    # Unclosed events are intentionally not dispatched.


def playback_stall(chunks, initial_buffer=0.):
    if not chunks:
        return 0.
    end = chunks[0]['arrival_s'] + initial_buffer
    stall = 0.
    for c in chunks:
        stall += max(0., c['arrival_s'] - end)
        end = max(end, c['arrival_s']) + c['audio_s']
    return stall


def load_inputs(wav, manifest, language, prompts):
    if manifest:
        path = Path(manifest).resolve()
        items = json.loads(path.read_text(encoding='utf-8'))
        root = path.parent
    else:
        items = [{'id': 'single', 'wav': wav, 'language': language}]
        root = Path.cwd()
    if not isinstance(items, list) or not items:
        raise ValueError('manifest must be a nonempty list')
    prepared = []
    ids = set()
    for item in items:
        ident = item['id']
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError('input IDs must be unique nonempty strings')
        ids.add(ident)
        lang = item.get('language', language)
        if lang not in prompts:
            raise ValueError(f'missing system prompt for {lang}')
        p = root / item['wav']
        raw = p.read_bytes()
        audio, rate = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f'invalid input: {ident}')
        # Request declares WAV; reject mislabeled input before timed requests.
        with sf.SoundFile(io.BytesIO(raw)) as f:
            if f.format not in ('WAV', 'WAVEX', 'RF64'):
                raise ValueError(f'input must be WAV: {ident}')
        prepared.append(dict(id=ident, language=lang, sha256=hashlib.sha256(raw).hexdigest(),
                             duration_s=len(audio)/rate, data=base64.b64encode(raw).decode()))
    return prepared


async def request(session, url, body, ident, initial_buffer=0., threshold=.02, max_event_bytes=16 * 1024 * 1024):
    start = time.perf_counter()
    row = dict(request_id=ident, success=False, error_type=None, error_message=None,
               first_audio_received_s=None, first_audio_decoded_s=None,
               first_audible_s=None, audio_duration_s=0., audio_chunk_count=0,
               chunk_trace=[])
    try:
        async with session.post(url + '/v1/chat/completions', json=body) as response:
            response.raise_for_status()
            if 'text/event-stream' not in response.headers.get('Content-Type', ''):
                raise ValueError('expected text/event-stream response')
            done = False
            async for data in sse_events(response.content, max_event_bytes):
                received = time.perf_counter() - start
                if data.strip() == '[DONE]':
                    done = True
                    break
                event = json.loads(data)
                if 'error' in event:
                    raise RuntimeError(str(event['error']))
                if event.get('modality') != 'audio':
                    continue
                for choice in event.get('choices', []):
                    value = (choice.get('delta') or {}).get('content')
                    if not value:
                        continue
                    pcm, rate = sf.read(io.BytesIO(base64.b64decode(value, validate=True)),
                                        dtype='float32', always_2d=True)
                    if not len(pcm) or not np.isfinite(pcm).all():
                        raise ValueError('invalid output audio')
                    decoded = time.perf_counter() - start
                    if row['first_audio_received_s'] is None:
                        row['first_audio_received_s'] = received
                        row['first_audio_decoded_s'] = decoded
                    if row['first_audible_s'] is None and np.max(np.abs(pcm)) > threshold:
                        row['first_audible_s'] = received
                    duration = len(pcm)/rate
                    row['chunk_trace'].append(dict(arrival_s=received, decoded_s=decoded, audio_s=duration))
                    row['audio_duration_s'] += duration
                    row['audio_chunk_count'] += 1
            if not done:
                raise RuntimeError('stream ended without [DONE]')
            if not row['audio_chunk_count']:
                raise RuntimeError('no audio')
            row['success'] = True
    except Exception as e:
        row['error_type'] = type(e).__name__
        row['error_message'] = str(e)
    row['e2e_s'] = time.perf_counter() - start
    row['stall_s'] = playback_stall(row['chunk_trace'], initial_buffer)
    return row


def summary(rows, wall, peak, settings, model, inputs):
    good = [r for r in rows if r['success']]
    result = dict(schema_version=1, settings=settings, model=model,
                  inputs=[{k:v for k,v in x.items() if k != 'data'} for x in inputs],
                  success=len(good), total=len(rows), success_rate=len(good)/len(rows),
                  wall_s=wall, peak_inflight=peak, requests_per_s=len(good)/wall,
                  total_output_audio_s=sum(r['audio_duration_s'] for r in good))
    result['audio_seconds_per_s'] = result['total_output_audio_s']/wall
    result['errors'] = {}
    for r in rows:
        if not r['success']:
            k=r['error_type']; result['errors'][k]=result['errors'].get(k,0)+1
    for field in ('first_audio_received_s','first_audio_decoded_s','first_audible_s','stall_s','e2e_s'):
        values=[r[field] for r in good if r[field] is not None]
        result[field] = dict(zip(('p50','p90','p95'),map(float,np.percentile(values,[50,90,95])))) if values else None
    return result


async def run(a):
    prompts=json.loads(Path(a.system_prompt).read_text(encoding='utf-8'))
    inputs=load_inputs(a.wav, a.manifest, a.language, prompts)
    out=Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=a.timeout),
                                    connector=aiohttp.TCPConnector(limit=0)) as session:
        async with session.get(a.url.rstrip('/')+'/v1/models') as r:
            r.raise_for_status(); models=(await r.json())['data']
        model=a.model
        if not model:
            if len(models)!=1:
                raise ValueError('--model required when server exposes multiple models')
            model=models[0]['id']
        rng=random.Random(a.seed)
        def body(x):
            return dict(model=model, stream=True, modalities=['text','audio'], language=x['language'],
                        voice=a.voice, max_tokens=a.max_tokens, temperature=.7, top_p=.9,
                        messages=[dict(role='system',content=prompts[x['language']]),
                                  dict(role='user',content=[dict(type='text',text='<|audio|>'),
                                    dict(type='input_audio',input_audio=dict(format='wav',data=x['data']))])])
        bodies=[body(x) for x in inputs]
        for i in range(a.warmup):
            r=await request(
                session, a.url.rstrip('/'), bodies[i % len(inputs)], f'warmup-{i}',
                initial_buffer=a.initial_buffer, threshold=a.audible_threshold,
                max_event_bytes=getattr(a, 'max_event_bytes', 16 * 1024 * 1024),
            )
            if not r['success']:
                raise RuntimeError(f'warmup failed: {r["error_message"]}')
        indices=[rng.randrange(len(inputs)) if a.selection=='random' else i%len(inputs) for i in range(a.requests)]
        sem=asyncio.Semaphore(a.concurrency or 1)
        active=peak=0
        begin=time.perf_counter()
        rows=[]
        with (out/'requests.jsonl').open('w',encoding='utf-8') as f:
            async def execute(i):
                nonlocal active,peak
                j=indices[i]; x=inputs[j]
                target=begin+(a.arrival_window*i/max(1,a.requests-1) if a.arrival_window is not None else 0)
                active+=1;peak=max(peak,active)
                actual=time.perf_counter()
                try:
                    r=await request(session,a.url.rstrip('/'),bodies[j],str(i),a.initial_buffer,a.audible_threshold,getattr(a, "max_event_bytes", 16 * 1024 * 1024))
                finally:
                    active-=1
                r.update(input_id=x['id'],input_sha256=x['sha256'],input_duration_s=x['duration_s'],
                         actual_start_s=actual-begin,scheduled_start_s=target-begin if a.arrival_window is not None else None,
                         send_lag_s=max(0.,actual-target) if a.arrival_window is not None else None)
                rows.append(r);f.write(json.dumps(r,allow_nan=False)+'\n');f.flush()
            async def launch(i):
                if a.arrival_window is not None:
                    target=begin+a.arrival_window*i/max(1,a.requests-1)
                    while time.perf_counter() < target:
                        await asyncio.sleep(max(0.,target-time.perf_counter()))
                    await execute(i)
                else:
                    async with sem:
                        await execute(i)
            await asyncio.gather(*(launch(i) for i in range(a.requests)))
        result=summary(rows,time.perf_counter()-begin,peak,vars(a),model,inputs)
        (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps(result,indent=2,allow_nan=False))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--wav');g.add_argument('--manifest')
    p.add_argument('--url',default='http://127.0.0.1:8000')
    p.add_argument('--model');p.add_argument('--language',default='zh')
    p.add_argument('--voice',default='chelsie')
    p.add_argument('--system-prompt',default=str(Path(__file__).parent/'config/system_prompt.json'))
    p.add_argument('--selection',choices=('round-robin','random'),default='round-robin')
    p.add_argument('--seed',type=int,default=0)
    load=p.add_mutually_exclusive_group()
    load.add_argument('--concurrency',type=int);load.add_argument('--arrival-window',type=float)
    p.add_argument('--requests',type=int,default=30);p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--timeout',type=float,default=300)
    p.add_argument('--max-tokens',type=int,default=300)
    p.add_argument('--initial-buffer',type=float,default=0)
    p.add_argument('--audible-threshold',type=float,default=.02)
    p.add_argument('--max-event-bytes',type=int,default=16 * 1024 * 1024)
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    for name in ('timeout', 'initial_buffer', 'audible_threshold', 'arrival_window'):
        value = getattr(a, name)
        if value is not None and not math.isfinite(value):
            p.error('--' + name.replace('_', '-') + ' must be finite')
    if a.max_event_bytes < 1:
        p.error('--max-event-bytes must be positive')
    if a.requests<1 or a.warmup<0 or (a.concurrency is not None and a.concurrency<1) or a.timeout<=0 or a.max_tokens<1 or a.initial_buffer<0 or not 0<=a.audible_threshold<=1 or (a.arrival_window is not None and a.arrival_window<0):
        p.error('invalid numeric setting')
    if a.concurrency is None and a.arrival_window is None:
        a.concurrency=1
    return a


if __name__=='__main__':
    asyncio.run(run(parse_args()))
