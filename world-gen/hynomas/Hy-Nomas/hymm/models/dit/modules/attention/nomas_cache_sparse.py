"""Inference-only spatial routing for rectangular segment-AR KV attention.

The cache remains raster ordered and complete. Only attention reads are sparse.
All ranges for a query share one softmax, including its permitted text keys.
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import json
import os
import time

import torch
from hymm.utils.inference_metrics import profiled


_AUDIT = Counter()


def reset_sparse_audit():
    _AUDIT.clear()


def sparse_audit():
    return {key: _AUDIT[key] for key in (
        'router_calls', 'torch_router_calls', 'cute_router_calls', 'triton_router_calls',
        'sparse_flex_calls', 'sparse_magi_calls', 'dense_flex_calls',
        'text_flex_calls', 'cache_write_calls')}


def note_dense_flex():
    _AUDIT['dense_flex_calls'] += 1


@dataclass
class CacheVideo:
    start: int
    image_tokens: int
    end: int
    time_start: int
    height: int
    width: int
    # None defers pooling of the actual committed K until the next route refresh.
    prototypes: torch.Tensor | None


def _reuse_key(context, cache_valid_len, q, k):
    cfg = context['config']
    return (tuple(context['grid']), context['time_start'], cache_valid_len,
            tuple(q.shape), tuple(k.shape), str(q.device), q.dtype, k.dtype,
            json.dumps(dict(cfg), sort_keys=True, default=list))


def _route_reuse_state(cache, context, cache_valid_len, q, k,
                       kv_mask, txt_mask, batch, update_cache):
    """Reuse only selected block IDs; values and queries are always current.

    Masks are compared by value so in-place changes and guidance batch changes
    invalidate the route. The first denoise step clears the owning layer cache.
    """
    refresh = tuple(context['config'].get('route_refresh_steps', ()))
    if not refresh:
        cache.pop('route_reuse', None)
        return None, None, None
    step = int(context.get('step_index', -1)) + 1
    if step <= 0:
        raise ValueError('Route reuse requires a 0-based step_index in context')
    key = _reuse_key(context, cache_valid_len, q, k)
    entries = cache.setdefault('route_reuse', {})
    old = entries.get(batch)
    scheduled = not update_cache and step in refresh
    valid = (old is not None and old['key'] == key and old['step'] <= step
             and torch.equal(old['kv_mask'], kv_mask[batch])
             and torch.equal(old['txt_mask'], txt_mask[batch]))
    stats = cache.setdefault('route_reuse_stats', {})
    if valid and not scheduled:
        name = 'cache_write_reuses' if update_cache else 'denoise_reuses'
        stats[name] = stats.get(name, 0) + 1
        return old['selected'].to(device=q.device, dtype=torch.int64), key, step
    # No consumer needs the stale route after invalidation/refresh.
    if context['config'].get('memory_efficient', False):
        entries.pop(batch, None)
    name = 'scheduled_refreshes' if scheduled else 'fallback_refreshes'
    stats[name] = stats.get(name, 0) + 1
    steps = stats.setdefault('refresh_steps', {})
    label = 'cache_write' if update_cache else str(step)
    steps[label] = steps.get(label, 0) + 1
    return None, key, step


def trim_cache(cache, valid_len, evict=0):
    records = cache.get('sparse_video', [])
    kept = []
    for record in records:
        if record.end > valid_len or record.end <= evict:
            continue
        if record.start < evict:
            raise ValueError('Sparse eviction must align with a cached chunk')
        if evict:
            record.start -= evict
            record.end -= evict
        kept.append(record)
    cache['sparse_video'] = kept
    return kept


@lru_cache(maxsize=64)
@torch.inference_mode(False)
def spatial_layout(frames, height, width, device, block_size=8):
    """Pack real pixels into tiles; edge tiles remain short and never attend padding."""
    if block_size != 8 or min(frames, height, width) < 1:
        raise ValueError("Sparse routing requires positive 8x8 tile geometry")
    grid = torch.arange(frames * height * width, device=device).reshape(frames, height, width)
    parts, lengths, prototype_slots = [], [], []
    cursor = 0
    for frame in range(frames):
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                tile = grid[frame, y:y+block_size, x:x+block_size]
                h, w = tile.shape
                parts.append(tile.flatten())
                lengths.append(h*w)
                # Replicate edge pixels only while pooling routing prototypes.
                # Attention Q/K/V retain exactly the real, unpadded pixels.
                yi = torch.arange(block_size, device=device).clamp(max=h-1)
                xi = torch.arange(block_size, device=device).clamp(max=w-1)
                prototype_slots.append(cursor + (yi[:,None]*w + xi[None,:]).flatten())
                cursor += h*w
    return torch.cat(parts), tuple(lengths), torch.stack(prototype_slots)


def spatial_order(frames, height, width, device, block_size=8):
    return spatial_layout(frames, height, width, device, block_size)[0]


def pool_blocks(x, mode='2x2', block_size=8, prototype_slots=None):
    if mode != '2x2' or block_size != 8:
        raise ValueError("Sparse router requires 8x8 tiles and 2x2 pooling")
    heads, dim = x.shape[-2:]
    if prototype_slots is not None:
        x = x.index_select(0, prototype_slots.flatten())
    return x.float().reshape(-1,4,2,4,2,heads,dim).mean(dim=(2,4)).reshape(
        -1,16,heads,dim).permute(2,0,1,3).contiguous()


def bounded_pool_blocks(x, mode, block_size, slots):
    """Pool independent spatial blocks with bounded FP32 conversion storage."""
    if slots is None:
        return pool_blocks(x, mode, block_size, slots)
    heads, dim = x.shape[-2:]
    count = slots.shape[0]
    result = torch.empty((heads, count, 16, dim), device=x.device, dtype=torch.float32)
    blocks = max(1, (64 * 2**20) // (64 * heads * dim * 6))
    for first in range(0, count, blocks):
        indices = slots[first:first+blocks].reshape(-1)
        packed = x.index_select(0, indices)
        pooled = pool_blocks(packed, mode, block_size)
        result[:, first:first+blocks].copy_(pooled)
        del packed, pooled
    return result


_compiled_pool = None


def inference_pool_blocks(x, mode_name, block_size, prototype_slots):
    from hymm.models.dit.modules.attention.nomas_inference_optim import mode, compare, note
    policy=mode('POOL')
    if policy=='off' or not x.is_cuda or torch.is_grad_enabled():
        return pool_blocks(x,mode_name,block_size,prototype_slots)
    global _compiled_pool
    if _compiled_pool is None:
        _compiled_pool=torch.compile(pool_blocks,fullgraph=True,dynamic=True)
    result=_compiled_pool(x,mode_name,block_size,prototype_slots)
    note('pool.hit')
    if policy=='shadow':
        ref=pool_blocks(x,mode_name,block_size,prototype_slots)
        compare('pool',ref,result)
        return ref
    return result


@lru_cache(maxsize=1)
def cute_router_module():
    import importlib.util
    from pathlib import Path
    import sys
    path = Path(__file__).with_name('cute_route')
    spec = importlib.util.spec_from_file_location('nomas_cute_route_impl', path/'__init__.py',
                                                submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def route(qp,kp,q_ids,k_ids,blocks_per_frame,config,stats=None):
    if config.get('router_mode') not in ('cute_bf16', 'torch_reference', 'triton_bf16'):
        raise ValueError('Unknown sparse router_mode')
    if config.get('selection_scope') != 'global_fraction':
        raise ValueError('This sparse path requires selection_scope=global_fraction')
    if (int(config.get('temporal_block_size',1)) != 1
            or config.get('spatial_shift_mode','none') != 'none'
            or int(config.get('query_time_group_size',1)) != 1):
        raise ValueError('Global routing requires unshifted 8x8x1 blocks')
    stats=stats if stats is not None else {}
    if config.get('router_mode') == 'triton_bf16':
        from .triton_route import full_route
        result=full_route(qp,kp,q_ids,k_ids,blocks_per_frame,config)
        stats['triton_router_calls']=stats.get('triton_router_calls',0)+1
        _AUDIT['triton_router_calls'] += 1
        _AUDIT['router_calls'] += 1
        return result
    if config.get('router_mode') == 'torch_reference':
        from .torch_route import full_route
        result=full_route(qp,kp,q_ids,k_ids,blocks_per_frame,config)
        stats['torch_router_calls']=stats.get('torch_router_calls',0)+1
        _AUDIT['torch_router_calls'] += 1
        _AUDIT['router_calls'] += 1
        return result
    module=cute_router_module()
    reason,workspace=module.eligibility(qp,kp,q_ids,k_ids,blocks_per_frame,config)
    if reason:
        raise RuntimeError('Global CuTe route unavailable: '+reason)
    from hymm.models.dit.modules.attention.nomas_inference_optim import mode
    result=module.full_route(qp,kp,q_ids,k_ids,blocks_per_frame,config,
                             _eligibility_checked=mode('ROUTER')!='off')
    stats['full_hits']=stats.get('full_hits',0)+1
    stats['cute_hits']=stats.get('cute_hits',0)+1
    stats['cute_router_calls']=stats.get('cute_router_calls',0)+1
    _AUDIT['cute_router_calls'] += 1
    _AUDIT['router_calls'] += 1
    stats['workspace_guard_bytes']=workspace
    return result


def ranges_for_head(selected, nq, nv, nhisttxt, ntxt, offsets, windows, device,
                    q_lengths=None, k_lengths=None):
    """H3-style int32 GPU ranges, retaining Nomas cache/action visibility.

    Allocate final arrays once; broadcast Q block intervals instead of doing
    arithmetic on expanded int64 arrays. Text descriptors stay compact on the
    host and are expanded on-device, without Python per-block lists/uploads.
    Range ordering matches the original builder, including text query rows.
    """
    n, count = selected.shape
    q_lengths = tuple(q_lengths or (64,)*n)
    k_lengths = tuple(k_lengths or (64,)*(nv//64))
    if sum(q_lengths) != nq or sum(k_lengths) != nv or len(q_lengths) != n:
        raise ValueError("Sparse tile lengths do not cover real Q/K video tokens")
    qend = torch.tensor(q_lengths, dtype=torch.int32, device=device).cumsum(0).to(torch.int32)
    qstart = torch.cat((qend.new_zeros(1), qend[:-1]))
    kend = torch.tensor(k_lengths, dtype=torch.int32, device=device).cumsum(0).to(torch.int32)
    kstart = torch.cat((kend.new_zeros(1), kend[:-1]))
    qr = [torch.stack((qstart,qend),-1)[:,None,:].expand(n,count,2).reshape(-1,2)]
    kr = [torch.stack((kstart[selected],kend[selected]),-1).reshape(-1,2)]
    qboundaries = [0]
    for length in q_lengths:
        qboundaries.append(qboundaries[-1]+length)
    def emit(a,b,c,d):
        if b<=a or d<=c:
            return
        if a<nq:
            video_end=min(b,nq)
            intervals=[(max(a,left),min(video_end,right)) for left,right in zip(qboundaries,qboundaries[1:])
                       if right>a and left<video_end]
            if intervals:
                qr.append(torch.tensor(intervals,dtype=torch.int32,device=device))
                kr.append(torch.tensor([[c,d]],dtype=torch.int32,device=device).expand(len(intervals),2))
            a=video_end
        if b>a:
            qr.append(torch.tensor([[a,b]],dtype=torch.int32,device=device))
            kr.append(torch.tensor([[c,d]],dtype=torch.int32,device=device))
    emit(nq,nq+ntxt,0,nv+nhisttxt)
    emit(0,nq,nv,nv+nhisttxt)
    current_text_start=nv+nhisttxt
    if offsets is None:
        emit(0,nq+ntxt,current_text_start,current_text_start+ntxt)
    else:
        offs=[max(0,min(int(o),ntxt)) for o in offsets]
        if not offs or offs!=sorted(offs) or offs[-1]!=ntxt:
            raise ValueError("Sparse action offsets must cover valid text")
        emit(0,nq+ntxt,current_text_start,current_text_start+offs[0])
        for left,right,action in windows:
            if action>=0:
                emit(left,right,current_text_start+offs[action],current_text_start+offs[action+1])
        for action in range(len(offs)-1):
            emit(nq+offs[action],nq+offs[action+1],current_text_start+offs[action],current_text_start+offs[action+1])
    qr,kr=torch.cat(qr).contiguous(),torch.cat(kr).contiguous()
    return qr,kr,torch.zeros(qr.shape[0],dtype=torch.int32,device=device)


@lru_cache(maxsize=1)
def batched_attention_module():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('nomas_batched_magi_impl',
        Path(__file__).with_name('nomas_batched_magi.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@profiled("sparse_attention")
def cache_sparse_attention(q_full, full_k, full_v, kv_mask, cur_txt_mask,
                           cache, cache_valid_len, update_cache, cache_txt,
                           context, offsets=None, windows=None):
    if torch.is_grad_enabled():
        raise RuntimeError('Nomas cache sparse attention is inference-only')
    config = context['config']
    pooling = config.get('router_pooling', '2x2')
    head_packing = (config.get('memory_efficient', False) and
                    context.get('backend', 'magi') == 'flex' and
                    config.get('flex_sparse_layout', 'compact') == 'aligned')
    pool = bounded_pool_blocks if config.get('memory_efficient', False) else inference_pool_blocks
    spatial_block = int(config.get('spatial_block_size', 8))
    frames, height, width = context['grid']
    nq = frames*height*width
    tpf = height*width
    if spatial_block != 8:
        raise ValueError("Sparse routing requires spatial_block_size=8")
    bpf = ((height+7)//8)*((width+7)//8)
    device = q_full.device
    if q_full.shape[1] != nq+cur_txt_mask.shape[1]:
        raise ValueError('SP padding/true query length mismatch in sparse attention')
    if not cache_txt:
        raise ValueError('Initial Nomas sparse implementation requires seg_ar (cache_txt=True)')
    order, q_lengths, prototype_slots = spatial_layout(frames,height,width,device,spatial_block)
    records = trim_cache(cache, cache_valid_len)
    if sum(r.end-r.start for r in records) != cache_valid_len:
        raise ValueError('Sparse metadata does not cover the existing KV cache')
    if any((r.height, r.width) != (height, width) for r in records):
        raise ValueError('Sparse cache spatial geometry changed')
    end = full_k.shape[1]
    out = None if head_packing else torch.zeros_like(q_full)
    new_protos = []
    timings = {}
    tick = time.perf_counter
    started = tick()
    for batch in range(q_full.shape[0]):
        reused, reuse_key, route_step = _route_reuse_state(
            cache, context, cache_valid_len, q_full, full_k,
            kv_mask, cur_txt_mask, batch, update_cache)
        refresh_route = reused is None
        txt_idx = cur_txt_mask[batch].bool().nonzero().flatten()
        qorder = torch.cat((order, nq+txt_idx))
        q = None if head_packing else q_full[batch].index_select(0, qorder)
        current_protos = qprotos = None
        if refresh_route:
            if head_packing:
                # Compose spatial and prototype indices before gathering: avoid
                # full-head packed Q/current K and allocate only pool chunks.
                raster_slots = order[prototype_slots]
                current_protos = pool(full_k[batch, cache_valid_len:cache_valid_len+nq],
                                      pooling, spatial_block, raster_slots)
                qprotos = pool(q_full[batch, :nq], pooling, spatial_block, raster_slots)
            else:
                current = full_k[batch, cache_valid_len:cache_valid_len+nq].index_select(0, order)
                current_protos = pool(current, pooling, spatial_block, prototype_slots)
                qprotos = pool(q[:nq], pooling, spatial_block, prototype_slots)
                if config.get('memory_efficient', False):
                    del current
            if reuse_key is not None:
                rs = cache['route_reuse_stats']
                rs['pool_calls'] = rs.get('pool_calls', 0) + 2
        if update_cache and not config.get('route_refresh_steps'):
            new_protos.append(current_protos)
        video_orders, proto_parts, block_ids, text_orders, all_k_lengths = [], [], [], [], []
        for record in records:
            nt = record.image_tokens//tpf
            history_order, history_lengths, _ = spatial_layout(nt,height,width,device,spatial_block)
            idx = history_order + record.start
            visible = kv_mask[batch].index_select(0,idx).reshape(nt,tpf)
            if not torch.equal(visible.any(-1),visible.all(-1)):
                raise ValueError("Partially visible video frames are unsupported")
            keep_frames = visible.all(-1)
            keep_blocks = keep_frames[:,None].expand(-1,bpf).reshape(-1)
            video_orders.append(idx.reshape(nt,tpf)[keep_frames].flatten())
            all_k_lengths.extend(length for length,keep in zip(history_lengths,keep_blocks.tolist()) if keep)
            if refresh_route:
                if record.prototypes is None:
                    # Pool committed (not step-6) K: its values changed at KV write.
                    slots = spatial_layout(nt,height,width,device,spatial_block)[2]
                    if head_packing:
                        raster_slots = idx[slots]
                        record.prototypes = torch.stack([
                            pool(full_k[b], pooling, spatial_block, raster_slots)
                            for b in range(full_k.shape[0])]).detach()
                    else:
                        record.prototypes = torch.stack([
                            pool(full_k[b].index_select(0, idx),
                                                  pooling, spatial_block, slots)
                            for b in range(full_k.shape[0])]).detach()
                    if reuse_key is not None:
                        rs = cache['route_reuse_stats']
                        rs['history_pool_calls'] = rs.get('history_pool_calls', 0) + full_k.shape[0]
                # Triton already rounds FP32 prototypes to BF16 before its dot.
                if config.get('memory_efficient', False) and config.get('router_mode') == 'triton_bf16':
                    record.prototypes = record.prototypes.to(
                        device='cpu' if config.get('memory_cache_device', 'cpu') == 'cpu' else device,
                        dtype=torch.bfloat16)
                cb = 0 if record.prototypes.shape[0] == 1 else batch
                # Select after restoring the current layer; other layers stay off device.
                history_protos = record.prototypes[cb].to(device=device, dtype=torch.float32)
                proto_parts.append(history_protos[:, keep_blocks])
                del history_protos
            ids = torch.arange(nt*bpf, device=device)+record.time_start*bpf
            block_ids.append(ids[keep_blocks])
            tidx = torch.arange(record.start+record.image_tokens, record.end, device=device)
            text_orders.append(tidx[kv_mask[batch].index_select(0, tidx)])
        video_orders.append(cache_valid_len+order)
        all_k_lengths.extend(q_lengths)
        if refresh_route:
            proto_parts.append(current_protos)
        qids = torch.arange(frames*bpf, device=device)+context['time_start']*bpf
        block_ids.append(qids)
        kvidx = torch.cat(video_orders)
        histtxt = torch.cat(text_orders) if text_orders else order[:0]
        korder = torch.cat((kvidx, histtxt, cache_valid_len+nq+txt_idx))
        if not config.get('memory_efficient', False):
            k = full_k[batch].index_select(0, korder)
            v = full_v[batch].index_select(0, korder)
        kp = torch.cat(proto_parts, dim=1) if refresh_route else None
        if head_packing and refresh_route:
            # torch.cat owns its storage: source prototypes are no longer needed.
            proto_parts.clear()
            current_protos = None
        kids = torch.cat(block_ids)
        ready = tick()
        timings['pack_pool'] = timings.get('pack_pool', 0)+ready-started
        temporal = int(config.get('temporal_block_size',1))
        if temporal != 1 or spatial_block != 8 or pooling != '2x2':
            raise ValueError('This sparse path requires 8x8x1 blocks and 2x2 pooling')
        timing_recorder = context.get('timing_recorder')
        route_event_start = None
        if temporal == 1 and timing_recorder is not None and os.environ.get('HY_INFER_LINEAR_PROFILE_STEP', '0') != '0':
            route_event_start = torch.cuda.Event(enable_timing=True)
            route_event_start.record()
        if reused is not None:
            selected = reused
        elif context.get("dense", False):
            selected = torch.arange(kp.shape[1],device=device).view(1,1,-1).expand(
                qprotos.shape[0],qprotos.shape[1],-1).contiguous()
        else:
            selected = route(
                qprotos, kp, qids, kids, bpf, config,
                cache.setdefault("router_stats",{}))
        if reuse_key is not None and refresh_route:
            cached_selected = selected
            if config.get('memory_efficient', False):
                dtype = torch.uint16 if kids.numel() <= 65536 else (
                    torch.int32 if kids.numel() <= 2**31 else torch.int64)
                cached_selected = selected.to(dtype=dtype)
                if config.get('memory_cache_device', 'cpu') == 'cpu':
                    cached_selected = cached_selected.cpu()
            cache['route_reuse'][batch] = dict(
                key=reuse_key, step=route_step, selected=cached_selected.detach(),
                kv_mask=kv_mask[batch].detach().clone(),
                txt_mask=cur_txt_mask[batch].detach().clone())
        if route_event_start is not None:
            route_event_end = torch.cuda.Event(enable_timing=True)
            route_event_end.record()
            timing_recorder('double_blocks_router', route_event_start, route_event_end)
        if config.get('memory_efficient', False):
            # Prototypes and their concatenation are dead after selection.
            del kp, qprotos, current_protos
            proto_parts.clear()
            # The aligned path gathers only the active head group below.
            if not head_packing:
                k = full_k[batch].index_select(0, korder)
                v = full_v[batch].index_select(0, korder)
        routed = tick()
        timings['route'] = timings.get('route', 0)+routed-ready
        offs = offsets[batch] if offsets is not None else None
        wins = [[int(a)*tpf, int(b)*tpf, int(c)] for a,b,c in (windows or [])]
        if offs is not None and (not wins or wins[0][0] != 0 or wins[-1][1] != nq):
            raise ValueError('Sparse action windows must cover the current video')
        backend = context.get('backend', 'magi')
        if backend == 'flex':
            layout = config.get('flex_sparse_layout', 'compact')
            if layout == 'compact':
                from . import nomas_batched_flex as from_module
            elif layout == 'aligned':
                from . import nomas_aligned_flex as from_module
            else:
                raise ValueError('Unknown flex_sparse_layout: '+str(layout))
        elif backend == 'magi':
            from_module = batched_attention_module()
        else:
            raise ValueError('Unknown sparse attention backend: '+backend)
        before_kernel = tick()
        magi_event_start = None
        if timing_recorder is not None and os.environ.get('HY_INFER_LINEAR_PROFILE_STEP', '0') != '0':
            magi_event_start = torch.cuda.Event(enable_timing=True)
            magi_event_start.record()
        def variable_ranges(*args):
            return ranges_for_head(*args,q_lengths=q_lengths,k_lengths=all_k_lengths)
        if head_packing:
            # Slice heads before index_select: never materialize packed full-head KV.
            head_chunk = int(config.get('memory_attention_heads', 1))
            if head_chunk < 1:
                raise ValueError('memory_attention_heads must be positive')
            if out is None:
                out = torch.zeros_like(q_full)
            if config.get('memory_adaptive_heads', False):
                gather_plan = from_module.composed_gather_plan(
                    q_lengths, all_k_lengths, txt_idx.numel(), histtxt.numel(),
                    qorder, korder, q_full.device)
                head_chunk=from_module.choose_head_group(q_full.shape[2],head_chunk,
                    q_lengths,all_k_lengths,txt_idx.numel(),histtxt.numel(),q_full.device)
                astats=cache.setdefault('attention_stats', {})
                group_key='head_group_'+str(head_chunk)
                astats[group_key]=astats.get(group_key,0)+1
                for first_head in range(0, q_full.shape[2], head_chunk):
                    last_head = min(first_head + head_chunk, q_full.shape[2])
                    hs = slice(first_head, last_head)
                    qh = q_full[batch, :, hs]
                    kh = full_k[batch, :, hs]
                    vh = full_v[batch, :, hs]
                    part = from_module.attention(qh, kh, vh, selected[hs],
                        nq, kvidx.numel(), histtxt.numel(), txt_idx.numel(), offs,
                        wins, variable_ranges, cache.setdefault('attention_stats', {}),
                        q_lengths=q_lengths, k_lengths=all_k_lengths, gather_plan=gather_plan)
                    out[batch, :, hs].index_copy_(0, qorder, part)
                    del qh, kh, vh, part
                del gather_plan
            else:
                for first_head in range(0, q_full.shape[2], head_chunk):
                    last_head = min(first_head + head_chunk, q_full.shape[2])
                    hs = slice(first_head, last_head)
                    qh = q_full[batch, :, hs].index_select(0, qorder)
                    kh = full_k[batch, :, hs].index_select(0, korder)
                    vh = full_v[batch, :, hs].index_select(0, korder)
                    part = from_module.attention(qh, kh, vh, selected[hs],
                        nq, kvidx.numel(), histtxt.numel(), txt_idx.numel(), offs,
                        wins, variable_ranges, cache.setdefault('attention_stats', {}),
                        q_lengths=q_lengths, k_lengths=all_k_lengths)
                    out[batch, :, hs].index_copy_(0, qorder, part)
                    del qh, kh, vh, part
        else:
            packed = from_module.attention(q, k, v, selected, nq, kvidx.numel(),
                histtxt.numel(), txt_idx.numel(), offs, wins, variable_ranges,
                cache.setdefault('attention_stats', {}),
                q_lengths=q_lengths,k_lengths=all_k_lengths)
        _AUDIT['sparse_'+backend+'_calls'] += 1
        _AUDIT['cache_write_calls'] += int(update_cache)
        if backend=='flex':
            _AUDIT['text_flex_calls'] += int(txt_idx.numel()>0)
        if backend=='magi':
            astats=cache['attention_stats']
            astats['sparse_magi_calls']=astats.get('sparse_magi_calls',0)+1
        if magi_event_start is not None:
            magi_event_end = torch.cuda.Event(enable_timing=True)
            magi_event_end.record()
            timing_recorder('double_blocks_sparse_'+backend, magi_event_start, magi_event_end)
        timings[backend] = timings.get(backend, 0)+tick()-before_kernel
        if not head_packing:
            out[batch].index_copy_(0, qorder, packed)
            del packed, k, v
        started = tick()
    if update_cache:
        # Scheduled reuse defers prototype materialization to the next segment.
        stored = None if config.get('route_refresh_steps') else torch.stack(new_protos).detach()
        cache['sparse_video'].append(CacheVideo(cache_valid_len, nq, end, context['time_start'],
                                               height, width, stored))
        cache.pop('route_reuse', None)
    stats = cache.setdefault('sparse_stats', dict(calls=0, writes=0))
    stats['calls'] += 1
    stats['writes'] += int(update_cache)
    for name, seconds in timings.items():
        stats[name] = stats.get(name, 0.)+seconds
    if context['layer'] == 0 and (stats['calls'] == 1 or update_cache):
        print('NOMAS_SPARSE '+json.dumps(dict(rank=int(os.environ.get('RANK', 0)),
            layer=context['layer'], q_grid=[frames,height,width], history_tokens=cache_valid_len,
            kv_video_frames=int(kvidx.numel()//tpf), blocks_per_frame=bpf,
            selection_scope=config.get('selection_scope','per_time_group'),
            global_keep_fraction=config.get('global_keep_fraction'),
            selected_video_blocks_per_query=int(selected.shape[-1]) if selected is not None else None,
            top_k=None if config.get('selection_scope')=='global_fraction' else min(int(config.get('top_k_per_kv_frame',5)),bpf),
            kv_time_group_size=int(config.get('kv_time_group_size',1)),
            top_k_per_group=None if config.get('selection_scope')=='global_fraction' else min(int(config.get('top_k_per_kv_frame',5)),bpf)*int(config.get('kv_time_group_size',1)),
            router_mode=config.get('router_mode','chunked'), router_stats=cache.get('router_stats',{}),
            router_pooling=pooling, spatial_block_size=spatial_block,
            attention_backend=backend, attention_stats=cache.get('attention_stats',{}),
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
            temporal_stats=cache.get('temporal_stats'), temporal_block_size=int(config.get('temporal_block_size',1)),
            block4x4x4_stats=cache.get('block4x4x4_stats'),
            local_heads=q_full.shape[2], update_cache=update_cache,
            backend=('flex_head_sparse' if backend=='flex' else 'magi_range_merge_sparse_load'),
            timing_mode='host_enqueue', stats=stats)), flush=True)
    return out
