"""Explicit, model-scoped inference policies. No environment-variable overrides."""
from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps
import json
import torch

_NAMES = ('router', 'pool', 'guards', 'ranges', 'text', 'magi_static',
          'head_view')
_ACTIVE = ContextVar('nomas_sparse_optim', default=({}, None))


def validate(config):
    cfg = dict(config or {})
    unknown = set(cfg) - set(_NAMES) - {'enabled', 'text_kchunk'}
    if unknown:
        raise ValueError('Unknown sparse_optim options: '+str(sorted(unknown)))
    if type(cfg.get('enabled', False)) is not bool:
        raise ValueError('sparse_optim.enabled must be boolean')
    for key in _NAMES:
        if cfg.get(key, 'off') not in ('off', 'on', 'shadow'):
            raise ValueError('sparse_optim.'+key+' must be off/on/shadow (quote YAML values)')
    chunk = cfg.get('text_kchunk', 0)
    if type(chunk) is not int or chunk < 0:
        raise ValueError('text_kchunk must be a nonnegative integer')
    return cfg


def mode(name):
    cfg, _ = _ACTIVE.get()
    return cfg.get(name.lower(), 'off') if cfg.get('enabled', False) else 'off'


def option(name, default=None):
    return _ACTIVE.get()[0].get(name, default)


def note(name, count=1):
    _, counters = _ACTIVE.get()
    if counters is not None:
        counters[name] += count


@contextmanager
def scope(config, counters=None):
    token = _ACTIVE.set((config, counters))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def scoped_forward(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        cfg = getattr(self, 'sparse_optim', {})
        if not cfg.get('enabled', False):
            return fn(self, *args, **kwargs)
        if torch.is_grad_enabled() or self.training:
            raise RuntimeError('Sparse optimizations are inference-only')
        with scope(cfg, self.sparse_optim_counts):
            return fn(self, *args, **kwargs)
    return wrapped


def compare(name, reference, candidate, *, exact=False):
    a, b = reference.float(), candidate.float()
    d = a-b
    row = dict(kind=name, max_abs=d.abs().max().item(), mean_abs=d.abs().mean().item(),
               relative_l2=(d.norm()/a.norm().clamp_min(1e-20)).item(),
               equal=torch.equal(a,b), finite=bool(torch.isfinite(b).all()))
    note(name+'.shadow')
    print('NOMAS_OPT_COMPARE '+json.dumps(row), flush=True)
    if not row['finite'] or (exact and not row['equal']):
        raise RuntimeError('Sparse optimization comparison failed: '+name)
    if not exact:
        torch.testing.assert_close(candidate, reference, rtol=0.02, atol=0.02,
                                   msg='Sparse optimization shadow mismatch: '+name)
    return row
