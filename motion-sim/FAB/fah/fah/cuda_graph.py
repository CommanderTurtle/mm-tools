import torch


class GraphedForward:
    """Wraps fn(batch) -> dict of tensors as CUDA graphs, one per input-shape signature.

    Tensor entries of `batch` are copied into static buffers and the captured graph is replayed;
    non-tensor entries are baked in at capture time and must not change between calls. Outputs are
    the graph's static buffers: consume (e.g. .cpu()) them before the next call."""

    def __init__(self, fn, warmup=3):
        self.fn = fn
        self.warmup = warmup
        self.graphs = {}
        self.pool = None

    def _key(self, batch):
        return tuple((k, tuple(v.shape), v.dtype) if torch.is_tensor(v) else (k, v)
                     for k, v in sorted(batch.items()) if torch.is_tensor(v) or isinstance(v, (bool, int, float, str)))

    def _capture(self, batch):
        static_in = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.warmup):
                self.fn(dict(static_in))
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out = self.fn(dict(static_in))
        if self.pool is None:
            self.pool = graph.pool()
        static_out = {k: v for k, v in out.items() if torch.is_tensor(v)}
        return graph, static_in, static_out

    @torch.no_grad()
    def __call__(self, batch):
        key = self._key(batch)
        if key not in self.graphs:
            self.graphs[key] = self._capture(batch)
        graph, static_in, static_out = self.graphs[key]
        for k, v in batch.items():
            if torch.is_tensor(v):
                static_in[k].copy_(v, non_blocking=True)
        graph.replay()
        return static_out
