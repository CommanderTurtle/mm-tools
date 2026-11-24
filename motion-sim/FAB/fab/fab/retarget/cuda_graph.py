"""The retargeter's optimisation loops as CUDA graphs (fab-retarget's default; --no_graph runs the plain loops). Each iteration's
forward + loss + backward is captured once and replayed; the Adam step stays eager (the same optimizer object and python
learning rates, reading the graph's static .grad tensors), so the results are bitwise those of the plain loops."""
import torch


def graph_steps(params, opt, loss_fn, iters, warm=3, pre=None):
    """iters Adam steps of loss_fn() over params: warm eager steps on a side stream, then one captured forward + backward
    (grads zeroed in place inside the graph) replayed per step, each followed by the eager opt.step(); pre(it) runs before
    each step (e.g. the learning-rate schedule, as in the plain loop)."""
    def fwd_bwd():
        for p_ in params:
            p_.grad.zero_()
        loss_fn().backward()

    for p_ in params:
        p_.grad = torch.zeros_like(p_)
    W = min(warm, iters)
    st_ = torch.cuda.Stream()
    st_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st_):
        for it in range(W):
            if pre:
                pre(it)
            fwd_bwd()
            opt.step()
    torch.cuda.current_stream().wait_stream(st_)
    if iters > W:
        g_ = torch.cuda.CUDAGraph()
        # captured on the warm-up stream: a warm-up autograd graph a caller keeps alive (scene_retarget's self_pos) keeps the
        # params' AccumulateGrad nodes on that stream, and a capture on another stream would run the gradient accumulation on
        # a second stream inside the graph (torch's AccumulateGrad stream-mismatch warning)
        with torch.cuda.graph(g_, stream=st_):
            fwd_bwd()
        for it in range(W, iters):
            if pre:
                pre(it)
            g_.replay()
            opt.step()
    for p_ in params:
        p_.grad = None
