import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import functools
import pickle
import logging
from collections import OrderedDict
from torch.autograd import Function

__all__ = [
    'is_dist_initialized',
    'get_world_size',
    'get_rank',
    'get_global_rank',
    'new_group',
    'destroy_process_group',
    'barrier',
    'broadcast',
    'broadcast_object_list',
    'all_reduce',
    'easy_reduce',
    'reduce',
    'gather',
    'all_gather',
    'get_global_gloo_group',
    'generalized_all_gather',
    'generalized_gather',
    'send',
    'recv',
    'isend',
    'irecv',
    'all_to_all',
    'shared_random_seed',
    'diff_all_to_all',
    'split_forward_gather_backward',
    'gather_forward_split_backward',
    'distributed_kmeans',
    'sinkhorn',
    'frechet_inception_distance',
    'init_model_parallel_groups',
    'is_model_parallel_initialized',
    'get_data_parallel_group',
    'get_tensor_parallel_group',
    'get_pipeline_parallel_group',
    'get_data_parallel_rank',
    'get_data_parallel_world_size',
    'get_tensor_parallel_rank',
    'get_tensor_parallel_world_size',
    'get_tensor_parallel_src_rank',
    'get_pipeline_parallel_ranks',
    'destroy_model_parallel_groups'
]


#------------------------ basic operations ------------------------#

def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size(group=None):
    return dist.get_world_size(group) if is_dist_initialized() else 1


def get_rank(group=None):
    return dist.get_rank(group) if is_dist_initialized() else 0


def get_global_rank(group, group_rank):
    return dist.get_global_rank(group, group_rank) if is_dist_initialized() else 0


def new_group(ranks=None, **kwargs):
    if is_dist_initialized():
        return dist.new_group(ranks, **kwargs)
    return None


def destroy_process_group():
    if is_dist_initialized():
        dist.destroy_process_group()


def barrier(group=None, **kwargs):
    if get_world_size(group) > 1:
        dist.barrier(group, **kwargs)


def broadcast(tensor, src, group=None, **kwargs):
    if get_world_size(group) > 1:
        return dist.broadcast(tensor, src, group, **kwargs)


def broadcast_object_list(object_list, src, group=None, device=None):
    if get_world_size(group) > 1:
        return dist.broadcast_object_list(object_list, src, group, device)


def all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, **kwargs):
    if get_world_size(group) > 1:
        return dist.all_reduce(tensor, op, group, **kwargs)


def easy_reduce(tensor, op=dist.ReduceOp.SUM, group=None, **kwargs):
    tensor = tensor.clone()
    if get_world_size(group) > 1:
        dist.all_reduce(tensor, op, group, **kwargs)
    return tensor


def reduce(tensor, dst, op=dist.ReduceOp.SUM, group=None, **kwargs):
    if get_world_size(group) > 1:
        return dist.reduce(tensor, dst, op, group, **kwargs)


def gather(tensor, dst=0, group=None, **kwargs):
    rank = get_rank()  # global rank
    world_size = get_world_size(group)
    if world_size == 1:
        return [tensor]
    tensor_list = [
        torch.empty_like(tensor)
        for _ in range(world_size)] if rank == dst else None
    dist.gather(tensor, tensor_list, dst, group, **kwargs)
    return tensor_list


def all_gather(tensor, uniform_size=True, group=None, **kwargs):
    world_size = get_world_size(group)
    if world_size == 1:
        return [tensor]
    assert tensor.is_contiguous(), \
        'ops.all_gather requires the tensor to be contiguous()'
    
    if uniform_size:
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor, group, **kwargs)
        return tensor_list
    else:
        # collect tensor shapes across GPUs
        shape = tuple(tensor.shape)
        shape_list = generalized_all_gather(shape, group)

        # flatten the tensor
        tensor = tensor.reshape(-1)
        size = int(np.prod(shape))
        size_list = [int(np.prod(u)) for u in shape_list]
        max_size = max(size_list)

        # pad to maximum size
        if size != max_size:
            padding = tensor.new_zeros(max_size - size)
            tensor = torch.cat([tensor, padding], dim=0)
        
        # all_gather
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor, group, **kwargs)

        # reshape tensors
        tensor_list = [t[:n].view(s) for t, n, s in zip(
            tensor_list, size_list, shape_list)]
        return tensor_list

@functools.lru_cache()
def get_global_gloo_group():
    backend = dist.get_backend()
    assert backend in ['gloo', 'nccl']
    if backend == 'nccl':
        return dist.new_group(backend='gloo')
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ['gloo', 'nccl']
    device = torch.device('cpu' if backend == 'gloo' else 'cuda')

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            'Rank {} trying to all-gather {:.2f} GB of data on device'
            '{}'.format(get_rank(), len(buffer) / (1024 ** 3), device)
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    world_size = dist.get_world_size(group=group)
    assert world_size >= 1, \
        'gather/all_gather must be called from ranks within' \
        'the give group!'
    local_size = torch.tensor(
        [tensor.numel()], dtype=torch.int64, device=tensor.device
    )
    size_list = [torch.zeros(
        [1], dtype=torch.int64, device=tensor.device
    ) for _ in range(world_size)]

    # gather tensors and compute the maximum size
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # pad tensors to the same size
    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size, ),
            dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def generalized_all_gather(data, group=None):
    if get_world_size(group) == 1:
        return [data]
    if group is None:
        group = get_global_gloo_group()
    
    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving tensors from all ranks
    tensor_list = [torch.empty(
        (max_size, ), dtype=torch.uint8, device=tensor.device
    ) for _ in size_list]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
    return data_list


def generalized_gather(data, dst=0, group=None):
    world_size = get_world_size(group)
    if world_size == 1:
        return [data]
    if group is None:
        group = get_global_gloo_group()
    rank = dist.get_rank()  # global rank

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving tensors from all ranks to dst
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [torch.empty(
            (max_size, ), dtype=torch.uint8, device=tensor.device
        ) for _ in size_list]
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        dist.gather(tensor, [], dst=dst, group=group)
        return []


def send(tensor, dst, group=None, **kwargs):
    if get_world_size(group) > 1:
        assert tensor.is_contiguous(), \
            'ops.send requires the tensor to be contiguous()'
        return dist.send(tensor, dst, group, **kwargs)


def recv(tensor, src=None, group=None, **kwargs):
    if get_world_size(group) > 1:
        assert tensor.is_contiguous(), \
            'ops.recv requires the tensor to be contiguous()'
        return dist.recv(tensor, src, group, **kwargs)


def isend(tensor, dst, group=None, **kwargs):
    if get_world_size(group) > 1:
        assert tensor.is_contiguous(), \
            'ops.isend requires the tensor to be contiguous()'
        return dist.isend(tensor, dst, group, **kwargs)


def irecv(tensor, src=None, group=None, **kwargs):
    if get_world_size(group) > 1:
        assert tensor.is_contiguous(), \
            'ops.irecv requires the tensor to be contiguous()'
        return dist.irecv(tensor, src, group, **kwargs)


def all_to_all(x, scatter_dim, gather_dim, group=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.
    """
    world_size = get_world_size(group)
    if world_size > 1:
        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=group, **kwargs)
        x = torch.cat(outputs, dim=gather_dim).contiguous()
    return x


def shared_random_seed(group=None):
    seed = np.random.randint(2 ** 31)
    all_seeds = generalized_all_gather(seed, group)
    return all_seeds[0]


#-------------------- differentiable operations -------------------#

def _split(input, dim, group):
    # skip if world_size == 1
    rank = get_rank(group=group)
    world_size = get_world_size(group=group)
    if world_size == 1:
        return input
    
    # split sequence
    assert input.size(dim) % world_size == 0
    return input.chunk(world_size, dim=dim)[rank].contiguous()


def _gather(input, dim, group):
    # skip if world_size == 1
    world_size = get_world_size(group=group)
    if world_size == 1:
        return input
    
    # gather sequence
    output = all_gather(input, uniform_size=True, group=group)
    return torch.cat(output, dim=dim).contiguous()


class AllToAll(Function):

    @staticmethod
    def forward(ctx, input, scatter_dim, gather_dim, group):
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.group = group
        return all_to_all(input, scatter_dim, gather_dim, group)
    
    @staticmethod
    def backward(ctx, grad_output):
        return (
            all_to_all(grad_output, ctx.gather_dim, ctx.scatter_dim, ctx.group),
            None, None, None
        )


class SplitForwardGatherBackward(Function):

    @staticmethod
    def forward(ctx, input, dim, group=None, grad_scale=None):
        ctx.dim = dim
        ctx.group = group
        ctx.grad_scale = grad_scale
        return _split(input, dim, group)
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == 'up':
            grad_output = grad_output * get_world_size(group=ctx.group)
        elif ctx.grad_scale == 'down':
            grad_output = grad_output / get_world_size(group=ctx.group)
        return _gather(grad_output, ctx.dim, ctx.group), None, None, None


class GatherForwardSplitBackward(Function):

    @staticmethod
    def forward(ctx, input, dim, group=None, grad_scale=None):
        ctx.dim = dim
        ctx.group = group
        ctx.grad_scale = grad_scale
        return _gather(input, dim, group)
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * get_world_size(group=ctx.group)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / get_world_size(group=ctx.group)
        return _split(grad_output, ctx.dim, ctx.group), None, None, None


def diff_all_to_all(input, scatter_dim, gather_dim, group=None):
    return AllToAll.apply(input, scatter_dim, gather_dim, group)


def split_forward_gather_backward(input, dim, group=None, grad_scale=None):
    return SplitForwardGatherBackward.apply(input, dim, group, grad_scale)


def gather_forward_split_backward(input, dim, group=None, grad_scale=None):
    return GatherForwardSplitBackward.apply(input, dim, group, grad_scale)


#--------------------------- algorithms ---------------------------#

@torch.no_grad()
def distributed_kmeans(
    feats,
    num_clusters,
    num_iters=10,
    metric='inner_product',
    memory_budget=2 ** 34,
    seed=8888,
    echo=False
):
    assert metric in ('inner_product', 'l2')

    # params
    k, n, c, device = num_clusters, *feats.size(), feats.device
    rank = get_rank()
    world_size = get_world_size()
    ones = feats.new_ones(n, dtype=torch.long)

    # similarity function
    def sim_fn(x, y):
        if metric == 'inner_product':
            return torch.mm(x, y.t())
        else:
            return -torch.cdist(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0)

    # init clusters
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    rand_inds = torch.randperm(
        n, device=device, generator=g
    )[:int(np.ceil(k / world_size))]
    clusters = torch.cat(all_gather(feats[rand_inds]), dim=0)[:k]

    # variables
    new_clusters = feats.new_zeros(k, c)
    counts = feats.new_zeros(k, dtype=torch.long)

    # iterative Expectation-Maximization
    for step in range(num_iters + 1):
        # Expectation step
        if n * k * 4 <= memory_budget:
            simmat = sim_fn(feats, clusters)
            scores, assigns = simmat.max(dim=1)
        else:
            # NOTE: for large n and k, the simmat could be super large!
            scores, assigns = [], []
            for chunk in feats.split(int(memory_budget / (k * 4))):
                simmat = sim_fn(chunk, clusters)
                _scores, _assigns = simmat.max(dim=1)
                scores.append(_scores)
                assigns.append(_assigns)
            scores = torch.cat(scores)
            assigns = torch.cat(assigns)
        
        # logging
        if echo:
            print(
                f'rank [{rank}/{world_size}] kmeans step: {step}/{num_iters} '
                f'score: {scores.mean().item():.4f}',
                flush=True
            )

        # skip the maximization step for the last iteration
        if step == num_iters:
            break

        # Maximization step
        new_clusters.zero_().scatter_add_(0, assigns.unsqueeze(1).expand(-1, c), feats)
        all_reduce(new_clusters)

        counts.zero_()
        counts.index_add_(0, assigns, ones)
        all_reduce(counts)

        mask = (counts > 0)
        clusters[mask] = new_clusters[mask] / counts[mask].view(-1, 1)
        if metric == 'inner_product':
            clusters = F.normalize(clusters, p=2, dim=1)
    return clusters, assigns, scores


@torch.no_grad()
def sinkhorn(Q, eps=0.5, num_iters=3):
    # normalize Q
    Q = torch.exp(Q / eps).t()
    sum_Q = Q.sum()
    all_reduce(sum_Q)
    Q /= sum_Q

    # variables
    n, m = Q.size()
    u = Q.new_zeros(n)
    r = Q.new_ones(n) / n
    c = Q.new_ones(m) / (m * get_world_size())

    # iterative update
    cur_sum = Q.sum(dim=1)
    all_reduce(cur_sum)
    for i in range(num_iters):
        u = cur_sum
        Q *= (r / u).unsqueeze(1)
        Q *= (c / Q.sum(dim=0)).unsqueeze(0)
        cur_sum = Q.sum(dim=1)
        all_reduce(cur_sum)
    return (Q / Q.sum(dim=0, keepdim=True)).t().float()


def _approximation_error(matrix, s_matrix):
    norm_of_matrix = torch.norm(matrix)
    error = matrix - torch.mm(s_matrix, s_matrix)
    error = torch.norm(error) / norm_of_matrix
    return error


def _sqrtm_newton_schulz(matrix, num_iters=100):
    """
    Square root of matrix using Newton-Schulz Iterative method
    Source: https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
    Args:
        matrix: matrix or batch of matrices
        num_iters: Number of iteration of the method
    Returns:
        Square root of matrix
        Error
    """
    dim = matrix.size(0)
    norm_of_matrix = matrix.norm(p='fro')
    Y = matrix.div(norm_of_matrix)
    I = torch.eye(dim, dim, dtype=matrix.dtype, device=matrix.device)  # noqa: E741
    Z = torch.eye(dim, dim, dtype=matrix.dtype, device=matrix.device)

    s_matrix = torch.empty_like(matrix)
    error = torch.empty(1, dtype=matrix.dtype, device=matrix.device)

    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)

        s_matrix = Y * torch.sqrt(norm_of_matrix)
        error = _approximation_error(matrix, s_matrix)
        if torch.isclose(error, torch.tensor(
            [0.], dtype=error.dtype, device=error.device), atol=1e-5
        ):
            break

    return s_matrix, error


def frechet_inception_distance(x, y, eps=1e-6, group=None):
    """
    x: [Nx, C].
    y: [Ny, C].
    """
    assert x.size(1) == y.size(1)
    nx, ny, nw, c = x.size(0), y.size(0), get_world_size(group=group), x.size(1)

    # preprocess
    x = x.detach().to(torch.float64)
    y = y.detach().to(torch.float64)

    # statistics of x
    mu_x = easy_reduce(x.sum(dim=0), group=group) / (nx * nw)
    x = x - mu_x.unsqueeze(0)
    sigma_x = easy_reduce(x.T @ x, group=group) / (nx * nw - 1)

    # statistics of y
    mu_y = easy_reduce(y.sum(dim=0), group=group) / (ny * nw)
    y = y - mu_y.unsqueeze(0)
    sigma_y = easy_reduce(y.T @ y, group=group) / (ny * nw - 1)

    # square root of cov production
    s, _ = _sqrtm_newton_schulz(sigma_x @ sigma_y)
    if not torch.isfinite(s).all():
        print(
            'FID calculation produces singular product; '
            f'adding {eps} to diagonal of cov estimates',
            flush=True
        )
        offset = eps * torch.eye(c, dtype=x.dtype, device=x.device)
        s, _ = _sqrtm_newton_schulz((sigma_x + offset) @ (sigma_y + offset))
    
    # compute fid
    m = mu_x - mu_y
    fid = (
        torch.dot(m, m) + torch.trace(sigma_x) +
        torch.trace(sigma_y) - 2 * torch.trace(s)
    )
    return fid


#------------------------- model parallelism -------------------------#

# parallel states
DATA_PARALLEL_GROUP = None
TENSOR_PARALLEL_GROUP = None
PIPELINE_PARALLEL_GROUP = None
PIPELINE_PARALLEL_RANKS = None


def init_model_parallel_groups(tensor_parallel_size=1, pipeline_parallel_size=1):
    """
    Initialize model and data parallel groups.

    Arguments:
        tensor_parallel_size: #GPUs used to parallelize model tensor.
        pipeline_parallel_size: #GPUs used to parallelize model pipeline.
    
    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we use 2 GPUs to
    parallelize the model tensor, and 4 GPUs to parallelize the model pipeline. The
    present function will create 8 tensor model-parallel groups, 4 pipeline
    model-parallel groups and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7],
            [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7],
            [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks are on the same
    DGX box. For example if we are using 2 DGX-1 boxes with a total of 16 GPUs, rank 0
    to 7 belong to the first box and ranks 8 to 15 belong to the second box.
    """
    # parallel group sizes
    assert dist.is_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size % tensor_parallel_size == 0
    assert world_size % (tensor_parallel_size * pipeline_parallel_size) == 0
    data_parallel_size = world_size // (tensor_parallel_size * pipeline_parallel_size)
    if rank == 0:
        print(f'>> initialize data parallel with size {data_parallel_size}')
        print(f'>> initialize tensor parallel with size {tensor_parallel_size}')
        print(f'>> initialize pipeline parallel with size {pipeline_parallel_size}')
    
    # mesh to facilitate rank indexing
    mesh = torch.arange(world_size).view(
        data_parallel_size, pipeline_parallel_size, tensor_parallel_size
    )
    index = torch.where(mesh == rank)
    assert all(u.numel() == 1 for u in index)
    index = [u.item() for u in index]

    # data parallel groups
    global DATA_PARALLEL_GROUP
    assert DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'
    for j in range(pipeline_parallel_size):
        for k in range(tensor_parallel_size):
            group = dist.new_group(mesh[:, j, k].tolist())
            if j == index[1] and k == index[2]:
                DATA_PARALLEL_GROUP = group
    
    # tensor parallel groups
    global TENSOR_PARALLEL_GROUP
    assert TENSOR_PARALLEL_GROUP is None, 'tensor parallel group is already initialized'
    for i in range(data_parallel_size):
        for j in range(pipeline_parallel_size):
            group = dist.new_group(mesh[i, j, :].tolist())
            if i == index[0] and j == index[1]:
                TENSOR_PARALLEL_GROUP = group
    
    # pipeline parallel group
    global PIPELINE_PARALLEL_GROUP
    global PIPELINE_PARALLEL_RANKS
    assert PIPELINE_PARALLEL_GROUP is None and PIPELINE_PARALLEL_RANKS is None, \
        'pipeline parallel group is already initialized'
    for i in range(data_parallel_size):
        for k in range(tensor_parallel_size):
            ranks = mesh[i, :, k].tolist()
            group = dist.new_group(ranks)
            if i == index[0] and k == index[2]:
                PIPELINE_PARALLEL_GROUP = group
                PIPELINE_PARALLEL_RANKS = ranks


def is_model_parallel_initialized():
    return (
        DATA_PARALLEL_GROUP is not None and
        TENSOR_PARALLEL_GROUP is not None and
        PIPELINE_PARALLEL_GROUP is not None
    )


def get_data_parallel_group():
    assert DATA_PARALLEL_GROUP is not None, 'data parallel group is not initialized'
    return DATA_PARALLEL_GROUP


def get_tensor_parallel_group():
    assert TENSOR_PARALLEL_GROUP is not None, 'tensor parallel group is not initialized'
    return TENSOR_PARALLEL_GROUP


def get_pipeline_parallel_group():
    assert PIPELINE_PARALLEL_GROUP is not None, \
        'pipeline parallel group is not initialized'
    return PIPELINE_PARALLEL_GROUP


def get_data_parallel_rank():
    return dist.get_rank(group=get_data_parallel_group())


def get_data_parallel_world_size():
    return dist.get_world_size(group=get_data_parallel_group())


def get_tensor_parallel_rank():
    return dist.get_rank(group=get_tensor_parallel_group())


def get_tensor_parallel_world_size():
    return dist.get_world_size(group=get_tensor_parallel_group())


def get_tensor_parallel_src_rank():
    rank = dist.get_rank()
    tensor_parallel_size = get_tensor_parallel_world_size()
    return (rank // tensor_parallel_size) * tensor_parallel_size


def get_pipeline_parallel_ranks():
    assert PIPELINE_PARALLEL_RANKS is not None, \
        'pipeline parallel group is not initialized'
    return PIPELINE_PARALLEL_RANKS


def destroy_model_parallel_groups():
    global DATA_PARALLEL_GROUP
    global TENSOR_PARALLEL_GROUP
    global PIPELINE_PARALLEL_GROUP
    global PIPELINE_PARALLEL_RANKS
    DATA_PARALLEL_GROUP = None
    TENSOR_PARALLEL_GROUP = None
    PIPELINE_PARALLEL_GROUP = None
    PIPELINE_PARALLEL_RANKS = None
