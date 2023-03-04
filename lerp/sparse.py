import torch
from multiprocessing import pool


class BatchMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor1, tensor2):
        ctx.save_for_backward(tensor1, tensor2)
        return torch.bmm(tensor1, tensor2)

    @staticmethod
    def backward(ctx, grad_output):
        tensor1, tensor2 = ctx.saved_tensors
        # grad1 = torch.bmm(grad_output, tensor2.transpose(1, 2))
        grad1 = None
        grad2 = torch.bmm(tensor1.transpose(1, 2), grad_output)
        return grad1, grad2


sparse_bmm = BatchMatMul.apply


def sparse_lambda(a, fn):
    if a.is_sparse:
        a = a.coalesce()
        return torch.sparse_coo_tensor(
            a.indices(), fn(a.values()), a.shape
        )
    else:
        return fn(a)


def sparse_scalarmul(a, c):
    if a.is_sparse:
        a = a.coalesce()
        return torch.sparse_coo_tensor(
            a.indices(), a.values() * c, a.shape
        )
    else:
        return a * c


def sparse_vectormul(a, b, dim):
    if a.is_sparse:
        threadpool = pool.ThreadPool(100)
        aT = a.transpose(0, dim)
        results = torch.stack(threadpool.starmap(
            sparse_scalarmul, zip(aT, b)
        )).transpose(0, dim)
        return results
    else:
        return a * b.view(*(1,)*(dim), -1, *(1,)*(a.ndim-dim-1))


def sparse_select(tensor, dim, index):
    tensor = tensor.coalesce()
    indices = tensor.indices()
    indexed = indices[dim] == index
    new_indices = indices[:, indexed]
    dim_ind = list(range(tensor.ndim))
    dim_ind.remove(dim)
    return torch.sparse_coo_tensor(
        new_indices[dim_ind],
        tensor.values()[indexed],
        tensor.shape[:dim]+tensor.shape[dim+1:]
    )


def spmatmul(a, b, dim=-1):
    if a.is_sparse:
        mm = torch.sparse.mm
        if a.ndim == 2 and b.ndim == 2:
            return mm(a, b)
        elif a.ndim == b.ndim and a.shape[:-2] == b.shape[:-2]:
            return torch.stack([
                spmatmul(_a, _b)
                for _a, _b in zip(a, b)
            ])
        elif b.ndim == 2:
            aT = a.transpose(0, dim)
            output = []
            # print("A", end="", flush=True)
            threadpool = pool.ThreadPool(100)
            for j in range(b.shape[1]):
                mapped = threadpool.starmap(
                    sparse_scalarmul, zip(aT, b.select(1, j)))
                _sum = mapped[0]
                for i in range(1, b.shape[0]):
                    _sum = _sum + mapped[i]
                output.append(_sum)
            output = torch.stack(output, dim)
            # print("B", end="", flush=True)
            return output
        else:
            raise NotImplementedError()
    else:
        if dim == -1:
            return torch.matmul(a, b)
        else:
            return torch.matmul(
                a.transpose(dim, -1), b).transpose(dim, -1)


def squeeze(tensor, dim):
    if tensor.is_sparse:
        tensor = tensor.coalesce()
        return torch.sparse_coo_tensor(
            torch.cat([
                tensor.indices()[:dim], tensor.indices()[dim+1:]
            ]),
            tensor.values(),
            tensor.shape[:dim] + tensor.shape[dim+1:]
        )
    else:
        return tensor.squeeze(dim)


def sparse_extend(tensor, to_shape):
    # no grad here
    if tensor.shape == to_shape:
        return tensor
    return torch.sparse_coo_tensor(
        tensor._indices(),
        tensor._values(),
        to_shape
    )


def sparse_stack(tensor, dim, n):
    return torch.stack([tensor]*n, dim)


def sparse_single_add(tensor, index, value):
    tensor.add_(
        torch.sparse_coo_tensor(
            [[_i] for _i in index], [value],
            tensor.shape
        )
    )


def sparse_dropout_last(tensor, num):
    tensor = tensor.coalesce()
    if tensor._nnz() <= num or num <= 0:
        return tensor
    else:
        argsort = tensor.values().argsort()
        top = argsort < num
        return torch.sparse_coo_tensor(
            tensor.indices()[:, top],
            tensor.values()[top],
            tensor.shape
        )


def sparse_max(tensor):
    tensor = tensor.coalesce()
    if tensor._nnz() != 0:
        return tensor.values().max()
    else:
        return torch.sparse.sum(tensor)


def sparse_bias(tensor, bias):
    exp = (-bias).exp()
    if tensor.is_sparse:
        tensor = tensor.coalesce()
        tensor = torch.sparse_coo_tensor(
            tensor.indices(),
            ((1 - tensor.values()) / tensor.values() * exp +
             1).reciprocal(),
            tensor.shape
        )
    else:
        tensor = (1+(1-tensor)/tensor*exp).reciprocal()


def to_dense(tensor):
    if tensor.is_sparse:
        return tensor.to_dense()
    else:
        return tensor


def to_sparse(tensor):
    if not tensor.is_sparse:
        return tensor.to_sparse()
    else:
        return tensor


def normalize_max(tensor):
    if tensor.is_sparse:
        max_score = sparse_max(tensor)
    else:
        max_score = tensor.max()
    tensor = sparse_scalarmul(tensor, 1 / max(1, max_score))
    return tensor
