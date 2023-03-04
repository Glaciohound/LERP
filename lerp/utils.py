import os
import json
import random
import torch
import torch.nn.functional as F
import numpy as np
import time
import pickle


def prob_to_logit(prob):
    eps = 1e-4
    if isinstance(prob, torch.Tensor):
        prob = prob.float().clamp(eps, 1-eps)
        return (prob / (1 - prob)).log()
    else:
        if not isinstance(prob, np.ndarray):
            prob = np.array(prob)
        prob = prob.clip(eps, 1-eps)
        return np.log(prob / (1 - prob))


def logit_to_prob(logit):
    if isinstance(logit, torch.Tensor):
        return logit.sigmoid()
    else:
        return 1 / (1 + np.exp(-logit))


def expand_and_permute(tensor, from_vars, to_vars, n_nodes=None):
    if tensor is None:
        return None
    assert all(ind in to_vars for ind in from_vars)
    if from_vars == to_vars:
        return tensor
    elif tensor.is_sparse:
        if len(to_vars) == 2:
            if len(from_vars) == 2:
                tensor = tensor.transpose(-1, -2)
            elif from_vars[0] == to_vars[0]:
                tensor = torch.stack([tensor]*tensor.shape[-1], -1)
            else:
                tensor = torch.stack([tensor]*tensor.shape[-1], -2)
        else:
            assert len(to_vars) == 1 and n_nodes is not None
            tensor = torch.stack([tensor]*n_nodes, -1)
        return tensor
    to_shape = tensor.shape + (1,) * (len(to_vars) - len(from_vars))
    expanded = tensor.view(*to_shape)
    permutation = [0, 1]
    tail_vars = len(from_vars)
    for name in to_vars:
        if name in from_vars:
            permutation.append(from_vars.index(name)+2)
        else:
            permutation.append(tail_vars+2)
            tail_vars += 1
    permuted = expanded.permute(*permutation)
    return permuted


def align_results(results, to_variables, stack_axis=0):
    aligned_scores = [
        expand_and_permute(scores, variables, to_variables)
        for scores, _, variables in results
    ]
    aligned_mask = [
        mask if mask is not None else
        expand_and_permute(mask, variables, to_variables)
        for _, mask, variables in results
    ]
    max_shape = torch.Size(torch.tensor(
        [list(_score.shape) for _score in aligned_scores]
    ).max(0)[0])
    scores = torch.stack([
        _score.expand(max_shape) if max_shape != _score.shape else _score
        for _score in aligned_scores
    ], axis=stack_axis)
    if aligned_mask[0] is not None:
        valid_mask = torch.stack([
            _mask.expand(max_shape) if max_shape != _mask.shape else _mask
            for _mask in aligned_mask
        ], axis=stack_axis)
    else:
        valid_mask = None
    return scores, valid_mask


def vocab_to_list(vocab):
    assert len(vocab) == max(vocab.values()) + 1
    return [
        name for name, i in
        sorted(vocab.items(), key=lambda x: x[1])
    ]


def pad_sequence(batch, padding_value):
    max_shape = torch.stack(
        [torch.tensor(_tensor.shape) for _tensor in batch]
    ).max(0)[0]
    raw_padded = []
    for tensor in batch:
        pad_shape = (max_shape - torch.tensor(tensor.shape)).flip(0)
        pad_arg = torch.stack(
            [torch.zeros_like(pad_shape), pad_shape],
            1
        ).view(-1)
        raw_padded.append(F.pad(
            tensor, pad_arg.numpy().tolist(), "constant", padding_value))
    output = torch.stack(raw_padded)
    return output


def batch_tensor_with_padding(tensor_batch, padding_value):
    padded = pad_sequence(tensor_batch, padding_value)
    valid_mask = pad_sequence(
        [torch.ones_like(_tensor, dtype=bool) for _tensor in
         tensor_batch],
        padding_value=False
    )
    return padded, valid_mask


def mask_with_value(tensor, mask, value):
    output = torch.ones_like(tensor) * value
    output[mask] = tensor[mask]
    return output


def categorical_softmax_entropy(tensor, dim):
    log_softmax = F.log_softmax(tensor, dim)
    P = F.softmax(tensor, dim)
    entropy = -(P * log_softmax).sum(dim)
    return entropy


def mask_out_diagonal(mask):
    n_nodes = mask.shape[-1]
    n_vars = mask.ndim - 1
    diag = torch.diag(
        torch.ones(n_nodes, dtype=bool, device=mask.device)
    ).logical_not()
    for i in range(2, n_vars+1):
        for j in range(1, i):
            this_diag = diag.view(
                *((1,)*j), n_nodes, *((1,)*(i-j-1)), n_nodes,
                *((1,)*(n_vars-i))

            )
            mask = mask.logical_and(this_diag)
    return mask


def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def unwrap_json(filename):
    if filename.endswith(".json"):
        with open(filename, 'r') as f:
            data = json.load(f)
    elif filename.endswith(".jsonl"):
        with open(filename, 'r') as f:
            lines = list(f)
            data = [json.loads(line) for line in lines]
        filename = filename[:-1]
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def timeit(fn, args):
    a = time.time()
    result = fn(*args)
    print("time: ", time.time() - a)
    return result


cache_dir = ".cache"


def cache_name(name):
    return os.path.join(cache_dir, name+".pkl")


def is_cached(name):
    return os.path.exists(cache_name(name))


def get_cached(name):
    with open(cache_name(name), 'rb') as f:
        return pickle.load(f)


def cache(content, name):
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_name(name), 'wb') as f:
        pickle.dump(content, f)


class RunningMean:
    def __init__(self, gamma):
        self.gamma = gamma
        self.count = 0
        self._value = None

    def update(self, value):
        value = value.detach().cpu()
        if value.ndim == 0:
            self._update(value)
        else:
            for _v in value:
                self._update(_v)

    def _update(self, value):
        self.count += 1
        if self._value is None:
            self._value = value
        else:
            w1 = self.gamma * (1 - self.gamma ** (self.count - 1))
            w2 = (1 - self.gamma)
            wt = w1 + w2
            w1 = w1 / wt
            w2 = w2 / wt
            self._value = w1 * self._value + w2 * value

    @property
    def value(self):
        if self._value is None:
            return 0
        return self._value * 1


class PlainMean:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value):
        value = value.detach().cpu()
        if value.ndim == 0:
            self._update(value)
        else:
            for _v in value:
                self._update(_v)

    def _update(self, value):
        self.sum = self.sum + value
        self.count += 1

    @property
    def value(self):
        if self.count == 0:
            return 0
        else:
            return self.sum / self.count


def parser_set_new_default(parser, name, new_value):
    for _action in parser._actions:
        if _action.dest == name:
            _action.default = new_value
