import torch
import torch.nn as nn

from lerp.sparse import sparse_lambda
from lerp.utils import categorical_softmax_entropy


class And_or_Not(nn.Module):
    def __init__(self, rank, width):
        super().__init__()
        self.width = width
        self.rank = rank
        self.select_weights = nn.Parameter(torch.zeros(rank, width))
        self.gate_weights = nn.Parameter(torch.zeros(rank))

    def forward(self, chain, lerp):
        lerp_selected = torch.mm(
            self.select_weights.softmax(1), lerp
        )
        and_lerp = chain * lerp_selected[None]
        gated = and_lerp + (chain-and_lerp) * \
            self.gate_weights.sigmoid()[None, :, None]
        # gated = (lerp_selected.log() *
        #          self.gate_weights.sigmoid()[:, None]).exp()
        # output = chain * gated[None]
        return gated

    def entropy(self):
        gate_weights = self.gate_weights.sigmoid()
        gate_entropy = -(gate_weights * gate_weights.log()).sum()
        entropy = (
            categorical_softmax_entropy(self.select_weights, 1).sum() +
            gate_entropy
        )
        return entropy


class LerpChaining(nn.Module):
    def __init__(self, width, n_relations, with_equity_relation, dense):
        super().__init__()
        self.width = width
        self.n_relations = n_relations
        self.with_equity_relation = with_equity_relation
        self.dense = dense
        self.weights = nn.Parameter(torch.zeros(width, 2*n_relations))
        self.equity_weight = nn.Parameter(torch.zeros(width, 2))

    def forward(self, inputs, database):
        # inputs shape == (query, width, n_node)
        weights = self.weights.softmax(1).transpose(0, 1)
        if self.with_equity_relation:
            equity_weight = self.equity_weight.softmax(1)
        batch_size, width, n_node = inputs.shape
        n_rel = self.n_relations
        output = 0
        if self.dense:
            # database shape == (n_rel, n_node, n_node)
            averaged_rel = (
                database.transpose(0, 2).matmul(weights[:n_rel]) +
                database.permute(1, 2, 0).matmul(weights[n_rel:])
            ).transpose(0, 2)
            # output.shape == inputs.shape
            output = torch.bmm(
                inputs.transpose(0, 1), averaged_rel
            ).transpose(0, 1)
        else:
            database = database + [A_rel.transpose(0, 1) for A_rel in database]
            for A_rel, _weight in zip(database, weights):
                chained = torch.sparse.mm(
                    A_rel.transpose(0, 1),
                    inputs.transpose(0, 2).reshape(n_node, -1)
                ).reshape(n_node, width, batch_size).transpose(0, 2)
                output = output + chained * _weight[:, None]

        output = sparse_lambda(output, lambda x: 1-(-x).exp())
        if self.with_equity_relation:
            output = output * equity_weight[:, 0, None] + \
                inputs * equity_weight[:, 1, None]
        return output

    def entropy(self):
        entropy = (
            categorical_softmax_entropy(self.weights, 1).sum() +
            categorical_softmax_entropy(self.equity_weight, 1).sum()
        )
        return entropy


class LogicLerpInputLayer(nn.Module):
    def __init__(self, n_relations, width, dense, n_attributes=0):
        super().__init__()
        self.n_relations = n_relations
        self.n_attributes = n_attributes
        self.width = width
        self.dense = dense
        self.weights = nn.Parameter(torch.zeros(
            2*n_relations+n_attributes, width))

    def forward(self, database, attributes=None):
        if self.dense:
            quantified = torch.cat([
                database.sum(1), database.sum(2)
            ], 0)
            if attributes is not None:
                assert self.n_attributes > 0
                quantified = torch.cat([quantified, attributes * 2], 0)
            quantified = 1 - (-quantified).exp()
            outputs = torch.mm(self.weights.softmax(0).transpose(0, 1),
                               quantified)
        else:
            # shape == (2 * n_rel, n_node)
            assert attributes is None and self.n_attributes == 0
            quantified = torch.stack([
                torch.sparse.sum(A_rel, dim)
                for A_rel in database for dim in (0, 1)
            ], 0)
            quantified = sparse_lambda(quantified, lambda x: 1 - (-x).exp())
            # shape == (width, n_node)
            outputs = torch.sparse.mm(quantified.transpose(0, 1),
                                      self.weights.softmax(0)).transpose(0, 1)
        return outputs, quantified

    def entropy(self):
        entropy = categorical_softmax_entropy(self.weights, 0).sum()
        return entropy


class LogicMetaLerpLayer(nn.Module):
    def __init__(self, width, n_relations, ops, with_bias, temperature, dense):
        super().__init__()
        self.width = width
        self.ops = ops
        self.with_bias = with_bias
        self.temperature = temperature
        self.arg1_weights = nn.Parameter(torch.zeros(width, width))
        self.arg2_weights = nn.Parameter(torch.zeros(width, width))
        self.op_weights = nn.Parameter(torch.zeros(width, len(ops)))
        self.bias = nn.Parameter(torch.zeros(width))
        self.dense = dense
        self.chain = LerpChaining(width, n_relations, False, dense)

    def forward(self, inputs, database):
        # shape == (width, n_node)
        arg1 = torch.mm(self.arg1_weights.softmax(0).transpose(0, 1),
                        inputs)
        arg2 = torch.mm(self.arg2_weights.softmax(0).transpose(0, 1),
                        inputs)
        op_weights = self.op_weights.softmax(1).transpose(0, 1)
        op_results = 0
        for _op, _op_weight in zip(self.ops, op_weights):
            if _op == "Copy":
                this_op_result = arg2
            elif _op == "And":
                this_op_result = arg1 * arg2
            elif _op == "Or":
                this_op_result = arg1 + arg2 - arg1 * arg2
            elif _op == "Chaining":
                this_op_result = self.chain(arg2[None], database)[0]
            elif _op == "Not":
                this_op_result = 1 - arg1
            op_results = op_results + this_op_result * _op_weight[:, None]
        return op_results

    def entropy(self):
        entropy = (
            categorical_softmax_entropy(self.arg1_weights, 0).sum() +
            categorical_softmax_entropy(self.arg2_weights, 0).sum() +
            categorical_softmax_entropy(self.op_weights, 1).sum() +
            self.chain.entropy()
        )
        return entropy
