import numpy as np
import torch
from torch.nn import Parameter

from lerp.utils import \
    align_results, categorical_softmax_entropy, expand_and_permute, \
    prob_to_logit
from lerp.sparse import \
    spmatmul, sparse_vectormul, to_dense, to_sparse, normalize_max, \
    sparse_select, sparse_bmm
from lerp.logic import \
    And, Exist, LogicalEquation, Wrapper, Attribute, Relation, Copy, \
    Binary, Unary, Quantified, ReplaceVariable


class LogicNetElement(LogicalEquation):
    def __init__(self, variables, no_grad=False):
        super().__init__(no_grad)
        self.variables = variables

    def forward(self, batch, soft_logic, final_forall_logic=None):
        raise NotImplementedError()

    def str_info(self, replace_vars, all_variables):
        raise NotImplementedError()


class VectorBinaryChainAnd(Wrapper, Binary):
    def __init__(self, formula1: LogicalEquation, formula2: LogicalEquation,
                 chain_head=None, chain_tail=None,
                 no_grad=False):
        assert formula1.n_vars == 2 and formula2.n_vars == 2
        super().__init__(formula1, formula2, no_grad)
        hinge_var = next(self.tmp_names)
        if chain_head is None:
            chain_head = self.variables[1]
            chain_tail = self.variables[0]
        self.inner_formula = Exist(
            (hinge_var,),
            And(
                ReplaceVariable(chain_head, hinge_var, formula1),
                ReplaceVariable(chain_tail, hinge_var, formula2))
        )

    def str_info(self, replace_vars, all_variables):
        return self.inner_formula.str_info(replace_vars, all_variables)

    def inner_scoring(self, batch, soft_logic: str):
        if soft_logic == "naive-prob-matmul":
            pre_results = self.score_leaves(batch, soft_logic)
            scores, valid_mask, _ = self.static_scoring(
                pre_results + (self.variables,), soft_logic
            )
            return scores, valid_mask, self.variables
        return self.inner_formula.scoring(batch, soft_logic)

    @staticmethod
    def spvecmatul(matrices, weights, score):
        output = None
        for _type, _weight in enumerate(weights):
            mat = sparse_select(matrices, 1, _type)
            this_one = sparse_bmm(mat, score.transpose(1, 2)).transpose(1, 2)
            this_one = sparse_vectormul(this_one, _weight, 1)
            if output is None:
                output = this_one
            else:
                output = output + this_one
        return output

    @staticmethod
    def static_scoring(pre_results, soft_logic):
        (prim_matrix, prim_matrix_mask, argmat_weights), score2, \
            valid_mask, variables = pre_results
        scores = VectorBinaryChainAnd.spvecmatul(
            prim_matrix, argmat_weights, score2)
        scores = normalize_max(scores)
        return scores, valid_mask, variables


class LogicVectorInputLayer(LogicNetElement):
    def __init__(self, variables, attributes, relations, to_sparse=False,
                 no_quantified=False, no_grad=False, remove_inputs=[]):
        super().__init__(variables, no_grad)
        var1, var2 = self.variables
        self.attribute_input_list = [
            Attribute(attribute_name, var1)
            for attribute_name in attributes
            if attribute_name not in remove_inputs
        ]
        if not no_quantified:
            self.attribute_input_list.extend(
                [
                    Exist((var2,), Relation(relation_name, var1, var2))
                    for relation_name in relations
                    if relation_name not in remove_inputs+["Equal"]
                ] + [
                    Exist((var2,), Relation(relation_name, var2, var1))
                    for relation_name in relations
                    if relation_name not in remove_inputs+["Equal"]
                ])
        self.relation_input_list = [
            Relation(relation_name, variable1, variable2)
            for relation_name in relations
            for variable1 in self.variables for variable2 in self.variables
            if variable1 != variable2
            if relation_name not in remove_inputs + ["Equal"]
        ]
        self.input_list = self.attribute_input_list*2+self.relation_input_list
        self.n_width = len(self.attribute_input_list)
        self.n_prim_vector = len(self.attribute_input_list)
        self.n_prim_matrix = len(self.relation_input_list)
        self.to_sparse = to_sparse
        self.no_quantified = no_quantified

    def inner_scoring(self, batch, soft_logic):
        attribute_result_list = [
            _element.scoring(batch, soft_logic)
            for _element in self.attribute_input_list
        ]
        attribute_result_list = [
            (tensor.to_dense() if tensor.is_sparse else tensor,
             valid_mask, variables)
            for tensor, valid_mask, variables in attribute_result_list
        ]
        attribute_scores, attribute_valid_mask = align_results(
            attribute_result_list, self.variables[:1], 1)
        if not self.to_sparse:
            attribute_scores = to_dense(attribute_scores)
        else:
            attribute_scores = to_sparse(attribute_scores)
        if len(self.relation_input_list) != 0:
            relation_result_list = [
                _element.scoring(batch, soft_logic)
                for _element in self.relation_input_list
            ]
            relation_scores, relation_valid_mask = align_results(
                relation_result_list, self.variables, 1)
        else:
            relation_scores, relation_valid_mask = None, None
        return attribute_scores, attribute_valid_mask, self.variables[:1], (
            attribute_scores, attribute_valid_mask,
            relation_scores, relation_valid_mask)

    def str_info(self, replace_vars, all_variables):
        info = self.input_list[self.repr_index.pop(0)].str_info(
            replace_vars, all_variables)
        return info

    def entropy(self):
        return 0


class LogicMetaVectorLayer(LogicNetElement):
    repr_mode = "max"

    def __init__(self, last_layer, n_width, n_args, ops,
                 with_bias, direct_scores_only,
                 init_var, temperature, no_grad=False):
        super().__init__(last_layer.variables, no_grad)
        self.init_var = init_var
        self.with_bias = with_bias
        self.temperature = temperature
        self.direct_scores_only = direct_scores_only
        self.last_layer = last_layer
        input_width = self.last_layer.n_width
        self.input_width = input_width
        self.attribute_input_list = last_layer.attribute_input_list
        self.relation_input_list = last_layer.relation_input_list
        self.n_prim_vector = last_layer.n_prim_vector
        self.n_prim_matrix = last_layer.n_prim_matrix
        self.n_width = n_width
        self.n_args = n_args
        # ops: [(op_type, extra_info), ...]
        self.ops = ops
        whole_input_width = input_width + self.n_prim_vector
        self.whole_input_width = whole_input_width
        self.arg1_weights = Parameter(
            torch.randn(whole_input_width, n_width) * init_var
        )
        if n_args == 2:
            self.arg2_weights = Parameter(
                torch.randn(whole_input_width, n_width) * init_var
            )
            self.argmat_weights = Parameter(
                torch.randn(self.n_prim_matrix, n_width) * init_var
            )
        if self.with_bias:
            self.bias = Parameter(torch.randn(n_width) * init_var)
        self.op_weights = Parameter(
            torch.randn(n_width, len(ops)) * init_var
        )

    def inner_scoring(self, batch, soft_logic):
        input_scores, valid_mask, variables, prim = \
            self.last_layer.scoring(batch, soft_logic)
        prim_vector, prim_vector_mask, prim_matrix, prim_matrix_mask = prim
        bs = input_scores.shape[0]
        n_nodes = input_scores.shape[-1]
        nv = 1
        nw = self.n_width
        niw = self.input_width
        assert variables == self.variables[:1]
        op_weights = (self.op_weights * self.temperature).softmax(1)
        arg1_weights = (self.arg1_weights * self.temperature).softmax(0)
        arg1 = spmatmul(input_scores, arg1_weights[:niw], 1)
        if self.n_prim_vector != 0:
            arg1 += spmatmul(prim_vector, arg1_weights[niw:], 1)
        if self.n_args == 2:
            argmat_weights = (self.argmat_weights*self.temperature).softmax(0)
            arg2_weights = (self.arg2_weights * self.temperature).softmax(0)
            arg2 = spmatmul(input_scores, arg2_weights[:niw], 1)
            if self.n_prim_vector != 0:
                arg2 += spmatmul(prim_vector, arg2_weights[niw:], 1)
        if valid_mask is not None:
            valid_mask = (
                valid_mask.min(1) + prim_vector.min(1)
            )[0].unsqueeze(1).repeat(1, nw, *((1,)*nv)).view(*arg1.shape)
        op_args = []
        for _op, _info in self.ops:
            if issubclass(_op, Quantified):
                op_args.append(
                    (arg1, valid_mask, variables, _info["reduce_var"]))
            elif issubclass(_op, Unary):
                op_args.append((arg1, valid_mask, variables))
            elif issubclass(_op, VectorBinaryChainAnd):
                op_args.append(
                    ((prim_matrix, prim_matrix_mask, argmat_weights), arg2,
                     valid_mask, variables))
            else:
                op_args.append((arg1, arg2, valid_mask, variables))

        if arg1.is_sparse:
            layer_scores = torch.sparse_coo_tensor(
                [[]] * (2+nv), [], (bs, nw, *((n_nodes,)*nv)),
                device=arg1.device, dtype=arg1.dtype
            )
            layer_mask = None
            for _i, ((_op, _), _arg) in enumerate(zip(self.ops, op_args)):
                op_score, op_mask, op_variables = _op.static_scoring(
                    _arg, soft_logic)
                op_score = expand_and_permute(
                    op_score, op_variables, self.variables[:1], n_nodes)
                layer_scores = layer_scores + sparse_vectormul(
                    op_score, op_weights[:, _i], 1
                )

        else:
            layer_scores = torch.zeros(
                (bs, nw, *((n_nodes,)*nv)),
                device=arg1.device, dtype=arg1.dtype)
            layer_mask = torch.ones(
                (bs, nw, *((n_nodes,)*nv)), device=arg1.device, dtype=bool)
            for (_op, _), _arg, _wght in zip(
                    self.ops, op_args, op_weights.split(1, 1)):
                op_score, op_mask, op_variables = _op.static_scoring(
                    _arg, soft_logic)
                op_score = expand_and_permute(
                    op_score, op_variables, self.variables[:1])
                layer_scores = layer_scores + \
                    op_score * _wght.view(1, -1, *((1,)*nv))
                if op_mask is not None:
                    op_mask = expand_and_permute(
                        op_mask, op_variables, self.variables[:1])
                    layer_mask = layer_mask.min(op_mask)

        if self.with_bias:
            layer_scores = (prob_to_logit(layer_scores) +
                            self.bias[None, :, None]).sigmoid()
        if self.direct_scores_only:
            return layer_scores, layer_mask, self.variables[:1]
        else:
            return layer_scores, layer_mask, self.variables[:1], prim

    def str_info(self, replace_vars, all_variables):
        index = self.repr_index.pop(0)
        if index >= self.n_width + self.n_prim_vector:
            return self.relation_input_list[
                index - self.n_width - self.n_prim_vector
            ].str_info(replace_vars, all_variables)
        elif index >= self.n_width:
            return self.attribute_input_list[
                index - self.n_width
            ].str_info(replace_vars, all_variables)

        def _select_fn(weights):
            assert weights.ndim == 1
            if self.repr_mode == "max":
                return weights.max(0)
            elif self.repr_mode == "sampling":
                choice_num = weights.shape[0]
                choice_index = np.random.choice(
                    range(choice_num), p=weights.detach().cpu().numpy())
                return weights[choice_index], choice_index
            else:
                raise Exception()

        arg1_confidence, arg1_index = _select_fn((
            self.arg1_weights[:, index] * self.temperature
        ).softmax(0))
        if self.n_args == 2:
            arg2_confidence, arg2_index = _select_fn((
                self.arg2_weights[:, index] * self.temperature
            ).softmax(0))
            if self.n_prim_matrix != 0:
                argmat_confidence, argmat_index = _select_fn((
                    self.argmat_weights[:, index] * self.temperature
                ).softmax(0))
        op_confidence, op_index = _select_fn((
            self.op_weights[index] * self.temperature
        ).softmax(0))

        last_layer = self.last_layer
        _op_type, _info = self.ops[op_index]
        if issubclass(_op_type, Unary):
            last_layer.repr_index = [arg1_index]
            op = _op_type(last_layer)
        elif issubclass(_op_type, VectorBinaryChainAnd):
            last_layer.repr_index = [
                self.whole_input_width + argmat_index, arg2_index]
            op = _op_type(last_layer, last_layer)
        elif issubclass(_op_type, Binary):
            last_layer.repr_index = [arg1_index, arg2_index]
            op = _op_type(last_layer, last_layer)
        else:
            last_layer.repr_index = [arg1_index]
            op = _op_type(_info["reduce_var"], last_layer)
        info = op.str_info(replace_vars, all_variables)
        info["confidence"] *= op_confidence.cpu().detach().numpy()
        if not issubclass(_op_type, VectorBinaryChainAnd):
            info["confidence"] *= arg1_confidence.cpu().detach().numpy()
        if issubclass(_op_type, Binary):
            info["confidence"] *= arg2_confidence.cpu().detach().numpy()
        return info

    def entropy(self):
        entropy = (
            categorical_softmax_entropy(self.arg1_weights, 0) +
            categorical_softmax_entropy(self.op_weights, 1)
        )
        if self.n_args == 2:
            entropy = entropy + categorical_softmax_entropy(
                self.arg2_weights, 0) + categorical_softmax_entropy(
                    self.argmat_weights, 0
                )
        entropy = entropy.sum()
        return entropy


class LogicSelectionVectorLayer(LogicMetaVectorLayer):
    def __init__(self, layer, n_width, with_bias, direct_scores_only,
                 init_var, temperature, no_grad=False):
        super().__init__(
            layer, n_width, 1, [(Copy, {})], with_bias, direct_scores_only,
            init_var, temperature, no_grad)


class LogicIndexLayer(LogicNetElement):
    def __init__(self, layer, index, no_grad=False):
        super().__init__(layer.variables)
        self.last_layer = layer
        self.index = index

    def inner_scoring(self, batch, soft_logic):
        scores, valid_mask, variables = self.last_layer.scoring(
            batch, soft_logic)
        scores = scores.select(1, self.index)
        if valid_mask is not None:
            valid_mask = valid_mask.select(1, self.index)
        return scores, valid_mask, self.variables

    def str_info(self, replace_vars, all_variables):
        self.last_layer.repr_index = [self.index]
        return self.last_layer.str_info(replace_vars, all_variables)

    def entropy(self):
        return 0
