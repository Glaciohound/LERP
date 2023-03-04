import torch
from torch.nn import Module
from copy import deepcopy
from .graph import Graph
from .utils import \
    logit_to_prob, expand_and_permute, \
    batch_tensor_with_padding, mask_out_diagonal
from .sparse import \
    spmatmul, sparse_scalarmul, sparse_select, sparse_max


# ---------- Definition of Nodes ---------- #

class Node:
    def __init__(self):
        pass


class Variable(Node):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __eq__(self, another):
        if isinstance(another, Variable) and self.name == another.name:
            return True
        else:
            return False

    def __repr__(self):
        return f"\"{self.name}\""

    def __hash__(self):
        return hash(self.name)


class Constant(Node):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __eq__(self, another):
        if isinstance(another, Constant) and self.name == another.name:
            return True
        else:
            return False

    def __repr__(self):
        return f"Constant(\"{self.name}\")"

    def __hash__(self):
        return hash(self.name)

# ---------- Definition of Logic ---------- #


class LogicalEquation(Module):
    soft_logic_types = (
        "naive-prob",
        "minmax-prob",
        "naive-prob-matmul",
    )
    tmp_names = (Variable(f"TMP_{i}") for i in range(10000))
    global_cachestamp = None
    sparse_dropout_last = -1

    def __init__(self, no_grad=False):
        super().__init__()
        self.variables = ()
        self.no_grad = no_grad
        self.cachestamp = 0
        self.cached_result = None
        self.caching = False

    @property
    def n_vars(self):
        return len(self.variables)

    def scoring(self, batch, soft_logic):
        if self.caching and self.cachestamp == self.global_cachestamp:
            return self.cached_result
        self.assert_before_scoring(batch, soft_logic)
        if self.no_grad:
            with torch.no_grad():
                result = self.inner_scoring(batch, soft_logic)
        else:
            result = self.inner_scoring(batch, soft_logic)
        if self.caching and self.global_cachestamp is not None:
            self.cached_result = result
            self.cachestamp = self.global_cachestamp
        else:
            self.cached_result = None
        return result

    def inner_scoring(self, batch, soft_logic):
        raise NotImplementedError()

    @staticmethod
    def static_scoring(pre_results, soft_logic):
        raise NotImplementedError()

    def forward(self, batch, soft_logic, final_forall_logic=None,
                distinct_variables=False):
        if isinstance(batch[0], Graph):
            batch_size = len(batch)
        else:
            batch_size = batch[0].shape[0] if batch[0] is not None \
                else batch[2].shape[0]
        score_tensor, valid_mask, _ = self.scoring(batch, soft_logic)
        if final_forall_logic is None:
            final_forall_logic = soft_logic
        if distinct_variables and valid_mask is not None:
            valid_mask = mask_out_diagonal(valid_mask)
        # score_tensor = mask_with_value(score_tensor, valid_mask, 1)
        if valid_mask is not None:
            score_tensor = 1 - (1 - score_tensor).min(valid_mask)
        scores = score_tensor.reshape(batch_size, -1)
        if final_forall_logic.startswith("naive-prob"):
            scores = scores.prod(-1)
        elif final_forall_logic == "minmax-prob":
            scores = scores.min(-1)[0]
        return scores, score_tensor, valid_mask

    def assert_before_scoring(self, batch, soft_logic):
        if isinstance(batch[0], Graph):
            for graph in batch:
                assert graph.on_torch
        assert soft_logic in self.soft_logic_types

    def __repr__(self):
        info = self.str_info({}, self.all_variables)
        return f"confidence={info['confidence']},  " + info["str"]

    def str_info(self, replace_vars, all_variables):
        variables = self.replace_variables(self.variables, replace_vars)
        return {
            "str": f"{self.__class__.__name__}{variables}",
            "replace_vars": replace_vars,
            "all_variables": all_variables,
            "valid_variables": set(variables),
            "confidence": 1,
            "class": self.__class__
        }

    @property
    def all_variables(self):
        return set(self.variables)

    @staticmethod
    def replace_variables(variables, replace_vars):
        while len(set(variables).intersection(replace_vars.keys())) != 0:
            variables = tuple(
                replace_vars.get(var, var) for var in variables
            )
        return variables

    @staticmethod
    def get_new_variable(variables):
        for i in range(100000):
            x = Variable(f"X{i}")
            if x not in variables:
                return x

    @staticmethod
    def forward_multiple(equations, *args, **kwargs):
        scores = torch.stack([
            _equation(*args, **kwargs)[0] for _equation in equations
        ])
        return scores


class Unary(LogicalEquation):
    def __init__(self, formula: LogicalEquation, no_grad=False):
        super().__init__(no_grad)
        self.formula = formula
        self.variables = formula.variables

    def inner_scoring(self, batch, soft_logic):
        scores, valid_mask, variables = self.formula.scoring(
            batch, soft_logic)
        assert variables == self.variables
        output = self.static_scoring(
            (scores, valid_mask, variables), soft_logic)
        return output

    def str_info(self, replace_vars, all_variables):
        # FIXME
        if hasattr(self, "repr_index"):
            self.formula.repr_index = deepcopy(self.repr_index)
        info = self.formula.str_info(replace_vars, all_variables)
        info["str"] = f"{self.__class__.__name__}({info['str']})"
        return info

    @property
    def all_variables(self):
        return self.formula.all_variables


class Binary(LogicalEquation):
    def __init__(self, formula1: LogicalEquation, formula2: LogicalEquation,
                 no_grad=False):
        super().__init__(no_grad)
        self.formula1 = formula1
        self.formula2 = formula2
        variables = list(formula1.variables)
        for _var in formula2.variables:
            if _var not in variables:
                variables.append(_var)
        self.variables = tuple(variables)

    def score_leaves(self, graph: Graph, soft_logic: str):
        scores1, valid_mask1, variables1 = self.formula1.scoring(
            graph, soft_logic)
        scores2, valid_mask2, variables2 = self.formula2.scoring(
            graph, soft_logic)
        assert set(variables1).union(set(variables2)) == set(self.variables)
        scores1 = expand_and_permute(scores1, variables1, self.variables)
        scores2 = expand_and_permute(scores2, variables2, self.variables)
        valid_mask1 = expand_and_permute(
            valid_mask1, variables1, self.variables)
        valid_mask2 = expand_and_permute(
            valid_mask2, variables2, self.variables)
        if valid_mask1 is None or valid_mask2 is None:
            valid_mask = None
        else:
            valid_mask = torch.logical_and(valid_mask1, valid_mask2)
        return scores1, scores2, valid_mask

    def inner_scoring(self, batch, soft_logic: str):
        pre_results = self.score_leaves(batch, soft_logic)
        results = self.static_scoring(
            pre_results + (self.variables,), soft_logic)
        return results

    def str_info(self, replace_vars, all_variables):
        # FIXME
        if hasattr(self, "repr_index"):
            self.formula1.repr_index = deepcopy(self.repr_index)
            self.formula2.repr_index = deepcopy(self.repr_index)
        info1 = self.formula1.str_info(replace_vars, all_variables)
        all_variables = info1["all_variables"]
        info2 = self.formula2.str_info(replace_vars, all_variables)
        str1 = info1["str"]
        str2 = info2["str"]
        if issubclass(info1["class"], Binary):
            str1 = f"({str1})"
        # and not issubclass(info1["class"], self.__class__):
        if issubclass(info2["class"], Binary):
            str2 = f"({str2})"
            # and \ not issubclass(info2["class"], self.__class__):
        info2["str"] = self.binary_str(str1, str2)
        info2["confidence"] *= info1["confidence"]
        info2["valid_variables"] = info2["valid_variables"].union(
            info1["valid_variables"])
        info2["class"] = self.__class__
        return info2

    @property
    def all_variables(self):
        return self.formula1.all_variables.union(
            self.formula2.all_variables
        )


class Triple(LogicalEquation):
    def __init__(self, formula1: LogicalEquation, formula2: LogicalEquation,
                 formula3: LogicalEquation, no_grad=False):
        super().__init__(no_grad)
        self.formula1 = formula1
        self.formula2 = formula2
        self.formula3 = formula3
        variables = list(formula1.variables)
        for _var in formula2.variables + formula3.variables:
            if _var not in variables:
                variables.append(_var)
        self.variables = tuple(variables)

    def score_leaves(self, graph: Graph, soft_logic: str):
        scores1, valid_mask1, variables1 = self.formula1.scoring(
            graph, soft_logic)
        scores2, valid_mask2, variables2 = self.formula2.scoring(
            graph, soft_logic)
        scores3, valid_mask3, variables3 = self.formula3.scoring(
            graph, soft_logic)
        assert set(variables1).union(set(variables2)).union(set(variables3)) \
            == set(self.variables)
        scores1 = expand_and_permute(scores1, variables1, self.variables)
        scores2 = expand_and_permute(scores2, variables2, self.variables)
        scores3 = expand_and_permute(scores3, variables3, self.variables)
        valid_mask1 = expand_and_permute(
            valid_mask1, variables1, self.variables)
        valid_mask2 = expand_and_permute(
            valid_mask2, variables2, self.variables)
        valid_mask3 = expand_and_permute(
            valid_mask3, variables3, self.variables)
        if valid_mask1 is None or valid_mask2 is None or valid_mask3 is None:
            valid_mask = None
        else:
            valid_mask = torch.logical_and(
                torch.logical_and(valid_mask1, valid_mask2),
                valid_mask3
            )
        return scores1, scores2, scores3, valid_mask

    def inner_scoring(self, batch, soft_logic: str):
        pre_results = self.score_leaves(batch, soft_logic)
        results = self.static_scoring(
            pre_results + (self.variables,), soft_logic)
        return results

    def str_info(self, replace_vars, all_variables):
        info1 = self.formula1.str_info(replace_vars, all_variables)
        all_variables = info1["all_variables"]
        info2 = self.formula2.str_info(replace_vars, all_variables)
        info3 = self.formula3.str_info(replace_vars, all_variables)
        str1 = info1["str"]
        str2 = info2["str"]
        str3 = info3["str"]
        if issubclass(info1["class"], Binary) and \
                not issubclass(info1["class"], self.__class__):
            str1 = f"({str1})"
        if issubclass(info2["class"], Binary) and \
                not issubclass(info2["class"], self.__class__):
            str2 = f"({str2})"
        if issubclass(info3["class"], Binary) and \
                not issubclass(info3["class"], self.__class__):
            str3 = f"({str3})"
        info3["str"] = self.triple_str(str1, str2, str3)
        info3["confidence"] *= info1["confidence"] * info2["confidence"]
        info3["valid_variables"] = info3["valid_variables"].union(
            info1["valid_variables"]).union(info2["valid_variables"])
        info3["class"] = self.__class__
        return info3

    @property
    def all_variables(self):
        return self.formula1.all_variables.union(
            self.formula2.all_variables
        ).union(self.formula3.all_variables)


class Quantified(LogicalEquation):
    def __init__(self, variables, formula: LogicalEquation, no_grad=False):
        super().__init__(no_grad)
        self.formula = formula
        assert all(variables.count(var) == 1 for var in variables)
        self.reduce_variables = variables
        self.variables = tuple(
            var for var in formula.variables
            if var not in variables
        )

    def inner_scoring(self, batch, soft_logic: str):
        pre_results = self.formula.scoring(batch, soft_logic)
        results = self.static_scoring(
            pre_results + (self.reduce_variables,), soft_logic
        )
        assert results[2] == self.variables
        return results

    def str_info(self, replace_vars, all_variables):
        replace_vars = deepcopy(replace_vars)
        all_variables = deepcopy(all_variables)
        for var in self.reduce_variables:
            # if var not in replace_vars:
            new_var = self.get_new_variable(all_variables)
            all_variables.add(new_var)
            replace_vars[var] = new_var

        info = self.formula.str_info(replace_vars, all_variables)
        reduce_variables = tuple(filter(
            lambda x: x in info["valid_variables"],
            self.replace_variables(
                self.reduce_variables, replace_vars)
        ))
        info["valid_variables"] = info["valid_variables"].difference(
            set(reduce_variables))

        if len(reduce_variables) > 0:
            if issubclass(info["class"], self.__class__):
                old_vars = info["reduce_variables"]
                reduce_variables = reduce_variables + old_vars
                info["str"] = info["inner_str"]
            reduce_str = ",".join([
                str(_var) for _var in reduce_variables])
            info["reduce_variables"] = reduce_variables
            info["inner_str"] = info["str"]
            info["str"] = f"{self.header}{reduce_str}:[{info['str']}]"
            info["class"] = self.__class__
        return info

    @property
    def all_variables(self):
        return self.formula.all_variables

    @property
    def header(self):
        return self.__class__.__name__


class Wrapper(LogicalEquation):
    def inner_scoring(self, batch, soft_logic: str):
        return self.inner_formula.scoring(batch, soft_logic)


class Attribute(LogicalEquation):
    def __init__(self, attribute_name: str, node: Node, no_grad=False):
        super().__init__(no_grad)
        assert isinstance(node, Node)
        self.attribute_name = attribute_name
        self.node = node
        if isinstance(node, Constant):
            self.variables = ()
        else:
            self.variables = (node,)

    def inner_scoring(self, batch, soft_logic: str):
        if isinstance(batch[0], Graph):
            # graph list
            logit_batch = [
                graph[
                    self.attribute_name,
                    self.node.name if isinstance(self.node, Constant) else None
                ] for graph in batch
            ]
            logits, valid_mask = batch_tensor_with_padding(logit_batch, 0)
            scores = logit_to_prob(logits)
        else:
            # collated batch
            attributes, valid_mask, _, _, attribute_names, _ = batch
            index = attribute_names.index(self.attribute_name)
            if attributes.is_sparse:
                scores = sparse_select(attributes, 1, index)
            else:
                scores = attributes[:, index]
            if valid_mask is not None:
                valid_mask = valid_mask[:, index]
        return scores, valid_mask, self.variables

    def str_info(self, replace_vars, all_variables):
        info = super().str_info(replace_vars, all_variables)
        node = self.replace_variables((self.node,), replace_vars)[0]
        info["str"] = f"{self.attribute_name}({node})"
        return info


class Relation(LogicalEquation):
    def __init__(self, relation_name: str, node1: Node, node2: Node,
                 no_grad=False):
        super().__init__(no_grad)
        assert isinstance(node1, Node)
        assert isinstance(node2, Node)
        self.relation_name = relation_name
        self.nodes = (node1, node2)
        self.variables = tuple(
            node for node in self.nodes if isinstance(node, Variable))

    def inner_scoring(self, batch, soft_logic: str):
        if isinstance(batch[0], Graph):
            # graph list
            logit_batch = [
                graph[
                    self.relation_name,
                    self.nodes[0].name if isinstance(self.nodes[0], Constant)
                    else None,
                    self.nodes[1].name if isinstance(self.nodes[1], Constant)
                    else None
                ] for graph in batch
            ]
            logits, valid_mask = batch_tensor_with_padding(logit_batch, 0)
            scores = logit_to_prob(logits)
        else:
            # collated batch
            _, _, relations, valid_mask, _, relation_names = batch
            index = relation_names.index(self.relation_name)
            if relations.is_sparse:
                scores = sparse_select(relations, 1, index)
            else:
                scores = relations[:, index]
            if valid_mask is not None:
                valid_mask = valid_mask[:, index]
        return scores, valid_mask, self.variables

    def str_info(self, replace_vars, all_variables):
        info = super().str_info(replace_vars, all_variables)
        nodes = self.replace_variables(self.nodes, replace_vars)
        info["str"] = f"{self.relation_name}{nodes}"
        return info


class Not(Unary):
    @staticmethod
    def static_scoring(results, soft_logic):
        scores, valid_mask, variables = results
        if soft_logic.startswith("naive-prob"):
            scores = 1 - scores
        elif soft_logic == "minmax-prob":
            scores = 1 - scores
        return scores, valid_mask, variables


class Copy(Unary):
    @staticmethod
    def static_scoring(results, soft_logic):
        scores, valid_mask, variables = results
        return scores, valid_mask, variables

    def str_info(self, replace_vars, all_variables):
        info = self.formula.str_info(replace_vars, all_variables)
        return info


class Or(Binary):
    @staticmethod
    def static_scoring(results, soft_logic):
        scores1, scores2, valid_mask, variables = results
        if soft_logic.startswith("naive-prob"):
            # print("C", end="", flush=True)
            if scores1.is_sparse:
                scores1 = scores1.coalesce()
                scores2 = scores2.coalesce()
                if scores1._nnz() == scores2._nnz() and \
                        torch.all(scores1.indices() == scores2.indices()):
                    scores = torch.sparse_coo_tensor(
                        scores1.indices(),
                        scores1.values() + scores2.values() -
                        scores1.values() * scores2.values(),
                        scores1.shape
                    )
                else:
                    scores = scores1 + scores2 - scores1 * scores2
            else:
                scores = scores1 + scores2 - scores1 * scores2
            # print("D", end="", flush=True)
        elif soft_logic == "minmax-prob":
            scores = torch.max(scores1, scores2)
        return scores, valid_mask, variables

    def binary_str(self, str1, str2):
        return f"{str1} ∨ {str2}"


class And(Binary):
    @staticmethod
    def static_scoring(results, soft_logic):
        scores1, scores2, valid_mask, variables = results
        if soft_logic.startswith("naive-prob"):
            scores = scores1 * scores2
        elif soft_logic == "minmax-prob":
            scores = torch.min(scores1, scores2)
        return scores, valid_mask, variables

    def binary_str(self, str1, str2):
        return f"{str1} ∧ {str2}"


class Xor(Binary):
    @staticmethod
    def static_scoring(results, soft_logic):
        scores1, scores2, valid_mask, variables = results
        if soft_logic.startswith("naive-prob"):
            scores = scores1 + scores2 - 2 * scores1 * scores2
        elif soft_logic == "minmax-prob":
            scores = torch.max(torch.min(scores1, 1-scores2),
                               torch.min(1-scores1, scores2))
        return scores, valid_mask, variables

    def binary_str(self, str1, str2):
        return f"{str1} ≠ {str2}"


class Diff(Binary):
    @staticmethod
    def static_scoring(results, soft_logic):
        scores1, scores2, valid_mask, variables = results
        if soft_logic.startswith("naive-prob"):
            scores = scores1 * (1 - scores2)
        elif soft_logic == "minmax-prob":
            scores = torch.min(scores1, 1 - scores2)
        return scores, valid_mask, variables

    def binary_str(self, str1, str2):
        return f"{str1} ≠ {str2}"


class Equals(Binary):
    @staticmethod
    def static_scoring(results, soft_logic):
        scores1, scores2, valid_mask, variables = results
        if soft_logic.startswith("naive-prob"):
            scores = scores1 * scores2 + (1-scores1) * (1-scores2)
        elif soft_logic == "minmax-prob":
            scores = torch.max(torch.min(scores1, scores2),
                               torch.min(1-scores1, 1-scores2))
        return scores, valid_mask, variables

    def binary_str(self, str1, str2):
        return f"{str1} == {str2}"


class Nand(Binary):
    @staticmethod
    def static_scoring(results, soft_logic):
        scores1, scores2, valid_mask, variables = results
        if soft_logic.startswith("naive-prob"):
            scores = 1 - scores1 * scores2
        elif soft_logic == "minmax-prob":
            scores = 1 - torch.min(scores1, scores2)
        return scores, valid_mask, variables

    def binary_str(self, str1, str2):
        return f"{str1} NAND {str2}"


class Exist(Quantified):
    @staticmethod
    def static_scoring(results, soft_logic):
        scores, valid_mask, variables, reduce_variables = results
        final_variables = tuple(_var for _var in variables if _var not in
                                reduce_variables)
        ndim = scores.ndim
        dims = sorted(tuple(
            ndim - len(variables) + variables.index(var)
            for var in reduce_variables
        ))
        # scores = mask_with_value(scores, valid_mask, 0)
        if valid_mask is not None:
            scores = scores.min(valid_mask)
            valid_mask = torch.amax(valid_mask, dims)
        if soft_logic.startswith("naive-prob"):
            if scores.is_sparse:
                if scores._nnz() == 0:
                    scores = torch.sparse_coo_tensor(
                        [[]]*(ndim-len(dims)), [],
                        [scores.shape[_i] for _i in range(ndim)
                         if _i not in dims],
                        dtype=scores.dtype, device=scores.device
                    )
                else:
                    scores = torch.sparse.sum(scores, dims)
                    # scores = normalize_max(scores)
                    scores = scores.coalesce()
                    scores = torch.sparse_coo_tensor(
                        scores.indices(),
                        scores.values().clamp_max(1), scores.shape
                    )
            else:
                scores = 1 - scores
                for dim in reversed(sorted(dims)):
                    scores = scores.prod(dim)
                scores = 1 - scores
        elif soft_logic == "minmax-prob":
            scores = torch.amax(scores, dims)
        return scores, valid_mask, final_variables

    @property
    def header(self):
        return "∃"


class ForAll(Quantified):
    @staticmethod
    def static_scoring(results, soft_logic):
        scores, valid_mask, variables, reduce_variables = results
        final_variables = tuple(_var for _var in variables if _var not in
                                reduce_variables)
        ndim = scores.ndim
        dims = sorted(tuple(
            ndim - len(variables) + variables.index(var)
            for var in reduce_variables
        ))
        # scores = mask_with_value(scores, valid_mask, 1)
        if valid_mask is not None:
            scores = 1 - (1 - scores).min(valid_mask)
            valid_mask = torch.amax(valid_mask, dims)
        if soft_logic.startswith("naive-prob"):
            for dim in reversed(sorted(dims)):
                scores = scores.prod(dim)
        elif soft_logic == "minmax-prob":
            scores = torch.amin(scores, dims)
        return scores, valid_mask, final_variables

    @property
    def header(self):
        return "∀"


class DeduceTo(Wrapper, Binary):
    def __init__(self, formula1: LogicalEquation, formula2: LogicalEquation,
                 no_grad=False):
        super().__init__(formula1, formula2, no_grad)
        self.inner_formula = Or(Not(formula1), formula2)
        self.formula1 = formula1
        self.formula2 = formula2

    @staticmethod
    def static_scoring(results, soft_logic: str):
        score1, score2, valid_mask, variables = results
        score1, _, _ = Not.static_scoring(
            (score1, valid_mask, variables), soft_logic)
        return Or.static_scoring(
            (score1, score2, valid_mask, variables), soft_logic
        )

    def binary_str(self, str1, str2):
        return f"{str1} ~> {str2}"


class HardDeduceTo(Wrapper, Binary):
    def __init__(self, formula1: LogicalEquation, formula2: LogicalEquation,
                 no_grad=False):
        super().__init__(formula1, formula2, no_grad)
        self.inner_formula = Or(Not(formula1, True), formula2)
        self.formula1 = formula1
        self.formula2 = formula2

    @staticmethod
    def static_scoring(results, soft_logic: str):
        score1, score2, valid_mask, variables = results
        with torch.no_grad():
            score1, _, _ = Not.static_scoring(
                (score1, valid_mask, variables), soft_logic)
        return Or.static_scoring(
            (score1, score2, valid_mask, variables), soft_logic
        )

    def binary_str(self, str1, str2):
        return f"{str1} -> {str2}"


class ReplaceVariable(Unary):
    def __init__(self, from_var, to_var,
                 formula: LogicalEquation, no_grad=False):
        super().__init__(formula, no_grad)
        self.variables = tuple(var if var != from_var else to_var
                               for var in formula.variables)
        self.formula = formula
        self.from_var = from_var
        self.to_var = to_var

    def inner_scoring(self, batch, soft_logic):
        scores, valid_mask, variables = self.formula.scoring(
            batch, soft_logic)
        return scores, valid_mask, self.variables

    def str_info(self, replace_vars, all_variables):
        replace_vars = deepcopy(replace_vars)
        replace_vars[self.from_var] = self.to_var
        info = self.formula.str_info(replace_vars, all_variables)
        return info


class BinaryChainAnd(Wrapper, Binary):
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
    def static_scoring(pre_results, soft_logic):
        score1, score2, valid_mask, variables = pre_results
        scores = spmatmul(score1, score2)
        if scores.is_sparse:
            max_score = sparse_max(scores)
        else:
            max_score = scores.max()
        scores = sparse_scalarmul(scores, 1 / max(1, max_score))
        return scores, valid_mask, variables
