import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lerp.utils import set_seed

from lerp.logic import Variable, Or, Copy, And, Exist, Wrapper
from lerp.bridged_lerpnet import VectorBinaryChainAnd, LogicIndexLayer, \
    LogicVectorInputLayer, LogicMetaVectorLayer, LogicSelectionVectorLayer
from lerp.utils import RunningMean

from model_base import LearnerBase


class Bridged_LerpModel(Wrapper):
    def __init__(self, rank, width, depth, attributes, relations,
                 init_var, temperature, no_grad=False):
        super().__init__(no_grad)
        self.n_variables = 2
        self.variables = tuple([Variable(f"X{i}") for i in range(2)])
        self.width = width
        self.depth = depth
        self.rank = rank
        self.init_var = init_var
        self.temperature = temperature
        self.build_model(attributes, relations)

    def str_info(self, replace_vars, all_variables):
        return self.inner_formula.str_info(replace_vars, all_variables)

    def build_model(self, attributes, relations):
        self.input_layer = LogicVectorInputLayer(
                self.variables, attributes, relations, to_sparse=False,
                no_quantified=True)
        layers = [self.input_layer]
        for i in range(self.depth):
            layers.append(
                LogicMetaVectorLayer(
                    layers[-1], self.width, 2,
                    [(Copy, {}), (VectorBinaryChainAnd, {}), (And, {}),
                     (Or, {}),
                     (Exist, {"reduce_var": self.variables[:1]})],
                    True, False, self.init_var, self.temperature
                )
            )
        end_layer = layers[-1]
        end_layer.caching = True
        for i in range(self.rank):
            layers.append(LogicSelectionVectorLayer(
                end_layer, 1, True, True,
                self.init_var, self.temperature
            ))
        inner_formula = Or(layers[-1], layers[-2])
        inner_formula.variables = self.variables[:1]
        for i in range(3, self.rank+1):
            inner_formula = Or(inner_formula, layers[-i])
            inner_formula.variables = self.variables[:1]
        inner_formula = LogicIndexLayer(inner_formula, 0)
        self.formula_list = nn.ModuleList(layers)
        self.inner_formula = inner_formula

    def classify(self, batch, y, soft_logic, final_forall_logic, loss_fn,
                 distinct_variables=False):
        scores, score_tensor, valid_mask = self(
            batch, soft_logic, final_forall_logic, distinct_variables)
        if loss_fn == "nll":
            loss = - (0.5 - y * (0.5 - scores)).log()
        elif loss_fn == "square":
            loss = (0.5 + y * (0.5 - scores)).pow(2)
        else:
            raise NotImplementedError()
        return loss, scores, score_tensor

    def __repr__(self):
        info = self.str_info({}, self.all_variables)
        return f"confidence={info['confidence']},  " + info["str"]

    def entropy(self):
        entropy = self.inner_formula.entropy()
        for _formula in self.formula_list:
            entropy = entropy + _formula.entropy()
        return entropy


class Learner(LearnerBase):
    """
    This class builds a computation graph that represents the
    neural ILP model and handles related graph running acitivies,
    including update, predict, and get_attentions for given queries.

    Args:
        option: hyper-parameters
    """

    def __init__(self, option):
        super().__init__()
        self.seed = option.seed
        self.num_step = option.num_step
        self.rank = option.rank
        self.num_layer = option.num_layer
        self.rnn_state_size = option.rnn_state_size

        self.norm = not option.no_norm
        self.thr = option.thr
        self.dropout = option.dropout
        self.learning_rate = option.learning_rate
        self.accuracy = option.accuracy
        self.top_k = option.top_k

        self.num_entity = option.num_entity
        self.num_operator = option.num_operator
        self.query_is_language = option.query_is_language

        if not option.query_is_language:
            self.num_query = option.num_query
            self.query_embed_size = option.query_embed_size
        else:
            self.vocab_embed_size = option.vocab_embed_size
            self.query_embed_size = self.vocab_embed_size
            self.num_vocab = option.num_vocab
            self.num_word = option.num_word

        set_seed(self.seed)
        self.headwise = option.headwise
        self.no_rev_in_model = option.no_rev_in_model
        self.num_relation = option.num_relation
        self.width = option.width
        self.depth = option.depth
        self.init_var = option.init_var
        self.model = option.model
        model_class = Bridged_LerpModel
        self.logic_fns = nn.ModuleList([
            model_class(
                self.rank, self.width, self.depth, ["IsTail"],
                list(range(self.num_relation)), self.init_var, 1
            )
            for i in range(self.num_relation * 2)
        ])
        self.soft_logic = option.soft_logic
        self.sparse = option.sparse
        self.running_mean = {
            split: {
                cat: RunningMean(0.97)
                for cat in ("loss", "in_top")
            } for split in ("train", "test")
        }

    def _random_uniform_unit(self, r, c):
        """ Initialize random and unit row norm matrix of size (r, c). """
        bound = 6. / np.sqrt(c)
        init_matrix = np.random.uniform(-bound, bound, (r, c))
        init_matrix = np.array(list(map(
            lambda row: row / np.linalg.norm(row), init_matrix)))
        # print('init',init_matrix)
        init_matrix = torch.tensor(init_matrix, dtype=torch.float32)
        return init_matrix

    def _clip_if_not_None(self, g, v, low, high):
        """ Clip not-None gradients to (low, high). """
        """ Gradient of T is None if T not connected to the objective. """
        if g is not None:
            # return (tf.clip_by_value(g, low, high), v)
            return (torch.clamp(g, low, high), v)
        else:
            return (g, v)

    def _inner_run_graph(self, queries, heads, tails, database):
        # print("start training", time.ctime())
        targets = F.one_hot(heads, num_classes=self.num_entity)

        def _dense(tensor):
            if self.sparse:
                return tensor
            else:
                return tensor.to_dense()

        def _sparse(tensor):
            if self.sparse:
                return tensor.to_sparse()
            else:
                return tensor

        data_matrix = torch.stack(
            [
                _dense(database[_i])
                for _i in range(self.num_relation)
            ]
        )

        query_onehot = F.one_hot(tails, num_classes=self.num_entity).float()[
            :, None].to_sparse()
        batch_size = queries.shape[0]
        batch = (query_onehot, None,
                 torch.stack([data_matrix]*batch_size), None,
                 ["IsTail"], list(range(self.num_operator)))
        _query = queries[0, 0]
        predictions, _, _ = self.logic_fns[_query].scoring(
            batch, self.soft_logic)

        if self.norm:
            predictions /= predictions.sum(1).unsqueeze(1)

        final_loss = - torch.sum(
            targets * predictions.clamp_min(self.thr).log(), 1)

        if not self.accuracy:
            topk = predictions.topk(self.top_k)[1]
            in_top = (heads.unsqueeze(1) == topk).any(dim=1)
        else:
            _, indices = predictions.topk(self.top_k, sorted=False)[1]
            in_top = torch.squeeze(indices) == heads

        results = {
            "final_loss": final_loss,
            "in_top": in_top,
            "predictions": predictions,
            # "attention_operators": attention_operators_list,
            # "vocab_embedding": self.vocab_embedding,
        }
        return results

    def update(self, qq, hh, tt, mdb):
        results = self._run_graph(qq, hh, tt, mdb)
        self.running_mean["train"]["loss"].update(results["final_loss"].mean())
        self.running_mean["train"]["in_top"].update(
            results["in_top"].float().mean())
        return results["final_loss"], results["in_top"]

    def predict(self, qq, hh, tt, mdb):
        with torch.no_grad():
            results = self._run_graph(qq, hh, tt, mdb)
        self.running_mean["test"]["loss"].update(results["final_loss"].mean())
        self.running_mean["test"]["in_top"].update(
            results["in_top"].float().mean())
        return results["final_loss"], results["in_top"]

    def get_predictions_given_queries(self, qq, hh, tt, mdb):
        with torch.no_grad():
            results = self._run_graph(qq, hh, tt, mdb)
            return results["in_top"], results["predictions"]

    def get_attentions_given_queries(self, queries):
        with torch.no_grad():
            qq = queries
            hh = [0] * len(queries)
            tt = [0] * len(queries)
            mdb = {r: ([(0, 0)], [0.], (self.num_entity, self.num_entity))
                   for r in range(self.num_operator // 2)}
            results = self._run_graph(qq, hh, tt, mdb)
            return results["attention_operators"]

    def get_vocab_embedding(self):
        return self.vocab_embedding.weight
