import torch
import torch.nn as nn
import torch.nn.functional as F

from lerp.utils import set_seed, RunningMean
from lerp.lerpnet import \
    LogicLerpInputLayer, LogicMetaLerpLayer, LerpChaining, \
    And_or_Not
from model_base import LearnerBase


class LerpModel(nn.Module):
    def __init__(self, rank, length, depth, width, n_relations,
                 init_var, temperature, dense):
        super().__init__()
        self.n_variables = 1
        self.rank = rank
        self.length = length
        self.depth = depth
        self.width = width
        self.n_relations = n_relations
        self.init_var = init_var
        self.temperature = temperature
        self.dense = dense
        self.build_model(n_relations)

    def build_model(self, relations):
        # Building Lerp
        self.lerp_input_layer = LogicLerpInputLayer(
            self.n_relations, self.width, self.dense)
        self.lerp_hidden_layers = nn.ModuleList([
            LogicMetaLerpLayer(self.width, self.n_relations,
                               ("And", "Or", "Chaining", "Copy", "Not"),
                               True, self.temperature, self.dense)
            for _ in range(max(0, self.depth-1))
        ])

        # Building Chains
        self.constraints = nn.ModuleList([
            And_or_Not(self.rank, self.width)
            for _ in range(2 * self.n_relations * (self.length + 1))
        ])
        self.chaining_layers = nn.ModuleList([
            LerpChaining(self.rank, self.n_relations, True, self.dense)
            for _ in range(2 * self.n_relations * self.length)
        ])
        self.rule_weights = nn.Parameter(torch.zeros(
            self.n_relations*2, self.rank))
        # self.rule_weights = nn.Parameter(torch.zeros(self.rank))

        for params in self.parameters():
            nn.init.normal_(params, 0, self.init_var)

    def interpret_lerp(self, layer, index, relations):
        if layer == -1:
            layer = self.depth-1
        assert self.n_relations == len(relations)
        if layer == 0:
            weight, subindex = self.lerp_input_layer.weights[
                :, index].softmax(0).max(0)
            if subindex >= self.n_relations:
                output = weight, relations[subindex - self.n_relations]+"^T"
            else:
                output = weight, relations[subindex]
        else:
            lerp_layer = self.lerp_hidden_layers[layer-1]
            op_weight, op_index = lerp_layer.op_weights[index].softmax(
                0).max(0)
            arg1_weight, arg1_index = lerp_layer.arg1_weights[
                :, index].softmax(0).max(0)
            arg2_weight, arg2_index = lerp_layer.arg2_weights[
                :, index].softmax(0).max(0)
            op = lerp_layer.ops[op_index]
            chain_weight, chain_index = lerp_layer.chain.weights[
                index].softmax(0).max(0)
            if chain_index >= self.n_relations:
                chain_rel = relations[chain_index-self.n_relations]+"^T"
            else:
                chain_rel = relations[chain_index]

            arg1_result = self.interpret_lerp(
                layer-1, arg1_index, relations)
            arg1_weight = arg1_weight * arg1_result[0]
            arg2_result = self.interpret_lerp(
                layer-1, arg2_index, relations)
            arg2_weight = arg2_weight * arg2_result[0]

            if op == "Copy":
                output = op_weight * arg2_weight, arg2_result[1]
            if op == "Chaining":
                output = op_weight * arg2_weight * chain_weight, \
                    f"{chain_rel}--{arg2_result[1]}"
            if op == "Not":
                output = op_weight * arg1_weight, f"Not({arg1_result[1]})"
            elif op == "And":
                output = op_weight * arg1_weight * arg2_weight,\
                    f"And({arg1_result[1]}, {arg2_result[1]})"
            elif op == "Or":
                output = op_weight * arg1_weight * arg2_weight,\
                    f"Or({arg1_result[1]}, {arg2_result[1]})"
        return output[0].item(), output[1]

    def interpret_rule(self, query, rule_index, relations):
        weight = self.rule_weights[query, rule_index].sigmoid()
        # weight = self.rule_weights[rule_index].sigmoid()
        text = "X"
        for i in range(self.length+1):
            if i > 0:
                _chain = self.chaining_layers[query*self.length+i-1]
                rel_weight, rel = _chain.weights[rule_index].softmax(0).max(0)
                equity_weight, equity = _chain.equity_weight[
                    rule_index].softmax(0).max(0)
                if equity == 1:
                    weight = weight * equity_weight
                else:
                    weight = weight * equity_weight * rel_weight
                    if rel >= self.n_relations:
                        rel_text = relations[rel-self.n_relations]
                        text = text + "--" + rel_text + "^T"
                    else:
                        rel_text = relations[rel]
                        text = text + "--" + rel_text

            if self.depth > 0:
                _constraint = self.constraints[query*(self.length+1)+i]
                gate_weight = _constraint.gate_weights[rule_index].sigmoid()
                if gate_weight > 0.5:
                    weight = weight * gate_weight
                else:
                    select_weight, select_index = _constraint.select_weights[
                        rule_index].softmax(0).max(0)
                    lerp_weight, lerp_text = self.interpret_lerp(
                        -1, select_index, relations)
                    weight = weight * (1-gate_weight) * lerp_weight * \
                        select_weight
                    text = text + f"(AND {lerp_text})"
        return weight.item(), text

    def forward(self, query, tails, heads, num_entity, database):
        tails_onehot = F.one_hot(tails, num_classes=num_entity)
        # Getting Lerp
        database = [database[i] for i in range(self.n_relations)]
        if self.dense:
            database = torch.stack(database).to_dense()
            if query < self.n_relations:
                database[query % self.n_relations, heads, tails] = 0
            else:
                database[query % self.n_relations, tails, heads] = 0
        lerp, quantified = self.lerp_input_layer(database)
        for _layer in self.lerp_hidden_layers:
            lerp = _layer(lerp, database)

        if self.depth > 0:
            x = self.constraints[query*(self.length+1)](
                tails_onehot[:, None], lerp)
        else:
            x = tails_onehot[:, None].expand(
                tails_onehot.shape[0], self.rank, num_entity).float()
        # Chaining
        chaining_layers = self.chaining_layers[
            query*self.length:(query+1)*self.length
        ]
        constraints = self.constraints[
            query*(self.length+1)+1:(query+1)*(self.length+1)
        ]
        for _chaining, _constraint in zip(
                chaining_layers, constraints):
            x = _chaining(x, database)
            if self.depth > 0:
                x = _constraint(x, lerp)
        x = x * (self.rule_weights[query] - 2).sigmoid()[None, :, None]
        x = x.sum(1)

        return x


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
        self.no_rev_in_model = option.no_rev_in_model
        self.num_relation = option.num_relation
        self.width = option.width
        self.length = option.length
        self.depth = option.depth
        self.init_var = option.init_var
        self.model_name = option.model
        self.soft_logic = option.soft_logic
        self.sparse = option.sparse
        self.dense = option.dense
        self.lerp = LerpModel(
            self.rank, self.length, self.depth, self.width,
            self.num_relation, self.init_var, 1, self.dense)
        self.running_mean = {
            split: {
                cat: RunningMean(0.97)
                for cat in ("loss", "in_top")
            } for split in ("train", "test")
        }

    def _inner_run_graph(self, queries, heads, tails, database):
        # print("start training", time.ctime())
        targets = F.one_hot(heads, num_classes=self.num_entity)

        # data_matrix = torch.stack(
        #     [
        #         _dense(database[_i])
        #         for _i in range(self.num_relation)
        #     ]
        # )

        _query = queries[0, 0]
        # predictions, _, _ = self.lerp.chains[_query].scoring(
        #     batch, self.soft_logic)
        predictions = self.lerp(
            _query, tails, heads, self.num_entity, database
        )

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
