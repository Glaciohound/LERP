import torch
import torch.nn as nn
import torch.nn.functional as F
from lerp.utils import set_seed

from lerp.utils import RunningMean
from lerp.lerpnet import \
    LogicLerpInputLayer, LogicMetaLerpLayer
from model_base import LearnerBase


class TransEModel(nn.Module):
    def __init__(self, depth, width, embed_dim,
                 is_logical, num_entity, num_relation,
                 margin, init_var, temperature, dense):
        super().__init__()
        self.n_variables = 1
        self.depth = depth
        self.width = width
        self.num_relation = num_relation
        self.num_entity = num_entity
        self.is_logical = is_logical
        self.init_var = init_var
        self.margin = margin
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.dense = dense
        self.build_model(num_relation)

    def build_model(self, relations):
        # Building Lerp
        self.lerp_input_layer = LogicLerpInputLayer(
            self.num_relation, self.width, self.dense)
        self.lerp_hidden_layers = nn.ModuleList([
            LogicMetaLerpLayer(self.width, self.num_relation,
                               ("And", "Or", "Chaining", "Copy", "Not"),
                               True, self.temperature, self.dense)
            for _ in range(max(0, self.depth-1))
        ])

        self.relation_embedding = nn.Embedding(
            2*self.num_relation, self.embed_dim)
        self.entity_embedding = nn.Embedding(
            self.num_entity, self.embed_dim)
        self.lerp_to_embed_rotation = nn.Parameter(
            torch.zeros(self.width, self.embed_dim)
        )

        for params in self.parameters():
            nn.init.normal_(params, 0, self.init_var)

    def entropy(self):
        entropy = (
            self.lerp_input_layer.entropy() +
            sum([layer.entropy() for layer in self.lerp_hidden_layers])
        )
        return entropy

    def forward(self, query, tails, heads, num_entity, database):
        if self.is_logical:
            database = [database[i] for i in range(self.num_relation)]
            if self.dense:
                database = torch.stack(database).to_dense()
            lerp, quantified = self.lerp_input_layer(database)
            lerp = lerp.transpose(0, 1)
            tails_embedding = lerp[tails].matmul(
                self.lerp_to_embed_rotation
            ) + self.entity_embedding(tails)
            target = lerp.matmul(self.lerp_to_embed_rotation) + \
                self.entity_embedding.weight
        else:
            tails_embedding = self.entity_embedding(tails)
            target = self.entity_embedding.weight
            lerp = None

        tails_embedding = F.normalize(tails_embedding, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)

        relation_embedding = self.relation_embedding(query)
        translated = tails_embedding + relation_embedding
        distance = (translated[:, None] - target[None]).abs().sum(-1)
        score = 1 / distance
        # for i in range(self.width):
        #     print(self.interpret_lerp(
        #         -1, i, ["aunt", "brother", "daughter", "father", "husband", "mother",
        #                 "nephew", "niece", "sister", "son", "uncle", "wife"]))
        # from IPython import embed; embed(); exit()
        return lerp, distance, score

    def interpret_lerp(self, layer, index, relations):
        if layer == -1:
            layer = self.depth-1
        assert self.num_relation == len(relations)
        if layer == 0:
            weight, subindex = self.lerp_input_layer.weights[
                :, index].softmax(0).max(0)
            if subindex >= self.num_relation:
                output = weight, relations[subindex - self.num_relation]+"^T"
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
            if chain_index >= self.num_relation:
                chain_rel = relations[chain_index-self.num_relation]+"^T"
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
        self.margin = option.margin
        self.embed_dim = option.embed_dim
        self.regulation = option.regulation
        self.entropy_regulation = option.entropy_regulation
        self.lerp = TransEModel(
            self.depth, self.width, self.embed_dim,
            self.model_name == "logical-transE",
            self.num_entity, self.num_relation, self.margin,
            self.init_var, 1, self.dense)
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
        lerp, distance, predictions = self.lerp(
            _query, tails, heads, self.num_entity, database
        )

        if self.norm:
            predictions /= predictions.sum(1).unsqueeze(1)

        positive_distance = distance[torch.arange(0, heads.shape[0]), heads]
        if self.model_name == "transE":
            final_loss = (
                positive_distance[:, None] - distance + self.margin
            ).clamp_min(0).sum(1)
        else:
            final_loss = (
                positive_distance[:, None] - distance + self.margin
            ).clamp_min(0).sum(1) + (
                self.lerp.entity_embedding.weight.pow(2).sum(1).mean() /
                lerp.matmul(self.lerp.lerp_to_embed_rotation
                            ).pow(2).sum(1).mean()
            ) * self.regulation + self.lerp.entropy() * self.entropy_regulation
        # final_loss = - torch.sum(
        #     targets * predictions.clamp_min(self.thr).log(), 1)

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
        self.running_mean["train"]["in_top"].update(results["in_top"].float().mean())
        return results["final_loss"], results["in_top"]

    def predict(self, qq, hh, tt, mdb):
        with torch.no_grad():
            results = self._run_graph(qq, hh, tt, mdb)
        self.running_mean["test"]["loss"].update(results["final_loss"].mean())
        self.running_mean["test"]["in_top"].update(results["in_top"].float().mean())
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
