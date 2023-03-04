import torch
from torch.nn import Parameter
import numpy as np
from copy import deepcopy
from .utils import prob_to_logit, logit_to_prob
from .sparse import sparse_single_add


class Graph(torch.nn.Module):
    def __init__(self, n_nodes, n_attributes, n_relations,
                 default_logit, on_torch, requires_grad, sparse):
        super().__init__()
        self.on_torch = on_torch
        self.requires_grad = requires_grad
        self.n_nodes = n_nodes
        self.n_attributes = n_attributes
        self.n_relations = n_relations + 1
        self.sparse = sparse

        self.node_names = [f"node_{i}" for i in range(n_nodes)]
        self.attribute_names = [f"attribute_{i}" for i in range(n_nodes)]
        self.relation_names = [f"relation_{i}" for i in range(n_nodes)] + [
            "Equal"
        ]
        self.other_info = {}

        # logits
        if sparse:
            self.attributes = torch.sparse_coo_tensor(
                [[], []], [], (n_attributes, n_nodes), dtype=torch.float)
            self.relations = torch.sparse_coo_tensor(
                [[n_relations]*n_nodes, list(range(n_nodes)),
                 list(range(n_nodes))],
                [1]*n_nodes, (n_relations+1, n_nodes, n_nodes),
                dtype=torch.float)
        elif on_torch:
            self.attributes = Parameter(
                torch.zeros(n_attributes, n_nodes),
                requires_grad=requires_grad)
            self.relations = Parameter(
                torch.zeros(n_relations + 1, n_nodes, n_nodes),
                requires_grad=requires_grad)
            self.attributes.data[:] = default_logit
            self.relations.data[:] = default_logit
            self.relations.data[-1] = prob_to_logit(0)
            self.relations.data[-1].fill_diagonal_(prob_to_logit(1))
        else:
            self.attributes = np.zeros(
                (n_attributes, n_nodes))
            self.relations = np.zeros(
                (n_relations + 1, n_nodes, n_nodes))
            self.attributes[:] = default_logit
            self.relations[:] = default_logit
            self.relations[-1] = prob_to_logit(-1)
            np.fill_diagonal(self.relations[-1], prob_to_logit(1))

    @classmethod
    def empty_from_names(cls, node_names, attribute_names, relation_names,
                         default_logit, on_torch, requires_grad, sparse):
        graph = cls(len(node_names), len(attribute_names), len(relation_names),
                    default_logit, on_torch, requires_grad, sparse)
        graph.node_names = node_names
        graph.attribute_names = attribute_names
        graph.relation_names = relation_names + ["Equal"]
        return graph

    @classmethod
    def from_prob_dict(cls, graph_dict):
        graph = cls.empty_from_names(
            graph_dict["node_names"],
            graph_dict["attribute_names"],
            graph_dict["relation_names"],
            prob_to_logit(graph_dict["default_probability"]),
            graph_dict.get("on_torch", True),
            graph_dict.get("requires_grad", True),
        )
        if len(graph_dict["attributes"]) > 0:
            for node_name, attributes in graph_dict["attributes"].items():
                for attribute, prob in attributes.items():
                    graph[attribute, node_name] = prob_to_logit(prob)
        if len(graph_dict["relations"]) > 0:
            for node_pair, relations in graph_dict["relations"].items():
                if isinstance(node_pair, str):
                    node_pair = node_pair.split(" || ")
                node1, node2 = node_pair
                for relation, prob in relations.items():
                    graph[relation, node1, node2] = prob_to_logit(prob)
        return graph

    def __getitem__(self, index):
        if len(index) == 2:
            category = "attribute"
            predicate, node = index
        else:
            category = "relation"
            predicate, node1, node2 = index

        if category == "attribute":
            if isinstance(predicate, str):
                predicate = self.attribute_names.index(predicate)
            if isinstance(predicate, str):
                predicate = self.attribute_names.index(predicate)
            if isinstance(node, str):
                node = self.node_names.index(node)
            if node is not None:
                return self.attributes[predicate, node]
            else:
                return self.attributes[predicate]
        elif category == "relation":
            if isinstance(predicate, str):
                predicate = self.relation_names.index(predicate)
            if isinstance(node1, str):
                node1 = self.node_names.index(node1)
            if isinstance(node2, str):
                node2 = self.node_names.index(node2)
            if node1 is not None and node2 is not None:
                return self.relations[predicate, node1, node2]
            elif node1 is not None:
                return self.relations[predicate, node1, :]
            elif node2 is not None:
                return self.relations[predicate, :, node2]
            else:
                return self.relations[predicate]

    def __setitem__(self, index, logit):
        if len(index) == 2:
            category = "attribute"
            predicate, nodes = index
        else:
            category = "relation"
            predicate, node1, node2 = index
        if category == "attribute":
            if isinstance(predicate, str):
                predicate = self.attribute_names.index(predicate)
                nodes = self.node_names.index(nodes)
            if self.sparse:
                sparse_single_add(
                    self.attributes, [predicate, nodes], logit
                )
            elif self.on_torch:
                self.attributes.data[predicate, nodes] = logit
            else:
                self.attributes[predicate, nodes] = logit
        elif category == "relation":
            if isinstance(predicate, str):
                predicate = self.relation_names.index(predicate)
                node1 = self.node_names.index(node1)
                node2 = self.node_names.index(node2)
            if self.sparse:
                sparse_single_add(
                    self.relations, [predicate, node1, node2], logit
                )
            elif self.on_torch:
                self.relations.data[predicate, node1, node2] = logit
            else:
                self.relations[predicate, node1, node2] = logit

    def to_dict(self, as_prob=False, value_only=False):
        output = {
            "attributes":
            logit_to_prob(self.attributes) if as_prob
            else self.attributes,
            "relations":
            logit_to_prob(self.relations) if as_prob
            else self.relations
        }
        if not value_only:
            output.update({
                "n_nodes": self.n_nodes,
                "n_attributes": self.n_attributes,
                "n_relations": self.n_relations,
                "node_names": self.node_names,
                "attribute_names": self.attribute_names,
                "relation_names": self.relation_names,
            })
        return output

    def decode_dict(self, threshold=0.5):
        attributes = logit_to_prob(self.attributes)
        relations = logit_to_prob(self.relations)
        output = {"attributes": [], "relations": []}
        for i, node1 in enumerate(self.node_names):
            for attr, score in zip(self.attribute_names, attributes[:, i]):
                if score > threshold:
                    output["attributes"].append([node1, attr])
            for j, node2 in enumerate(self.node_names):
                for rel, score in zip(self.relation_names, relations[:, i, j]):
                    if score > threshold:
                        output["relations"].append([node1, node2, rel])
        return output

    def to_torch(self):
        graph = deepcopy(self)

        if graph.on_torch:
            return graph
        graph.attributes = torch.tensor(graph.attributes)
        graph.relations = torch.tensor(graph.relations)
        graph.on_torch = True
        return graph

    def to_numpy(self):
        graph = deepcopy(self)

        if not graph.on_torch:
            return graph
        graph.attributes = graph.attributes.numpy()
        graph.relations = graph.relations.numpy()
        graph.on_torch = False
        return graph
