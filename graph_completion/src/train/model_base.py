import numpy as np
import torch
import torch.nn as nn
import time

from lerp.logic import LogicalEquation


class LearnerBase(nn.Module):
    def _run_graph(self, qq, hh, tt, mdb):
        LogicalEquation.global_cachestamp = time.time()
        if not self.query_is_language:
            queries = [[q] * (self.num_step-1) + [self.num_query]
                       for q in qq]
        else:
            queries = [[q] * (self.num_step-1)
                       + [[self.num_vocab] * self.num_word] for q in qq]

        heads = hh
        tails = tt
        database = {}
        for r in range(self.num_operator // 2):
            indices, values, ndim = mdb[r]
            indices = np.array(indices).transpose()
            if indices.shape[0] == 0:
                indices = [[], []]
            database[r] = torch.sparse_coo_tensor(
                indices, values, ndim, dtype=torch.float).to(self.device)
        queries = torch.LongTensor(queries).to(self.device)
        heads = torch.LongTensor(heads).to(self.device)
        tails = torch.LongTensor(tails).to(self.device)
        graph_output = self._inner_run_graph(queries, heads, tails, database)
        LogicalEquation.global_cachestamp = None
        return graph_output
