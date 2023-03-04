import numpy as np
import os
import copy
from math import ceil
from collections import Counter

def resplit(train, facts, no_link_percent):
    num_train = len(train)
    num_facts = len(facts)
    all = train + facts

    if no_link_percent == 0.:
        np.random.shuffle(all)
        new_train = all[:num_train]
        new_facts = all[num_train:]
    else:
        link_cntr = Counter()
        for tri in all:
            link_cntr[(tri[1], tri[2])] += 1
        tmp_train = []
        tmp_facts = []
        for tri in all:
            if link_cntr[(tri[1], tri[2])] + link_cntr[(tri[2], tri[1])] > 1:
                if np.random.random() < no_link_percent:
                    tmp_facts.append(tri)
                else:
                    tmp_train.append(tri)
            else:
                tmp_train.append(tri)

        if len(tmp_train) > num_train:
            np.random.shuffle(tmp_train)
            new_train = tmp_train[:num_train]
            new_facts = tmp_train[num_train:] + tmp_facts
        else:
            np.random.shuffle(tmp_facts)
            num_to_fill = num_train - len(tmp_train)
            new_train = tmp_train + tmp_facts[:num_to_fill]
            new_facts = tmp_facts[num_to_fill:]

    assert(len(new_train) == num_train)
    assert(len(new_facts) == num_facts)

    return new_train, new_facts

class Data(object):
    def __init__(self, folder, seed, type_check, domain_size, no_extra_facts,
                 limit_supernode, induction, data_relink, train_no_facts):
        np.random.seed(seed)
        self.seed = seed
        self.type_check = type_check
        self.domain_size = domain_size
        self.use_extra_facts = not no_extra_facts
        self.query_include_reverse = True
        self.induction = induction
        self.data_relink = data_relink
        self.train_no_facts = train_no_facts
        assert not (induction and data_relink)

        self.relation_file = os.path.join(folder, "relations.txt")
        self.entity_file = os.path.join(folder, "entities.txt")

        self.relation_to_number, self.entity_to_number = self._numerical_encode()
        self.number_to_entity = {v: k for k, v in self.entity_to_number.items()}
        self.num_relation = len(self.relation_to_number)
        self.num_query = self.num_relation * 2
        self.num_entity = len(self.entity_to_number)

        self.test_file = os.path.join(folder, "test.txt")
        self.train_file = os.path.join(folder, "train.txt")
        self.valid_file = os.path.join(folder, "valid.txt")

        if os.path.isfile(os.path.join(folder, "facts.txt")):
            self.facts_file = os.path.join(folder, "facts.txt")
            self.share_db = True
        else:
            self.train_facts_file = os.path.join(folder, "train_facts.txt")
            self.test_facts_file = os.path.join(folder, "test_facts.txt")
            self.share_db = False
            if not os.path.exists(self.train_facts_file):
                self.facts_file = self.train_file
                self.train_facts_file = self.train_file
                self.test_facts_file = self.train_file
                self.share_db = True

        self.test, self.num_test = self._parse_triplets(self.test_file)
        self.train, self.num_train = self._parse_triplets(self.train_file)
        if os.path.isfile(self.valid_file):
            self.valid, self.num_valid = self._parse_triplets(self.valid_file)
        else:
            self.valid, self.train = self._split_valid_from_train()
            self.num_valid = len(self.valid)
            self.num_train = len(self.train)

        if self.share_db:
            self.facts, self.num_fact = self._parse_triplets(self.facts_file)
            if self.induction:
                self.facts, self.train = self.resplit_induction(
                    self.facts, self.train, self.test)
            if self.data_relink:
                self.facts, self.train = self.resplit_relink(
                    self.facts, self.train, self.test)
            self.matrix_db = self._db_to_matrix_db(self.facts)
            self.matrix_db_train = self.matrix_db
            self.matrix_db_test = self.matrix_db
            self.matrix_db_valid = self.matrix_db
            if self.use_extra_facts:
                extra_mdb = self._db_to_matrix_db(self.train)
                self.augmented_mdb = self._combine_two_mdbs(self.matrix_db, extra_mdb)
                self.augmented_mdb_valid = self.augmented_mdb
                self.augmented_mdb_test = self.augmented_mdb
        else:
            self.train_facts, self.num_train_fact \
                = self._parse_triplets(self.train_facts_file)
            self.test_facts, self.num_test_fact \
                = self._parse_triplets(self.test_facts_file)
            self.matrix_db_train = self._db_to_matrix_db(self.train_facts)
            self.matrix_db_test = self._db_to_matrix_db(self.test_facts)
            self.matrix_db_valid = self._db_to_matrix_db(self.train_facts)

        if self.type_check:
            self.domains_file = os.path.join(folder, "stats/domains.txt")
            self.domains = self._parse_domains_file(self.domains_file)
            self.train = sorted(self.train, key=lambda x: x[0])
            self.test = sorted(self.test, key=lambda x: x[0])
            self.valid = sorted(self.valid, key=lambda x: x[0])
            self.num_operator = 2 * self.domain_size
        else:
            self.domains = None
            self.num_operator = 2 * self.num_relation

        # get rules for queries and their inverses appeared in train and test
        self.query_for_rules = list(set(list(zip(*self.train))[0]) | set(list(zip(*self.test))[0]) | set(list(zip(*self._augment_with_reverse(self.train)))[0]) | set(list(zip(*self._augment_with_reverse(self.test)))[0]))
        self.parser = self._create_parser()
        self.limit_supernode = limit_supernode

    def resplit_induction(self, facts, train, test):
        test_entities = set([item[i] for item in test for i in range(1, 3)])
        all_data = facts + train
        overlapped = np.array(
            [item[1] in test_entities or item[2] in test_entities
             for item in all_data])
        all_data = np.array(all_data)
        facts1 = all_data[overlapped]
        remains = all_data[np.logical_not(overlapped)]
        permutation = np.random.permutation(remains.shape[0])
        new_train = remains[permutation[:self.num_train]]
        facts2 = remains[permutation[self.num_train:]]
        new_facts = np.concatenate([facts1, facts2])
        new_train = new_train[new_train[:, 0].argsort(0)].tolist()
        new_facts = new_facts.tolist()
        return new_facts, new_train

    def resplit_relink(self, facts, train, test):
        test_queries = set([item[0] for item in test])
        test_entities = set([item[i] for item in test for i in range(1, 3)])

    def _create_parser(self):
        """Create a parser that maps numbers to queries and operators given queries"""
        assert(self.num_query==2*len(self.relation_to_number)==2*self.num_relation)
        parser = {"query":{}, "operator":{}}
        number_to_relation = {value: key for key, value
                                         in self.relation_to_number.items()}
        for key, value in self.relation_to_number.items():
            parser["query"][value] = key
            parser["query"][value + self.num_relation] = "inv_" + key
        for query in range(self.num_relation):
            d = {}
            if self.type_check:
                for i, o in enumerate(self.domains[query]):
                    d[i] = number_to_relation[o]
                    d[i + self.domain_size] = "inv_" + number_to_relation[o]
            else:
                for k, v in number_to_relation.items():
                    d[k] = v
                    d[k + self.num_relation] = "inv_" + v
            parser["operator"][query] = d
            parser["operator"][query + self.num_relation] = d
        return parser

    def _parse_domains_file(self, file_name):
        result = {}
        with open(file_name, "r") as f:
            for line in f:
                l = line.strip().split(",")
                l = [self.relation_to_number[i] for i in l]
                relation = l[0]
                this_domain = l[1:1+self.domain_size]
                if len(this_domain) == self.domain_size:
                    pass
                else:
                    # fill in blanks
                    num_remain = self.domain_size - len(this_domain)
                    remains = [i for i in range(self.num_relation)
                                 if i not in this_domain]
                    pads = np.random.choice(remains, num_remain, replace=False)
                    this_domain += list(pads)
                this_domain.sort()
                assert(len(set(this_domain)) == self.domain_size)
                assert(len(this_domain) == self.domain_size)
                result[relation] = this_domain
        for r in range(self.num_relation):
            if r not in result.keys():
                result[r] = np.random.choice(range(self.num_relation),
                                             self.domain_size,
                                             replace=False)
        return result

    def _numerical_encode(self):
        relation_to_number = {}
        with open(self.relation_file) as f:
            for line in f:
                l = line.strip().split()
                assert(len(l) == 1)
                relation_to_number[l[0]] = len(relation_to_number)

        entity_to_number = {}
        with open(self.entity_file) as f:
            for line in f:
                l = line.strip().split()
                assert(len(l) == 1)
                entity_to_number[l[0]] = len(entity_to_number)
        return relation_to_number, entity_to_number

    def _parse_triplets(self, file):
        """Convert (head, relation, tail) to (relation, head, tail)"""
        output = []
        with open(file) as f:
            for line in f:
                l = line.strip().split("\t")
                assert(len(l) == 3)
                output.append((self.relation_to_number[l[1]],
                               self.entity_to_number[l[0]],
                               self.entity_to_number[l[2]]))
        return output, len(output)

    def _split_valid_from_train(self):
        valid = []
        new_train = []
        for fact in self.train:
            dice = np.random.uniform()
            if dice < 0.1:
                valid.append(fact)
            else:
                new_train.append(fact)
        np.random.shuffle(new_train)
        return valid, new_train

    def _db_to_matrix_db(self, db):
        matrix_db = {r: ([[0,0]], [0.], [self.num_entity, self.num_entity])
                     for r in range(self.num_relation)}
        for i, fact in enumerate(db):
            rel = fact[0]
            head = fact[1]
            tail = fact[2]
            value = 1.
            matrix_db[rel][0].append([head, tail])
            matrix_db[rel][1].append(value)
        return matrix_db

    def _combine_two_mdbs(self, mdbA, mdbB):
        """Assume mdbA and mdbB contain distinct elements."""
        new_mdb = {}
        for key, value in mdbA.items():
            new_mdb[key] = value
        for key, value in mdbB.items():
            try:
                value_A = mdbA[key]
                new_mdb[key] = [value_A[0] + value[0], value_A[1] + value[1], value_A[2]]
            except KeyError:
                new_mdb[key] = value
        return new_mdb

    def _count_batch(self, samples, batch_size):
        relations = zip(*samples)[0]
        relations_counts = Counter(relations)
        num_batches = [ceil(1. * x / batch_size) for x in relations_counts.values()]
        return int(sum(num_batches))

    def reset(self, batch_size):
        self.batch_size = batch_size
        self.train_start = 0
        self.valid_start = 0
        self.test_start = 0
        if not self.type_check:
            self.num_batch_train = self.num_train // batch_size + 1
            self.num_batch_valid = self.num_valid // batch_size + 1
            self.num_batch_test = self.num_test // batch_size + 1
        else:
            self.num_batch_train = self._count_batch(self.train, batch_size)
            self.num_batch_valid = self._count_batch(self.valid, batch_size)
            self.num_batch_test = self._count_batch(self.test, batch_size)

    def train_resplit(self, no_link_percent):
      new_train, new_facts = resplit(self.train, self.facts, no_link_percent)
      self.train = new_train
      self.matrix_db_train = self._db_to_matrix_db(new_facts)

    #########################################################################

    def _subset_of_matrix_db(self, matrix_db, domain):
        subset_matrix_db = {}
        for i, r in enumerate(domain):
            subset_matrix_db[i] = matrix_db[r]
        return subset_matrix_db

    def _augment_with_reverse(self, triplets):
        augmented = []
        for triplet in triplets:
            augmented += [triplet, (triplet[0]+self.num_relation,
                                    triplet[2],
                                    triplet[1])]
        return augmented

    def _get_reverse(self, triplets):
        reverse = []
        for triplet in triplets:
            reverse += [(triplet[0]+self.num_relation,
                         triplet[2], triplet[1])]
        return reverse

    def _next_batch(self, start, size, samples):
        assert(start < size)
        end = min(start + self.batch_size, size)
        if self.type_check:
            this_batch_tmp = samples[start:end]
            major_relation = this_batch_tmp[0][0]
            # assume sorted by relations
            batch_size = next((i for i in range(len(this_batch_tmp))
                                if this_batch_tmp[i][0] != major_relation),
                              len(this_batch_tmp))
            end = start + batch_size
            assert(end <= size)
        next_start = end % size
        this_batch = samples[start:end]
        if self.query_include_reverse:
            this_batch = self._augment_with_reverse(this_batch)
        this_batch_id = range(start, end)
        return next_start, this_batch, this_batch_id

    def _triplet_to_feed(self, triplets):
        if len(triplets) == 0:
            return None, None, None
        queries, heads, tails = zip(*triplets)
        return queries, heads, tails

    def limit_supernode_fn(self, matrix_db, this_batch):
        if self.limit_supernode <= 0:
            return matrix_db
        n_step = 2
        this_batch = np.array(this_batch)
        if this_batch.shape[0] == 0:
            shape = matrix_db[0][2]
            return {
                i: ([], [], shape)
                for i in matrix_db.keys()
            }
        all_mdb = np.concatenate([
            np.concatenate([np.array(values, dtype=int)[:, None] * key,
                            np.array(indices)], axis=1
                           )
            for key, (indices, values, shape) in matrix_db.items()
        ])
        all_mdb = np.concatenate([
            all_mdb,
            np.stack([-all_mdb[:, 0] - 1, all_mdb[:, 2], all_mdb[:, 1]], 1)
        ])

        entries = this_batch
        chosen_nodes = []
        filtered_entries = []
        for step in range(n_step):
            chosen_nodes = np.array(list(set(
                np.concatenate([entries[:, 1], entries[:, 2]])
            ).difference(chosen_nodes)))
            indices = (all_mdb[:, 1, None] == chosen_nodes[None]
                       ).max(1).astype(bool)
            entries = all_mdb[indices]
            np.random.shuffle(entries)
            entries = entries[np.argsort(entries[:, 1])]
            counter = np.zeros(entries.shape[0], dtype=int)
            for i in range(1, entries.shape[0]):
                if entries[i, 1] == entries[i-1, 1]:
                    counter[i] = counter[i-1] + 1
            filtered_entries.append(entries[counter < self.limit_supernode])
        filtered_entries = np.concatenate(filtered_entries)

        neg_entries = filtered_entries[filtered_entries[:, 0] < 0]
        neg_entries = np.stack([
            -neg_entries[:, 0] - 1, neg_entries[:, 2], neg_entries[:, 1]
        ], 1)
        final_entries = np.concatenate([
            filtered_entries[filtered_entries[:, 0] >= 0],
            neg_entries
        ])
        final_entries = final_entries[np.argsort(final_entries[:, 0])]
        new_mdb = dict()
        shape = matrix_db[0][2]
        for i in matrix_db.keys():
            entries_i = final_entries[final_entries[:, 0] == i]
            new_mdb[i] = [
                entries_i[:, 1:].tolist() if entries_i.shape[0] > 0
                else [],
                [1] * entries_i.shape[0],
                shape
            ]
        return new_mdb

    def next_test(self):
        self.test_start, this_batch, _ = self._next_batch(self.test_start,
                                                       self.num_test,
                                                       self.test)
        if self.share_db and self.use_extra_facts:
            matrix_db = self.augmented_mdb_test
        else:
            matrix_db = self.matrix_db_test

        if self.type_check:
            query = this_batch[0][0]
            matrix_db = self._subset_of_matrix_db(matrix_db,
                                                  self.domains[query])
        matrix_db = self.limit_supernode_fn(matrix_db, this_batch)
        return self._triplet_to_feed(this_batch), matrix_db

    def next_valid(self):
        self.valid_start, this_batch, _ = self._next_batch(self.valid_start,
                                                        self.num_valid,
                                                        self.valid)
        if self.share_db and self.use_extra_facts:
            matrix_db = self.augmented_mdb_valid
        else:
            matrix_db = self.matrix_db_valid

        if self.type_check:
            query = this_batch[0][0]
            matrix_db = self._subset_of_matrix_db(matrix_db,
                                                  self.domains[query])
        matrix_db = self.limit_supernode_fn(matrix_db, this_batch)
        return self._triplet_to_feed(this_batch), matrix_db

    def next_train(self):
        self.train_start, this_batch, this_batch_id = self._next_batch(self.train_start,
                                                        self.num_train,
                                                        self.train)

        if self.share_db and self.use_extra_facts:
            extra_facts = [fact for i, fact in enumerate(self.train) if i not in this_batch_id]
            extra_mdb = self._db_to_matrix_db(extra_facts)
            if not self.train_no_facts:
                augmented_mdb = self._combine_two_mdbs(extra_mdb, self.matrix_db_train)
            else:
                augmented_mdb = extra_mdb
            matrix_db = augmented_mdb
        else:
            matrix_db = self.matrix_db_train

        if self.type_check:
            query = this_batch[0][0]
            matrix_db = self._subset_of_matrix_db(matrix_db, self.domains[query])

        matrix_db = self.limit_supernode_fn(matrix_db, this_batch)
        return self._triplet_to_feed(this_batch), matrix_db


class DataPlus(Data):
    def __init__(self, folder, seed):
        np.random.seed(seed)
        self.seed = seed
        self.kb_relation_file = os.path.join(folder, "kb_relations.txt")
        self.kb_entity_file = os.path.join(folder, "kb_entities.txt")
        self.query_vocab_file = os.path.join(folder, "query_vocabs.txt")

        self.kb_relation_to_number = self._numerical_encode(self.kb_relation_file)
        self.kb_entity_to_number = self._numerical_encode(self.kb_entity_file)
        self.query_vocab_to_number = self._numerical_encode(self.query_vocab_file)

        self.test_file = os.path.join(folder, "test.txt")
        self.train_file = os.path.join(folder, "train.txt")
        self.valid_file = os.path.join(folder, "valid.txt")
        self.facts_file = os.path.join(folder, "facts.txt")

        self.test, self.num_test = self._parse_examples(self.test_file)
        self.train, self.num_train = self._parse_examples(self.train_file)
        self.valid, self.num_valid = self._parse_examples(self.valid_file)
        self.facts, self.num_fact = self._parse_facts(self.facts_file)
        self.all_exams = set([tuple(q + [h, t]) for (q, h, t) in self.train + self.test + self.valid])

        self.num_word = len(self.test[0][0])
        self.num_vocab = len(self.query_vocab_to_number)
        self.num_relation = len(self.kb_relation_to_number)
        self.num_operator = 2 * self.num_relation
        self.num_entity = len(self.kb_entity_to_number)

        self.matrix_db = self._db_to_matrix_db(self.facts)
        self.matrix_db_train = self.matrix_db
        self.matrix_db_test = self.matrix_db
        self.matrix_db_valid = self.matrix_db

        self.type_check = False
        self.domain_size = None
        self.use_extra_facts = False
        self.query_include_reverse = False
        self.share_db = False

        self.parser = self._create_parser()
        #self.query_for_rules = [list(q) for q in Counter([tuple(q) for (q, _, _) in self.test]).keys()]
        self.query_for_rules = [list(q) for q in set([tuple(q) for (q, _, _) in self.test + self.train])]

    def _numerical_encode(self, file_name):
        lines = [l.strip() for l in open(file_name, "r").readlines()]
        line_to_number = {line: i for i, line in enumerate(lines)}
        return line_to_number

    def _parse_examples(self, file_name):
        lines = [l.strip().split("\t") for l in open(file_name, "r").readlines()]
        triplets = [[[self.query_vocab_to_number[w] for w in l[1].split(",")],
                      self.kb_entity_to_number[l[0]],
                      self.kb_entity_to_number[l[2]],]
                    for l in lines]
        return triplets, len(triplets)

    def _parse_facts(self, file_name):
        lines = [l.strip().split("\t") for  l in open(file_name, "r").readlines()]
        facts = [[self.kb_relation_to_number[l[1]],
                  self.kb_entity_to_number[l[0]],
                  self.kb_entity_to_number[l[2]]]
                 for l in lines]
        return facts, len(facts)

    def _create_parser(self):
        parser = {"operator":{}}
        number_to_relation = {value: key for key, value
                                         in self.kb_relation_to_number.items()}
        number_to_query_vocab = {value: key for key, value
                                            in self.query_vocab_to_number.items()}

        parser["query"] = lambda ws: ",".join([number_to_query_vocab[w] for w in ws]) + " "

        d = {}
        for k, v in number_to_relation.items():
            d[k] = v
            d[k + self.num_relation] = "inv_" + v
        parser["operator"] = d

        return parser

    def is_true(self, q, h, t):
        if tuple(q + [h, t]) in self.all_exams:
            return True
        else:
            return False


# class DataHeadwise(Data):
#     def reset(self, batch_size):
#         self.batch_size = batch_size
#         self.train_start = 0
#         self.valid_start = 0
#         self.test_start = 0
#         self.num_batch_train = self.num_relation * 2
#         self.num_batch_valid = self.num_relation * 2
#         self.num_batch_test = self.num_relation * 2
#
#     def _next_batch(self, start, size, samples):
#         original_start = start
#         start = original_start // 2
#         use_reverse = original_start % 2
#         assert(start < size * 2)
#         this_batch_id = [
#             _i for _i, _sample in enumerate(samples)
#             if _sample[0] == start
#         ]
#         this_batch = [samples[_i] for _i in this_batch_id]
#         next_start = (original_start + 1) % (self.num_relation * 2)
#         if use_reverse == 1:
#             this_batch = self._get_reverse(this_batch)
#         return next_start, this_batch, this_batch_id


class DataHeadwise(Data):
    def reset(self, batch_size):
        self.batch_size = batch_size
        self.batches_train, self.num_batch_train = self.batching(
            self.train, batch_size)
        self.batches_valid, self.num_batch_valid = self.batching(
            self.valid, batch_size)
        self.batches_test, self.num_batch_test = self.batching(
            self.test, batch_size)
        self.train_start = (0, self.batches_train)
        self.valid_start = (0, self.batches_valid)
        self.test_start = (0, self.batches_test)

    def batching(self, samples, batch_size):
        head_batches = [[] for _ in range(self.num_relation)]
        for _i, _sample in enumerate(samples):
            head_batches[_sample[0]].append(_i)
        if batch_size != -1:
            head_batches = [
                one_head[_i*batch_size: (_i+1)*batch_size]
                for one_head in head_batches
                for _i in range((len(one_head)-1)//batch_size+1)
            ]
        return head_batches, len(head_batches) * 2

    def _next_batch(self, start, size, samples):
        start, batches = start
        batch_index = start // 2
        is_reverse = start % 2 == 1
        # # FIXME
        # batch_index, is_reverse = 0, False
        this_batch_id = batches[batch_index]
        this_batch = [samples[_i] for _i in this_batch_id]
        next_start = (start + 1) % (len(batches) * 2)
        if is_reverse:
            this_batch = self._get_reverse(this_batch)
        return (next_start, batches), this_batch, this_batch_id
