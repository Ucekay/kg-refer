import collections
import logging
import os
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from numpy.typing import NDArray

from kgat.config import KGATConfig


class DataLoader:
    config: KGATConfig
    data_name: str
    use_pretrain: int
    pretrain_embedding_dir: str

    data_dir: str
    train_file: str
    test_file: str
    kg_file: str

    cf_train_data: tuple[NDArray[np.int32], NDArray[np.int32]]
    train_user_dict: dict[int, list[int]]
    cf_test_data: tuple[NDArray[np.int32], NDArray[np.int32]]
    test_user_dict: dict[int, list[int]]

    rng: np.random.Generator

    cf_batch_size: int
    kg_batch_size: int
    test_batch_size: int

    n_users: int
    n_items: int
    n_cf_train: int
    n_cf_test: int

    kg_data: pd.DataFrame

    n_relations: int
    n_entities: int
    n_users_entities: int

    train_kg_dict: collections.defaultdict[int, list[tuple[int, int]]]
    train_relation_dict: collections.defaultdict[int, list[tuple[int, int]]]

    laplacian_type: str

    def __init__(self, config: KGATConfig, logger: logging.Logger) -> None:
        self.config = config
        self.data_name = config.data_name
        self.use_pretrain = config.use_pretrain
        self.pretrain_embedding_dir = config.pretrain_embedding_dir

        self.data_dir = os.path.join(config.data_dir, self.data_name)
        self.train_file = os.path.join(self.data_dir, "train.txt")
        self.test_file = os.path.join(self.data_dir, "test.txt")
        self.kg_file = os.path.join(self.data_dir, "kg_final.txt")

        self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file)
        self.cf_test_data, self.test_user_dict = self.load_cf(self.test_file)
        self.rng = np.random.default_rng(seed=config.seed)
        self.statistic_cf()

        if self.use_pretrain == 1:
            self.load_pretrain_data()

        self.cf_batch_size = config.cf_batch_size
        self.kg_batch_size = config.kg_batch_size
        self.test_batch_size = config.test_batch_size

        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)
        self.print_info(logger)

        self.laplacian_type = config.laplacian_type
        self.create_adjacency_dict()
        self.create_laplacian_dict()

    def load_cf(self, filename: str):
        user_list: list[int] = []
        item_list: list[int] = []
        user_dict: dict[int, list[int]] = {}

        lines = open(filename, "r", encoding="utf-8").readlines()
        for line in lines:
            tmp = line.strip()
            inter = [int(i) for i in tmp.split()]

            if len(inter) > 1:
                user_id, item_ids = inter[0], inter[1:]
                item_ids = list(set(item_ids))

                for item_id in item_ids:
                    user_list.append(user_id)
                    item_list.append(item_id)
                user_dict[user_id] = item_ids

        user_array = np.array(user_list, dtype=np.int32)
        item_array = np.array(item_list, dtype=np.int32)
        return (user_array, item_array), user_dict

    def statistic_cf(self):
        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0])) + 1
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1])) + 1
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])

    def load_kg(self, filename: str):
        try:
            kg_data = pd.read_csv(
                filename,
                sep=" ",
                names=["h", "r", "t"],
                engine="python",
                dtype={"h": int, "r": int, "t": int},
            )
        except ValueError:
            # 型変換に失敗した場合は文字列として読み込んで変換
            kg_data = pd.read_csv(
                filename, sep=" ", names=["h", "r", "t"], engine="python"
            )
            kg_data = kg_data.drop_duplicates()

            # 後で変換
            kg_data["h"] = pd.to_numeric(kg_data["h"], errors="coerce")
            kg_data["r"] = pd.to_numeric(kg_data["r"], errors="coerce")
            kg_data["t"] = pd.to_numeric(kg_data["t"], errors="coerce")

            # NaNを含む行を削除
            kg_data = kg_data.dropna()

            kg_data["h"] = kg_data["h"].astype(int)
            kg_data["r"] = kg_data["r"].astype(int)
            kg_data["t"] = kg_data["t"].astype(int)

        kg_data = kg_data.drop_duplicates()
        kg_data = kg_data.drop_duplicates()

        return kg_data

    def sample_pos_items_for_u(self, user_dict, user_id: int, n_sample_pos_items):
        pos_items = user_dict[user_id]
        n_pos_items = len(pos_items)
        sample_pos_items = []
        while True:
            if len(sample_pos_items) == n_sample_pos_items:
                break

            pos_item_idx = self.rng.integers(low=0, high=n_pos_items)
            pos_item_id = pos_items[pos_item_idx]
            if pos_item_id not in sample_pos_items:
                sample_pos_items.append(pos_item_id)
        return sample_pos_items

    def sample_neg_items_for_u(self, user_dict, user_id: int, n_sample_neg_items):
        neg_items = user_dict[user_id]

        sampled_neg_items = []
        while True:
            if len(sampled_neg_items) == n_sample_neg_items:
                break

            neg_item_id = self.rng.integers(low=0, high=self.n_items)
            if neg_item_id not in neg_items and neg_item_id not in sampled_neg_items:
                sampled_neg_items.append(neg_item_id)
        return sampled_neg_items

    def generate_cf_batch(self, user_dict, batch_size):
        exist_users = list(user_dict.keys())
        if batch_size <= len(exist_users):
            batch_user = random.sample(exist_users, batch_size)

        else:
            batch_user = [random.choice(exist_users) for _ in range(batch_size)]

        batch_pos_item = []
        batch_neg_item = []
        for u in batch_user:
            batch_pos_item += self.sample_pos_items_for_u(user_dict, u, 1)
            batch_neg_item += self.sample_neg_items_for_u(user_dict, u, 1)

        batch_user = torch.LongTensor(batch_user)
        batch_pos_item = torch.LongTensor(batch_pos_item)
        batch_neg_item = torch.LongTensor(batch_neg_item)
        return batch_user, batch_pos_item, batch_neg_item

    def sanple_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break
            pos_triple_idx = self.rng.integers(low=0, high=n_pos_triples)
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails

    def sample_neg_triples_for_h(
        self, kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx
    ):
        pos_triples = kg_dict[head]

        sampled_neg_tails = []
        while True:
            if len(sampled_neg_tails) == n_sample_neg_triples:
                break

            tail = self.rng.integers(low=0, high=highest_neg_idx)
            if (tail, relation) not in pos_triples and tail not in sampled_neg_tails:
                sampled_neg_tails.append(tail)
        return sampled_neg_tails

    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        exist_heads = list(kg_dict.keys())
        if batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, batch_size)
        else:
            batch_head = [random.choice(exist_heads) for _ in range(batch_size)]

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for h in batch_head:
            relation, pos_tail = self.sanple_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail

            neg_tail = self.sample_neg_triples_for_h(
                kg_dict, h, relation[0], 1, highest_neg_idx
            )
            batch_neg_tail += neg_tail
        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail

    def load_pretrain_data(self):
        pre_model = "mf"
        pretrain_path = (
            f"{self.pretrain_embedding_dir}/{self.data_name}/{pre_model}.npz"
        )
        pretrain_data = np.load(pretrain_path)
        self.user_pre_embed = pretrain_data["user_embed"]
        self.item_pre_embed = pretrain_data["item_embed"]

        assert self.user_pre_embed.shape[0] == self.n_users
        assert self.item_pre_embed.shape[0] == self.n_items
        assert self.user_pre_embed.shape[1] == self.config.embed_dim
        assert self.item_pre_embed.shape[1] == self.config.embed_dim

    def construct_data(self, kg_data: pd.DataFrame):
        # add inverse kg data

        n_relations: int = max(kg_data["r"]) + 1
        inversed_kg_data = kg_data.copy()
        inversed_kg_data = inversed_kg_data.rename({"h": "t", "t": "h"}, axis="columns")
        inversed_kg_data["r"] += n_relations
        kg_data = pd.concat(
            [kg_data, inversed_kg_data], axis=0, ignore_index=True, sort=False
        )

        kg_data["r"] += 2
        self.n_relations = max(kg_data["r"]) + 1
        self.n_entities = max(max(kg_data["h"]), max(kg_data["t"])) + 1
        self.n_users_entities = self.n_users + self.n_entities

        self.cf_train_data = (
            np.array(
                list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))
            ).astype(np.int32),
            self.cf_train_data[1].astype(np.int32),
        )
        self.cf_test_data = (
            np.array(
                list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))
            ).astype(np.int32),
            self.cf_test_data[1].astype(np.int32),
        )

        self.train_user_dict = {
            k + self.n_entities: v for k, v in self.train_user_dict.items()
        }
        self.test_user_dict = {
            k + self.n_entities: v for k, v in self.test_user_dict.items()
        }

        cf2kg_train_data = pd.DataFrame(
            np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=["h", "r", "t"]
        )
        cf2kg_train_data["h"] = self.cf_train_data[0]
        cf2kg_train_data["t"] = self.cf_train_data[1]

        inversed_cf2kg_train_data = pd.DataFrame(
            np.ones((self.n_cf_train, 3), dtype=np.int32), columns=["h", "r", "t"]
        )
        inversed_cf2kg_train_data["h"] = self.cf_train_data[1]
        inversed_cf2kg_train_data["t"] = self.cf_train_data[0]

        self.kg_train_data = pd.concat(
            [kg_data, cf2kg_train_data, inversed_cf2kg_train_data],
            ignore_index=True,
        )
        self.n_kg_train = len(self.kg_train_data)

        h_list: list[int] = []
        t_list: list[int] = []
        r_list: list[int] = []

        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)

        for idx, (h, r, t) in self.kg_train_data.iterrows():
            h: int = int(h)
            r: int = int(r)
            t: int = int(t)
            h_list.append(h)
            r_list.append(r)
            t_list.append(t)

            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)

    def convert_coo2_tensor(self, coo_matrix):
        values = coo_matrix.data
        indices = np.vstack((coo_matrix.row, coo_matrix.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo_matrix.shape
        return torch.sparse_coo_tensor(i, v, torch.Size(shape), dtype=torch.float32)

    def create_adjacency_dict(self) -> None:
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]
            vals = [1] * len(rows)
            adj = sp.coo_matrix(
                (vals, (rows, cols)),
                shape=(self.n_users_entities, self.n_users_entities),
            )
            self.adjacency_dict[r] = adj

    def create_laplacian_dict(self) -> None:
        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            dmp_inv = np.power(rowsum, -1.0).flatten()
            dmp_inv[np.isinf(dmp_inv)] = 0
            d_mat_inv = sp.diags(dmp_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == "symmetric":
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == "random-walk":
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.adjacency_dict.values())
        A_in = sp.coo_matrix(A_in)
        self.A_in = self.convert_coo2_tensor(A_in)

    def print_info(self, logger: logging.Logger) -> None:
        logger.info("n_users:           %d" % self.n_users)
        logger.info("n_items:           %d" % self.n_items)
        logger.info("n_entities:        %d" % self.n_entities)
        logger.info("n_users_entities:  %d" % self.n_users_entities)
        logger.info("n_relations:       %d" % self.n_relations)

        logger.info("n_h_list:          %d" % len(self.h_list))
        logger.info("n_t_list:          %d" % len(self.t_list))
        logger.info("n_r_list:          %d" % len(self.r_list))

        logger.info("n_cf_train:        %d" % self.n_cf_train)
        logger.info("n_cf_test:         %d" % self.n_cf_test)

        logger.info("n_kg_train:        %d" % self.n_kg_train)
