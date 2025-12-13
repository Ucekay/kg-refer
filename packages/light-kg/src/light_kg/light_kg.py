import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from recbole.data.dataset.kg_dataset import KnowledgeBasedDataset
from recbole.model.abstract_recommender import GeneralRecommender, KnowledgeRecommender
from recbole.model.init import (
    xavier_normal_initialization,
    xavier_uniform_initialization,
)
from recbole.model.layers import SparseDropout
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from scipy.sparse import coo_matrix


class LightKG(KnowledgeRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset: KnowledgeBasedDataset):
        super(LightKG, self).__init__(config, dataset)

        self.embedding_size = config["embedding_size"]
        self.layer = config["layer"]
        self.mess_dropout_rate = config["mess_dropout_rate"]
        self.cos_loss = config["cos_loss"]
        self.beta_i = config["item_loss"]
        self.beta_u = config["user_loss"]
        self.temperature = 0.6

        self.mess_dropout = nn.Dropout1d(self.mess_dropout_rate)

        inter_result = dataset.inter_matrix(form="coo").astype(np.float32)
        # Type assertion: form="coo" guarantees coo_matrix
        assert isinstance(inter_result, coo_matrix), (
            "inter must be coo_matrix when form='coo'"
        )
        self.inter: coo_matrix = inter_result

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(2 * self.n_relations - 1, 1)

        kg_graph_result = dataset.kg_graph(form="coo", value_field="relation_id")
        # Type assertion: form="coo" guarantees coo_matrix
        assert isinstance(kg_graph_result, coo_matrix), (
            "kg_graph must be coo_matrix when form='coo'"
        )
        self.kg_graph: coo_matrix = kg_graph_result

        self.CKG = self.get_ckg()

        self.Degree = self.get_degree_matrix()

        self.Similarity_matrix = self.get_Similarity_matrix()

        self.apply(xavier_uniform_initialization)

        self.mf_loss = BPRLoss()
        self.test = False

        self.all_user_embeddings, self.all_entity_embeddings = (
            self.user_embedding.weight.to(self.device),
            self.entity_embedding.weight.to(self.device),
        )
        self.mf_loss = BPRLoss()
        self.test = False

        self.all_user_embeddings, self.all_entity_embeddings = (
            self.user_embedding.weight.to(self.device),
            self.entity_embedding.weight.to(self.device),
        )

    def get_ckg(self):
        """
        Construct the Collaborative Knowledge Graph (CKG) by combining user_item interactions and the knowledge graph (KG).
        The CKG is a sparse tendor that represents relationships between users, items, and entities in the KG.

        Returns:
            torch.sparse.Tensor: A sparse tensor representing the CKG.
        """

        # Extract KG data: head entities, tail entities, and relations
        kg_head = self.kg_graph.row
        kg_tail = self.kg_graph.col
        kg_relation = self.kg_graph.data

        # Create reversed relations for bidirectional edges in the KG
        kg_relation_reversed = kg_relation + (self.n_relations - 2)

        # Combine original and reversed KG edges
        kg_head_bidirectional = np.concatenate((kg_head, kg_tail))
        kg_tail_bidirectional = np.concatenate((kg_tail, kg_head))
        kg_relation_bidirectional = np.concatenate((kg_relation, kg_relation_reversed))

        # Extract user-item interaction data
        inter_head = self.inter.row
        inter_tail = self.inter.col

        # Define relation IDs for user-item interactions
        inter_relation = np.array([(self.n_relations - 2) * 2 + 1] * len(inter_head))
        inter_relation_reversed = np.array(
            [(self.n_relations - 2) * 2 + 2] * len(inter_head)
        )

        # Combine all head entities (users, items, and KG entities)
        all_head = torch.tensor(
            np.concatenate(
                (
                    inter_head,
                    inter_tail + self.n_users,
                    kg_head_bidirectional + self.n_users,
                )
            )
        )

        # Combine all tail entities (items, users, and KG entities)
        all_tail = torch.tensor(
            np.concatenate(
                (
                    inter_tail + self.n_users,
                    inter_head,
                    kg_tail_bidirectional + self.n_users,
                )
            )
        )

        # Combine all relations (user-item, item-usern, and KG relations)
        all_relation = torch.tensor(
            np.concatenate(
                (
                    inter_relation,
                    inter_relation_reversed,
                    kg_relation_bidirectional,
                )
            )
        )

        # Define the size of the CKG sparse tensor
        ckg_size = torch.Size(
            [self.n_entities + self.n_users, self.n_entities + self.n_users]
        )

        # Create the CKG sparse tensor
        ckg_sparse_tensor = torch.sparse_coo_tensor(
            torch.stack([all_head, all_tail]),
            all_relation,
            size=ckg_size,
        )

        return ckg_sparse_tensor.to(self.device)

    def get_degree_matrix(self):
        """
        Compute the degree matrix for normalization.
        The degree matrix records the number of neighbors for each node in the graph.
        Returns:
            A tensor containing the degree of each node, normalized by 1/sqrt(degree).
        """
        # ---- 1) User-Item interaction degrees (bincount is faster + shape-stable) ----
        inter_user = torch.as_tensor(self.inter.row, dtype=torch.long)
        inter_item = torch.as_tensor(self.inter.col, dtype=torch.long)

        n_users = int(self.n_users)
        n_items = int(self.n_items)
        n_entities = int(self.n_entities)

        user_degree = torch.bincount(inter_user, minlength=n_users).to(torch.float32)
        item_degree_inter = torch.bincount(inter_item, minlength=n_items).to(torch.float32)

        # ---- 2) KG entity degrees (make edges bidirectional, de-duplicate, then count) ----
        kg_head = self.kg_graph.row
        kg_tail = self.kg_graph.col

        # Bidirectional directed edges; outgoing-degree == number of neighbors
        kg_head_bi = np.concatenate((kg_head, kg_tail))
        kg_tail_bi = np.concatenate((kg_tail, kg_head))

        # Remove duplicate directed edges with stable, vectorized unique
        edges = np.unique(np.stack((kg_head_bi, kg_tail_bi), axis=1), axis=0)
        head_idx = torch.from_numpy(edges[:, 0]).to(torch.long)

        entity_degree = torch.bincount(head_idx, minlength=n_entities).to(torch.float32)

        # ---- 3) Combine degrees into (n_users + n_entities,) and normalize ----
        item_degree_kg = entity_degree[:n_items]
        item_degree_total = item_degree_inter + item_degree_kg

        degree_vec = torch.cat((user_degree, item_degree_total, entity_degree[n_items:]))

        # Normalize by 1/sqrt(degree); set 0-degree nodes to 0 (avoid inf)
        inv_sqrt = torch.rsqrt(degree_vec)
        inv_sqrt = torch.where(torch.isfinite(inv_sqrt), inv_sqrt, torch.zeros_like(inv_sqrt))

        return inv_sqrt.to(self.device)

