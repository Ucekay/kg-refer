import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from recbole.data.dataset.kg_dataset import KnowledgeBasedDataset
from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import (
    xavier_uniform_initialization,
)
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from scipy.sparse import coo_matrix


class LightKG(KnowledgeRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset: KnowledgeBasedDataset):
        super(LightKG, self).__init__(config, dataset)

        self.embedding_size = config["embedding_size"]
        self.layer = config["layer"]
        self.mess_dropout_rate = config["mess_dropout_rate"]
        self.cos_loss = bool(config["cos_loss"])  # Convert 0/1 to boolean
        self.beta_i = config["item_loss"]
        self.beta_u = config["user_loss"]
        self.temperature = 0.6
        self.fix_relation_weights = config["fix_relation_weights"]

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

        self.Similarity_matrix = self.get_similarity_matrix()

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
        item_degree_inter = torch.bincount(inter_item, minlength=n_items).to(
            torch.float32
        )

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

        degree_vec = torch.cat(
            (user_degree, item_degree_total, entity_degree[n_items:])
        )

        # Normalize by 1/sqrt(degree); set 0-degree nodes to 0 (avoid inf)
        inv_sqrt = torch.rsqrt(degree_vec)
        inv_sqrt = torch.where(
            torch.isfinite(inv_sqrt), inv_sqrt, torch.zeros_like(inv_sqrt)
        )

        return inv_sqrt.to(self.device)

    def get_similarity_matrix(self):
        """
        Compute the similarity matrix for contrastive learning.
        The similarity matrix captures relationships between users and items in the CKG.

        Returns:
            A sparse tensor representing the similarity matrix.
        """
        # Extract and filter CKG indices to user-item subgraph
        ckg_indices = self.CKG.coalesce().indices()

        n_ui = int(self.n_users + self.n_items)
        user_item_mask = (ckg_indices[0] < n_ui) & (ckg_indices[1] < n_ui)
        filtered_indices = ckg_indices[:, user_item_mask]

        # Build sparse adjacency for user-item subgraph
        interaction_matrix = torch.sparse_coo_tensor(
            filtered_indices,
            torch.ones(
                filtered_indices.shape[1], device=self.device, dtype=torch.float16
            ),
            size=(n_ui, n_ui),
        )

        # Compute co-occurrence: A^T * A (transpose for proper similarity)
        sim = torch.sparse.mm(interaction_matrix.t(), interaction_matrix)

        # Add direct connections
        sim = (sim + interaction_matrix).coalesce()

        # Normalize by degree product: -values * (1/sqrt(d_i)) * (1/sqrt(d_j))
        indices = sim.indices()
        values = sim.values()

        degree_i = self.Degree[indices[0]]
        degree_j = self.Degree[indices[1]]
        normalized_values = -values * degree_i * degree_j

        return torch.sparse_coo_tensor(
            indices,
            normalized_values,
            size=(n_ui, n_ui),
            device=self.device,
        )

    def get_normal_matrix(self):
        """
        Compute the normalized adjacency matrix for message passing in the graph.
        The normalization is done using the degree to scale the relation embeddings.

        Returns:
            indices_A (torch.Tensor): Indices of the normalized adjacency matrix.
            value_A (torch.Tensor): Calues of rhe normalized adjacency matrix.
        """
        # Extract the values (relation embeddings) from the Collaborative Knowledge Graph (CKG)
        relation_values = self.CKG._values()

        # Get the indices of the edges in the CKG
        edge_indices = self.CKG._indices()

        # Retrieve the relation embeddings and flatten them into a 1D tensor
        # If fix_relation_weights is True, set all relation weights to 1 (equivalent to LightGCN)
        if self.fix_relation_weights:
            relation_embeddings = torch.ones_like(relation_values, dtype=torch.float32, device=self.device)
        else:
            relation_embeddings = self.relation_embedding(relation_values).view(-1)

        # Compute the degree normalization factor for each node
        # Degree normalization is applied to both nodes connected by each edge
        degree_normalization = (
            self.Degree[edge_indices[0]] * self.Degree[edge_indices[1]]
        )

        # Apply degree normalization to the relation embeddings
        normalized_values = relation_embeddings * degree_normalization

        return edge_indices, normalized_values

    def get_all_embeddings(self):
        """
        Retrieve and concatenate the embeddings for users and entities.
        Applies dropout to embeddings duaring training if dropout rate is greated than 0.

        Returns:
            torch.Tensor: Concatenated embeddings of users and entities.
        """
        # Retrieve user and entity embeddings from the embedding layers
        user_embeddings = self.user_embedding.weight
        entity_embeddings = self.entity_embedding.weight

        # Apply dropout to embeddings during training (if dropout rate > 0 and not in test mode)
        if self.mess_dropout_rate > 0.0 and not self.test:
            entity_embeddings = self.mess_dropout(entity_embeddings)
            user_embeddings = self.mess_dropout(user_embeddings)

        # Concatenate user and entity embeddings along the first dimention
        all_embeddings = torch.cat([user_embeddings, entity_embeddings], dim=0)

        return all_embeddings

    def forward(self):
        """
        Perform the forward pass of the LightKG model.
        This method propagates user and entity embeddings through multiple GNN layers and aggregates the results to produce final embeddings.

        Returns:
            user_embeddings: Embeddings for users after GNN.
            entity_embeddings: Embeddings for entities after GNN.
        """

        # Step 1: Get initial all embeddings (user and entity embeddings)
        initial_embeddings = self.get_all_embeddings()

        # Step 2: Store embeddings for each layer
        layer_embeddings = [initial_embeddings]

        # Step 3: Get the normalized adjacency matrix for message passing
        adjacency_indices, adjacency_values = self.get_normal_matrix()

        # Step 4: Perform message passing through each GCN layer
        for layer_idx in range(self.layer):
            # Propagate embeddings using sparse matrix multiplication
            initial_embeddings = torch_sparse.spmm(
                adjacency_indices,
                adjacency_values,
                self.n_entities + self.n_users,
                self.n_entities + self.n_users,
                initial_embeddings,
            )
            # Store the embeddings for current layer
            layer_embeddings.append(initial_embeddings)

        # Step 5: Aggregate embeddings from all layers
        # Stock embeddings from all layers and compute their mean
        aggregated_embeddings = torch.stack(layer_embeddings, dim=1).mean(dim=1)

        # Step 6: Split aggregated embeddings into user and entity embeddings
        user_embeddings, entity_embeddings = torch.split(
            aggregated_embeddings, [self.n_users, self.n_entities]
        )

        return user_embeddings, entity_embeddings

    def _get_rec_embedding(self, user, pos_item, neg_item):
        """
        Retrieve 0-layer embeddings for users, positive items, and negative items.
        """
        user_e = self.user_embedding(user)
        pos_item_e = self.entity_embedding(pos_item)
        neg_item_e = self.entity_embedding(neg_item)
        return user_e, pos_item_e, neg_item_e

    def get_user_similarity(self, node: torch.Tensor):
        """
        Compute the similarity matrix for a given set of user nodes.

        Args:
            node (torch.Tensor): Indices of the user nodes.

        Returns:
            torch.Tensor: Dense similarity matrix (|node| × |node|) with bias +1.
        """
        # Extract node×node submatrix from sparse similarity matrix
        sim_submatrix = (
            self.Similarity_matrix.index_select(0, node)
            .index_select(1, node)
            .to_dense()
        )
        return 1 + sim_submatrix

    def get_user_loss(self, node: torch.Tensor, embedding: torch.Tensor):
        """
        Compute the contrastive loss for users based on their embeddings and similarities.

        Args:
            node (torch.Tensor): Indices of the user nodes (batch_size,).
            embedding (torch.Tensor): Embeddings of the user nodes (batch_size, dim).

        Returns:
            torch.Tensor: The computed user contrastive loss (scalar).
        """
        # Get similarity and normalize embeddings
        sim = self.get_user_similarity(node)
        normalized_emb = F.normalize(embedding, p=2, dim=1)

        # Compute degree-based mask: 1 - degree_i * degree_j
        degree = self.Degree[node]
        degree_product = torch.outer(degree, degree)
        mask = 1 - degree_product

        # Contrastive loss: sum(mask * exp((emb · emb^T) * sim / temp))
        similarity_score = (normalized_emb @ normalized_emb.T) * sim / self.temperature
        loss = torch.sum(mask * torch.exp(similarity_score))

        return loss

    def get_item_similarity(self, pos_item: torch.Tensor, neg_item: torch.Tensor):
        """
        Compute the similarity matrix between positive and negative items.
        Args:
            pos_item : Indices of the positive item.
            neg_item : Indices of the negative item.
        Returns:
            torch.Tensor: A dense similarity matrix between positive and negative items.
        """
        # Extract pos_item×neg_item submatrix from sparse similarity matrix
        sim_submatrix = (
            self.Similarity_matrix.index_select(0, pos_item)
            .index_select(1, neg_item)
            .to_dense()
        )
        return 1 + sim_submatrix

    def get_item_loss(
        self,
        pos_item: torch.Tensor,
        neg_item: torch.Tensor,
        pos_e: torch.Tensor,
        neg_e: torch.Tensor,
    ):
        """
        Compute the contrastive loss for items based on their embeddings and similarities.

        Args:
            pos_item: Indices of the positive items.
            neg_item: Indices of the negative items.
            pos_e: Embeddings of the positive items.
            neg_e: Embeddings of the negative items.
        Returns:
            torch.Tensor: The computed item contrastive loss.
        """
        # Get similarity and normalize embeddings
        sim = self.get_item_similarity(pos_item, neg_item)
        normalized_pos = F.normalize(pos_e, p=2, dim=1)
        normalized_neg = F.normalize(neg_e, p=2, dim=1)

        # Compute degree-based mask: 1 - degree_pos * degree_neg
        pos_degree = self.Degree[pos_item]
        neg_degree = self.Degree[neg_item]
        degree_product = torch.outer(pos_degree, neg_degree)
        mask = 1 - degree_product

        # Contrastive loss: sum(mask * exp((pos · neg^T) * sim / temp))
        similarity_score = (normalized_pos @ normalized_neg.T) * sim / self.temperature
        loss = torch.sum(mask * torch.exp(similarity_score))

        return loss

    def calculate_loss(self, interaction):
        """
        Calculate the total loss for training, including BPR loss and contrastive loss (if enabled).

        Args:
            interaction: RecBole Interaction object containing user, positive item, and negative item IDs.

        Returns:
            torch.Tensor: The total loss value.
        """
        # Set test mode to False (training mode)
        self.test = False

        # Extract user, positive item, and negative item IDs from interaction
        user_ids = interaction[self.USER_ID]
        pos_item_ids = interaction[self.ITEM_ID]
        neg_item_ids = interaction[self.NEG_ITEM_ID]

        # Forward pass to get user and item embeddings
        user_embeddings, item_embeddings = self.forward()
        self.all_user_embeddings, self.all_entity_embeddings = (
            user_embeddings,
            item_embeddings,
        )

        # Get embeddings for users, positive items, and negative items
        user_emb = self.all_user_embeddings[user_ids]
        pos_item_emb = self.all_entity_embeddings[pos_item_ids]
        neg_item_emb = self.all_entity_embeddings[neg_item_ids]

        # Compute positive and negative scores for BPR loss
        pos_scores = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_scores = torch.mul(user_emb, neg_item_emb).sum(dim=1)

        # Compute Bayesian Personalized Ranking (BPR) loss
        bpr_loss = self.mf_loss(pos_scores, neg_scores)

        # Get embeddings for contrastive loss
        user_emb_rec, pos_item_emb_rec, neg_item_emb_rec = self._get_rec_embedding(
            user_ids, pos_item_ids, neg_item_ids
        )

        # Initialize contrastive loss to 0
        contrastive_loss = 0

        # Compute contrastive loss
        if self.cos_loss:
            user_contrastive_loss = self.get_user_loss(user_ids, user_emb_rec)

            # Compute item contrastive loss
            item_contrastive_loss = self.get_item_loss(
                pos_item_ids + self.n_users,
                neg_item_ids + self.n_users,
                pos_item_emb_rec,
                neg_item_emb_rec,
            )

            # Weighted sum of user and item contrastive losses
            contrastive_loss = (
                self.beta_u * user_contrastive_loss
                + self.beta_i * item_contrastive_loss
            )

        # Total loss is the sum of BPR loss and contrastive loss
        total_loss = bpr_loss + contrastive_loss

        return total_loss

    def predict(self, interaction):
        """
        Predict the interaction score between a user and an item.

        Args:
            interaction: RecBole Interaction object containing user and item IDs.

        Returns:
            torch.Tensor: Predicted scores for the user-item pairs.
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        if not self.test:
            self.test = True
            self.all_user_embeddings, self.all_entity_embeddings = self.forward()

        user_lightgcn_embeddings, item_lightgcn_embeddings = (
            self.all_user_embeddings,
            self.all_entity_embeddings,
        )

        scores = torch.mul(
            user_lightgcn_embeddings[user], item_lightgcn_embeddings[item]
        ).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if not self.test:
            self.test = True
            self.all_user_embeddings, self.all_entity_embeddings = self.forward()

        u_embs = self.all_user_embeddings[user]
        i_embs = self.all_entity_embeddings[: self.n_items]
        scores = torch.matmul(u_embs, i_embs.transpose(0, 1))
        return scores.view(-1)
