"""Attention score calculation for KGAT edges."""

from typing import Dict, Tuple

import torch
import torch.nn as nn


class AttentionCalculator:
    """Calculates attention scores for edges in the knowledge graph."""

    def __init__(
        self,
        entity_user_embed: nn.Embedding,
        relation_embed: nn.Embedding,
        trans_M: nn.Parameter,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize the attention calculator.

        Args:
            entity_user_embed: Entity and user embedding layer
            relation_embed: Relation embedding layer
            trans_M: Transformation matrices for each relation
            device: Device to run computations on
        """
        self.entity_user_embed = entity_user_embed
        self.relation_embed = relation_embed
        self.trans_M = trans_M
        self.device = device

    @torch.no_grad()
    def calculate_edge_attention(self, h: int, t: int, r: int) -> float:
        """Calculate attention score for a single edge.

        Args:
            h: Head entity/user ID
            t: Tail entity/user ID
            r: Relation ID

        Returns:
            Attention score for the edge (h, r, t)
        """
        r_embed = self.relation_embed.weight[r]
        W_r = self.trans_M[r]

        h_embed = self.entity_user_embed.weight[h]
        t_embed = self.entity_user_embed.weight[t]

        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)

        v = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed))

        return v.item()

    @torch.no_grad()
    def calculate_batch_attention(
        self, h_list: torch.Tensor, t_list: torch.Tensor, r_idx: int
    ) -> torch.Tensor:
        """Calculate attention scores for a batch of edges with the same relation.

        Args:
            h_list: Head entity/user IDs
            t_list: Tail entity/user IDs
            r_idx: Relation ID

        Returns:
            Attention scores for the edges
        """
        r_embed = self.relation_embed.weight[r_idx]
        W_r = self.trans_M[r_idx]

        h_embed = self.entity_user_embed.weight[h_list]
        t_embed = self.entity_user_embed.weight[t_list]

        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)

        return v_list

    @torch.no_grad()
    def precompute_all_attention_scores(
        self,
        kg_data: Dict[int, list[Tuple[int, int]]],
    ) -> Dict[Tuple[int, int, int], float]:
        """Precompute attention scores for all edges in the knowledge graph.

        Args:
            kg_data: Dictionary mapping head -> [(tail, relation), ...]

        Returns:
            Dictionary mapping (head, relation, tail) -> attention_score
        """
        attention_scores = {}

        for head, tail_rel_list in kg_data.items():
            for tail, relation in tail_rel_list:
                score = self.calculate_edge_attention(head, tail, relation)
                attention_scores[(head, relation, tail)] = score

        return attention_scores

    @torch.no_grad()
    def precompute_attention_scores_by_relation(
        self,
        kg_data_by_relation: Dict[int, list[Tuple[int, int]]],
    ) -> Dict[Tuple[int, int, int], float]:
        """Precompute attention scores grouped by relation for efficiency.

        Args:
            kg_data_by_relation: Dictionary mapping relation -> [(head, tail), ...]

        Returns:
            Dictionary mapping (head, relation, tail) -> attention_score
        """
        attention_scores = {}

        for relation, head_tail_list in kg_data_by_relation.items():
            if not head_tail_list:
                continue

            heads = torch.LongTensor([h for h, _ in head_tail_list]).to(self.device)
            tails = torch.LongTensor([t for _, t in head_tail_list]).to(self.device)

            scores = self.calculate_batch_attention(heads, tails, relation)

            for (h, t), score in zip(head_tail_list, scores):
                attention_scores[(h, relation, t)] = score.item()

        return attention_scores
