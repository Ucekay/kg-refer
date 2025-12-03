"""Attention score calculation for KGAT edges."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class AttentionCalculator:
    """Calculates attention scores for edges in the knowledge graph.

    Supports two modes:
    1. Raw attention scores: Calculated from embeddings (not normalized)
    2. Normalized attention scores: Retrieved from A_in matrix (softmax normalized)
    """

    def __init__(
        self,
        entity_user_embed: nn.Embedding,
        relation_embed: nn.Embedding,
        trans_M: nn.Parameter,
        device: torch.device = torch.device("cpu"),
        A_in: Optional[torch.Tensor] = None,
    ):
        """Initialize the attention calculator.

        Args:
            entity_user_embed: Entity and user embedding layer
            relation_embed: Relation embedding layer
            trans_M: Transformation matrices for each relation
            device: Device to run computations on
            A_in: Optional normalized attention matrix (sparse tensor from trained model)
        """
        self.entity_user_embed = entity_user_embed
        self.relation_embed = relation_embed
        self.trans_M = trans_M
        self.device = device
        self._A_in = A_in.coalesce() if A_in is not None else None

    @property
    def has_A_in(self) -> bool:
        """Check if A_in matrix is available."""
        return self._A_in is not None

    def set_A_in(self, A_in: torch.Tensor) -> None:
        """Set the A_in matrix.

        Args:
            A_in: Normalized attention matrix (sparse tensor)
        """
        self._A_in = A_in.coalesce()

    @torch.no_grad()
    def get_normalized_attention(self, h: int, t: int) -> float:
        """Get normalized attention score from A_in for a single edge.

        Args:
            h: Head entity/user ID
            t: Tail entity/user ID

        Returns:
            Normalized attention score (0.0 if edge not found or A_in not set)

        Raises:
            ValueError: If A_in is not set
        """
        if self._A_in is None:
            raise ValueError("A_in matrix is not set. Use set_A_in() or pass A_in to constructor.")

        indices = self._A_in.indices()  # [2, num_edges]
        values = self._A_in.values()

        mask = (indices[0] == h) & (indices[1] == t)
        if mask.any():
            return values[mask].item()
        return 0.0

    @torch.no_grad()
    def get_normalized_attention_batch(
        self, h_list: List[int], t_list: List[int]
    ) -> List[float]:
        """Get normalized attention scores from A_in for multiple edges.

        Args:
            h_list: List of head entity/user IDs
            t_list: List of tail entity/user IDs

        Returns:
            List of normalized attention scores (0.0 for edges not found)

        Raises:
            ValueError: If A_in is not set
        """
        if self._A_in is None:
            raise ValueError("A_in matrix is not set. Use set_A_in() or pass A_in to constructor.")

        indices = self._A_in.indices()
        values = self._A_in.values()

        results = []
        for h, t in zip(h_list, t_list):
            mask = (indices[0] == h) & (indices[1] == t)
            if mask.any():
                results.append(values[mask].item())
            else:
                results.append(0.0)

        return results

    @torch.no_grad()
    def get_outgoing_edges(self, head_id: int) -> Dict[int, float]:
        """Get all outgoing edges and their normalized attention scores for a node.

        Args:
            head_id: Head entity/user ID

        Returns:
            Dictionary mapping tail_id -> normalized attention score

        Raises:
            ValueError: If A_in is not set
        """
        if self._A_in is None:
            raise ValueError("A_in matrix is not set. Use set_A_in() or pass A_in to constructor.")

        indices = self._A_in.indices()
        values = self._A_in.values()

        mask = indices[0] == head_id
        tail_ids = indices[1][mask].tolist()
        scores = values[mask].tolist()

        return dict(zip(tail_ids, scores))

    @torch.no_grad()
    def get_incoming_edges(self, tail_id: int) -> Dict[int, float]:
        """Get all incoming edges and their normalized attention scores for a node.

        Args:
            tail_id: Tail entity/user ID

        Returns:
            Dictionary mapping head_id -> normalized attention score

        Raises:
            ValueError: If A_in is not set
        """
        if self._A_in is None:
            raise ValueError("A_in matrix is not set. Use set_A_in() or pass A_in to constructor.")

        indices = self._A_in.indices()
        values = self._A_in.values()

        mask = indices[1] == tail_id
        head_ids = indices[0][mask].tolist()
        scores = values[mask].tolist()

        return dict(zip(head_ids, scores))

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
