"""A_inからリレーション重みを抽出するモジュール"""

import logging

import numpy as np
import torch
from scipy.sparse import coo_matrix


class RelationWeightExtractor:
    """A_inからリレーション重みを抽出するクラス"""

    def __init__(
        self,
        A_in: torch.Tensor,
        laplacian_dict: dict[int, coo_matrix],
        logger: logging.Logger,
    ):
        """
        Args:
            A_in: KGATのA_inテンソル（スパーステンソル）
            laplacian_dict: {relation_id: adjacency_matrix}の辞書
            logger: ロガー
        """
        self.A_in = A_in
        self.laplacian_dict = laplacian_dict
        self.logger = logger
        self._build_edge_weight_dict()

    def _build_edge_weight_dict(self) -> None:
        """
        A_inとlaplacian_dictから、各エッジの重みを抽出して辞書を構築する
        edge_weights[(head, relation, tail)] = weight
        """
        self.edge_weights: dict[tuple[int, int, int], float] = {}

        # A_inをCPUに移動してnumpyに変換
        if self.A_in.is_sparse:
            A_in_coo = self.A_in.coalesce().cpu()
            indices = A_in_coo.indices().numpy()
            values = A_in_coo.values().numpy()
        else:
            # 密行列の場合
            A_in_dense = self.A_in.cpu().numpy()
            # スパース形式に変換
            A_in_sparse = coo_matrix(A_in_dense)
            indices = np.array([A_in_sparse.row, A_in_sparse.col])
            values = A_in_sparse.data

        # A_inのエッジ情報を辞書に保存
        A_in_edges = {}
        for i in range(indices.shape[1]):
            head = int(indices[0, i])
            tail = int(indices[1, i])
            weight = float(values[i])
            A_in_edges[(head, tail)] = weight

        # 各リレーションの隣接行列を確認して、エッジの重みを設定
        for relation_id, adj_matrix in self.laplacian_dict.items():
            # adjacency matrixのエッジを取得
            row = adj_matrix.row
            col = adj_matrix.col

            for i in range(len(row)):
                head = int(row[i])
                tail = int(col[i])

                # A_inに対応するエッジがある場合、その重みを使用
                if (head, tail) in A_in_edges:
                    weight = A_in_edges[(head, tail)]
                    self.edge_weights[(head, relation_id, tail)] = weight

        self.logger.info(f"Extracted {len(self.edge_weights)} edge weights from A_in")

    def get_edge_weight(self, head: int, relation: int, tail: int) -> float:
        """
        指定されたエッジの重みを取得する

        Args:
            head: 先頭エンティティID
            relation: リレーションID
            tail: 末尾エンティティID

        Returns:
            float: エッジの重み（存在しない場合は0.0）
        """
        return self.edge_weights.get((head, relation, tail), 0.0)

    def calculate_path_importance(
        self, path_nodes: list[int], path_relations: list[int]
    ) -> float:
        """
        パスの重要度を計算する（各エッジの重みの積）

        Args:
            path_nodes: パス上のノード [node0, node1, node2, node3]
            path_relations: パス上のリレーション [rel1, rel2, rel3]

        Returns:
            float: パスの重要度スコア
        """
        if len(path_nodes) != len(path_relations) + 1:
            raise ValueError(
                f"Invalid path: {len(path_nodes)} nodes and {len(path_relations)} relations"
            )

        importance = 1.0
        for i, relation in enumerate(path_relations):
            head = path_nodes[i]
            tail = path_nodes[i + 1]
            weight = self.get_edge_weight(head, relation, tail)

            # 重みが0の場合、パス全体の重要度は0
            if weight == 0.0:
                return 0.0

            importance *= weight

        return importance
