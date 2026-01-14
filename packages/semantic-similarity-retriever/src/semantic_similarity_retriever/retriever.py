"""意味的類似度に基づく類似ノード検索モジュール"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class SemanticSimilarityRetriever:
    """意味的類似度に基づいて類似ノードを検索するクラス"""

    def __init__(
        self,
        user_profiles: dict[int, str],
        item_profiles: dict[int, str],
        interactions_df: pd.DataFrame,
        text_encoder: str = "sentence-transformers/multi-qa-distilbert-cos-v1",
        pruning_score: float = 0.0,
        device: torch.device | None = None,
    ):
        """
        Args:
            user_profiles: ユーザーIDからプロファイルテキストへのマッピング
            item_profiles: アイテムIDからプロファイルテキストへのマッピング
            interactions_df: uid, iidカラムを持つインタラクションデータ
            text_encoder: 使用するテキストエンコーダーモデル名
            pruning_score: 類似度スコアの閾値（デフォルト: 0.0、この値以上のスコアを持つノードのみを返す）
            device: 計算に使用するデバイス
        """
        self.user_profiles = user_profiles
        self.item_profiles = item_profiles
        self.interactions_df = interactions_df
        self.pruning_score = pruning_score
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logging.info(f"テキストエンコーダーを初期化中: {text_encoder}")
        self.encoder = SentenceTransformer(text_encoder, device=str(self.device))
        
        self._compute_embeddings()
        self._build_interaction_graph()

    def _compute_embeddings(self):
        """ユーザーとアイテムの埋め込みを計算"""
        logging.info("ユーザーとアイテムの埋め込みを計算中...")
        
        # ユーザー埋め込み
        user_ids = sorted(self.user_profiles.keys())
        user_texts = [self.user_profiles[uid] for uid in user_ids]
        user_embeddings = self.encoder.encode(
            user_texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=32,
        )
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}
        self.user_embeddings = user_embeddings
        
        # アイテム埋め込み
        item_ids = sorted(self.item_profiles.keys())
        item_texts = [self.item_profiles[iid] for iid in item_ids]
        item_embeddings = self.encoder.encode(
            item_texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=32,
        )
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
        self.idx_to_item_id = {idx: iid for iid, idx in self.item_id_to_idx.items()}
        self.item_embeddings = item_embeddings
        
        logging.info(f"ユーザー埋め込み: {self.user_embeddings.shape}")
        logging.info(f"アイテム埋め込み: {self.item_embeddings.shape}")

    def _build_interaction_graph(self):
        """インタラクショングラフを構築（どのユーザーがどのアイテムを購入したか）"""
        logging.info("インタラクショングラフを構築中...")
        
        # user_id -> [item_ids]
        self.user_to_items = {}
        # item_id -> [user_ids]
        self.item_to_users = {}
        # グラフ構造を保持（2ホップサブグラフ抽出用）
        # node_id -> set of neighbor_node_ids
        # ユーザーIDはそのまま、アイテムIDは user_offset + item_id として格納
        self.user_offset = len(self.user_profiles)  # アイテムIDのオフセット
        self.graph = {}
        
        for _, row in self.interactions_df.iterrows():
            uid = int(row["uid"])
            iid = int(row["iid"])
            
            if uid not in self.user_to_items:
                self.user_to_items[uid] = []
            self.user_to_items[uid].append(iid)
            
            if iid not in self.item_to_users:
                self.item_to_users[iid] = []
            self.item_to_users[iid].append(uid)
            
            # グラフ構造を構築（双方向）
            user_node = uid
            item_node = self.user_offset + iid
            
            if user_node not in self.graph:
                self.graph[user_node] = set()
            if item_node not in self.graph:
                self.graph[item_node] = set()
            
            self.graph[user_node].add(item_node)
            self.graph[item_node].add(user_node)

    def _extract_k_hop_subgraph(self, user_id: int, item_id: int, k: int = 2) -> set[int]:
        """kホップのサブグラフを抽出（G-Referと同じ実装）
        
        Args:
            user_id: ユーザーID
            item_id: アイテムID
            k: ホップ数（デフォルト: 2）
        
        Returns:
            サブグラフ内のノードIDのセット（ユーザーIDとアイテムID（オフセット付き）を含む）
        """
        user_node = user_id
        item_node = self.user_offset + item_id
        
        # BFSでkホップ以内のノードを収集
        visited = set()
        queue = [(user_node, 0), (item_node, 0)]  # (node, hop_count)
        
        while queue:
            current_node, hop_count = queue.pop(0)
            
            if current_node in visited or hop_count > k:
                continue
            
            visited.add(current_node)
            
            if hop_count < k and current_node in self.graph:
                for neighbor in self.graph[current_node]:
                    if neighbor not in visited:
                        queue.append((neighbor, hop_count + 1))
        
        return visited

    def cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """コサイン類似度を計算

        Args:
            x: 1つ目の埋め込みベクトル
            y: 2つ目の埋め込みベクトル

        Returns:
            コサイン類似度
        """
        return F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0))

    def retrieve_similar_nodes(
        self,
        user_id: int,
        item_id: int,
        topk: int = 5,
    ) -> dict[str, Any]:
        """指定されたユーザーとアイテムに対して類似ノードを検索

        Args:
            user_id: ユーザーID
            item_id: アイテムID
            topk: 取得する類似ノードの最大数

        Returns:
            類似ノードの情報を含む辞書
        """
        if user_id not in self.user_id_to_idx or item_id not in self.item_id_to_idx:
            return {
                "user_id": user_id,
                "item_id": item_id,
                "similar_users": [],
                "similar_items": [],
            }
        
        user_idx = self.user_id_to_idx[user_id]
        item_idx = self.item_id_to_idx[item_id]
        
        user_emb = self.user_embeddings[user_idx]
        item_emb = self.item_embeddings[item_idx]
        
        # G-Referと同じく2ホップのサブグラフを抽出
        subgraph_nodes = self._extract_k_hop_subgraph(user_id, item_id, k=2)
        
        # G-Referと同じ方法：サブグラフ内のエッジから直接候補を抽出
        user_node = user_id
        item_node = self.user_offset + item_id
        
        # このアイテムを購入した他のユーザー（サブグラフ内のエッジから抽出）
        # item_nodeに接続されているユーザーノード（user_offset未満）で、サブグラフ内に含まれるもの
        users_who_bought_item = set()
        if item_node in self.graph:
            for neighbor in self.graph[item_node]:
                # neighborがユーザーノード（user_offset未満）で、サブグラフ内に含まれる場合
                if neighbor < self.user_offset and neighbor in subgraph_nodes:
                    users_who_bought_item.add(neighbor)
        users_who_bought_item.discard(user_node)
        
        similar_users = []
        for other_uid in users_who_bought_item:
            if other_uid not in self.user_id_to_idx:
                continue
            
            other_user_idx = self.user_id_to_idx[other_uid]
            other_user_emb = self.user_embeddings[other_user_idx]
            similarity = F.cosine_similarity(
                user_emb.unsqueeze(0), other_user_emb.unsqueeze(0)
            ).item()
            
            if similarity >= self.pruning_score:
                similar_users.append((other_uid, similarity))
        
        # スコアでソートしてtop-kを取得
        similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[:topk]
        
        # このユーザーが購入した他のアイテム（サブグラフ内のエッジから抽出）
        # user_nodeに接続されているアイテムノード（user_offset以上）で、サブグラフ内に含まれるもの
        items_bought_by_user = set()
        if user_node in self.graph:
            for neighbor in self.graph[user_node]:
                # neighborがアイテムノード（user_offset以上）で、サブグラフ内に含まれる場合
                if neighbor >= self.user_offset and neighbor in subgraph_nodes:
                    items_bought_by_user.add(neighbor - self.user_offset)
        items_bought_by_user.discard(item_id)
        
        similar_items = []
        for other_iid in items_bought_by_user:
            if other_iid not in self.item_id_to_idx:
                continue
            
            other_item_idx = self.item_id_to_idx[other_iid]
            other_item_emb = self.item_embeddings[other_item_idx]
            similarity = F.cosine_similarity(
                item_emb.unsqueeze(0), other_item_emb.unsqueeze(0)
            ).item()
            
            if similarity >= self.pruning_score:
                similar_items.append((other_iid, similarity))
        
        # スコアでソートしてtop-kを取得
        similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)[:topk]
        
        return {
            "user_id": user_id,
            "item_id": item_id,
            "similar_users": similar_users,
            "similar_items": similar_items,
        }

    def retrieve_batch(
        self,
        eval_interactions: pd.DataFrame,
        topk: int = 5,
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        """複数のユーザー・アイテムペアに対して類似ノードを検索

        Args:
            eval_interactions: uid, iidカラムを持つ評価用DataFrame
            topk: 取得する類似ノードの最大数
            show_progress: プログレスバーを表示するかどうか

        Returns:
            各ペアの類似ノード情報のリスト
        """
        results = []
        iterator = (
            tqdm(eval_interactions.iterrows(), total=len(eval_interactions), desc="類似ノード検索中")
            if show_progress
            else eval_interactions.iterrows()
        )

        user_count = 0
        item_count = 0

        for _, row in iterator:
            user_id = int(row["uid"])
            item_id = int(row["iid"])

            result = self.retrieve_similar_nodes(user_id, item_id, topk)
            results.append(result)

            user_count += len(result["similar_users"])
            item_count += len(result["similar_items"])

        avg_users = user_count / len(eval_interactions) if len(eval_interactions) > 0 else 0
        avg_items = item_count / len(eval_interactions) if len(eval_interactions) > 0 else 0

        logging.info(f"平均類似ユーザー数: {avg_users:.2f}")
        logging.info(f"平均類似アイテム数: {avg_items:.2f}")

        return results


def load_profiles(user_profile_path: str, item_profile_path: str) -> tuple[dict[int, str], dict[int, str]]:
    """ユーザーとアイテムのプロファイルをロード

    Args:
        user_profile_path: ユーザープロファイルJSONLファイルのパス
        item_profile_path: アイテムプロファイルJSONLファイルのパス

    Returns:
        (user_profiles, item_profiles)のタプル
    """
    # ユーザープロファイルをロード（JSON Lines形式）
    user_profiles = {}
    with open(user_profile_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            uid = int(data["uid"])
            # user summaryがJSON文字列の場合はパース
            user_summary = data.get("user summary", "")
            if isinstance(user_summary, str) and user_summary.startswith("{"):
                try:
                    summary_data = json.loads(user_summary)
                    user_summary = summary_data.get("summarization", "")
                except json.JSONDecodeError:
                    pass
            user_profiles[uid] = user_summary
    
    # アイテムプロファイルをロード（JSON Lines形式）
    item_profiles = {}
    with open(item_profile_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            iid = int(data["iid"])
            # business summaryまたはitem summaryがJSON文字列の場合はパース
            item_summary = data.get("business summary", data.get("item summary", ""))
            if isinstance(item_summary, str) and item_summary.startswith("{"):
                try:
                    summary_data = json.loads(item_summary)
                    item_summary = summary_data.get("summarization", "")
                except json.JSONDecodeError:
                    pass
            item_profiles[iid] = item_summary
    
    return user_profiles, item_profiles
