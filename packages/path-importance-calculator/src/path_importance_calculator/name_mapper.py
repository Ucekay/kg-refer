"""エンティティとリレーションの名前マッピング"""

import csv
from pathlib import Path as FilePath


class NameMapper:
    """エンティティIDとリレーションIDを名前にマッピングするクラス"""

    def __init__(
        self,
        entity_list_file: str,
        relation_list_file: str,
        n_items: int,
        n_entities: int,
    ):
        """
        Args:
            entity_list_file: entity_list.txtのパス
            relation_list_file: relation_list.txtのパス
            n_items: アイテム数
            n_entities: エンティティ数（アイテム + KGエンティティ）
        """
        self.n_items = n_items
        self.n_entities = n_entities
        self.entity_names = self._load_entity_names(entity_list_file)
        self.relation_names = self._load_relation_names(relation_list_file)

    def _load_entity_names(self, entity_list_file: str) -> dict[int, str]:
        """エンティティ名を読み込む"""
        entity_names = {}
        
        with open(entity_list_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=" ")
            for row in reader:
                entity_id = int(row["remap_id"])
                # クォートを除去
                entity_name = row["entity_name"].strip('"')
                entity_names[entity_id] = entity_name
        
        return entity_names

    def _load_relation_names(self, relation_list_file: str) -> dict[int, str]:
        """リレーション名を読み込む（KGATの変換を考慮）"""
        relation_names = {}
        
        # 特殊リレーション（CFインタラクション）
        relation_names[0] = "user_interacts_with_item"
        relation_names[1] = "item_interacted_by_user"
        
        with open(relation_list_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=" ")
            for row in reader:
                remap_id = int(row["remap_id"])
                # クォートを除去
                relation_name = row["relation_name"].strip('"')
                
                # KGATでは元のリレーションに+2される
                # 順方向
                relation_names[remap_id + 2] = relation_name
                # 逆方向（+2+元のリレーション数）
                # 逆方向は後で計算
        
        # 逆方向のリレーション名を追加
        forward_relations = [k for k in relation_names.keys() if k >= 2]
        if forward_relations:
            max_forward = max(forward_relations)
            n_original_relations = max_forward - 1  # -2 + 2 = 元のリレーション数
            
            for rel_id, rel_name in list(relation_names.items()):
                if rel_id >= 2:
                    # 逆方向のID
                    inverse_id = rel_id + n_original_relations
                    relation_names[inverse_id] = f"inverse_{rel_name}"
        
        return relation_names

    def get_node_info(self, node_id: int) -> dict[str, str | int]:
        """
        ノードの情報を取得する

        Args:
            node_id: ノードID

        Returns:
            dict: ノードの情報
                - node_id: ノードID
                - node_type: "user" / "item" / "entity"
                - node_name: ノードの名前
        """
        if node_id >= self.n_entities:
            # ユーザー
            user_id = node_id - self.n_entities
            return {
                "node_id": node_id,
                "node_type": "user",
                "node_name": f"User_{user_id}",
                "original_id": user_id,
            }
        elif node_id < self.n_items:
            # アイテム
            return {
                "node_id": node_id,
                "node_type": "item",
                "node_name": f"Item_{node_id}",
                "original_id": node_id,
            }
        else:
            # KGエンティティ
            entity_name = self.entity_names.get(node_id, f"Entity_{node_id}")
            return {
                "node_id": node_id,
                "node_type": "entity",
                "node_name": entity_name,
                "original_id": node_id,
            }

    def get_relation_name(self, relation_id: int) -> str:
        """
        リレーション名を取得する

        Args:
            relation_id: リレーションID

        Returns:
            str: リレーション名
        """
        return self.relation_names.get(relation_id, f"Relation_{relation_id}")

    def enrich_path(self, path_result: dict) -> dict:
        """
        パス結果にノード名とリレーション名を追加する

        Args:
            path_result: 元のパス結果

        Returns:
            dict: エンリッチされたパス結果
        """
        enriched = path_result.copy()
        
        # ノード情報を追加
        node_infos = []
        for node_id in path_result["path_nodes"]:
            node_infos.append(self.get_node_info(node_id))
        enriched["path_node_infos"] = node_infos
        
        # リレーション名を追加
        relation_names = []
        for relation_id in path_result["path_relations"]:
            relation_names.append(self.get_relation_name(relation_id))
        enriched["path_relation_names"] = relation_names
        
        return enriched
