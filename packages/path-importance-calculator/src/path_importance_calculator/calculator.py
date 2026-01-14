"""パス重要度計算のメイン処理"""

import json
import logging
from pathlib import Path as FilePath
from typing import Any

import pandas as pd
from tqdm import tqdm

from path_importance_calculator.data_loader import (
    load_kg_as_dict,
    load_user_item_pairs,
)
from path_importance_calculator.model_loader import KGATModelLoader
from path_importance_calculator.name_mapper import NameMapper
from path_importance_calculator.path_finder import PathFinder
from path_importance_calculator.weight_extractor import RelationWeightExtractor


class PathImportanceCalculator:
    """パス重要度計算のメインクラス"""

    def __init__(
        self,
        model_loader: KGATModelLoader,
        max_hops: int = 3,
        min_hops: int = 1,
        ignore_relations: list[int] | None = None,
        entity_list_file: str | None = None,
        relation_list_file: str | None = None,
    ):
        """
        Args:
            model_loader: KGATモデルローダー
            max_hops: 最大ホップ数（デフォルト: 3）
            min_hops: 最小ホップ数（デフォルト: 1）
            ignore_relations: 無視するリレーションIDのリスト（KG元のremap_id）
            entity_list_file: entity_list.txtのパス
            relation_list_file: relation_list.txtのパス
        """
        self.logger = model_loader.logger
        self.max_hops = max_hops
        self.min_hops = min_hops

        # モデルとデータをロード
        self.logger.info("Loading KGAT model...")
        self.model, self.data_loader = model_loader.load_model()

        # 名前マッパーを初期化
        self.name_mapper = None
        if entity_list_file and relation_list_file:
            self.logger.info("Loading entity and relation names...")
            self.name_mapper = NameMapper(
                entity_list_file,
                relation_list_file,
                self.data_loader.n_items,
                self.data_loader.n_entities,
            )

        # リレーションIDを変換（+2 for forward, +2+n_relations for inverse）
        self.ignore_relations_set: set[int] = set()
        if ignore_relations:
            # KGの元のリレーション数を取得（construct_dataで+2される前）
            # train_relation_dictのリレーション数 / 2 で順方向の数
            n_original_relations = (
                max(self.data_loader.train_relation_dict.keys()) + 1
            ) // 2
            for rel_id in ignore_relations:
                # 順方向: rel_id + 2
                self.ignore_relations_set.add(rel_id + 2)
                # 逆方向: rel_id + 2 + n_original_relations
                self.ignore_relations_set.add(rel_id + 2 + n_original_relations)
            self.logger.info(f"Ignoring relations: {self.ignore_relations_set}")

        # KGデータをdata_loaderから取得（CFデータ統合済み）
        self.logger.info("Building graph from train_kg_dict...")
        self.forward_edges, self.backward_edges = self._build_edges_from_kg_dict()

        # パスファインダーを初期化
        self.path_finder = PathFinder(
            self.forward_edges,
            self.backward_edges,
            max_hops=max_hops,
            ignore_relations=self.ignore_relations_set,
        )

        # 重み抽出器を初期化
        self.logger.info("Initializing weight extractor...")
        self.weight_extractor = RelationWeightExtractor(
            self.model.A_in, self.data_loader.laplacian_dict, self.logger
        )

    def _build_edges_from_kg_dict(
        self,
    ) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
        """
        data_loader.train_kg_dictからエッジ辞書を構築する

        Returns:
            tuple: (forward_edges, backward_edges)
                - forward_edges[head] = [(relation, tail), ...]
                - backward_edges[tail] = [(relation, head), ...]
        """
        forward_edges: dict[int, list[tuple[int, int]]] = {}
        backward_edges: dict[int, list[tuple[int, int]]] = {}

        # train_kg_dict[head] = [(tail, relation), ...]
        for head, tail_rel_list in self.data_loader.train_kg_dict.items():
            for tail, relation in tail_rel_list:
                # Forward edge: head -> tail
                if head not in forward_edges:
                    forward_edges[head] = []
                forward_edges[head].append((relation, tail))

                # Backward edge: tail -> head
                if tail not in backward_edges:
                    backward_edges[tail] = []
                backward_edges[tail].append((relation, head))

        self.logger.info(
            f"Built graph with {len(forward_edges)} nodes and "
            f"{sum(len(v) for v in forward_edges.values())} edges"
        )

        return forward_edges, backward_edges

    def calculate_for_pairs(
        self, user_item_file: str, topk: int = 2, selection_mode: str = "topk"
    ) -> list[dict[str, Any]]:
        """
        ユーザー・アイテムペアに対してパスの重要度を計算する

        Args:
            user_item_file: ユーザー・アイテムペアのファイルパス
            topk: 各ペアごとに保存するパスの最大数（重要度スコア上位、デフォルト: 2）
            selection_mode: パス選択モード ('topk' または 'diverse')
                - 'topk': 重要度スコア上位のパスを取得（デフォルト）
                - 'diverse': interactionパス1つと、それ以外の関係が含まれるパス1つを取得

        Returns:
            list[dict]: 計算結果のリスト
        """
        results: list[dict[str, Any]] = []

        # ユーザー・アイテムペアをロード
        user_item_pairs = list(load_user_item_pairs(user_item_file))
        self.logger.info(f"Loaded {len(user_item_pairs)} user-item pairs")
        self.logger.info(f"Selection mode: {selection_mode}")

        # 各ペアに対して処理
        for user_id, item_id in tqdm(user_item_pairs, desc="Processing pairs"):
            # ユーザーIDをエンティティIDに変換
            user_entity_id = self.data_loader.n_entities + user_id
            item_entity_id = item_id

            # パスを探索
            paths = self.path_finder.find_all_paths(user_entity_id, item_entity_id)

            # 各パスの重要度を計算
            pair_results: list[dict[str, Any]] = []
            for path in paths:
                path_length = len(path.relations)
                
                # パス長のフィルタリング
                if path_length < self.min_hops or path_length > self.max_hops:
                    continue
                
                importance = self.weight_extractor.calculate_path_importance(
                    path.nodes, path.relations
                )
                
                # 各エッジの重みを取得
                edge_weights = []
                for i, relation in enumerate(path.relations):
                    head = path.nodes[i]
                    tail = path.nodes[i + 1]
                    weight = self.weight_extractor.get_edge_weight(head, relation, tail)
                    edge_weights.append(weight)

                result = {
                    "user_id": user_id,
                    "item_id": item_id,
                    "path_nodes": path.nodes,
                    "path_relations": path.relations,
                    "edge_weights": edge_weights,
                    "importance_score": importance,
                    "path_length": path_length,
                }
                
                # 名前情報を追加
                if self.name_mapper:
                    result = self.name_mapper.enrich_path(result)
                
                pair_results.append(result)

            # 重要度スコアでソート
            pair_results.sort(key=lambda x: x["importance_score"], reverse=True)

            # パス選択モードに応じてパスを選択
            if selection_mode == "diverse":
                selected_paths = self._select_diverse_paths(pair_results)
            else:  # 'topk'
                selected_paths = pair_results[:topk]

            results.extend(selected_paths)

        self.logger.info(f"Found {len(results)} paths in total ({selection_mode} mode)")
        return results

    def _select_diverse_paths(
        self, pair_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        interactionパス1つと、それ以外の関係が含まれるパス1つを選択する

        Args:
            pair_results: パス情報のリスト（重要度スコアでソート済み）

        Returns:
            list[dict]: 選択されたパスのリスト（最大2つ）
        """
        if not pair_results:
            return []

        selected_paths = []

        # interactionのみのパスを探す（user_interacts_with_item と item_interacted_by_user のみ）
        interaction_path = None
        for path in pair_results:
            if self._is_interaction_only_path(path):
                interaction_path = path
                break

        # それ以外の関係が含まれるパスを探す
        other_relation_path = None
        for path in pair_results:
            if not self._is_interaction_only_path(path):
                other_relation_path = path
                break

        # interactionパスを追加
        if interaction_path is not None:
            selected_paths.append(interaction_path)

        # それ以外の関係が含まれるパスを追加
        if other_relation_path is not None and other_relation_path != interaction_path:
            selected_paths.append(other_relation_path)

        # interactionパスが見つからない場合、上位のパスを追加
        if not selected_paths and pair_results:
            selected_paths.append(pair_results[0])

        return selected_paths

    def _is_interaction_only_path(self, path: dict[str, Any]) -> bool:
        """
        パスがinteraction関係のみで構成されているかを判定する

        Args:
            path: パス情報（path_relation_namesを含む）

        Returns:
            bool: interactionのみのパスならTrue、それ以外ならFalse
        """
        # path_relation_namesが存在しない場合はFalse（古い形式）
        if "path_relation_names" not in path:
            return False

        relation_names = path["path_relation_names"]
        interaction_relations = {"user_interacts_with_item", "item_interacted_by_user"}

        # すべてのリレーションがinteraction関係かどうか
        return all(rel in interaction_relations for rel in relation_names)

    def save_results(
        self, results: list[dict[str, Any]], output_file: str, format: str = "jsonl"
    ) -> None:
        """
        結果を保存する

        Args:
            results: 計算結果のリスト
            output_file: 出力ファイルパス
            format: 出力フォーマット ('jsonl', 'json' または 'csv')
        """
        output_path = FilePath(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            # JSONL形式で保存（1行1JSONオブジェクト、メモリ効率が良い）
            with open(output_path, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            self.logger.info(f"Results saved to {output_path} (JSONL format)")

        elif format == "json":
            # 後方互換性のためJSON配列形式もサポート
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Results saved to {output_path} (JSON format)")

        elif format == "csv":
            # CSVの場合、ノードとリレーションをカンマ区切りの文字列に変換
            csv_results = []
            for r in results:
                csv_results.append(
                    {
                        "user_id": r["user_id"],
                        "item_id": r["item_id"],
                        "path_nodes": ",".join(map(str, r["path_nodes"])),
                        "path_relations": ",".join(map(str, r["path_relations"])),
                        "importance_score": r["importance_score"],
                        "path_length": r["path_length"],
                    }
                )
            df = pd.DataFrame(csv_results)
            df.to_csv(output_path, index=False)
            self.logger.info(f"Results saved to {output_path} (CSV format)")

        else:
            raise ValueError(f"Unsupported format: {format}")
