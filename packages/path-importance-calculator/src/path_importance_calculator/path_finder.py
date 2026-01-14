"""パス探索アルゴリズム"""

from collections import deque
from typing import NamedTuple


class Path(NamedTuple):
    """パス情報を保持するクラス"""

    nodes: list[int]  # パス上のノードのリスト [source, node1, node2, target]
    relations: list[int]  # パス上のリレーションのリスト [rel1, rel2, rel3]


class PathFinder:
    """KG上でのパス探索を行うクラス"""

    def __init__(
        self,
        forward_edges: dict[int, list[tuple[int, int]]],
        backward_edges: dict[int, list[tuple[int, int]]],
        max_hops: int = 3,
        ignore_relations: set[int] | None = None,
    ):
        """
        Args:
            forward_edges: head -> [(relation, tail), ...]
            backward_edges: tail -> [(relation, head), ...]
            max_hops: 最大ホップ数（デフォルト: 3）
            ignore_relations: 無視するリレーションIDのセット
        """
        self.forward_edges = forward_edges
        self.backward_edges = backward_edges
        self.max_hops = max_hops
        self.ignore_relations = ignore_relations or set()

    def find_all_paths(self, source: int, target: int) -> list[Path]:
        """
        sourceからtargetまでの全てのパスを探索する（最大max_hopsホップ）

        Args:
            source: 開始ノード
            target: 終了ノード

        Returns:
            list[Path]: 見つかった全パスのリスト
        """
        all_paths: list[Path] = []

        # BFSで探索
        # キュー内の各要素: (現在のノード, パス上のノード, パス上のリレーション)
        queue: deque[tuple[int, list[int], list[int]]] = deque()
        queue.append((source, [source], []))

        while queue:
            current_node, path_nodes, path_relations = queue.popleft()

            # 最大ホップ数に達したら終了
            if len(path_relations) >= self.max_hops:
                continue

            # 現在のノードから出ているエッジを探索
            neighbors = self.forward_edges.get(current_node, [])

            for relation, next_node in neighbors:
                # 無視するリレーションはスキップ
                if relation in self.ignore_relations:
                    continue
                
                # 既に訪問済みのノードはスキップ（サイクル回避）
                if next_node in path_nodes:
                    continue

                new_path_nodes = path_nodes + [next_node]
                new_path_relations = path_relations + [relation]

                # ターゲットに到達した場合
                if next_node == target:
                    all_paths.append(Path(new_path_nodes, new_path_relations))
                else:
                    # まだ最大ホップ数に達していない場合は探索を続ける
                    if len(new_path_relations) < self.max_hops:
                        queue.append((next_node, new_path_nodes, new_path_relations))

        return all_paths

    def find_paths_batch(
        self, user_item_pairs: list[tuple[int, int]], n_users: int
    ) -> dict[tuple[int, int], list[Path]]:
        """
        複数のユーザー・アイテムペアに対してパスを探索する

        Args:
            user_item_pairs: [(user_id, item_id), ...]のリスト
            n_users: ユーザー数（エンティティIDへの変換に使用）

        Returns:
            dict: {(user_id, item_id): [Path, ...]}
        """
        results: dict[tuple[int, int], list[Path]] = {}

        for user_id, item_id in user_item_pairs:
            # KGではユーザーIDはn_entitiesから始まる
            user_entity_id = n_users + user_id
            item_entity_id = item_id

            paths = self.find_all_paths(user_entity_id, item_entity_id)
            results[(user_id, item_id)] = paths

        return results
