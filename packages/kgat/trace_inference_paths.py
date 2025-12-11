"""
学習済みKGATモデルを使って推薦を計算し、テストセットでヒットした
ユーザー・アイテムペアをランダムに1件サンプリングして、
A_in の注意重みを用いた推論パスを出力するユーティリティ。
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

# kgat パッケージを import できるように src をパスへ追加
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from kgat.config import KGATConfig  # noqa: E402
from kgat.core.kgat import KGAT  # noqa: E402
from kgat.data.dataloader import DataLoader  # noqa: E402
from kgat.utils.model_helper import load_model  # noqa: E402


@dataclass(frozen=True)
class PathSearchConfig:
    max_hops: int = 3
    top_neighbors_per_node: int = 64


def load_id_map(file_path: Path) -> Dict[int, str]:
    """`entity_list.txt` / `user_list.txt` を remap_id -> org_id の辞書で読み込む。"""
    id_map: Dict[int, str] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        next(f, None)  # ヘッダーをスキップ
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                remap_id = int(parts[1])
            except ValueError:
                continue
            id_map[remap_id] = parts[0]
    return id_map


def build_relation_name_lookup(data_dir: Path) -> Dict[int, str]:
    """
    最終的な relation id (user/item エッジ + KG + 逆向き) -> 関係名の辞書を作成する。
    """
    relation_path = data_dir / "relation_list.txt"
    kg_rel_names: Dict[int, str] = {}
    with open(relation_path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                rid = int(parts[1])
            except ValueError:
                continue
            kg_rel_names[rid] = parts[0]

    n_relations = (max(kg_rel_names) + 1) if kg_rel_names else 0
    lookup: Dict[int, str] = {
        0: "user->item",
        1: "item->user",
    }
    for rid, name in kg_rel_names.items():
        lookup[2 + rid] = name
        lookup[2 + n_relations + rid] = f"{name}_inv"
    return lookup


def build_edge_relation_lookup(train_relation_dict: Dict[int, List[Tuple[int, int]]]):
    """(h, t) -> relation id のリストを構築する。"""
    edge_rel = defaultdict(list)
    for r, ht_list in train_relation_dict.items():
        for h, t in ht_list:
            edge_rel[(int(h), int(t))].append(int(r))
    return edge_rel


def build_attention_graph(
    A_in: torch.Tensor, top_neighbors: int
) -> tuple[dict[int, list[tuple[int, float]]], dict[tuple[int, int], float]]:
    """
    A_in から隣接辞書とエッジ重み辞書を作る。
    top_neighbors を指定すると各ノードで上位のみ保持し探索を軽量化する。
    """
    coo = A_in.detach().cpu().coalesce()
    idx = coo.indices().numpy()
    vals = coo.values().numpy()

    adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
    weight_lookup: dict[tuple[int, int], float] = {}
    for h, t, w in zip(idx[0], idx[1], vals):
        h_i = int(h)
        t_i = int(t)
        w_f = float(w)
        adjacency[h_i].append((t_i, w_f))
        weight_lookup[(h_i, t_i)] = w_f

    for h, neighbors in adjacency.items():
        neighbors.sort(key=lambda x: x[1], reverse=True)
        adjacency[h] = neighbors[:top_neighbors] if top_neighbors else neighbors

    return adjacency, weight_lookup


def enumerate_paths_all(
    start: int,
    target: int,
    adjacency: dict[int, list[tuple[int, float]]],
    max_hops: int,
) -> list[list[int]]:
    """
    貪欲な深さ優先で、ホップ数上限までの単純経路をすべて列挙する。
    """
    paths: list[list[int]] = []

    def dfs(node: int, path: list[int], depth: int) -> None:
        if depth > max_hops:
            return
        if node == target and len(path) > 1:
            paths.append(path.copy())
            # 目的地に到達したらそこで終了（さらなる伸長はしない）
            return
        if depth == max_hops:
            return
        for neigh, _ in adjacency.get(node, []):
            if neigh in path:
                continue
            path.append(neigh)
            dfs(neigh, path, depth + 1)
            path.pop()

    dfs(start, [start], 0)
    return paths


def score_paths(
    paths: list[list[int]], weight_lookup: dict[tuple[int, int], float]
) -> list[tuple[list[int], float]]:
    """列挙した経路に対し、A_in の重み積でスコア付けして降順ソート。"""
    scored: list[tuple[list[int], float]] = []
    for p in paths:
        w_prod = 1.0
        valid = True
        for a, b in zip(p[:-1], p[1:]):
            w = weight_lookup.get((a, b))
            if w is None:
                valid = False
                break
            w_prod *= w
        if valid:
            scored.append((p, w_prod))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def format_node_label(
    node_id: int,
    n_entities: int,
    entity_map: Dict[int, str],
    user_map: Dict[int, str],
) -> str:
    if node_id >= n_entities:
        u_id = node_id - n_entities
        raw = user_map.get(u_id, "?")
        return f"user[{u_id}] ({raw})"
    raw = entity_map.get(node_id, "?")
    return f"entity[{node_id}] ({raw})"


def describe_path(
    path_nodes: Iterable[int],
    weight_lookup: dict[tuple[int, int], float],
    edge_rel: dict[tuple[int, int], list[int]],
    relation_names: dict[int, str],
    n_entities: int,
    entity_map: Dict[int, str],
    user_map: Dict[int, str],
) -> list[dict]:
    """パスを人間可読な形に整形する。"""
    nodes = list(path_nodes)
    steps: list[dict] = []
    for src, dst in zip(nodes[:-1], nodes[1:]):
        rel_ids = edge_rel.get((src, dst), [])
        rel_names = [relation_names.get(r, f"rel_{r}") for r in rel_ids] or ["unknown"]
        steps.append(
            {
                "from": format_node_label(src, n_entities, entity_map, user_map),
                "to": format_node_label(dst, n_entities, entity_map, user_map),
                "weight": weight_lookup.get((src, dst)),
                "relations": rel_names,
            }
        )
    return steps


def collect_hit_pairs(
    model: KGAT,
    data: DataLoader,
    device: torch.device,
    top_k: int,
    batch_size: int,
) -> list[dict]:
    """テストユーザーの Top-K 推薦でヒットした (user, item) の一覧を返す。"""
    model.eval()

    test_users = list(data.test_user_dict.keys())
    item_ids = torch.arange(data.n_items, device=device)
    hits: list[dict] = []

    for start in range(0, len(test_users), batch_size):
        batch_users = test_users[start : start + batch_size]
        batch_tensor = torch.LongTensor(batch_users).to(device)
        with torch.no_grad():
            scores = model(batch_tensor, item_ids, mode="predict")

        top_scores, top_items = torch.topk(scores, k=min(top_k, data.n_items), dim=1)
        top_scores = top_scores.cpu().numpy()
        top_items = top_items.cpu().numpy()

        for idx, u in enumerate(batch_users):
            test_items = set(data.test_user_dict[u])
            for rank, (item_id, score) in enumerate(
                zip(top_items[idx], top_scores[idx]), start=1
            ):
                if int(item_id) in test_items:
                    hits.append(
                        {
                            "user_id": int(u),
                            "item_id": int(item_id),
                            "rank": rank,
                            "score": float(score),
                        }
                    )
    return hits


def main():
    parser = argparse.ArgumentParser(
        description=(
            "学習済み KGAT モデルで推薦を行い、テストセットでヒットしたペアの推論パスを抽出"
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="学習済みモデル (.pth) へのパス",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("packages/kgat/datasets"),
        help="データセットディレクトリ（yelp2018 を含む親ディレクトリ）",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default="yelp2018",
        help="データセット名（サブディレクトリ名）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="ヒット判定に使う推薦上位件数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="ユーザーバッチサイズ（予測時）",
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=3,
        help="パス探索の最大ホップ数",
    )
    parser.add_argument(
        "--num-paths",
        type=int,
        default=5,
        help="表示する上位パス数（列挙した経路を重み順に上位から）",
    )
    parser.add_argument(
        "--top-neighbors",
        type=int,
        default=64,
        help="各ノードで保持する A_in 上位エッジ数（探索軽量化用）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="ランダムシード（ヒットサンプリング用）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="使用デバイス。未指定なら cuda が利用可なら cuda、それ以外は cpu。",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=64,
        help="モデルの埋め込み次元（学習時設定に合わせる）",
    )
    parser.add_argument(
        "--relation-dim",
        type=int,
        default=64,
        help="モデルの関係埋め込み次元（学習時設定に合わせる）",
    )
    parser.add_argument(
        "--aggregation-type",
        type=str,
        default="bi-interaction",
        choices=["gcn", "graphsage", "bi-interaction"],
        help="学習時と同じ集約方式",
    )
    parser.add_argument(
        "--laplacian-type",
        type=str,
        default="random-walk",
        choices=["random-walk", "symmetric"],
        help="学習時と同じ正規化ラプラシアン種別",
    )
    parser.add_argument(
        "--conv-dims",
        type=str,
        default="[64,32,16]",
        help="学習時 conv_dim_list（文字列そのまま渡す）",
    )
    parser.add_argument(
        "--use-pretrain",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="学習時の use_pretrain 値",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.model_path.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {args.model_path}")

    dataset_dir = args.data_dir / args.data_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"データセットディレクトリが見つかりません: {dataset_dir}")

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logging.info(f"device: {device}")

    # DataLoader 準備
    config = KGATConfig()
    config.data_dir = str(args.data_dir)
    config.data_name = args.data_name
    config.use_pretrain = args.use_pretrain
    config.embed_dim = args.embed_dim
    config.relation_dim = args.relation_dim
    config.aggregation_type = args.aggregation_type
    config.laplacian_type = args.laplacian_type
    config.conv_dim_list = args.conv_dims
    config.predict = True

    logger = logging.getLogger("trace_paths")
    data = DataLoader(config, logger)
    logging.info(
        f"n_users={data.n_users}, n_items={data.n_items}, n_entities={data.n_entities}, n_relations={data.n_relations}"
    )

    # モデル読み込み
    model = KGAT(
        config,
        data.n_users,
        data.n_entities,
        data.n_relations,
        data.A_in,
    )
    model = load_model(model, str(args.model_path))
    model.to(device)
    logging.info(f"モデルをロード: {args.model_path.name}")

    # 推薦ヒットを収集
    hits = collect_hit_pairs(
        model=model,
        data=data,
        device=device,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )
    if not hits:
        logging.warning("指定した Top-K ではテストセットのヒットが見つかりませんでした。")
        return

    rng = np.random.default_rng(args.seed)
    selected = hits[rng.integers(low=0, high=len(hits))]
    user_id = selected["user_id"]
    item_id = selected["item_id"]
    logging.info(
        f"ヒットを1件サンプリング: user_id={user_id}, item_id={item_id}, rank={selected['rank']}, score={selected['score']:.4f}"
    )

    # A_in から推論パス探索
    path_cfg = PathSearchConfig(
        max_hops=args.max_hops,
        top_neighbors_per_node=args.top_neighbors,
    )
    adjacency, weight_lookup = build_attention_graph(
        model.A_in, top_neighbors=path_cfg.top_neighbors_per_node
    )
    edge_rel = build_edge_relation_lookup(data.train_relation_dict)
    relation_names = build_relation_name_lookup(dataset_dir)
    entity_map = load_id_map(dataset_dir / "entity_list.txt")
    user_map = load_id_map(dataset_dir / "user_list.txt")

    paths_raw = enumerate_paths_all(
        start=user_id,
        target=item_id,
        adjacency=adjacency,
        max_hops=path_cfg.max_hops,
    )
    paths = score_paths(paths_raw, weight_lookup)

    if not paths:
        logging.warning(
            "指定条件内でユーザーからアイテムへのパスを発見できませんでした。パラメータを緩めて再実行してください。"
        )
        return

    print("\n=== 推論パス ===")
    print(f"ユーザー: {format_node_label(user_id, data.n_entities, entity_map, user_map)}")
    print(f"アイテム: {format_node_label(item_id, data.n_entities, entity_map, user_map)}")

    for path_idx, (path_nodes, path_score) in enumerate(
        paths[: args.num_paths], start=1
    ):
        steps = describe_path(
            path_nodes,
            weight_lookup,
            edge_rel,
            relation_names,
            data.n_entities,
            entity_map,
            user_map,
        )
        print(f"\n--- パス {path_idx} ---")
        print(
            f"パス長: {len(path_nodes) - 1} (最大 {args.max_hops} ホップ), 重み積: {path_score:.6f}"
        )
        for idx, step in enumerate(steps, start=1):
            rel_str = ", ".join(step["relations"])
            w = step["weight"]
            w_str = f"{w:.6f}" if w is not None else "N/A"
            print(f"{idx}. {step['from']} --{rel_str} (A_in={w_str})--> {step['to']}")


if __name__ == "__main__":
    main()

