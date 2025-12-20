"""
テストデータからユーザー・アイテムペアをランダムに選択し、
対象ユーザー・アイテム間の直接エッジを除いた状態で、
アイテムに接続するエンティティからの埋め込み伝播を無視した場合と
対象アイテムと接続する他のユーザーとのエッジをランダムに無視した場合のスコア変化を計算するスクリプト。
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# kgat パッケージを import できるように src をパスへ追加
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from kgat.config import KGATConfig  # noqa: E402
from kgat.core.kgat import KGAT  # noqa: E402
from kgat.data.dataloader import DataLoader  # noqa: E402
from kgat.utils.model_helper import load_model  # noqa: E402


def get_item_connected_entities(
    item_id: int, A_in: torch.Tensor, n_items: int, n_entities: int
) -> list[int]:
    """
    A_inからアイテムに接続する他のエンティティ（n_items以上のエンティティ）のリストを取得する。
    アイテムへの入力エッジまたは出力エッジを持つエンティティを返す（双方向）。
    """
    A_coo = A_in.cpu().coalesce()
    indices = A_coo.indices()
    
    entities_set = set()
    for i in range(indices.shape[1]):
        src = int(indices[0, i])
        dst = int(indices[1, i])
        
        # アイテムへの入力エッジで、送信元がアイテム以外のエンティティの場合
        if dst == item_id and n_items <= src < n_entities:
            entities_set.add(src)
        # アイテムからの出力エッジで、送信先がアイテム以外のエンティティの場合
        if src == item_id and n_items <= dst < n_entities:
            entities_set.add(dst)
    
    return list(entities_set)


def create_modified_A_in(
    original_A_in: torch.Tensor,
    edges_to_remove: list[tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    """
    A_inから特定のエッジを削除した新しいスパーステンソルを作成する。
    """
    # COO形式に変換
    A_coo = original_A_in.cpu().coalesce()
    indices = A_coo.indices()
    values = A_coo.values()
    
    # 削除するエッジのセットを作成
    edges_to_remove_set = set(edges_to_remove)
    
    # 削除しないエッジのみを保持
    keep_mask = []
    for i in range(indices.shape[1]):
        edge = (int(indices[0, i]), int(indices[1, i]))
        if edge not in edges_to_remove_set:
            keep_mask.append(i)
    
    if not keep_mask:
        # すべてのエッジが削除される場合、空のテンソルを作成
        size = original_A_in.shape
        empty_indices = torch.zeros((2, 0), dtype=torch.long)
        empty_values = torch.zeros(0)
        return torch.sparse_coo_tensor(empty_indices, empty_values, size).to(device)
    
    # 新しいインデックスと値を作成
    new_indices = indices[:, keep_mask]
    new_values = values[keep_mask]
    
    # 新しいスパーステンソルを作成
    new_A_in = torch.sparse_coo_tensor(
        new_indices, new_values, original_A_in.shape
    )
    
    # 正規化（softmaxを再適用）
    new_A_in = torch.sparse.softmax(new_A_in.cpu(), dim=1)
    
    return new_A_in.to(device)


def get_item_connected_users(
    item_id: int, A_in: torch.Tensor, n_entities: int, target_user_id: int
) -> list[tuple[int, float]]:
    """
    A_inからアイテムに接続する他のユーザー（target_user_idを除く）とその重みのリストを取得する。
    双方向のエッジを考慮する。
    """
    A_coo = A_in.cpu().coalesce()
    indices = A_coo.indices()
    values = A_coo.values()
    
    # ユーザーIDをキーとして重みを保持（双方向のうち一方の重みを保持）
    connected_users_dict = {}
    for i in range(indices.shape[1]):
        src = int(indices[0, i])
        dst = int(indices[1, i])
        weight = float(values[i])
        
        # アイテムへの入力エッジで、送信元がユーザー（n_entities以上）で、target_user_idでない場合
        if dst == item_id and src >= n_entities and src != target_user_id:
            if src not in connected_users_dict:
                connected_users_dict[src] = weight
        # アイテムからの出力エッジで、送信先がユーザー（n_entities以上）で、target_user_idでない場合
        if src == item_id and dst >= n_entities and dst != target_user_id:
            if dst not in connected_users_dict:
                connected_users_dict[dst] = weight
    
    return [(user_id, weight) for user_id, weight in connected_users_dict.items()]


def calculate_score_with_modified_A_in(
    model: KGAT,
    user_id: int,
    item_id: int,
    modified_A_in: torch.Tensor,
    device: torch.device,
) -> float:
    """
    修正されたA_inを使用してスコアを計算する。
    """
    # 元のA_inを保存
    original_A_in = model.A_in.data.clone()
    
    # 一時的にA_inを置き換え
    model.A_in.data = modified_A_in
    
    # スコアを計算
    user_tensor = torch.LongTensor([user_id]).to(device)
    item_tensor = torch.LongTensor([item_id]).to(device)
    
    with torch.no_grad():
        scores = model(user_tensor, item_tensor, mode="predict")
        score = float(scores[0, 0])
    
    # 元のA_inを復元
    model.A_in.data = original_A_in
    
    return score


def main():
    parser = argparse.ArgumentParser(
        description="対象ユーザー・アイテム間の直接エッジを除いた状態で、エンティティ→アイテムの伝播と他のユーザー→アイテムのエッジを無視した場合のスコア変化を分析"
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
        help="データセットディレクトリ",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default="yelp",
        help="データセット名",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("packages/kgat/output/entity_propagation_analysis.csv"),
        help="出力CSVパス",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="サンプリングするユーザー・アイテムペアの数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="ランダムシード",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="使用デバイス",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=64,
        help="埋め込み次元",
    )
    parser.add_argument(
        "--relation-dim",
        type=int,
        default=64,
        help="関係埋め込み次元",
    )
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    if not args.model_path.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {args.model_path}")
    
    dataset_dir = args.data_dir / args.data_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"データセットが見つかりません: {dataset_dir}")
    
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logging.info(f"device: {device}")
    
    # DataLoader準備
    config = KGATConfig()
    config.data_dir = str(args.data_dir)
    config.data_name = args.data_name
    config.use_pretrain = 0
    config.embed_dim = args.embed_dim
    config.relation_dim = args.relation_dim
    config.aggregation_type = "bi-interaction"
    config.laplacian_type = "random-walk"
    config.conv_dim_list = "[64,32,16]"
    config.predict = True
    config.seed = args.seed
    
    logger = logging.getLogger("analyze_entity_propagation")
    data = DataLoader(config, logger)
    logging.info(
        f"n_users={data.n_users}, n_items={data.n_items}, " +
        f"n_entities={data.n_entities}, n_relations={data.n_relations}"
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
    model.eval()
    logging.info(f"モデルをロード: {args.model_path.name}")
    
    # テストデータからランダムにサンプリング
    # すべてのテストペアを取得（後でエンティティ接続をチェック）
    rng = np.random.default_rng(args.seed)
    all_test_pairs = []
    for user_id, item_ids in data.test_user_dict.items():
        for item_id in item_ids:
            all_test_pairs.append((user_id, item_id))
    
    if len(all_test_pairs) < args.n_samples:
        logging.warning(
            f"エンティティに接続するアイテムを持つテストペアが{len(all_test_pairs)}件しかありません。"
        )
        sampled_pairs = all_test_pairs
    else:
        indices = rng.choice(len(all_test_pairs), size=args.n_samples, replace=False)
        sampled_pairs = [all_test_pairs[i] for i in indices]
    
    logging.info(f"{len(sampled_pairs)}件のユーザー・アイテムペアをサンプリング")
    
    # 結果を保存
    results = []
    
    for idx, (user_id, item_id) in enumerate(sampled_pairs, start=1):
        logging.info(f"\n処理中: {idx}/{len(sampled_pairs)} - user={user_id}, item={item_id}")
        
        # 対象ユーザー↔対象アイテム間のエッジを削除した状態をベースとする
        edges_to_remove_base = [
            (user_id, item_id),  # ユーザー → アイテム
            (item_id, user_id),  # アイテム → ユーザー（逆向き）
        ]
        base_A_in = create_modified_A_in(
            model.A_in.data, edges_to_remove_base, device
        )
        
        # ベースとなるスコアを計算（対象ユーザー・アイテム間の直接エッジなし）
        user_tensor = torch.LongTensor([user_id]).to(device)
        item_tensor = torch.LongTensor([item_id]).to(device)
        
        base_score = calculate_score_with_modified_A_in(
            model, user_id, item_id, base_A_in, device
        )
        
        logging.info(f"  ベーススコア（対象ユーザー・アイテム間エッジなし）: {base_score:.6f}")
        
        # アイテムに接続するエンティティを取得
        connected_entities = get_item_connected_entities(
            item_id, base_A_in, data.n_items, data.n_entities
        )
        n_connected_entities = len(connected_entities)
        logging.info(f"  接続エンティティ数: {n_connected_entities}")
        
        if n_connected_entities == 0:
            logging.warning("  接続エンティティがないのでスキップ")
            continue
        
        # ベース（対象ユーザー・アイテム間エッジ削除）+ エンティティ↔アイテムのエッジを削除（双方向）
        edges_to_remove_entities = edges_to_remove_base.copy()
        for entity_id in connected_entities:
            edges_to_remove_entities.append((entity_id, item_id))  # エンティティ → アイテム
            edges_to_remove_entities.append((item_id, entity_id))  # アイテム → エンティティ（逆向き）
        
        modified_A_in_entities = create_modified_A_in(
            model.A_in.data, edges_to_remove_entities, device
        )
        score_without_entities = calculate_score_with_modified_A_in(
            model, user_id, item_id, modified_A_in_entities, device
        )
        
        logging.info(f"  エンティティ伝播なしスコア: {score_without_entities:.6f}")
        
        # アイテムに接続する他のユーザーを取得
        item_connected_users = get_item_connected_users(
            item_id, base_A_in, data.n_entities, user_id
        )
        n_other_users = len(item_connected_users)
        logging.info(f"  アイテムに接続する他のユーザー数: {n_other_users}")
        
        # ベース（対象ユーザー・アイテム間エッジ削除）+ 同じ数だけランダムにユーザー↔アイテムのエッジを削除（双方向）
        if n_other_users >= n_connected_entities:
            selected_indices = rng.choice(
                n_other_users, size=n_connected_entities, replace=False
            )
            edges_to_remove_random = edges_to_remove_base.copy()
            for i in selected_indices:
                user_id_to_remove = item_connected_users[i][0]
                edges_to_remove_random.append((user_id_to_remove, item_id))  # ユーザー → アイテム
                edges_to_remove_random.append((item_id, user_id_to_remove))  # アイテム → ユーザー（逆向き）
        else:
            # アイテムに接続するユーザーが少ない場合はすべて削除
            edges_to_remove_random = edges_to_remove_base.copy()
            for user in item_connected_users:
                edges_to_remove_random.append((user[0], item_id))  # ユーザー → アイテム
                edges_to_remove_random.append((item_id, user[0]))  # アイテム → ユーザー（逆向き）
        
        modified_A_in_random = create_modified_A_in(
            model.A_in.data, edges_to_remove_random, device
        )
        score_without_random = calculate_score_with_modified_A_in(
            model, user_id, item_id, modified_A_in_random, device
        )
        
        logging.info(f"  他ユーザー→アイテムエッジ削除スコア: {score_without_random:.6f}")
        
        # 結果を記録
        results.append({
            "user_id": user_id,
            "item_id": item_id,
            "base_score": base_score,
            "n_connected_entities": n_connected_entities,
            "score_without_entities": score_without_entities,
            "score_diff_entities": base_score - score_without_entities,
            "score_diff_entities_pct": (
                (base_score - score_without_entities) / abs(base_score) * 100
                if base_score != 0 else 0
            ),
            "n_other_users_connected": n_other_users,
            "n_removed_random_edges": len(edges_to_remove_random) - 2,  # ベースの2エッジを除く
            "score_without_random": score_without_random,
            "score_diff_random": base_score - score_without_random,
            "score_diff_random_pct": (
                (base_score - score_without_random) / abs(base_score) * 100
                if base_score != 0 else 0
            ),
        })
    
    # CSVに保存
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        if results:
            fieldnames = [
                "user_id",
                "item_id",
                "base_score",
                "n_connected_entities",
                "score_without_entities",
                "score_diff_entities",
                "score_diff_entities_pct",
                "n_other_users_connected",
                "n_removed_random_edges",
                "score_without_random",
                "score_diff_random",
                "score_diff_random_pct",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    logging.info(f"\n結果を保存しました: {args.output_csv}")
    logging.info(f"処理件数: {len(results)}件")
    
    # 統計情報を表示
    if results:
        avg_entities = np.mean([r["n_connected_entities"] for r in results])
        avg_diff_entities = np.mean([r["score_diff_entities"] for r in results])
        avg_diff_entities_pct = np.mean([r["score_diff_entities_pct"] for r in results])
        avg_diff_random = np.mean([r["score_diff_random"] for r in results])
        avg_diff_random_pct = np.mean([r["score_diff_random_pct"] for r in results])
        
        logging.info("\n=== 統計情報 ===")
        logging.info("※ベーススコアは対象ユーザー・アイテム間の直接エッジを削除した状態")
        logging.info(f"平均接続エンティティ数: {avg_entities:.2f}")
        logging.info(f"エンティティ伝播なし時の平均スコア差: {avg_diff_entities:.6f} ({avg_diff_entities_pct:.2f}%)")
        logging.info(f"ランダム接続削除時の平均スコア差: {avg_diff_random:.6f} ({avg_diff_random_pct:.2f}%)")


if __name__ == "__main__":
    main()

