"""
学習済みKGATモデルから、各エンティティにつながる各リレーションの重みの平均をA_inから計算するスクリプト
"""

import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.kgat.config import KGATConfig
from src.kgat.core.kgat import KGAT
from src.kgat.data.dataloader import DataLoader
from src.kgat.utils.model_helper import load_model


def setup_logging():
    """ロギングの設定"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def load_kg_data(kg_file_path: str) -> pd.DataFrame:
    """
    kg_final.txtからKGデータを読み込む

    Args:
        kg_file_path: kg_final.txtへのパス

    Returns:
        DataFrame with columns ['h', 'r', 't']
    """
    try:
        kg_data = pd.read_csv(
            kg_file_path,
            sep=" ",
            names=["h", "r", "t"],
            engine="python",
            dtype={"h": int, "r": int, "t": int},
            on_bad_lines="skip",
        )
    except ValueError:
        kg_data = pd.read_csv(
            kg_file_path,
            sep=" ",
            names=["h", "r", "t"],
            engine="python",
            on_bad_lines="skip",
        )
        kg_data = kg_data.drop_duplicates()
        kg_data["h"] = pd.to_numeric(kg_data["h"], errors="coerce")
        kg_data["r"] = pd.to_numeric(kg_data["r"], errors="coerce")
        kg_data["t"] = pd.to_numeric(kg_data["t"], errors="coerce")
        kg_data = kg_data.dropna()
        kg_data["h"] = kg_data["h"].astype(int)
        kg_data["r"] = kg_data["r"].astype(int)
        kg_data["t"] = kg_data["t"].astype(int)

    kg_data = kg_data.drop_duplicates()
    return kg_data


def get_attention_weights_from_A_in(
    A_in: torch.Tensor, head: int, tail: int
) -> float:
    """
    A_inから特定の(head, tail)ペアの重みを取得

    Args:
        A_in: スパーステンソル
        head: ヘッドエンティティID
        tail: テールエンティティID

    Returns:
        重み（存在しない場合は0.0）
    """
    # A_inをcoalesceしてから値を取得
    A_in_coalesced = A_in.coalesce()
    indices = A_in_coalesced.indices()
    values = A_in_coalesced.values()

    # (head, tail)に一致するインデックスを探す
    mask = (indices[0] == head) & (indices[1] == tail)
    if mask.any():
        return values[mask].item()
    return 0.0


def load_user_item_interactions(train_file_path: str, n_entities: int) -> pd.DataFrame:
    """
    train.txtからuser-item interactionを読み込む

    Args:
        train_file_path: train.txtへのパス
        n_entities: エンティティ数（ユーザーIDのオフセット）

    Returns:
        DataFrame with columns ['h', 'r', 't']
        - リレーション0: user -> item
        - リレーション1: item -> user
    """
    interactions = []
    
    with open(train_file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                user_id = int(parts[0])
                item_ids = [int(x) for x in parts[1:]]
                
                # ユーザーIDにn_entitiesを加算（KGATの内部表現）
                user_id_internal = user_id + n_entities
                
                for item_id in item_ids:
                    # リレーション0: user -> item
                    interactions.append({
                        "h": user_id_internal,
                        "r": 0,
                        "t": item_id
                    })
                    # リレーション1: item -> user
                    interactions.append({
                        "h": item_id,
                        "r": 1,
                        "t": user_id_internal
                    })
    
    return pd.DataFrame(interactions)


def calculate_relation_weight_averages(
    model: KGAT, kg_data: pd.DataFrame, train_file_path: str, n_entities: int, n_users: int
) -> dict[int, float]:
    """
    全体を通して各リレーションの重みの平均を計算

    Args:
        model: 学習済みKGATモデル
        kg_data: KGデータ（DataFrame with columns ['h', 'r', 't']）
        train_file_path: train.txtへのパス
        n_entities: エンティティ数
        n_users: ユーザー数

    Returns:
        relation_id -> average_weight の辞書
    """
    A_in = model.A_in
    logger = logging.getLogger(__name__)

    # user-item interactionを読み込む
    logger.info("user-item interactionを読み込み中...")
    user_item_data = load_user_item_interactions(train_file_path, n_entities)
    logger.info(f"user-item interaction数: {len(user_item_data)}")

    # KGデータとuser-item interactionを結合
    all_data = pd.concat([kg_data, user_item_data], ignore_index=True)
    logger.info(f"総エッジ数: {len(all_data)}")

    # リレーションごとに重みを集計（エンティティごとではなく全体）
    relation_weights = defaultdict(list)

    logger.info("A_inから重みを取得中...")
    total_edges = len(all_data)
    processed = 0

    for idx, row in all_data.iterrows():
        head = int(row["h"])
        relation = int(row["r"])
        tail = int(row["t"])

        # エンティティ（item）のみを対象（ユーザーは除外）
        if relation == 0:
            # リレーション0: user -> item (Interact)
            # headがuser、tailがitemなので、tail（item）側を対象
            if tail < n_entities:
                weight = get_attention_weights_from_A_in(A_in, head, tail)
                if weight > 0:
                    relation_weights[relation].append(weight)
        elif relation == 1:
            # リレーション1: item -> user (InteractedBy)
            # headがitem、tailがuserなので、head（item）側を対象
            if head < n_entities:
                weight = get_attention_weights_from_A_in(A_in, head, tail)
                if weight > 0:
                    relation_weights[relation].append(weight)
        else:
            # KGリレーション（2以上）
            # kg_final.txtのリレーションIDは0から始まるが、KGAT内部では+2される
            # ただし、user-item interactionを追加したので、そのまま使う
            # headがエンティティの場合のみ対象
            if head < n_entities:
                weight = get_attention_weights_from_A_in(A_in, head, tail)
                if weight > 0:
                    relation_weights[relation].append(weight)

        processed += 1
        if processed % 10000 == 0:
            logger.info(f"処理済み: {processed}/{total_edges}")

    # 各リレーションごとの平均を計算
    logger.info("平均を計算中...")
    result = {}
    for relation_id, weights in relation_weights.items():
        if len(weights) > 0:
            result[relation_id] = np.mean(weights)
            logger.info(f"リレーション {relation_id}: 平均重み = {result[relation_id]:.6f} (エッジ数: {len(weights)})")
        else:
            result[relation_id] = 0.0

    return result


def load_relation_names(relation_file_path: str) -> dict[int, str]:
    """
    relation_list.txtからリレーション名を読み込む

    Args:
        relation_file_path: relation_list.txtへのパス

    Returns:
        relation_id -> relation_name の辞書
    """
    relation_names = {}
    if os.path.exists(relation_file_path):
        # ファイル形式: "relation_name" remap_id (ヘッダー行あり)
        # ヘッダー行をスキップして読み込む
        df = pd.read_csv(
            relation_file_path,
            delimiter=" ",
            names=["relation_name", "remap_id"],
            skiprows=1,
        )
        for _, row in df.iterrows():
            relation_id = int(row["remap_id"])
            relation_name = str(row["relation_name"]).strip('"')
            relation_names[relation_id] = relation_name
    return relation_names


def load_entity_names(entity_file_path: str) -> dict[int, str]:
    """
    entity_list.txtからエンティティ名を読み込む

    Args:
        entity_file_path: entity_list.txtへのパス

    Returns:
        entity_id -> entity_name の辞書
    """
    entity_names = {}
    if os.path.exists(entity_file_path):
        # ファイル形式: "entity_name" remap_id (ヘッダー行あり)
        # ヘッダー行をスキップして読み込む
        df = pd.read_csv(
            entity_file_path,
            delimiter=" ",
            names=["entity_name", "remap_id"],
            skiprows=1,
        )
        for _, row in df.iterrows():
            entity_id = int(row["remap_id"])
            entity_name = str(row["entity_name"]).strip('"')
            entity_names[entity_id] = entity_name
    return entity_names


def save_results(
    results: dict[int, float],
    output_path: str,
    relation_names: dict[int, str] = None,
):
    """
    結果をCSVファイルに保存

    Args:
        results: relation_id -> average_weight の辞書
        output_path: 出力ファイルパス
        relation_names: リレーション名の辞書（オプション）
    """
    rows = []
    for relation_id, avg_weight in results.items():
        relation_name = (
            relation_names.get(relation_id, f"relation_{relation_id}")
            if relation_names
            else f"relation_{relation_id}"
        )
        rows.append(
            {
                "relation_id": relation_id,
                "relation_name": relation_name,
                "average_weight": avg_weight,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("relation_id")
    df.to_csv(output_path, index=False)
    logger = logging.getLogger(__name__)
    logger.info(f"結果を {output_path} に保存しました（{len(df)} 行）")


def main():
    parser = argparse.ArgumentParser(
        description="学習済みKGATモデルから各エンティティの各リレーションの重み平均を計算"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="学習済みモデルのパス (.pth)",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="yelp",
        help="データセット名",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/",
        help="データセットディレクトリ（スクリプトからの相対パス）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/relation_weight_averages.csv",
        help="出力ファイルパス",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=64,
        help="埋め込み次元",
    )
    parser.add_argument(
        "--relation_dim",
        type=int,
        default=64,
        help="リレーション次元",
    )
    parser.add_argument(
        "--conv_dims",
        type=str,
        default="[64,32,16]",
        help="畳み込み層の次元リスト",
    )
    parser.add_argument(
        "--aggregation_type",
        type=str,
        default="bi-interaction",
        choices=["gcn", "graphsage", "bi-interaction"],
        help="集約タイプ",
    )
    parser.add_argument(
        "--laplacian_type",
        type=str,
        default="random-walk",
        choices=["random-walk", "symmetric"],
        help="ラプラシアンタイプ",
    )
    parser.add_argument(
        "--file_prefix",
        type=str,
        default="total_",
        help="train/test/valファイルのプレフィックス（例: 'total_' で total_train.txt）",
    )

    args = parser.parse_args()
    logger = setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"デバイス: {device}")

    # 設定を作成
    config = KGATConfig()
    config.data_name = args.data_name
    # data_dirを絶対パスに変換
    if not os.path.isabs(args.data_dir):
        # スクリプトのディレクトリ（packages/kgat/）を基準に絶対パスに変換
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config.data_dir = os.path.abspath(os.path.join(script_dir, args.data_dir))
    else:
        config.data_dir = args.data_dir
    
    # total_trn.txt に対応するため、file_prefixを調整
    # DataLoaderは {prefix}train.txt という形式でファイル名を構築するため
    # total_trn.txt を使うには、prefixを "total_trn" にして train.txt の部分を空にする必要がある
    # しかし、DataLoaderのコードを変更するのは難しいため、
    # file_prefixを "total_trn" にして、DataLoaderの後にファイル名を直接変更する
    config.file_prefix = args.file_prefix
    config.embed_dim = args.embed_dim
    config.relation_dim = args.relation_dim
    config.conv_dim_list = args.conv_dims
    config.aggregation_type = args.aggregation_type
    config.laplacian_type = args.laplacian_type
    config.mess_dropout = "[0.1,0.1,0.1]"
    config.use_pretrain = 0

    # データローダーを初期化
    logger.info("データをロード中...")
    
    # total_trn.txt に対応するため、DataLoaderを拡張
    class CustomDataLoader(DataLoader):
        def __init__(self, config: KGATConfig, logger: logging.Logger):
            self.config = config
            self.data_name = config.data_name
            self.use_pretrain = config.use_pretrain
            self.pretrain_embedding_dir = config.pretrain_embedding_dir

            self.data_dir = os.path.join(config.data_dir, self.data_name)
            prefix = config.file_prefix
            
            # 通常のファイル名構築
            self.train_file = os.path.join(self.data_dir, f"{prefix}train.txt")
            self.val_file = os.path.join(self.data_dir, f"{prefix}val.txt")
            self.test_file = os.path.join(self.data_dir, f"{prefix}test.txt")
            
            self.kg_file = os.path.join(self.data_dir, "kg_final.txt")

            self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file)
            self.cf_val_data, self.val_user_dict = self.load_cf(self.val_file)
            self.cf_test_data, self.test_user_dict = self.load_cf(self.test_file)
            self.rng = np.random.default_rng(seed=config.seed)
            self.statistic_cf()

            if self.use_pretrain == 1:
                self.load_pretrain_data()

            self.cf_batch_size = config.cf_batch_size
            self.kg_batch_size = config.kg_batch_size
            self.test_batch_size = config.test_batch_size

            kg_data = self.load_kg(self.kg_file)
            self.construct_data(kg_data)
            self.print_info(logger)

            self.laplacian_type = config.laplacian_type
            self.create_adjacency_dict()
            self.create_laplacian_dict()
    
    data = CustomDataLoader(config, logger)

    # モデルをロード
    logger.info(f"モデルをロード中: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    # チェックポイントから実際のサイズを取得
    entity_user_embed_size = checkpoint["model_state_dict"][
        "entity_user_embed.weight"
    ].shape[0]
    n_relations_ckpt = checkpoint["model_state_dict"]["relation_embed.weight"].shape[
        0
    ]

    logger.info(f"チェックポイントのエンティティ+ユーザー数: {entity_user_embed_size}")
    logger.info(f"チェックポイントのリレーション数: {n_relations_ckpt}")

    # チェックポイントのサイズに合わせてモデルを作成
    ratio = entity_user_embed_size / data.n_users_entities
    n_entities_ckpt = int(data.n_entities * ratio)
    n_users_ckpt = int(data.n_users * ratio)

    logger.info(f"推定: n_users={n_users_ckpt}, n_entities={n_entities_ckpt}")

    model = KGAT(
        config, n_users_ckpt, n_entities_ckpt, n_relations_ckpt, A_in=data.A_in.to(device)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    logger.info("A_inを確認...")
    logger.info(f"A_inの非ゼロ要素数: {model.A_in._nnz()}")

    # KGデータを読み込む
    kg_file = os.path.join(config.data_dir, config.data_name, "kg_final.txt")
    logger.info(f"KGデータを読み込み中: {kg_file}")
    kg_data = load_kg_data(kg_file)

    # リレーション名とエンティティ名を読み込む（オプション）
    relation_file = os.path.join(
        config.data_dir, config.data_name, "relation_list.txt"
    )
    entity_file = os.path.join(config.data_dir, config.data_name, "entity_list.txt")
    relation_names = load_relation_names(relation_file) if os.path.exists(relation_file) else {}
    
    # user-item interactionのリレーション名を追加
    # KGATでは、リレーション0と1がuser-item interaction
    # ただし、relation_list.txtのIDは0から始まるので、KGAT内部のIDに変換する必要がある
    # KGAT内部: 0=Interact, 1=InteractedBy, 2~=KG relations (relation_list.txtのID + 2)
    relation_names[0] = "Interact"  # user -> item
    relation_names[1] = "InteractedBy"  # item -> user
    
    # relation_list.txtのIDをKGAT内部IDに変換（+2）
    if os.path.exists(relation_file):
        original_relation_names = load_relation_names(relation_file)
        for orig_id, name in original_relation_names.items():
            kgat_id = orig_id + 2  # KGAT内部ID = 元のID + 2
            relation_names[kgat_id] = name
    
    entity_names = load_entity_names(entity_file) if os.path.exists(entity_file) else None

    # 重みの平均を計算
    train_file = os.path.join(config.data_dir, config.data_name, f"{config.file_prefix}train.txt")
    logger.info("全体を通して各リレーションの重み平均を計算中...")
    results = calculate_relation_weight_averages(
        model, kg_data, train_file, n_entities_ckpt, n_users_ckpt
    )

    logger.info(f"計算完了: {len(results)} リレーション")

    # 結果を保存
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_results(results, args.output, relation_names)

    # 統計情報を表示
    logger.info("\n=== 統計情報 ===")
    all_weights = list(results.values())
    
    if all_weights:
        logger.info(f"リレーション数: {len(results)}")
        logger.info(f"重みの平均: {np.mean(all_weights):.6f}")
        logger.info(f"重みの標準偏差: {np.std(all_weights):.6f}")
        logger.info(f"重みの最小値: {np.min(all_weights):.6f}")
        logger.info(f"重みの最大値: {np.max(all_weights):.6f}")


if __name__ == "__main__":
    main()
