"""
ユーザーの集約埋め込みとアイテムに接続する属性エンティティの集約埋め込みの内積を計算するモジュール
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# kgatパッケージをインポートするためにパスを追加
PACKAGES_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PACKAGES_DIR / "kgat" / "src"))

from kgat.config import KGATConfig  # noqa: E402
from kgat.core.kgat import KGAT  # noqa: E402
from kgat.data.dataloader import DataLoader  # noqa: E402

logger = logging.getLogger(__name__)


def load_kg_connections(
    kg_file_path: str, exclude_relations: list[int] | None = None
) -> dict[int, list[int]]:
    """
    kg_final.txtからアイテムとそれに接続するエンティティのマッピングを作成

    Args:
        kg_file_path: kg_final.txtへのパス
        exclude_relations: 除外するリレーションIDのリスト（kg_final.txtのリレーションID = relation_list.txtのremap_id）

    Returns:
        head -> [tail entities] のマッピング辞書
    """
    kg_connections = {}
    exclude_relations = exclude_relations or []

    with open(kg_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                head, relation, tail = int(parts[0]), int(parts[1]), int(parts[2])

                # 除外するリレーションをスキップ
                if relation in exclude_relations:
                    continue

                if head not in kg_connections:
                    kg_connections[head] = []
                kg_connections[head].append(tail)

    return kg_connections


def load_interactions(interactions_file: str) -> pd.DataFrame:
    """
    interactions_yelp_evaluation.txtを読み込む

    Args:
        interactions_file: interactions_yelp_evaluation.txtへのパス

    Returns:
        DataFrame with columns ['uid', 'iid']
    """
    return pd.read_csv(interactions_file, sep=",")


def load_entity_names(entity_list_path: str) -> dict[int, str]:
    """
    entity_list.txtからエンティティIDと名前のマッピングを作成

    Args:
        entity_list_path: entity_list.txtへのパス

    Returns:
        entity_id -> entity_name のマッピング辞書
    """
    entity_names = {}

    with open(entity_list_path, "r", encoding="utf-8") as f:
        # ヘッダー行をスキップ
        header = f.readline()

        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                # 最後の要素がID、それ以外が名前
                entity_id = int(parts[-1])
                entity_name = " ".join(parts[:-1]).strip('"')
                entity_names[entity_id] = entity_name

    return entity_names


def get_attribute_entities_for_item(
    item_id: int,
    kg_connections: dict[int, list[int]],
    n_items: int,
    n_entities: int,
) -> list[int]:
    """
    アイテムに接続する属性エンティティ（ユーザーでもアイテムでもない）を取得

    Args:
        item_id: アイテムID
        kg_connections: KG接続の辞書
        n_items: アイテム数
        n_entities: エンティティ数（アイテム + 属性）
        n_users: ユーザー数

    Returns:
        属性エンティティIDのリスト
    """
    connected_entities = kg_connections.get(item_id, [])
    attribute_entities = []

    for entity_id in connected_entities:
        # アイテムでもユーザーでもないエンティティ（属性）を抽出
        # エンティティIDの範囲: [0, n_entities-1] のうち、[n_items, n_entities-1]が属性
        # ユーザーIDの範囲: [n_entities, n_entities + n_users - 1]
        if n_items <= entity_id < n_entities:
            attribute_entities.append(entity_id)

    return attribute_entities


def calculate_scores(
    interactions_file: str,
    model_path: str,
    kg_file: str,
    data_dir: str,
    data_name: str,
    output_file: str,
    exclude_relations: list[int] | None = None,
) -> None:
    """
    ユーザー・アイテムペアについて、ユーザーの集約埋め込みと
    アイテムに接続する属性エンティティの集約埋め込みの内積を計算

    Args:
        interactions_file: interactions_yelp_evaluation.txtへのパス
        model_path: 学習済みモデルのパス
        kg_file: kg_final.txtへのパス
        data_dir: データディレクトリ
        data_name: データセット名（yelp）
        output_file: 出力ファイルのパス
    """
    logger.info("データをロード中...")

    # 設定を読み込み
    config = KGATConfig()
    config.data_name = data_name
    config.data_dir = data_dir
    config.use_pretrain = 0
    config.file_prefix = (
        "total_"  # total_train.txt, total_test.txt, total_val.txt を使用
    )

    # モデル設定（学習済みモデルに合わせる）
    config.embed_dim = 64
    config.relation_dim = 64
    config.conv_dim_list = "[64,32,16]"
    config.aggregation_type = "bi-interaction"
    config.laplacian_type = "random-walk"
    config.mess_dropout = "[0.1,0.1,0.1]"

    # データローダーを初期化
    data = DataLoader(config, logger)

    logger.info(
        f"n_users: {data.n_users}, n_entities: {data.n_entities}, n_items: {data.n_items}"
    )

    # モデルをロード
    logger.info("モデルをロード中...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用デバイス: {device}")

    checkpoint = torch.load(model_path, map_location=device)
    logger.info(f"チェックポイントのエポック: {checkpoint.get('epoch', 'N/A')}")

    # チェックポイントから実際のサイズを取得
    entity_user_embed_size = checkpoint["model_state_dict"][
        "entity_user_embed.weight"
    ].shape[0]
    n_relations_ckpt = checkpoint["model_state_dict"]["relation_embed.weight"].shape[0]

    logger.info(f"チェックポイントのエンティティ+ユーザー数: {entity_user_embed_size}")
    logger.info(f"チェックポイントのリレーション数: {n_relations_ckpt}")

    # チェックポイントのサイズに合わせてモデルを作成
    ratio = entity_user_embed_size / data.n_users_entities
    n_entities_ckpt = int(data.n_entities * ratio)
    n_users_ckpt = int(data.n_users * ratio)

    logger.info(f"推定: n_users={n_users_ckpt}, n_entities={n_entities_ckpt}")

    # A_inを設定（埋め込み計算に必要）
    model = KGAT(
        config,
        n_users_ckpt,
        n_entities_ckpt,
        n_relations_ckpt,
        A_in=data.A_in.to(device),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    # 埋め込みを計算
    logger.info("埋め込みを計算中...")
    with torch.no_grad():
        all_embeddings = model.calc_cf_embeddings()
        logger.info(f"全埋め込み形状: {all_embeddings.shape}")

    # KG接続を読み込む
    logger.info("KG接続を読み込み中...")
    if exclude_relations:
        logger.info(f"除外するリレーション: {exclude_relations}")
    kg_connections = load_kg_connections(kg_file, exclude_relations=exclude_relations)

    # エンティティ名を読み込む
    entity_list_file = os.path.join(data_dir, data_name, "entity_list.txt")
    logger.info("エンティティ名を読み込み中...")
    entity_names = load_entity_names(entity_list_file)
    logger.info(f"読み込んだエンティティ数: {len(entity_names)}")

    # インタラクションファイルを読み込む
    logger.info("インタラクションファイルを読み込み中...")
    interactions_df = load_interactions(interactions_file)
    logger.info(f"読み込んだインタラクション数: {len(interactions_df)}")

    # 各ユーザー・アイテムペアについてスコアを計算
    logger.info("スコアを計算中...")
    results = []

    for row_idx, (_, row) in enumerate(interactions_df.iterrows()):
        uid = int(row["uid"])
        iid = int(row["iid"])

        # ユーザーIDを内部表現に変換（n_entities + uid）
        user_id_internal = n_entities_ckpt + uid

        # ユーザー埋め込みを取得
        if user_id_internal >= len(all_embeddings):
            logger.warning(
                f"ユーザーID {uid} (内部ID: {user_id_internal}) が範囲外です"
            )
            continue

        user_embedding = all_embeddings[user_id_internal].cpu().numpy()

        # アイテムに接続する属性エンティティを取得
        # アイテムIDの範囲: [0, n_items-1]
        # 属性エンティティIDの範囲: [n_items, n_entities-1]
        # ユーザーIDの範囲: [n_entities, n_entities+n_users-1]
        attribute_entities = get_attribute_entities_for_item(
            iid, kg_connections, data.n_items, n_entities_ckpt
        )

        if len(attribute_entities) == 0:
            # 属性エンティティがない場合はスキップ
            continue

        # 各属性エンティティごとにユーザー埋め込みとのコサイン類似度を計算
        for attr_id in attribute_entities:
            if attr_id >= len(all_embeddings):
                continue

            # エンティティ名を取得
            attribute_name = entity_names.get(attr_id, f"Unknown_{attr_id}")

            # user, users, individual, individuals, people, those, groups で始まる属性は除外
            # 大文字小文字を区別しない
            # attribute_name_lower = attribute_name.lower()
            # if any(
            #     attribute_name_lower.startswith(prefix)
            #     for prefix in [
            #         "user ",
            #         "users ",
            #         "individual ",
            #         "individuals ",
            #         "people ",
            #         "those ",
            #         "groups ",
            #     ]
            # ):
            #     continue

            # 属性エンティティの埋め込みを取得
            attr_emb = all_embeddings[attr_id].cpu().numpy()

            # ユーザー埋め込みと属性エンティティ埋め込みのコサイン類似度を計算
            norm_user = np.linalg.norm(user_embedding)
            norm_attr = np.linalg.norm(attr_emb)
            score = np.dot(user_embedding, attr_emb) / max(norm_user * norm_attr, 1e-8)

            results.append(
                {
                    "uid": uid,
                    "iid": iid,
                    "attribute_id": attr_id,
                    "attribute_name": attribute_name,
                    "score": float(score),
                }
            )

        if (row_idx + 1) % 100 == 0:
            current_idx = row_idx + 1
            total = len(interactions_df)
            logger.info(f"処理済み: {current_idx}/{total}")

    # 結果をDataFrameに変換して保存
    logger.info("結果を保存中...")
    results_df = pd.DataFrame(results)

    # スコア順にソート
    results_df = results_df.sort_values(
        ["uid", "iid", "score"], ascending=[True, True, False]
    )

    results_df.to_csv(output_file, index=False)
    logger.info(f"結果を保存: {output_file}")
    logger.info(f"総レコード数: {len(results_df)}")
    logger.info(
        f"ユニークなユーザー・アイテムペア数: {results_df[['uid', 'iid']].drop_duplicates().shape[0]}"
    )
    logger.info(f"平均スコア: {results_df['score'].mean():.4f}")
    logger.info(f"スコアの標準偏差: {results_df['score'].std():.4f}")

    # 各ユーザー・アイテムペアごとの統計も表示
    if len(results_df) > 0:
        logger.info("\n=== サンプル結果（上位10件） ===")
        print(results_df.head(10).to_string(index=False))
