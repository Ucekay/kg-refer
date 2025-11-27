"""
KGATモデルを使用して、ユーザーとアイテムのエンティティ埋め込みを分析するスクリプト

このスクリプトは以下を実行します：
1. データセットからランダムにユーザーを1人選択
2. 学習済みKGATモデルからそのユーザーの埋め込みを取得
3. テストデータにあるアイテムに対してkg_finalから接続するエンティティを取得
4. エンティティの埋め込みも学習済みモデルから取得
5. ユーザー埋め込みと各アイテムのエンティティ埋め込みの内積を計算してスコア順に並べる
6. 結果をoutput/に保存
"""

import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.kgat.config import KGATConfig
from src.kgat.core.kgat import KGAT
from src.kgat.data.dataloader import DataLoader
from src.kgat.utils.model_helper import load_model


def load_kg_connections(kg_file_path: str) -> dict[int, list[int]]:
    """
    kg_final.txtからアイテム（またはエンティティ）とそれに接続するエンティティのマッピングを作成

    Args:
        kg_file_path: kg_final.txtへのパス

    Returns:
        head -> [tail entities] のマッピング辞書
    """
    kg_connections = {}

    with open(kg_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                head, relation, tail = int(parts[0]), int(parts[1]), int(parts[2])
                if head not in kg_connections:
                    kg_connections[head] = []
                kg_connections[head].append(tail)

    return kg_connections


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


def main():
    # ===== ユーザーID指定 =====
    # 特定のユーザーを選択する場合はここでIDを指定、Noneの場合はランダム選択
    uid = 1326  # ユーザーID（元の表現）。Noneにするとランダム選択
    # =========================

    # ロギング設定
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # 設定を読み込み
    config = KGATConfig()
    config.data_name = "yelp"
    config.data_dir = "datasets/"
    config.use_pretrain = 0  # pretrainなしでモデルをロード

    # モデルのアーキテクチャに合わせて設定を調整
    # チェックポイントの形状から判断: 実際には2層のアグリゲータ
    # 内部で [embed_dim] + conv_dim_list = [64] + [32, 16] = [64, 32, 16] となる
    config.embed_dim = 64
    config.relation_dim = 64
    config.conv_dim_list = "[64,32,16]"  # 内部で [64, 32, 16] に展開される
    config.aggregation_type = "bi-interaction"
    config.laplacian_type = "random-walk"
    config.mess_dropout = "[0.1,0.1,0.1]"  # 2層なので2つのdropout値

    # モデルパスを設定（最良のモデルを選択）
    model_path = "trained_model/KGAT/yelp/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain0/model_epoch1000.pth"
    config.pretrain_model_path = model_path

    logger.info(f"モデルパス: {model_path}")

    # データローダーを初期化
    logger.info("データをロード中...")
    data = DataLoader(config, logger)

    # 1. ユーザーを選択（指定されていればそれを使用、なければランダム）
    test_users = list(data.test_user_dict.keys())

    if uid is not None:
        # 指定されたユーザーIDを使用
        logger.info(f"指定されたユーザーID: {uid}")
        random_user_id = uid + data.n_entities

        # このユーザーがテストデータに存在するか確認
        if random_user_id not in test_users:
            logger.error(f"ユーザーID {uid} はテストデータに存在しません")
            logger.info(
                f"利用可能なユーザーIDの範囲: 0 - {max(test_users) - data.n_entities}"
            )
            return

        original_user_id = uid
    else:
        # ランダムに選択
        logger.info("ランダムにユーザーを選択...")
        random_user_id = random.choice(test_users)
        original_user_id = random_user_id - data.n_entities

    logger.info(f"選択されたユーザーID（内部表現）: {random_user_id}")
    logger.info(f"選択されたユーザーID（元の表現）: {original_user_id}")

    # 2. モデルをロードしてユーザー埋め込みを取得
    logger.info("モデルをロード中...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # チェックポイントを読み込んで実際のモデルサイズを確認
    checkpoint = torch.load(model_path, map_location=device)
    logger.info(f"チェックポイントのエポック: {checkpoint.get('epoch', 'N/A')}")

    # チェックポイントから実際のサイズを取得
    entity_user_embed_size = checkpoint["model_state_dict"][
        "entity_user_embed.weight"
    ].shape[0]
    n_relations_ckpt = checkpoint["model_state_dict"]["relation_embed.weight"].shape[0]

    logger.info(f"チェックポイントのエンティティ+ユーザー数: {entity_user_embed_size}")
    logger.info(f"チェックポイントのリレーション数: {n_relations_ckpt}")
    logger.info(
        f"現在のデータ: エンティティ+ユーザー={data.n_users_entities}, リレーション={data.n_relations}"
    )

    # チェックポイントのサイズに合わせてモデルを作成
    # n_users_entitiesからn_usersとn_entitiesを逆算
    # entity_user_embed_size = n_entities + n_users なので
    # 現在のデータから比率を使って推定
    ratio = entity_user_embed_size / data.n_users_entities
    n_entities_ckpt = int(data.n_entities * ratio)
    n_users_ckpt = int(data.n_users * ratio)

    logger.info(f"推定: n_users={n_users_ckpt}, n_entities={n_entities_ckpt}")

    model = KGAT(config, n_users_ckpt, n_entities_ckpt, n_relations_ckpt, A_in=None)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    logger.info("埋め込みを計算中...")
    with torch.no_grad():
        # 全埋め込みを計算
        all_embeddings = model.calc_cf_embeddings()

        # ユーザー埋め込みを取得
        user_embedding = all_embeddings[random_user_id].cpu().numpy()
        logger.info(f"ユーザー埋め込み形状: {user_embedding.shape}")

    # 3. このユーザーがインタラクションを持つアイテムを取得
    user_items = data.test_user_dict.get(random_user_id, np.array([]))
    logger.info(f"ユーザーがインタラクションを持つアイテム数: {len(user_items)}")

    if len(user_items) == 0:
        logger.warning("このユーザーはテストデータにインタラクションがありません")
        return

    # 4. それらのアイテムに接続する属性エンティティを取得
    kg_file = os.path.join(config.data_dir, config.data_name, "kg_final.txt")
    entity_list_file = os.path.join(
        config.data_dir, config.data_name, "entity_list.txt"
    )

    logger.info("KG接続を読み込み中...")
    kg_connections = load_kg_connections(kg_file)

    logger.info("エンティティ名を読み込み中...")
    entity_names = load_entity_names(entity_list_file)

    # ユーザーのアイテムごとに接続するエンティティを収集
    logger.info("アイテムごとに接続するエンティティを収集中...")
    item_to_entities = {}  # アイテム -> それに接続しているエンティティのリスト

    for item_id in user_items:
        connected_entities = kg_connections.get(item_id, [])
        item_to_entities[item_id] = connected_entities

    logger.info(f"アイテム数: {len(item_to_entities)}")

    # 5. 各アイテムについて、接続するエンティティのスコアを計算
    logger.info("各アイテムのエンティティスコアを計算中...")

    all_item_entity_scores = []  # 全データを保存するリスト

    for item_id, entity_ids in item_to_entities.items():
        logger.info(f"\nアイテム {item_id}: {len(entity_ids)}個のエンティティ")

        entity_scores_for_item = []

        for entity_id in entity_ids:
            # エンティティの埋め込みを取得
            if entity_id < len(all_embeddings):
                entity_emb = all_embeddings[entity_id].cpu().numpy()

                # ユーザー埋め込みとの内積を計算
                score = np.dot(user_embedding, entity_emb)

                entity_scores_for_item.append(
                    {
                        "item_id": item_id,
                        "entity_id": entity_id,
                        "entity_name": entity_names.get(
                            entity_id, f"Unknown_{entity_id}"
                        ),
                        "score": score,
                    }
                )

        # このアイテムのエンティティをスコア順にソート
        entity_scores_for_item_sorted = sorted(
            entity_scores_for_item, key=lambda x: x["score"], reverse=True
        )

        # 上位5件を表示
        if entity_scores_for_item_sorted:
            logger.info(f"  上位5エンティティ:")
            for i, entry in enumerate(entity_scores_for_item_sorted[:5], 1):
                logger.info(
                    f"    {i}. [{entry['entity_id']}] {entry['entity_name']}: score={entry['score']:.4f}"
                )

        all_item_entity_scores.extend(entity_scores_for_item_sorted)

    # 6. 結果をDataFrameに変換して保存
    logger.info("\n結果を保存中...")
    results_df = pd.DataFrame(all_item_entity_scores)

    # 出力ディレクトリを作成
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # アイテムごとに整列した結果を保存
    output_file = output_dir / f"user_{original_user_id}_entity_scores_by_item.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"結果を保存: {output_file}")

    # アイテムごとの統計も作成
    item_stats = []
    for item_id in item_to_entities.keys():
        item_entities = results_df[results_df["item_id"] == item_id]
        if len(item_entities) > 0:
            item_stats.append(
                {
                    "item_id": item_id,
                    "num_entities": len(item_entities),
                    "avg_score": item_entities["score"].mean(),
                    "max_score": item_entities["score"].max(),
                    "min_score": item_entities["score"].min(),
                }
            )

    item_stats_df = pd.DataFrame(item_stats)
    item_stats_file = output_dir / f"user_{original_user_id}_item_stats.csv"
    item_stats_df.to_csv(item_stats_file, index=False)
    logger.info(f"アイテム統計を保存: {item_stats_file}")

    # 全体の統計情報を保存
    stats = {
        "selected_user_id": original_user_id,
        "selected_user_id_internal": random_user_id,
        "num_user_items": len(user_items),
        "total_entities": len(results_df),
        "avg_score": results_df["score"].mean(),
        "max_score": results_df["score"].max(),
        "min_score": results_df["score"].min(),
    }

    stats_file = output_dir / f"user_{original_user_id}_entity_stats.txt"
    with open(stats_file, "w") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

    logger.info(f"\n統計情報を保存: {stats_file}")

    # サマリー表示
    logger.info("\n=== サマリー ===")
    logger.info(f"ユーザー {original_user_id} のアイテム数: {len(user_items)}")
    logger.info(f"全エンティティ数: {len(results_df)}")
    logger.info("\nアイテムごとの統計:")
    print(item_stats_df.to_string(index=False))

    logger.info("\n完了!")


if __name__ == "__main__":
    main()
