"""意味的類似度検索のCLI"""

import json
import logging
from pathlib import Path

import pandas as pd
from cyclopts import App

from .config import SemanticSimilarityConfig
from .retriever import SemanticSimilarityRetriever, load_profiles

app = App()

# ロガーの設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@app.default
def main(config: SemanticSimilarityConfig = SemanticSimilarityConfig()):
    """意味的類似度に基づいて類似ノードを検索する

    Args:
        config: 類似度検索の設定
    """
    logger.info("意味的類似度検索を開始します")
    logger.info(f"ユーザープロファイル: {config.user_profile_path}")
    logger.info(f"アイテムプロファイル: {config.item_profile_path}")
    logger.info(f"学習用インタラクション: {config.train_interactions_path}")
    logger.info(f"評価用インタラクション: {config.eval_interactions_path}")

    # プロファイルをロード
    logger.info("プロファイルをロード中...")
    user_profiles, item_profiles = load_profiles(
        config.user_profile_path, config.item_profile_path
    )
    logger.info(f"ユーザー数: {len(user_profiles)}")
    logger.info(f"アイテム数: {len(item_profiles)}")

    # 学習用インタラクションをロード（グラフ構築用）
    logger.info(f"学習用インタラクションをロード中: {config.train_interactions_path}")
    train_interactions_path = Path(config.train_interactions_path)
    
    if not train_interactions_path.exists():
        raise FileNotFoundError(
            f"学習用インタラクションファイルが見つかりません: {config.train_interactions_path}"
        )

    # ファイル形式を確認（CSVかスペース区切りか、またはKGATの特殊な形式か）
    with open(train_interactions_path, "r") as f:
        first_line = f.readline().strip()
    
    if "," in first_line:
        # CSVファイルの場合
        if first_line.startswith("uid"):
            # ヘッダーあり
            train_interactions_df = pd.read_csv(train_interactions_path)
        else:
            # ヘッダーなし
            train_interactions_df = pd.read_csv(
                train_interactions_path, names=["uid", "iid"]
            )
    else:
        # KGAT形式の場合（各行: user_id item_id1 item_id2 ...）
        interactions = []
        with open(train_interactions_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                user_id = int(parts[0])
                for item_id_str in parts[1:]:
                    interactions.append({"uid": user_id, "iid": int(item_id_str)})
        
        train_interactions_df = pd.DataFrame(interactions)

    logger.info(f"学習用インタラクション数: {len(train_interactions_df)}")

    # 評価用インタラクションをロード
    logger.info(f"評価用インタラクションをロード中: {config.eval_interactions_path}")
    eval_interactions_path = Path(config.eval_interactions_path)
    
    if not eval_interactions_path.exists():
        raise FileNotFoundError(
            f"評価用インタラクションファイルが見つかりません: {config.eval_interactions_path}"
        )

    eval_interactions_df = pd.read_csv(eval_interactions_path)
    logger.info(f"評価用インタラクション数: {len(eval_interactions_df)}")

    # 類似ノード検索を実行
    retriever = SemanticSimilarityRetriever(
        user_profiles=user_profiles,
        item_profiles=item_profiles,
        interactions_df=train_interactions_df,
        text_encoder=config.text_encoder,
        pruning_score=config.pruning_score,
    )

    results = retriever.retrieve_batch(eval_interactions_df, topk=config.topk)

    # 結果を保存
    output_path = Path(config.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"結果を保存中: {config.output_file}")
    # JSONL形式で保存（1行1JSONオブジェクト、メモリ効率が良い）
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info("類似ノード検索が完了しました")

    # 統計情報を表示
    total_similar_users = sum(len(r["similar_users"]) for r in results)
    total_similar_items = sum(len(r["similar_items"]) for r in results)

    logger.info(f"\n=== 統計情報 ===")
    logger.info(f"処理したインタラクション数: {len(results)}")
    logger.info(f"検索された類似ユーザー総数: {total_similar_users}")
    logger.info(f"検索された類似アイテム総数: {total_similar_items}")
    if len(results) > 0:
        logger.info(
            f"平均類似ユーザー数/インタラクション: {total_similar_users / len(results):.2f}"
        )
        logger.info(
            f"平均類似アイテム数/インタラクション: {total_similar_items / len(results):.2f}"
        )

    # いくつかの例を表示
    logger.info(f"\n=== 例 (最初の3件) ===")
    for i, result in enumerate(results[:3]):
        logger.info(
            f"\n例 {i + 1}: ユーザー {result['user_id']} - アイテム {result['item_id']}"
        )
        if result["similar_users"]:
            logger.info(f"  類似ユーザー (top 3): {result['similar_users'][:3]}")
        else:
            logger.info(f"  類似ユーザー: なし")
        
        if result["similar_items"]:
            logger.info(f"  類似アイテム (top 3): {result['similar_items'][:3]}")
        else:
            logger.info(f"  類似アイテム: なし")


if __name__ == "__main__":
    app()
