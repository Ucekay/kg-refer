"""設定モジュール"""

from dataclasses import dataclass

from cyclopts import Parameter


@Parameter(name="*")
@dataclass
class SemanticSimilarityConfig:
    """意味的類似度検索の設定"""

    user_profile_path: str = "datasets/yelp/user_profile.json"
    """ユーザープロファイルJSONファイルのパス"""

    item_profile_path: str = "datasets/yelp/item_profile.json"
    """アイテムプロファイルJSONファイルのパス"""

    train_interactions_path: str = "datasets/yelp/total_train.txt"
    """学習用インタラクションファイルのパス（グラフ構築に使用）"""

    eval_interactions_path: str = "datasets/yelp/interactions_yelp_evaluation.txt"
    """評価用インタラクションファイルのパス"""

    output_file: str = "packages/semantic-similarity-retriever/output/semantic_similarity_results.jsonl"
    """結果を保存するファイルのパス"""

    text_encoder: str = "sentence-transformers/multi-qa-distilbert-cos-v1"
    """使用するテキストエンコーダーモデル名"""

    topk: int = 2
    """取得する類似ノードの最大数"""

    pruning_score: float = 0.0
    """類似度スコアの閾値"""
