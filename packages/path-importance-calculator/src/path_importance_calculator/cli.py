"""コマンドラインインターフェース"""

import logging
import sys
from dataclasses import dataclass
from typing import Literal

from cyclopts import App, Parameter

from kgat.config import KGATConfig
from path_importance_calculator.calculator import PathImportanceCalculator
from path_importance_calculator.model_loader import KGATModelLoader


app = App(help="Calculate path importance between user-item pairs using trained KGAT model")


@Parameter(name="*")
@dataclass
class Config:
    """パス重要度計算の設定"""

    model_path: str
    "Path to trained KGAT model (e.g., model_epoch50.pth)"
    interaction_file: str
    "Path to user-item interaction file"
    output_file: str
    "Path to output file"

    data_dir: str = "packages/kgat/datasets/"
    "Directory containing data files"
    data_name: str = "yelp"
    "Dataset name"
    max_hops: int = 3
    "Maximum number of hops for path search"
    min_hops: int = 1
    "Minimum number of hops for path search"
    output_format: Literal["jsonl", "json", "csv"] = "jsonl"
    "Output format (jsonl, json, or csv)"

    embed_dim: int = 64
    "Embedding dimension"
    relation_dim: int = 64
    "Relation dimension"
    laplacian_type: Literal["random-walk", "symmetric"] = "random-walk"
    "Laplacian type (random-walk or symmetric)"
    aggregation_type: Literal["gcn", "graphsage", "bi-interaction"] = "bi-interaction"
    "Aggregation type"
    conv_dim_list: str = "[64,32,16]"
    "Convolution dimension list"
    ignore_relations: str = "[]"
    "List of relation IDs to ignore in path search (e.g., '[5]' to ignore 'located in')"
    entity_list_file: str = ""
    "Path to entity_list.txt for entity names (optional)"
    relation_list_file: str = ""
    "Path to relation_list.txt for relation names (optional)"
    topk: int = 2
    "Number of top paths to retrieve per pair (used only when selection_mode='topk')"
    selection_mode: Literal["topk", "diverse"] = "topk"
    "Path selection mode ('topk' for top-k paths, 'diverse' for one interaction path and one diverse path)"


@app.default
def main(config: Config) -> None:
    """パス重要度を計算してファイルに保存する"""

    # ロガーの設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    # KGAT設定を作成
    kgat_config = KGATConfig(
        data_dir=config.data_dir,
        data_name=config.data_name,
        embed_dim=config.embed_dim,
        relation_dim=config.relation_dim,
        laplacian_type=config.laplacian_type,
        aggregation_type=config.aggregation_type,
        conv_dim_list=config.conv_dim_list,
        use_pretrain=0,  # 学習済みモデルをロードするので不要
    )

    # モデルローダーを作成
    model_loader = KGATModelLoader(config.model_path, kgat_config, logger)

    # ignore_relationsをパース
    ignore_relations = eval(config.ignore_relations) if config.ignore_relations else []

    # entity_list_fileとrelation_list_fileを処理
    entity_list_file = config.entity_list_file if config.entity_list_file else None
    relation_list_file = config.relation_list_file if config.relation_list_file else None

    # 計算器を作成
    calculator = PathImportanceCalculator(
        model_loader,
        max_hops=config.max_hops,
        min_hops=config.min_hops,
        ignore_relations=ignore_relations,
        entity_list_file=entity_list_file,
        relation_list_file=relation_list_file,
    )

    # パス重要度を計算
    logger.info("Calculating path importance...")
    results = calculator.calculate_for_pairs(
        config.interaction_file, topk=config.topk, selection_mode=config.selection_mode
    )

    # 結果を保存
    calculator.save_results(results, config.output_file, format=config.output_format)

    logger.info("Done!")


if __name__ == "__main__":
    app()
