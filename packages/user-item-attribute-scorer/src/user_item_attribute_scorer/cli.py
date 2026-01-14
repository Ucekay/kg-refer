"""
CLIエントリーポイント
"""

import argparse
import logging
import sys
from pathlib import Path

from user_item_attribute_scorer.score_calculator import calculate_scores

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="ユーザーの集約埋め込みとアイテムに接続する属性エンティティの集約埋め込みの内積を計算"
    )
    parser.add_argument(
        "--interactions-file",
        type=str,
        required=True,
        help="interactions_yelp_evaluation.txtへのパス",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="学習済みモデルのパス",
    )
    parser.add_argument(
        "--kg-file",
        type=str,
        default=None,
        help="kg_final.txtへのパス（デフォルト: data_dir/data_name/kg_final.txt）",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets/",
        help="データディレクトリ（デフォルト: datasets/）",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default="yelp",
        help="データセット名（デフォルト: yelp）",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="出力ファイルのパス",
    )
    parser.add_argument(
        "--exclude-relations",
        type=str,
        default=None,
        help="除外するリレーションIDのカンマ区切りリスト（kg_final.txtのID = relation_list.txtのremap_id。例: '5' で 'located in' を除外）",
    )

    args = parser.parse_args()

    # kg_fileが指定されていない場合はデフォルトパスを使用
    if args.kg_file is None:
        args.kg_file = str(Path(args.data_dir) / args.data_name / "kg_final.txt")

    # パスの存在確認
    if not Path(args.interactions_file).exists():
        logger.error(f"インタラクションファイルが見つかりません: {args.interactions_file}")
        sys.exit(1)

    if not Path(args.model_path).exists():
        logger.error(f"モデルファイルが見つかりません: {args.model_path}")
        sys.exit(1)

    if not Path(args.kg_file).exists():
        logger.error(f"KGファイルが見つかりません: {args.kg_file}")
        sys.exit(1)

    # 除外するリレーションIDのリストを作成
    exclude_relations = None
    if args.exclude_relations:
        try:
            exclude_relations = [int(r.strip()) for r in args.exclude_relations.split(",")]
            logger.info(f"除外するリレーション: {exclude_relations}")
        except ValueError:
            logger.error(f"無効なリレーションID: {args.exclude_relations}")
            sys.exit(1)

    # スコア計算を実行
    try:
        calculate_scores(
            interactions_file=args.interactions_file,
            model_path=args.model_path,
            kg_file=args.kg_file,
            data_dir=args.data_dir,
            data_name=args.data_name,
            output_file=args.output_file,
            exclude_relations=exclude_relations,
        )
        logger.info("完了!")
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
