"""一時ファイルをソートするだけのスクリプト"""

import argparse
from pathlib import Path

from .embedder import AttributeEmbedder


def main() -> None:
    """ソートのみを実行するメイン関数"""
    parser = argparse.ArgumentParser(
        description="JSON Lines形式の一時ファイルをソートしてJSON Lines形式で出力"
    )
    
    parser.add_argument(
        "temp_file",
        type=str,
        help="一時ファイルのパス（JSON Lines形式、例: similarities_temp.jsonl）",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="出力ファイルのパス（JSON Lines形式、例: similarities.jsonl）",
    )
    
    parser.add_argument(
        "--chunk-lines",
        type=int,
        default=100000,
        help="外部ソート時の1チャンクあたりの行数（デフォルト: 100000）",
    )
    
    args = parser.parse_args()
    
    temp_path = Path(args.temp_file)
    output_path = Path(args.output)
    
    if not temp_path.exists():
        print(f"エラー: ファイルが見つかりません: {temp_path}")
        return
    
    print(f"入力ファイル: {temp_path}")
    print(f"出力ファイル: {output_path}")
    print()
    
    # AttributeEmbedderのソート機能を使用
    embedder = AttributeEmbedder()
    embedder._sort_and_save_pairs(temp_path, output_path)
    
    print("\nソート完了!")


if __name__ == "__main__":
    main()
