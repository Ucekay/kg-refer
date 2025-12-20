"""CLIインターフェース"""

import argparse
from pathlib import Path

from .embedder import AttributeEmbedder


def main() -> None:
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="知識グラフの属性埋め込みと類似性分析"
    )
    
    parser.add_argument(
        "kg_path",
        type=str,
        help="知識グラフJSONファイルのパス",
    )
    
    parser.add_argument(
        "--embeddings-output",
        type=str,
        default="embeddings.npz",
        help="埋め込みの出力ファイルパス（デフォルト: embeddings.npz）",
    )
    
    parser.add_argument(
        "--similarities-output",
        type=str,
        default="similarities.jsonl",
        help="類似ペアの出力ファイルパス（JSON Lines形式、デフォルト: similarities.jsonl）",
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="類似度の閾値（デフォルト: 0.8）",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="埋め込み生成時のバッチサイズ（デフォルト: 32）",
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="類似度計算時のチャンクサイズ（デフォルト: 1000）",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/multi-qa-distilbert-cos-v1",
        help="使用するSentence Transformerモデル",
    )
    
    args = parser.parse_args()
    
    # AttributeEmbedderインスタンスを作成
    embedder = AttributeEmbedder(model_name=args.model)
    
    # パイプライン実行
    embedder.process_pipeline(
        kg_path=args.kg_path,
        embeddings_output_path=args.embeddings_output,
        similarities_output_path=args.similarities_output,
        similarity_threshold=args.threshold,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )
    
    print("\n処理が完了しました。")


if __name__ == "__main__":
    main()

