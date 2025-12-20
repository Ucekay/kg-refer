"""使用例スクリプト"""

from pathlib import Path

from kg_attribute_embedder import AttributeEmbedder

# データファイルのパス
kg_path = Path(__file__).parent.parent.parent / "datasets" / "yelp" / "merged_kg_2.json"
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# 出力ファイルのパス
embeddings_path = output_dir / "embeddings.npz"
similarities_path = output_dir / "similarities.json"

# AttributeEmbedderインスタンスを作成
embedder = AttributeEmbedder(
    model_name="sentence-transformers/multi-qa-distilbert-cos-v1"
)

# パイプライン全体を実行
embedder.process_pipeline(
    kg_path=kg_path,
    embeddings_output_path=embeddings_path,
    similarities_output_path=similarities_path,
    similarity_threshold=0.8,
    batch_size=64,  # より大きなバッチサイズで高速化
    chunk_size=1000,  # メモリ使用量を制御
)

print("\n処理が完了しました。")
print(f"埋め込みファイル: {embeddings_path}")
print(f"類似ペアファイル: {similarities_path}")

