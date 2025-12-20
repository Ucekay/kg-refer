#!/bin/bash

# Yelpデータセットに対してkg-attribute-embedderを実行するスクリプト

# データセットのパス
KG_PATH="../../datasets/yelp/cleaned_kg_20251219_230727.json"

# 出力ディレクトリ
OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"

# kg-attribute-embedderを実行
echo "知識グラフの属性埋め込みと類似性分析を開始します..."
echo "データセット: $KG_PATH"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo ""

uv run kg-attribute-embedder \
    "$KG_PATH" \
    --embeddings-output "$OUTPUT_DIR/embeddings.npz" \
    --similarities-output "$OUTPUT_DIR/similarities.jsonl" \
    --threshold 0.8 \
    --batch-size 64 \
    --chunk-size 250

echo ""
echo "処理が完了しました。"
echo "埋め込みファイル: $OUTPUT_DIR/embeddings.npz"
echo "類似ペアファイル: $OUTPUT_DIR/similarities.jsonl (JSON Lines形式)"

