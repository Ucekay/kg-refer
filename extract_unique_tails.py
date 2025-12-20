#!/usr/bin/env python3
"""
ユニークなtailを抽出してテキストファイルに保存するスクリプト
"""
import json
import sys
from pathlib import Path


def extract_unique_tails(input_json_path: str, output_txt_path: str):
    """
    JSONファイルからユニークなtailを抽出してテキストファイルに保存
    
    Args:
        input_json_path: 入力JSONファイルのパス
        output_txt_path: 出力テキストファイルのパス
    """
    input_path = Path(input_json_path)
    output_path = Path(output_txt_path)
    
    if not input_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"JSONファイルを読み込んでいます: {input_path}")
    
    # JSONファイルを読み込む
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # すべてのtailを収集
    tails = set()
    total_triplets = 0
    
    for entry in data:
        if "triplets" in entry:
            for triplet in entry["triplets"]:
                if len(triplet) >= 3:
                    tail = triplet[2]  # tailは3番目の要素（インデックス2）
                    tails.add(tail)
                    total_triplets += 1
    
    # ユニークなtailをソートして保存
    unique_tails = sorted(tails)
    
    print(f"総triplet数: {total_triplets}")
    print(f"ユニークなtail数: {len(unique_tails)}")
    print(f"テキストファイルに保存中: {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for tail in unique_tails:
            f.write(f"{tail}\n")
    
    print(f"完了: {len(unique_tails)}個のユニークなtailを保存しました")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python extract_unique_tails.py <input_json> [output_txt]")
        print("例: python extract_unique_tails.py datasets/yelp/cleaned_kg_20251219_230727.json unique_tails.txt")
        sys.exit(1)
    
    input_json = sys.argv[1]
    output_txt = sys.argv[2] if len(sys.argv) > 2 else "unique_tails.txt"
    
    extract_unique_tails(input_json, output_txt)
