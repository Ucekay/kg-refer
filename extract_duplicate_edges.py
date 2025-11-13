#!/usr/bin/env python3
"""
Knowledge Graphで同じ(head, tail)ペアに対して複数の関係があるエッジを
元のkg_final.txtと同じ形式で出力するスクリプト
"""

import sys
from collections import defaultdict


def extract_duplicate_edges(input_file, output_file):
    """
    重複エッジを抽出して出力する

    Args:
        input_file: 入力KGファイルパス
        output_file: 出力ファイルパス（重複エッジのみ）
    """
    # (head, tail) -> [(relation, line_num), ...] のマッピング
    edge_relations = defaultdict(list)
    all_triples = []

    # ファイルを読み込む
    print(f"Reading: {input_file}")
    with open(input_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 3:
                print(f"Warning: Line {line_num} has invalid format: {line}")
                continue

            head, relation, tail = map(int, parts)
            edge_key = (head, tail)
            edge_relations[edge_key].append((relation, line_num))
            all_triples.append((head, relation, tail))

    # 重複があるエッジのみを抽出
    duplicate_triples = []

    for edge_key, relation_list in edge_relations.items():
        if len(relation_list) > 1:
            head, tail = edge_key
            # このエッジに対応するすべてのトリプルを追加
            for relation, _ in relation_list:
                duplicate_triples.append((head, relation, tail))

    # 出力ファイルに書き込み
    print(f"Writing: {output_file}")
    with open(output_file, "w") as f:
        for head, relation, tail in duplicate_triples:
            f.write(f"{head} {relation} {tail}\n")

    # 統計情報
    total_edges = len(edge_relations)
    duplicate_edges = sum(1 for rels in edge_relations.values() if len(rels) > 1)
    duplicate_triples_count = len(duplicate_triples)

    print()
    print("=" * 60)
    print("処理完了")
    print("=" * 60)
    print(f"入力ファイル: {input_file}")
    print(f"出力ファイル: {output_file}")
    print()
    print(f"総エッジ数（ユニークな(head, tail)ペア）: {total_edges}")
    print(f"総トリプル数: {len(all_triples)}")
    print(
        f"重複があったエッジ数: {duplicate_edges} ({duplicate_edges / total_edges * 100:.2f}%)"
    )
    print(f"重複エッジのトリプル数: {duplicate_triples_count}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_duplicate_edges.py <input_kg_file> <output_file>")
        print()
        print("Example:")
        print(
            "  python extract_duplicate_edges.py packages/kgat-pt/datasets/yelp/kg_final.txt \\"
        )
        print(
            "                                     packages/kgat-pt/datasets/yelp/kg_duplicates.txt"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        extract_duplicate_edges(input_file, output_file)
        print("成功しました！")
    except FileNotFoundError:
        print(f"Error: ファイル '{input_file}' が見つかりません。")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
