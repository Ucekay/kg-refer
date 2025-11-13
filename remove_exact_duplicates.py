#!/usr/bin/env python3
"""
kg_final.txtから完全に同一のトリプル (h r t) を削除するスクリプト
"""

import sys


def remove_exact_duplicates(input_file, output_file):
    """
    完全に同一のトリプルを削除

    Args:
        input_file: 入力KGファイルパス
        output_file: 出力KGファイルパス
    """
    print(f"Reading: {input_file}")

    # setを使ってユニークなトリプルのみを保持
    unique_triples = set()
    total_triples = 0

    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total_triples += 1

            # (h, r, t) のタプルとしてsetに追加
            parts = line.split()
            if len(parts) != 3:
                print(f"Warning: Invalid line format: {line}")
                continue

            try:
                h, r, t = map(int, parts)
                unique_triples.add((h, r, t))
            except ValueError:
                print(f"Warning: Cannot parse line: {line}")
                continue

    # ソートして出力（元の順序に近づけるため）
    unique_triples = sorted(unique_triples)

    print(f"Writing: {output_file}")
    with open(output_file, "w") as f:
        for h, r, t in unique_triples:
            f.write(f"{h} {r} {t}\n")

    # 統計情報
    num_unique = len(unique_triples)
    num_duplicates = total_triples - num_unique

    print()
    print("=" * 60)
    print("処理完了")
    print("=" * 60)
    print(f"入力ファイル: {input_file}")
    print(f"出力ファイル: {output_file}")
    print()
    print(f"元のトリプル数: {total_triples}")
    print(f"ユニークなトリプル数: {num_unique}")
    print(f"削除された重複数: {num_duplicates}")
    print(f"重複率: {num_duplicates / total_triples * 100:.2f}%")
    print()

    return num_unique, num_duplicates


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python remove_exact_duplicates.py <input_kg_file> <output_file>")
        print()
        print("Example:")
        print(
            "  python remove_exact_duplicates.py packages/kgat-pt/datasets/yelp/kg_final.txt \\"
        )
        print(
            "                                     packages/kgat-pt/datasets/yelp/kg_final_no_duplicates.txt"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        remove_exact_duplicates(input_file, output_file)
        print("成功しました！")
    except FileNotFoundError:
        print(f"Error: ファイル '{input_file}' が見つかりません。")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
