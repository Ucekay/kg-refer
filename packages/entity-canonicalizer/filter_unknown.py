#!/usr/bin/env python3
"""
Unknownを含むトリプレットを削除し、クリーンなデータを作成するスクリプト
"""

import json


def contains_filtered_terms(text):
    """テキストにフィルタリング対象の用語が含まれるかチェックする"""
    if not isinstance(text, str):
        return False

    filtered_terms = [
        "Unknown",
        "Cuisine",
        "cuisine",
        "Food",
        "food",
        "Users",
        "users",
        "unspecified",
        "dishes",
        "Dishes",
        "not specified",
        "Not specified",
    ]
    return any(term in text for term in filtered_terms)


def filter_triplets(data):
    """フィルタリング対象の用語を含むトリプレットを削除する"""
    filtered_data = []

    for entry in data:
        if "triplets" in entry:
            filtered_triplets = []
            for triplet in entry["triplets"]:
                if len(triplet) >= 3:
                    subject, relation, obj = triplet[0], triplet[1], triplet[2]

                    # フィルタリング対象の用語を含むトリプレットは除外
                    if not (
                        contains_filtered_terms(subject) or contains_filtered_terms(obj)
                    ):
                        filtered_triplets.append(triplet)

            # トリプレットが残っている場合のみエントリを保持
            if filtered_triplets:
                filtered_entry = entry.copy()
                filtered_entry["triplets"] = filtered_triplets
                filtered_data.append(filtered_entry)

    return filtered_data


def main():
    input_file = (
        "/home/kimura/repos/kg-refer/packages/entity-canonicalizer/data/canon_kg.json"
    )
    output_file = "/home/kimura/repos/kg-refer/packages/entity-canonicalizer/data/canon_kg_filtered.json"

    print("元データを読み込み中...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"元データ: {len(data)} 件のエントリ")

    # トリプレットをフィルタリング
    print("Unknownを含むトリプレットを削除中...")
    filtered_data = filter_triplets(data)

    # 結果を保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print(f"フィルタリング後データ: {len(filtered_data)} 件のエントリ")
    print(f"結果は {output_file} に保存されました。")

    # 削除されたトリプレットの数を計算
    original_triplets = sum(len(entry.get("triplets", [])) for entry in data)
    filtered_triplets = sum(len(entry.get("triplets", [])) for entry in filtered_data)
    removed_triplets = original_triplets - filtered_triplets

    print(f"元のトリプレット数: {original_triplets}")
    print(f"削除されたトリプレット数: {removed_triplets}")
    print(f"残ったトリプレット数: {filtered_triplets}")
    print(f"削除率: {removed_triplets / original_triplets * 100:.1f}%")

    print(f"削除率: {removed_triplets / original_triplets * 100:.1f}%")


if __name__ == "__main__":
    main()
