#!/usr/bin/env python3
"""
エンティティ出現回数を集計するスクリプト
IDを除くすべてのエンティティの出現回数をカウントし、JSON形式で出力する
"""

import json
import re
from collections import Counter


def is_numeric_id(text):
    """数値ID（純粋な数字）かどうかを判定する"""
    if isinstance(text, (int, float)):
        return True
    return isinstance(text, str) and text.isdigit()


def extract_entities_from_triplets(triplets):
    """tripletsからID以外のエンティティを抽出する（relationは除く）"""
    entities = []
    for triplet in triplets:
        if len(triplet) >= 3:
            subject, relation, obj = triplet[0], triplet[1], triplet[2]

            # 主体（subject）が数値IDでなければ追加
            if not is_numeric_id(subject):
                entities.append(subject)

            # 対象（object）が数値IDでなければ追加
            if not is_numeric_id(obj):
                entities.append(obj)

    return entities


def count_entities(json_file_path):
    """JSONファイルからエンティティの出現回数をカウントする"""
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_entities = []

    for entry in data:
        if "triplets" in entry:
            entities = extract_entities_from_triplets(entry["triplets"])
            all_entities.extend(entities)

    # エンティティの出現回数をカウント
    entity_counts = Counter(all_entities)

    return entity_counts


def generate_statistics(entity_counts):
    """統計情報を生成する"""
    total_entities = sum(entity_counts.values())
    unique_entity_types = len(entity_counts)

    # 割合を計算
    entity_stats = {}
    for entity, count in entity_counts.most_common():
        percentage = (count / total_entities) * 100
        entity_stats[entity] = {"count": count, "percentage": round(percentage, 2)}

    return {
        "entity_counts": entity_stats,
        "total_entities": total_entities,
        "unique_entity_types": unique_entity_types,
    }


def main():
    input_file = "/home/kimura/repos/kg-refer/packages/entity-canonicalizer/data/canon_kg_normalized.json"
    output_file = "/home/kimura/repos/kg-refer/packages/entity-canonicalizer/data/entity_counts_normalized.json"

    # エンティティをカウント
    entity_counts = count_entities(input_file)

    # 統計情報を生成
    statistics = generate_statistics(entity_counts)

    # 結果をJSONファイルに出力
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)

    print(f"エンティティ出現回数の集計完了！")
    print(f"総エンティティ数: {statistics['total_entities']}")
    print(f"ユニークエンティティタイプ数: {statistics['unique_entity_types']}")
    print(f"結果は {output_file} に出力されました。")

    # 上位10件を表示
    print("\n出現回数トップ10:")
    for i, (entity, count) in enumerate(entity_counts.most_common(10), 1):
        percentage = (count / statistics["total_entities"]) * 100
        print(f"{i}. {entity}: {count}回 ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
