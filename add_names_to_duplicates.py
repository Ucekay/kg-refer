#!/usr/bin/env python3
"""
kg_duplicates.txtに関係名とtailエンティティ名を追加するスクリプト

出力フォーマット:
head relation tail relation_name tail_entity_name
"""

import sys


def load_entity_names(entity_file):
    """
    entity_list.txtからエンティティ名を読み込む

    Returns:
        dict: {entity_id: entity_name}
    """
    entity_names = {}

    with open(entity_file, "r", encoding="utf-8") as f:
        # ヘッダー行をスキップ
        next(f)

        for line in f:
            line = line.strip()
            if not line:
                continue

            # entity_name remap_id の形式
            parts = line.rsplit(None, 1)  # 最後の空白で分割
            if len(parts) != 2:
                continue

            entity_name = parts[0].strip('"')  # クォートを削除
            entity_id = int(parts[1])
            entity_names[entity_id] = entity_name

    return entity_names


def load_relation_names(relation_file):
    """
    relation_list.txtから関係名を読み込む

    Returns:
        dict: {relation_id: relation_name}
    """
    relation_names = {}

    with open(relation_file, "r", encoding="utf-8") as f:
        # ヘッダー行をスキップ
        next(f)

        for line in f:
            line = line.strip()
            if not line:
                continue

            # relation_name remap_id の形式
            parts = line.rsplit(None, 1)  # 最後の空白で分割
            if len(parts) != 2:
                continue

            relation_name = parts[0].strip('"')  # クォートを削除
            relation_id = int(parts[1])
            relation_names[relation_id] = relation_name

    return relation_names


def add_names_to_duplicates(kg_file, entity_file, relation_file, output_file):
    """
    kg_duplicates.txtに関係名とエンティティ名を追加

    Args:
        kg_file: kg_duplicates.txtのパス
        entity_file: entity_list.txtのパス
        relation_file: relation_list.txtのパス
        output_file: 出力ファイルのパス
    """
    print(f"Loading entity names from: {entity_file}")
    entity_names = load_entity_names(entity_file)
    print(f"  Loaded {len(entity_names)} entities")

    print(f"Loading relation names from: {relation_file}")
    relation_names = load_relation_names(relation_file)
    print(f"  Loaded {len(relation_names)} relations")

    print(f"Processing: {kg_file}")
    print(f"Writing to: {output_file}")

    processed = 0
    missing_entities = set()
    missing_relations = set()

    with open(kg_file, "r") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 3:
                print(f"Warning: Invalid line format: {line}")
                continue

            head, relation, tail = map(int, parts)

            # 関係名を取得
            relation_name = relation_names.get(relation, f"UNKNOWN_RELATION_{relation}")
            if relation not in relation_names:
                missing_relations.add(relation)

            # tailエンティティ名を取得
            tail_name = entity_names.get(tail, f"UNKNOWN_ENTITY_{tail}")
            if tail not in entity_names:
                missing_entities.add(tail)

            # 出力: head relation tail relation_name tail_entity_name
            f_out.write(f"{head} {relation} {tail} {relation_name} {tail_name}\n")
            processed += 1

    print(f"\n処理完了: {processed} トリプル")

    if missing_relations:
        print(f"\n警告: {len(missing_relations)} 個の関係IDが見つかりませんでした:")
        print(f"  {sorted(missing_relations)}")

    if missing_entities:
        print(
            f"\n警告: {len(missing_entities)} 個のエンティティIDが見つかりませんでした:"
        )
        print(
            f"  {sorted(list(missing_entities))[:20]}{'...' if len(missing_entities) > 20 else ''}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(
            "Usage: python add_names_to_duplicates.py <kg_duplicates> <entity_list> <relation_list> <output_file>"
        )
        print()
        print("Example:")
        print("  python add_names_to_duplicates.py \\")
        print("         packages/kgat-pt/datasets/yelp/kg_duplicates.txt \\")
        print("         data/yelp/entity_list.txt \\")
        print("         data/yelp/relation_list.txt \\")
        print("         packages/kgat-pt/datasets/yelp/kg_duplicates_with_names.txt")
        sys.exit(1)

    kg_file = sys.argv[1]
    entity_file = sys.argv[2]
    relation_file = sys.argv[3]
    output_file = sys.argv[4]

    try:
        add_names_to_duplicates(kg_file, entity_file, relation_file, output_file)
        print("\n成功しました！")
    except FileNotFoundError as e:
        print(f"Error: ファイルが見つかりません: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
