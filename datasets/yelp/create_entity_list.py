#!/usr/bin/env python3
"""
プログラム: create_entity_list.py
説明: merged_kg_2.jsonからtailエンティティ（非アイテム）を抽出し、
      max_item_id + 1から始まるremap_idを割り当ててentity_list.txtを生成する

前提条件:
  - merged_kg_2.jsonのトリプレットのtailエンティティはすべて非アイテム
  - total.csvからアイテムIDの最大値を取得
  - 非アイテムエンティティには max_item_id + 1 から始まるIDを割り当てる

KGATの前提条件を満たす:
  - アイテムID: [0, max_item_id]
  - 非アイテムエンティティID: [max_item_id + 1, ...]
"""

import csv
import json
from collections import OrderedDict


def get_max_item_id(csv_file_path):
    """
    total.csvからアイテムIDの最大値を取得する

    Args:
        csv_file_path (str): total.csvのパス

    Returns:
        int: アイテムIDの最大値
    """
    max_item_id = -1

    with open(csv_file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = int(row["item"])
            if item_id > max_item_id:
                max_item_id = item_id

    return max_item_id


def extract_tail_entities(json_file_path):
    """
    merged_kg_2.jsonからtailエンティティ（非アイテム）を抽出する

    前提: トリプレットのtailエンティティ（3番目の要素）はすべて非アイテム

    Args:
        json_file_path (str): merged_kg_2.jsonのパス

    Returns:
        list: ユニークなtailエンティティのリスト（出現順）
    """
    tail_entities = OrderedDict()  # 順序を保持する辞書

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # データ内の全てのトリプレットからtailエンティティを抽出
    for item in data:
        if "triplets" in item:
            for triplet in item["triplets"]:
                if len(triplet) >= 3:
                    # tailエンティティ（3番目の要素）を抽出
                    tail_entity = str(triplet[2])
                    if tail_entity not in tail_entities:
                        tail_entities[tail_entity] = None

    return list(tail_entities.keys())


def create_entity_list(entities, output_file_path, start_id):
    """
    エンティティリストからentity_list.txtを生成する

    Args:
        entities (list): 非アイテムエンティティのリスト
        output_file_path (str): 出力ファイルのパス
        start_id (int): remap_idの開始値（max_item_id + 1）
    """
    with open(output_file_path, "w", encoding="utf-8") as f:
        # ヘッダーを書き込み
        f.write("entity_name remap_id\n")

        # 各エンティティにstart_idから始まるremap_idを割り当てて書き込み
        for i, entity in enumerate(entities):
            remap_id = start_id + i
            f.write(f'"{entity}" {remap_id}\n')


def main():
    """メイン関数"""
    input_file = "merged_kg_2.json"
    total_csv_file = "total.csv"
    output_file = "entity_list.txt"

    print("=" * 80)
    print("Entity List 生成プログラム")
    print("=" * 80)

    try:
        # Step 1: total.csvからアイテムIDの最大値を取得
        print(f"\nStep 1: '{total_csv_file}' からアイテムIDの最大値を取得中...")
        max_item_id = get_max_item_id(total_csv_file)
        n_items = max_item_id + 1
        print(f"   最大アイテムID: {max_item_id}")
        print(f"   アイテム数: {n_items} (ID: 0 ~ {max_item_id})")

        # Step 2: 非アイテムエンティティの開始IDを計算
        entity_start_id = max_item_id + 1
        print(f"   非アイテムエンティティの開始ID: {entity_start_id}")

        # Step 3: merged_kg_2.jsonからtailエンティティを抽出
        print(f"\nStep 2: '{input_file}' からtailエンティティ（非アイテム）を抽出中...")
        tail_entities = extract_tail_entities(input_file)
        n_entities = len(tail_entities)
        print(f"   抽出された非アイテムエンティティ数: {n_entities}")

        # エンティティのサンプルを表示
        print("\n   非アイテムエンティティ一覧（最初の20個）:")
        for i, entity in enumerate(tail_entities[:20]):
            remap_id = entity_start_id + i
            print(f"     [{remap_id}] {entity}")

        if n_entities > 20:
            print(f"     ... 他 {n_entities - 20} 個")

        # Step 4: entity_list.txtを生成
        print(f"\nStep 3: '{output_file}' を生成中...")
        create_entity_list(tail_entities, output_file, entity_start_id)

        # 結果サマリー
        print(f"\n✓ '{output_file}' を正常に生成しました!")
        print("\n" + "=" * 80)
        print("ID割り当て結果:")
        print("=" * 80)
        print(f"  アイテム:              ID [0, {max_item_id}] ({n_items}個)")
        print(
            f"  非アイテムエンティティ: ID [{entity_start_id}, {entity_start_id + n_entities - 1}] ({n_entities}個)"
        )
        print(f"  総エンティティ数:      {n_items + n_entities}")
        print("=" * 80)

        # KGATの前提条件チェック
        print("\n✓ KGATの前提条件を満たしています:")
        print(f"  - embedding[0 ~ {max_item_id}]: アイテム用")
        print(
            f"  - embedding[{entity_start_id} ~ {entity_start_id + n_entities - 1}]: 非アイテムエンティティ用"
        )
        print(
            f"  - embedding[{n_items + n_entities} ~ ...]: ユーザー用（KGATが自動的に割り当て）"
        )
        print("=" * 80)

    except FileNotFoundError as e:
        print("\nエラー: ファイルが見つかりません。")
        print("カレントディレクトリに以下のファイルがあることを確認してください:")
        print(f"  - {input_file}")
        print(f"  - {total_csv_file}")
    except json.JSONDecodeError:
        print(f"\nエラー: '{input_file}' のJSON形式が不正です。")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
