#!/usr/bin/env python3
"""
プログラム: create_entity_list.py
説明: canon_kg_normalized.jsonから全てのユニークなエンティティを抽出し、
remap_idを割り当ててentity_list.txtを生成する
"""

import json
from collections import OrderedDict


def extract_entities(json_file_path):
    """
    JSONファイルから全てのユニークなエンティティを抽出する

    Args:
        json_file_path (str): JSONファイルのパス

    Returns:
        list: ユニークなエンティティのリスト（出現順）
    """
    entities = OrderedDict()  # 順序を保持する辞書
    item_ids = set()  # item IDを収集するセット

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # まず全てのitem IDを収集
    for item in data:
        if "iid" in item:
            item_ids.add(str(item["iid"]))

    # データ内の全てのトリプレットを走査
    for item in data:
        if "triplets" in item:
            for triplet in item["triplets"]:
                if len(triplet) >= 3:
                    # エンティティは1番目と3番目の要素
                    subject = str(triplet[0])  # 文字列に変換
                    object_entity = str(triplet[2])  # 文字列に変換

                    # item IDと一致しないエンティティのみを追加
                    if subject not in item_ids and subject not in entities:
                        entities[subject] = None
                    if object_entity not in item_ids and object_entity not in entities:
                        entities[object_entity] = None

    return list(entities.keys())


def create_entity_list(entities, output_file_path):
    """
    エンティティリストからentity_list.txtを生成する

    Args:
        entities (list): エンティティのリスト
        output_file_path (str): 出力ファイルのパス
    """
    with open(output_file_path, "w", encoding="utf-8") as f:
        # ヘッダーを書き込み
        f.write("entity_name remap_id\n")

        # 各エンティティにremap_idを割り当てて書き込み
        for i, entity in enumerate(entities):
            f.write(f'"{entity}" {i}\n')


def main():
    """メイン関数"""
    input_file = "canon_kg_normalized.json"
    output_file = "entity_list.txt"

    print(f"JSONファイル '{input_file}' からエンティティを抽出中...")

    try:
        # エンティティを抽出
        entities = extract_entities(input_file)

        print(f"見つかったエンティティ数: {len(entities)}")
        print("エンティティ一覧（最初の20個）:")
        for i, entity in enumerate(entities[:20]):
            print(f"  {i}: {entity}")

        if len(entities) > 20:
            print(f"  ... 他 {len(entities) - 20} 個")

        # entity_list.txtを生成
        create_entity_list(entities, output_file)

        print(f"\n'{output_file}' を正常に生成しました!")

    except FileNotFoundError:
        print(f"エラー: '{input_file}' が見つかりません。")
        print("カレントディレクトリにファイルがあることを確認してください。")
    except json.JSONDecodeError:
        print(f"エラー: '{input_file}' のJSON形式が不正です。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
