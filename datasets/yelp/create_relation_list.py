#!/usr/bin/env python3
"""
プログラム: create_relation_list.py
説明: canon_kg_normalized.jsonから全てのユニークなリレーションを抽出し、
remap_idを割り当ててrelation_list.txtを生成する
"""

import json
from collections import OrderedDict


def extract_relations(json_file_path):
    """
    JSONファイルから全てのユニークなリレーションを抽出する

    Args:
        json_file_path (str): JSONファイルのパス

    Returns:
        list: ユニークなリレーションのリスト（出現順）
    """
    relations = OrderedDict()  # 順序を保持する辞書

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # データ内の全てのトリプレットを走査
    for item in data:
        if "triplets" in item:
            for triplet in item["triplets"]:
                if len(triplet) >= 2:
                    relation = triplet[1]  # リレーションは2番目の要素
                    if relation not in relations:
                        relations[relation] = None

    return list(relations.keys())


def create_relation_list(relations, output_file_path):
    """
    リレーションリストからrelation_list.txtを生成する

    Args:
        relations (list): リレーションのリスト
        output_file_path (str): 出力ファイルのパス
    """
    with open(output_file_path, "w", encoding="utf-8") as f:
        # ヘッダーを書き込み
        f.write("relation_name remap_id\n")

        # 各リレーションにremap_idを割り当てて書き込み
        for i, relation in enumerate(relations):
            f.write(f'"{relation}" {i}\n')


def main():
    """メイン関数"""
    input_file = "canon_kg_normalized.json"
    output_file = "relation_list.txt"

    print(f"JSONファイル '{input_file}' からリレーションを抽出中...")

    try:
        # リレーションを抽出
        relations = extract_relations(input_file)

        print(f"見つかったリレーション数: {len(relations)}")
        print("リレーション一覧:")
        for i, relation in enumerate(relations):
            print(f"  {i}: {relation}")

        # relation_list.txtを生成
        create_relation_list(relations, output_file)

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
