#!/usr/bin/env python3
"""
プログラム: create_kg_final.py
説明: canon_kg_normalized.jsonとentity_list.txtから
h r t 形式（head, relation, tailのID）でkg_final.txtを生成する
"""

import json

import pandas as pd


def load_entity_mapping(entity_file_path):
    """
    entity_list.txtからエンティティ名→IDのマッピングを読み込む

    Args:
        entity_file_path (str): entity_list.txtのパス

    Returns:
        dict: エンティティ名→IDのマッピング辞書
    """
    df = pd.read_csv(entity_file_path, delimiter=" ")
    # entity_nameからダブルクォーテーションを除去
    df["entity_name"] = df["entity_name"].str.strip('"')
    return dict(zip(df["entity_name"], df["remap_id"]))


def load_relation_mapping(relation_file_path):
    """
    relation_list.txtからrelation名→IDのマッピングを読み込む

    Args:
        relation_file_path (str): relation_list.txtのパス

    Returns:
        dict: relation名→IDのマッピング辞書
    """
    df = pd.read_csv(relation_file_path, delimiter=" ")
    # relation_nameからダブルクォーテーションを除去
    df["relation_name"] = df["relation_name"].str.strip('"')
    return dict(zip(df["relation_name"], df["remap_id"]))


def create_kg_final(json_file_path, entity_mapping, relation_mapping, output_file_path):
    """
    h r t 形式でkg_final.txtを生成する

    Args:
        json_file_path (str): JSONファイルのパス
        entity_mapping (dict): エンティティ名→IDのマッピング
        relation_mapping (dict): relation名→IDのマッピング
        output_file_path (str): 出力ファイルのパス
    """
    with open(output_file_path, "w", encoding="utf-8") as f:
        with open(json_file_path, "r", encoding="utf-8") as json_f:
            data = json.load(json_f)

        for item in data:
            if "triplets" in item:
                for triplet in item["triplets"]:
                    if len(triplet) >= 3:
                        # triplet: [item_id, relation, entity]
                        head = str(triplet[0])  # item IDはそのまま使用
                        relation = str(triplet[1])
                        tail_entity = str(triplet[2])

                        # relation IDを取得
                        if relation not in relation_mapping:
                            continue  # 見つからない場合はスキップ
                        relation_id = relation_mapping[relation]

                        # tail entity IDを取得
                        if tail_entity not in entity_mapping:
                            continue  # 見つからない場合はスキップ
                        tail_id = entity_mapping[tail_entity]

                        # h r t 形式で書き込み
                        f.write(f"{head} {relation_id} {tail_id}\n")


def main():
    """メイン関数"""
    json_file = "canon_kg_normalized.json"
    entity_file = "entity_list.txt"
    relation_file = "relation_list.txt"
    output_file = "kg_final.txt"

    print("エンティティマッピングを読み込み中...")
    try:
        entity_mapping = load_entity_mapping(entity_file)
        print(f"エンティティ数: {len(entity_mapping)}")
    except FileNotFoundError:
        print(f"エラー: '{entity_file}' が見つかりません。")
        return

    print("リレーションマッピングを読み込み中...")
    try:
        relation_mapping = load_relation_mapping(relation_file)
        print(f"リレーション数: {len(relation_mapping)}")
    except FileNotFoundError:
        print(f"エラー: '{relation_file}' が見つかりません。")
        return

    print(f"kg_final.txtを生成中...")
    try:
        create_kg_final(json_file, entity_mapping, relation_mapping, output_file)
        print(f"'{output_file}' を正常に生成しました!")
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
