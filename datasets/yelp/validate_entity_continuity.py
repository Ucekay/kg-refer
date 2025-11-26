#!/usr/bin/env python3
"""
kg_final.txtのエンティティIDの連続性を検証し、欠落IDを特定する
"""

import pandas as pd


def validate_kg_final(kg_file, entity_file):
    """
    kg_final.txtのエンティティIDを検証
    
    Args:
        kg_file: kg_final.txtのパス
        entity_file: entity_list.txtのパス
    """
    print("=" * 80)
    print("kg_final.txt エンティティID連続性検証")
    print("=" * 80)
    
    # 1. kg_final.txtからエンティティIDを収集
    print("\n[1] kg_final.txtを読み込み中...")
    kg_data = pd.read_csv(kg_file, sep=" ", names=["h", "r", "t"], dtype=int)
    
    all_entities = set()
    all_entities.update(kg_data["h"].unique())
    all_entities.update(kg_data["t"].unique())
    
    entities_sorted = sorted(all_entities)
    n_entities = len(entities_sorted)
    min_id = min(entities_sorted)
    max_id = max(entities_sorted)
    
    print(f"  エンティティ総数: {n_entities}")
    print(f"  ID範囲: [{min_id}, {max_id}]")
    
    # 2. entity_list.txtを読み込み
    print("\n[2] entity_list.txtを読み込み中...")
    entity_df = pd.read_csv(entity_file, sep=" ")
    entity_df["entity_name"] = entity_df["entity_name"].str.strip('"')
    
    entity_ids_from_list = set(entity_df["remap_id"])
    min_entity_list_id = entity_df["remap_id"].min()
    max_entity_list_id = entity_df["remap_id"].max()
    n_entity_list = len(entity_ids_from_list)
    
    print(f"  非アイテムエンティティ数: {n_entity_list}")
    print(f"  ID範囲: [{min_entity_list_id}, {max_entity_list_id}]")
    
    # 3. 連続性をチェック
    print("\n[3] ID連続性をチェック中...")
    expected_ids = set(range(max_id + 1))
    missing_ids = expected_ids - set(entities_sorted)
    extra_ids = set(entities_sorted) - expected_ids
    
    if not missing_ids and not extra_ids:
        print("  ✓ すべてのIDは0から連続しています")
    else:
        if missing_ids:
            print(f"  ✗ 欠落ID数: {len(missing_ids)}")
            print(f"    欠落ID: {sorted(missing_ids)}")
            
            # 欠落IDがアイテム範囲か非アイテム範囲かを判定
            for missing_id in sorted(missing_ids):
                if missing_id < min_entity_list_id:
                    print(f"      ID {missing_id}: アイテム範囲内の欠落")
                else:
                    entity_index = missing_id - min_entity_list_id
                    print(f"      ID {missing_id}: 非アイテムエンティティ範囲内の欠落")
                    print(f"                      (entity_listの{entity_index}番目)")
        
        if extra_ids:
            print(f"  ✗ 余分なID数: {len(extra_ids)}")
            print(f"    余分なID: {sorted(extra_ids)}")
    
    # 4. アイテムIDの範囲を推定
    print("\n[4] アイテムID範囲を推定中...")
    print(f"  entity_listの最小ID: {min_entity_list_id}")
    print(f"  推定: アイテムID = [0, {min_entity_list_id - 1}]")
    print(f"  推定: アイテム数 = {min_entity_list_id}")
    
    item_ids = [e for e in entities_sorted if e < min_entity_list_id]
    n_items = len(item_ids)
    
    if item_ids:
        print(f"  実際のアイテムID: [{min(item_ids)}, {max(item_ids)}]")
        print(f"  実際のアイテム数: {n_items}")
        
        # アイテムIDの連続性チェック
        expected_item_ids = set(range(min_entity_list_id))
        missing_item_ids = expected_item_ids - set(item_ids)
        if missing_item_ids:
            print(f"  ✗ 欠落しているアイテムID: {len(missing_item_ids)}個")
            print(f"    例（最初の20個）: {sorted(list(missing_item_ids))[:20]}")
        else:
            print(f"  ✓ アイテムIDは連続しています")
    
    # 5. kg_finalに存在するがentity_listにないエンティティ
    print("\n[5] entity_listとの整合性チェック...")
    non_item_entities_in_kg = set([e for e in entities_sorted if e >= min_entity_list_id])
    missing_in_entity_list = non_item_entities_in_kg - entity_ids_from_list
    missing_in_kg = entity_ids_from_list - non_item_entities_in_kg
    
    if missing_in_entity_list:
        print(f"  ✗ kg_finalに存在するがentity_listにないID: {len(missing_in_entity_list)}個")
        print(f"    ID: {sorted(list(missing_in_entity_list))[:20]}")
    else:
        print(f"  ✓ kg_finalの非アイテムエンティティはすべてentity_listに存在")
    
    if missing_in_kg:
        print(f"  ⚠ entity_listに存在するがkg_finalにないID: {len(missing_in_kg)}個")
        print(f"    ID（最初の20個）: {sorted(list(missing_in_kg))[:20]}")
        
        # 対応するエンティティ名を表示
        print("\n    対応するエンティティ名:")
        for eid in sorted(list(missing_in_kg))[:10]:
            entity_name = entity_df[entity_df["remap_id"] == eid]["entity_name"].values
            if len(entity_name) > 0:
                print(f"      ID {eid}: {entity_name[0]}")
    else:
        print(f"  ✓ entity_listのすべてのエンティティがkg_finalに存在")
    
    # 6. 結論
    print("\n" + "=" * 80)
    print("結論")
    print("=" * 80)
    
    has_issues = bool(missing_ids or extra_ids or missing_item_ids or missing_in_entity_list)
    
    if has_issues:
        print("\n⚠ データに問題があります:")
        if missing_ids:
            print(f"  • エンティティIDに{len(missing_ids)}個の欠落があります")
        if missing_item_ids:
            print(f"  • アイテムIDに{len(missing_item_ids)}個の欠落があります")
        if missing_in_entity_list:
            print(f"  • entity_listにない非アイテムエンティティが{len(missing_in_entity_list)}個あります")
        
        print("\n推奨される対応:")
        print("  1. canon_kg_normalized.jsonとmerged_kg_2.jsonが同じか確認")
        print("  2. entity_listを再生成")
        print("  3. kg_finalを再生成")
    else:
        print("\n✓ データは正常です")
        print("  すべてのIDは連続しており、entity_listとkg_finalは整合しています")


def main():
    kg_file = "packages/kgat/datasets/yelp/kg_final.txt"
    entity_file = "packages/kgat/datasets/yelp/entity_list.txt"
    
    try:
        validate_kg_final(kg_file, entity_file)
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません - {e}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


