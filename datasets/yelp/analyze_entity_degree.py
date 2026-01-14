#!/usr/bin/env python3
"""
アイテムとユーザー以外のエンティティの次数を調べてヒストグラムに表示するスクリプト
"""

import matplotlib.pyplot as plt
from collections import defaultdict

def load_entity_ids(entity_list_file):
    """entity_list.txtからエンティティIDのリストを読み込む（アイテムとユーザー以外のエンティティ）"""
    entity_ids = set()
    with open(entity_list_file, 'r', encoding='utf-8') as f:
        next(f)  # ヘッダーをスキップ
        for line in f:
            line = line.strip()
            if not line:
                continue
            # エンティティ名にスペースが含まれる可能性があるため、最後のスペースで分割
            parts = line.rsplit(' ', 1)
            if len(parts) >= 2:
                try:
                    entity_id = int(parts[1])
                    entity_ids.add(entity_id)
                except ValueError:
                    continue
    return entity_ids

def calculate_entity_degrees(kg_file, entity_ids):
    """
    kg_final.txtからエンティティの次数を計算する
    
    Args:
        kg_file: kg_final.txtのパス（形式: head relation tail）
        entity_ids: entity_list.txtに含まれるエンティティIDのセット
    
    Returns:
        dict: エンティティID -> 次数
    """
    degree = defaultdict(int)
    
    with open(kg_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            try:
                head = int(parts[0])
                tail = int(parts[2])
                
                # headが非アイテムエンティティの場合
                if head in entity_ids:
                    degree[head] += 1
                
                # tailが非アイテムエンティティの場合
                if tail in entity_ids:
                    degree[tail] += 1
                    
            except ValueError:
                continue
    
    return degree

def main():
    kg_file = "kg_final.txt"
    entity_list_file = "entity_list.txt"
    
    print("エンティティリストを読み込み中...")
    entity_ids = load_entity_ids(entity_list_file)
    print(f"エンティティ数（アイテムとユーザー以外）: {len(entity_ids)}")
    
    print("次数を計算中...")
    degrees = calculate_entity_degrees(kg_file, entity_ids)
    
    # 次数の分布を計算
    degree_distribution = defaultdict(int)
    for entity_id, deg in degrees.items():
        degree_distribution[deg] += 1
    
    # 次数1のエンティティ数を表示
    degree_one_count = degree_distribution.get(1, 0)
    print(f"\n次数が1のエンティティ数: {degree_one_count}")
    
    # 次数20までのヒストグラム用データを準備
    max_degree_for_hist = 20
    degrees_for_hist = []
    counts_for_hist = []
    
    for deg in range(1, max_degree_for_hist + 1):
        count = degree_distribution.get(deg, 0)
        degrees_for_hist.append(deg)
        counts_for_hist.append(count)
    
    # ヒストグラムを表示
    plt.figure(figsize=(12, 6))
    plt.bar(degrees_for_hist, counts_for_hist, color='steelblue', edgecolor='black')
    plt.xlabel('次数', fontsize=12)
    plt.ylabel('エンティティ数', fontsize=12)
    plt.title('アイテムとユーザー以外のエンティティの次数分布（次数1-20）', fontsize=14)
    plt.xticks(range(1, max_degree_for_hist + 1))
    plt.grid(axis='y', alpha=0.3)
    
    # 各バーの上に数値を表示
    for deg, count in zip(degrees_for_hist, counts_for_hist):
        if count > 0:
            plt.text(deg, count, str(count), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('entity_degree_histogram.png', dpi=150, bbox_inches='tight')
    print(f"\nヒストグラムを 'entity_degree_histogram.png' に保存しました。")
    
    # 統計情報を表示
    print("\n=== 次数分布（次数1-20）===")
    print(f"{'次数':<6} {'エンティティ数':<15}")
    print("-" * 25)
    for deg, count in zip(degrees_for_hist, counts_for_hist):
        print(f"{deg:<6} {count:<15}")
    
    # 全次数の統計
    all_degrees = list(degrees.values())
    if all_degrees:
        print(f"\n=== 全エンティティの次数統計 ===")
        print(f"平均次数: {sum(all_degrees) / len(all_degrees):.2f}")
        print(f"最大次数: {max(all_degrees)}")
        print(f"最小次数: {min(all_degrees)}")
        print(f"次数0のエンティティ数: {len(entity_ids) - len(degrees)}")

if __name__ == "__main__":
    main()
