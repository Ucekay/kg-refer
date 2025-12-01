import csv
import json

# explanation.json から (uid, iid) のペアを読み込む
explanation_pairs = set()
with open('/home/kimura/repos/kg-refer/datasets/yelp/explanation.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        explanation_pairs.add((data['uid'], data['iid']))

print(f"explanation.json の組み合わせ数: {len(explanation_pairs)}")

# users_with_top20_hits.csv を読み込み、フィルタリング
filtered_rows = []
with open('/home/kimura/repos/kg-refer/packages/kgat/output/users_with_top20_hits.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        user_id = int(row['user_id'])
        hit_items = [int(x) for x in row['hit_items'].split(';')]
        hit_ranks = [int(x) for x in row['hit_ranks'].split(';')]
        
        # explanation.json に存在する組み合わせだけをフィルタリング
        filtered_items = []
        filtered_ranks = []
        for item, rank in zip(hit_items, hit_ranks):
            if (user_id, item) in explanation_pairs:
                filtered_items.append(item)
                filtered_ranks.append(rank)
        
        if filtered_items:
            filtered_rows.append({
                'user_id': user_id,
                'n_hits': len(filtered_items),
                'n_test_items': row['n_test_items'],
                'hit_items': ';'.join(map(str, filtered_items)),
                'hit_ranks': ';'.join(map(str, filtered_ranks))
            })

print(f"フィルタリング後のユーザー数: {len(filtered_rows)}")
total_hits = sum(row['n_hits'] for row in filtered_rows)
print(f"フィルタリング後の総ヒット数: {total_hits}")

# 結果を保存
output_path = '/home/kimura/repos/kg-refer/packages/kgat/output/users_with_top20_hits_filtered.csv'
with open(output_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['user_id', 'n_hits', 'n_test_items', 'hit_items', 'hit_ranks'])
    writer.writeheader()
    writer.writerows(filtered_rows)

print(f"結果を保存しました: {output_path}")

