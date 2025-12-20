# エンティティ伝播分析スクリプト実行方法

## 概要

このスクリプトは、yelpデータセットのテストデータからランダムに10個のユーザー・アイテムペアを選択し、以下を分析します：

1. アイテムに接続するエンティティ（属性）からの埋め込み伝播をなしにしたスコア
2. 同じ数だけランダムに対象アイテムと接続する他のユーザーとのエッジを削除したスコア
3. 元のスコアとの比較

## 実行コマンド

### 基本的な実行

```bash
cd /home/kimura/repos/kg-refer/packages/kgat

python analyze_entity_propagation.py \
  --model-path trained_model/KGAT/yelp/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.001_pretrain0/model_epoch700.pth \
  --data-dir datasets \
  --data-name yelp \
  --output-csv output/entity_propagation_analysis.csv \
  --n-samples 10 \
  --seed 42
```

### GPUを使用する場合

```bash
python analyze_entity_propagation.py \
  --model-path trained_model/KGAT/yelp/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.001_pretrain0/model_epoch700.pth \
  --data-dir datasets \
  --data-name yelp \
  --device cuda
```

### より多くのサンプルを分析する場合

```bash
python analyze_entity_propagation.py \
  --model-path trained_model/KGAT/yelp/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.001_pretrain0/model_epoch700.pth \
  --data-dir datasets \
  --data-name yelp \
  --n-samples 50 \
  --seed 123
```

## 出力ファイル

デフォルトでは、`packages/kgat/output/entity_propagation_analysis.csv` に結果が保存されます。

### 出力CSV形式

| 列名 | 説明 |
|------|------|
| user_id | ユーザーID（n_entities加算後） |
| item_id | アイテムID |
| original_score | 元のスコア |
| n_connected_entities | アイテムに接続するエンティティの数 |
| score_without_entities | エンティティ伝播なしのスコア |
| score_diff_entities | スコア差（元 - エンティティなし） |
| score_diff_entities_pct | スコア差の割合（%） |
| n_other_users_connected | アイテムに接続する他のユーザー数 |
| n_removed_random_edges | ランダムに削除したエッジの数 |
| score_without_random | ランダム接続削除時のスコア |
| score_diff_random | スコア差（元 - ランダム削除） |
| score_diff_random_pct | スコア差の割合（%） |

## 実行時のログ出力例

```
device: cuda
n_users=30838, n_items=14284, n_entities=38304, n_relations=44
モデルをロード: model_epoch700.pth
10件のユーザー・アイテムペアをサンプリング

処理中: 1/10 - user=45123, item=1234
  元のスコア: 0.456789
  接続エンティティ数: 5
  エンティティ伝播なしスコア: 0.423456
  アイテムに接続する他のユーザー数: 12
  他ユーザー→アイテムエッジ削除スコア: 0.445678

...

結果を保存しました: packages/kgat/output/entity_propagation_analysis.csv
処理件数: 10件

=== 統計情報 ===
平均接続エンティティ数: 4.80
エンティティ伝播なし時の平均スコア差: 0.033333 (7.30%)
ランダム接続削除時の平均スコア差: 0.011111 (2.43%)
```

## 結果の解釈

- **score_diff_entities**: エンティティ→アイテムの伝播がスコアに与える影響
  - 正の値：エンティティからの伝播がスコアを上昇させている
  - 負の値：エンティティからの伝播がスコアを低下させている

- **score_diff_random**: 他のユーザー→アイテムのエッジ削除の影響（ベースライン）
  - 対象アイテムに接続する他のユーザーからのエッジをランダムに削除

- **比較**: `score_diff_entities` と `score_diff_random` を比較することで、エンティティからの伝播が他のユーザーからの伝播と比べて相対的にどれだけ重要かがわかります。

## 注意事項

1. スクリプトは、アイテムに接続するエンティティ（n_items以上のエンティティID）を持つテストペアのみを分析します
2. エンティティに接続していないアイテムのペアはスキップされます
3. A_inからエッジを削除した後、softmaxによる正規化が再適用されます
4. ユーザーIDはn_entitiesが加算された値で表示されます（内部表現）
5. ランダム削除では、対象ユーザー以外でアイテムに接続している他のユーザー→アイテムのエッジを削除します
6. エンティティ削除とランダム削除は、どちらもアイテムへの入力エッジを削除するため、公平な比較が可能です

## トラブルシューティング

### モデルが見つからない

```
FileNotFoundError: モデルファイルが見つかりません
```

→ `--model-path` のパスを確認してください

### データセットが見つからない

```
FileNotFoundError: データセットが見つかりません
```

→ `--data-dir` と `--data-name` の組み合わせを確認してください

### メモリ不足

大規模なモデルやデータセットの場合、メモリが不足する可能性があります。
その場合は `--n-samples` を減らしてください。

