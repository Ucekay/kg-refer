# エンティティ伝播分析スクリプト

## 概要

このスクリプトは、KGATモデルにおいてアイテムに接続するエンティティからの埋め込み伝播がスコアに与える影響を分析します。

## 機能

1. テストデータからランダムに10個のユーザー・アイテムペアをサンプリング
2. 各ペアについて以下を計算：
   - 元のスコア
   - アイテムに接続するエンティティからの埋め込み伝播をなしにしたスコア（エンティティ→アイテムのエッジを削除）
   - 無視したエンティティと同じ数だけランダムに対象アイテムと接続する他のユーザーとのエッジを削除したスコア（他のユーザー→アイテムのエッジを削除）
3. 結果をCSVファイルに保存

## 実行方法

### 基本的な使い方

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

### オプション

- `--model-path`: 学習済みモデル(.pth)のパス（必須）
- `--data-dir`: データセットディレクトリ（デフォルト: `packages/kgat/datasets`）
- `--data-name`: データセット名（デフォルト: `yelp`）
- `--output-csv`: 出力CSVファイルのパス（デフォルト: `packages/kgat/output/entity_propagation_analysis.csv`）
- `--n-samples`: サンプリングするユーザー・アイテムペアの数（デフォルト: 10）
- `--seed`: ランダムシード（デフォルト: 42）
- `--device`: 使用デバイス（未指定の場合は自動選択）
- `--embed-dim`: 埋め込み次元（デフォルト: 64）
- `--relation-dim`: 関係埋め込み次元（デフォルト: 64）

### GPUを使用する場合

```bash
python analyze_entity_propagation.py \
  --model-path trained_model/KGAT/yelp/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.001_pretrain0/model_epoch700.pth \
  --data-dir datasets \
  --data-name yelp \
  --device cuda
```

### サンプル数を変更する場合

```bash
python analyze_entity_propagation.py \
  --model-path trained_model/KGAT/yelp/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.001_pretrain0/model_epoch700.pth \
  --data-dir datasets \
  --data-name yelp \
  --n-samples 50
```

## 出力ファイル形式

出力CSVファイルには以下の列が含まれます：

| 列名 | 説明 |
|------|------|
| `user_id` | ユーザーID |
| `item_id` | アイテムID |
| `original_score` | 元のスコア |
| `n_connected_entities` | アイテムに接続するエンティティの数 |
| `score_without_entities` | エンティティ伝播なしのスコア |
| `score_diff_entities` | エンティティ伝播なし時のスコア差 |
| `score_diff_entities_pct` | エンティティ伝播なし時のスコア差（%） |
| `n_other_users_connected` | アイテムに接続する他のユーザー数 |
| `n_removed_random_edges` | ランダムに削除したエッジの数 |
| `score_without_random` | ランダム接続削除時のスコア |
| `score_diff_random` | ランダム接続削除時のスコア差 |
| `score_diff_random_pct` | ランダム接続削除時のスコア差（%） |

## 分析結果の読み方

- `score_diff_entities`: 正の値はエンティティ→アイテムの伝播がスコアを上げていることを示します
- `score_diff_random`: 比較用のベースライン。他のユーザー→アイテムのエッジをランダムに削除した影響
- 2つの差を比較することで、エンティティからの伝播が他のユーザーからの伝播と比べてどれだけ重要かがわかります
- どちらもアイテムへの入力エッジを削除するため、公平な比較が可能です

## 注意事項

- スクリプトは、アイテムがエンティティに接続しているテストペアのみを分析対象とします
- 接続エンティティの数がアイテムに接続する他のユーザー数より多い場合、すべての他のユーザー→アイテムエッジが削除されます
- スコア計算時にA_inの正規化（softmax）が再適用されます
- ランダム削除では、対象ユーザー以外でアイテムに接続している他のユーザーのエッジを削除します

