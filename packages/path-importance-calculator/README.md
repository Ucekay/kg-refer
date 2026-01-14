# Path Importance Calculator

学習済みKGATモデルを使用して、ユーザー・アイテムペア間の3ホップパスを探索し、A_inに基づいてパスの重要度を計算するツールです。

## 機能

- 学習済みKGATモデルのロード
- ユーザー・アイテムペア間の最大3ホップまでのパス探索
- A_inのアテンション重みに基づくパスの重要度計算
- 各パス上のリレーション重みの積による重要度スコア算出
- JSON/CSV形式での結果保存

## インストール

```bash
cd packages/path-importance-calculator
uv sync
```

## 使用方法

### コマンドライン

```bash
path-importance-calculator \
  --model-path packages/kgat/trained_model/KGAT/yelp/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.001_pretrain0/model_epoch50.pth \
  --interaction-file datasets/yelp/interactions_yelp_evaluation.txt \
  --output-file packages/path-importance-calculator/output/path_importance_scores.json \
  --data-dir packages/kgat/datasets/ \
  --data-name yelp \
  --max-hops 3 \
  --min-hops 3 \
  --ignore-relations "[5]" \
  --entity-list-file datasets/yelp/entity_list.txt \
  --relation-list-file datasets/yelp/relation_list.txt \
  --output-format json
```

### パラメータ

- `--model-path`: 学習済みKGATモデルのパス（必須）
- `--interaction-file`: ユーザー・アイテムペアのファイルパス（必須）
- `--output-file`: 出力ファイルのパス（必須）
- `--data-dir`: データディレクトリ（デフォルト: packages/kgat/datasets/）
- `--data-name`: データセット名（デフォルト: yelp）
- `--max-hops`: 最大ホップ数（デフォルト: 3）
- `--output-format`: 出力形式（json または csv、デフォルト: json）
- `--embed-dim`: 埋め込み次元（デフォルト: 64）
- `--relation-dim`: リレーション次元（デフォルト: 64）
- `--laplacian-type`: ラプラシアンタイプ（デフォルト: random-walk）
- `--aggregation-type`: 集約タイプ（デフォルト: bi-interaction）
- `--conv-dim-list`: 畳み込み次元リスト（デフォルト: [64,32,16]）

### Pythonスクリプトから使用

```python
import logging
from kgat.config import KGATConfig
from path_importance_calculator import PathImportanceCalculator, KGATModelLoader

# ロガーの設定
logger = logging.getLogger(__name__)

# KGAT設定
kgat_config = KGATConfig(
    data_dir="packages/kgat/datasets/",
    data_name="yelp",
    embed_dim=64,
    relation_dim=64,
    use_pretrain=0,
)

# モデルローダー
model_loader = KGATModelLoader(
    model_path="packages/kgat/trained_model/KGAT/yelp/.../model_epoch50.pth",
    config=kgat_config,
    logger=logger,
)

# 計算器
calculator = PathImportanceCalculator(
    model_loader=model_loader,
    max_hops=3,
)

# 計算実行
results = calculator.calculate_for_pairs("datasets/yelp/interactions_yelp_evaluation.txt")

# 結果保存
calculator.save_results(results, "packages/path-importance-calculator/output/path_importance_scores.json", format="json")
```

## 出力形式

### JSON形式

```json
[
  {
    "user_id": 4991,
    "item_id": 11821,
    "path_nodes": [20935, 14123, 10456, 11821],
    "path_relations": [5, 12, 3],
    "importance_score": 0.00234,
    "path_length": 3
  },
  ...
]
```

### CSV形式

| user_id | item_id | path_nodes | path_relations | importance_score | path_length |
|---------|---------|------------|----------------|------------------|-------------|
| 4991 | 11821 | 20935,14123,10456,11821 | 5,12,3 | 0.00234 | 3 |

## アルゴリズム

1. ユーザーIDをエンティティIDに変換（user_entity_id = n_entities + user_id）
2. BFSを使用してユーザーエンティティからアイテムエンティティまでの全パスを探索（最大3ホップ）
3. 各パスに対して、A_inから各エッジ（head, relation, tail）の重みを抽出
4. パスの重要度 = ∏(各エッジの重み)
5. 結果をユーザーID、アイテムID、パス、重要度スコアと共に保存

## 注意事項

- A_inは学習済みモデルに保存されているスパーステンソルです
- ユーザーエンティティIDはn_entitiesから始まります
- パス探索ではサイクルを回避します（同じノードを2回訪問しない）
- 重みが0のエッジがある場合、パス全体の重要度は0になります
