# user-item-attribute-scorer

ユーザーの集約埋め込みとアイテムに接続する属性エンティティの集約埋め込みの内積を計算するパッケージです。

## 概要

このパッケージは、KGATモデルを使用して以下の処理を行います：

1. ユーザー・アイテムペアのリストを読み込む
2. 学習済みKGATモデルからユーザーの集約埋め込みを取得
3. 各アイテムに接続する属性エンティティ（ユーザーでもアイテムでもない）の埋め込みを取得
4. 各属性エンティティごとにユーザー埋め込みとの内積を計算
5. 結果をCSVファイルに保存

## インストール

```bash
cd packages/user-item-attribute-scorer
uv sync
```

## 使用方法

### 基本的な使用方法

```bash
user-item-attribute-scorer \
    --interactions-file datasets/yelp/interactions_yelp_evaluation.txt \
    --model-path packages/kgat/trained_model/KGAT/yelp/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.001_pretrain0/model_epoch60.pth \
    --data-dir datasets/ \
    --data-name yelp \
    --output-file packages/user-item-attribute-scorer/output/user_item_attribute_scores.csv
```

### "located in"リレーションを除外する場合

"located in"のリレーションを無視して学習したモデル（`model_epoch50.pth`）に対応する場合：

```bash
user-item-attribute-scorer \
    --interactions-file datasets/yelp/interactions_yelp_evaluation.txt \
    --model-path packages/kgat/trained_model/KGAT/yelp/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.001_pretrain0/model_epoch50.pth \
    --data-dir datasets/ \
    --data-name yelp \
    --output-file packages/user-item-attribute-scorer/output/user_item_attribute_scores_no_location.csv \
    --exclude-relations 5
```

**注意**: `--exclude-relations 5` は、`kg_final.txt`のリレーションID 5（"located in"のremap_id）を除外します。

### 引数

- `--interactions-file`（必須）: ユーザー・アイテムペアのCSVファイル（`uid,iid`形式）
- `--model-path`（必須）: 学習済みKGATモデルのパス（`.pth`ファイル）
- `--output-file`（必須）: 出力CSVファイルのパス
- `--data-dir`（オプション）: データディレクトリ（デフォルト: `datasets/`）
- `--data-name`（オプション）: データセット名（デフォルト: `yelp`）
- `--kg-file`（オプション）: `kg_final.txt`のパス（デフォルト: `data_dir/data_name/kg_final.txt`）
- `--exclude-relations`（オプション）: 除外するリレーションIDのカンマ区切りリスト（`kg_final.txt`のリレーションID = `relation_list.txt`のremap_id）

### リレーションIDの対応

`kg_final.txt`のリレーションIDは、`relation_list.txt`の`remap_id`と同じです：

- 例: "located in"の`remap_id`は5 → `kg_final.txt`でのIDも5

主なリレーション（`relation_list.txt`より）:
- 0: "serves"
- 1: "offers"
- 2: "has atmosphere"
- 5: "located in"
- 6: "is a"

### 出力形式

出力CSVファイルには以下の列が含まれます：

- `uid`: ユーザーID
- `iid`: アイテムID
- `attribute_id`: 属性エンティティID
- `attribute_name`: 属性エンティティ名
- `score`: ユーザー埋め込みと属性エンティティ埋め込みの内積スコア

各ユーザー・アイテムペアについて、接続する各属性エンティティとのスコアが個別に記録されます。スコアは各ユーザー・アイテムペア内で降順にソートされます。
