# Semantic Similarity Retriever

意味的類似度に基づいて類似ノードを検索するパッケージです。G-Referのプロファイルベースの類似ノード検索機能をkg-referに実装しました。

## 概要

このパッケージは、ユーザーとアイテムのプロファイル（テキスト説明）から意味的埋め込みを計算し、コサイン類似度に基づいて類似するユーザーやアイテムを検索します。

## 機能

- **テキストベースの埋め込み生成**: Sentence-BERTなどのテキストエンコーダーを使用してユーザー・アイテムのプロファイルから埋め込みベクトルを生成
- **類似ノード検索**: コサイン類似度に基づいて類似するユーザーとアイテムをTop-K検索
- **インタラクショングラフ構築**: 学習データから誰がどのアイテムを購入したかのグラフを構築し、より関連性の高い候補から検索

## インストール

```bash
cd kg-refer
uv sync
```

## 使用方法

### 基本的な使用

```bash
cd kg-refer
uv run semantic-similarity-retriever
```

### カスタム設定での実行

```bash
uv run semantic-similarity-retriever \
  --user-profile-path datasets/yelp/user_profile.json \
  --item-profile-path datasets/yelp/item_profile.json \
  --train-interactions-path datasets/yelp/total_train.txt \
  --eval-interactions-path datasets/yelp/interactions_yelp_evaluation.txt \
  --output-file packages/semantic-similarity-retriever/output/semantic_similarity_results.json \
  --text-encoder sentence-transformers/all-MiniLM-L6-v2 \
  --topk 5 \
  --pruning-score 0.0  # デフォルト値（0以上のスコアを持つノードを返す）
```

### パラメータ説明

- `--user-profile-path`: ユーザープロファイルJSONLファイルのパス
- `--item-profile-path`: アイテムプロファイルJSONLファイルのパス
- `--train-interactions-path`: 学習用インタラクションファイル（グラフ構築に使用）
- `--eval-interactions-path`: 評価用インタラクションファイル（類似ノードを検索する対象）
- `--output-file`: 結果を保存するJSONファイルのパス（デフォルト: `packages/semantic-similarity-retriever/output/semantic_similarity_results.json`）
- `--text-encoder`: 使用するテキストエンコーダーモデル名
- `--topk`: 取得する類似ノードの最大数
- `--pruning-score`: 類似度スコアの閾値（デフォルト: 0.0、この値以上のスコアを持つノードのみを返す）

## 出力形式

結果はJSON形式で出力されます：

```json
[
  {
    "user_id": 4991,
    "item_id": 11821,
    "similar_users": [
      [87, 0.7793341875076294],
      [7075, 0.700798749923706],
      [5582, 0.6903373599052429]
    ],
    "similar_items": [
      [11692, 0.684956431388855],
      [2969, 0.6254342794418335],
      [7198, 0.5033300518989563]
    ]
  }
]
```

各エントリには以下が含まれます：
- `user_id`: ユーザーID
- `item_id`: アイテムID  
- `similar_users`: 類似ユーザーのリスト（[ユーザーID, 類似度スコア]のタプル）
- `similar_items`: 類似アイテムのリスト（[アイテムID, 類似度スコア]のタプル）

## 実装詳細

### G-Referとの違い

G-Referの`dense_retriever.py`の実装を参考にしていますが、以下の点で異なります：

1. **PyTorch Geometricへの依存なし**: G-Referは事前に計算されたPyGデータを使用していますが、このパッケージはプロファイルJSONファイルから直接埋め込みを計算します
2. **動的な埋め込み生成**: Sentence-Transformersを使用して実行時に埋め込みを生成します
3. **柔軟なテキストエンコーダー**: 任意のSentence-Transformersモデルを指定可能です

### アルゴリズム

1. ユーザーとアイテムのプロファイルテキストをロード
2. テキストエンコーダー（Sentence-BERT）で全ての埋め込みを計算
3. 学習用インタラクションからグラフを構築
4. 評価用の各ユーザー・アイテムペアに対して：
   - そのアイテムを購入した他のユーザーとの類似度を計算
   - そのユーザーが購入した他のアイテムとの類似度を計算
   - Top-Kの類似ノードを返す

## 例

Yelpデータセットで実行した結果：

```
=== 統計情報 ===
処理したインタラクション数: 3000
検索された類似ユーザー総数: 14997
検索された類似アイテム総数: 14999
平均類似ユーザー数/インタラクション: 5.00
平均類似アイテム数/インタラクション: 5.00

=== 例 (最初の3件) ===
例 1: ユーザー 4991 - アイテム 11821
  類似ユーザー (top 3): [(87, 0.779), (7075, 0.701), (5582, 0.690)]
  類似アイテム (top 3): [(11692, 0.685), (2969, 0.625), (7198, 0.503)]
```

## ライセンス

MITライセンス
