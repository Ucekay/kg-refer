# kg-attribute-embedder

知識グラフの属性埋め込みと類似性分析を行うパッケージです。

## 概要

このパッケージは、知識グラフ内の**ユニークなrelation-tailの組み合わせ**に対して以下の処理を行います：

1. ユニークなrelation-tailの組み合わせを抽出（item idには依存しない）
2. 各組み合わせに対して "This business {relation} {tail}" という文を生成
3. Sentence Transformers（multi-qa-distilbert-cos-v1）で埋め込みを生成
4. すべてのユニークなrelation-tail間の類似度を計算
5. 類似するペアを列挙して保存（どのitemがそのrelation-tailを持っているかも記録）
6. 属性値（tail）の出現回数をカウントし、1-2回しか登場しないレアな属性にマークを付与

**注意**: item idは関係なく、relation-tailの組み合わせがユニークなものに対してのみ類似度を計算します。これにより計算量が大幅に削減され、より意味のある類似度分析が可能になります。

## インストール

```bash
cd packages/kg-attribute-embedder
uv sync
```

## 使用方法

### CLIから実行

```bash
kg-attribute-embedder \
    path/to/merged_kg.json \
    --embeddings-output embeddings.npz \
    --similarities-output similarities.json \
    --threshold 0.8 \
    --batch-size 32 \
    --chunk-size 1000
```

#### 引数

- `kg_path`: 知識グラフJSONファイルのパス（必須）
- `--embeddings-output`: 埋め込みの出力ファイルパス（デフォルト: embeddings.npz）
- `--similarities-output`: 類似ペアの出力ファイルパス（デフォルト: similarities.json）
- `--threshold`: 類似度の閾値（デフォルト: 0.8）
- `--batch-size`: 埋め込み生成時のバッチサイズ（デフォルト: 32）
- `--chunk-size`: 類似度計算時のチャンクサイズ。大きなデータセットではメモリ使用量を制御します（デフォルト: 1000）
- `--model`: 使用するSentence Transformerモデル（デフォルト: sentence-transformers/multi-qa-distilbert-cos-v1）

### Pythonから使用

```python
from kg_attribute_embedder import AttributeEmbedder

# インスタンス作成
embedder = AttributeEmbedder(model_name="sentence-transformers/multi-qa-distilbert-cos-v1")

# パイプライン全体を実行
embedder.process_pipeline(
    kg_path="path/to/merged_kg.json",
    embeddings_output_path="embeddings.npz",
    similarities_output_path="similarities.json",
    similarity_threshold=0.8,
    batch_size=32,
)

# または、個別に実行
embedder.load_kg_data("path/to/merged_kg.json")
embedder.count_tails()
embedder.generate_embeddings(batch_size=32)
embedder.save_embeddings("embeddings.npz")
similar_pairs = embedder.compute_similarities(threshold=0.8, output_path="similarities.json")
```

## 入力データフォーマット

入力JSONファイルは以下の形式である必要があります：

```json
[
    {
        "iid": 5,
        "triplets": [
            ["5", "serves", "craft beers"],
            ["5", "serves", "beers"],
            ["5", "offers", "diverse food options"]
        ]
    },
    ...
]
```

## 出力データフォーマット

### 埋め込みファイル（.npz）

NumPy圧縮形式で保存され、以下の配列を含みます：

- `embeddings`: 埋め込みベクトル配列
- `keys`: (relation, tail) のタプル配列（ユニークな組み合わせのみ）
- `sentences`: 生成された文の配列
- `tail_counts`: 各tailの出現回数
- `is_rare`: レアな属性かどうか（1-2回出現）
- `item_ids`: 各relation-tailがどのitemに属しているかのリスト

### 類似ペアファイル（.jsonl）

JSON Lines形式で保存され、1行に1つの類似ペアが記録されます。最小限の情報のみを含むコンパクト形式です：

```jsonl
{"r1":"serves","t1":"craft beers","c1":15,"r2":"serves","t2":"artisan beers","c2":3,"s":0.92}
{"r1":"has atmosphere","t1":"cozy","c1":8,"r2":"has atmosphere","t2":"comfortable","c2":5,"s":0.85}
...
```

**フィールド説明**:
- `r1`, `r2`: relation1, relation2（短縮キー）
- `t1`, `t2`: tail1, tail2（短縮キー）
- `c1`, `c2`: tail1, tail2の出現回数（count）
- `s`: similarity（類似度、小数点以下3桁）

**注意**: 
- JSON Lines形式（.jsonl）で保存されるため、メモリ効率的です
- 各ペアは類似度の降順でソートされています
- 最小限の情報のみを含むコンパクト形式です

## ライセンス

MIT

