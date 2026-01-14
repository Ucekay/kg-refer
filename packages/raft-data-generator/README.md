# RAFT Data Generator

G-Referのraft_dataと同じ形式のfine-tuning用データを生成するパッケージです。

## 概要

このパッケージは、以下の既存パッケージを統合してRAFT形式のデータを生成します：

- **explainable-rec-prompt-generator**: 説明プロンプトの生成
- **explanation-generator**: 説明の生成（LLM使用）
- **path-importance-calculator**: パス重要度スコアの取得
- **semantic-similarity-retriever**: 類似ノードの検索と類似度スコアの計算
- **user-item-attribute-scorer**: ユーザー・属性親和度の取得

## インストール

プロジェクトルートから:

```bash
uv sync
```

## 使い方

### 1. 新規データ生成

プロジェクトルートから実行します:

```bash
uv run raft-data-generator \
    --interactions datasets/yelp/interactions_yelp_evaluation.txt \
    --user-profiles datasets/yelp/user_profile.json \
    --item-profiles datasets/yelp/item_profile.json \
    --attribute-scores packages/user-item-attribute-scorer/output/user_item_attribute_scores_no_location.csv \
    --path-scores packages/path-importance-calculator/output/path_importance_scores_with_names.json \
    --similarity packages/semantic-similarity-retriever/output/semantic_similarity_results.json \
    --output-dir packages/raft-data-generator/output/yelp \
    --dataset-name yelp \
    --train-ratio 0.8 \
    --eval-ratio 0.1 \
    --explanation-provider openai \
    --explanation-model gpt-3.5-turbo \
    --explanation-max-tokens 150 \
    --explanation-temperature 0.7
```

### 2. インタラクションファイルから直接生成（推奨）

インタラクションファイルから類似ノード、パス重要度、ユーザー・属性親和度を計算してRAFTデータを生成します:

```bash
uv run raft-data-generator from-interactions \
    --train-interactions datasets/yelp/interactions_yelp_train.txt \
    --eval-interactions datasets/yelp/interactions_yelp_eval.txt \
    --test-interactions datasets/yelp/interactions_yelp_test.txt \
    --user-profiles datasets/yelp/user_profile.json \
    --item-profiles datasets/yelp/item_profile.json \
    --model-path packages/kgat/trained_model/KGAT/yelp/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.001_pretrain0/model_epoch50.pth \
    --output-dir packages/raft-data-generator/output/yelp \
    --data-dir datasets/ \
    --data-name yelp \
    --entity-list-file datasets/yelp/entity_list.txt \
    --relation-list-file datasets/yelp/relation_list.txt \
    --ignore-relations "[5]" \
    --exclude-relations "[5]" \
    --explanation-provider openai \
    --explanation-model gpt-3.5-turbo
```

このコマンドは以下の処理を自動的に実行します：
1. **類似ノードの計算**: 訓練データと評価データから類似ユーザー・アイテムを検索
2. **パス重要度の計算**: ユーザー・アイテムペア間のパスを探索して重要度を計算
3. **ユーザー・属性親和度の計算**: ユーザーとアイテム属性の親和度を計算
4. **RAFTデータの生成**: 上記の結果を使ってプロンプトと説明を生成

### 3. 既存データのプロンプト置き換え（計算付き）

G-Referのraft_dataから`uid`, `iid`, `chosen`, `similarity_score`を取得し、類似ノード・パス・属性スコアを計算してから`prompt`だけをオリジナルのものに置き換えます:

```bash
uv run raft-data-generator replace-prompts-with-computation \
    --input-dir /home/kimura/repos/G-Refer/raft_data/yelp \
    --train-interactions datasets/yelp/interactions_yelp_train.txt \
    --user-profiles datasets/yelp/user_profile.json \
    --item-profiles datasets/yelp/item_profile.json \
    --model-path packages/kgat/trained_model/KGAT/yelp/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.001_pretrain0/model_epoch50.pth \
    --output-dir packages/raft-data-generator/output/yelp \
    --data-dir datasets/ \
    --data-name yelp \
    --entity-list-file datasets/yelp/entity_list.txt \
    --relation-list-file datasets/yelp/relation_list.txt \
    --ignore-relations "[5]" \
    --exclude-relations "[5]" \
    --min-attribute-score 0.0 \
    --top-paths 2 \
    --top-similar 2
```

このコマンドは以下の処理を自動的に実行します：
1. **類似ノードの計算**: 訓練データから類似ユーザー・アイテムを検索
2. **パス重要度の計算**: ユーザー・アイテムペア間のパスを探索して重要度を計算
3. **ユーザー・属性親和度の計算**: ユーザーとアイテム属性の親和度を計算
4. **プロンプトの生成**: 上記の結果を使ってプロンプトを再生成して置き換え

既存の`train.json`, `eval.json`, `test.json`から`uid`, `iid`, `chosen`, `similarity_score`を読み込み、新しく計算したデータを使ってプロンプトだけを再生成します。

### 4. 既存データのプロンプト置き換え（計算済みデータを使用）

既に計算済みの類似ノード・パス・属性スコアのファイルがある場合、それらを使ってプロンプトだけを置き換えます:

```bash
uv run raft-data-generator replace-prompts \
    --input-dir /home/kimura/repos/G-Refer/raft_data/yelp \
    --user-profiles datasets/yelp/user_profile.json \
    --item-profiles datasets/yelp/item_profile.json \
    --attribute-scores packages/user-item-attribute-scorer/output/user_item_attribute_scores_no_location.csv \
    --path-scores packages/path-importance-calculator/output/path_importance_scores_with_names.json \
    --similarity packages/semantic-similarity-retriever/output/semantic_similarity_results.json \
    --output-dir packages/raft-data-generator/output/yelp \
    --min-attribute-score 0.0 \
    --top-paths 2 \
    --top-similar 2
```

このコマンドは、既存の`train.json`, `eval.json`, `test.json`から`uid`, `iid`, `chosen`, `similarity_score`を読み込み、計算済みのデータを使ってプロンプトだけを再生成して置き換えます。

### OpenAI APIキーの設定

環境変数で設定するか、`--explanation-api-key`で指定できます:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### HuggingFaceモデルを使用する場合

```bash
uv run raft-data-generator \
    --interactions datasets/yelp/interactions_yelp_evaluation.txt \
    --user-profiles datasets/yelp/user_profile.json \
    --item-profiles datasets/yelp/item_profile.json \
    --attribute-scores packages/user-item-attribute-scorer/output/user_item_attribute_scores_no_location.csv \
    --path-scores packages/path-importance-calculator/output/path_importance_scores_with_names.json \
    --similarity packages/semantic-similarity-retriever/output/semantic_similarity_results.json \
    --output-dir packages/raft-data-generator/output/yelp \
    --explanation-provider huggingface \
    --explanation-model meta-llama/Llama-2-7b-chat-hf \
    --explanation-batch-size 4
```

## パラメータ

### 新規データ生成コマンド（デフォルト）

#### 必須パラメータ

- `--interactions`: 評価用のユーザー・アイテムペアファイル（CSV形式、uid, iidカラム）
- `--user-profiles`: ユーザープロファイルJSONのパス
- `--item-profiles`: アイテムプロファイルJSONのパス
- `--attribute-scores`: 属性スコアCSVのパス
- `--path-scores`: パス重要度スコアJSONのパス
- `--similarity`: セマンティック類似度結果JSONのパス

### インタラクションファイルから直接生成コマンド（`from-interactions`）

#### 必須パラメータ

- `--train-interactions`: 訓練用インタラクションファイル（CSV形式、uid, iidカラム）
- `--eval-interactions`: 評価用インタラクションファイル（CSV形式、uid, iidカラム）
- `--test-interactions`: テスト用インタラクションファイル（CSV形式、uid, iidカラム）
- `--user-profiles`: ユーザープロファイルJSONのパス
- `--item-profiles`: アイテムプロファイルJSONのパス
- `--model-path`: 学習済みKGATモデルのパス

#### オプションパラメータ

- `--data-dir`: データディレクトリ（デフォルト: `datasets/`）
- `--data-name`: データセット名（デフォルト: `yelp`）
- `--kg-file`: kg_final.txtのパス（Noneの場合は`data_dir/data_name/kg_final.txt`を使用）
- `--entity-list-file`: entity_list.txtのパス（オプション）
- `--relation-list-file`: relation_list.txtのパス（オプション）
- `--ignore-relations`: パス探索で無視するリレーションIDのリスト（例: `[5]`）
- `--exclude-relations`: 属性スコア計算で除外するリレーションIDのリスト（例: `[5]`）
- `--text-encoder`: テキストエンコーダーモデル名（デフォルト: `sentence-transformers/multi-qa-distilbert-cos-v1`）
- `--similarity-topk`: 類似ノードの最大数（デフォルト: `2`）
- `--similarity-pruning-score`: 類似度スコアの閾値（デフォルト: `0.0`）
- `--max-hops`: 最大ホップ数（デフォルト: `3`）
- `--min-hops`: 最小ホップ数（デフォルト: `3`）

### プロンプト置き換えコマンド（`replace-prompts`）

#### 必須パラメータ

- `--input-dir`: 既存のRAFTデータディレクトリ（train.json, eval.json, test.jsonを含む）
- `--user-profiles`: ユーザープロファイルJSONのパス
- `--item-profiles`: アイテムプロファイルJSONのパス
- `--attribute-scores`: 属性スコアCSVのパス
- `--path-scores`: パス重要度スコアJSONのパス
- `--similarity`: セマンティック類似度結果JSONのパス

### オプションパラメータ

- `--output-dir`: 出力ディレクトリのパス（デフォルト: `packages/raft-data-generator/output`）
- `--dataset-name`: データセット名（デフォルト: `yelp`）
- `--train-ratio`: 訓練データの割合（デフォルト: `0.8`）
- `--eval-ratio`: 評価データの割合（デフォルト: `0.1`、残りがテストデータ）
- `--min-attribute-score`: 属性スコアの最小閾値（デフォルト: `0.0`）
- `--top-paths`: 含めるトップパスの数（デフォルト: `2`）
- `--top-similar`: 含める類似ユーザー/アイテムの数（デフォルト: `2`）
- `--explanation-provider`: 説明生成のプロバイダー（`openai` または `huggingface`、デフォルト: `openai`）
- `--explanation-model`: 説明生成に使用するモデル名（デフォルト: `gpt-3.5-turbo`）
- `--explanation-api-key`: OpenAI APIキー（環境変数`OPENAI_API_KEY`からも読み込み可能）
- `--explanation-max-tokens`: 説明生成の最大トークン数（デフォルト: `150`）
- `--explanation-temperature`: 説明生成の温度パラメータ（デフォルト: `0.7`）
- `--explanation-batch-size`: 説明生成のバッチサイズ（HuggingFaceの場合、デフォルト: `16`）

## 出力形式

出力はG-Referのraft_dataと同じ形式のJSONLファイルです。各データセット（train, eval, test）に対して、以下の形式のJSONファイルが生成されます：

```json
{
  "uid": 1260,
  "iid": 11980,
  "prompt": "Given the business title, business profile, and user profile...",
  "chosen": "### The user would enjoy...",
  "reject": "I DO NOT KNOW",
  "similarity_score": -0.07674673199653625
}
```

各エントリには以下が含まれます：

- `uid`: ユーザーID
- `iid`: アイテムID
- `prompt`: 説明プロンプト（explainable-rec-prompt-generatorで生成）
- `chosen`: 選択された説明（explanation-generatorで生成）
- `reject`: 拒否された説明（常に "I DO NOT KNOW"）
- `similarity_score`: 類似度スコア（類似ユーザーと類似アイテムの平均スコア）

## 出力ファイル

指定した出力ディレクトリに以下のファイルが生成されます：

- `train.json`: 訓練データ
- `eval.json`: 評価データ
- `test.json`: テストデータ

## 実装詳細

### データ生成フロー

1. **プロンプト生成**: `explainable-rec-prompt-generator`を使用して、ユーザー・アイテムペアごとに説明プロンプトを生成
2. **説明生成**: `explanation-generator`を使用して、プロンプトから説明を生成
3. **類似度スコア計算**: `semantic-similarity-retriever`の結果から、類似ユーザーと類似アイテムの平均スコアを計算
4. **データ分割**: 指定された割合でtrain/eval/testに分割
5. **JSON形式で出力**: G-Referのraft_dataと同じ形式で保存

### 依存パッケージ

このパッケージは以下のワークスペースパッケージに依存しています：

- `explainable-rec-prompt-generator`: プロンプト生成
- `explanation-generator`: 説明生成

これらのパッケージは自動的にワークスペースから解決されます。
