# Explanation Generator

推薦システムのためのLLMベースの説明生成パッケージです。G-Referの説明生成機能をkg-referに実装しました。

## 概要

このパッケージは、ユーザー・アイテムペアのプロンプトから、LLM（OpenAI APIまたはHuggingFace Transformers）を使用して推薦の説明を生成します。

## 機能

- **OpenAI API対応**: GPT-3.5-turbo、GPT-4などのOpenAIモデルを使用
- **HuggingFace Transformers対応**: ローカルまたはHuggingFace Hubのモデルを使用（LLaMA、Mistralなど）
- **バッチ処理**: HuggingFaceモデルでのバッチ処理に対応
- **レジューム機能**: 中断した場合も続きから処理を再開可能
- **柔軟な設定**: 最大トークン数、温度パラメータなどを調整可能

## インストール

```bash
cd kg-refer
uv sync
```

## 使用方法

### OpenAI APIを使用する場合

```bash
# 環境変数でAPIキーを設定
export OPENAI_API_KEY="your-api-key-here"

# 実行
cd kg-refer
uv run explanation-generator \
  --provider openai \
  --model-name gpt-3.5-turbo \
  --prompts-file packages/explainable-rec-prompt-generator/output/prompts.csv \
  --output-file packages/explanation-generator/output/explanations.jsonl \
  --max-tokens 150 \
  --temperature 0.7
```

または、コマンドラインで直接APIキーを指定：

```bash
uv run explanation-generator \
  --provider openai \
  --model-name gpt-3.5-turbo \
  --api-key "your-api-key-here"
```

### HuggingFace Transformersを使用する場合

```bash
cd kg-refer
uv run explanation-generator \
  --provider huggingface \
  --model-name meta-llama/Llama-2-7b-chat-hf \
  --prompts-file packages/explainable-rec-prompt-generator/output/prompts.csv \
  --output-file packages/explanation-generator/output/explanations.jsonl \
  --max-tokens 150 \
  --temperature 0.7 \
  --batch-size 4
```

### 部分実行

一部のサンプルのみ処理する場合：

```bash
# 最初の100件のみ処理
uv run explanation-generator --max-samples 100

# 1000件目から100件処理
uv run explanation-generator --start-index 1000 --max-samples 100
```

## パラメータ

- `--prompts-file`: プロンプトCSVファイルのパス
- `--output-file`: 出力JSONLファイルのパス
- `--ground-truth-file`: 正解説明を含むJSONLファイルのパス（オプション、デフォルト: `datasets/gen_explanations/G-Refer/yelp_pred.jsonl`）
- `--provider`: LLMプロバイダー (`openai` または `huggingface`)
- `--model-name`: 使用するモデル名
  - OpenAI: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`, etc.
  - HuggingFace: モデルパスまたはHuggingFace Hub ID
- `--api-key`: OpenAI APIキー（環境変数`OPENAI_API_KEY`からも読み込み可能）
- `--max-tokens`: 生成する最大トークン数（デフォルト: 1024、G-Referと同じ）
- `--temperature`: 生成時の温度パラメータ（デフォルト: 0.0、greedy decoding）
- `--batch-size`: バッチサイズ（HuggingFaceの場合のみ、デフォルト: 16、G-Referと同じ）
- `--start-index`: 開始インデックス（レジューム用、デフォルト: 0）
- `--max-samples`: 処理する最大サンプル数（-1で全件処理、デフォルト: -1）

## 入力形式

プロンプトCSVファイルは以下の形式である必要があります：

```csv
uid,iid,prompt
4991,11821,"Given the business title, business profile, and user profile, please explain why the user would enjoy this business..."
1291,11950,"Given the business title..."
```

必須カラム：
- `uid`: ユーザーID
- `iid`: アイテムID
- `prompt`: LLMへの入力プロンプト

## 出力形式

結果はJSON Lines形式で出力されます（G-Refer互換形式）：

```jsonl
{"index": 0, "source_data": {"uid": 4991, "iid": 11821, "prompt": "...", "chosen": "", "reject": "I DO NOT KNOW"}, "input_str": "...", "output_str": "..."}
```

各行には以下が含まれます：
- `index`: サンプルのインデックス
- `source_data`: ソースデータ
  - `uid`: ユーザーID
  - `iid`: アイテムID
  - `prompt`: 入力プロンプト
  - `chosen`: 正解の説明（ground_truth_fileから読み込み、ない場合は空文字列）
  - `reject`: リジェクト応答（"I DO NOT KNOW"）
- `input_str`: 入力プロンプト（promptと同じ）
- `output_str`: 生成された説明

## レジューム機能

処理が中断された場合、既存の出力ファイルがあれば自動的にそこから再開します：

```bash
# 最初の実行（途中で中断）
uv run explanation-generator

# 再実行すると自動的に続きから処理
uv run explanation-generator
```

明示的に開始位置を指定することもできます：

```bash
uv run explanation-generator --start-index 500
```

## 実装詳細

### G-Referとの違い

G-Referの`ds_inference/infer.py`を参考にしていますが、以下の点で異なります：

1. **CSVからの直接読み込み**: G-ReferはJSONファイルを使用していますが、このパッケージはCSVファイルから直接読み込みます
2. **OpenAI API対応**: G-ReferはHuggingFace Transformersのみですが、このパッケージはOpenAI APIにも対応しています
3. **柔軟な設定**: cycloptsを使用してCLIから全てのパラメータを設定可能です

### プロンプトの構造

プロンプトには以下の情報が含まれます：
- ビジネスタイトル
- ビジネスプロファイル
- ユーザープロファイル
- ユーザーの属性選好スコア
- 類似ユーザー・アイテム情報
- パス情報（グラフ検索結果）

## 注意事項

### OpenAI API使用時
- APIキーが必要です
- 使用料金が発生します
- レート制限に注意してください

### HuggingFace Transformers使用時
- 大きなモデルは大量のメモリを消費します
- GPUの使用を推奨します
- 一部のモデルは認証が必要です（LLaMAなど）

## ライセンス

MITライセンス
