# 使用例

## OpenAI APIを使用する場合

### 環境変数でAPIキーを設定

```bash
# APIキーを環境変数に設定
export OPENAI_API_KEY="sk-..."

# 全3000件を処理
cd kg-refer
uv run explanation-generator

# 最初の10件のみ処理（テスト用）
uv run explanation-generator --max-samples 10

# GPT-4を使用
uv run explanation-generator --model-name gpt-4

# 温度パラメータを調整（より創造的な出力、デフォルトは0.0）
# 注: G-Referではtemperature=0.0（greedy decoding）を使用
uv run explanation-generator --temperature 0.0 --max-tokens 200
```

### コマンドラインでAPIキーを指定

```bash
cd kg-refer
uv run explanation-generator --api-key "sk-..."
```

## HuggingFace Transformersを使用する場合

### LLaMA 2を使用

```bash
cd kg-refer

# LLaMA 2 7B Chat（事前にHuggingFaceでアクセス許可が必要）
uv run explanation-generator \
  --provider huggingface \
  --model-name meta-llama/Llama-2-7b-chat-hf \
  --batch-size 4

# LLaMA 2 13B Chat（より大きなモデル）
uv run explanation-generator \
  --provider huggingface \
  --model-name meta-llama/Llama-2-13b-chat-hf \
  --batch-size 2
```

### Mistralを使用

```bash
cd kg-refer

# Mistral 7B Instruct
uv run explanation-generator \
  --provider huggingface \
  --model-name mistralai/Mistral-7B-Instruct-v0.2 \
  --batch-size 4 \
  --temperature 0.0
```

### ローカルのファインチューンモデルを使用

```bash
cd kg-refer

# ローカルに保存されたモデルを使用
uv run explanation-generator \
  --provider huggingface \
  --model-name /path/to/your/finetuned/model \
  --batch-size 4
```

## レジューム機能

処理が中断された場合、自動的に続きから再開します：

```bash
# 最初の実行（例: 500件処理したところで中断）
uv run explanation-generator

# 再実行すると自動的に501件目から処理
uv run explanation-generator
```

明示的に開始位置を指定：

```bash
# 1000件目から処理
uv run explanation-generator --start-index 1000

# 1000件目から100件だけ処理
uv run explanation-generator --start-index 1000 --max-samples 100
```

## データセット別の実行

### Yelpデータセット（デフォルト）

```bash
cd kg-refer
uv run explanation-generator
```

### カスタムプロンプトファイル

```bash
cd kg-refer
uv run explanation-generator \
  --prompts-file path/to/custom/prompts.csv \
  --output-file path/to/custom/output.jsonl
```

## 出力の確認

生成された説明を確認：

```bash
# 最初の3件を確認
head -3 packages/explanation-generator/output/explanations.jsonl | jq

# 特定のユーザー・アイテムペアを検索
cat packages/explanation-generator/output/explanations.jsonl | jq 'select(.uid == 4991 and .iid == 11821)'

# 説明の長さを確認
cat packages/explanation-generator/output/explanations.jsonl | jq '.explanation | length'

# 処理済み件数を確認
wc -l packages/explanation-generator/output/explanations.jsonl
```

## パフォーマンスチューニング

### OpenAI API

- **レート制限**: OpenAI APIにはレート制限があります。大量の処理を行う場合は、アカウントの制限を確認してください
- **コスト**: トークン数に応じて課金されます。`--max-tokens`で制限を設定できます

### HuggingFace Transformers

- **バッチサイズ**: GPUメモリに応じて調整してください
  - 7Bモデル + 16GB GPU: `--batch-size 4-8`
  - 13Bモデル + 24GB GPU: `--batch-size 2-4`
  - 70Bモデル: マルチGPUが必要

- **量子化**: メモリを節約する場合は、モデルを量子化してください（コード修正が必要）

## トラブルシューティング

### OpenAI APIエラー

```
Error code: 401 - Invalid API key
```
→ APIキーが正しく設定されているか確認してください

```
Error code: 429 - Rate limit exceeded
```
→ レート制限に達しています。しばらく待ってから再実行してください

### HuggingFace Transformersエラー

```
CUDA out of memory
```
→ バッチサイズを減らしてください：`--batch-size 1`

```
You do not have permission to access this model
```
→ HuggingFace Hubでモデルへのアクセス許可を取得してください

## 推奨設定

### 高品質な説明が必要な場合

```bash
uv run explanation-generator \
  --provider openai \
  --model-name gpt-4 \
  --temperature 0.7 \
  --max-tokens 200
```

### コストを抑えたい場合

```bash
uv run explanation-generator \
  --provider openai \
  --model-name gpt-3.5-turbo \
  --temperature 0.5 \
  --max-tokens 100
```

### ローカルで実行したい場合

```bash
uv run explanation-generator \
  --provider huggingface \
  --model-name mistralai/Mistral-7B-Instruct-v0.2 \
  --batch-size 4 \
  --temperature 0.0 \
  --max-tokens 1024
```
