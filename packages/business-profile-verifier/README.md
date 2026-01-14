# Business Profile Verifier

VeriFastScoreをベースにした、Business profileをソースとして説明文を検証するパッケージです。

## 概要

このパッケージは、explanation-generatorで生成された説明文（output_str）を、input_strに含まれるBusiness profileをエビデンスとして検証します。外部検索APIは使用せず、Business profileのみをソースとして使用します。

## 機能

- **Business profile抽出**: input_strからBusiness profileを自動抽出
- **クレーム分解**: 説明文を検証可能なクレームに分解
- **検証**: Business profileに基づいて各クレームを検証（Supported/Unsupported）
- **VeriFastScore算出**: F1スコアベースの検証スコアを計算

## インストール

```bash
cd kg-refer
uv sync
```

追加でspaCyの英語モデルをインストール:

```bash
uv run python -m spacy download en_core_web_sm
```

## 使用方法

### 基本的な使い方

```bash
cd kg-refer
uv run business-profile-verifier \
  --input-file packages/explanation-generator/output/explanations_kg_5.jsonl \
  --output-dir packages/business-profile-verifier/output \
  --model-name rishanthrajendhran/VeriFastScore
```

### パラメータ

- `--input-file`: 入力JSONLファイルのパス（必須）
  - explanation-generatorの出力ファイルを指定
- `--output-dir`: 出力ディレクトリ（デフォルト: `./output`）
- `--model-name`: 使用するモデル名（デフォルト: `rishanthrajendhran/VeriFastScore`）
- `--cache-dir`: キャッシュディレクトリ（デフォルト: `./data/cache`）

## 入力形式

explanation-generatorの出力JSONL形式:

```json
{
  "index": 0,
  "source_data": {
    "uid": 4991,
    "iid": 11821,
    "prompt": "...",
    "chosen": "...",
    "reject": "I DO NOT KNOW"
  },
  "input_str": "Business title: Taco Riendo. Business profile: Mexican food enthusiasts...",
  "output_str": "The user would enjoy Taco Riendo because..."
}
```

## 出力形式

検証結果はJSONL形式で `{output_dir}/model_output/` に保存されます:

```json
{
  "index": 0,
  "source_data": {...},
  "input_str": "...",
  "output_str": "...",
  "business_profile": "Mexican food enthusiasts...",
  "evidence": "Mexican food enthusiasts...",
  "claim_verification_result": [
    {
      "claim": "Taco Riendo provides Mexican food",
      "verification_result": "supported"
    }
  ]
}
```

## VeriFastScoreについて

VeriFastScoreは、以下の式で計算されます:

- **Precision**: サポートされたクレームの割合
- **Recall**: `min(1, クレーム数 / median_claims)` （median_claims = 17）
- **F1 Score**: `2 * precision * recall / (precision + recall)`

## 実装詳細

### VeriFastScoreとの違い

1. **エビデンスソース**: 外部検索APIの代わりにBusiness profileを使用
2. **抽出ロジック**: `extract_business_profile()`関数でinput_strからBusiness profileを抽出
3. **依存関係の削減**: 検索API関連のコードを削除

### Business profile抽出

正規表現を使用してinput_strからBusiness profileを抽出:

```python
pattern = r"Business profile:\s*([^\n]*?)(?:\.\s|User profile:|$)"
```

## 注意事項

- GPUを使用することを推奨します（CUDAが利用可能な場合自動的に使用されます）
- 大きなモデルは大量のメモリを消費します
- spaCyモデル `en_core_web_sm` が必要です

## ライセンス

MITライセンス
