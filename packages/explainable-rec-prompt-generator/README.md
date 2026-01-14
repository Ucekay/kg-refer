# Explainable Recommendation Prompt Generator

説明可能な推薦のためのプロンプトを生成するパッケージです。ユーザーとアイテムのペアに対して、属性嗜好度、類似ノード、説明パスを含む包括的なプロンプトを生成します。

## インストール

プロジェクトルートから:

```bash
uv sync
```

## 使い方

プロジェクトルートから実行します:

```bash
uv run explainable-rec-prompt-generator \
    --interactions datasets/yelp/interactions_yelp_evaluation.txt \
    --user-profiles datasets/yelp/user_profile.json \
    --item-profiles datasets/yelp/item_profile.json \
    --attribute-scores packages/user-item-attribute-scorer/output/user_item_attribute_scores_no_location.csv \
    --path-scores packages/path-importance-calculator/output/path_importance_scores_with_names.json \
    --similarity packages/semantic-similarity-retriever/output/semantic_similarity_results.json \
    --output packages/explainable-rec-prompt-generator/output/prompts.csv \
    --min-attribute-score 0.0 \
    --top-paths 2 \
    --top-similar 2
```

## パラメータ

- `--interactions`: 評価用のユーザー・アイテムペアファイル（必須）
- `--user-profiles`: ユーザープロファイルJSON（必須）
- `--item-profiles`: アイテムプロファイルJSON（必須）
- `--attribute-scores`: 属性スコアCSV（必須）
- `--path-scores`: パス重要度スコアJSON（必須）
- `--similarity`: セマンティック類似度結果JSON（必須）
- `--output`: 出力ファイルパス（デフォルト: `packages/explainable-rec-prompt-generator/output/prompts.csv`）
- `--min-attribute-score`: 属性スコアの最小閾値（デフォルト: 0.0）
- `--top-paths`: 含めるトップパスの数（デフォルト: 2）
- `--top-similar`: 含める類似ユーザー/アイテムの数（デフォルト: 2）

## 出力形式

CSV形式で出力されます。各行には以下の列が含まれます:

- `uid`: ユーザーID
- `iid`: アイテムID
- `prompt`: 生成されたプロンプト

生成されるプロンプトは以下の情報を含みます:

1. **アイテムタイトル**: アイテムの識別子
2. **アイテムプロファイル**: アイテムの説明
3. **ユーザープロファイル**: ユーザーの嗜好傾向
4. **属性嗜好度**: ユーザーの属性名（0以上のスコアを持つ属性のみ、スコアは非表示）
5. **類似ユーザー**: 類似ユーザーのプロファイル（IDとスコアなし）
6. **類似アイテム**: 類似アイテムのプロファイル（IDとスコアなし）
7. **説明パス**: ユーザーとアイテムを接続するパス（IDとスコアなし）

## 例

生成されるプロンプトの例:

```
Given the business title, business profile, user profile, and the user's attribute preference scores for specific item attributes, please explain why the user would enjoy this business within 50 words. Business title: Item 11821. Business profile: Fans of gourmet burgers, unique salads, and flavorful appetizers in a lively and popular setting would enjoy Bru Burger Bar in Indianapolis. User profile: This user is likely to enjoy businesses with a focus on quality ingredients, cozy atmospheres, and excellent service. They appreciate a variety of menu options, traditional flavors, and unique dining experiences. User's Attribute Preferences: good quality burgers, cozy atmosphere, unique menu items
### For the user-item pair, here are some related users and items: Users: This user is likely to enjoy businesses with a variety of food and drink options..., This user appreciates unique and thoughtfully prepared dishes... Items: Users who enjoy gourmet burgers and craft beers..., Fans of American cuisine with a focus on quality ingredients... 
### For the given user-item pair, here are several related paths connecting users and items through their interactions: 1. User (Profile: ...) -> user_interacts_with_item -> Item (Profile: ...) 2. User (Profile: ...) -> user_interacts_with_item -> Item (Profile: ...) -> has atmosphere -> cozy -> inverse_has atmosphere -> Item (Profile: ...)
### Explanation:
```

プロンプト内の改行は、CSV形式でダブルクォートで囲まれて正しく保存されます。
