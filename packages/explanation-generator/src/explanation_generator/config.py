"""設定モジュール"""

from dataclasses import dataclass
from typing import Literal

from cyclopts import Parameter


@Parameter(name="*")
@dataclass
class ExplanationGeneratorConfig:
    """説明生成の設定"""

    prompts_file: str = "packages/explainable-rec-prompt-generator/output/prompts.csv"
    """プロンプトCSVファイルのパス"""

    output_file: str = "packages/explanation-generator/output/explanations.jsonl"
    """生成された説明を保存するJSONLファイルのパス"""

    ground_truth_file: str = "datasets/gen_explanations/G-Refer/yelp_pred.jsonl"
    """正解説明を含むJSONLファイルのパス（オプション）"""

    provider: Literal["openai", "huggingface"] = "openai"
    """LLMプロバイダー (openai または huggingface)"""

    model_name: str = "gpt-3.5-turbo"
    """使用するモデル名 (OpenAI: gpt-3.5-turbo, gpt-4, etc. / HuggingFace: model path)"""

    api_key: str = ""
    """OpenAI APIキー（環境変数OPENAI_API_KEYからも読み込み可能）"""

    max_tokens: int = 1024
    """生成する最大トークン数（G-Referと同じ）"""

    temperature: float = 0.0
    """生成時の温度パラメータ（0.0でgreedy decoding）"""

    batch_size: int = 16
    """バッチサイズ (HuggingFaceの場合のみ、G-Referと同じ)"""

    start_index: int = 0
    """開始インデックス（レジューム用）"""

    max_samples: int = -1
    """処理する最大サンプル数（-1で全件処理）"""

    use_two_pass: bool = False
    """2パス生成を行うかどうか（1回目の結果を履歴に含めて2回目を生成）"""

    use_chat_template: bool = False
    """チャットテンプレートを使用するかどうか（QwenなどのInstructモデル用）"""
