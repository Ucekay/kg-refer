"""説明生成のCLI"""

import logging

from cyclopts import App

from .config import ExplanationGeneratorConfig
from .generator import ExplanationGenerator

app = App()

# ロガーの設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@app.default
def main(config: ExplanationGeneratorConfig = ExplanationGeneratorConfig()):
    """プロンプトCSVから説明を生成する

    Args:
        config: 説明生成の設定
    """
    logger.info("説明生成を開始します")
    logger.info(f"プロバイダー: {config.provider}")
    logger.info(f"モデル: {config.model_name}")
    logger.info(f"プロンプトファイル: {config.prompts_file}")
    logger.info(f"出力ファイル: {config.output_file}")

    # ジェネレーターを初期化
    generator = ExplanationGenerator(
        provider=config.provider,
        model_name=config.model_name,
        api_key=config.api_key,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        batch_size=config.batch_size,
        use_chat_template=config.use_chat_template,
    )

    # 説明を生成
    generator.generate_from_csv(
        prompts_file=config.prompts_file,
        output_file=config.output_file,
        ground_truth_file=config.ground_truth_file,
        start_index=config.start_index,
        max_samples=config.max_samples,
        use_two_pass=config.use_two_pass,
    )

    logger.info("説明生成が完了しました！")


if __name__ == "__main__":
    app()
