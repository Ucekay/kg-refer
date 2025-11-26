"""Command-line interface for KGAT Explainer."""

from cyclopts import App

from kgat_explainer.config import KGATExplainerConfig
from kgat_explainer.explain import explain_correct_predictions, explain_user_predictions
from kgat_explainer.stats import compute_statistics, evaluate_model

app = App()


@app.default
def main(config: KGATExplainerConfig = KGATExplainerConfig()):
    """KGAT Explainer - Explain KGAT model predictions through path analysis."""
    if config.explain_correct:
        print("Explaining correct predictions mode activated.")
        explain_correct_predictions(config)
    elif config.explain_user is not None:
        print(f"Explaining predictions for user {config.explain_user}.")
        explain_user_predictions(config)
    elif config.statistics:
        print("Computing path statistics mode activated.")
        compute_statistics(config)
    elif config.evaluate:
        print("Evaluation mode activated.")
        evaluate_model(config)
    else:
        print(
            "No mode selected. Use --explain-correct, --explain-user, --statistics, or --evaluate"
        )
        print(
            "Example: kgat-explainer --explain-correct --model-path path/to/model.pth"
        )


if __name__ == "__main__":
    app()
