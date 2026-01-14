"""CLI for explainable recommendation prompt generator."""

import sys
from pathlib import Path

import pandas as pd
from cyclopts import App

from explainable_rec_prompt_generator.data_loader import DataLoader
from explainable_rec_prompt_generator.prompt_generator import PromptGenerator

app = App()


def load_evaluation_pairs(interactions_path: str | Path) -> pd.DataFrame:
    """Load user-item pairs from the evaluation file.

    Args:
        interactions_path: Path to interactions_yelp_evaluation.txt

    Returns:
        DataFrame with uid and iid columns
    """
    return pd.read_csv(interactions_path)


@app.default
def main(
    interactions: str,
    user_profiles: str,
    item_profiles: str,
    attribute_scores: str,
    path_scores: str,
    similarity: str,
    item_list: str | None = None,
    output: str = "packages/explainable-rec-prompt-generator/output/prompts.csv",
    min_attribute_score: float = 0.0,
    top_paths: int = 2,
    top_similar: int = 2,
    simple_mode: bool = True,
    include_examples: bool = False,
) -> None:
    """Generate explainable recommendation prompts for user-item pairs.

    Args:
        interactions: Path to interactions_yelp_evaluation.txt
        user_profiles: Path to user_profile.json
        item_profiles: Path to item_profile.json
        attribute_scores: Path to user_item_attribute_scores_no_location.csv
        path_scores: Path to path_importance_scores_with_names.json
        similarity: Path to semantic_similarity_results.json
        item_list: Path to item_list.txt (optional, for item names)
        output: Output file path for generated prompts (default: packages/explainable-rec-prompt-generator/output/prompts.csv)
        min_attribute_score: Minimum attribute score threshold (default: 0.0)
        top_paths: Number of top paths to include (default: 2)
        top_similar: Number of similar users/items to include (default: 2)
        simple_mode: If True, use simple template (default: True)
        include_examples: If True, include Examples section (default: False, only used when simple_mode=False)
    """
    # Initialize data loader
    print("Loading data...")
    data_loader = DataLoader(
        user_profile_path=user_profiles,
        item_profile_path=item_profiles,
        attribute_scores_path=attribute_scores,
        path_scores_path=path_scores,
        similarity_path=similarity,
    )

    # Load item titles if item_list is provided
    if item_list:
        print(f"Loading item titles from {item_list}...")
        data_loader.load_item_titles(item_list)

    # Initialize prompt generator
    prompt_generator = PromptGenerator(data_loader)

    # Load evaluation pairs
    print(f"Loading evaluation pairs from {interactions}...")
    eval_pairs = load_evaluation_pairs(interactions)
    print(f"Found {len(eval_pairs)} user-item pairs")

    # Create output directory if it doesn't exist
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate prompts for each pair
    print("Generating prompts...")
    results = []
    for i, (idx, row) in enumerate(eval_pairs.iterrows()):
        uid = int(row["uid"])
        iid = int(row["iid"])

        print(f"Processing pair {i + 1}/{len(eval_pairs)}: User {uid}, Item {iid}")

        try:
            prompt = prompt_generator.generate_prompt(
                uid=uid,
                iid=iid,
                min_attribute_score=min_attribute_score,
                top_paths=top_paths,
                top_similar=top_similar,
                simple_mode=simple_mode,
                include_examples=include_examples,
            )
            results.append({"uid": uid, "iid": iid, "prompt": prompt})

        except Exception as e:
            print(
                f"Error processing pair {i + 1} (User {uid}, Item {iid}): {e}",
                file=sys.stderr,
            )
            results.append({"uid": uid, "iid": iid, "prompt": f"Error: {e}"})
            continue

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Done! Prompts saved to {output_path}")


if __name__ == "__main__":
    app()
