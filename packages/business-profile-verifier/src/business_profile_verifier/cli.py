"""CLI entry point for business-profile-verifier."""

import json
import logging
import os
from pathlib import Path

from cyclopts import App
from business_profile_verifier.verifier import BusinessProfileVerifier


app = App()


@app.default
def main(
    input_file: Path,
    output_dir: str = "./output",
    model_name: str = "rishanthrajendhran/VeriFastScore",
    cache_dir: str = "./data/cache",
):
    """Verify explanation outputs using Business profile as evidence.

    Args:
        input_file: Path to input JSONL file containing explanations.
        output_dir: Directory for output files.
        model_name: Name or path of the model to use for verification.
        cache_dir: Directory for caching.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found.")
        return

    # Read input data
    with open(input_file, "r") as f:
        data = [json.loads(x) for x in f.readlines() if x.strip()]

    print(f"Loaded {len(data)} samples from {input_file}")

    # Initialize verifier
    verifier = BusinessProfileVerifier(
        model_name=model_name,
        cache_dir=cache_dir,
        output_dir=output_dir,
    )

    # Get input file name without extension for output naming
    input_file_name = input_file.stem

    # Run verification
    time_taken = verifier.verify(data, input_file_name)

    print("Verification complete!")
    print(f"Results saved to: {os.path.join(output_dir, 'model_output')}")


if __name__ == "__main__":
    app()
