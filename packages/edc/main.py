import asyncio
import json

import torch.multiprocessing as multiprocessing
from cyclopts import App

from edc.edc_framework import EDC
from edc.types.config import EDCConfig

app = App()


def load_input_data(file_path: str):
    """
    Load input data from either text file or JSONL format.

    Args:
        file_path: Path to the input file

    Returns:
        tuple: (list of texts, list of ids)
    """
    texts = []
    ids = []

    with open(file_path, "r", encoding="utf-8") as f:
        # Try to detect if it's JSONL by reading first line
        first_line = f.readline().strip()
        f.seek(0)  # Reset file pointer

        # Check if first line is valid JSON
        try:
            json.loads(first_line)
            # It's JSONL format
            for line in f:
                data = json.loads(line.strip())
                if "iid" in data and "business summary" in data:
                    # Yelp JSONL format
                    iid = data["iid"]
                    # Parse the nested JSON in business summary
                    summary_data = json.loads(data["business summary"])
                    text = summary_data.get("summarization", "")
                elif "text" in data:
                    # Generic JSONL with text field
                    iid = data.get("id", len(texts))
                    text = data["text"]
                else:
                    # Use entire JSON as text if no specific field found
                    iid = data.get("id", len(texts))
                    text = json.dumps(data)

                texts.append(text)
                ids.append(iid)

        except json.JSONDecodeError:
            # It's a regular text file (one text per line)
            for line_num, line in enumerate(f):
                texts.append(line.strip())
                ids.append(line_num)

    return texts, ids


@app.default
def main(config: EDCConfig = EDCConfig()):
    """Main entry point for EDC framework."""
    #
    # Initialize EDC framework with config
    edc = EDC(config=config)

    # Load input data - support both text files and JSONL format
    input_text_list, input_ids_list = load_input_data(config.input_text_file_path)

    if config.enable_parallel_requests:
        # Use async extraction for parallel processing
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass
        output_kg = asyncio.run(
            edc.extract_kg_async(
                input_text_list,
                config.output_dir,
                refinement_iterations=config.refinement_iterations,
                input_ids_list=input_ids_list,
            )
        )
    else:
        # Use synchronous extraction
        output_kg = edc.extract_kg(
            input_text_list,
            config.output_dir,
            refinement_iterations=config.refinement_iterations,
            input_ids_list=input_ids_list,
        )


if __name__ == "__main__":
    app()
