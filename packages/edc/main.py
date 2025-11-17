import asyncio
import json

import torch.multiprocessing as multiprocessing
from cyclopts import App

from edc.edc_framework import EDC
from edc.types.config import EDCConfig

app = App()


def load_kg_data(kg_file_path: str):
    """
    Load KG data from JSON file.

    Args:
        kg_file_path: Path to the KG JSON file

    Returns:
        list: List of KG items with iid and triplets
    """
    with open(kg_file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_empty_triplet_iids(kg_data):
    """
    Get IIDs of items with empty triplets.

    Args:
        kg_data: List of KG items

    Returns:
        set: Set of IIDs with empty triplets
    """
    empty_iids = set()
    for item in kg_data:
        if "iid" in item and "triplets" in item:
            if not item["triplets"] or len(item["triplets"]) == 0:
                empty_iids.add(item["iid"])
    return empty_iids


def load_input_data(file_path: str, filter_iids=None):
    """
    Load input data from either text file or JSONL format.

    Args:
        file_path: Path to the input file
        filter_iids: Optional set of IIDs to filter (only process these IIDs)

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

                    # Skip if filtering is enabled and this IID is not in the filter set
                    if filter_iids is not None and iid not in filter_iids:
                        continue

                    # Parse the nested JSON in business summary
                    summary_data = json.loads(data["business summary"])
                    text = summary_data.get("summarization", "")
                elif "text" in data:
                    # Generic JSONL with text field
                    iid = data.get("id", len(texts))

                    # Skip if filtering is enabled and this IID is not in the filter set
                    if filter_iids is not None and iid not in filter_iids:
                        continue

                    text = data["text"]
                else:
                    # Use entire JSON as text if no specific field found
                    iid = data.get("id", len(texts))

                    # Skip if filtering is enabled and this IID is not in the filter set
                    if filter_iids is not None and iid not in filter_iids:
                        continue

                    text = json.dumps(data)

                texts.append(text)
                ids.append(iid)

        except json.JSONDecodeError:
            # It's a regular text file (one text per line)
            for line_num, line in enumerate(f):
                # For text files, if filtering is enabled, skip items not in filter
                if filter_iids is not None and line_num not in filter_iids:
                    continue

                texts.append(line.strip())
                ids.append(line_num)

    return texts, ids


@app.default
def main(config: EDCConfig = EDCConfig()):
    """Main entry point for EDC framework."""
    #
    # Initialize EDC framework with config
    edc = EDC(config=config)

    # Determine which IIDs to process
    filter_iids = None
    if config.only_empty_triplets:
        if not config.kg_file_path:
            raise ValueError(
                "kg_file_path must be specified when only_empty_triplets is True"
            )

        print(f"Loading KG data from {config.kg_file_path}")
        kg_data = load_kg_data(config.kg_file_path)
        filter_iids = get_empty_triplet_iids(kg_data)
        print(f"Found {len(filter_iids)} items with empty triplets")
        print(
            f"Empty triplet IIDs: {sorted(filter_iids)[:20]}{'...' if len(filter_iids) > 20 else ''}"
        )

    # Load input data - support both text files and JSONL format
    input_text_list, input_ids_list = load_input_data(
        config.input_text_file_path, filter_iids=filter_iids
    )

    print(f"Processing {len(input_text_list)} items")

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
