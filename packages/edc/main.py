import asyncio

import torch.multiprocessing as multiprocessing
from cyclopts import App

from edc.edc_framework import EDC
from edc.types.config import EDCConfig

app = App()


@app.default
def main(config: EDCConfig = EDCConfig()):
    """Main entry point for EDC framework."""
    #
    # Initialize EDC framework with config
    edc = EDC(config=config)

    input_text_list = open(
        config.input_text_file_path, "r", encoding="utf-8"
    ).readlines()

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
            )
        )
    else:
        # Use synchronous extraction
        output_kg = edc.extract_kg(
            input_text_list,
            config.output_dir,
            refinement_iterations=config.refinement_iterations,
        )


if __name__ == "__main__":
    app()
