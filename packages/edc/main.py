import asyncio

from cyclopts import App

from edc.edc_framework import EDC
from edc.types.config import EDCConfig

app = App()


@app.default
def main(config: EDCConfig = EDCConfig()):
    """Main entry point for EDC framework."""
    # Initialize EDC framework with config
    edc = EDC(config=config)

    input_text_list = open(
        config.input_text_file_path, "r", encoding="utf-8"
    ).readlines()

    # Choose processing mode based on configuration
    if config.enable_parallel_processing:
        print(
            f"Running in PARALLEL mode with {config.max_concurrent_requests} concurrent requests"
        )
        # Run async processing
        output_kg = asyncio.run(
            edc.extract_kg_async(
                input_text_list,
                config.output_dir,
                refinement_iterations=config.refinement_iterations,
            )
        )
    else:
        print("Running in SEQUENTIAL mode")
        # Run synchronous processing
        output_kg = edc.extract_kg(
            input_text_list,
            config.output_dir,
            refinement_iterations=config.refinement_iterations,
        )

    print(f"Processing completed. Results saved to: {config.output_dir}")


if __name__ == "__main__":
    app()
