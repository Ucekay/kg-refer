import logging
import sys

from cyclopts import App

from kg_formatter.config import FormatterConfig
from kg_formatter.formatter import KGFormatter
from kg_formatter.merger import KGMerger

app = App(
    name="kg-formatter",
    help="Format and remap Knowledge Graph IDs for training",
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@app.default
def main(config: FormatterConfig = FormatterConfig()):
    """Format and remap Knowledge Graph IDs or merge KG files.

    Mode 1 - ID Remapping (default):
    This tool performs:
    1. Assigns incremental IDs to all unique relations
    2. Assigns incremental IDs to all unique tail entities
    3. Generates remapped KG triplets using the assigned IDs

    Outputs:
    - relation_list.txt: relation_name (quoted) and remap_id (space-separated)
    - entity_list.txt: entity_name (quoted) and remap_id (space-separated)
    - kg_final.txt: head relation_id tail_id (space-separated, no header)

    Mode 2 - Merge Mode (--merge-mode):
    Merges base KG (with empty triplets) with supplement KG to create complete KG.
    Requires --base-kg-path and --supplement-kg-path to be specified.
    """
    # Setup logging
    setup_logging(config.verbose)
    logger = logging.getLogger(__name__)

    if config.merge_mode:
        # Merge mode
        if not config.base_kg_path or not config.supplement_kg_path:
            logger.error(
                "Merge mode requires --base-kg-path and --supplement-kg-path to be specified"
            )
            sys.exit(1)

        if not config.base_kg_path.exists():
            logger.error(f"Base KG file not found: {config.base_kg_path}")
            sys.exit(1)

        if not config.supplement_kg_path.exists():
            logger.error(f"Supplement KG file not found: {config.supplement_kg_path}")
            sys.exit(1)

        try:
            # Run merger
            merger = KGMerger()
            merger.merge_kg(
                config.base_kg_path,
                config.supplement_kg_path,
                config.merged_output,
            )
            sys.exit(0)

        except Exception as e:
            logger.error(f"Error during merging: {e}", exc_info=config.verbose)
            sys.exit(1)
    else:
        # ID remapping mode (default)
        if not config.input.exists():
            logger.error(f"Input file not found: {config.input}")
            sys.exit(1)

        # Create output directory if needed
        config.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Run formatter
            formatter = KGFormatter()
            formatter.format_kg(
                config.input,
                config.output_dir,
            )
            sys.exit(0)

        except Exception as e:
            logger.error(f"Error during formatting: {e}", exc_info=config.verbose)
            sys.exit(1)


if __name__ == "__main__":
    app()
