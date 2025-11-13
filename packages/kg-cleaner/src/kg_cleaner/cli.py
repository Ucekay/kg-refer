import logging
import sys
from pathlib import Path

from cyclopts import App

from kg_cleaner.cleaner import KGCleaner
from kg_cleaner.config import CleanerConfig

app = App(
    name="kg-cleaner",
    help="Clean and normalize Knowledge Graph triplets",
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
def main(config: CleanerConfig = CleanerConfig()):
    """Clean and normalize Knowledge Graph triplets.

    This tool performs:
    1. Smart case normalization for relations and tails (only when variations exist)
    2. Duplicate triplet removal
    3. Conflict detection for (h,t) pairs with multiple relations
    """
    # Setup logging
    setup_logging(config.verbose)
    logger = logging.getLogger(__name__)

    # Validate that at least one feature is enabled
    if not (
        config.normalize
        or config.deduplicate
        or config.find_conflicts
        or config.filter_terms
        or config.filter_relations
        or config.expand_entities
        or config.replace_entities
        or config.replace_combinations
        or config.unify_relations
    ):
        logger.error(
            "At least one feature must be enabled. "
            "Use --normalize, --deduplicate, --find-conflicts, --filter-terms, "
            "--filter-relations, --expand-entities, --replace-entities, --replace-combinations, or --unify-relations"
        )
        sys.exit(1)

    # Validate input file exists
    if not config.input.exists():
        logger.error(f"Input file not found: {config.input}")
        sys.exit(1)

    # Create output directories if needed
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.conflicts.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Run cleaner
        cleaner = KGCleaner()
        cleaner.clean_file(
            config.input,
            config.output,
            config.conflicts,
            normalize=config.normalize,
            deduplicate=config.deduplicate,
            find_conflicts=config.find_conflicts,
            filter_terms=config.filter_terms,
            filter_relations=config.filter_relations,
            expand_entities=config.expand_entities,
            replace_entities=config.replace_entities,
            replace_combinations=config.replace_combinations,
            unify_relations=config.unify_relations,
        )
        logger.info("✓ Cleaning completed successfully")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Error during cleaning: {e}", exc_info=config.verbose)
        sys.exit(1)


if __name__ == "__main__":
    app()
