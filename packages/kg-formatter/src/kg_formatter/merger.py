import json
import logging
from pathlib import Path
from typing import Dict, List


class KGMerger:
    """Merge base KG with supplementary KG to create complete KG."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def merge_kg(
        self,
        base_kg_path: Path,
        supplement_kg_path: Path,
        output_path: Path,
    ) -> Dict[str, int]:
        """
        Merge base KG (with empty triplets) with supplement KG.

        Args:
            base_kg_path: Path to base KG file (may contain empty triplets)
            supplement_kg_path: Path to supplement KG file (fills empty triplets)
            output_path: Path to save merged KG

        Returns:
            Dictionary with merge statistics
        """
        self.logger.info(f"Loading base KG from {base_kg_path}")
        with open(base_kg_path, "r", encoding="utf-8") as f:
            base_kg = json.load(f)

        self.logger.info(f"Loading supplement KG from {supplement_kg_path}")
        with open(supplement_kg_path, "r", encoding="utf-8") as f:
            supplement_kg = json.load(f)

        # Create a dictionary for quick lookup by iid
        supplement_dict = {item["iid"]: item for item in supplement_kg}

        stats = {
            "base_items": len(base_kg),
            "supplement_items": len(supplement_kg),
            "empty_filled": 0,
            "empty_remaining": 0,
            "non_empty_preserved": 0,
        }

        # Merge the KGs
        merged_kg = []
        for base_item in base_kg:
            iid = base_item["iid"]
            base_triplets = base_item.get("triplets", [])

            # If base has empty triplets and supplement has data, use supplement
            if (
                not base_triplets or len(base_triplets) == 0
            ) and iid in supplement_dict:
                supplement_item = supplement_dict[iid]
                supplement_triplets = supplement_item.get("triplets", [])

                if supplement_triplets and len(supplement_triplets) > 0:
                    # Use supplement triplets
                    merged_item = {
                        "iid": iid,
                        "triplets": supplement_triplets,
                    }
                    stats["empty_filled"] += 1
                    self.logger.debug(
                        f"IID {iid}: Filled empty triplets with {len(supplement_triplets)} triplets"
                    )
                else:
                    # Both are empty
                    merged_item = base_item
                    stats["empty_remaining"] += 1
                    self.logger.debug(f"IID {iid}: Remains empty (no supplement data)")
            elif not base_triplets or len(base_triplets) == 0:
                # Base is empty but no supplement available
                merged_item = base_item
                stats["empty_remaining"] += 1
                self.logger.debug(f"IID {iid}: Remains empty (no supplement available)")
            else:
                # Base has triplets, preserve them
                merged_item = base_item
                stats["non_empty_preserved"] += 1

            merged_kg.append(merged_item)

        # Save merged KG
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Writing merged KG to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_kg, f, indent=4, ensure_ascii=False)

        # Log statistics
        self.logger.info("Merge statistics:")
        self.logger.info(f"  Base items: {stats['base_items']}")
        self.logger.info(f"  Supplement items: {stats['supplement_items']}")
        self.logger.info(f"  Empty triplets filled: {stats['empty_filled']}")
        self.logger.info(f"  Empty triplets remaining: {stats['empty_remaining']}")
        self.logger.info(
            f"  Non-empty triplets preserved: {stats['non_empty_preserved']}"
        )
        self.logger.info("✓ Merge completed successfully")

        return stats
