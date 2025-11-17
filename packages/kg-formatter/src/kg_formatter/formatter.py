import json
import logging
from pathlib import Path
from typing import Dict, Set, Tuple


class KGFormatter:
    """Knowledge Graph formatter for ID remapping and file generation."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def format_kg(
        self,
        input_path: Path,
        output_dir: Path,
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Create ID mappings and formatted files from cleaned KG JSON.

        Args:
            input_path: Path to the cleaned_kg JSON file
            output_dir: Directory to save output files

        Returns:
            Tuple of (relation_to_id, entity_to_id) dictionaries
        """
        self.logger.info(f"Loading KG data from {input_path}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load JSON data
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.logger.info(f"Loaded {len(data)} items")

        # Collect unique relations and tail entities
        relations: Set[str] = set()
        tail_entities: Set[str] = set()

        for item in data:
            triplets = item.get("triplets", [])
            for triplet in triplets:
                if len(triplet) >= 3:
                    _, relation, tail = triplet[0], triplet[1], triplet[2]
                    relations.add(relation)
                    tail_entities.add(tail)

        self.logger.info(f"Found {len(relations)} unique relations")
        self.logger.info(f"Found {len(tail_entities)} unique tail entities")

        # Sort for consistent ordering
        sorted_relations = sorted(relations)
        sorted_entities = sorted(tail_entities)

        # Create ID mappings (0-indexed)
        relation_to_id = {rel: idx for idx, rel in enumerate(sorted_relations)}
        entity_to_id = {ent: idx for idx, ent in enumerate(sorted_entities)}

        # Save relation_list.txt
        relation_list_path = output_dir / "relation_list.txt"
        self.logger.info(f"Writing relation list to {relation_list_path}")
        with open(relation_list_path, "w", encoding="utf-8") as f:
            for relation, remap_id in relation_to_id.items():
                f.write(f'"{relation}" {remap_id}\n')

        # Save entity_list.txt
        entity_list_path = output_dir / "entity_list.txt"
        self.logger.info(f"Writing entity list to {entity_list_path}")
        with open(entity_list_path, "w", encoding="utf-8") as f:
            for entity, remap_id in entity_to_id.items():
                f.write(f'"{entity}" {remap_id}\n')

        # Save kg_final.txt
        kg_final_path = output_dir / "kg_final.txt"
        self.logger.info(f"Writing final KG to {kg_final_path}")

        triplet_count = 0
        skipped_count = 0
        with open(kg_final_path, "w", encoding="utf-8") as f:
            for item in data:
                triplets = item.get("triplets", [])
                for triplet in triplets:
                    if len(triplet) >= 3:
                        head, relation, tail = triplet[0], triplet[1], triplet[2]

                        # Check if head is numeric (valid item ID)
                        try:
                            head_id = int(head)
                        except (ValueError, TypeError):
                            self.logger.warning(
                                f"Skipping triplet with non-numeric head: {triplet}"
                            )
                            skipped_count += 1
                            continue

                        relation_id = relation_to_id[relation]
                        tail_id = entity_to_id[tail]
                        # Use validated head_id
                        f.write(f"{head_id} {relation_id} {tail_id}\n")
                        triplet_count += 1

        self.logger.info(f"Wrote {triplet_count} triplets")
        if skipped_count > 0:
            self.logger.warning(
                f"Skipped {skipped_count} triplets with non-numeric heads"
            )
        self.logger.info("✓ Formatting completed successfully")

        return relation_to_id, entity_to_id
