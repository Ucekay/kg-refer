import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Terms to filter out from entities
FILTERED_TERMS = [
    "Unknown",
    "unknown",
    "Cuisine",
    "cuisine",
    "Food",
    "food",
    "Users",
    "users",
    "unspecified",
    "dishes",
    "Dishes",
    "not specified",
    "Not specified",
    "food options",
    "meal",
]

# Relations to filter out (remove triplets with these relations)
# Format: ["relation1", "relation2", ...]
FILTERED_RELATIONS = [
    # Example: "appreciates", "likes", etc.
    # Add relations to filter here:
    "appreciates",
    "interested in",
]

# Entity-only replacement rules: Replace tail entity regardless of relation
# Format: {old_tail: new_tail}
ENTITY_REPLACEMENTS = {
    # Example: "Food allergies" -> "food allergies" (normalize case)
    # Add your entity replacements here:
    "Café": "Cafe",
}

# Entity expansion rules: Replace one tail entity with multiple tail entities
# Format: {tail_to_replace: [list_of_replacement_tails]}
ENTITY_EXPANSION_RULES = {
    # Example: "Cajun/Creole" will create two triplets with "Cajun" and "Creole"
    # "Cajun/Creole": ["Cajun", "Creole"],
    # Add your expansion rules here:
    "Cajun/Creole": ["Cajun cuisine", "Creole cuisine"],
    "Cajun/Creole cuisine": ["Cajun cuisine", "Creole cuisine"],
}

# Replacement rules for (relation, tail) combinations
# Format: [(relation_pattern, tail_pattern, new_relation, new_tail), ...]
# Use None to keep the original value
RELATION_TAIL_REPLACEMENTS = [
    # Example: ("serves", "Italian cuisine", "serves", "Italian")
    # Example: ("has feature", "big portions", "has characteristic", "big portions")
    # Add your rules here:
    # ("old_relation", "old_tail", "new_relation", "new_tail"),
    ("serves", "Italian cuisine", "serves", "Italian"),
    ("serves", "Italian food", "serves", "Italian"),
    ("serves", "Polish cuisine", "serves", "Polish"),
    ("serves", "Polish food", "serves", "Polish"),
    ("appeals to", "Fans of Polish food", "appeals to", "Fans of Polish"),
    ("appeals to", "Fans of Polish cuisine", "appeals to", "Fans of Polish"),
    ("appeals to", "Fans of Italian food", "appeals to", "Fans of Italian"),
    ("appeals to", "Fans of Italian cuisine", "appeals to", "Fans of Italian"),
    ("serves", "Cajun food", "serves", "Cajun cuisine"),
    ("serves", "Creole food", "serves", "Creole cuisine"),
    ("serves", "French cuisine", "serves", "French"),
    ("serves", "French food", "serves", "French"),
    ("serves", "Japanese food", "serves", "Japanese cuisine"),
    ("serves", "seafood dishes", "serves", "seafood"),
    ("has atmosphere", "cozy atmosphere", "has atmosphere", "cozy"),
    ("serves", "American", "serves", "American cuisine"),
    ("serves", "American food", "serves", "American cuisine"),
    ("serves", "Mexican cuisine", "serves", "Mexican"),
    ("serves", "Mexican food", "serves", "Mexican"),
    ("serves", "Thai dishes", "serves", "Thai cuisine"),
    ("serves", "Thai food", "serves", "Thai cioisine"),
    ("cators to", "food allergies", "accommodates", "food allergies"),
    ("serves", "Asian fusion cuisine", "serves", "Asian fusion"),
    ("offers", "Asian fusion cuisine", "serves", "Asian fusion"),
    ("price range", "reasonable prices", "price range", "reasonable"),
    ("price range", "reasonable price", "price range", "reasonable"),
    ("serves", "Greek food", "serves", "Greek cuisine"),
    ("serves", "restaurant", "is a", "restaurant"),
    ("category", "restaurant", "is a", "restaurant"),
    ("category", "cafe", "is a", "cafe"),
    ("category", "bar", "is a", "bar"),
    ("category", "pub", "is a", "pub"),
    ("has feature", "family-friendly atmosphere", "has atmosphere", "family-friendly"),
    (
        "has atmosphere",
        "family-friendly atmosphere",
        "has atmosphere",
        "family-friendly",
    ),
]

# Tail-based relation unification rules
# When the same tail appears with multiple relations in one iid,
# if the specified (relation, tail) combination exists, keep only that one.
# Format: {tail: preferred_relation}
# Example: {"big portions": "serves"} means:
#   If tail "big portions" appears with multiple relations (e.g., "has feature" and "serves"),
#   keep only the triplet with relation "serves" and remove others.
TAIL_RELATION_UNIFICATION_RULES = {
    # Add your tail-based unification rules here:
    "coffee": "serves",
    "cakes": "serves",
    "soups": "serves",
    "cozy": "has atmosphere",
    "restaurant": "is a",
    "cafe": "is a",
}


class KGCleaner:
    """Knowledge Graph cleaner for normalizing and deduplicating triplets."""

    def __init__(self):
        self.stats = {
            "total_triplets": 0,
            "normalized_count": 0,
            "duplicate_count": 0,
            "conflict_count": 0,
            "filtered_count": 0,
            "replaced_count": 0,
            "filtered_relations_count": 0,
            "entity_replaced_count": 0,
            "expanded_count": 0,
            "unified_count": 0,
        }

    def _should_normalize(self, value: str) -> bool:
        """Check if a value has case variations that should be normalized."""
        return value != value.lower() and value.lower() not in ["", "_"]

    def _collect_global_variations(
        self, input_data: List[Dict[str, Any]]
    ) -> Tuple[set, set]:
        """
        Collect global case variations across all data.
        Returns sets of relations and tails that should be normalized.
        """
        # Collect all unique relations and tails globally
        relations = defaultdict(set)
        tails = defaultdict(set)

        for item in input_data:
            for h, r, t in item["triplets"]:
                relations[r.lower()].add(r)
                tails[t.lower()].add(t)

        # Identify which should be normalized (have multiple case variations)
        relations_to_normalize = {
            orig
            for lower, variations in relations.items()
            if len(variations) > 1
            for orig in variations
        }
        tails_to_normalize = {
            orig
            for lower, variations in tails.items()
            if len(variations) > 1
            for orig in variations
        }

        logger.info(
            f"Found {len(relations_to_normalize)} relations and {len(tails_to_normalize)} tails to normalize globally"
        )

        return relations_to_normalize, tails_to_normalize

    def _normalize_relation_and_tail(
        self,
        triplets: List[List[str]],
        relations_to_normalize: set,
        tails_to_normalize: set,
    ) -> List[List[str]]:
        """
        Normalize relation and tail entities by converting to lowercase
        based on global case variations.
        """
        # Apply normalization
        normalized_triplets = []
        for h, r, t in triplets:
            normalized_r = r.lower() if r in relations_to_normalize else r
            normalized_t = t.lower() if t in tails_to_normalize else t

            if normalized_r != r or normalized_t != t:
                self.stats["normalized_count"] += 1

            normalized_triplets.append([h, normalized_r, normalized_t])

        return normalized_triplets

    def _expand_entities(self, triplets: List[List[str]]) -> List[List[str]]:
        """Expand tail entities based on expansion rules (one entity -> multiple entities)."""
        if not ENTITY_EXPANSION_RULES:
            return triplets

        expanded = []

        for h, r, t in triplets:
            # Check if tail matches any expansion rule
            if t in ENTITY_EXPANSION_RULES:
                # Create multiple triplets with expanded entities
                for new_t in ENTITY_EXPANSION_RULES[t]:
                    expanded.append([h, r, new_t])
                    self.stats["expanded_count"] += 1
                logger.debug(
                    f"Expanded: [{h}, {r}, {t}] → {len(ENTITY_EXPANSION_RULES[t])} triplets"
                )
            else:
                expanded.append([h, r, t])

        if self.stats["expanded_count"] > 0:
            logger.info(
                f"Expanded {self.stats['expanded_count']} tail entities into multiple triplets"
            )
        return expanded

    def _replace_entities(self, triplets: List[List[str]]) -> List[List[str]]:
        """Replace tail entities based on entity replacement rules."""
        if not ENTITY_REPLACEMENTS:
            return triplets

        replaced = []

        for h, r, t in triplets:
            # Check if tail matches any entity replacement rule
            if t in ENTITY_REPLACEMENTS:
                new_t = ENTITY_REPLACEMENTS[t]
                replaced.append([h, r, new_t])
                self.stats["entity_replaced_count"] += 1
                logger.debug(f"Replaced entity: [{h}, {r}, {t}] → [{h}, {r}, {new_t}]")
            else:
                replaced.append([h, r, t])

        if self.stats["entity_replaced_count"] > 0:
            logger.info(f"Replaced {self.stats['entity_replaced_count']} tail entities")
        return replaced

    def _replace_relation_tail_combinations(
        self, triplets: List[List[str]]
    ) -> List[List[str]]:
        """Replace specific (relation, tail) combinations based on rules."""
        if not RELATION_TAIL_REPLACEMENTS:
            return triplets

        processed = []

        for h, r, t in triplets:
            replaced = False
            # Check each replacement rule
            for old_r, old_t, new_r, new_t in RELATION_TAIL_REPLACEMENTS:
                if r == old_r and t == old_t:
                    # Apply replacement (None means keep original)
                    new_r = new_r if new_r is not None else r
                    new_t = new_t if new_t is not None else t
                    self.stats["replaced_count"] += 1
                    logger.debug(f"Replaced: [{h}, {r}, {t}] → [{h}, {new_r}, {new_t}]")
                    processed.append([h, new_r, new_t])
                    replaced = True
                    break

            if not replaced:
                processed.append([h, r, t])

        if self.stats["replaced_count"] > 0:
            logger.info(
                f"Replaced {self.stats['replaced_count']} relation-tail combinations"
            )
        return processed

    def _filter_triplets(self, triplets: List[List[str]]) -> List[List[str]]:
        """Remove triplets with filtered terms in head, relation, or tail."""
        filtered = []

        for h, r, t in triplets:
            # Check if any field exactly matches filtered terms
            if h in FILTERED_TERMS or r in FILTERED_TERMS or t in FILTERED_TERMS:
                self.stats["filtered_count"] += 1
                logger.debug(f"Filtered triplet: [{h}, {r}, {t}]")
                continue

            filtered.append([h, r, t])

        logger.info(
            f"Removed {self.stats['filtered_count']} triplets with filtered terms"
        )
        return filtered

    def _filter_relations(self, triplets: List[List[str]]) -> List[List[str]]:
        """Remove triplets with specific relations."""
        if not FILTERED_RELATIONS:
            return triplets

        filtered = []

        for h, r, t in triplets:
            if r in FILTERED_RELATIONS:
                self.stats["filtered_relations_count"] += 1
                logger.debug(f"Filtered relation: [{h}, {r}, {t}]")
                continue

            filtered.append([h, r, t])

        if self.stats["filtered_relations_count"] > 0:
            logger.info(
                f"Removed {self.stats['filtered_relations_count']} triplets with filtered relations"
            )
        return filtered

    def _deduplicate_triplets(self, triplets: List[List[str]]) -> List[List[str]]:
        """Remove exact duplicate triplets (h, r, t)."""
        seen = set()
        deduplicated = []

        for h, r, t in triplets:
            triplet_tuple = (h, r, t)
            if triplet_tuple not in seen:
                seen.add(triplet_tuple)
                deduplicated.append([h, r, t])
            else:
                self.stats["duplicate_count"] += 1

        logger.info(f"Removed {self.stats['duplicate_count']} duplicate triplets")
        return deduplicated

    def _unify_tail_relations(self, triplets: List[List[str]]) -> List[List[str]]:
        """
        Unify relations for tails that appear with multiple relations.
        If a tail matches the unification rules and the preferred relation exists,
        remove all other relations for that tail.

        Example:
            If TAIL_RELATION_UNIFICATION_RULES = {"big portions": "serves"}
            and triplets contain:
                ["Restaurant A", "has feature", "big portions"]
                ["Restaurant A", "serves", "big portions"]
            Then remove "has feature" and keep only "serves".
        """
        if not TAIL_RELATION_UNIFICATION_RULES:
            return triplets

        # Group triplets by tail to find tails with multiple relations
        tail_to_triplets = defaultdict(list)
        for h, r, t in triplets:
            tail_to_triplets[t].append([h, r, t])

        unified = []

        for tail, tail_triplets in tail_to_triplets.items():
            # If only one triplet for this tail, keep as is
            if len(tail_triplets) == 1:
                unified.extend(tail_triplets)
                continue

            # Multiple triplets with this tail - check for unification rule
            if tail in TAIL_RELATION_UNIFICATION_RULES:
                preferred_relation = TAIL_RELATION_UNIFICATION_RULES[tail]

                # Check if the preferred relation exists
                relations = {triplet[1] for triplet in tail_triplets}
                if preferred_relation in relations:
                    # Keep only triplets with the preferred relation
                    for triplet in tail_triplets:
                        if triplet[1] == preferred_relation:
                            unified.append(triplet)
                        else:
                            self.stats["unified_count"] += 1
                            logger.debug(
                                f"Unified: [{triplet[0]}, {triplet[1]}, {triplet[2]}] -> prefer relation '{preferred_relation}'"
                            )
                else:
                    # Preferred relation doesn't exist, keep all
                    unified.extend(tail_triplets)
            else:
                # No unification rule for this tail, keep all
                unified.extend(tail_triplets)

        if self.stats["unified_count"] > 0:
            logger.info(
                f"Unified {self.stats['unified_count']} relations based on tail unification rules"
            )

        return unified

    def _find_conflicts(
        self, triplets: List[List[str]]
    ) -> Dict[Tuple[str, str], List[str]]:
        """
        Find (h, t) pairs that have multiple relations.
        Returns dict: (head, tail) -> [list of relations]
        """
        ht_to_relations = defaultdict(set)

        for h, r, t in triplets:
            ht_to_relations[(h, t)].add(r)

        # Filter only conflicts (multiple relations for same h, t pair)
        conflicts = {
            ht: sorted(relations)
            for ht, relations in ht_to_relations.items()
            if len(relations) > 1
        }

        # Don't update stats here - will be updated after all iids are processed
        if conflicts:
            logger.debug(
                f"Found {len(conflicts)} (h, t) pairs with multiple relations in this iid"
            )

        return conflicts

    def clean_kg_data(
        self,
        input_data: List[Dict[str, Any]],
        normalize: bool = True,
        deduplicate: bool = True,
        find_conflicts: bool = True,
        filter_terms: bool = True,
        filter_relations: bool = True,
        expand_entities: bool = True,
        replace_entities: bool = True,
        replace_combinations: bool = True,
        unify_relations: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], List[str]]]:
        """
        Clean knowledge graph data.

        Args:
            input_data: List of dicts with 'iid' and 'triplets' keys
            normalize: Apply case normalization to relations and tails
            deduplicate: Remove exact duplicate triplets
            find_conflicts: Find (h, t) pairs with multiple relations
            filter_terms: Remove triplets containing filtered terms
            filter_relations: Remove triplets with specific relations
            expand_entities: Expand tail entities into multiple triplets
            replace_entities: Replace tail entities regardless of relation
            replace_combinations: Replace specific (relation, tail) combinations
            unify_relations: Unify relations for tails with multiple relations

        Returns:
            Tuple of (cleaned_data, conflicts_dict)
        """
        self.stats["total_triplets"] = sum(len(item["triplets"]) for item in input_data)
        logger.info(f"Processing {self.stats['total_triplets']} total triplets")

        # Step 0: Collect global case variations if normalize is enabled
        relations_to_normalize = set()
        tails_to_normalize = set()
        if normalize:
            relations_to_normalize, tails_to_normalize = (
                self._collect_global_variations(input_data)
            )

        cleaned_data = []
        all_conflicts = {}

        for item in input_data:
            iid = item["iid"]
            triplets = item["triplets"]

            # Step 0: Filter terms (optional)
            if filter_terms:
                triplets = self._filter_triplets(triplets)

            # Step 1: Filter relations (optional)
            if filter_relations:
                triplets = self._filter_relations(triplets)

            # Step 2: Expand entities (optional)
            if expand_entities:
                triplets = self._expand_entities(triplets)

            # Step 3: Replace entities (optional)
            if replace_entities:
                triplets = self._replace_entities(triplets)

            # Step 4: Replace combinations (optional)
            if replace_combinations:
                triplets = self._replace_relation_tail_combinations(triplets)

            # Step 5: Normalize relations and tails (optional)
            if normalize:
                triplets = self._normalize_relation_and_tail(
                    triplets, relations_to_normalize, tails_to_normalize
                )

            # Step 6: Deduplicate (optional)
            if deduplicate:
                triplets = self._deduplicate_triplets(triplets)

            # Step 7: Unify relations (optional)
            if unify_relations:
                triplets = self._unify_tail_relations(triplets)

            # Step 8: Find conflicts (optional)
            if find_conflicts:
                conflicts = self._find_conflicts(triplets)
                if conflicts:
                    all_conflicts.update(conflicts)

            cleaned_data.append({"iid": iid, "triplets": triplets})

        # Update conflict count after all iids are processed
        if find_conflicts:
            self.stats["conflict_count"] = len(all_conflicts)
            logger.info(
                f"Found {len(all_conflicts)} total (h, t) pairs with multiple relations"
            )

        logger.info(f"Cleaning statistics: {self.stats}")
        return cleaned_data, all_conflicts

    def clean_file(
        self,
        input_path: Path,
        output_path: Path,
        conflicts_path: Path,
        normalize: bool = True,
        deduplicate: bool = True,
        find_conflicts: bool = True,
        filter_terms: bool = True,
        filter_relations: bool = True,
        expand_entities: bool = True,
        replace_entities: bool = True,
        replace_combinations: bool = True,
        unify_relations: bool = True,
    ):
        """
        Clean a KG file and save results.

        Args:
            input_path: Path to input JSON file
            output_path: Path to save cleaned data
            conflicts_path: Path to save conflicts
            normalize: Apply case normalization to relations and tails
            deduplicate: Remove exact duplicate triplets
            find_conflicts: Find (h, t) pairs with multiple relations
            filter_terms: Remove triplets containing filtered terms
            filter_relations: Remove triplets with specific relations
            expand_entities: Expand tail entities into multiple triplets
            replace_entities: Replace tail entities regardless of relation
            replace_combinations: Replace specific (relation, tail) combinations
            unify_relations: Unify relations for tails with multiple relations
        """
        logger.info(f"Loading data from {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cleaned_data, conflicts = self.clean_kg_data(
            data,
            normalize=normalize,
            deduplicate=deduplicate,
            find_conflicts=find_conflicts,
            filter_terms=filter_terms,
            filter_relations=filter_relations,
            expand_entities=expand_entities,
            replace_entities=replace_entities,
            replace_combinations=replace_combinations,
            unify_relations=unify_relations,
        )

        # Save cleaned data
        logger.info(f"Saving cleaned data to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

        # Save conflicts in the same format as input canon_kg.json
        if find_conflicts and conflicts:
            # Use head entity as iid
            conflicts_data = []
            for (h, t), relations in sorted(conflicts.items()):
                triplets = [[h, r, t] for r in relations]
                conflicts_data.append(
                    {
                        "iid": h,
                        "triplets": triplets,
                    }
                )

            logger.info(
                f"Saving {len(conflicts_data)} conflict groups to {conflicts_path}"
            )
            with open(conflicts_path, "w", encoding="utf-8") as f:
                json.dump(conflicts_data, f, indent=2, ensure_ascii=False)
        else:
            logger.info("Conflict detection disabled or no conflicts found")

        logger.info("Cleaning complete!")
        logger.info(f"  Total triplets: {self.stats['total_triplets']}")
        if filter_terms:
            logger.info(f"  Filtered: {self.stats['filtered_count']}")
        if filter_relations:
            logger.info(
                f"  Filtered relations: {self.stats['filtered_relations_count']}"
            )
        if expand_entities:
            logger.info(f"  Entities expanded: {self.stats['expanded_count']}")
        if replace_entities:
            logger.info(f"  Entities replaced: {self.stats['entity_replaced_count']}")
        if replace_combinations:
            logger.info(f"  Combinations replaced: {self.stats['replaced_count']}")
        if normalize:
            logger.info(f"  Normalized: {self.stats['normalized_count']}")
        if deduplicate:
            logger.info(f"  Duplicates removed: {self.stats['duplicate_count']}")
        if unify_relations:
            logger.info(f"  Relations unified: {self.stats['unified_count']}")
        if find_conflicts:
            logger.info(f"  Conflicts found: {self.stats['conflict_count']}")
