#!/usr/bin/env python3
import json
from collections import defaultdict


def find_case_variations(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entities = []
    for entry in data:
        if "triplets" in entry:
            for triplet in entry["triplets"]:
                if len(triplet) >= 3:
                    subject, relation, obj = triplet[0], triplet[1], triplet[2]

                    if not (
                        isinstance(subject, (int, float))
                        or (isinstance(subject, str) and subject.isdigit())
                    ):
                        entities.append(subject)
                    if not (
                        isinstance(obj, (int, float))
                        or (isinstance(obj, str) and obj.isdigit())
                    ):
                        entities.append(obj)

    case_groups = defaultdict(set)
    for entity in entities:
        if isinstance(entity, str):
            lower_entity = entity.lower()
            case_groups[lower_entity].add(entity)

    variations = {}
    for lower_entity, variants in case_groups.items():
        if len(variants) > 1:
            variations[lower_entity] = set(variants)

    return variations


def normalize_case_in_triplets(data, variations):
    normalized_data = []

    for entry in data:
        if "triplets" in entry:
            normalized_triplets = []
            for triplet in entry["triplets"]:
                if len(triplet) >= 3:
                    subject, relation, obj = triplet[0], triplet[1], triplet[2]

                    # 主体を正規化
                    if isinstance(subject, str) and not subject.isdigit():
                        for lower_entity, variants in variations.items():
                            if subject in variants:
                                subject = lower_entity
                                break

                    # 対象を正規化
                    if isinstance(obj, str) and not obj.isdigit():
                        for lower_entity, variants in variations.items():
                            if obj in variants:
                                obj = lower_entity
                                break

                    normalized_triplets.append([subject, relation, obj])

            if normalized_triplets:
                normalized_entry = entry.copy()
                normalized_entry["triplets"] = normalized_triplets
                normalized_data.append(normalized_entry)

    return normalized_data


def main():
    input_file = "/home/kimura/repos/kg-refer/packages/entity-canonicalizer/data/canon_kg_filtered.json"
    output_file = "/home/kimura/repos/kg-refer/packages/entity-canonicalizer/data/canon_kg_normalized.json"

    variations = find_case_variations(input_file)
    print(f"Found {len(variations)} case variations")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized_data = normalize_case_in_triplets(data, variations)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=2)

    print(f"Normalized data saved to {output_file}")
    print(f"Original entries: {len(data)}")
    print(f"Normalized entries: {len(normalized_data)}")


if __name__ == "__main__":
    main()
