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
            variations[lower_entity] = {
                "variants": sorted(list(variants)),
                "count": len(variants),
                "total_occurrences": sum(entities.count(v) for v in variants),
            }

    return variations


def main():
    input_file = "/home/kimura/repos/kg-refer/packages/entity-canonicalizer/data/canon_kg_filtered.json"
    output_file = "/home/kimura/repos/kg-refer/packages/entity-canonicalizer/data/case_variations.json"

    variations = find_case_variations(input_file)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(variations, f, ensure_ascii=False, indent=2)

    print(f"{len(variations)} variations found")

    text_output_file = "/home/kimura/repos/kg-refer/packages/entity-canonicalizer/data/case_variations.txt"
    with open(text_output_file, "w", encoding="utf-8") as f:
        sorted_variations = sorted(
            variations.items(), key=lambda x: x[1]["total_occurrences"], reverse=True
        )

        for i, (lower_entity, info) in enumerate(sorted_variations, 1):
            f.write(
                f"{i}. {lower_entity}: {', '.join(info['variants'])} ({info['total_occurrences']})\n"
            )


if __name__ == "__main__":
    main()
