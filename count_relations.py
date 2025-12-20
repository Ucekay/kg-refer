import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_kg(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_relations(data: list[dict[str, Any]]) -> Counter:
    counter: Counter = Counter()
    for item in data:
        triplets = item.get("triplets", [])
        for triplet in triplets:
            if len(triplet) < 2:
                continue
            relation = triplet[1]
            counter[relation] += 1
    return counter


def save_counts(counter: Counter, output_path: Path) -> None:
    """
    関係とその出現頻度を多い順に並べて保存する。
    フォーマット: relation<TAB>count
    """
    lines = [
        f"{relation}\t{count}"
        for relation, count in counter.most_common()
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Yelp 知識グラフから関係の登場頻度を集計して保存するスクリプト"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/yelp/merged_kg_2.json"),
        help="入力となる知識グラフ JSON ファイルのパス",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/yelp/relation_frequencies.tsv"),
        help="出力ファイルのパス (relation<TAB>count)",
    )
    args = parser.parse_args()

    data = load_kg(args.input)
    counter = count_relations(data)
    save_counts(counter, args.output)


if __name__ == "__main__":
    main()




