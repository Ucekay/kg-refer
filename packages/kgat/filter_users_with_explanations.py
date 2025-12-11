import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_explanation_pairs(explanation_path: Path) -> Set[Tuple[int, int]]:
    """
    explanation.json は 1 行 1 JSON オブジェクト (uid, iid, explanation) の
    JSON Lines 形式になっている前提で、(uid, iid) のセットを返す。
    """
    pairs: Set[Tuple[int, int]] = set()

    with explanation_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            uid = int(obj["uid"])
            iid = int(obj["iid"])
            pairs.add((uid, iid))

    return pairs


def filter_users_with_top_hits(
    input_csv: Path,
    explanation_json: Path,
    output_csv: Path,
) -> None:
    """
    users_with_top20_hits.csv から、datasets/yelp/explanation.json に存在する
    (user_id, item_id) の組み合わせだけを残して新しい CSV を出力する。

    - 入力 CSV のカラム: user_id,n_hits,n_test_items,hit_items,hit_ranks
    - hit_items は ';' 区切りの item_id 群
    - hit_ranks は ';' 区切りの rank 群 (hit_items と同じ長さを想定)
    """
    explanation_pairs = load_explanation_pairs(explanation_json)

    with input_csv.open("r", encoding="utf-8", newline="") as fin, output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("入力 CSV にヘッダーがありません")

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            user_id = int(row["user_id"])
            hit_items_str = row["hit_items"]
            hit_ranks_str = row["hit_ranks"]

            if not hit_items_str:
                continue

            item_ids: List[int] = [int(x) for x in hit_items_str.split(";")]
            ranks: List[str] = hit_ranks_str.split(";") if hit_ranks_str else []

            # ranks の長さが item_ids と違う場合は、短い方に合わせる
            if ranks and len(ranks) != len(item_ids):
                min_len = min(len(ranks), len(item_ids))
                item_ids = item_ids[:min_len]
                ranks = ranks[:min_len]

            kept_items: List[int] = []
            kept_ranks: List[str] = []

            for idx, item_id in enumerate(item_ids):
                if (user_id, item_id) in explanation_pairs:
                    kept_items.append(item_id)
                    if ranks:
                        kept_ranks.append(ranks[idx])

            # 一件も残らなければそのユーザ行はスキップ
            if not kept_items:
                continue

            row["n_hits"] = str(len(kept_items))
            row["hit_items"] = ";".join(str(i) for i in kept_items)
            row["hit_ranks"] = ";".join(kept_ranks) if kept_ranks else ""

            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "users_with_top20_hits.csv から "
            "datasets/yelp/explanation.json に存在する (uid, iid) だけを残した CSV を作成します。"
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("packages/kgat/output/users_with_top20_hits.csv"),
        help="入力 CSV パス (デフォルト: packages/kgat/output/users_with_top20_hits.csv)",
    )
    parser.add_argument(
        "--explanation-json",
        type=Path,
        default=Path("datasets/yelp/explanation.json"),
        help="explanation.json パス (JSON Lines 形式, デフォルト: datasets/yelp/explanation.json)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("packages/kgat/output/users_with_top20_hits.filtered.csv"),
        help="出力 CSV パス (デフォルト: packages/kgat/output/users_with_top20_hits.filtered.csv)",
    )

    args = parser.parse_args()

    filter_users_with_top_hits(
        input_csv=args.input_csv,
        explanation_json=args.explanation_json,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()





