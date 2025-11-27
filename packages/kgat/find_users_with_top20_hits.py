"""
Top20にヒットがあるユーザー一覧を取得するスクリプト
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Top20にヒットがあるユーザー一覧を取得"
    )
    parser.add_argument(
        "--cf-scores-path",
        type=str,
        default="packages/kgat/trained_model/KGAT/yelp/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain2/cf_scores.npy",
        help="cf_scores.npyのパス",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="packages/kgat/datasets/yelp",
        help="データディレクトリのパス",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="packages/kgat/output/users_with_top20_hits.csv",
        help="出力CSVファイルのパス",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="上位何件を対象とするか",
    )
    parser.add_argument(
        "--show-limit",
        type=int,
        default=50,
        help="画面に表示するユーザー数",
    )

    args = parser.parse_args()

    # データ読み込み
    print("データを読み込み中...")
    cf_scores = np.load(args.cf_scores_path)
    print(f"cf_scores shape: {cf_scores.shape}")

    test_file = Path(args.data_dir) / "test.txt"
    test_user_dict = {}
    with open(test_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                user_id = int(parts[0])
                item_ids = [int(x) for x in parts[1:]]
                test_user_dict[user_id] = np.array(item_ids, dtype=np.int32)

    print(f"テストユーザー数: {len(test_user_dict)}")

    # Top-Kにヒットがあるユーザーを抽出
    print(f"\nTop{args.top_k}にヒットがあるユーザーを抽出中...")
    users_with_hits = []
    for user_id, test_items in test_user_dict.items():
        if user_id >= cf_scores.shape[0]:
            continue
        user_scores = cf_scores[user_id]
        top_k = np.argsort(user_scores)[::-1][: args.top_k]

        # 推薦順位順にヒットを取得
        hit_items = []
        hit_ranks = []
        for rank, item in enumerate(top_k, 1):
            if item in test_items:
                hit_items.append(item)
                hit_ranks.append(rank)

        if len(hit_items) > 0:
            users_with_hits.append(
                {
                    "user_id": user_id,
                    "n_hits": len(hit_items),
                    "n_test_items": len(test_items),
                    "hit_items": hit_items,
                    "hit_ranks": hit_ranks,
                }
            )

    # ヒット数でソート（多い順）
    users_with_hits.sort(key=lambda x: (-x["n_hits"], x["user_id"]))

    # 結果表示
    print(f"\n{'='*70}")
    print(f"Top{args.top_k}にヒットがあるユーザー数: {len(users_with_hits)} / {len(test_user_dict)}")
    print(f"割合: {len(users_with_hits) / len(test_user_dict) * 100:.2f}%")

    print(f"\n=== ヒット数別の分布 ===")
    hit_counts = Counter(u["n_hits"] for u in users_with_hits)
    for n_hits in sorted(hit_counts.keys(), reverse=True):
        print(f"  {n_hits}ヒット: {hit_counts[n_hits]}ユーザー")

    print(f"\n=== Top{args.top_k}にヒットがあるユーザー一覧（上位{args.show_limit}件） ===")
    print(f"{'ユーザーID':>10} | {'ヒット数':>8} | {'テスト数':>8} | ヒットアイテム（順位）")
    print("-" * 80)
    for u in users_with_hits[: args.show_limit]:
        hit_info = ", ".join(
            [f"{item}({rank}位)" for item, rank in zip(u["hit_items"], u["hit_ranks"])]
        )
        print(f"{u['user_id']:>10} | {u['n_hits']:>8} | {u['n_test_items']:>8} | {hit_info}")

    # CSVとして保存
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "n_hits", "n_test_items", "hit_items", "hit_ranks"])
        for u in users_with_hits:
            writer.writerow(
                [
                    u["user_id"],
                    u["n_hits"],
                    u["n_test_items"],
                    ";".join(map(str, u["hit_items"])),
                    ";".join(map(str, u["hit_ranks"])),
                ]
            )

    print(f"\n全ユーザーリストを保存: {output_path}")
    print(f"保存件数: {len(users_with_hits)}件")


if __name__ == "__main__":
    main()

