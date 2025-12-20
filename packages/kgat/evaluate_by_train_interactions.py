"""
訓練データのインタラクション数ごとにユーザーをビン分割し，
各ビンのユーザーに対して Top@20 で推薦性能を評価するスクリプト。

評価には，既に計算済みの `cf_scores.npy` を利用する。
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_interactions(path: str) -> Dict[int, np.ndarray]:
    """train.txt / test.txt を読み込み，ユーザーごとのアイテム ID 配列を返す"""
    user_dict: Dict[int, np.ndarray] = {}
    file_path = Path(path)

    with file_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                user_id = int(parts[0])
                item_ids = [int(x) for x in parts[1:]]
                user_dict[user_id] = np.array(item_ids, dtype=np.int32)

    return user_dict


def assign_bin(n_interactions: int) -> str:
    """
    ユーザーの訓練インタラクション数からビンラベルを決定する。

    - 0〜59: 10 ごとのビン（例: 0-9, 10-19, ..., 50-59）
    - 60 以上: 60+
    """
    if n_interactions >= 60:
        return "60+"
    lower = (n_interactions // 10) * 10
    upper = lower + 9
    return f"{lower}-{upper}"


@dataclass
class UserMetrics:
    hits: int
    precision: float
    recall: float
    ndcg: float


def dcg_at_k(hit_ranks: List[int], k: int) -> float:
    """与えられたヒット順位から DCG@k を計算する（rank は 1 始まり）。"""
    dcg = 0.0
    for r in hit_ranks:
        if r <= k:
            dcg += 1.0 / np.log2(r + 1.0)
    return dcg


def compute_user_metrics(
    user_scores: np.ndarray, test_items: np.ndarray, top_k: int = 20
) -> UserMetrics:
    """
    1 ユーザーについて Top@K の Precision/Recall/NDCG を計算する。
    """
    if test_items.size == 0:
        return UserMetrics(hits=0, precision=0.0, recall=0.0, ndcg=0.0)

    test_set = set(int(x) for x in test_items.tolist())
    top_indices = np.argsort(user_scores)[::-1][:top_k]

    hits = 0
    hit_ranks: List[int] = []
    for rank, item_id in enumerate(top_indices, 1):
        if item_id in test_set:
            hits += 1
            hit_ranks.append(rank)

    precision = hits / top_k
    recall = hits / len(test_set) if len(test_set) > 0 else 0.0

    # NDCG@K: DCG を理想 DCG で正規化
    dcg = dcg_at_k(hit_ranks, top_k)
    ideal_hits = min(len(test_set), top_k)
    ideal_ranks = list(range(1, ideal_hits + 1))
    ideal_dcg = dcg_at_k(ideal_ranks, top_k) if ideal_hits > 0 else 1.0
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    return UserMetrics(hits=hits, precision=precision, recall=recall, ndcg=ndcg)


def evaluate_by_bins(
    cf_scores_path: str,
    data_dir: str,
    top_k: int = 20,
) -> Tuple[
    List[Tuple[str, int, float, float, float]],
    Tuple[float, float, float],
]:
    """
    ビンごとにユーザーを集計し，Top@K の平均 Precision/Recall/NDCG を返す。

    Returns:
        (
            per_bin_results,  # List of (bin_label, n_users, precision, recall, ndcg)
            overall_metrics,  # (precision, recall, ndcg) for all users
        )
    """
    cf_scores = np.load(cf_scores_path)

    train_path = Path(data_dir) / "train.txt"
    test_path = Path(data_dir) / "test.txt"

    train_user_dict = load_interactions(str(train_path))
    test_user_dict = load_interactions(str(test_path))

    # ビンごとのメトリクス集計
    bin_metrics: Dict[str, List[UserMetrics]] = {}
    all_user_metrics: List[UserMetrics] = []

    for user_id, test_items in test_user_dict.items():
        if user_id >= cf_scores.shape[0]:
            # cf_scores に存在しないユーザーはスキップ
            continue

        train_items = train_user_dict.get(user_id)
        n_train = len(train_items) if train_items is not None else 0
        bin_label = assign_bin(n_train)

        user_scores = cf_scores[user_id]
        metrics = compute_user_metrics(user_scores, test_items, top_k=top_k)

        bin_metrics.setdefault(bin_label, []).append(metrics)
        all_user_metrics.append(metrics)

    # ビンラベルを昇順ソート（0-9, 10-19, ... , 50-59, 60+）
    def bin_sort_key(label: str):
        if label.endswith("+"):
            return (999, 0)
        lower = int(label.split("-")[0])
        return (lower, 0)

    results: List[Tuple[str, int, float, float, float]] = []
    for bin_label in sorted(bin_metrics.keys(), key=bin_sort_key):
        metrics_list = bin_metrics[bin_label]
        n_users = len(metrics_list)
        if n_users == 0:
            continue
        avg_precision = float(
            np.mean([m.precision for m in metrics_list], dtype=np.float64)
        )
        avg_recall = float(
            np.mean([m.recall for m in metrics_list], dtype=np.float64)
        )
        avg_ndcg = float(np.mean([m.ndcg for m in metrics_list], dtype=np.float64))
        results.append((bin_label, n_users, avg_precision, avg_recall, avg_ndcg))

    # 全ユーザーの平均
    if all_user_metrics:
        overall_precision = float(
            np.mean([m.precision for m in all_user_metrics], dtype=np.float64)
        )
        overall_recall = float(
            np.mean([m.recall for m in all_user_metrics], dtype=np.float64)
        )
        overall_ndcg = float(
            np.mean([m.ndcg for m in all_user_metrics], dtype=np.float64)
        )
    else:
        overall_precision = overall_recall = overall_ndcg = 0.0

    return results, (overall_precision, overall_recall, overall_ndcg)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "訓練インタラクション数ごとのビンでユーザーを分割し，"
            "各ビンについて Top@20 で推薦性能を評価して CSV に保存する"
        )
    )
    parser.add_argument(
        "--cf-scores-path",
        type=str,
        default=(
            "packages/kgat/trained_model/KGAT/yelp/"
            "embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.001_pretrain0/cf_scores.npy"
        ),
        help="cf_scores.npy のパス（既に predict モードで生成済みのもの）",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="packages/kgat/datasets/yelp",
        help="train.txt / test.txt を含むデータディレクトリ",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top@K で評価する K の値（デフォルト: 20）",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=(
            "packages/kgat/output/"
            "evaluation_by_train_interactions_top20.csv"
        ),
        help="ビンごとの評価結果を保存する CSV ファイルパス",
    )

    args = parser.parse_args()

    print("cf_scores とデータを読み込み中...")
    per_bin_results, overall = evaluate_by_bins(
        cf_scores_path=args.cf_scores_path,
        data_dir=args.data_dir,
        top_k=args.top_k,
    )
    overall_precision, overall_recall, overall_ndcg = overall

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"結果を CSV に保存中: {output_path}")
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "interaction_bin",
                "n_users",
                f"precision@{args.top_k}",
                f"recall@{args.top_k}",
                f"ndcg@{args.top_k}",
                f"precision@{args.top_k}_norm",
                f"recall@{args.top_k}_norm",
                f"ndcg@{args.top_k}_norm",
            ]
        )
        # 先頭に全体平均行を追加（比較しやすくするため）
        writer.writerow(
            [
                "ALL",
                "",
                overall_precision,
                overall_recall,
                overall_ndcg,
                1.0,  # 全体に対する正規化なので常に1
                1.0,
                1.0,
            ]
        )
        # 各ビンの値と，全体との比（正規化値）を出力
        for bin_label, n_users, precision, recall, ndcg in per_bin_results:
            prec_norm = precision / overall_precision if overall_precision > 0 else 0.0
            rec_norm = recall / overall_recall if overall_recall > 0 else 0.0
            ndcg_norm = ndcg / overall_ndcg if overall_ndcg > 0 else 0.0
            writer.writerow(
                [
                    bin_label,
                    n_users,
                    precision,
                    recall,
                    ndcg,
                    prec_norm,
                    rec_norm,
                    ndcg_norm,
                ]
            )

    print("完了しました。")


if __name__ == "__main__":
    main()


