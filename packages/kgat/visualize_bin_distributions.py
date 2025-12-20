"""
訓練インタラクション数ビンごとのスコア分布を，
モデル間で箱ひげ図として比較するスクリプト。

- 各モデルの `cf_scores.npy` と `train.txt` / `test.txt` を使って，
  ユーザーごとの precision@K / recall@K / ndcg@K を再計算
- 訓練インタラクション数に応じてビン分割（0-9, 10-19, ..., 50-59, 60+）
- ビン × モデルごとにスコア分布を箱ひげ図で可視化
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
        return UserMetrics(precision=0.0, recall=0.0, ndcg=0.0)

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

    return UserMetrics(precision=precision, recall=recall, ndcg=ndcg)


def collect_user_bin_metrics_for_model(
    model_name: str,
    cf_scores_path: str,
    data_dir: str,
    top_k: int,
) -> List[Dict[str, object]]:
    """
    1 つのモデルについて，ユーザーごとのビンとスコアを収集し，
    可視化用のレコード一覧を返す。
    """
    cf_scores = np.load(cf_scores_path)

    train_path = Path(data_dir) / "train.txt"
    test_path = Path(data_dir) / "test.txt"

    train_user_dict = load_interactions(str(train_path))
    test_user_dict = load_interactions(str(test_path))

    records: List[Dict[str, object]] = []

    for user_id, test_items in test_user_dict.items():
        if user_id >= cf_scores.shape[0]:
            # cf_scores に存在しないユーザーはスキップ
            continue

        train_items = train_user_dict.get(user_id)
        n_train = len(train_items) if train_items is not None else 0
        bin_label = assign_bin(n_train)

        user_scores = cf_scores[user_id]
        metrics = compute_user_metrics(user_scores, test_items, top_k=top_k)

        records.append(
            {
                "model": model_name,
                "user_id": user_id,
                "interaction_bin": bin_label,
                f"precision@{top_k}": metrics.precision,
                f"recall@{top_k}": metrics.recall,
                f"ndcg@{top_k}": metrics.ndcg,
            }
        )

    return records


def main():
    parser = argparse.ArgumentParser(
        description=(
            "訓練インタラクション数ビンごとのスコア分布を，"
            "モデル間で箱ひげ図として可視化する"
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("NAME", "CF_SCORES_PATH"),
        help="モデル名と cf_scores.npy のパスをペアで指定（複数指定可）",
        required=True,
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
        "--output-dir",
        type=str,
        default="packages/kgat/output/boxplots",
        help="箱ひげ図を保存するディレクトリ",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default="",
        help=(
            "ユーザーごとのスコアを CSV として保存するパス "
            "(指定しない場合は保存しない)"
        ),
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, object]] = []
    for name, cf_path in args.model:
        print(f"モデル {name}: {cf_path} からスコアを読み込み中...")
        records = collect_user_bin_metrics_for_model(
            model_name=name,
            cf_scores_path=cf_path,
            data_dir=args.data_dir,
            top_k=args.top_k,
        )
        all_records.extend(records)

    if not all_records:
        print("レコードが空です。入力を確認してください。")
        return

    df = pd.DataFrame(all_records)

    # ビンの順序を明示的に指定（categorical 型に変換）
    bin_order = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+"]
    df["interaction_bin"] = pd.Categorical(
        df["interaction_bin"], categories=bin_order, ordered=True
    )

    # 任意で生データを CSV 保存
    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"ユーザーごとのスコアを保存しました: {csv_path}")

    sns.set(style="whitegrid")

    metrics = [f"precision@{args.top_k}", f"recall@{args.top_k}", f"ndcg@{args.top_k}"]
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=df,
            x="interaction_bin",
            y=metric,
            hue="model",
            showfliers=True,  # 外れ値も表示
        )
        plt.title(f"{metric} by interaction bin (Top@{args.top_k})")
        plt.xlabel("Interaction bin (train)")
        plt.ylabel(metric)
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        out_path = output_dir / f"boxplot_{metric.replace('@', '_at_')}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"箱ひげ図を保存しました: {out_path}")


if __name__ == "__main__":
    main()


