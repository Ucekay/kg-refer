"""
推薦結果と実際のインタラクションを比較するスクリプト

cf_scores.npyから予測結果を取得し、テストセットの実際のインタラクションと比較します。
"""

import argparse
import numpy as np
from pathlib import Path


def load_test_data(data_dir: str) -> dict[int, np.ndarray]:
    """test.txtを読み込んでユーザーごとのインタラクションを返す"""
    test_file = Path(data_dir) / "test.txt"
    test_user_dict = {}
    
    with open(test_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                user_id = int(parts[0])
                item_ids = [int(x) for x in parts[1:]]
                test_user_dict[user_id] = np.array(item_ids, dtype=np.int32)
    
    return test_user_dict


def load_train_data(data_dir: str) -> dict[int, np.ndarray]:
    """train.txtを読み込んでユーザーごとのインタラクションを返す"""
    train_file = Path(data_dir) / "train.txt"
    train_user_dict = {}
    
    with open(train_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                user_id = int(parts[0])
                item_ids = [int(x) for x in parts[1:]]
                train_user_dict[user_id] = np.array(item_ids, dtype=np.int32)
    
    return train_user_dict


def analyze_user_recommendations(
    user_id: int,
    cf_scores: np.ndarray,
    test_user_dict: dict[int, np.ndarray],
    train_user_dict: dict[int, np.ndarray],
    top_k: int = 100,
    ks: list[int] = [20, 40, 60, 80, 100],
) -> dict:
    """
    特定ユーザーの推薦結果を分析する
    
    Returns:
        分析結果の辞書
    """
    if user_id not in test_user_dict:
        return {"error": f"User {user_id} not found in test set"}
    
    user_scores = cf_scores[user_id]
    test_items = set(test_user_dict[user_id])
    train_items = set(train_user_dict.get(user_id, []))
    
    # 上位K件の推薦アイテムを取得
    top_indices = np.argsort(user_scores)[::-1][:top_k]
    
    # ヒットしたアイテムを特定
    hits = []
    for rank, item_id in enumerate(top_indices, 1):
        if item_id in test_items:
            hits.append({
                "item_id": int(item_id),
                "rank": rank,
                "score": float(user_scores[item_id]),
            })
    
    # 各K値でのメトリクス計算
    metrics_at_k = {}
    for k in ks:
        top_k_items = set(top_indices[:k])
        hits_at_k = len(top_k_items & test_items)
        precision = hits_at_k / k
        recall = hits_at_k / len(test_items) if test_items else 0
        metrics_at_k[k] = {
            "hits": hits_at_k,
            "precision": precision,
            "recall": recall,
        }
    
    return {
        "user_id": user_id,
        "n_test_items": len(test_items),
        "n_train_items": len(train_items),
        "test_items": sorted(list(test_items)),
        "hits": hits,
        "n_hits_in_top_k": len(hits),
        "metrics_at_k": metrics_at_k,
        "top_recommendations": [
            {
                "rank": rank,
                "item_id": int(item_id),
                "score": float(user_scores[item_id]),
                "is_hit": item_id in test_items,
            }
            for rank, item_id in enumerate(top_indices[:20], 1)
        ],
    }


def print_user_analysis(result: dict) -> None:
    """ユーザー分析結果を表示"""
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"\n{'='*60}")
    print(f"User {result['user_id']} の推薦分析")
    print(f"{'='*60}")
    
    print(f"\n【基本情報】")
    print(f"  訓練セットのアイテム数: {result['n_train_items']}")
    print(f"  テストセットのアイテム数: {result['n_test_items']}")
    print(f"  テストアイテムID: {result['test_items']}")
    
    print(f"\n【推薦トップ20】")
    print(f"{'順位':>4} | {'アイテムID':>10} | {'スコア':>10} | {'ヒット':>6}")
    print("-" * 45)
    for rec in result["top_recommendations"]:
        hit_mark = "✓" if rec["is_hit"] else ""
        print(f"{rec['rank']:>4} | {rec['item_id']:>10} | {rec['score']:>10.4f} | {hit_mark:>6}")
    
    print(f"\n【ヒットしたアイテム（トップ100内）】")
    if result["hits"]:
        for hit in result["hits"]:
            print(f"  アイテムID {hit['item_id']}: {hit['rank']}位 (スコア: {hit['score']:.4f})")
    else:
        print("  なし")
    
    print(f"\n【各K値でのメトリクス】")
    print(f"{'K':>6} | {'ヒット数':>8} | {'Precision':>10} | {'Recall':>10}")
    print("-" * 45)
    for k, metrics in result["metrics_at_k"].items():
        print(f"{k:>6} | {metrics['hits']:>8} | {metrics['precision']:>10.4f} | {metrics['recall']:>10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="推薦結果と実際のインタラクションを比較"
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
        "--user-id",
        type=int,
        default=None,
        help="分析するユーザーID（指定しない場合はランダム）",
    )
    parser.add_argument(
        "--n-users",
        type=int,
        default=5,
        help="ランダムに選択するユーザー数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="分析する推薦上位数",
    )
    
    args = parser.parse_args()
    
    print("データを読み込み中...")
    cf_scores = np.load(args.cf_scores_path)
    test_user_dict = load_test_data(args.data_dir)
    train_user_dict = load_train_data(args.data_dir)
    
    print(f"cf_scores shape: {cf_scores.shape}")
    print(f"テストユーザー数: {len(test_user_dict)}")
    print(f"訓練ユーザー数: {len(train_user_dict)}")
    
    # 分析するユーザーを決定
    if args.user_id is not None:
        user_ids = [args.user_id]
    else:
        rng = np.random.default_rng(args.seed)
        all_user_ids = list(test_user_dict.keys())
        user_ids = rng.choice(all_user_ids, size=min(args.n_users, len(all_user_ids)), replace=False)
    
    # 各ユーザーの分析
    all_results = []
    for user_id in user_ids:
        result = analyze_user_recommendations(
            user_id=user_id,
            cf_scores=cf_scores,
            test_user_dict=test_user_dict,
            train_user_dict=train_user_dict,
            top_k=args.top_k,
        )
        all_results.append(result)
        print_user_analysis(result)
    
    # 全体サマリー
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("全体サマリー")
        print(f"{'='*60}")
        
        for k in [20, 40, 60, 80, 100]:
            avg_recall = np.mean([r["metrics_at_k"][k]["recall"] for r in all_results if "error" not in r])
            avg_precision = np.mean([r["metrics_at_k"][k]["precision"] for r in all_results if "error" not in r])
            print(f"  @{k}: 平均Precision={avg_precision:.4f}, 平均Recall={avg_recall:.4f}")


if __name__ == "__main__":
    main()

