"""
ランダムなアイテムノードに対して、それぞれの関係-属性に対する
A_inからのアテンションスコアをすべて取得するプログラム

注意: A_inは関係を統合したマトリックスなので、同じ(head, tail)ペアに対して
複数の関係があっても1つの値しか持たない（relation-agnostic）
"""

import csv
import json
import logging
import os
import random
from pathlib import Path

import torch

# パッケージのパス設定
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / "src"
kgat_src_dir = current_dir.parent / "kgat" / "src"

import sys

sys.path.append(str(src_dir))
sys.path.append(str(kgat_src_dir))

from kgat.config import KGATConfig
from kgat.core.kgat import KGAT
from kgat.data.dataloader import DataLoader

from kgat_explainer import KGATExplainer, KGATModelLoader


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def load_entity_names(entity_list_path: str) -> dict[int, str]:
    """entity_list.txtからエンティティ名を取得"""
    entity_names = {}
    try:
        with open(entity_list_path, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    entity_id = int(parts[-1])
                    entity_name = " ".join(parts[:-1]).strip('"')
                    entity_names[entity_id] = entity_name
    except FileNotFoundError:
        print(f"Warning: {entity_list_path} not found.")
    return entity_names


def load_relation_names(relation_list_path: str) -> dict[int, str]:
    """relation_list.txtからリレーション名を取得"""
    relation_names = {}
    try:
        with open(relation_list_path, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    rel_id = int(parts[-1])
                    rel_name = " ".join(parts[:-1]).strip('"')
                    relation_names[rel_id] = rel_name
    except FileNotFoundError:
        print(f"Warning: {relation_list_path} not found.")
    return relation_names


def load_trained_model(
    config: KGATConfig,
    data_loader: DataLoader,
    device: torch.device,
    model_path: str = None,
    epoch: int = None,
):
    """学習済みモデルをロード"""
    if model_path:
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            if (Path("packages") / model_path).exists():
                model_path_obj = Path("packages") / model_path
            elif (Path("../kgat") / model_path).exists():
                model_path_obj = Path("../kgat") / model_path
            else:
                raise FileNotFoundError(f"Specified model not found: {model_path}")
        model_path = str(model_path_obj)
    elif epoch is not None:
        model_dir = Path("../kgat/trained_model/KGAT") / config.data_name
        if not model_dir.exists():
            model_dir = Path("packages/kgat/trained_model/KGAT") / config.data_name

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        model_filename = f"model_epoch{epoch}.pth"
        model_path_obj = model_dir / model_filename

        if not model_path_obj.exists():
            model_files = list(model_dir.rglob(model_filename))
            if model_files:
                model_path_obj = model_files[0]
            else:
                raise FileNotFoundError(
                    f"Model file not found: {model_filename} in {model_dir}"
                )

        model_path = str(model_path_obj)
    else:
        model_dir = Path("../kgat/trained_model/KGAT") / config.data_name
        model_files = list(model_dir.rglob("*.pth"))

        if not model_files:
            model_dir = Path("packages/kgat/trained_model/KGAT") / config.data_name
            model_files = list(model_dir.rglob("*.pth"))

        if not model_files:
            raise FileNotFoundError(f"No trained model found in {model_dir}")

        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        model_path = str(model_files[0])

    print(f"Loading model from: {model_path}")

    user_pre_embed = None
    item_pre_embed = None
    if config.use_pretrain == 1 and hasattr(data_loader, "user_pre_embed"):
        user_pre_embed = torch.FloatTensor(data_loader.user_pre_embed)
        item_pre_embed = torch.FloatTensor(data_loader.item_pre_embed)

    loader = KGATModelLoader(model_path, device)

    model = loader.load_model(
        model_class=KGAT,
        config=config,
        n_users=data_loader.n_users,
        n_entities=data_loader.n_entities,
        n_relations=data_loader.n_relations,
        A_in=data_loader.A_in.to(device),
        user_pre_embed=user_pre_embed,
        item_pre_embed=item_pre_embed,
    )

    return model


import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Get attention scores for random item nodes from A_in matrix"
    )
    parser.add_argument(
        "--num_items",
        type=int,
        default=10,
        help="Number of random items to analyze",
    )
    parser.add_argument(
        "--item_id",
        type=int,
        default=None,
        help="Specific item ID to analyze (overrides random selection)",
    )
    parser.add_argument("--data_name", type=str, default="yelp", help="Dataset name")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to specific model file (.pth)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch number to load model (e.g., 700 for model_epoch700.pth)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for item selection",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize attention scores using softmax per item (all edges from the same head)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()

    logger.info(f"Analyzing attention scores for items (num_items={args.num_items})")

    # 設定
    kgat_path = Path("packages/kgat").resolve()
    if not kgat_path.exists():
        kgat_path = Path("../kgat").resolve()

    if not kgat_path.exists():
        logger.warning(
            "Could not find 'packages/kgat' directory. Using default relative paths."
        )
        data_dir_path = "datasets/"
    else:
        data_dir_path = str(kgat_path / "datasets") + "/"

    config = KGATConfig(
        data_name=args.data_name,
        data_dir=data_dir_path,
        use_pretrain=0,
        embed_dim=64,
        relation_dim=64,
        aggregation_type="bi-interaction",
        conv_dim_list="[64,32,16]",
        mess_dropout="[0.1,0.1,0.1]",
    )

    if not os.path.exists(config.data_dir) and os.path.exists("../kgat/datasets/"):
        config.data_dir = "../kgat/datasets/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # データロード
    logger.info("Loading data...")
    try:
        data_loader = DataLoader(config, logger)
    except FileNotFoundError:
        config.data_dir = "packages/kgat/datasets/"
        data_loader = DataLoader(config, logger)

    # エンティティ名とリレーション名のロード
    entity_list_path = os.path.join(
        config.data_dir, config.data_name, "entity_list.txt"
    )
    relation_list_path = os.path.join(
        config.data_dir, config.data_name, "relation_list.txt"
    )

    entity_names = load_entity_names(entity_list_path)
    raw_relation_names = load_relation_names(relation_list_path)

    # KGAT内部のリレーションIDマッピング
    kgat_relation_names = {0: "Interact", 1: "InteractedBy"}
    n_raw_relations_in_kg = (data_loader.n_relations - 2) // 2

    for raw_id, name in raw_relation_names.items():
        if raw_id < n_raw_relations_in_kg:
            kgat_relation_names[raw_id + 2] = name
            kgat_relation_names[raw_id + 2 + n_raw_relations_in_kg] = f"{name} (Inverse)"

    relation_names = kgat_relation_names

    # モデルロード
    logger.info("Loading trained model...")
    model = load_trained_model(config, data_loader, device, args.model_path, args.epoch)

    # KG辞書の準備
    logger.info("Preparing KG data structures...")
    kg_dict_by_relation = {}
    for relation, ht_list in data_loader.train_relation_dict.items():
        kg_dict_by_relation[relation] = ht_list

    # Explainer初期化（A_inを使用）
    logger.info("Initializing KGAT Explainer with model's A_in matrix...")
    explainer = KGATExplainer(
        model=model,
        data_loader=data_loader,
        kg_dict=data_loader.train_kg_dict,
        kg_dict_by_relation=kg_dict_by_relation,
        device=device,
        precompute_attention=False,
        use_model_A_in=True,  # A_inからアテンションスコアを取得
    )

    # アイテム選択
    if args.item_id is not None:
        item_ids = [args.item_id]
        logger.info(f"Analyzing specific item: {args.item_id}")
    else:
        # ランダムなアイテムを選択（エッジを持つアイテムのみ）
        random.seed(args.seed)
        items_with_edges = [
            item_id
            for item_id in range(data_loader.n_items)
            if item_id in data_loader.train_kg_dict
        ]
        if len(items_with_edges) < args.num_items:
            logger.warning(
                f"Only {len(items_with_edges)} items have edges, using all of them"
            )
            item_ids = items_with_edges
        else:
            item_ids = random.sample(items_with_edges, args.num_items)
        logger.info(f"Selected {len(item_ids)} random items: {item_ids}")

    # 各アイテムについてアテンションスコアを取得
    all_results = []

    for item_id in item_ids:
        logger.info(f"\n=== Analyzing Item {item_id} ===")

        # アイテムノードから出るすべてのエッジを取得
        if item_id not in data_loader.train_kg_dict:
            logger.warning(f"Item {item_id} has no outgoing edges")
            continue

        edges = data_loader.train_kg_dict[item_id]  # [(tail, relation), ...]

        # A_inからアテンションスコアを取得（関係非依存）
        # 注意: A_inは関係を統合したマトリックスなので、同じ(head, tail)ペアに対して
        # 複数の関係があっても1つの値しか持たない
        # ただし、head-tailペアが一組一つ（同じ(head, tail)ペアに対して複数の関係が存在しない）なら問題ない
        a_in_scores = {}  # (tail_id, relation_id) -> A_in score
        for tail_id, relation_id in edges:
            # A_inから正規化済みアテンションスコアを取得（関係情報は使用されない）
            # 同じ(head, tail)ペアに対して複数の関係が存在しない場合、各関係ごとに異なる(head, tail)ペアが存在するため、
            # A_inから取得したスコアも関係ごとに異なる値になる
            attention_score = explainer.attention_calculator.get_normalized_attention(
                item_id, tail_id
            )
            a_in_scores[(tail_id, relation_id)] = attention_score

        # 正規化は既にA_inで行われているため、normalizeフラグは無視される
        normalized_scores = a_in_scores

        # すべてのエッジを1つのリストにまとめる（関係に関係なく）
        all_edges_list = []
        for tail_id, relation_id in edges:
            attention_score = normalized_scores.get((tail_id, relation_id), 0.0)
            
            relation_name = relation_names.get(relation_id, f"Relation_{relation_id}")
            tail_name = (
                entity_names.get(tail_id, f"Entity_{tail_id}")
                if tail_id < data_loader.n_entities
                else f"User_{tail_id - data_loader.n_entities}"
            )
            
            all_edges_list.append({
                "tail_id": tail_id,
                "tail_name": tail_name,
                "relation_id": relation_id,
                "relation_name": relation_name,
                "attention_score": float(attention_score),
            })

        # アテンションスコアの高い順にソート（降順）
        all_edges_list.sort(key=lambda x: x["attention_score"], reverse=True)

        # 結果を保存
        item_results = {
            "item_id": item_id,
            "item_name": entity_names.get(item_id, f"Unknown_{item_id}"),
            "num_edges": len(edges),
            "normalized": True,  # A_inは既に正規化済み
            "note": "A_in scores are relation-agnostic. Sorted by attention score (descending).",
            "edges": all_edges_list,
        }

        all_results.append(item_results)

        logger.info(
            f"Item {item_id}: {item_results['num_edges']} edges (sorted by attention score)"
        )

    # 結果の保存（JSON）
    output_data = {
        "attention_mode": "A_in (normalized, relation-agnostic)",
        "note": "Attention scores are retrieved from model's A_in matrix. "
                "A_in integrates all relations, so the same (head, tail) pair has the same score "
                "regardless of relation type. Scores are already normalized via softmax. "
                "Edges are sorted by attention score (descending).",
        "normalized": True,  # A_inは既に正規化済み
        "num_items_analyzed": len(all_results),
        "items": all_results,
    }

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    if args.item_id is not None:
        json_file = output_dir / f"item_{args.item_id}_attention_scores.json"
        csv_file = output_dir / f"item_{args.item_id}_attention_scores.csv"
    else:
        json_file = output_dir / f"random_items_attention_scores_{args.num_items}.json"
        csv_file = output_dir / f"random_items_attention_scores_{args.num_items}.csv"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\nJSON results saved to {json_file}")

    # CSVファイルに保存（関係に関係なく、アテンションスコアの高い順）
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # ヘッダー
        writer.writerow([
            "item_id", "item_name", "tail_id", "tail_name", 
            "relation_id", "relation_name", "attention_score"
        ])
        
        # 各アイテムのエッジを書き込み
        for item_result in all_results:
            for edge in item_result["edges"]:
                writer.writerow([
                    item_result["item_id"],
                    item_result["item_name"],
                    edge["tail_id"],
                    edge["tail_name"],
                    edge["relation_id"],
                    edge["relation_name"],
                    edge["attention_score"],
                ])

    logger.info(f"CSV results saved to {csv_file}")

    # サマリー表示（アテンションスコアの高い順）
    logger.info("\n=== Summary (sorted by attention score, descending) ===")
    for item_result in all_results:
        logger.info(f"\nItem {item_result['item_id']} ({item_result['item_name']}):")
        logger.info(f"  Total edges: {item_result['num_edges']}")
        logger.info("  Top 10 edges by attention score:")
        for i, edge in enumerate(item_result["edges"][:10], 1):
            logger.info(
                f"    {i:2d}. {edge['relation_name']} (ID: {edge['relation_id']}) -> "
                f"{edge['tail_name']} (ID: {edge['tail_id']}): "
                f"attention_score={edge['attention_score']:.6f}"
            )


if __name__ == "__main__":
    main()

