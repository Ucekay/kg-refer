"""
指定ノードに接続する A_in の全エッジ（入出力）を関係名・エンティティ/ユーザー名付きでCSV出力するスクリプト。
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

from kgat.config import KGATConfig
from kgat.core.kgat import KGAT
from kgat.data.dataloader import DataLoader
from kgat.utils.model_helper import load_model


def load_id_map(file_path: Path) -> Dict[int, str]:
    """remap_id -> org_id の辞書を作成（行末に余分な列があっても2列目をIDとみなす）。"""
    id_map: Dict[int, str] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                rid = int(parts[1])
            except ValueError:
                continue
            id_map[rid] = parts[0]
    return id_map


def load_relation_names(file_path: Path) -> Dict[int, str]:
    """
    relation_list.txt を読み込み、最終的な relation id へのマッピングを構築。
    KGATの前処理と同じく、user/item エッジを 0,1 とし、
    KG relation を +2、逆向きを +2 + n_relations として付与する。
    """
    rel_map: Dict[int, str] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                rid = int(parts[-1])
            except ValueError:
                continue
            name = " ".join(parts[:-1])
            rel_map[rid] = name

    n_rel = (max(rel_map) + 1) if rel_map else 0
    full_rel = {0: "user->item", 1: "item->user"}
    for rid, name in rel_map.items():
        full_rel[2 + rid] = name
        full_rel[2 + n_rel + rid] = f"{name}_inv"
    return full_rel


def build_edge_relation_lookup(train_relation_dict):
    """(h,t) -> relation id のリスト"""
    edge_rel = defaultdict(list)
    for r, ht_list in train_relation_dict.items():
        for h, t in ht_list:
            edge_rel[(int(h), int(t))].append(int(r))
    return edge_rel


def main():
    parser = argparse.ArgumentParser(
        description="A_in で指定ノードに接続する全エッジをCSV出力（入出力両方向）"
    )
    parser.add_argument(
        "--node-id",
        type=int,
        required=True,
        help="A_in 上のノードID（0〜n_entities+n_users-1）",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="学習済みモデル .pth のパス",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("packages/kgat/datasets"),
        help="データセットディレクトリ（親）",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default="yelp2018",
        help="データセット名（サブディレクトリ名）",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="出力CSVパス（未指定なら packages/kgat/output/node_edges_node{ID}.csv）",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("dump_node_edges")

    dataset_dir = args.data_dir / args.data_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"データセットが見つかりません: {dataset_dir}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {args.model_path}")

    # KGATConfigをモデル設定に合わせて初期化
    config = KGATConfig()
    config.data_dir = str(args.data_dir)
    config.data_name = args.data_name
    config.use_pretrain = 0
    config.embed_dim = 64
    config.relation_dim = 64
    config.aggregation_type = "bi-interaction"
    config.laplacian_type = "random-walk"
    config.conv_dim_list = "[64,32,16]"
    config.predict = True

    data = DataLoader(config, logger)
    model = KGAT(
        config,
        data.n_users,
        data.n_entities,
        data.n_relations,
        data.A_in,
    )
    model = load_model(model, str(args.model_path))

    entity_map = load_id_map(dataset_dir / "entity_list.txt")
    user_map = load_id_map(dataset_dir / "user_list.txt")
    rel_name = load_relation_names(dataset_dir / "relation_list.txt")
    edge_rel = build_edge_relation_lookup(data.train_relation_dict)

    def node_label(nid: int) -> str:
        if nid >= data.n_entities:
            uid = nid - data.n_entities
            return f"user[{uid}] ({user_map.get(uid, '?')})"
        return f"entity[{nid}] ({entity_map.get(nid, '?')})"

    A = model.A_in.detach().cpu().coalesce()
    idx = A.indices()
    vals = A.values()

    outgoing: List[tuple[int, float, List[int]]] = []
    incoming: List[tuple[int, float, List[int]]] = []

    for r, c, v in zip(idx[0], idx[1], vals):
        r_i = int(r)
        c_i = int(c)
        w = float(v)
        rels = edge_rel.get((r_i, c_i), [])
        if r_i == args.node_id:
            outgoing.append((c_i, w, rels))
        if c_i == args.node_id:
            incoming.append((r_i, w, rels))

    outgoing.sort(key=lambda x: -x[1])
    incoming.sort(key=lambda x: -x[1])

    out_path = (
        args.output_csv
        if args.output_csv is not None
        else Path(f"packages/kgat/output/node_edges_node{args.node_id}.csv")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["direction", "src_id", "src_label", "dst_id", "dst_label", "weight", "relations"]
        )
        for dst, w, rels in outgoing:
            writer.writerow(
                [
                    "out",
                    args.node_id,
                    node_label(args.node_id),
                    dst,
                    node_label(dst),
                    f"{w:.8f}",
                    ";".join(rel_name.get(r, f"rel_{r}") for r in rels) if rels else "unknown",
                ]
            )
        for src, w, rels in incoming:
            writer.writerow(
                [
                    "in",
                    src,
                    node_label(src),
                    args.node_id,
                    node_label(args.node_id),
                    f"{w:.8f}",
                    ";".join(rel_name.get(r, f"rel_{r}") for r in rels) if rels else "unknown",
                ]
            )

    logger.info(
        f"保存しました: {out_path} (outgoing {len(outgoing)} 本, incoming {len(incoming)} 本)"
    )


if __name__ == "__main__":
    main()


