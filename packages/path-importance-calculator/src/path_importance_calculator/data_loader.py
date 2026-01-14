"""データ読み込み用モジュール"""

from pathlib import Path
from typing import Generator

import pandas as pd


def load_user_item_pairs(file_path: str) -> Generator[tuple[int, int], None, None]:
    """
    ユーザー・アイテムペアを読み込む

    Args:
        file_path: インタラクションファイルのパス

    Yields:
        tuple[int, int]: (ユーザーID, アイテムID)のペア
    """
    with open(file_path, "r", encoding="utf-8") as f:
        # ヘッダー行をスキップ
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == 2:
                user_id = int(parts[0])
                item_id = int(parts[1])
                yield user_id, item_id


def load_kg_as_dict(
    kg_file: str,
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    """
    KGデータを辞書形式で読み込む

    Args:
        kg_file: KGファイルのパス (head relation tail形式)

    Returns:
        tuple: (forward_edges, backward_edges)
            - forward_edges[head] = [(relation, tail), ...]
            - backward_edges[tail] = [(relation, head), ...]
    """
    forward_edges: dict[int, list[tuple[int, int]]] = {}
    backward_edges: dict[int, list[tuple[int, int]]] = {}

    with open(kg_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue

            head, relation, tail = int(parts[0]), int(parts[1]), int(parts[2])

            # Forward edge: head -> tail
            if head not in forward_edges:
                forward_edges[head] = []
            forward_edges[head].append((relation, tail))

            # Backward edge: tail -> head
            if tail not in backward_edges:
                backward_edges[tail] = []
            backward_edges[tail].append((relation, head))

    return forward_edges, backward_edges
