"""
特定のユーザーとアイテム間の説明パスを取得・ランキング付けして保存するスクリプト
"""

import json
import logging
import os
from pathlib import Path

import pandas as pd
import torch

# パッケージのパス設定
# srcディレクトリをパスに追加してインポートできるようにする
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
):
    """学習済みモデルをロード"""
    if model_path:
        # 指定されたパスを使用
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            # 相対パスで探索
            if (Path("packages") / model_path).exists():
                model_path_obj = Path("packages") / model_path
            elif (Path("../kgat") / model_path).exists():
                model_path_obj = Path("../kgat") / model_path
            else:
                raise FileNotFoundError(f"Specified model not found: {model_path}")
        model_path = str(model_path_obj)
    else:
        # モデルディレクトリのパスを調整
        # kgatパッケージ内のtrained_modelを参照
        model_dir = Path("../kgat/trained_model/KGAT") / config.data_name

        # モデルファイルを探す
        model_files = list(model_dir.rglob("*.pth"))

        if not model_files:
            # カレントディレクトリからの相対パスでも探してみる
            model_dir = Path("packages/kgat/trained_model/KGAT") / config.data_name
            model_files = list(model_dir.rglob("*.pth"))

        if not model_files:
            raise FileNotFoundError(f"No trained model found in {model_dir}")

        # 最初のモデルを使用（通常はエポックが進んだものを使いたいが、ここでは最初に見つかったものを使用）
        # 必要であればソートロジックを追加
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # 新しい順
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
        description="Explain prediction for a specific user-item pair"
    )
    parser.add_argument("--user_id", type=int, default=4646, help="Target User ID")
    parser.add_argument("--item_id", type=int, default=159, help="Target Item ID")
    parser.add_argument(
        "--max_hops", type=int, default=3, help="Maximum hops for path finding"
    )
    parser.add_argument(
        "--top_k", type=int, default=50, help="Number of top paths to save"
    )
    parser.add_argument("--data_name", type=str, default="yelp", help="Dataset name")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to specific model file (.pth)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()

    # ターゲット設定
    TARGET_UID = args.user_id
    TARGET_IID = args.item_id
    MAX_HOPS = args.max_hops
    TOP_K_PATHS = args.top_k  # 保存するパスの数

    logger.info(f"Target: User={TARGET_UID}, Item={TARGET_IID}, Hops={MAX_HOPS}")

    # 設定
    # データセットの絶対パスを解決
    # パッケージ構成を考慮して柔軟にパスを探す
    kgat_path = Path("packages/kgat").resolve()
    if not kgat_path.exists():
        # カレントディレクトリがpackages/kgat-explainerの場合など
        kgat_path = Path("../kgat").resolve()

    if not kgat_path.exists():
        # プロジェクトルートから実行されていない場合など
        # 仕方がないので相対パスで試みる
        logger.warning(
            "Could not find 'packages/kgat' directory. Using default relative paths."
        )
        data_dir_path = "datasets/"
    else:
        data_dir_path = str(kgat_path / "datasets") + "/"

    config = KGATConfig(
        data_name=args.data_name,
        data_dir=data_dir_path,
        use_pretrain=0,  # 推論時は事前学習データのロードは不要（モデル重みで上書きされるため）
        embed_dim=64,
        relation_dim=64,
        aggregation_type="bi-interaction",
        conv_dim_list="[64,32,16]",
        mess_dropout="[0.1,0.1,0.1]",
    )

    # パスの補正（念のため残すが、上記ロジックでカバーされているはず）
    if not os.path.exists(config.data_dir) and os.path.exists("../kgat/datasets/"):
        config.data_dir = "../kgat/datasets/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # データロード
    logger.info("Loading data...")
    try:
        data_loader = DataLoader(config, logger)
    except FileNotFoundError:
        # パスを再調整してトライ
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
    relation_names = load_relation_names(relation_list_path)

    # インタラクション用リレーション名を追加
    relation_names[0] = "Interact"
    relation_names[1] = "InteractedBy"

    # 逆リレーション名（Relation_X）を分かりやすくする
    # リレーション数（Interact含む）を計算
    n_relations = (
        len(relation_names) - 2
    ) + 2  # 既存の辞書にはInteract/InteractedByはないはずだが念のため
    # 実際には relation_list.txt の行数から分かるが、ここでは簡易的に
    # リレーションIDの最大値を探してオフセットを推定する
    max_rel_id = max([k for k in relation_names.keys() if isinstance(k, int)])

    # オフセットの推定 (relation_list.txt のエントリ数)
    # load_relation_names は relation_list.txt の内容のみを返す
    # relation_list.txt の最大IDは offset - 1
    # KGATでは 0, 1 がインタラクション用、2〜offset+1 が順方向、offset+2〜 が逆方向
    # relation_names のIDは 0, 1, ... (raw ID from file) なので、表示時に +2 されているか確認が必要
    # load_relation_names はファイルのID (0 start) をそのまま返す

    # KGAT内部のリレーションIDは:
    # 0: Interact
    # 1: InteractedBy
    # 2 ~ K+1: Original Relations (kg_final.txtのrelation ID 0 ~ K-1)
    # K+2 ~ 2K+1: Inverse Relations
    # ここで K = kg_final.txtの最大relation ID + 1

    # relation_names辞書のキーはファイルのID (0 ~ K-1)
    # これをKGAT内部IDにマッピングして新しい辞書を作る
    kgat_relation_names = {0: "Interact", 1: "InteractedBy"}

    # ファイルから読み込んだリレーション名を追加
    raw_relation_names = load_relation_names(relation_list_path)

    # 重要: DataLoaderはkg_final.txtの実際の最大relation IDから逆方向のオフセットを計算する
    # n_relations = (kg_final.txtの最大relation ID + 1) * 2 + 2
    # n_rawはkg_final.txtの最大relation ID + 1
    n_raw_relations_in_kg = (data_loader.n_relations - 2) // 2  # DataLoaderで計算された値から逆算

    # relation_list.txtには使われていない関係も含まれる可能性がある
    # kg_final.txtで実際に使われているのは 0 ~ (n_raw_relations_in_kg - 1) のみ
    for raw_id, name in raw_relation_names.items():
        # kg_final.txtで使われている関係のみをマッピング
        if raw_id < n_raw_relations_in_kg:
            # 順方向: ID + 2
            kgat_relation_names[raw_id + 2] = name
            # 逆方向: ID + 2 + n_raw_relations_in_kg
            kgat_relation_names[raw_id + 2 + n_raw_relations_in_kg] = f"{name} (Inverse)"

    # relation_names を上書き（explainループで使いやすいように）
    relation_names = kgat_relation_names

    # モデルロード
    logger.info("Loading trained model...")
    model = load_trained_model(config, data_loader, device, args.model_path)

    # KG辞書の準備（Attention計算用）
    logger.info("Preparing KG data structures...")
    kg_dict_by_relation = {}
    for relation, ht_list in data_loader.train_relation_dict.items():
        kg_dict_by_relation[relation] = ht_list

    # Explainer初期化
    logger.info("Initializing KGAT Explainer...")
    explainer = KGATExplainer(
        model=model,
        data_loader=data_loader,
        kg_dict=data_loader.train_kg_dict,
        kg_dict_by_relation=kg_dict_by_relation,
        device=device,
        precompute_attention=True,
    )

    # ユーザーIDの調整（エンティティ数だけオフセットを加算）
    adjusted_uid = TARGET_UID + data_loader.n_entities

    # パスの探索と説明
    logger.info(
        f"Finding paths for User {TARGET_UID} (internal: {adjusted_uid}) and Item {TARGET_IID}..."
    )

    explanation = explainer.explain_prediction(
        user_id=adjusted_uid,
        item_id=TARGET_IID,
        max_hops=MAX_HOPS,
        max_paths=TOP_K_PATHS,
    )

    logger.info(f"Found {explanation['num_paths']} paths.")

    # 結果の整形
    formatted_paths = []
    for i, path_info in enumerate(explanation["paths"]):
        formatted_path = []
        path_str_parts = []

        # 開始ノード（ユーザー）
        start_node_name = f"User:{TARGET_UID}"
        path_str_parts.append(start_node_name)

        for edge in path_info["path"]:
            h, r, t = edge["head"], edge["relation"], edge["tail"]

            # 名前解決
            # Headはパスの最初の要素以外はエンティティ
            h_name = (
                entity_names.get(h, f"Entity_{h}")
                if h < data_loader.n_entities
                else f"User_{h - data_loader.n_entities}"
            )
            if h == adjusted_uid:
                h_name = f"User_{TARGET_UID}"

            t_name = (
                entity_names.get(t, f"Entity_{t}")
                if t < data_loader.n_entities
                else f"User_{t - data_loader.n_entities}"
            )
            r_name = relation_names.get(r, f"Relation_{r}")

            edge_data = {
                "head_id": h,
                "head_name": h_name,
                "relation_id": r,
                "relation_name": r_name,
                "tail_id": t,
                "tail_name": t_name,
                "attention_score": float(edge["attention_score"]),
            }
            formatted_path.append(edge_data)
            # スコアをパース文字列に含める (小数点4桁)
            score_str = f"{float(edge['attention_score']):.4f}"
            path_str_parts.append(f"--[{r_name} ({score_str})]--> {t_name}")

        formatted_paths.append(
            {
                "rank": i + 1,
                "total_score": float(path_info["total_score"]),
                "avg_score": float(path_info["avg_score"]),
                "length": path_info["length"],
                "path_sequence": formatted_path,
                "path_string": " ".join(path_str_parts),
            }
        )

    # 結果の保存
    output_data = {
        "user_id": TARGET_UID,
        "item_id": TARGET_IID,
        "item_name": entity_names.get(TARGET_IID, f"Unknown_{TARGET_IID}"),
        "num_paths_found": explanation["num_paths"],
        "paths": formatted_paths,
    }

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"paths_u{TARGET_UID}_i{TARGET_IID}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_file}")

    # 上位パスの表示
    logger.info("\n=== Top 3 Paths ===")
    for i, path in enumerate(formatted_paths[:3]):
        logger.info(f"Rank {path['rank']} (Score: {path['total_score']:.4f}):")
        logger.info(f"  {path['path_string']}")


if __name__ == "__main__":
    main()
