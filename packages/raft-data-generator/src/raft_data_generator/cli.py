"""RAFTデータ生成のCLI"""

import logging
from pathlib import Path

import pandas as pd
from cyclopts import App

from raft_data_generator.generator import RaftDataGenerator

app = App()

# ロガーの設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@app.default
def main(
    interactions: str,
    user_profiles: str,
    item_profiles: str,
    attribute_scores: str,
    path_scores: str,
    similarity: str,
    output_dir: str = "packages/raft-data-generator/output",
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    min_attribute_score: float = 0.0,
    top_paths: int = 2,
    top_similar: int = 2,
    dataset_name: str = "yelp",
) -> None:
    """RAFT形式のfine-tuningデータを生成（プロンプトのみ、chosenは空文字列）

    Args:
        interactions: 評価用のユーザー・アイテムペアファイル（CSV形式、uid, iidカラム）
        user_profiles: ユーザープロファイルJSONのパス
        item_profiles: アイテムプロファイルJSONのパス
        attribute_scores: 属性スコアCSVのパス
        path_scores: パス重要度スコアJSONのパス
        similarity: セマンティック類似度結果JSONのパス
        output_dir: 出力ディレクトリのパス
        train_ratio: 訓練データの割合（デフォルト: 0.8）
        eval_ratio: 評価データの割合（デフォルト: 0.1、残りがテストデータ）
        min_attribute_score: 属性スコアの最小閾値（デフォルト: 0.0）
        top_paths: 含めるトップパスの数（デフォルト: 2）
        top_similar: 含める類似ユーザー/アイテムの数（デフォルト: 2）
        dataset_name: データセット名（出力ファイル名に使用、デフォルト: "yelp"）
    """
    logger.info("RAFTデータ生成を開始します")
    logger.info(f"インタラクションファイル: {interactions}")
    logger.info(f"出力ディレクトリ: {output_dir}")
    logger.info(f"データセット名: {dataset_name}")

    # インタラクションデータをロード
    logger.info(f"インタラクションデータをロード中: {interactions}")
    interactions_df = pd.read_csv(interactions)
    logger.info(f"インタラクション数: {len(interactions_df)}")

    # データを分割
    logger.info(
        f"データを分割中（train: {train_ratio}, eval: {eval_ratio}, test: {1 - train_ratio - eval_ratio}）"
    )
    n_total = len(interactions_df)
    n_train = int(n_total * train_ratio)
    n_eval = int(n_total * eval_ratio)

    train_df = interactions_df.iloc[:n_train]
    eval_df = interactions_df.iloc[n_train : n_train + n_eval]
    test_df = interactions_df.iloc[n_train + n_eval :]

    logger.info(f"訓練データ: {len(train_df)}件")
    logger.info(f"評価データ: {len(eval_df)}件")
    logger.info(f"テストデータ: {len(test_df)}件")

    # RAFTデータジェネレーターを初期化（説明生成は不要、chosenは空文字列）
    generator = RaftDataGenerator(
        user_profiles=user_profiles,
        item_profiles=item_profiles,
        attribute_scores=attribute_scores,
        path_scores=path_scores,
        similarity=similarity,
        explanation_provider=None,  # 説明生成は不要（chosenは空文字列）
    )

    # 各データセットを生成
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in [
        ("train", train_df),
        ("eval", eval_df),
        ("test", test_df),
    ]:
        if len(split_df) == 0:
            logger.warning(f"{split_name}データが空です。スキップします。")
            continue

        logger.info(f"{split_name}データを生成中...")
        data = generator.generate_batch(
            split_df,
            min_attribute_score=min_attribute_score,
            top_paths=top_paths,
            top_similar=top_similar,
            show_progress=True,
        )

        output_file = output_dir_path / f"{split_name}.json"
        generator.save_to_jsonl(data, output_file)

    logger.info("RAFTデータ生成が完了しました！")


@app.command
def replace_prompts(
    input_dir: str,
    user_profiles: str,
    item_profiles: str,
    attribute_scores: str,
    path_scores: str,
    similarity: str,
    output_dir: str = "packages/raft-data-generator/output",
    min_attribute_score: float = 0.0,
    top_paths: int = 2,
    top_similar: int = 2,
) -> None:
    """既存のRAFTデータファイルからプロンプトだけを置き換える

    Args:
        input_dir: 既存のRAFTデータディレクトリ（train.json, eval.json, test.jsonを含む）
        user_profiles: ユーザープロファイルJSONのパス
        item_profiles: アイテムプロファイルJSONのパス
        attribute_scores: 属性スコアCSVのパス
        path_scores: パス重要度スコアJSONのパス
        similarity: セマンティック類似度結果JSONのパス
        output_dir: 出力ディレクトリのパス
        min_attribute_score: 属性スコアの最小閾値（デフォルト: 0.0）
        top_paths: 含めるトップパスの数（デフォルト: 2）
        top_similar: 含める類似ユーザー/アイテムの数（デフォルト: 2）
    """
    logger.info("プロンプト置き換えを開始します")
    logger.info(f"入力ディレクトリ: {input_dir}")
    logger.info(f"出力ディレクトリ: {output_dir}")

    # RAFTデータジェネレーターを初期化（プロンプト生成のみなので説明生成は不要）
    generator = RaftDataGenerator(
        user_profiles=user_profiles,
        item_profiles=item_profiles,
        attribute_scores=attribute_scores,
        path_scores=path_scores,
        similarity=similarity,
        explanation_provider=None,  # 既存のchosenを使うので説明生成は不要
    )

    # 出力ディレクトリを作成
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    input_dir_path = Path(input_dir)

    # 各データセット（train, eval, test）を処理
    for split_name in ["train", "eval", "test"]:
        input_file = input_dir_path / f"{split_name}.json"

        if not input_file.exists():
            logger.warning(f"{split_name}.jsonが見つかりません。スキップします。")
            continue

        logger.info(f"{split_name}データのプロンプトを置き換え中...")
        data = generator.replace_prompts_from_existing_data(
            input_file,
            min_attribute_score=min_attribute_score,
            top_paths=top_paths,
            top_similar=top_similar,
            show_progress=True,
        )

        output_file = output_dir_path / f"{split_name}.json"
        generator.save_to_jsonl(data, output_file)

    logger.info("プロンプト置き換えが完了しました！")


@app.command
def replace_prompts_with_computation(
    input_dir: str,
    train_interactions: str,
    user_profiles: str,
    item_profiles: str,
    model_path: str,
    output_dir: str = "packages/raft-data-generator/output",
    data_dir: str = "datasets/",
    data_name: str = "yelp",
    kg_file: str | None = None,
    entity_list_file: str | None = None,
    relation_list_file: str | None = None,
    item_list: str | None = None,
    min_attribute_score: float = 0.0,
    top_paths: int = 2,
    top_similar: int = 2,
    text_encoder: str = "sentence-transformers/multi-qa-distilbert-cos-v1",
    similarity_topk: int = 2,
    similarity_pruning_score: float = 0.0,
    max_hops: int = 3,
    min_hops: int = 3,
    ignore_relations: str = "[]",
    exclude_relations: str = "[]",
    path_selection_mode: str = "topk",
) -> None:
    """既存のRAFTデータから類似ノード・パス・属性スコアを計算してプロンプトを置き換える

    Args:
        input_dir: 既存のRAFTデータディレクトリ（train.json, eval.json, test.jsonを含む）
        train_interactions: 訓練用インタラクションファイル（類似ノード計算のグラフ構築に使用）
        user_profiles: ユーザープロファイルJSONのパス
        item_profiles: アイテムプロファイルJSONのパス
        model_path: 学習済みKGATモデルのパス
        output_dir: 出力ディレクトリのパス
        data_dir: データディレクトリ（デフォルト: datasets/）
        data_name: データセット名（デフォルト: yelp）
        kg_file: kg_final.txtのパス（Noneの場合はデフォルトパスを使用）
        entity_list_file: entity_list.txtのパス（オプション）
        relation_list_file: relation_list.txtのパス（オプション）
        item_list: item_list.txtのパス（オプション、店名取得用）
        min_attribute_score: 属性スコアの最小閾値（デフォルト: 0.0）
        top_paths: 含めるトップパスの数（デフォルト: 2）
        top_similar: 含める類似ユーザー/アイテムの数（デフォルト: 2）
        text_encoder: テキストエンコーダーモデル名（デフォルト: sentence-transformers/multi-qa-distilbert-cos-v1）
        similarity_topk: 類似ノードの最大数（デフォルト: 2）
        similarity_pruning_score: 類似度スコアの閾値（デフォルト: 0.0）
        max_hops: 最大ホップ数（デフォルト: 3）
        min_hops: 最小ホップ数（デフォルト: 3）
        ignore_relations: パス探索で無視するリレーションIDのリスト（例: "[5]"）
        exclude_relations: 属性スコア計算で除外するリレーションIDのリスト（例: "[5]"）
        path_selection_mode: パス選択モード（'topk' または 'diverse'、デフォルト: 'topk'）
    """
    import json

    from kgat.config import KGATConfig
    from path_importance_calculator.calculator import PathImportanceCalculator
    from path_importance_calculator.model_loader import KGATModelLoader
    from semantic_similarity_retriever.retriever import (
        SemanticSimilarityRetriever,
        load_profiles,
    )
    from user_item_attribute_scorer.score_calculator import calculate_scores

    logger.info("プロンプト置き換え（計算付き）を開始します")
    logger.info(f"入力ディレクトリ: {input_dir}")
    logger.info(f"出力ディレクトリ: {output_dir}")

    # 出力ディレクトリを作成
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    input_dir_path = Path(input_dir)
    temp_dir = output_dir_path / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # プロファイルと訓練データをロード（共通で使用）
    logger.info("プロファイルと訓練データをロード中...")
    user_profiles_dict, item_profiles_dict = load_profiles(user_profiles, item_profiles)
    train_df = pd.read_csv(train_interactions)

    # カラム名をuid, iidに統一（user/item, uid/iid, その他の形式に対応）
    if "uid" in train_df.columns and "iid" in train_df.columns:
        # 既にuid, iidの場合はそのまま
        pass
    elif "user" in train_df.columns and "item" in train_df.columns:
        # user, itemの場合はリネーム
        train_df = train_df.rename(columns={"user": "uid", "item": "iid"})
    elif len(train_df.columns) >= 2:
        # ヘッダーなしまたは他の形式の場合、最初の2列をuid, iidとして使用
        train_df.columns = ["uid", "iid"] + list(train_df.columns[2:])
        train_df = train_df[["uid", "iid"]]
    else:
        raise ValueError(
            f"CSVファイルには少なくとも2列（ユーザーIDとアイテムID）が必要です。現在のカラム: {train_df.columns.tolist()}"
        )

    # 類似ノード検索器を初期化（訓練データでグラフ構築）
    retriever = SemanticSimilarityRetriever(
        user_profiles=user_profiles_dict,
        item_profiles=item_profiles_dict,
        interactions_df=train_df,
        text_encoder=text_encoder,
        pruning_score=similarity_pruning_score,
    )

    # KGAT設定とモデルローダーを初期化（共通で使用）
    kgat_config = KGATConfig(
        data_dir=data_dir,
        data_name=data_name,
        file_prefix="total_",  # total_train.txt, total_test.txt, total_val.txt を使用
        embed_dim=64,
        relation_dim=64,
        laplacian_type="random-walk",
        aggregation_type="bi-interaction",
        conv_dim_list="[64,32,16]",
        use_pretrain=0,
    )

    model_loader = KGATModelLoader(str(model_path), kgat_config, logger)

    if kg_file is None:
        kg_file = str(Path(data_dir) / data_name / "kg_final.txt")

    # 各データセット（train/eval/test）を処理
    for split_name in ["train", "eval", "test"]:
        input_file = input_dir_path / f"{split_name}.json"

        if not input_file.exists():
            logger.warning(f"{split_name}.jsonが見つかりません。スキップします。")
            continue

        logger.info("=" * 60)
        logger.info(f"{split_name.upper()}データの処理を開始します")

        # 既存データを読み込む
        logger.info(f"{split_name}データを読み込み中...")
        existing_data = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_data.append(json.loads(line))

        logger.info(f"読み込んだデータ数: {len(existing_data)}")

        # uid, iidのリストを作成
        uids = [item["uid"] for item in existing_data]
        iids = [item["iid"] for item in existing_data]
        split_df = pd.DataFrame({"uid": uids, "iid": iids})

        # 1. 類似ノードを計算（既に存在する場合はスキップ）
        similarity_file = temp_dir / f"semantic_similarity_results_{split_name}.json"
        if similarity_file.exists():
            logger.info(
                f"ステップ1: {split_name}の類似ノードは既に計算済みです。スキップします: {similarity_file}"
            )
        else:
            logger.info(f"ステップ1: {split_name}の類似ノードを計算中...")
            similarity_results = retriever.retrieve_batch(
                split_df, topk=similarity_topk
            )
            # JSONL形式で保存（1行1JSONオブジェクト、メモリ効率が良い）
            with open(similarity_file, "w", encoding="utf-8") as f:
                for result in similarity_results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            logger.info(f"類似ノードを保存しました: {similarity_file}")

        # 2. パス重要度を計算（既に存在する場合はスキップ）
        path_suffix = "_diverse" if path_selection_mode == "diverse" else ""
        path_file = (
            temp_dir / f"path_importance_scores_with_names{path_suffix}_{split_name}.json"
        )
        if path_file.exists():
            logger.info(
                f"ステップ2: {split_name}のパス重要度は既に計算済みです。スキップします: {path_file}"
            )
        else:
            logger.info(f"ステップ2: {split_name}のパス重要度を計算中... (selection_mode={path_selection_mode})")

            calculator = PathImportanceCalculator(
                model_loader,
                max_hops=max_hops,
                min_hops=min_hops,
                ignore_relations=eval(ignore_relations) if ignore_relations else [],
                entity_list_file=entity_list_file,
                relation_list_file=relation_list_file,
            )

            # 一時ファイルに保存してから計算
            temp_interactions_file = temp_dir / f"interactions_{split_name}.csv"
            split_df.to_csv(temp_interactions_file, index=False)

            calculator_results = calculator.calculate_for_pairs(
                str(temp_interactions_file),
                topk=top_paths,
                selection_mode=path_selection_mode,
            )
            calculator.save_results(calculator_results, str(path_file), format="jsonl")
            logger.info(f"パス重要度を保存しました: {path_file}")

        # 3. ユーザー・属性親和度を計算（既に存在する場合はスキップ）
        attribute_file = temp_dir / f"user_item_attribute_scores_{split_name}.csv"
        if attribute_file.exists():
            logger.info(
                f"ステップ3: {split_name}のユーザー・属性親和度は既に計算済みです。スキップします: {attribute_file}"
            )
        else:
            logger.info(f"ステップ3: {split_name}のユーザー・属性親和度を計算中...")

            # 一時ファイルが存在しない場合は作成
            temp_interactions_file = temp_dir / f"interactions_{split_name}.csv"
            if not temp_interactions_file.exists():
                split_df.to_csv(temp_interactions_file, index=False)

            calculate_scores(
                interactions_file=str(temp_interactions_file),
                model_path=str(model_path),
                kg_file=kg_file,
                data_dir=data_dir,
                data_name=data_name,
                output_file=str(attribute_file),
                exclude_relations=eval(exclude_relations)
                if exclude_relations
                else None,
            )
            logger.info(f"ユーザー・属性親和度を保存しました: {attribute_file}")

        # 4. プロンプトを生成して置き換え（レジューム可能）
        logger.info(f"ステップ4: {split_name}のプロンプトを生成中...")

        output_file = output_dir_path / f"{split_name}.json"
        temp_output_file = temp_dir / f"{split_name}_prompts_temp.json"

        # 既存の出力ファイルまたは一時ファイルから処理済みデータを読み込む
        processed_uids_iids = set()
        results = []

        if output_file.exists():
            logger.info(f"既存の出力ファイルを検出: {output_file}")
            logger.info("既存のデータを読み込み中...")
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        existing_result = json.loads(line)
                        results.append(existing_result)
                        processed_uids_iids.add(
                            (existing_result["uid"], existing_result["iid"])
                        )
            logger.info(f"既存のデータ数: {len(results)}")
        elif temp_output_file.exists():
            logger.info(f"一時ファイルを検出: {temp_output_file}")
            logger.info("一時ファイルからデータを読み込み中...")
            with open(temp_output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        existing_result = json.loads(line)
                        results.append(existing_result)
                        processed_uids_iids.add(
                            (existing_result["uid"], existing_result["iid"])
                        )
            logger.info(f"一時ファイルのデータ数: {len(results)}")

        generator = RaftDataGenerator(
            user_profiles=user_profiles,
            item_profiles=item_profiles,
            attribute_scores=str(attribute_file),
            path_scores=str(path_file),
            similarity=str(similarity_file),
            explanation_provider=None,  # プロンプト生成のみなので不要
            item_list=item_list,  # 店名取得用
        )

        # 未処理の項目のみ処理
        from tqdm import tqdm

        remaining_items = [
            item
            for item in existing_data
            if (item["uid"], item["iid"]) not in processed_uids_iids
        ]

        if remaining_items:
            logger.info(f"未処理の項目数: {len(remaining_items)}/{len(existing_data)}")

            for item in tqdm(remaining_items, desc=f"{split_name}のプロンプト生成中"):
                uid = item["uid"]
                iid = item["iid"]
                chosen = item.get("chosen", "")  # chosenが存在しない場合は空文字列

                try:
                    # 新しいプロンプトを生成
                    new_prompt = generator.prompt_generator.generate_prompt(
                        uid=uid,
                        iid=iid,
                        min_attribute_score=min_attribute_score,
                        top_paths=top_paths,
                        top_similar=top_similar,
                    )

                    # 既存のデータを保持し、プロンプトだけを置き換え
                    result = {
                        "uid": uid,
                        "iid": iid,
                        "prompt": new_prompt,
                        "chosen": chosen,
                        "reject": item.get("reject", "I DO NOT KNOW"),
                    }
                    # similarity_scoreが存在する場合のみ追加
                    if "similarity_score" in item:
                        result["similarity_score"] = item["similarity_score"]

                    results.append(result)

                    # 定期的に一時ファイルに保存（レジューム用）
                    if len(results) % 100 == 0:
                        generator.save_to_jsonl(
                            results, temp_output_file, verbose=False
                        )

                except Exception as e:
                    logger.error(f"エラーが発生しました (uid={uid}, iid={iid}): {e}")
                    # エラーが発生しても処理を続行
                    continue
        else:
            logger.info(f"すべての項目が既に処理済みです。")

        # 最終的な結果を保存
        generator.save_to_jsonl(results, output_file)
        # 一時ファイルを削除
        if temp_output_file.exists():
            temp_output_file.unlink()
        logger.info(
            f"{split_name}のデータを保存しました: {output_file} ({len(results)}件)"
        )

    logger.info("=" * 60)
    logger.info("プロンプト置き換え（計算付き）が完了しました！")


@app.command
def from_interactions(
    train_interactions: str,
    eval_interactions: str,
    test_interactions: str,
    user_profiles: str,
    item_profiles: str,
    model_path: str,
    output_dir: str = "packages/raft-data-generator/output",
    data_dir: str = "datasets/",
    data_name: str = "yelp",
    kg_file: str | None = None,
    entity_list_file: str | None = None,
    relation_list_file: str | None = None,
    min_attribute_score: float = 0.0,
    top_paths: int = 2,
    top_similar: int = 2,
    text_encoder: str = "sentence-transformers/multi-qa-distilbert-cos-v1",
    similarity_topk: int = 2,
    similarity_pruning_score: float = 0.0,
    max_hops: int = 3,
    min_hops: int = 3,
    ignore_relations: list[int] | None = None,
    exclude_relations: list[int] | None = None,
    path_selection_mode: str = "topk",
) -> None:
    """インタラクションファイルから直接RAFT形式のデータを生成（類似ノード、パス、属性スコアも計算）

    Args:
        train_interactions: 訓練用インタラクションファイル（CSV形式、uid, iidカラム）
        eval_interactions: 評価用インタラクションファイル（CSV形式、uid, iidカラム）
        test_interactions: テスト用インタラクションファイル（CSV形式、uid, iidカラム）
        user_profiles: ユーザープロファイルJSONのパス
        item_profiles: アイテムプロファイルJSONのパス
        model_path: 学習済みKGATモデルのパス
        output_dir: 出力ディレクトリのパス
        data_dir: データディレクトリ（デフォルト: datasets/）
        data_name: データセット名（デフォルト: yelp）
        kg_file: kg_final.txtのパス（Noneの場合はデフォルトパスを使用）
        entity_list_file: entity_list.txtのパス（オプション）
        relation_list_file: relation_list.txtのパス（オプション）
        min_attribute_score: 属性スコアの最小閾値（デフォルト: 0.0）
        top_paths: 含めるトップパスの数（デフォルト: 2）
        top_similar: 含める類似ユーザー/アイテムの数（デフォルト: 2）
        text_encoder: テキストエンコーダーモデル名（デフォルト: sentence-transformers/multi-qa-distilbert-cos-v1）
        similarity_topk: 類似ノードの最大数（デフォルト: 2）
        similarity_pruning_score: 類似度スコアの閾値（デフォルト: 0.0）
        max_hops: 最大ホップ数（デフォルト: 3）
        min_hops: 最小ホップ数（デフォルト: 3）
        ignore_relations: パス探索で無視するリレーションIDのリスト（オプション）
        exclude_relations: 属性スコア計算で除外するリレーションIDのリスト（オプション）
        path_selection_mode: パス選択モード（'topk' または 'diverse'、デフォルト: 'topk'）
    """
    import tempfile

    logger.info("インタラクションファイルからRAFTデータ生成を開始します")
    logger.info(f"訓練用インタラクション: {train_interactions}")
    logger.info(f"評価用インタラクション: {eval_interactions}")
    logger.info(f"テスト用インタラクション: {test_interactions}")
    logger.info(f"出力ディレクトリ: {output_dir}")

    # 一時ディレクトリを作成して中間ファイルを保存
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir_path / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # プロファイルと訓練データをロード（共通で使用）
    import json

    from kgat.config import KGATConfig
    from path_importance_calculator.calculator import PathImportanceCalculator
    from path_importance_calculator.model_loader import KGATModelLoader
    from semantic_similarity_retriever.retriever import (
        SemanticSimilarityRetriever,
        load_profiles,
    )
    from user_item_attribute_scorer.score_calculator import calculate_scores

    logger.info("プロファイルと訓練データをロード中...")
    user_profiles_dict, item_profiles_dict = load_profiles(user_profiles, item_profiles)
    train_df = pd.read_csv(train_interactions)

    # カラム名をuid, iidに統一（user/item, uid/iid, その他の形式に対応）
    if "uid" in train_df.columns and "iid" in train_df.columns:
        # 既にuid, iidの場合はそのまま
        pass
    elif "user" in train_df.columns and "item" in train_df.columns:
        # user, itemの場合はリネーム
        train_df = train_df.rename(columns={"user": "uid", "item": "iid"})
    elif len(train_df.columns) >= 2:
        # ヘッダーなしまたは他の形式の場合、最初の2列をuid, iidとして使用
        train_df.columns = ["uid", "iid"] + list(train_df.columns[2:])
        train_df = train_df[["uid", "iid"]]
    else:
        raise ValueError(
            f"CSVファイルには少なくとも2列（ユーザーIDとアイテムID）が必要です。現在のカラム: {train_df.columns.tolist()}"
        )

    # 類似ノード検索器を初期化（訓練データでグラフ構築）
    retriever = SemanticSimilarityRetriever(
        user_profiles=user_profiles_dict,
        item_profiles=item_profiles_dict,
        interactions_df=train_df,
        text_encoder=text_encoder,
        pruning_score=similarity_pruning_score,
    )

    # KGAT設定とモデルローダーを初期化（共通で使用）
    kgat_config = KGATConfig(
        data_dir=data_dir,
        data_name=data_name,
        file_prefix="total_",  # total_train.txt, total_test.txt, total_val.txt を使用
        embed_dim=64,
        relation_dim=64,
        laplacian_type="random-walk",
        aggregation_type="bi-interaction",
        conv_dim_list="[64,32,16]",
        use_pretrain=0,
    )

    model_loader = KGATModelLoader(str(model_path), kgat_config, logger)

    if kg_file is None:
        kg_file = str(Path(data_dir) / data_name / "kg_final.txt")

    # 各データセット（train/eval/test）ごとに計算してRAFTデータを生成
    for split_name, interactions_file in [
        ("train", train_interactions),
        ("eval", eval_interactions),
        ("test", test_interactions),
    ]:
        logger.info("=" * 60)
        logger.info(f"{split_name.upper()}データの処理を開始します")

        split_df = pd.read_csv(interactions_file)

        if len(split_df) == 0:
            logger.warning(f"{split_name}データが空です。スキップします。")
            continue

        # 1. 類似ノードを計算（既に存在する場合はスキップ）
        similarity_file = temp_dir / f"semantic_similarity_results_{split_name}.json"
        if similarity_file.exists():
            logger.info(
                f"ステップ1: {split_name}の類似ノードは既に計算済みです。スキップします: {similarity_file}"
            )
        else:
            logger.info(f"ステップ1: {split_name}の類似ノードを計算中...")
            similarity_results = retriever.retrieve_batch(
                split_df, topk=similarity_topk
            )
            # JSONL形式で保存（1行1JSONオブジェクト、メモリ効率が良い）
            with open(similarity_file, "w", encoding="utf-8") as f:
                for result in similarity_results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            logger.info(f"類似ノードを保存しました: {similarity_file}")

        # 2. パス重要度を計算（既に存在する場合はスキップ）
        path_suffix = "_diverse" if path_selection_mode == "diverse" else ""
        path_file = (
            temp_dir / f"path_importance_scores_with_names{path_suffix}_{split_name}.json"
        )
        if path_file.exists():
            logger.info(
                f"ステップ2: {split_name}のパス重要度は既に計算済みです。スキップします: {path_file}"
            )
        else:
            logger.info(f"ステップ2: {split_name}のパス重要度を計算中... (selection_mode={path_selection_mode})")

            calculator = PathImportanceCalculator(
                model_loader,
                max_hops=max_hops,
                min_hops=min_hops,
                ignore_relations=eval(ignore_relations) if ignore_relations else [],
                entity_list_file=entity_list_file,
                relation_list_file=relation_list_file,
            )

            calculator_results = calculator.calculate_for_pairs(
                str(interactions_file), topk=top_paths, selection_mode=path_selection_mode
            )
            calculator.save_results(calculator_results, str(path_file), format="jsonl")
            logger.info(f"パス重要度を保存しました: {path_file}")

        # 3. ユーザー・属性親和度を計算（既に存在する場合はスキップ）
        attribute_file = temp_dir / f"user_item_attribute_scores_{split_name}.csv"
        if attribute_file.exists():
            logger.info(
                f"ステップ3: {split_name}のユーザー・属性親和度は既に計算済みです。スキップします: {attribute_file}"
            )
        else:
            logger.info(f"ステップ3: {split_name}のユーザー・属性親和度を計算中...")

            calculate_scores(
                interactions_file=str(interactions_file),
                model_path=str(model_path),
                kg_file=kg_file,
                data_dir=data_dir,
                data_name=data_name,
                output_file=str(attribute_file),
                exclude_relations=eval(exclude_relations)
                if exclude_relations
                else None,
            )
            logger.info(f"ユーザー・属性親和度を保存しました: {attribute_file}")

        # 4. RAFTデータを生成
        logger.info(f"ステップ4: {split_name}のRAFTデータを生成中...")

        generator = RaftDataGenerator(
            user_profiles=user_profiles,
            item_profiles=item_profiles,
            attribute_scores=str(attribute_file),
            path_scores=str(path_file),
            similarity=str(similarity_file),
            explanation_provider=None,  # 説明生成は不要（chosenは空文字列）
        )

        data = generator.generate_batch(
            split_df,
            min_attribute_score=min_attribute_score,
            top_paths=top_paths,
            top_similar=top_similar,
            show_progress=True,
        )

        output_file = output_dir_path / f"{split_name}.json"
        generator.save_to_jsonl(data, output_file)
        logger.info(f"{split_name}のRAFTデータを保存しました: {output_file}")

    logger.info("=" * 60)
    logger.info("RAFTデータ生成が完了しました！")
    logger.info(f"出力ディレクトリ: {output_dir_path}")


if __name__ == "__main__":
    app()
