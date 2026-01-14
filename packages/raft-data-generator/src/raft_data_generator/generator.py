"""RAFT形式のデータ生成モジュール"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from explainable_rec_prompt_generator.data_loader import DataLoader
from explainable_rec_prompt_generator.prompt_generator import PromptGenerator
from explanation_generator.generator import ExplanationGenerator

logger = logging.getLogger(__name__)


class RaftDataGenerator:
    """RAFT形式のfine-tuningデータを生成するクラス"""

    def __init__(
        self,
        user_profiles: str | Path,
        item_profiles: str | Path,
        attribute_scores: str | Path,
        path_scores: str | Path,
        similarity: str | Path,
        explanation_provider: str | None = "openai",
        explanation_model: str = "gpt-3.5-turbo",
        explanation_api_key: str = "",
        explanation_max_tokens: int = 150,
        explanation_temperature: float = 0.7,
        explanation_batch_size: int = 16,
        item_list: str | Path | None = None,
    ):
        """RAFTデータジェネレーターを初期化

        Args:
            user_profiles: ユーザープロファイルJSONのパス
            item_profiles: アイテムプロファイルJSONのパス
            attribute_scores: 属性スコアCSVのパス
            path_scores: パス重要度スコアJSONのパス
            similarity: セマンティック類似度結果JSONのパス
            explanation_provider: 説明生成のプロバイダー ("openai", "huggingface", または None)
            explanation_model: 説明生成に使用するモデル名
            explanation_api_key: OpenAI APIキー（オプション）
            explanation_max_tokens: 説明生成の最大トークン数
            explanation_temperature: 説明生成の温度パラメータ
            explanation_batch_size: 説明生成のバッチサイズ（HuggingFaceの場合）
            item_list: item_list.txtのパス（オプション、店名取得用）
        """
        # データローダーとプロンプトジェネレーターを初期化
        self.data_loader = DataLoader(
            user_profile_path=user_profiles,
            item_profile_path=item_profiles,
            attribute_scores_path=attribute_scores,
            path_scores_path=path_scores,
            similarity_path=similarity,
        )

        # Load item titles if item_list is provided
        if item_list is not None:
            self.data_loader.load_item_titles(item_list)

        self.prompt_generator = PromptGenerator(self.data_loader)

        # 説明ジェネレーターを初期化（プロンプト生成のみの場合はNone）
        if explanation_provider is not None:
            self.explanation_generator = ExplanationGenerator(
                provider=explanation_provider,
                model_name=explanation_model,
                api_key=explanation_api_key,
                max_tokens=explanation_max_tokens,
                temperature=explanation_temperature,
                batch_size=explanation_batch_size,
            )
        else:
            self.explanation_generator = None

        # 類似度データをロード
        self.similarity_data = self._load_similarity_data(similarity)

    def _load_similarity_data(
        self, similarity_path: str | Path
    ) -> dict[tuple[int, int], dict[str, Any]]:
        """類似度データをロードして辞書形式に変換（JSONL形式対応）

        Args:
            similarity_path: 類似度データJSONLまたはJSONファイルのパス

        Returns:
            (uid, iid) -> 類似度データの辞書
        """
        similarity_dict = {}
        with open(similarity_path, encoding="utf-8") as f:
            # まずJSONL形式を試す（1行1JSONオブジェクト）
            try:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        uid = entry["user_id"]
                        iid = entry["item_id"]
                        similarity_dict[(uid, iid)] = entry
            except json.JSONDecodeError:
                # JSONL形式でない場合は、JSON配列形式として読み込む（後方互換性）
                f.seek(0)
                data = json.load(f)
                for entry in data:
                    uid = entry["user_id"]
                    iid = entry["item_id"]
                    similarity_dict[(uid, iid)] = entry
        return similarity_dict

    def _calculate_similarity_score(self, uid: int, iid: int) -> float:
        """類似度スコアを計算

        Args:
            uid: ユーザーID
            iid: アイテムID

        Returns:
            類似度スコア（類似ユーザーと類似アイテムの平均スコア、または0.0）
        """
        entry = self.similarity_data.get((uid, iid))
        if not entry:
            return 0.0

        similar_users = entry.get("similar_users", [])
        similar_items = entry.get("similar_items", [])

        scores = []
        if similar_users:
            # 類似ユーザーのスコアの平均
            user_scores = [score for _, score in similar_users]
            scores.extend(user_scores)

        if similar_items:
            # 類似アイテムのスコアの平均
            item_scores = [score for _, score in similar_items]
            scores.extend(item_scores)

        if scores:
            return sum(scores) / len(scores)
        return 0.0

    def generate_single(
        self,
        uid: int,
        iid: int,
        min_attribute_score: float = 0.0,
        top_paths: int = 2,
        top_similar: int = 2,
    ) -> dict[str, Any]:
        """単一のユーザー・アイテムペアに対してRAFT形式のデータを生成

        Args:
            uid: ユーザーID
            iid: アイテムID
            min_attribute_score: 属性スコアの最小閾値
            top_paths: 含めるトップパスの数
            top_similar: 含める類似ユーザー/アイテムの数

        Returns:
            RAFT形式のデータ辞書
        """
        # プロンプトを生成
        prompt = self.prompt_generator.generate_prompt(
            uid=uid,
            iid=iid,
            min_attribute_score=min_attribute_score,
            top_paths=top_paths,
            top_similar=top_similar,
        )

        # 説明を生成
        if self.explanation_generator is None:
            raise ValueError("説明生成にはexplanation_providerを指定してください")

        if self.explanation_generator.provider == "openai":
            chosen = self.explanation_generator.generate_single_openai(prompt)
        else:
            # HuggingFaceの場合はバッチ処理が必要だが、単一の場合は1要素のリストとして処理
            explanations = self.explanation_generator.generate_batch_huggingface(
                [prompt]
            )
            chosen = explanations[0] if explanations else ""

        # 類似度スコアを計算
        similarity_score = self._calculate_similarity_score(uid, iid)

        # RAFT形式のデータを作成
        return {
            "uid": uid,
            "iid": iid,
            "prompt": prompt,
            "chosen": chosen,
            "reject": "I DO NOT KNOW",
            "similarity_score": similarity_score,
        }

    def generate_batch(
        self,
        interactions_df: pd.DataFrame,
        min_attribute_score: float = 0.0,
        top_paths: int = 2,
        top_similar: int = 2,
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        """複数のユーザー・アイテムペアに対してRAFT形式のデータを生成

        Args:
            interactions_df: uid, iidカラムを持つDataFrame
            min_attribute_score: 属性スコアの最小閾値
            top_paths: 含めるトップパスの数
            top_similar: 含める類似ユーザー/アイテムの数
            show_progress: プログレスバーを表示するかどうか

        Returns:
            RAFT形式のデータのリスト
        """
        from tqdm import tqdm

        results = []

        # プロンプトを一括生成
        prompts = []
        uids = []
        iids = []
        iterator = (
            tqdm(
                interactions_df.iterrows(),
                total=len(interactions_df),
                desc="プロンプト生成中",
            )
            if show_progress
            else interactions_df.iterrows()
        )

        for _, row in iterator:
            uid = int(row["uid"])
            iid = int(row["iid"])
            uids.append(uid)
            iids.append(iid)

            prompt = self.prompt_generator.generate_prompt(
                uid=uid,
                iid=iid,
                min_attribute_score=min_attribute_score,
                top_paths=top_paths,
                top_similar=top_similar,
            )
            prompts.append(prompt)

        # 説明を一括生成（explanation_generatorがNoneの場合は空文字列）
        if self.explanation_generator is None:
            logger.info("説明生成はスキップします（chosenは空文字列）")
            chosen_list = [""] * len(prompts)
        else:
            logger.info("説明を生成中...")
            if self.explanation_generator.provider == "openai":
                # OpenAIの場合は1件ずつ処理
                from tqdm import tqdm

                chosen_list = []
                for prompt in tqdm(prompts, desc="説明生成中"):
                    chosen = self.explanation_generator.generate_single_openai(prompt)
                    chosen_list.append(chosen)
            else:
                # HuggingFaceの場合はバッチ処理
                chosen_list = self.explanation_generator.generate_batch_huggingface(
                    prompts
                )

        # RAFT形式のデータを作成
        logger.info("RAFT形式のデータを作成中...")
        for uid, iid, prompt, chosen in zip(uids, iids, prompts, chosen_list):
            similarity_score = self._calculate_similarity_score(uid, iid)
            results.append(
                {
                    "uid": uid,
                    "iid": iid,
                    "prompt": prompt,
                    "chosen": chosen,
                    "reject": "I DO NOT KNOW",
                    "similarity_score": similarity_score,
                }
            )

        return results

    def replace_prompts_from_existing_data(
        self,
        input_file: str | Path,
        min_attribute_score: float = 0.0,
        top_paths: int = 2,
        top_similar: int = 2,
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        """既存のRAFTデータファイルからプロンプトだけを置き換える

        Args:
            input_file: 既存のRAFTデータファイル（JSONL形式）のパス
            min_attribute_score: 属性スコアの最小閾値
            top_paths: 含めるトップパスの数
            top_similar: 含める類似ユーザー/アイテムの数
            show_progress: プログレスバーを表示するかどうか

        Returns:
            プロンプトを置き換えたRAFT形式のデータのリスト
        """
        from tqdm import tqdm

        # 既存データを読み込む
        logger.info(f"既存のRAFTデータを読み込み中: {input_file}")
        existing_data = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_data.append(json.loads(line))

        logger.info(f"読み込んだデータ数: {len(existing_data)}")

        # プロンプトを生成して置き換え
        results = []
        iterator = (
            tqdm(existing_data, desc="プロンプト生成中")
            if show_progress
            else existing_data
        )

        for item in iterator:
            uid = item["uid"]
            iid = item["iid"]
            chosen = item.get("chosen", "")  # chosenが存在しない場合は空文字列

            # 新しいプロンプトを生成
            new_prompt = self.prompt_generator.generate_prompt(
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

        return results

    def save_to_jsonl(
        self,
        data: list[dict[str, Any]],
        output_path: str | Path,
        verbose: bool = True,
    ) -> None:
        """RAFT形式のデータをJSONL形式で保存

        Args:
            data: RAFT形式のデータのリスト
            output_path: 出力ファイルのパス
            verbose: ログ出力するかどうか（デフォルト: True）
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        if verbose:
            logger.info(f"データを保存しました: {output_path} ({len(data)}件)")

    def compute_semantic_similarity(
        self,
        train_interactions: str | Path,
        eval_interactions: str | Path,
        output_file: str | Path,
        text_encoder: str = "sentence-transformers/multi-qa-distilbert-cos-v1",
        topk: int = 2,
        pruning_score: float = 0.0,
    ) -> None:
        """類似ノードを計算して保存

        Args:
            train_interactions: 学習用インタラクションファイルのパス
            eval_interactions: 評価用インタラクションファイルのパス
            output_file: 出力ファイルのパス
            text_encoder: テキストエンコーダーモデル名
            topk: 取得する類似ノードの最大数（デフォルト: 2）
            pruning_score: 類似度スコアの閾値
        """
        import pandas as pd
        from semantic_similarity_retriever.retriever import (
            SemanticSimilarityRetriever,
            load_profiles,
        )

        logger.info("類似ノードを計算中...")
        logger.info(f"学習用インタラクション: {train_interactions}")
        logger.info(f"評価用インタラクション: {eval_interactions}")

        # プロファイルをロード
        user_profiles_dict, item_profiles_dict = load_profiles(
            str(self.data_loader.user_profile_path),
            str(self.data_loader.item_profile_path),
        )

        # 学習用インタラクションをロード
        train_df = pd.read_csv(train_interactions)

        # 評価用インタラクションをロード
        eval_df = pd.read_csv(eval_interactions)

        # 類似ノード検索を実行
        retriever = SemanticSimilarityRetriever(
            user_profiles=user_profiles_dict,
            item_profiles=item_profiles_dict,
            interactions_df=train_df,
            text_encoder=text_encoder,
            pruning_score=pruning_score,
        )

        results = retriever.retrieve_batch(eval_df, topk=topk)

        # 結果を保存（JSONL形式で保存、メモリ効率が良い）
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        logger.info(f"類似ノードを保存しました: {output_path}")

    def compute_path_importance(
        self,
        interactions_file: str | Path,
        model_path: str | Path,
        output_file: str | Path,
        data_dir: str = "packages/kgat/datasets/",
        data_name: str = "yelp",
        max_hops: int = 3,
        min_hops: int = 3,
        ignore_relations: list[int] | None = None,
        entity_list_file: str | None = None,
        relation_list_file: str | None = None,
        embed_dim: int = 64,
        relation_dim: int = 64,
        laplacian_type: str = "random-walk",
        aggregation_type: str = "bi-interaction",
        conv_dim_list: str = "[64,32,16]",
        topk: int = 2,
        selection_mode: str = "topk",
    ) -> None:
        """パス重要度を計算して保存

        Args:
            interactions_file: インタラクションファイルのパス
            model_path: 学習済みKGATモデルのパス
            output_file: 出力ファイルのパス
            data_dir: データディレクトリ
            data_name: データセット名
            max_hops: 最大ホップ数
            min_hops: 最小ホップ数
            ignore_relations: 無視するリレーションIDのリスト
            entity_list_file: entity_list.txtのパス
            relation_list_file: relation_list.txtのパス
            embed_dim: 埋め込み次元
            relation_dim: リレーション次元
            laplacian_type: ラプラシアンタイプ
            aggregation_type: 集約タイプ
            conv_dim_list: 畳み込み次元リスト
            topk: 各ペアごとに保存するパスの最大数（selection_mode='topk'の場合のみ使用）
            selection_mode: パス選択モード ('topk' または 'diverse')
        """
        from kgat.config import KGATConfig
        from path_importance_calculator.calculator import PathImportanceCalculator
        from path_importance_calculator.model_loader import KGATModelLoader

        logger.info("パス重要度を計算中...")
        logger.info(f"インタラクションファイル: {interactions_file}")
        logger.info(f"モデルパス: {model_path}")
        logger.info(f"選択モード: {selection_mode}")

        # KGAT設定を作成
        kgat_config = KGATConfig(
            data_dir=data_dir,
            data_name=data_name,
            embed_dim=embed_dim,
            relation_dim=relation_dim,
            laplacian_type=laplacian_type,
            aggregation_type=aggregation_type,
            conv_dim_list=conv_dim_list,
            use_pretrain=0,
        )

        # モデルローダーを作成
        model_loader = KGATModelLoader(str(model_path), kgat_config, logger)

        # 計算器を作成
        calculator = PathImportanceCalculator(
            model_loader,
            max_hops=max_hops,
            min_hops=min_hops,
            ignore_relations=ignore_relations or [],
            entity_list_file=entity_list_file,
            relation_list_file=relation_list_file,
        )

        # パス重要度を計算
        results = calculator.calculate_for_pairs(
            str(interactions_file), topk=topk, selection_mode=selection_mode
        )

        # 結果を保存
        calculator.save_results(results, str(output_file), format="json")

        logger.info(f"パス重要度を保存しました: {output_file}")

    def compute_attribute_scores(
        self,
        interactions_file: str | Path,
        model_path: str | Path,
        output_file: str | Path,
        data_dir: str = "datasets/",
        data_name: str = "yelp",
        kg_file: str | None = None,
        exclude_relations: list[int] | None = None,
    ) -> None:
        """ユーザー・属性親和度を計算して保存

        Args:
            interactions_file: インタラクションファイルのパス
            model_path: 学習済みKGATモデルのパス
            output_file: 出力ファイルのパス
            data_dir: データディレクトリ
            data_name: データセット名
            kg_file: kg_final.txtのパス（Noneの場合はデフォルトパスを使用）
            exclude_relations: 除外するリレーションIDのリスト
        """
        from user_item_attribute_scorer.score_calculator import calculate_scores

        logger.info("ユーザー・属性親和度を計算中...")
        logger.info(f"インタラクションファイル: {interactions_file}")
        logger.info(f"モデルパス: {model_path}")

        if kg_file is None:
            kg_file = str(Path(data_dir) / data_name / "kg_final.txt")

        # スコア計算を実行
        calculate_scores(
            interactions_file=str(interactions_file),
            model_path=str(model_path),
            kg_file=kg_file,
            data_dir=data_dir,
            data_name=data_name,
            output_file=str(output_file),
            exclude_relations=exclude_relations,
        )

        logger.info(f"ユーザー・属性親和度を保存しました: {output_file}")
