"""説明生成モジュール"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class ExplanationGenerator:
    """説明を生成するクラス"""

    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        api_key: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        batch_size: int = 16,
        use_chat_template: bool = False,
    ):
        """
        Args:
            provider: LLMプロバイダー ("openai" または "huggingface")
            model_name: 使用するモデル名
            api_key: OpenAI APIキー
            max_tokens: 生成する最大トークン数
            temperature: 生成時の温度パラメータ
            batch_size: バッチサイズ (HuggingFaceの場合)
            use_chat_template: チャットテンプレートを使用するかどうか
        """
        self.provider = provider
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.use_chat_template = use_chat_template

        if provider == "openai":
            # OpenAI APIキーの取得（引数 > 環境変数）
            api_key = api_key or os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "OpenAI APIキーが設定されていません。--api-keyで指定するか、環境変数OPENAI_API_KEYを設定してください。"
                )
            self.client = OpenAI(api_key=api_key)
            logging.info(f"OpenAI APIを使用: {model_name}")

        elif provider == "huggingface":
            logging.info(f"HuggingFace Transformersを使用: {model_name}")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"デバイス: {self.device}")

            # トークナイザーとモデルをロード
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            self.model.eval()

        else:
            raise ValueError(f"不明なプロバイダー: {provider}")

    def generate_single_openai(self, prompt: str, use_two_pass: bool = False) -> str:
        """OpenAI APIで単一のプロンプトから説明を生成

        Args:
            prompt: 入力プロンプト
            use_two_pass: 2パス生成を行うかどうか（1回目の結果を履歴に含めて2回目を生成）

        Returns:
            生成された説明
        """
        try:
            # 1回目の生成
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            first_output = response.choices[0].message.content.strip()

            if not use_two_pass:
                return first_output

            # 2回目の生成：1回目のプロンプトと結果を履歴に含めて再度生成
            response2 = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                    {
                        "role": "assistant",
                        "content": first_output,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            second_output = response2.choices[0].message.content.strip()

            return second_output

        except Exception as e:
            logging.error(f"OpenAI API呼び出しエラー: {e}")
            return f"[ERROR: {str(e)}]"

    def _apply_chat_template(self, prompt: str, first_output: str = "") -> str:
        """プロンプトをチャットテンプレートに変換（systemは固定、userは可変）

        Args:
            prompt: 元のプロンプト（ユーザーデータ部分）
            first_output: 1回目の出力（2パス目の場合に指定）

        Returns:
            チャットテンプレート適用後のプロンプト
        """
        # Systemプロンプト（固定）
        system_content = """Given the business title, business profile, and user profile, explain why the user would enjoy this business.

CRITICAL REQUIREMENTS:
- Write EXACTLY ONE SENTENCE (no periods in the middle, only at the end)
- Start with "The user would enjoy [business name] because" or "The user would enjoy this business because"
- Maximum 50 words total
- DO NOT make meta-comments about preferences or alignment
- Focus ONLY on describing the business's actual features: food, atmosphere, service, location, menu items
- NEVER say: "preferences", "aligns with",  "their profile", "they value", "that aligns with", "aligning withe"

### Examples (follow this structure - one sentence each):
Example 1: The user would enjoy Taco Riendo because it provides a convenient location, is open late, offers highly rated food with a fresh and flavorful al pastor burrito, and creates a welcoming atmosphere with Spanish mood music, making it a delightful dining experience.
Example 2: The user would enjoy this business for its unique Taiwanese shaved ice flavors, cozy atmosphere with board games, and friendly shop owners offering discounts, making it a perfect spot to hang out and cool down with friends.
Example 3: The user would enjoy J Devoti Trattoria for its excellent food quality, especially the charcuterie board, which exceeded their expectations, making it a standout dining experience worth returning for."""

        # Userプロンプト（promptからユーザーデータ部分を抽出）
        # "Business title:" から始まる部分をuser_contentに
        if "Business title:" in prompt:
            user_content = prompt
        else:
            # フォーマットが異なる場合は全体をuserに
            user_content = prompt

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        # 2パス目の場合、チャット履歴を追加
        if first_output:
            messages.append({"role": "assistant", "content": first_output})
            messages.append(
                {
                    "role": "user",
                    "content": f"{user_content}\n\nPlease provide a better explanation. Make sure it follows all the requirements: exactly ONE sentence, max 50 words, and no meta-comments about preferences or alignment.",
                }
            )

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.inference_mode()
    def generate_batch_huggingface(
        self, prompts: list[str], use_two_pass: bool = False
    ) -> list[str]:
        """HuggingFace Transformersでバッチのプロンプトから説明を生成

        Args:
            prompts: 入力プロンプトのリスト
            use_two_pass: 2パス生成を行うかどうか

        Returns:
            生成された説明のリスト
        """
        if not use_two_pass:
            # チャットテンプレートを適用（有効な場合）
            if self.use_chat_template:
                formatted_prompts = [self._apply_chat_template(p) for p in prompts]
            else:
                formatted_prompts = prompts

            # 1パス目の生成
            inputs = self.tokenizer(
                formatted_prompts,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            output_ids = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=GenerationConfig(
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                ),
            ).tolist()

            # 入力部分を除いた生成結果のみを取得
            real_output_ids = [
                output_id[len(inputs.input_ids[i]) :]
                for i, output_id in enumerate(output_ids)
            ]
            output_strs = self.tokenizer.batch_decode(
                real_output_ids, skip_special_tokens=True
            )

            return [output.strip() for output in output_strs]

        else:
            # 2パス生成
            # 1パス目（リトライなし）
            first_outputs = self.generate_batch_huggingface(prompts, use_two_pass=False)

            # 2パス目：1パス目の結果を踏まえて改善指示を追加
            # チャットテンプレートモードの場合は履歴に含める
            formatted_second_prompts = []
            for prompt, first_output in zip(prompts, first_outputs):
                if self.use_chat_template:
                    # チャットテンプレートモード：履歴に1パス目の結果を含める
                    formatted_prompt = self._apply_chat_template(prompt, first_output)
                    formatted_second_prompts.append(formatted_prompt)
                else:
                    # 通常モード：プロンプトに1パス目の結果を追加
                    second_prompt = f"{prompt}\n\nPrevious attempt (too long or violated constraints): {first_output}\n\n### Better Explanation (one sentence, max 50 words, do not mention preferences/alignment):"
                    formatted_second_prompts.append(second_prompt)

            inputs = self.tokenizer(
                formatted_second_prompts,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            output_ids = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=GenerationConfig(
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                ),
            ).tolist()

            # 入力部分を除いた生成結果のみを取得
            real_output_ids = [
                output_id[len(inputs.input_ids[i]) :]
                for i, output_id in enumerate(output_ids)
            ]
            output_strs = self.tokenizer.batch_decode(
                real_output_ids, skip_special_tokens=True
            )

            return [output.strip() for output in output_strs]

    def generate_from_csv(
        self,
        prompts_file: str,
        output_file: str,
        ground_truth_file: str = "",
        start_index: int = 0,
        max_samples: int = -1,
        use_two_pass: bool = False,
    ) -> None:
        """CSVファイルからプロンプトを読み込んで説明を生成

        Args:
            prompts_file: プロンプトCSVファイルのパス
            output_file: 出力JSONLファイルのパス
            ground_truth_file: 正解説明を含むJSONLファイルのパス（オプション）
            start_index: 開始インデックス（レジューム用）
            max_samples: 処理する最大サンプル数（-1で全件処理）
            use_two_pass: 2パス生成を行うかどうか
        """
        logging.info(f"プロンプトファイルをロード中: {prompts_file}")
        df = pd.read_csv(prompts_file)
        logging.info(f"プロンプト数: {len(df)}")

        # 正解説明の辞書を作成
        ground_truth_dict = {}
        if ground_truth_file and Path(ground_truth_file).exists():
            logging.info(f"正解説明をロード中: {ground_truth_file}")
            with open(ground_truth_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    uid = data["source_data"]["uid"]
                    iid = data["source_data"]["iid"]
                    chosen = data["source_data"]["chosen"]
                    ground_truth_dict[(uid, iid)] = chosen
            logging.info(f"正解説明数: {len(ground_truth_dict)}")
        else:
            if ground_truth_file:
                logging.warning(
                    f"正解説明ファイルが見つかりません: {ground_truth_file}"
                )
            logging.info("正解説明なしで実行します（chosenフィールドは空文字列）")

        # 出力ディレクトリを作成
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # レジューム機能：既に処理済みの行数を確認
        if output_path.exists() and start_index == 0:
            start_index = len(open(output_path).readlines())
            logging.info(f"既存の出力ファイルを検出。{start_index}行目から再開します。")

        # 処理する範囲を決定
        end_index = (
            len(df) if max_samples == -1 else min(start_index + max_samples, len(df))
        )
        logging.info(f"処理範囲: {start_index} ~ {end_index}")

        if self.provider == "openai":
            # OpenAI: 1件ずつ処理
            for i in tqdm(range(start_index, end_index), desc="説明生成中"):
                row = df.iloc[i]
                uid = int(row["uid"])
                iid = int(row["iid"])
                prompt = row["prompt"]

                explanation = self.generate_single_openai(
                    prompt, use_two_pass=use_two_pass
                )

                # 正解説明を取得
                chosen = ground_truth_dict.get((uid, iid), "")

                # 結果を保存（G-Refer形式）
                result = {
                    "index": i,
                    "source_data": {
                        "uid": uid,
                        "iid": iid,
                        "prompt": prompt,
                        "chosen": chosen,
                        "reject": "I DO NOT KNOW",
                    },
                    "input_str": prompt,
                    "output_str": explanation,
                }

                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        elif self.provider == "huggingface":
            # HuggingFace: バッチ処理
            for i in tqdm(
                range(start_index, end_index, self.batch_size), desc="説明生成中"
            ):
                batch_df = df.iloc[i : min(i + self.batch_size, end_index)]

                uids = batch_df["uid"].tolist()
                iids = batch_df["iid"].tolist()
                prompts = batch_df["prompt"].tolist()

                explanations = self.generate_batch_huggingface(
                    prompts, use_two_pass=use_two_pass
                )

                # 結果を保存（G-Refer形式）
                for j, (uid, iid, prompt, explanation) in enumerate(
                    zip(uids, iids, prompts, explanations)
                ):
                    # 正解説明を取得
                    chosen = ground_truth_dict.get((int(uid), int(iid)), "")

                    result = {
                        "index": i + j,
                        "source_data": {
                            "uid": int(uid),
                            "iid": int(iid),
                            "prompt": prompt,
                            "chosen": chosen,
                            "reject": "I DO NOT KNOW",
                        },
                        "input_str": prompt,
                        "output_str": explanation,
                    }

                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

        logging.info(f"説明生成完了！結果を保存しました: {output_file}")
