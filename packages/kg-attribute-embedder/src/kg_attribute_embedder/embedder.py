"""アイテムの属性の埋め込みと類似性分析を行うモジュール"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class AttributeEmbedder:
    """アイテムの属性を埋め込み、類似性を分析するクラス"""

    def __init__(self, model_name: str = "sentence-transformers/multi-qa-distilbert-cos-v1"):
        """
        Args:
            model_name: 使用するSentence Transformerのモデル名
        """
        self.model = SentenceTransformer(model_name)
        self.triplets_data: list[dict[str, Any]] = []
        self.tail_counts: dict[str, int] = {}
        # relation-tailのユニークな組み合わせをキーとする
        self.embeddings: dict[tuple[str, str], np.ndarray] = {}
        self.sentences: dict[tuple[str, str], str] = {}
        # 各relation-tailがどのitemに属しているか
        self.relation_tail_items: dict[tuple[str, str], set[str]] = {}

    def load_kg_data(self, kg_path: str | Path) -> None:
        """知識グラフデータを読み込む

        Args:
            kg_path: 知識グラフJSONファイルのパス
        """
        with open(kg_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.triplets_data = []
        # relation-tailのユニークな組み合わせを追跡
        relation_tail_set: set[tuple[str, str]] = set()
        
        for item in data:
            iid = item["iid"]
            for triplet in item["triplets"]:
                item_id, relation, tail = triplet
                self.triplets_data.append({
                    "iid": iid,
                    "item_id": item_id,
                    "relation": relation,
                    "tail": tail,
                })
                
                # relation-tailの組み合わせを記録
                rt_key = (relation, tail)
                relation_tail_set.add(rt_key)
                
                # どのitemがこのrelation-tailを持っているかを記録
                if rt_key not in self.relation_tail_items:
                    self.relation_tail_items[rt_key] = set()
                self.relation_tail_items[rt_key].add(item_id)

        print(f"読み込んだトリプレット数: {len(self.triplets_data)}")
        print(f"ユニークなrelation-tailの組み合わせ数: {len(relation_tail_set)}")

    def count_tails(self) -> None:
        """tail（属性値）の出現回数をカウントする"""
        tails = [t["tail"] for t in self.triplets_data]
        self.tail_counts = Counter(tails)
        print(f"ユニークなtail数: {len(self.tail_counts)}")
        
        # 出現回数が1-2回のtailの数を表示
        rare_tails = sum(1 for count in self.tail_counts.values() if count <= 2)
        print(f"出現回数が1-2回のtail数: {rare_tails}")

    def generate_embeddings(self, batch_size: int = 32) -> None:
        """ユニークなrelation-tailの組み合わせに対して文を生成し、埋め込みを計算する

        Args:
            batch_size: バッチサイズ
        """
        print("埋め込みを生成中（ユニークなrelation-tailのみ）...")
        
        # ユニークなrelation-tailの組み合わせに対して文を生成
        relation_tail_keys = []
        sentences = []
        
        # relation-tailのユニークな組み合わせを取得
        unique_relation_tails = sorted(self.relation_tail_items.keys())
        
        for relation, tail in unique_relation_tails:
            rt_key = (relation, tail)
            sentence = f"This business {relation} {tail}"
            relation_tail_keys.append(rt_key)
            sentences.append(sentence)
            self.sentences[rt_key] = sentence

        # バッチ処理で埋め込みを生成
        embeddings_list = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # 埋め込みを辞書に保存
        for rt_key, embedding in zip(relation_tail_keys, embeddings_list):
            self.embeddings[rt_key] = embedding

        print(f"埋め込みの生成完了: {len(self.embeddings)}個（ユニークなrelation-tail）")

    def save_embeddings(self, output_path: str | Path) -> None:
        """埋め込みデータを保存する

        Args:
            output_path: 出力ファイルパス（.npz形式）
        """
        # データを保存用に整形
        keys = list(self.embeddings.keys())
        embeddings_array = np.array([self.embeddings[k] for k in keys])
        
        # メタデータを準備（キーは(relation, tail)のタプル）
        metadata = {
            "keys": keys,
            "sentences": [self.sentences[k] for k in keys],
            "tail_counts": [self.tail_counts[k[1]] for k in keys],  # k[1]がtail
            "is_rare": [self.tail_counts[k[1]] <= 2 for k in keys],
            "item_ids": [list(self.relation_tail_items[k]) for k in keys],  # どのitemが持っているか
        }

        # 保存
        np.savez_compressed(
            output_path,
            embeddings=embeddings_array,
            keys=np.array(keys, dtype=object),
            sentences=np.array(metadata["sentences"], dtype=object),
            tail_counts=np.array(metadata["tail_counts"]),
            is_rare=np.array(metadata["is_rare"]),
            item_ids=np.array(metadata["item_ids"], dtype=object),
        )
        print(f"埋め込みを保存しました: {output_path}")

    def load_embeddings(self, input_path: str | Path) -> None:
        """保存された埋め込みデータを読み込む

        Args:
            input_path: 入力ファイルパス（.npz形式）
        """
        data = np.load(input_path, allow_pickle=True)
        
        keys = data["keys"]
        embeddings_array = data["embeddings"]
        sentences = data["sentences"]
        
        self.embeddings = {}
        self.sentences = {}
        
        for i, key in enumerate(keys):
            key_tuple = tuple(key)
            self.embeddings[key_tuple] = embeddings_array[i]
            self.sentences[key_tuple] = sentences[i]
            
        print(f"埋め込みを読み込みました: {len(self.embeddings)}個")

    def compute_similarities(
        self,
        threshold: float = 0.8,
        output_path: str | Path | None = None,
        chunk_size: int = 1000,
    ) -> int:
        """すべての埋め込み間の類似度を計算し、類似するペアを列挙する
        
        メモリ効率のため、バッチ処理で類似度を計算し、見つかったペアは
        即座にファイルに書き込みます（ストリーミング方式）。

        Args:
            threshold: 類似度の閾値（この値以上のペアを抽出）
            output_path: 結果を保存するJSONファイルのパス（必須）
            chunk_size: 一度に処理する埋め込みのチャンクサイズ

        Returns:
            見つかった類似ペア数
        """
        if output_path is None:
            raise ValueError("output_pathは必須です")
        
        print(f"類似度を計算中（閾値: {threshold}、チャンクサイズ: {chunk_size}）...")
        print("ストリーミング書き込みモード: メモリ効率を最適化")
        
        keys = list(self.embeddings.keys())
        embeddings_array = np.array([self.embeddings[k] for k in keys])
        n = len(keys)
        
        # 埋め込みを正規化（コサイン類似度の高速化）
        from sklearn.preprocessing import normalize
        embeddings_normalized = normalize(embeddings_array, norm='l2', axis=1)
        
        # 一時ファイル（JSON Lines形式）
        temp_path = Path(output_path).parent / f"{Path(output_path).stem}_temp.jsonl"
        pair_count = 0
        
        # ファイルを開いてストリーミング書き込み
        with open(temp_path, "w", encoding="utf-8") as f:
            # チャンクごとに類似度を計算
            for i in range(0, n, chunk_size):
                chunk_end = min(i + chunk_size, n)
                chunk = embeddings_normalized[i:chunk_end]
                
                # このチャンクと残りのすべての埋め込みとの類似度を計算
                # i以降のみを対象にすることで重複を避ける
                remaining = embeddings_normalized[i:n]
                similarities = chunk @ remaining.T  # 正規化済みなので内積=コサイン類似度
                
                # 閾値以上のペアを抽出
                for local_i in range(len(chunk)):
                    global_i = i + local_i
                    # local_i+1から開始することで対角成分と重複を避ける
                    for local_j in range(local_i + 1, len(remaining)):
                        global_j = i + local_j
                        similarity = similarities[local_i, local_j]
                        
                        if similarity >= threshold:
                            rt_key1 = keys[global_i]  # (relation, tail)
                            rt_key2 = keys[global_j]  # (relation, tail)
                            relation1, tail1 = rt_key1
                            relation2, tail2 = rt_key2
                            
                            # 最小限の情報のみ保存
                            pair_info = {
                                "r1": relation1,  # relation1
                                "t1": tail1,      # tail1
                                "c1": self.tail_counts[tail1],  # tail1の出現回数
                                "r2": relation2,  # relation2
                                "t2": tail2,      # tail2
                                "c2": self.tail_counts[tail2],  # tail2の出現回数
                                "s": round(float(similarity), 3),  # similarity (小数点以下3桁)
                            }
                            # JSON Lines形式で即座に書き込み（コンパクト形式）
                            f.write(json.dumps(pair_info, ensure_ascii=False, separators=(',', ':')) + "\n")
                            pair_count += 1
                
                # 進捗表示
                if (i // chunk_size) % 10 == 0 or i + chunk_size >= n:
                    print(f"  処理中: {i}/{n} ({i/n*100:.1f}%) - 見つかったペア: {pair_count}")
        
        print(f"\n類似するペア数: {pair_count}")
        print(f"一時ファイルに書き込み完了: {temp_path}")
        
        # 一時ファイルを読み込んでソートし、最終的なJSONファイルに書き込み
        print("類似度順にソート中...")
        self._sort_and_save_pairs(temp_path, output_path)
        
        # 一時ファイルを削除
        temp_path.unlink()
        print(f"類似ペアを保存しました: {output_path}")
        
        return pair_count
    
    def _sort_and_save_pairs(self, temp_path: Path, output_path: str | Path) -> None:
        """一時ファイルから類似ペアを読み込み、ソートして保存する
        
        大きなファイルの場合は外部ソート（チャンクに分割してソート）を使用します。
        出力はJSON Lines形式（.jsonl）で保存します。
        
        Args:
            temp_path: 一時ファイルのパス（JSON Lines形式）
            output_path: 最終的な出力ファイルのパス（JSON Lines形式）
        """
        import heapq
        import tempfile
        
        # ファイルサイズを確認
        file_size = temp_path.stat().st_size
        file_size_gb = file_size / (1024 ** 3)
        print(f"一時ファイルサイズ: {file_size_gb:.2f} GB")
        
        # 1GB以上の場合は外部ソートを使用
        if file_size > 1 * 1024 ** 3:
            print("大きなファイルのため、外部ソートを使用します...")
            self._external_sort(temp_path, output_path)
        else:
            print("通常のソートを使用します...")
            # 小さいファイルは通常のソート
            pairs = []
            with open(temp_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        pairs.append(json.loads(line))
            
            pairs.sort(key=lambda x: x["s"], reverse=True)  # "similarity" -> "s"
            
            # JSON Lines形式で保存（コンパクト形式）
            with open(output_path, "w", encoding="utf-8") as f:
                for pair in pairs:
                    f.write(json.dumps(pair, ensure_ascii=False, separators=(',', ':')) + "\n")
    
    def _external_sort(self, temp_path: Path, output_path: str | Path, chunk_lines: int = 100000) -> None:
        """外部ソート: 大きなファイルをメモリ効率的にソートする
        
        Args:
            temp_path: 入力ファイル（JSON Lines形式）
            output_path: 出力ファイル（JSON形式）
            chunk_lines: 一度にメモリに読み込む行数
        """
        import heapq
        import tempfile
        
        # ステップ1: ファイルをチャンクに分割し、各チャンクをソートして一時ファイルに保存
        chunk_files = []
        chunk_num = 0
        
        print("ステップ1: チャンクに分割してソート中...")
        with open(temp_path, "r", encoding="utf-8") as f:
            chunk = []
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    chunk.append(json.loads(line))
                
                if len(chunk) >= chunk_lines:
                    # チャンクをソートして一時ファイルに保存
                    chunk.sort(key=lambda x: x["s"], reverse=True)  # "similarity" -> "s"
                    chunk_file = temp_path.parent / f"chunk_{chunk_num}.jsonl"
                    with open(chunk_file, "w", encoding="utf-8") as cf:
                        for item in chunk:
                            cf.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')) + "\n")
                    chunk_files.append(chunk_file)
                    chunk_num += 1
                    chunk = []
                    print(f"  チャンク {chunk_num} 完了（処理済み行数: {line_num}）")
            
            # 残りのチャンクを保存
            if chunk:
                chunk.sort(key=lambda x: x["s"], reverse=True)  # "similarity" -> "s"
                chunk_file = temp_path.parent / f"chunk_{chunk_num}.jsonl"
                with open(chunk_file, "w", encoding="utf-8") as cf:
                    for item in chunk:
                        cf.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')) + "\n")
                chunk_files.append(chunk_file)
                chunk_num += 1
                print(f"  チャンク {chunk_num} 完了（最終チャンク）")
        
        print(f"合計 {len(chunk_files)} チャンクを作成しました")
        
        # ステップ2: K-wayマージを使用して全チャンクをマージ
        print("ステップ2: チャンクをマージ中...")
        self._merge_sorted_chunks(chunk_files, output_path)
        
        # チャンクファイルを削除
        print("一時チャンクファイルを削除中...")
        for chunk_file in chunk_files:
            chunk_file.unlink()
    
    def _merge_sorted_chunks(self, chunk_files: list[Path], output_path: str | Path) -> None:
        """ソート済みのチャンクファイルをマージして最終出力を生成
        
        Args:
            chunk_files: ソート済みチャンクファイルのリスト
            output_path: 最終出力ファイル
        """
        import heapq
        
        # 各チャンクファイルを開く
        file_handles = []
        heap = []
        
        for i, chunk_file in enumerate(chunk_files):
            f = open(chunk_file, "r", encoding="utf-8")
            file_handles.append(f)
            
            # 最初の行を読み込んでヒープに追加
            line = f.readline()
            if line.strip():
                item = json.loads(line)
                # (-similarity, file_index, item) の形式でヒープに追加（降順にするため負の値）
                heapq.heappush(heap, (-item["s"], i, item))  # "similarity" -> "s"
        
        # マージしてJSON Lines形式で出力
        merged_count = 0
        with open(output_path, "w", encoding="utf-8") as out_f:
            while heap:
                neg_similarity, file_idx, item = heapq.heappop(heap)
                
                # JSON Lines形式で書き込み（コンパクト形式）
                out_f.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')) + "\n")
                merged_count += 1
                
                if merged_count % 100000 == 0:
                    print(f"  マージ済み: {merged_count} ペア")
                
                # 次の行を読み込んでヒープに追加
                line = file_handles[file_idx].readline()
                if line.strip():
                    next_item = json.loads(line)
                    heapq.heappush(heap, (-next_item["s"], file_idx, next_item))  # "similarity" -> "s"
        
        # ファイルを閉じる
        for f in file_handles:
            f.close()
        
        print(f"マージ完了: 合計 {merged_count} ペア")

    def analyze_rare_attributes(self) -> dict[str, Any]:
        """レアな属性（1-2回しか登場しない）の分析を行う

        Returns:
            レアな属性の統計情報
        """
        rare_tails = [tail for tail, count in self.tail_counts.items() if count <= 2]
        
        analysis = {
            "total_unique_tails": len(self.tail_counts),
            "rare_tails_count": len(rare_tails),
            "rare_tails_percentage": len(rare_tails) / len(self.tail_counts) * 100,
            "rare_tails_examples": rare_tails[:20],  # 最初の20個を例として
        }
        
        return analysis

    def process_pipeline(
        self,
        kg_path: str | Path,
        embeddings_output_path: str | Path,
        similarities_output_path: str | Path,
        similarity_threshold: float = 0.8,
        batch_size: int = 32,
        chunk_size: int = 1000,
    ) -> None:
        """パイプライン全体を実行する

        Args:
            kg_path: 知識グラフJSONファイルのパス
            embeddings_output_path: 埋め込み保存先パス
            similarities_output_path: 類似ペア保存先パス
            similarity_threshold: 類似度の閾値
            batch_size: 埋め込み生成時のバッチサイズ
            chunk_size: 類似度計算時のチャンクサイズ
        """
        # 1. データ読み込み
        self.load_kg_data(kg_path)
        
        # 2. tailのカウント
        self.count_tails()
        
        # 3. 埋め込み生成
        self.generate_embeddings(batch_size=batch_size)
        
        # 4. 埋め込み保存
        self.save_embeddings(embeddings_output_path)
        
        # 5. 類似度計算と保存
        self.compute_similarities(
            threshold=similarity_threshold,
            output_path=similarities_output_path,
            chunk_size=chunk_size,
        )
        
        # 6. レアな属性の分析
        analysis = self.analyze_rare_attributes()
        print("\n=== レアな属性の分析 ===")
        print(f"ユニークなtail総数: {analysis['total_unique_tails']}")
        print(f"レアなtail数（1-2回出現）: {analysis['rare_tails_count']}")
        print(f"レアなtailの割合: {analysis['rare_tails_percentage']:.2f}%")
        print(f"\nレアなtailの例（最初の20個）:")
        for tail in analysis["rare_tails_examples"]:
            print(f"  - {tail} (出現回数: {self.tail_counts[tail]})")

