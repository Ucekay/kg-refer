import logging
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from src.kgat.config import KGATConfig
from src.kgat.core.kgat import KGAT
from src.kgat.data.dataloader import DataLoader

def main():
    # ターゲットユーザーID
    target_uid = 12719
    
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # 設定を読み込み (analyze_embeddings.pyと同じ設定)
    config = KGATConfig()
    config.data_name = "yelp"
    config.data_dir = "datasets/"
    config.use_pretrain = 0
    
    # モデル設定
    config.embed_dim = 64
    config.relation_dim = 64
    config.conv_dim_list = "[64,32,16]"
    config.aggregation_type = "bi-interaction"
    config.laplacian_type = "random-walk"
    config.mess_dropout = "[0.1,0.1,0.1]"

    # モデルパス
    model_path = "trained_model/KGAT/yelp/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain0/model_epoch900.pth"
    config.pretrain_model_path = model_path

    logger.info(f"モデルパス: {model_path}")

    # データローダーを初期化
    logger.info("データをロード中...")
    data = DataLoader(config, logger)
    
    # ターゲットユーザーの検証
    if target_uid >= data.n_users:
        logger.error(f"ユーザーID {target_uid} は無効です。最大ユーザーID: {data.n_users - 1}")
        return

    # モデルをロード
    logger.info("モデルをロード中...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # チェックポイントからサイズ情報を取得してモデルを構築
    entity_user_embed_size = checkpoint["model_state_dict"]["entity_user_embed.weight"].shape[0]
    n_relations_ckpt = checkpoint["model_state_dict"]["relation_embed.weight"].shape[0]
    
    # サイズ推定
    ratio = entity_user_embed_size / data.n_users_entities
    n_entities_ckpt = int(data.n_entities * ratio)
    n_users_ckpt = int(data.n_users * ratio)
    
    logger.info(f"モデル構築: n_users={n_users_ckpt}, n_entities={n_entities_ckpt}")
    
    model = KGAT(config, n_users_ckpt, n_entities_ckpt, n_relations_ckpt, A_in=None)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    
    # 埋め込み計算
    logger.info("全埋め込みを計算中...")
    with torch.no_grad():
        all_embeddings = model.calc_cf_embeddings()
        
        # ユーザー埋め込みの範囲を取得
        # エンティティの後にユーザーが続く
        user_embeddings = all_embeddings[n_entities_ckpt : n_entities_ckpt + n_users_ckpt]
        
        # ターゲットユーザーの埋め込み
        target_user_idx = target_uid  # user_embeddings内でのインデックスはそのままuid
        target_embedding = user_embeddings[target_user_idx]
        
        # コサイン類似度を計算
        # target_embedding: (dim,) -> (1, dim)
        # user_embeddings: (n_users, dim)
        
        # 正規化
        target_norm = torch.nn.functional.normalize(target_embedding.unsqueeze(0), p=2, dim=1)
        users_norm = torch.nn.functional.normalize(user_embeddings, p=2, dim=1)
        
        # 内積（正規化されているのでコサイン類似度になる）
        similarities = torch.mm(target_norm, users_norm.t()).squeeze(0) # (n_users,)
        
        # 自分自身を除外するために値を -1 に設定（または後でフィルタリング）
        similarities[target_user_idx] = -1.0
        
        # 上位5人を取得
        top_k = 5
        top_scores, top_indices = torch.topk(similarities, k=top_k)
        
        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        
        logger.info(f"\nユーザー {target_uid} に類似したユーザーTop 5:")
        
        results = []
        for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
            original_id = int(idx) # user_embeddingsのインデックス = オリジナルID
            logger.info(f"{rank}. User ID: {original_id}, Similarity: {score:.4f}")
            results.append({
                "rank": rank,
                "user_id": original_id,
                "similarity": score
            })
            
        # 結果をCSVに保存
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"user_{target_uid}_similar_users.csv"
        pd.DataFrame(results).to_csv(output_file, index=False)
        logger.info(f"結果を保存しました: {output_file}")

if __name__ == "__main__":
    main()


