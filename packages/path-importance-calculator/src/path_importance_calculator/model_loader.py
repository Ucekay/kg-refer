"""KGATモデルをロードするためのモジュール"""

import logging
from pathlib import Path as FilePath

import torch

from kgat.config import KGATConfig
from kgat.core.kgat import KGAT
from kgat.data.dataloader import DataLoader


class KGATModelLoader:
    """学習済みKGATモデルをロードするクラス"""

    def __init__(self, model_path: str, config: KGATConfig, logger: logging.Logger):
        """
        Args:
            model_path: 学習済みモデルのパス
            config: KGATの設定
            logger: ロガー
        """
        self.model_path = FilePath(model_path)
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self) -> tuple[KGAT, DataLoader]:
        """
        学習済みKGATモデルとデータローダーをロードする

        Returns:
            tuple: (モデル, データローダー)
        """
        self.logger.info(f"Loading data from {self.config.data_dir}...")
        data_loader = DataLoader(self.config, self.logger)

        self.logger.info("Initializing KGAT model...")
        model = KGAT(
            config=self.config,
            n_users=data_loader.n_users,
            n_entities=data_loader.n_entities,
            n_relations=data_loader.n_relations,
            A_in=data_loader.A_in,
        )

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.logger.info(f"Loading model weights from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        
        # チェックポイントの形式を確認
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # 辞書形式の場合
            model.load_state_dict(checkpoint["model_state_dict"])
            self.logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # 直接state_dictの場合
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()

        self.logger.info("Model loaded successfully!")
        return model, data_loader
