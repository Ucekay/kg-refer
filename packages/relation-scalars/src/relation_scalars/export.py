import numpy as np

np.float_ = np.float64
np.complex_ = np.complex128
np.unicode_ = np.str_

import torch
from pathlib import Path
from recbole.config.configurator import Config
from recbole.data import create_dataset
from light_kg.light_kg import LightKG


def export_relation_scalars(
    ckpt_path: str | Path,
    config_path: str | Path,
    data_path: str | Path,
    dataset_name: str,
    output_path: str | Path,
):
    """
    学習済みLightKGモデルから各リレーションのスカラーを抽出してCSVに保存します。
    
    Args:
        ckpt_path: 学習済みモデルのチェックポイントパス (.pth)
        config_path: データセット設定ファイルパス (.yaml)
        data_path: データセットのルートディレクトリ
        dataset_name: データセット名
        output_path: 出力CSVファイルパス
    """
    ckpt_path = Path(ckpt_path)
    config_path = Path(config_path)
    data_path = Path(data_path)
    output_path = Path(output_path)
    
    # 1) リレーションID→名前のマッピング取得
    cfg = Config(
        model=LightKG,
        dataset=dataset_name,
        config_file_list=[str(config_path)],
        config_dict={"data_path": str(data_path)},
    )
    dataset = create_dataset(cfg)
    id2rel = dataset.field2id_token["relation_id"]

    # 2) チェックポイントから relation_embedding をロード
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    rel_scalars = state["relation_embedding.weight"].squeeze(-1)

    # 3) CSVに書き出し
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def rel_name(idx: int) -> str:
        R = len(id2rel)  # PAD含むrelation語彙数
        
        # 0-(R-1): 元のKGリレーション（PAD含む）
        if idx < R:
            return id2rel[idx]
        
        # R-(2R-3): 逆向きリレーション
        elif idx < (R-2)*2+1:
            base = idx - (R - 2)
            if base < R:
                return f"{id2rel[base]}_rev"
            return f"rel_{idx}_rev"
        
        # (R-2)*2+1: user→item
        elif idx == (R-2)*2+1:
            return "user_item"
        
        # (R-2)*2+2: item→user
        elif idx == (R-2)*2+2:
            return "item_user"
        
        return f"rel_{idx}"

    with output_path.open("w", encoding="utf-8") as f:
        f.write("relation_index,relation_name,scalar\n")
        for i, val in enumerate(rel_scalars.tolist()):
            f.write(f"{i},{rel_name(i)},{val}\n")

    print(f"saved: {output_path}")
    return output_path


def main():
    """CLI エントリーポイント"""
    root = Path("/home/kimura/repos/kg-refer")
    
    export_relation_scalars(
        ckpt_path=root / "saved/LightKG-Dec-15-2025_14-43-10.pth",
        config_path=root / "packages/light-kg/src/config_file/yelp.yaml",
        data_path=root / "packages/light-kg/datasets",
        dataset_name="yelp",
        output_path=root / "output/relation_scalars.csv",
    )


if __name__ == "__main__":
    main()



