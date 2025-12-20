from logging import getLogger
from pathlib import Path

import numpy as np
from cyclopts import App
from recbole.config.configurator import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import KGTrainer
from recbole.utils import init_logger, init_seed

from light_kg.light_kg import LightKG

from .config import LightKGConfig

np.float_ = np.float64
np.complex_ = np.complex128
np.unicode_ = np.str_

app = App()


@app.default
def main(config: LightKGConfig = LightKGConfig()):
    # Get the directory where this script is located
    # __file__ is in src/light_kg/cli.py, so parent.parent is src, and parent.parent.parent is packages/light-kg
    current_dir = Path(__file__).parent.parent.parent
    config_file_path = current_dir / "src" / "config_file" / f"{config.dataset}.yaml"
    
    # Set custom dataset path to prevent automatic download
    # RecBole expects data_path to point to the parent directory (datasets),
    # and it will automatically append the dataset name (yelp) to create datasets/yelp
    datasets_dir = current_dir / "datasets"
    data_path = str(datasets_dir)

    config_file_list = [str(config_file_path)]

    recbole_config = Config(
        model=LightKG,
        dataset=config.dataset,
        config_file_list=config_file_list,
        config_dict={
            "seed": config.seed,
            "data_path": data_path,
            "fix_relation_weights": config.fix_relation_weights,
        },
    )
    
    # Debug: Print config values to verify they are loaded correctly
    print(f"\n=== Config File Check ===")
    print(f"Config file path: {config_file_path}")
    print(f"Config file exists: {Path(config_file_path).exists()}")
    print(f"\nKey config values:")
    print(f"  embedding_size: {recbole_config['embedding_size']}")
    print(f"  layer: {recbole_config['layer']}")
    print(f"  learning_rate: {recbole_config['learning_rate']}")
    print(f"  mess_dropout_rate: {recbole_config['mess_dropout_rate']}")
    print(f"  weight_decay: {recbole_config['weight_decay']}")
    print(f"  cos_loss: {recbole_config['cos_loss']}")
    print(f"  user_loss: {recbole_config['user_loss']}")
    print(f"  item_loss: {recbole_config['item_loss']}")
    print(f"  train_neg_sample_num: {recbole_config['train_neg_sample_num']}")
    print(f"  stopping_steps: {recbole_config['stopping_steps']}")
    print(f"  benchmark_filename: {recbole_config.get('benchmark_filename', 'NOT SET') if hasattr(recbole_config, 'get') else recbole_config.final_config_dict.get('benchmark_filename', 'NOT SET')}")
    print(f"  fix_relation_weights: {recbole_config['fix_relation_weights']} (LightGCN mode: {'ON' if recbole_config['fix_relation_weights'] else 'OFF'})")
    print(f"=========================\n")
    
    init_seed(config.seed, recbole_config["reproducibility"])
    init_logger(recbole_config)
    logger = getLogger()

    data = create_dataset(recbole_config)
    logger.info(data)
    train_data, valid_data, test_data = data_preparation(recbole_config, data)

    init_seed(200, recbole_config["reproducibility"])

    model = LightKG(config=recbole_config, dataset=train_data._dataset).to(
        recbole_config["device"]
    )

    init_seed(config.seed, recbole_config["reproducibility"])

    logger.info(model)

    trainer = KGTrainer(config=recbole_config, model=model)

    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # Evaluate using the best model already loaded in memory
    # Set load_best_model=False to avoid torch.load weights_only issue in PyTorch 2.6
    test_restult = trainer.evaluate(test_data, load_best_model=False, show_progress=False)

    logger.info(f"Best valid result: {best_valid_result}")
    logger.info(f"Test result: {test_restult}")


if __name__ == "__main__":
    app()
