from logging import getLogger
from pathlib import Path

import numpy as np
from cyclopts import App
from recbole.config.configurator import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed

from .config import LightKGConfig

np.float_ = np.float64
np.complex_ = np.complex128
np.unicode_ = np.str_

app = App()


@app.default
def main(config: LightKGConfig = LightKGConfig()):
    # Get the directory where this script is located
    current_dir = Path(__file__).parent.parent
    config_file_path = current_dir / "config_file" / f"{config.dataset}.yaml"

    config_file_list = [str(config_file_path)]

    recbole_config = Config(
        model=LightKG,
        dataset=config.dataset,
        config_file_list=config_file_list,
        config_dict={"seed": config.seed},
    )
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

    test_restult = trainer.evaluate(test_data)

    logger.info(f"Best valid result: {best_valid_result}")
    logger.info(f"Test result: {test_restult}")


if __name__ == "__main__":
    app()
