from cyclopts import App

from kgat.training.predict import predict
from kgat.training.train import train

from .config import KGATConfig

app = App()


@app.default
def main(config: KGATConfig = KGATConfig()):
    if config.train:
        print("Training mode activated.")
        train(config)
    if config.predict:
        print("Prediction mode activated.")
        predict(config)


if __name__ == "__main__":
    app()
