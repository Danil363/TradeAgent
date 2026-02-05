import pandas as pd
import numpy as np

from env.env import TradingEnvWithPrediction


def load_data(path):
    data = pd.read_csv(path)

    prices = data["Close"].to_numpy()
    preds = {
        "t1": data["predict_t1"].to_numpy(),
        "t3": data["predict_t3"].to_numpy(),
        "t8": data["predict_t8"].to_numpy(),
    }

    return prices, preds


def make_envs(csv_path, window=60, test_days=30):
    prices, preds = load_data(csv_path)

    test_size = test_days * 24  

    train_env = TradingEnvWithPrediction(
        prices=prices[300:-test_size],
        predictions={
            k: v[300:-test_size]
            for k, v in preds.items()
        },
        window=window,
        fee=0.000
    )

    test_env = TradingEnvWithPrediction(
        prices=prices[-test_size:],
        predictions={
            k: v[-test_size:]
            for k, v in preds.items()
        },
        window=window,
        fee=0.001
    )

    return train_env, test_env
