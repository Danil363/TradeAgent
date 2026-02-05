import numpy as np
from stable_baselines3 import DQN
from env.env import TradingEnvWithPrediction


class RLDecisionMaker:
    """
    Класс для загрузки RL модели и выполнения торговых решений.
    """

    def __init__(self, model_path, window=60):
        self.window = window

        # Загружаем RL модель
        self.model = DQN.load(model_path)

        print(f"⚡ RL модель загружена: {model_path}")

    def prepare_observation(self, prices_window, pred_t1, pred_t3, pred_t8):
        """
        Приводит входные данные к формату observation из среды.
        Обычно это:
        [нормализованные цены, предсказания моделей, позиция и т.п.]

        Здесь просто объединяем всё в одно наблюдение.
        """

        obs = np.array([
            *prices_window,
            pred_t1,
            pred_t3,
            pred_t8
        ], dtype=np.float32)

        return obs

    def decide(self, observation):
        """
        Возвращает действие RL модели:
        0 — Hold
        1 — Buy
        2 — Sell
        """

        action, _ = self.model.predict(observation, deterministic=True)

        return int(action)

    def readable_action(self, action: int):
        return {0: "HOLD", 1: "BUY", 2: "SELL"}[action]


# =============================================================
#   Удобная внешняя функция для использования в программе
# =============================================================

def make_trading_decision(prices_window, pred_t1, pred_t3, pred_t8,
                           model_path="models/AAPL/dqn_trading_model"):
    """
    Функция для интеграции в любую программу.
    На вход:
        prices_window — последние N цен (len = window)
        pred_t1, pred_t3, pred_t8 — предсказания моделей

    Возвращает:
        action (0/1/2)
        info dict (удобно для UI)
    """

    dm = RLDecisionMaker(model_path=model_path)

    obs = dm.prepare_observation(
        prices_window=prices_window,
        pred_t1=pred_t1,
        pred_t3=pred_t3,
        pred_t8=pred_t8
    )

    action = dm.decide(obs)

    info = {
        "action": action,
        "action_readable": dm.readable_action(action),
        "t1_prediction": float(pred_t1),
        "t3_prediction": float(pred_t3),
        "t8_prediction": float(pred_t8),
        "last_price": float(prices_window[-1]),
    }

    return action, info
