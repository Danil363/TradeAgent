import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnvWithPrediction(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, prices, predictions=None,
                 window=24, initial_balance=10000, fee=0.001):
        """
        prices: np.array или список цен
        predictions: dict, где ключ — название прогноза, значение — массив предсказаний
                     пример: {"t1": predict_t1, "t3": predict_t3, "t8": predict_t8}
        """
        super().__init__()

        self.prices = np.array(prices, dtype=np.float32).flatten()
        self.predictions = {}

        if predictions is not None:
            for name, arr in predictions.items():
                self.predictions[name] = np.array(arr, dtype=np.float32).flatten()

        self.window = window
        self.initial_balance = initial_balance
        self.fee = fee

        self.balance = float(initial_balance)
        self.position = 0.0
        self.current_step = window
        self.equity_curve = []

        self.buy_count = 0
        self.sell_count = 0

        # --- Actions: 0 = HOLD, 1 = BUY, 2 = SELL ---
        self.action_space = spaces.Discrete(3)

        # --- Observation size ---
        obs_dim = window + 2 + len(self.predictions)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = float(self.initial_balance)
        self.position = 0.0
        self.current_step = self.window
        self.equity_curve = []
        self.buy_count = 0
        self.sell_count = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        window_data = self.prices[self.current_step - self.window:self.current_step]
        obs = [*window_data, self.balance, self.position]

        # Добавляем все прогнозы
        for arr in self.predictions.values():
            if self.current_step < len(arr):
                obs.append(arr[self.current_step])
            else:
                obs.append(np.nan)  # если вышли за пределы массива

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        price = self.prices[self.current_step]
        prev_equity = self.balance + self.position * price
        
    
        # print(f"Step {self.current_step}: Price={price:.4f}, Balance={self.balance:.2f}, "
        #   f"Position={self.position:.4f}, Equity={prev_equity:.2f}")
    
        # --- Торговая логика с проверками ---
        trade_executed = False
        
        if action == 1:  # BUY
            if self.balance > 0:
                shares = self.balance / (price * (1 + self.fee))
                self.position += shares
                self.balance -= shares * price * (1 + self.fee)
                self.buy_count += 1
                trade_executed = True
            # else: можно добавить логирование попытки покупки без средств
        
        elif action == 2:  # SELL
            if self.position > 0:
                self.balance += self.position * price * (1 - self.fee)
                self.position = 0.0
                self.sell_count += 1
                trade_executed = True
            # else: можно добавить штраф за попытку продажи без позиции
        
        # --- Награда и метрики ---
        current_equity = self.balance + self.position * price
        # print(f"After action {action}: Equity={current_equity:.2f}, Change={current_equity-prev_equity:.2f}")
        self.equity_curve.append(current_equity)
        
        reward = 0.0
        if len(self.equity_curve) > 1:
            reward = (current_equity - prev_equity) / (prev_equity + 1e-9)
        
        # Штраф за бесполезные действия
        if not trade_executed and action != 0:  # действие не выполнено и это не HOLD
            reward -= 0.001  # небольшой штраф
        
        self.current_step += 1
        terminated = self.current_step >= len(self.prices) - 1
        
        return self._get_obs(), reward, terminated, False, {}

    def get_metrics(self):
        curve = np.array(self.equity_curve, dtype=np.float32)
        if len(curve) == 0:
            return {}

        final_balance = float(curve[-1])
        total_return = (final_balance - self.initial_balance) / self.initial_balance

        returns = np.diff(curve) / curve[:-1]
        sharpe = (
            np.mean(returns) / (np.std(returns) + 1e-9)
            if len(returns) > 1 else 0.0
        )

        running_max = np.maximum.accumulate(curve)
        max_dd = float(np.min((curve - running_max) / running_max))

        return {
            "final_balance": final_balance,
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd
        }

    def get_action_stats(self):
        return {
            "buys": self.buy_count,
            "sells": self.sell_count
        }