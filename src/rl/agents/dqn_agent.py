from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env


class DQNTradingAgent:
    """
    Обёртка над Stable-Baselines3 DQN,
    чтобы код был одинаковым для тренировки и теста.
    """

    def __init__(self, env, learning_rate=1e-4):
        check_env(env, warn=True)

        self.model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate,
            buffer_size=50_000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
        )

    def train(self, total_timesteps):
        print("🚀 Начинаем обучение модели...")
        self.model.learn(total_timesteps=total_timesteps)
        print("✅ Обучение завершено.")

    def save(self, path):
        self.model.save(path)

    def load(self, path, env):
        self.model = DQN.load(path, env=env)

    def predict(self, obs, deterministic=False):
        return self.model.predict(obs, deterministic=deterministic)
