from stable_baselines3.common.logger import configure
from env_init import make_envs
from agents.dqn_agent import DQNTradingAgent


def main():
    _, test_env = make_envs("data/models/AAPL/new_data.csv")

    agent = DQNTradingAgent(env=test_env)
    agent.load("models/AAPL/dqn_trading_model", env=test_env)

    obs, info = test_env.reset()
    done = False
    step = 0

    # Логгер
    agent.model._logger = configure("logs/", ["stdout"])
    agent.model._current_progress_remaining = 1.0

    while not done:
        action, _ = agent.predict(obs, deterministic=False)
        next_obs, reward, done, truncated, info = test_env.step(action)

        # Добавляем опыт
        agent.model.replay_buffer.add(
            obs, next_obs, action, reward, done, infos=[info]
        )

        # Онлайн-дообучение
        agent.model.train(batch_size=64, gradient_steps=1)

        obs = next_obs
        step += 1

        if step % 500 == 0:
            print(f"Онлайн-дообучение — шаг {step}")

    print("=== 📊 TEST METRICS ===")
    metrics = test_env.get_metrics()
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n📈 Действия:", test_env.get_action_stats())


if __name__ == "__main__":
    main()
