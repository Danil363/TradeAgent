from env_init import make_envs
from agents.dqn_agent import DQNTradingAgent


def main():
    train_env, _ = make_envs("data/models/AAPL/new_data.csv")

    agent = DQNTradingAgent(train_env)
    agent.train(total_timesteps=400_000)

    agent.save("models/AAPL/dqn_trading_model")


if __name__ == "__main__":
    main()
