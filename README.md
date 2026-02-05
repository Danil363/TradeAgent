# TradeAgent

**TradeAgent** is an intelligent trading system that combines
**time series price prediction** with an **autonomous trading agent**
to support decision-making on financial markets.

The system first predicts asset prices for the next few hours using an
**LSTM neural network**, and then passes this forecast to an agent that
decides whether to buy, sell, or hold.

---

## Project Overview

TradeAgent follows a two-stage pipeline:

1. **Price Prediction**
   - Historical market data is processed
   - An LSTM neural network predicts prices for the upcoming hours

2. **Decision-Making Agent**
   - The predicted prices are used as input features
   - The agent analyzes trends, signals, and risk
   - A trading decision is produced

This architecture separates **forecasting** and **decision logic**,
making the system modular and extensible.

---

## 🧠 System Architecture

