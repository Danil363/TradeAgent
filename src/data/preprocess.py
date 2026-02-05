import yfinance as yf
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Поднимаемся на уровень выше
from data.dataset import TimeSeriesDataset

def normalize_yf_dataframe(df):    
    df = df.copy()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns.names = ['DataType', 'Ticker']
        df = df.stack(level=1).reset_index()
        df = df.rename(columns={'level_1': 'Ticker'})
    else:
        df = df.reset_index()
        df['Ticker'] = df.columns[1].split()[0] if ' ' in df.columns[1] else 'UNKNOWN'

    df = df.rename(columns=str.capitalize)
    cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df[[c for c in cols if c in df.columns]]

    return df

def make_sliding_window(data, feature_cols, target_col, window_size=60):
    X, y = [], []
    values = data[feature_cols].values
    targets = data[target_col].values

    if len(data) <= window_size:
        raise ValueError(f"Недостаточно данных для построения sliding window: "
                         f"len(data)={len(data)}, window_size={window_size}")

    for i in range(window_size, len(data)):
        X.append(values[i - window_size:i])
        y.append(targets[i])
    
    return np.array(X), np.array(y)


def scale(data, scaler):
    csale_d = data.reshape(-1, data.shape[2])
    csale_d = scaler.transform(csale_d)
    return csale_d.reshape(data.shape)

def prepare_data(df, seq_len, shift):
    df = normalize_yf_dataframe(df)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    target_col = f"Next_Close_t{shift}"
    df[target_col] = df["Close"].shift(-shift)
    df = df.dropna().reset_index(drop=True)

    features = ["Open", "High", "Low", "Close", "Volume"]
    features = [ "Close"]

    # ==== масштабування до sliding window ====
    X_raw = df[features].values           # (N, 5)
    y_raw = df[[target_col]].values       # (N, 1)

    scaler_x = StandardScaler().fit(X_raw)
    scaler_y = StandardScaler().fit(y_raw)

    df_scaled = df.copy()
    df_scaled[features] = scaler_x.transform(X_raw)
    df_scaled[target_col] = scaler_y.transform(y_raw)

    # ==== sliding window ====
    X, y = make_sliding_window(df_scaled, features, target_col, window_size=seq_len)

    # ==== split ====
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]

    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    # ==== torch loaders ====
    train_loader = DataLoader(
        TimeSeriesDataset(X_train, y_train),
        batch_size=32, shuffle=True
    )

    val_loader = DataLoader(
        TimeSeriesDataset(X_val, y_val),
        batch_size=32, shuffle=False
    )

    return (
        train_loader,
        val_loader,
        X_test,
        y_test,
        scaler_x,
        scaler_y,
        df_scaled
    )
