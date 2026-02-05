import torch
import pandas as pd
import numpy as np
import warnings
from training.train_pipeline import train_pipeline
import joblib 
import os


warnings.filterwarnings('ignore')

def create_sequences(X, window):
    X_seq = []
    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
    return np.array(X_seq)

def make_predictions(df, model, scaler_x, scaler_y, window, col_name, device):
    model.eval()

    features = ["Close"]
    X_raw = df[features].values

    X_all_scaled = scaler_x.transform(X_raw)

    X_seq = []
    for i in range(len(X_all_scaled) - window):
        X_seq.append(X_all_scaled[i:i+window])
    X_seq = np.array(X_seq)

    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy().flatten()

    y_pred_real = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    pred_column = np.full(len(df), np.nan)
    start_index = window
    pred_column[start_index:start_index + len(y_pred_real)] = y_pred_real

    df[col_name] = pred_column
    return df



def create_models(df: pd.DataFrame, shifts=(1, 3, 8), path_dir=None, use_atention=False):
    save_dir = 'data/models'
    df_new = df.copy()
    os.makedirs(save_dir, exist_ok=True)
    if path_dir:
        save_dir = f'data/models/{path_dir}'
        os.makedirs(save_dir, exist_ok=True)


    for shift in shifts:
        path = f'predict_t{shift}'
        os.makedirs(os.path.join(save_dir, path), exist_ok=True)

        model, scaler_x, scaler_y, optimizer = train_pipeline(df, 60, shift, 60, use_atention)

        model_path = os.path.join(save_dir, path, "model.pth")
        scaler_x_path = os.path.join(save_dir, path, "scaler_x.pkl")
        scaler_y_path = os.path.join(save_dir, path, "scaler_y.pkl")

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, model_path)

        joblib.dump(scaler_x, scaler_x_path)
        joblib.dump(scaler_y, scaler_y_path)

        df_new = make_predictions(df_new, model, scaler_x, scaler_y, 60, path, 'cuda')

        print(f"✅ Модель t={shift} сохранена и предсказания добавлены")

    df_new.to_csv(os.path.join(save_dir, 'new_data.csv'), index=False)
    print("✅ Все модели обучены и результаты сохранены в new_data.csv")
