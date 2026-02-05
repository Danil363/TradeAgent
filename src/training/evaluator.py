import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(model, X, y_true_scaled, scaler_y, device):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(X).cpu().numpy().flatten()

    # вернуть масштаб
    preds_real = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
    y_real = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_real, preds_real)
    rmse = mean_squared_error(y_real, preds_real) ** 0.5
    mape = np.mean(np.abs((y_real - preds_real) / y_real)) * 100
    r2 = r2_score(y_real, preds_real)

    print("\n=== METRICS ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²:   {r2:.4f}")

    return preds_real, y_real