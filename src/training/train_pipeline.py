import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Поднимаемся на уровень выше
from training.trainer import train
from training.evaluator import evaluate
import torch
import numpy as np
from data.preprocess import prepare_data
from models.lstm_model import LSTMModel
from models.cnn_model import CNN_BiLSTM_Attention
import torch.nn as nn

def train_pipeline(df, seq_len, shift, epochs, use_atention = False):
    train_loader, val_loader, X_test, y_test_sc, scaler_x, scaler_y,df = prepare_data(df, seq_len, shift)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = LSTMModel(input_size=X_test.shape[2], hidden_size=64)

    if use_atention:
        model = CNN_BiLSTM_Attention(input_size=X_test.shape[2], hidden_size=64).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model = model.to(device)

    train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs)
    preds, y_real = evaluate(model, X_test, y_test_sc, scaler_y, device)

    return model, scaler_x, scaler_y, optimizer
