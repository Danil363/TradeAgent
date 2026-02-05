import yfinance as yf
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def load_data(ticker):
    df = yf.download(ticker, period="2y", interval="1h")
    return df