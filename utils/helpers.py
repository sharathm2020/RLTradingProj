import yfinance as yf
import pandas as pd

def load_stock_data(ticker="AAPL", start="2022-01-01", end="2023-01-01"):
    df = yf.download(ticker, start=start, end=end)

    # Simple moving averages with pandas
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    df.dropna(inplace=True)
    return df
