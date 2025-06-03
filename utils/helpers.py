import yfinance as yf
import pandas as pd

def load_stock_data(ticker="AAPL", start="2022-01-01", end="2023-01-01"):
    print(f"[INFO] Attempting to download data for {ticker} from {start} to {end}")

    try:
        df = yf.download(ticker, start=start, end=end, threads=False)
        print(f"[INFO] Downloaded {len(df)} rows.")

        if df.empty:
            print(f"[WARNING] No data found for {ticker}.")
            return pd.DataFrame()

        print(f"[INFO] Data after cleaning has {len(df)} rows.")
        return df

    except Exception as e:
        print(f"[ERROR] Failed to download data for {ticker}: {e}")
        return pd.DataFrame()
