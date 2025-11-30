import yfinance as yf
import pandas as pd
import numpy as np

class FinancialFeatureEngineer:
    def preprocess_data(self, df):
        data = df.copy()
        asset_cols = [c for c in data.columns if c != 'SPY']
        
        log_ret = np.log(data[asset_cols] / data[asset_cols].shift(1)).fillna(0)
        
        # Volatility
        roll_std = log_ret.rolling(20).std()
        norm_vol = (roll_std - roll_std.rolling(252).mean()) / (roll_std.rolling(252).std() + 1e-8)
        
        # RSI
        delta = data[asset_cols].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        norm_rsi = (100 - (100 / (1 + rs)) - 50) / 50 
        
        # Trend
        sma_50 = data[asset_cols].rolling(50).mean()
        dist_sma = (data[asset_cols] / sma_50) - 1.0

        # Relative Strength
        asset_cum = data[asset_cols].pct_change(60)
        bench_cum = data['SPY'].pct_change(60)
        rel_str = asset_cum.sub(bench_cum, axis=0)
        
        # Beta
        market_ret = log_ret.mean(axis=1) 
        rolling_cov = log_ret.rolling(60).cov(market_ret)
        rolling_var = market_ret.rolling(60).var()
        beta = (rolling_cov.div(rolling_var, axis=0)).fillna(1.0)

        features = np.stack([
            norm_vol.fillna(0).clip(-3, 3).values, 
            norm_rsi.fillna(0).clip(-1, 1).values, 
            dist_sma.fillna(0).clip(-0.5, 0.5).values, 
            rel_str.fillna(0).clip(-0.5, 0.5).values,
            beta.fillna(1.0).clip(-2, 2).values, 
            log_ret.values
        ], axis=-1)
        
        return features.astype(np.float32), data[asset_cols].values.astype(np.float32), data['SPY'].values.astype(np.float32), asset_cols

def get_data():
    tickers = ['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG', 'SPY']
    print(f"📥 Downloading {len(tickers)} assets...")
    try:
        # Explicitly set auto_adjust=False to ensure 'Adj Close' is returned
        # group_by='column' ensures we can select 'Adj Close' easily
        raw_data = yf.download(tickers, start="2015-01-01", end="2024-01-01", auto_adjust=False, group_by='column')
        
        if 'Adj Close' in raw_data.columns:
             data = raw_data['Adj Close']
        elif isinstance(raw_data.columns, pd.MultiIndex):
             # Fallback: try to find Adj Close in levels or just take Close if Adj Close missing
             data = raw_data.xs('Adj Close', level=0, axis=1, drop_level=True)
        else:
             # If flat and no Adj Close, maybe just Close?
             data = raw_data['Close']
             
        if data.empty: raise ValueError('Data is empty')
        
        # Ensure we have a flat index (Tickers as columns)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(-1)
            
        return data.dropna()
    except Exception as e:
        print(f"⚠️ Data Download Failed or Empty. Check API. Error: {e}")
        return pd.DataFrame()

print("Running test...")
df = get_data()
print("Data columns:", df.columns)
print("Data shape:", df.shape)
if df.empty:
    print("Test failed: Data is empty")
else:
    engineer = FinancialFeatureEngineer()
    try:
        features, prices, benchmark, ticker_names = engineer.preprocess_data(df)
        print("Success! Features shape:", features.shape)
    except Exception as e:
        print(f"Test failed with error: {e}")
