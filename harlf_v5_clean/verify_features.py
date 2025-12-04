import pandas as pd
import numpy as np
import ta

def verify_features():
    print("Loading data...")
    price_data = pd.read_csv('data/price_data.csv', index_col=0, parse_dates=True)
    technical_features = pd.read_csv('data/technical_features.csv', index_col=0, parse_dates=True)
    
    print(f"Price data shape: {price_data.shape}")
    print(f"Technical features shape: {technical_features.shape}")
    
    # Assume first asset is NVDA
    asset_name = price_data.columns[0]
    print(f"Checking features for asset: {asset_name}")
    
    prices = price_data[asset_name]
    
    # Calculate common indicators
    print("Calculating indicators...")
    indicators = {}
    
    # RSI 14
    indicators['RSI_14'] = ta.momentum.RSIIndicator(prices, window=14).rsi()
    
    # SMA 20
    # indicators['SMA_20'] = ta.trend.SMAIndicator(prices, window=20).sma_indicator()
    
    # EMA 20
    # indicators['EMA_20'] = ta.trend.EMAIndicator(prices, window=20).ema_indicator()
    
    # MACD
    # macd = ta.trend.MACD(prices)
    # indicators['MACD'] = macd.macd()
    # indicators['MACD_Signal'] = macd.macd_signal()
    # indicators['MACD_Diff'] = macd.macd_diff()
    
    # Bollinger Bands
    # bb = ta.volatility.BollingerBands(prices, window=20, window_dev=2)
    # indicators['BB_High'] = bb.bollinger_hband()
    # indicators['BB_Low'] = bb.bollinger_lband()
    
    # Align dates
    common_dates = technical_features.index.intersection(prices.index)
    print(f"Common dates: {len(common_dates)}")
    
    if len(common_dates) == 0:
        print("No common dates found!")
        return

    tech_subset = technical_features.loc[common_dates]
    
    # Check correlations
    print("\nChecking correlations...")
    
    for name, series in indicators.items():
        series_subset = series.loc[common_dates]
        
        best_corr = 0
        best_col = -1
        best_lag = 0 # 0: aligned, 1: feature is lagged (t matches t-1), -1: feature is lookahead (t matches t+1)
        
        # Check against all columns in technical_features
        for i in range(technical_features.shape[1]):
            col_data = tech_subset.iloc[:, i]
            
            # Check alignment at t (Feature[t] vs Indicator[t])
            corr_t = series_subset.corr(col_data)
            
            # Check alignment at t-1 (Feature[t] vs Indicator[t-1]) -> Feature is lagged
            corr_lag = series_subset.shift(1).corr(col_data)
            
            # Check alignment at t+1 (Feature[t] vs Indicator[t+1]) -> Feature is lookahead
            corr_lead = series_subset.shift(-1).corr(col_data)
            
            if abs(corr_t) > abs(best_corr):
                best_corr = corr_t
                best_col = i
                best_lag = 0
            
            if abs(corr_lag) > abs(best_corr):
                best_corr = corr_lag
                best_col = i
                best_lag = 1
                
            if abs(corr_lead) > abs(best_corr):
                best_corr = corr_lead
                best_col = i
                best_lag = -1
        
        print(f"Best match for {name}:")
        print(f"  Column: {best_col}")
        print(f"  Correlation: {best_corr:.4f}")
        print(f"  Lag status: {best_lag} (0=Aligned, 1=Lagged, -1=Lookahead)")
        
        if abs(best_corr) > 0.9:
            print(f"Found match for {name}:")
            print(f"  Column: {best_col}")
            print(f"  Correlation: {best_corr:.4f}")
            print(f"  Lag status: {best_lag} (0=Aligned, 1=Lagged/Safe, -1=Lookahead)")
            
            # Verify exact values if correlation is very high
            if abs(best_corr) > 0.99:
                col_vals = tech_subset.iloc[:, best_col]
                
                if best_lag == 0:
                    ind_vals = series_subset
                    print("  -> Feature[t] matches Indicator[t] (calculated from prices up to t)")
                elif best_lag == 1:
                    ind_vals = series_subset.shift(1)
                    print("  -> Feature[t] matches Indicator[t-1] (calculated from prices up to t-1)")
                else:
                    ind_vals = series_subset.shift(-1)
                    print("  -> Feature[t] matches Indicator[t+1] (LOOKAHEAD BIAS!)")
                
                # Check diff
                diff = (col_vals - ind_vals).abs().mean()
                print(f"  Mean Absolute Difference: {diff:.6f}")

if __name__ == "__main__":
    verify_features()
