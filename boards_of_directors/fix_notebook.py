import json

file_path = '/Users/nataliamarko/Documents/GitHub_Projects/reinforcement_learning_in_finance/boards_of_directors/optimized_alpha_hunter.ipynb'

with open(file_path, 'r') as f:
    nb = json.load(f)

# Find the cell with get_data
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Check if this cell defines get_data
        # We join source to check, as it is a list of strings
        source_text = "".join(source)
        if 'def get_data():' in source_text and 'class FinancialFeatureEngineer:' in source_text:
            # This is the target cell
            
            new_full_source = [
                "class FinancialFeatureEngineer:\n",
                "    def preprocess_data(self, df):\n",
                "        data = df.copy()\n",
                "        asset_cols = [c for c in data.columns if c != 'SPY']\n",
                "        \n",
                "        log_ret = np.log(data[asset_cols] / data[asset_cols].shift(1)).fillna(0)\n",
                "        \n",
                "        # Volatility\n",
                "        roll_std = log_ret.rolling(20).std()\n",
                "        norm_vol = (roll_std - roll_std.rolling(252).mean()) / (roll_std.rolling(252).std() + 1e-8)\n",
                "        \n",
                "        # RSI\n",
                "        delta = data[asset_cols].diff()\n",
                "        gain = (delta.where(delta > 0, 0)).rolling(14).mean()\n",
                "        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()\n",
                "        rs = gain / (loss + 1e-8)\n",
                "        norm_rsi = (100 - (100 / (1 + rs)) - 50) / 50 \n",
                "        \n",
                "        # Trend\n",
                "        sma_50 = data[asset_cols].rolling(50).mean()\n",
                "        dist_sma = (data[asset_cols] / sma_50) - 1.0\n",
                "\n",
                "        # Relative Strength\n",
                "        asset_cum = data[asset_cols].pct_change(60)\n",
                "        bench_cum = data['SPY'].pct_change(60)\n",
                "        rel_str = asset_cum.sub(bench_cum, axis=0)\n",
                "        \n",
                "        # Beta\n",
                "        market_ret = log_ret.mean(axis=1) \n",
                "        rolling_cov = log_ret.rolling(60).cov(market_ret)\n",
                "        rolling_var = market_ret.rolling(60).var()\n",
                "        beta = (rolling_cov.div(rolling_var, axis=0)).fillna(1.0)\n",
                "\n",
                "        features = np.stack([\n",
                "            norm_vol.fillna(0).clip(-3, 3).values, \n",
                "            norm_rsi.fillna(0).clip(-1, 1).values, \n",
                "            dist_sma.fillna(0).clip(-0.5, 0.5).values, \n",
                "            rel_str.fillna(0).clip(-0.5, 0.5).values,\n",
                "            beta.fillna(1.0).clip(-2, 2).values, \n",
                "            log_ret.values\n",
                "        ], axis=-1)\n",
                "        \n",
                "        return features.astype(np.float32), data[asset_cols].values.astype(np.float32), data['SPY'].values.astype(np.float32), asset_cols\n",
                "\n",
                "def get_data():\n",
                "    tickers = ['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG', 'SPY']\n",
                "    print(f\"📥 Downloading {len(tickers)} assets...\")\n",
                "    try:\n",
                "        # Explicitly set auto_adjust=False to ensure 'Adj Close' is returned\n",
                "        # group_by='column' ensures we can select 'Adj Close' easily\n",
                "        raw_data = yf.download(tickers, start=\"2015-01-01\", end=\"2024-01-01\", auto_adjust=False, group_by='column')\n",
                "        \n",
                "        if 'Adj Close' in raw_data.columns:\n",
                "             data = raw_data['Adj Close']\n",
                "        elif isinstance(raw_data.columns, pd.MultiIndex):\n",
                "             # Fallback: try to find Adj Close in levels or just take Close if Adj Close missing\n",
                "             # But with auto_adjust=False, Adj Close should be there.\n",
                "             # If the user meant MultiIndex issues, it might be that we need to drop levels.\n",
                "             data = raw_data.xs('Adj Close', level=0, axis=1, drop_level=True)\n",
                "        else:\n",
                "             # If flat and no Adj Close, maybe just Close?\n",
                "             data = raw_data['Close']\n",
                "             \n",
                "        if data.empty: raise ValueError('Data is empty')\n",
                "        \n",
                "        # Ensure we have a flat index (Tickers as columns)\n",
                "        if isinstance(data.columns, pd.MultiIndex):\n",
                "            data.columns = data.columns.get_level_values(-1)\n",
                "            \n",
                "        return data.dropna()\n",
                "    except Exception as e:\n",
                "        print(f\"⚠️ Data Download Failed or Empty. Check API. Error: {e}\")\n",
                "        return pd.DataFrame()\n",
                "\n",
                "df = get_data()\n",
                "engineer = FinancialFeatureEngineer()\n",
                "features, prices, benchmark, ticker_names = engineer.preprocess_data(df)\n",
                "\n",
                "split = int(len(df) * 0.8)\n",
                "feat_train, feat_test = features[:split], features[split:]\n",
                "price_train, price_test = prices[:split], prices[split:]\n",
                "bench_train, bench_test = benchmark[:split], benchmark[split:]"
            ]
            
            cell['source'] = new_full_source
            found = True
            break

if found:
    with open(file_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print("Successfully updated notebook.")
else:
    print("Could not find the cell to update.")
