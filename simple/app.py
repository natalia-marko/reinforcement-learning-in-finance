from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import os
import yfinance as yf

# Import refactored modules
from rl_system import PortfolioEnv
from train_lstm import train_single_model
from data_eng import create_features_optimized
from config import START_DATE, END_DATE

app = FastAPI(title="RL Portfolio Manager")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Static Files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class PortfolioRequest(BaseModel):
    tickers: List[str]

@app.post("/analyze")
async def analyze_portfolio(request: PortfolioRequest):
    tickers = [t.upper().strip() for t in request.tickers]
    print(f"\n{'='*60}")
    print(f"ANALYZING PORTFOLIO: {', '.join(tickers)}")
    print(f"{'='*60}")
    
    if len(tickers) < 2:
        raise HTTPException(status_code=400, detail="Please provide at least 2 tickers.")
    
    try:
        # 1. Download Data
        print("\n[1/4] Downloading price and volume data...")
        prices_data = yf.download(
            tickers, 
            start=START_DATE, 
            end=END_DATE, 
            interval='1wk', 
            auto_adjust=False
        )
        
        if prices_data.empty:
            raise HTTPException(status_code=400, detail="No data found for these tickers.")
        
        prices = prices_data['Adj Close'].copy()
        volumes = prices_data['Volume'].copy()
        
        print(f"Downloaded {len(prices)} weeks of data")
        
        # 2. Create Features
        print("\n[2/4] Engineering features...")
        features = create_features_optimized(prices, volumes, tickers)
        
        if features.empty or len(features) < 50:
            raise HTTPException(status_code=400, detail="Insufficient data after feature engineering.")
        
        print(f"Created {features.shape[1]} features from {features.shape[0]} samples")
        
        # 3. Train Agent (quick training for web demo)
        print("\n[3/4] Training RL agent...")
        print("(Using 20k timesteps for speed - production would use 100k+)")
        
        model = train_single_model(
            df_features=features,
            df_prices=prices,
            tickers=tickers,
            total_timesteps=20000
        )
        
        print("Training complete!")
        
        # 4. Get Current Recommendation
        print("\n[4/4] Generating portfolio recommendation...")
        
        # Create environment for inference
        env = PortfolioEnv(features, prices_df=prices, tickers=tickers)
        obs, _ = env.reset()
        
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Convert to weights using environment's softmax
        weights = env._softmax_with_constraints(action)
        
        # Format result
        result = {t: float(w) for t, w in zip(tickers, weights)}
        result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
        
        print("\n✅ RECOMMENDATION:")
        for ticker, weight in result.items():
            print(f"  {ticker}: {weight*100:.1f}%")
        print(f"{'='*60}\n")
        
        return {"weights": result}
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {
        "message": "RL Portfolio API is running",
        "instructions": "Go to /static/index.html to use the app",
        "model": "LSTM-PPO with Regime Detection"
    }
