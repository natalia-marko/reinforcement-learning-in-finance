
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def check_feature_redundancy():
    print("="*70)
    print("CHECKING FEATURE REDUNDANCY")
    print("="*70)
    
    try:
        tech_features = pd.read_csv('data/technical_features.csv', index_col=0, parse_dates=True)
        sent_features = pd.read_csv('data/sentiment_features.csv', index_col=0, parse_dates=True)
        
        print(f"Technical Features: {tech_features.shape}")
        print(f"Sentiment Features: {sent_features.shape}")
        
        # Check Technical Features Collinearity
        print("\nAnalyzing Technical Features...")
        corr_matrix = tech_features.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        
        print(f"Found {len(to_drop)} highly correlated technical features (>0.95):")
        if len(to_drop) > 0:
            print(f"Examples: {to_drop[:5]}...")
        else:
            print("None found.")
            
        # Check Sentiment Features Collinearity
        print("\nAnalyzing Sentiment Features...")
        corr_matrix_sent = sent_features.corr().abs()
        upper_sent = corr_matrix_sent.where(np.triu(np.ones(corr_matrix_sent.shape), k=1).astype(bool))
        to_drop_sent = [column for column in upper_sent.columns if any(upper_sent[column] > 0.95)]
        
        print(f"Found {len(to_drop_sent)} highly correlated sentiment features (>0.95):")
        if len(to_drop_sent) > 0:
            print(f"Examples: {to_drop_sent[:5]}...")
        else:
            print("None found.")
            
    except Exception as e:
        print(f"Error checking redundancy: {e}")

if __name__ == "__main__":
    check_feature_redundancy()
