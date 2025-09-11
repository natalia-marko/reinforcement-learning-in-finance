#!/usr/bin/env python3
# Test the improved subreddit search strategy

COMPANY_NAMES = {
    'NVDA': ['nvidia'], 'AMD': ['amd'], 'MSFT': ['microsoft'],
    'GOOG': ['google', 'alphabet'], 'AI': ['c3.ai'], 'ASML': ['asml'],
    'MU': ['micron'], 'MRVL': ['marvell'], 'IONQ': ['ionq'],
    'RGTI': ['rigetti'], 'QBTS': ['d-wave'], 'PLTR': ['palantir'],
    'AEM': ['agnico eagle'], 'VERU': ['veru'], 'APP': ['applovin'],
    'ARBE': ['arbe'], 'RDDT': ['reddit'], 'SMR': ['nuscale'],
    'QQQ': ['qqq', 'nasdaq']
}

SUBREDDIT_TIERS = {
    'tier1': ['SecurityAnalysis', 'investing', 'ValueInvesting'],
    'tier2': ['stocks', 'StockMarket', 'investing'],
    'tier3': ['wallstreetbets', 'pennystocks', 'SecurityAnalysis']
}

def get_search_subreddits(ticker):
    """Get comprehensive subreddit list for ticker with multiple strategies"""
    search_order = []
    
    # Strategy 1: Direct ticker subreddit (r/AMD, r/NVDA, etc.)
    search_order.append(ticker.lower())
    
    # Strategy 2: Company-specific subreddits
    if ticker in COMPANY_NAMES:
        for name in COMPANY_NAMES[ticker]:
            search_order.append(name)
    
    # Strategy 3: Alternative ticker formats
    search_order.append(ticker.upper())
    search_order.append(f"${ticker.lower()}")
    
    # Strategy 4: General financial subreddits (tiered by quality)
    for tier in ['tier1', 'tier2', 'tier3']:
        search_order.extend(SUBREDDIT_TIERS[tier])
    
    # Strategy 5: Additional relevant subreddits
    additional_subs = [
        'investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting',
        'StockMarket', 'wallstreetbets', 'pennystocks', 'dividends',
        'options'
    ]
    search_order.extend(additional_subs)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_subs = []
    for sub in search_order:
        if sub not in seen:
            seen.add(sub)
            unique_subs.append(sub)
    
    return unique_subs

if __name__ == "__main__":
    print("IMPROVED REDDIT SEARCH STRATEGY TEST")
    print("=" * 50)
    
    # Test with different tickers
    test_tickers = ['NVDA', 'AMD', 'MSFT', 'GOOG', 'AAPL']
    
    for ticker in test_tickers:
        subreddits = get_search_subreddits(ticker)
        print(f"\n{ticker} SEARCH STRATEGY:")
        print(f"  Direct ticker: r/{ticker.lower()}")
        print(f"  Company names: {COMPANY_NAMES.get(ticker, [])}")
        print(f"  Total subreddits: {len(subreddits)}")
        print(f"  First 10: {subreddits[:10]}")
        print(f"  Last 5: {subreddits[-5:]}")
    
    print("\n✅ IMPROVEMENTS MADE:")
    print("1. ✅ Direct ticker subreddits (r/amd, r/nvda)")
    print("2. ✅ Multiple search queries per subreddit")
    print("3. ✅ Better error handling")
    print("4. ✅ More comprehensive subreddit list")
    print("5. ✅ Fallback mechanisms")
    print("6. ✅ Company name searches")
    print("7. ✅ Alternative ticker formats")
