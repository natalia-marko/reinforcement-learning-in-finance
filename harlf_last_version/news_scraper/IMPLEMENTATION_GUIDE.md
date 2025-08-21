# Implementation Guide - Google News RSS

### **Why Google News First?**
- ✅ No API keys required
- ✅ RSS feeds are reliable
- ✅ Complements GDELT (recent vs. historical)
- ✅ Easy to implement

### **Implementation Steps**

#### Step 1: Install Required Package
```bash
pip install feedparser
```

#### Step 2: Update `src/sources/google_news.py`
```python
from __future__ import annotations
import feedparser
import pandas as pd
from datetime import datetime, timedelta
from src.utils import url_domain, ensure_schema, dedupe_by_url

def collect(ticker_to_name: dict[str,str], quality_domains: set[str],
            start_date: str, end_date: str) -> pd.DataFrame:
    """
    Collect recent news from Google News RSS feeds
    """
    # Google News RSS feeds for business/finance
    rss_feeds = [
        "https://news.google.com/rss/headlines/section/topic/BUSINESS",
        "https://news.google.com/rss/headlines/section/topic/SCIENCE.TECHNOLOGY",
        "https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en"
    ]
    
    rows = []
    
    for feed_url in rss_feeds:
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:50]:  # Get recent articles
                title = getattr(entry, 'title', '')
                url = getattr(entry, 'link', '')
                
                if not title or not url:
                    continue
                
                # Check for ticker mentions
                mentioned_tickers = []
                for ticker, company_name in ticker_to_name.items():
                    if (ticker.lower() in title.lower() or 
                        company_name.lower() in title.lower()):
                        mentioned_tickers.append(ticker)
                
                if mentioned_tickers:
                    # Get publication date
                    pub_date = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    
                    # Create row for each mentioned ticker
                    for ticker in mentioned_tickers:
                        domain = url_domain(url)
                        rows.append({
                            "ticker": ticker,
                            "date": pub_date.strftime("%Y-%m-%d %H:%M:%S"),
                            "title": title,
                            "url": url,
                            "domain": domain,
                            "is_quality_source": domain in quality_domains,
                            "source": "google_news",
                        })
                        
        except Exception as e:
            print(f"Error fetching {feed_url}: {e}")
            continue
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(["ticker","date"]).reset_index(drop=True)
        df = dedupe_by_url(df)
    
    return ensure_schema(df)
```

#### Step 3: Test Google News
```bash
# Run again - now google_news.csv should have data
python run.py
```

## Data Validation Checklist

### **After Each Implementation**
- [ ] Source returns DataFrame with correct schema
- [ ] No duplicate URLs
- [ ] Dates are parseable
- [ ] Ticker matching works correctly
- [ ] Quality domain filtering works
- [ ] Source column is correctly labeled

### **Schema Validation**
```python
def validate_schema(df, source_name):
    """Validate DataFrame schema"""
    expected_cols = ["ticker","date","title","url","domain","is_quality_source","source"]
    
    # Check columns
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        print(f"❌ {source_name}: Missing columns: {missing_cols}")
        return False
    
    # Check source labeling
    if not df.empty and df['source'].iloc[0] != source_name:
        print(f"❌ {source_name}: Source column incorrect")
        return False
    
    print(f"✅ {source_name}: Schema valid, {len(df)} rows")
    return True

# Test each source
for source in ['gdelt', 'google_news', 'yahoo_finance', 'reddit']:
    df = pd.read_csv(f"out/{source}.csv")
    validate_schema(df, source)
```

## Performance Optimization

### **GDELT Rate Limiting**
Current: 0.25s between requests
- **Conservative**: Keep as-is (reliable)
- **Aggressive**: Reduce to 0.1s (may hit rate limits)
- **Balanced**: 0.15s with retry logic

### **Batch Processing**
```python
# Process tickers in smaller batches
def process_tickers_batch(tickers, batch_size=5):
    for i in range(0, len(tickers), batch_size):
        batch = dict(list(tickers.items())[i:i+batch_size])
        # Process batch
        time.sleep(1)  # Brief pause between batches
```

## Integration with Your Pipeline

### **Load News Data**
```python
# In your main analysis notebook
import pandas as pd

def load_news_data(out_dir="out"):
    """Load all news sources"""
    sources = ['gdelt', 'google_news', 'yahoo_finance', 'reddit']
    dfs = []
    
    for source in sources:
        try:
            df = pd.read_csv(f"{out_dir}/{source}.csv")
            if not df.empty:
                dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: {source}.csv not found")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()
```

### **Time Series Analysis**
```python
# Create daily news count
news_df = load_news_data()
news_df['date'] = pd.to_datetime(news_df['date'])
daily_counts = news_df.groupby([news_df['date'].dt.date, 'ticker']).size().unstack(fill_value=0)

# Plot news volume over time
daily_counts.plot(figsize=(15, 8))
plt.title('Daily News Volume by Ticker')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Troubleshooting Common Issues

### **GDELT Issues**
```python
# Check GDELT response
import requests
params = {
    "query": '("Microsoft Corporation" OR MSFT) AND sourcelang:english',
    "mode": "ArtList", 
    "format": "json",
    "startdatetime": "20240101000000",
    "enddatetime": "20240131235959",
    "maxrecords": "10"
}
response = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params)
print(f"Status: {response.status_code}")
print(f"Response: {response.text[:200]}")
```

### **Import Issues**
```bash
# Ensure you're in the right directory
cd /path/to/news_scraper

# Check Python path
python -c "import sys; print(sys.path)"

# Test imports
python -c "from src.sources import gdelt; print('Import successful')"
```

## Next Milestones

### **Week 1: Google News RSS**
- [ ] Implement RSS feed parsing
- [ ] Test with current tickers
- [ ] Validate data quality

### **Week 2: Yahoo Finance**
- [ ] Research scraping approach
- [ ] Implement basic scraping
- [ ] Handle rate limiting

### **Week 3: Data Integration**
- [ ] Merge all sources
- [ ] Create analysis pipeline
- [ ] Performance testing

### **Week 4: Sentiment Analysis**
- [ ] Integrate with existing tools
- [ ] Create sentiment scores
- [ ] Time series analysis

---

**Remember**: Start small, test thoroughly, then scale up. The current GDELT implementation gives you a solid foundation to build upon. 