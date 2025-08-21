# News Scraper - Financial News Collection System

- **Google News**: RSS-based recent news collection
- **Yahoo Finance**: Financial news scraping (requires implementation)
- **Reddit**: Social media sentiment (requires API keys)

## Configuration

start_date: "2015-01-01"    # Historical data start
end_date: null               # null = today


quality_domains:
     -- Tier 1: Institutional-grade financial sources
      'reuters.com',
      'bloomberg.com',
      'wsj.com',
      'ft.com',
      'financialtimes.com',
      
      -- Tier 2: Professional financial media
      'cnbc.com',
      'marketwatch.com',
      'barrons.com',
      'forbes.com',
      'thestreet.com',
      
      -- Tier 3: Established business & financial platforms  
      'finance.yahoo.com',
      'businessinsider.com',
      'economist.com',
      'morningstar.com',
      'seekingalpha.com',
      'investors.com',
      'fool.com',
      
      -- Tier 4: Major news outlets with strong business sections
      'nytimes.com',
      'washingtonpost.com',
      'theguardian.com'


tickers:                     # Your portfolio
  RDDT: "Reddit Inc"
  NVDA: "NVIDIA Corporation"
  SMR: "NuScale Power Corporation"
  MU: "Micron Technology Inc"
  MRVL: "Marvell Technology Inc"
  MSFT: "Microsoft Corporation"
  ASML: "ASML Holding NV"
  AEM: "Agnico Eagle Mines Limited"
  AMD: "Advanced Micro Devices Inc"
  VERU: "Veru Inc"
  AI: "C3.ai Inc"
  GOOGL: "Alphabet Inc"
  INGM: "Inogen Inc"
  PLUG: "Plug Power Inc"
  IONQ: "IonQ Inc"
  RGTI: "Rigetti Computing Inc"
  ARBE: "Arbe Robotics Ltd"
  APP: "AppLovin Corporation"
  QBTS: "D-Wave Quantum Inc"
  PLTR: "Palantir Technologies Inc"
```

### 4. **Data Schema**
Each article follows this structure:
```python
{
    "ticker": "MSFT",                    # Stock symbol
    "date": "2024-01-15 10:30:00",      # Publication date
    "title": "Microsoft Reports Earnings", # Article title
    "url": "https://...",                # Source URL
    "domain": "bloomberg.com",           # Source domain
    "is_quality_source": True,           # From trusted domain
    "source": "gdelt"                    # Data source
}
```


2. **Implement Google News RSS**
   - Most reliable stub to implement
   - Provides recent news complement to GDELT
   - No API keys required

3. **Data Quality Validation**
   - Verify GDELT data quality for your tickers
   - Check coverage gaps by time period
   - Validate domain filtering effectiveness

4. **Yahoo Finance Implementation**
   - Scrape financial news pages
   - Handle rate limiting and anti-bot measures
   - Focus on earnings and financial reports

5. **Reddit Integration**
   - Requires Reddit API keys
   - Collect r/wallstreetbets, r/investing sentiment
   - Implement sentiment scoring

### **Long-term Enhancements**
6. **Sentiment Analysis Pipeline**
   - Integrate with your existing sentiment tools
   - Add VADER or BERT-based scoring
   - Create sentiment time series

7. **Real-time Updates**
   - Schedule daily collection runs
   - Implement incremental updates
   - Add data freshness monitoring

8. **Advanced Filtering**
   - Topic-based filtering (earnings, M&A, etc.)
   - Relevance scoring
   - Custom quality metrics
