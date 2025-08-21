-- HARLF Portfolio Sentiment Analysis - Quality Sources Only
-- This query extracts sentiment data from trusted financial news sources
-- for your portfolio assets using GDELT database

WITH 
-- Step 1: Filter GDELT data for quality sources only (early filtering for efficiency)
gdelt_filtered AS (
  SELECT 
    SAFE.PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)) AS ts,
    LOWER(SourceCommonName) AS domain,
    DocumentIdentifier AS url,
    V2Tone,
    LOWER(IFNULL(V2Persons, '')) AS persons,
    LOWER(IFNULL(V2Organizations, '')) AS orgs
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE
    _PARTITIONTIME BETWEEN TIMESTAMP('2015-01-01') AND TIMESTAMP('2025-08-01')
    
    -- Ensure valid tone data
    AND V2Tone IS NOT NULL
    AND ARRAY_LENGTH(SPLIT(V2Tone, ',')) >= 7
    AND ABS(SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) <= 100
    AND SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(6)] AS INT64) >= 30  -- Min 30 words
    
    -- QUALITY SOURCES ONLY - Filter at source for efficiency
    AND REGEXP_REPLACE(LOWER(SourceCommonName), r'^www\.', '') IN (
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
    )
),

-- Step 2: Define ticker patterns for your HARLF portfolio
dictionary AS (
  -- Your current portfolio assets
  SELECT 'RDDT' AS ticker, r'\b(reddit|reddit inc)\b' AS pattern UNION ALL
  SELECT 'NVDA', r'\b(nvidia|geforce|rtx|cuda|jensen huang)\b' UNION ALL
  SELECT 'SMR',  r'\b(nuscale|nuscale power|smr reactor)\b' UNION ALL
  SELECT 'MU',   r'\b(micron|micron technology)\b' UNION ALL
  SELECT 'MRVL', r'\b(marvell|marvell technology)\b' UNION ALL
  SELECT 'MSFT', r'\b(microsoft|azure|xbox|windows|satya nadella)\b' UNION ALL
  SELECT 'ASML', r'\b(asml|asml holding)\b' UNION ALL
  SELECT 'AEM',  r'\b(agnico eagle|agnico eagle mines)\b' UNION ALL
  SELECT 'AMD',  r'\b(amd|advanced micro devices|ryzen|epyc)\b' UNION ALL
  SELECT 'VERU', r'\b(veru|veru inc)\b' UNION ALL
  SELECT 'AI',   r'\b(c3\.ai|c3 ai|c3ai)\b' UNION ALL
  SELECT 'GOOGL', r'\b(google|alphabet|waymo|deepmind|sundar pichai)\b' UNION ALL
  SELECT 'INGM', r'\b(inogen|inogen inc)\b' UNION ALL
  SELECT 'PLUG', r'\b(plug power|plug)\b' UNION ALL
  SELECT 'IONQ', r'\b(ionq|ionq inc)\b' UNION ALL
  SELECT 'RGTI', r'\b(rigetti|rigetti computing)\b' UNION ALL
  SELECT 'ARBE', r'\b(arbe|arbe robotics)\b' UNION ALL
  SELECT 'APP',  r'\b(applovin|app lovin)\b' UNION ALL
  SELECT 'QBTS', r'\b(d-wave|dwave|d\-wave systems)\b' UNION ALL
  SELECT 'PLTR', r'\b(palantir|alex karp)\b'
),
-- Step 3: Match articles to tickers
tagged_articles AS (
  SELECT
    d.ticker,
    g.url,
    g.domain,
    EXTRACT(YEAR FROM g.ts) AS year,
    EXTRACT(MONTH FROM g.ts) AS month,
    
    -- Sentiment scores from GDELT
    SAFE_CAST(SPLIT(g.V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tone,
    SAFE_CAST(SPLIT(g.V2Tone, ',')[OFFSET(1)] AS FLOAT64) AS positive,
    SAFE_CAST(SPLIT(g.V2Tone, ',')[OFFSET(2)] AS FLOAT64) AS negative,
    
    -- Article count for this calculation
    SAFE_CAST(SPLIT(g.V2Tone, ',')[OFFSET(6)] AS INT64) AS word_count
    
  FROM gdelt_filtered g
  JOIN dictionary d
    ON REGEXP_CONTAINS(
      LOWER(CONCAT(g.url, ' ', g.orgs, ' ', g.persons)), 
      d.pattern
    )
),

-- Step 4: Aggregate monthly sentiment
monthly_sentiment AS (
  SELECT
    ticker,
    year,
    month,
    COUNT(*) AS article_count,
    
    -- Core sentiment metrics
    AVG(tone) AS tone_mean,
    STDDEV(tone) AS tone_std,
    AVG(positive) AS positive_mean,
    AVG(negative) AS negative_mean,
    
    -- Calculate FinBERT-style sentiment score [-1, 1]
    (AVG(positive) - AVG(negative)) / 100 AS finbert_sentiment,
    
    -- Confidence metric based on article volume
    CASE 
      WHEN COUNT(*) >= 50 THEN 'high'
      WHEN COUNT(*) >= 20 THEN 'medium'
      WHEN COUNT(*) >= 7 THEN 'low'
      ELSE 'insufficient'
    END AS confidence_level,
    
    -- Sample URLs for validation
    ARRAY_AGG(DISTINCT url IGNORE NULLS LIMIT 5) AS sample_urls,
    
    -- Domain distribution for quality check
    ARRAY_AGG(DISTINCT domain IGNORE NULLS LIMIT 10) AS sources_used
    
  FROM tagged_articles
  WHERE ticker IS NOT NULL
  GROUP BY ticker, year, month
  HAVING COUNT(*) >= 7  -- Minimum threshold for reliability
),

-- Step 5: Add coverage statistics
final_with_stats AS (
  SELECT
    ticker,
    year,
    month,
    article_count,
    tone_mean,
    tone_std,
    positive_mean,
    negative_mean,
    finbert_sentiment,
    confidence_level,
    
    -- Add percentile ranks for context
    PERCENT_RANK() OVER (
      PARTITION BY ticker 
      ORDER BY article_count
    ) AS coverage_percentile,
    
    -- Add sentiment volatility flag
    CASE 
      WHEN tone_std > 10 THEN 'high_volatility'
      WHEN tone_std > 5 THEN 'moderate_volatility'
      ELSE 'stable'
    END AS sentiment_volatility,
    
    sample_urls,
    sources_used
    
  FROM monthly_sentiment
)

-- Final output
SELECT 
  ticker,
  year,
  month,
  article_count,
  ROUND(tone_mean, 2) AS tone_mean,
  ROUND(positive_mean, 2) AS positive_mean,
  ROUND(negative_mean, 2) AS negative_mean,
  ROUND(finbert_sentiment, 3) AS finbert_sentiment,
  confidence_level,
  sentiment_volatility,
  ROUND(coverage_percentile, 2) AS coverage_percentile,
  sample_urls,
  sources_used
FROM final_with_stats
ORDER BY ticker, year, month;
 