-- Complete V2Tone query: Expanded sources + Your proven validation + All tickers
WITH 
-- Step 1: Filter GDELT data with your proven validation logic
gdelt_filtered AS (
  SELECT 
    DATE,
    LOWER(SourceCommonName) AS domain,
    DocumentIdentifier AS url,
    V2Tone,
    LOWER(IFNULL(V2Persons, '')) AS persons,
    LOWER(IFNULL(V2Organizations, '')) AS orgs
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE
    _PARTITIONTIME BETWEEN TIMESTAMP('2025-07-19') AND TIMESTAMP('2025-08-01')
    
    -- Your complete V2Tone validation (proven approach)
    AND V2Tone IS NOT NULL
    AND V2Tone != '0'
    AND ARRAY_LENGTH(SPLIT(V2Tone, ',')) >= 7
    AND ABS(SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) <= 100
    AND SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(6)] AS INT64) >= 30  -- Min 30 words
    
    -- Your SourceCommonName trick: Efficient source filtering
    AND REGEXP_REPLACE(LOWER(SourceCommonName), r'^www\.', '') IN (
      'reuters.com',
      'bloomberg.com',
      'wsj.com',
      'cnn.com',
      'bbc.com',
      'yahoo.com',
      'cnbc.com',
      '4-traders.com',
      'marketwatch.com',
      'seekingalpha.com',
      'barrons.com',
      'forbes.com',
      'businessinsider.com',
      'morningstar.com',
      'investors.com',
      'fool.com',
      'nytimes.com',
      'washingtonpost.com',
      'theguardian.com',
      'ft.com',
      'financialtimes.com',
      'thestreet.com',
      'economist.com',
      'foxbusiness.com',
      'investing.com'
      )
),

-- Step 2: IMPROVED ticker patterns with better matching
dictionary AS (
  SELECT 'RDDT' AS ticker, r'\b(reddit|reddit inc)\b' AS pattern UNION ALL
  SELECT 'NVDA', r'\b(nvidia|geforce|rtx|cuda|jensen huang)\b' UNION ALL
  SELECT 'SMR',  r'\b(nuscale|nuscale power|smr reactor)\b' UNION ALL
  SELECT 'MU',   r'\b(micron|micron technology)\b' UNION ALL
  SELECT 'MRVL', r'\b(marvell|marvell technology)\b' UNION ALL
  SELECT 'MSFT', r'\b(microsoft|azure|xbox|windows|satya nadella)\b' UNION ALL
  SELECT 'ASML', r'\b(asml|asml holding)\b' UNION ALL
  SELECT 'AEM',  r'\b(agnico eagle|agnico eagle mines|agnico)\b' UNION ALL  -- IMPROVED
  SELECT 'AMD',  r'\b(amd|advanced micro devices|ryzen|epyc)\b' UNION ALL
  SELECT 'VERU', r'\b(veru|veru inc|veru pharmaceuticals)\b' UNION ALL  -- IMPROVED
  SELECT 'AI',   r'\b(c3\.ai|c3 ai|c3ai)\b' UNION ALL
  SELECT 'GOOGL', r'\b(google|alphabet|waymo|deepmind|sundar pichai)\b' UNION ALL
  SELECT 'INGM', r'\b(inogen|inogen inc|inogen medical)\b' UNION ALL  -- IMPROVED
  SELECT 'PLUG', r'\b(plug power|plug)\b' UNION ALL
  SELECT 'IONQ', r'\b(ionq|ionq inc|ionq quantum)\b' UNION ALL  -- IMPROVED
  SELECT 'RGTI', r'\b(rigetti|rigetti computing|rigetti quantum)\b' UNION ALL  -- IMPROVED
  SELECT 'ARBE', r'\b(arbe|arbe robotics|arbe systems)\b' UNION ALL  -- IMPROVED
  SELECT 'APP',  r'\b(applovin|app lovin|applovin corp)\b' UNION ALL  -- IMPROVED
  SELECT 'QBTS', r'\b(d-wave|dwave|d\-wave systems|dwave quantum)\b' UNION ALL  -- IMPROVED
  SELECT 'QQQ', r'\b(qqq|nasdaq|nasdaq-100|nasdaq 100)\b' UNION ALL  -- IMPROVED
  SELECT 'SPY', r'\b(spy|spdr|s&p 500|s&p500|standard & poor)\b' UNION ALL  -- IMPROVED
  SELECT 'PLTR', r'\b(palantir|alex karp|palantir technologies)\b'  -- IMPROVED
),

-- Step 3: Match articles to tickers (your proven approach)
tagged_articles AS (
  SELECT
    d.ticker,
    g.url,
    g.domain,
    g.DATE,
    EXTRACT(YEAR FROM SAFE.PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(g.DATE AS STRING))) AS year,
    EXTRACT(MONTH FROM SAFE.PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(g.DATE AS STRING))) AS month,
    
    -- Sentiment scores from GDELT (your proven approach)
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

-- Step 4: Monthly aggregation with your confidence levels
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
    
    -- Your confidence metric based on article volume
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
  HAVING COUNT(*) >= 7  -- Your minimum threshold for reliability
)

-- Final output with all your metrics
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
  sample_urls,
  sources_used
FROM monthly_sentiment
ORDER BY ticker, year, month;