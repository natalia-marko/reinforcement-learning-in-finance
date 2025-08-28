WITH 
-- Step 1: Filter GDELT data, V2Tone validation, better normalization
gdelt_filtered AS (
  SELECT 
    DATE,
    REGEXP_REPLACE(LOWER(COALESCE(NET.HOST(DocumentIdentifier), SourceCommonName)), r'^(www|m|amp)\.', '') AS domain,
    DocumentIdentifier AS url, -- DocumentIdentifier is the url of the news article
    V2Tone, -- V2Tone is precalculated sentiment score of the news article
    LOWER(IFNULL(V2Persons, '')) AS persons, -- V2Persons is the persons mentioned in the news article - ADDED BACK
    LOWER(IFNULL(V2Organizations, '')) AS orgs -- V2Organizations is the organizations mentioned in the news article
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE
    _PARTITIONTIME BETWEEN TIMESTAMP('2015-02-19') AND TIMESTAMP('2025-08-01')
    AND V2Tone IS NOT NULL
    AND V2Tone != '0' -- V2Tone is not 0  
    AND ARRAY_LENGTH(SPLIT(V2Tone, ',')) >= 7 -- V2Tone has 7 elements
    AND ABS(SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) <= 100 -- Tone score between -100 and 100
    
    -- TIERED WORD COUNT REQUIREMENTS
    AND (
      -- High-volume tickers need more words (reduce noise)
      (REGEXP_CONTAINS(LOWER(CONCAT(DocumentIdentifier, ' ', IFNULL(V2Organizations, ''), ' ', IFNULL(V2Persons, ''))), 
                      r'\b(google|microsoft|nvidia|nasdaq|s&p)\b') 
       AND SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(6)] AS INT64) >= 50)
      OR
      -- Low-volume tickers need fewer words (capture more signal) 
      (REGEXP_CONTAINS(LOWER(CONCAT(DocumentIdentifier, ' ', IFNULL(V2Organizations, ''), ' ', IFNULL(V2Persons, ''))), 
                      r'\b(ionq|nuscale|veru|rigetti|d-wave|inogen)\b')
       AND SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(6)] AS INT64) >= 20)
      OR
      -- Default threshold for all others
      SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(6)] AS INT64) >= 30
    )
    
    -- Source filtering with expanded domains
    AND REGEXP_REPLACE(LOWER(COALESCE(NET.HOST(DocumentIdentifier), SourceCommonName)),
                   r'^(www|m|amp)\.', '') IN (
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
      'investing.com',
      'prnewswire.com',
      'businesswire.com',
      'globenewswire.com',
      'sec.gov', 
      'investor.com', 
      'smallcappower.com',
      'earnings.com', 'stocknews.com', 'zacks.com', 'benzinga.com',
'biospace.com', 'mining.com', 'pharmanewsintel.com'
      )
),

-- Step 2: Enhanced ticker patterns
dictionary AS (
  SELECT 'RDDT' AS ticker, r'\b(reddit|reddit inc)\b' AS pattern UNION ALL
  SELECT 'NVDA', r'\b(nvidia|geforce|rtx|cuda|jensen huang)\b' UNION ALL
  SELECT 'MU',   r'\b(micron|micron technology)\b' UNION ALL
  SELECT 'MRVL', r'\b(marvell|marvell technology)\b' UNION ALL
  SELECT 'MSFT', r'\b(microsoft|azure|windows|satya nadella)\b' UNION ALL
  SELECT 'AMD',  r'\b(amd|advanced micro devices|ryzen|epyc|radeon)\b' UNION ALL
  SELECT 'GOOGL', r'\b(goog|google inc|google cloud|alphabet|waymo|deepmind|sundar pichai)\b' UNION ALL
  SELECT 'PLUG', r'\b(plug power|plug)\b' UNION ALL
  SELECT 'QQQ',  r'\b(qqq|nasdaq|nasdaq-100|nasdaq 100)\b' UNION ALL
  SELECT 'SPY', r'\b(spy|spdr|s&p.{0,3}500|sp500|s.?p.?500|standard.{0,10}poor)\b' UNION ALL
  SELECT 'PLTR', r'\b(palantir|alex karp|palantir technologies|big data platform)\b'
),

-- Step 3: FIXED - Proper matching logic (no domain in concatenation)
tagged_articles AS (
  SELECT
    d.ticker,
    g.url,
    g.domain,
    g.DATE,
    EXTRACT(YEAR FROM SAFE.PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(g.DATE AS STRING))) AS year,
    EXTRACT(MONTH FROM SAFE.PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(g.DATE AS STRING))) AS month,
    
    -- Sentiment scores from GDELT
    SAFE_CAST(SPLIT(g.V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tone,
    SAFE_CAST(SPLIT(g.V2Tone, ',')[OFFSET(1)] AS FLOAT64) AS positive,
    SAFE_CAST(SPLIT(g.V2Tone, ',')[OFFSET(2)] AS FLOAT64) AS negative,
    SAFE_CAST(SPLIT(g.V2Tone, ',')[OFFSET(6)] AS INT64) AS word_count
    
  FROM gdelt_filtered AS g
  JOIN dictionary AS d
    ON REGEXP_CONTAINS(
      LOWER(CONCAT(g.url, ' ', g.orgs, ' ', g.persons)),  -- FIXED: Back to original logic
      d.pattern
    )
  
  -- Optional: Add deduplication to prevent multiple matches per article
  QUALIFY ROW_NUMBER() OVER (PARTITION BY d.ticker, g.url ORDER BY g.DATE DESC) = 1
),

-- Step 4: Monthly aggregation  
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
    
    ARRAY_AGG(DISTINCT url IGNORE NULLS LIMIT 5) AS sample_urls,
    ARRAY_AGG(DISTINCT domain IGNORE NULLS LIMIT 10) AS sources_used
    
  FROM tagged_articles
  WHERE ticker IS NOT NULL
  GROUP BY ticker, year, month
  HAVING COUNT(*) >= 5
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
  sample_urls,
  sources_used
FROM monthly_sentiment
ORDER BY ticker, year, month;