WITH 
-- Step 1: Filter GDELT data for trusted sources and substantial articles
gdelt_filtered AS (
  SELECT 
    g.DATE,
    PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(g.DATE AS STRING)) AS ts,
    REGEXP_REPLACE(LOWER(COALESCE(NET.HOST(DocumentIdentifier), SourceCommonName)), r'^(www|m|amp)\.', '') AS domain,
    LOWER(DocumentIdentifier) AS url,
    LOWER(IFNULL(V2Organizations, '')) AS orgs,
    V2Tone
  FROM `gdelt-bq.gdeltv2.gkg_partitioned` g
  WHERE
    _PARTITIONTIME >= TIMESTAMP('2015-02-19')
    AND _PARTITIONTIME <  TIMESTAMP(CURRENT_DATE())  -- Always runs up to the current date
    AND V2Tone IS NOT NULL
    AND V2Tone != '0'
    AND ARRAY_LENGTH(SPLIT(V2Tone, ',')) >= 7
    AND ABS(SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) <= 100
    AND SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(6)] AS INT64) >= 30        -- Filter for articles with >=30 words
    AND REGEXP_REPLACE(LOWER(COALESCE(NET.HOST(DocumentIdentifier), SourceCommonName)), r'^(www|m|amp)\.', '') IN (
            -- High-quality, editorially-vetted financial and general news sources
            'barrons.com', 'bloomberg.com', 'businessinsider.com', 'cnbc.com',
            'economist.com', 'financialtimes.com', 'forbes.com', 'marketwatch.com',
            'morningstar.com', 'nasdaq.com', 'nytimes.com', 'reuters.com',
            'washingtonpost.com', 'wsj.com', 'finance.yahoo.com'
    )
),

-- Step 2: Define patterns for POPULAR assets only
dictionary AS (
  -- Big Tech & Semiconductors
  SELECT 'MSFT' AS ticker, r'(?i)\b(?:microsoft|azure|satya\s+nadella)\b' AS pattern UNION ALL
  SELECT 'GOOGL', r'(?i)\b(?:google|alphabet|google\s+cloud|waymo|deepmind|sundar\s+pichai)\b' UNION ALL
  SELECT 'NVDA',  r'(?i)\b(?:nvidia|geforce|cuda|jensen\s+huang)\b|(?:nvidia.{0,40}\brtx\b|\brtx\b.{0,40}nvidia)' UNION ALL
  SELECT SELECT 'NVDA', r'(?i)\b(?:nvidia|geforce|cuda|jensen\s+huang)\b|(?:nvidia.{0,40}\brtx\b|\brtx\b.{0,40}nvidia)' UNION ALL
  SELECT 'AMD',   r'(?i)\b(?:amd|advanced\s+micro\s+devices|ryzen|epyc)\b' UNION ALL
  SELECT 'MU',    r'(?i)\bmicron(?:\s+technology)?\b' UNION ALL
  SELECT 'MRVL',  r'(?i)\bmarvell(?:\s+technology)?\b' UNION ALL
  SELECT 'ASML',  r'(?i)\basml(?:\s+holding)?\b' UNION ALL
  SELECT 'AI',    r'(?i)\b(?:c3\.?ai|c3\s*ai|thomas\s+siebel)\b' UNION ALL
  SELECT 'APP',   r'(?i)\bapplovin|app\s*lovin|applovin\s+corp\b' UNION ALL

  -- Popular Specialized Tech
  SELECT 'PLTR',  r'(?i)\bpalantir(?:\s+technologies)?\b|\b(?:alex\s+karp|peter\s+thiel)\b' UNION ALL
  SELECT 'RDDT',  r'(?i)\breddit(?:,?\s+inc\.)?\b' UNION ALL
  
  -- Major Market ETFs
  SELECT 'QQQ',  r'(?i)(^|[^a-z0-9])qqq([^a-z0-9]|$)|\binvesco\s+qqq\b' UNION ALL
  SELECT 'SPY',  r'(?i)(^|[^a-z0-9])spy([^a-z0-9]|$)|\bspdr\s+s&p\s*500(?:\s+etf|\s+trust)?\b'
),

-- Step 3: Tag articles with tickers and deduplicate
tagged_articles AS (
  SELECT
    d.ticker,
    g.url,
    g.domain,
    g.DATE,
    EXTRACT(YEAR  FROM g.ts) AS year,
    EXTRACT(MONTH FROM g.ts) AS month,
    SAFE_CAST(SPLIT(g.V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tone,
    SAFE_CAST(SPLIT(g.V2Tone, ',')[OFFSET(1)] AS FLOAT64) AS positive,
    SAFE_CAST(SPLIT(g.V2Tone, ',')[OFFSET(2)] AS FLOAT64) AS negative,
    SAFE_CAST(SPLIT(g.V2Tone, ',')[OFFSET(6)] AS INT64)   AS word_count
  FROM gdelt_filtered AS g
  JOIN dictionary AS d
    ON REGEXP_CONTAINS(CONCAT(g.url, ' ', g.orgs), d.pattern)
  QUALIFY ROW_NUMBER() OVER (PARTITION BY d.ticker, g.url ORDER BY g.ts DESC) = 1  -- Deduplicate by URL per ticker
),

-- Step 4: Aggregate sentiment metrics by month
monthly_sentiment AS (
  SELECT
    ticker,
    year,
    month,
    COUNT(*) AS article_count,
    AVG(tone)     AS tone_mean,
    STDDEV(tone)  AS tone_std,
    AVG(positive) AS positive_mean,
    AVG(negative) AS negative_mean,
    (AVG(positive) - AVG(negative)) / 100.0 AS sentiment_balance, -- Renamed for clarity
    CASE 
      WHEN COUNT(*) >= 50 THEN 'high'
      WHEN COUNT(*) >= 20 THEN 'medium'
      WHEN COUNT(*) >= 7  THEN 'low'
      ELSE 'insufficient'
    END AS confidence_level,
    ARRAY_AGG(DISTINCT url    IGNORE NULLS LIMIT 5)  AS sample_urls,
    ARRAY_AGG(DISTINCT domain IGNORE NULLS LIMIT 10) AS sources_used
  FROM tagged_articles
  GROUP BY ticker, year, month
  HAVING COUNT(*) >= 10 -- Only include months with a meaningful number of articles
)

-- Final Step: Select and format the output
SELECT 
  ticker,
  year,
  month,
  article_count,
  ROUND(tone_mean, 2)     AS tone_mean,
  ROUND(positive_mean, 2) AS positive_mean,
  ROUND(negative_mean, 2) AS negative_mean,
  ROUND(sentiment_balance, 3) AS sentiment_balance, -- Renamed for clarity
  confidence_level,
  sample_urls,
  sources_used
FROM monthly_sentiment
ORDER BY ticker, year, month;