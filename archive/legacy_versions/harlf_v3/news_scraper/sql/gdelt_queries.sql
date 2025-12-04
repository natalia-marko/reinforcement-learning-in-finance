-- File: sql/gdelt_monthly_sentiment_orgs_only.sql
-- Purpose: Maximum precision for company sentiment (only orgs match, de-dup by content signature)

WITH 
params AS (
  SELECT [
    'reuters.com', 'bloomberg.com', 'wsj.com', 'financialtimes.com', 'cnbc.com',
    'marketwatch.com', 'ft.com', 'nytimes.com', 'finance.yahoo.com', 'seekingalpha.com', 'zacks.com', 'benzinga.com', 'investing.com', 'finviz.com', 'techcrunch.com', 'venturebeat.com', 'theverge.com',
    'thestreet.com', 'morningstar.com', 'bnnbloomberg.ca', 'mining.com',
    'prnewswire.com', 'globenewswire.com', 'fool.com', 'marketscreener.com', 'earnings.com'
  ] AS trusted_domains
),
dictionary AS (
  SELECT 'aem'  AS ticker, r'\bagnico eagle\b|\baem\b|\baem inc\b' AS pattern UNION ALL
  SELECT 'rddt' AS ticker, r'\breddit\b|\brddt\b' AS pattern UNION ALL
  SELECT 'nvda', r'\bnvidia\b|\bnvda\b|\bcuda\b|\bgeforce\b' UNION ALL
  SELECT 'smr', r'\bnuscale\b|\bsmr\b|\bvoygr\b|\bsmall modular reactor\b' UNION ALL
  SELECT 'rgti', r'\brigetti\b|\brgti\b' UNION ALL
  SELECT 'mu', r'\bmicron\b|\bmu\b|\bmicron technology\b' UNION ALL
  SELECT 'amd', r'\bamd\b|\badvanced micro devices\b|\bryzen\b|\bepyc\b|\bradeon\b' UNION ALL
  SELECT 'app', r'\bapplovin\b|\bapp[- ]lovin\b|\bapplovin corp\b' UNION ALL
  SELECT 'mrvl', r'\bmarvell\b|\bmrvl\b' UNION ALL
  SELECT 'msft', r'\bmicrosoft\b|\bmsft\b|\bazure\b' UNION ALL
  SELECT 'qbts', r'\bqbts\b|\bd[- ]?wave\b' UNION ALL
  SELECT 'pltr', r'\bpalantir\b|\bpltr\b' UNION ALL
  SELECT 'app' , r'\bapplovin\b|\bapp[- ]lovin\b|\bapplovin corp\b' UNION ALL
  SELECT 'asml', r'\basml\b|\blithograph\w*\b|\beuv\b' UNION ALL
  SELECT 'goog', r'\bgoogle\b|\balphabet\b|\bgoog\b|\bgoogle cloud\b' UNION ALL
  SELECT 'ionq', r'\bionq\b|\bionq aria\b|\bionq forte\b' UNION ALL
  SELECT 'veru', r'\bveru\b|\bveru inc\b|\bveru pharmaceuticals\b' UNION ALL
  SELECT 'ai', r'\bc3\.?ai\b|\bc3 ai\b|\bthomas siebel\b' UNION ALL
  SELECT 'arbe', r'\barbe\b|\barbe robotics\b|\barbe radar\b' UNION ALL
  SELECT 'qqq', r'\binvesco qqq\b|\bqqq etf\b|\bnasdaq[- ]?100 etf\b|\bnasdaq 100 etf\b'
),
gdelt_raw AS (
  SELECT 
    PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)) AS ts,
    REGEXP_REPLACE(
      LOWER(COALESCE(REGEXP_EXTRACT(DocumentIdentifier, r'https?://([^/]+)'), SourceCommonName, '')),
      r'^(www|m|amp)\.', ''
    ) AS domain,
    LOWER(DocumentIdentifier) AS url,
    LOWER(IFNULL(V2Organizations, '')) AS orgs,
    V2Tone
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE DATE >= 20150101000000
    AND DATE <= 20300101000000
    AND V2Tone IS NOT NULL
),
candidates AS (
  SELECT
    d.ticker,
    r.ts,
    r.domain,
    r.url,
    r.orgs,
    r.V2Tone,
    SAFE_CAST(SPLIT(r.V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tone,
    SAFE_CAST(SPLIT(r.V2Tone, ',')[OFFSET(1)] AS FLOAT64) AS positive,
    SAFE_CAST(SPLIT(r.V2Tone, ',')[OFFSET(2)] AS FLOAT64) AS negative,
    SAFE_CAST(SPLIT(r.V2Tone, ',')[OFFSET(6)] AS INT64)   AS word_count,
    -- Content signature for deduplication: canonical url + orgs + tone snippet
    TO_HEX(SHA256(CONCAT(
      LOWER(REGEXP_REPLACE(r.url, r'(\?.*|#.*)$', '')),
      '||',
      LOWER(SUBSTR(COALESCE(r.orgs,''),1,300)),
      '||',
      SUBSTR(SAFE_CAST(SPLIT(r.V2Tone, ',')[OFFSET(0)] AS STRING), 1, 16)
    ))) AS content_sig
  FROM gdelt_raw r
  JOIN dictionary d ON REGEXP_CONTAINS(r.orgs, d.pattern)
  WHERE r.domain IN UNNEST((SELECT trusted_domains FROM params))
    AND SAFE_CAST(SPLIT(r.V2Tone, ',')[OFFSET(6)] AS INT64) >= 30
),
-- Deduplicate by (ticker, content_sig): keep latest ts per unique content
candidates_dedup AS (
  SELECT *
  FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker, content_sig ORDER BY ts DESC) AS rn
    FROM candidates
  )
  WHERE rn = 1
),
monthly_sentiment AS (
  SELECT
    ticker,
    EXTRACT(YEAR FROM ts) AS year,
    EXTRACT(MONTH FROM ts) AS month,
    COUNT(*) AS article_count,
    AVG(positive - negative) AS mean_signed_score,
    AVG(tone) AS mean_tone
  FROM candidates_dedup
  GROUP BY ticker, year, month
)
SELECT *
FROM monthly_sentiment
ORDER BY ticker, year, month;




                       r'^(www|m|amp)\.', '') IN (
      'reuters.com','bloomberg.com','wsj.com','cnn.com','bbc.com','yahoo.com',
      'cnbc.com','4-traders.com','marketwatch.com','seekingalpha.com','barrons.com',
      'forbes.com','businessinsider.com','morningstar.com','investors.com','fool.com',
      'nytimes.com','washingtonpost.com','theguardian.com','ft.com','financialtimes.com',
      'thestreet.com','economist.com','foxbusiness.com','investing.com',
      'prnewswire.com','businesswire.com','globenewswire.com','sec.gov',
      'investor.com','smallcappower.com','earnings.com','stocknews.com','zacks.com',
      'benzinga.com','biospace.com','mining.com','pharmanewsintel.com'
    )
),

-- Step 2: tighter patterns (RE2-safe, low-noise)
dictionary AS (
  SELECT 'RDDT' AS ticker, r'(?i)\breddit(?:,?\s+inc\.)?\b' AS pattern UNION ALL
  SELECT 'NVDA', r'(?i)\b(?:nvidia|geforce|cuda|jensen\s+huang)\b|(?:nvidia.{0,40}\brtx\b|\brtx\b.{0,40}nvidia)' UNION ALL
  SELECT 'MU',   r'(?i)\bmicron(?:\s+technology)?\b' UNION ALL
  SELECT 'MRVL', r'(?i)\bmarvell(?:\s+technology)?\b' UNION ALL
  SELECT 'MSFT', r'(?i)\b(?:microsoft|azure|satya\s+nadella)\b' UNION ALL              -- drop "windows"
  SELECT 'ASML', r'(?i)\basml(?:\s+holding)?\b' UNION ALL
  SELECT 'AEM',  r'(?i)\bagnico(?:\s+eagle)?\b' UNION ALL
  SELECT 'AMD',  r'(?i)\b(?:amd|advanced\s+micro\s+devices|ryzen|epyc)\b' UNION ALL
  SELECT 'VERU', r'(?i)\b(?:veru(?:\s+inc)?|zucinone|enobosarm|sabizabulin)\b' UNION ALL
  SELECT 'AI',   r'(?i)\b(?:c3\.?ai|c3\s*ai|thomas\s+siebel)\b' UNION ALL
  SELECT 'GOOGL',r'(?i)\b(?:google|alphabet|google\s+cloud|waymo|deepmind|sundar\s+pichai)\b' UNION ALL
  SELECT 'INGN', r'(?i)\binogen(?:\s+inc|\s+medical)?\b' UNION ALL                      -- fixed
  SELECT 'PLUG', r'(?i)\bplug\s+power\b' UNION ALL                                      -- no plain "plug"
  SELECT 'IONQ', r'(?i)\bion\s*q\b|\bionq\b' UNION ALL                                  -- drop generic "quantum computing"
  SELECT 'SMR',  r'(?i)\bnuscale(?:\s+power)?\b|\bnyse:\s*smr\b' UNION ALL              -- ticker only (sector handled elsewhere)
  SELECT 'RGTI', r'(?i)\brigetti(?:\s+computing|\s+quantum)?\b' UNION ALL
  SELECT 'ARBE', r'(?i)\barbe(?:\s+robotics|\s+systems)?\b' UNION ALL
  SELECT 'APP',  r'(?i)\bapplovin|app\s*lovin|applovin\s+corp\b' UNION ALL
  SELECT 'QBTS', r'(?i)\bd-?wave(?:\s+systems)?|dwave\s+quantum\b' UNION ALL
  -- ETFs: RE2-safe boundaries (no lookarounds)
  SELECT 'QQQ',  r'(?i)(^|[^a-z0-9])qqq([^a-z0-9]|$)|\binvesco\s+qqq\b' UNION ALL
  SELECT 'SPY',  r'(?i)(^|[^a-z0-9])spy([^a-z0-9]|$)|\bspdr\s+s&p\s*500(?:\s+etf|\s+trust)?\b' UNION ALL
  SELECT 'PLTR', r'(?i)\bpalantir(?:\s+technologies)?\b|\balex\s+karp\b'
),

-- Step 3: join & tag (no domain in match; compute year/month once)
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
  QUALIFY ROW_NUMBER() OVER (PARTITION BY d.ticker, g.url ORDER BY g.ts DESC) = 1  -- light dedup
),

-- Step 4: Monthly aggregation (unchanged logic)
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
    (AVG(positive) - AVG(negative)) / 100.0 AS finbert_sentiment,   -- this is GDELT-derived balance; rename later if you run FinBERT
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
  HAVING COUNT(*) >= 7
)

SELECT 
  ticker,
  year,
  month,
  article_count,
  ROUND(tone_mean, 2)     AS tone_mean,
  ROUND(positive_mean, 2) AS positive_mean,
  ROUND(negative_mean, 2) AS negative_mean,
  ROUND(finbert_sentiment, 3) AS finbert_sentiment,
  confidence_level,
  sample_urls,
  sources_used
FROM monthly_sentiment
ORDER BY ticker, year, month;
