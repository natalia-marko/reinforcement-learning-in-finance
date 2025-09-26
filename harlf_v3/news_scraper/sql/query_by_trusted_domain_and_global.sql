WITH
dictionary AS (
  SELECT 'aem'  AS ticker, r'\bagnico eagle\b|\baem\b|\baem inc\b' AS pattern UNION ALL
  SELECT 'rddt', r'\breddit\b|\brddt\b' UNION ALL
  SELECT 'nvda', r'\bnvidia\b|\bnvda\b|\bcuda\b|\bgeforce\b' UNION ALL
  SELECT 'smr',  r'\bnuscale\b|\bsmr\b|\bvoygr\b|\bsmall modular reactor\b' UNION ALL
  SELECT 'rgti', r'\brigetti\b|\brgti\b' UNION ALL
  SELECT 'mu',   r'\bmicron\b|\bmu\b|\bmicron technology\b' UNION ALL
  SELECT 'amd',  r'\bamd\b|\badvanced micro devices\b|\bryzen\b|\bepyc\b|\bradeon\b' UNION ALL
  SELECT 'app',  r'\bapplovin\b|\bapp[- ]lovin\b|\bapplovin corp\b' UNION ALL
  SELECT 'mrvl', r'\bmarvell\b|\bmrvl\b' UNION ALL
  SELECT 'msft', r'\bmicrosoft\b|\bmsft\b|\bazure\b' UNION ALL
  SELECT 'qbts', r'\bqbts\b|\bd[- ]?wave\b|\bd[- ]?wave quantum\b|\bdwave quantum\b|\bdwave\b' UNION ALL
  SELECT 'pltr', r'\bpalantir\b|\bpltr\b' UNION ALL
  SELECT 'asml', r'\basml\b|\blithograph\w*\b|\beuv\b' UNION ALL
  SELECT 'goog', r'\bgoogle\b|\balphabet\b|\bgoog\b|\bgoogle cloud\b' UNION ALL
  SELECT 'ionq', r'\bionq\b|\bionq aria\b|\bionq forte\b' UNION ALL
  SELECT 'veru', r'\bveru\b|\bveru inc\b|\bveru pharmaceuticals\b' UNION ALL
  SELECT 'ai',   r'\bc3\.?ai\b|\bc3 ai\b|\bthomas siebel\b' UNION ALL
  SELECT 'arbe', r'\barbe\b|\barbe robotics\b|\barbe radar\b' UNION ALL
  SELECT 'qqq',  r'\binvesco qqq\b|\bqqq etf\b|\bnasdaq[- ]?100 etf\b|\bnasdaq 100 etf\b'
),

-- Per-ticker domain mapping (expand as needed for niche/low-coverage tickers)
trusted_domains AS (
  SELECT 'qbts' AS ticker, 'businesswire.com' AS domain UNION ALL
  SELECT 'qbts', 'prnewswire.com' AS domain UNION ALL
  SELECT 'qbts', 'quantumcomputingreport.com' AS domain UNION ALL
  SELECT 'qbts', 'globenewswire.com' AS domain UNION ALL
  SELECT 'qbts', 'finance.yahoo.com' AS domain UNION ALL
  SELECT 'qbts', 'marketwatch.com' AS domain UNION ALL
  SELECT 'qbts', 'ft.com' AS domain UNION ALL
  
  -- RGTI (Rigetti): quantum/PR
  SELECT 'rgti', 'businesswire.com' AS domain UNION ALL
  SELECT 'rgti', 'prnewswire.com' AS domain UNION ALL
  SELECT 'rgti', 'finance.yahoo.com' AS domain UNION ALL

  -- IONQ: quantum/PR
  SELECT 'ionq', 'prnewswire.com' AS domain UNION ALL
  SELECT 'ionq', 'finance.yahoo.com' AS domain UNION ALL
  SELECT 'ionq', 'businesswire.com' AS domain UNION ALL

  -- ASML: global tech, European, PR, semis
  SELECT 'asml', 'reuters.com' AS domain UNION ALL
  SELECT 'asml', 'bloomberg.com' AS domain UNION ALL
  SELECT 'asml', 'ft.com' AS domain UNION ALL
  SELECT 'asml', 'marketwatch.com' AS domain UNION ALL
  SELECT 'asml', 'finance.yahoo.com' AS domain UNION ALL
  SELECT 'asml', 'businesswire.com' AS domain UNION ALL
  SELECT 'asml', 'globenewswire.com' AS domain UNION ALL
  SELECT 'asml', 'seekingalpha.com' AS domain UNION ALL
  SELECT 'asml', 'prnewswire.com' AS domain UNION ALL
  SELECT 'asml', 'techcrunch.com' AS domain UNION ALL
  SELECT 'asml', 'marketscreener.com' AS domain UNION ALL

  -- ARBE: PR/newswire, auto-tech
  SELECT 'arbe', 'businesswire.com' AS domain UNION ALL
  SELECT 'arbe', 'prnewswire.com' AS domain UNION ALL
  SELECT 'arbe', 'globenewswire.com' AS domain UNION ALL
  SELECT 'arbe', 'marketwatch.com' AS domain UNION ALL
  SELECT 'arbe', 'finance.yahoo.com' AS domain UNION ALL
  SELECT 'arbe', 'streetinsider.com' AS domain UNION ALL
  SELECT 'arbe', 'benzinga.com' AS domain UNION ALL
  SELECT 'arbe', 'seekingalpha.com' AS domain UNION ALL

  -- VERU: biotech/pharma, PR, finance
  SELECT 'veru', 'businesswire.com' AS domain UNION ALL
  SELECT 'veru', 'prnewswire.com' AS domain UNION ALL
  SELECT 'veru', 'globenewswire.com' AS domain UNION ALL
  SELECT 'veru', 'marketwatch.com' AS domain UNION ALL
  SELECT 'veru', 'finance.yahoo.com' AS domain UNION ALL
  SELECT 'veru', 'seekingalpha.com' AS domain UNION ALL
  SELECT 'veru', 'benzinga.com' AS domain UNION ALL
  SELECT 'veru', 'thestreet.com' AS domain UNION ALL
  SELECT 'veru', 'marketscreener.com' AS domain UNION ALL
  SELECT 'veru', 'fool.com' AS domain
),
-- Global trusted domains (blue-chip and broad coverage)
global_domains AS (
  SELECT domain FROM UNNEST([
    'reuters.com', 'bloomberg.com', 'wsj.com', 'financialtimes.com', 'cnbc.com',
    'marketwatch.com', 'ft.com', 'nytimes.com', 'finance.yahoo.com',
    'seekingalpha.com', 'zacks.com', 'benzinga.com', 'investing.com', 'finviz.com',
    'techcrunch.com', 'venturebeat.com', 'theverge.com', 'thestreet.com', 'morningstar.com',
    'bnnbloomberg.ca', 'mining.com', 'prnewswire.com', 'globenewswire.com',
    'fool.com', 'marketscreener.com', 'earnings.com'
  ]) AS domain
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
  WHERE DATE >= 20150301000000
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
    TO_HEX(SHA256(CONCAT(
      LOWER(REGEXP_REPLACE(r.url, r'(\?.*|#.*)$', '')),
      '||',
      LOWER(SUBSTR(COALESCE(r.orgs,''),1,300)),
      '||',
      SUBSTR(SAFE_CAST(SPLIT(r.V2Tone, ',')[OFFSET(0)] AS STRING), 1, 16)
    ))) AS content_sig
  FROM gdelt_raw r
  JOIN dictionary d ON REGEXP_CONTAINS(r.orgs, d.pattern)
  LEFT JOIN trusted_domains td ON td.ticker = d.ticker AND r.domain = td.domain
  LEFT JOIN global_domains gd ON r.domain = gd.domain
  WHERE (
    td.domain IS NOT NULL  -- per-ticker domain match
    OR gd.domain IS NOT NULL  -- global domain match
  )
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
