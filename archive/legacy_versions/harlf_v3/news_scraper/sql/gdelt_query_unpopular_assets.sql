WITH 
-- Step 1: Filter GDELT data for unpopular assets with expanded sources
gdelt_filtered AS (
  SELECT 
    DATE,
    REGEXP_REPLACE(LOWER(COALESCE(NET.HOST(DocumentIdentifier), SourceCommonName)), r'^(www|m|amp)\.', '') AS domain,
    DocumentIdentifier AS url,
    V2Tone,
    LOWER(IFNULL(V2Persons, '')) AS persons,
    LOWER(IFNULL(V2Organizations, '')) AS orgs
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE
    _PARTITIONTIME BETWEEN TIMESTAMP('2015-02-19') AND TIMESTAMP('2025-08-01')
    AND V2Tone IS NOT NULL
    AND V2Tone != '0'
    AND ARRAY_LENGTH(SPLIT(V2Tone, ',')) >= 7
    AND ABS(SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) <= 100
    AND SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(6)] AS INT64) >= 10  -- Lower threshold for rescue
    
    -- Expanded domain sources for better coverage
    AND REGEXP_REPLACE(LOWER(COALESCE(NET.HOST(DocumentIdentifier), SourceCommonName)),
                   r'^(www|m|amp)\.', '') IN (
    'reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com', 'financialtimes.com',
    'cnbc.com', 'marketwatch.com', 'barrons.com', 'forbes.com',
    'businessinsider.com', 'morningstar.com', 'seekingalpha.com',
    'investing.com', 'thestreet.com', 'economist.com', 'foxbusiness.com',

    -- News wires & press releases
    'prnewswire.com', 'businesswire.com', 'globenewswire.com',
    'accesswire.com', 'newswire.com', 'einnews.com',

    -- Financial analysis sites
    'earnings.com', 'stocknews.com', 'zacks.com', 'benzinga.com',
    'stockanalysis.com', 'gurufocus.com', 'simplywall.st', 'marketbeat.com',
    'tradingview.com', 'finviz.com', 'alphastockn.com', 'investors.com', 'fool.com',

    -- Industry-specific sources
    'mining.com', 'kitco.com', 'goldseek.com',
    'biospace.com', 'fiercebitech.com', 'biopharmadive.com', 'pharmanewsintel.com',
    'techcrunch.com', 'theverge.com', 'arstechnica.com', 'venturebeat.com',
    'zdnet.com', 'computerworld.com',

    -- SEC and regulatory
    'sec.gov', 'investor.com', 'streetinsider.com', 'smallcappower.com',

    -- General news sources
    'nytimes.com', 'washingtonpost.com', 'theguardian.com',
    'bbc.com', 'cnn.com', 'yahoo.com', 'apnews.com', '4-traders.com'
    )
),

-- Step 2: Unpopular asset patterns - enhanced but focused
dictionary AS (
  SELECT 'AEM' AS ticker, r'(?i)\b(agnico eagle|aem|agnico gold|detour lake|canadian malartic|kittila mine)\b|(nyse|tse):\s?aem' AS pattern UNION ALL
  SELECT 'AI', r'(?i)\b(c3\.ai|c3 ai|c3ai|thomas siebel|c3 generative ai)\b|(nyse):\s?ai' UNION ALL
  SELECT 'APP', r'(?i)\b(applovin|machine zone|applovin corp|adam foroughi)\b|(nasdaq):\s?app' UNION ALL
  SELECT 'ARBE', r'\b(arbe|arbe robotics|arbe radar|perception radar|4d imaging radar|phoenix radar)\b|(nasdaq):\s?arbe' UNION ALL
  SELECT 'ASML', r'\b(asml|asml holding|lithography|euv|extreme ultraviolet|tsmc)\b' UNION ALL
  SELECT 'INGN', r'\b(inogen|inogen inc|inogen medical|one g3|one g4|one g5)\b|(nasdaq):\s?ingn' UNION ALL
  SELECT 'IONQ', r'(?i)\b(ionq|peter chapman|quantum computing|trapped ion|ionq aria|ionq forte)\b|(nyse):\s?ionq' UNION ALL
  SELECT 'PLTR', r'(?i)\b(palantir|alex karp|peter thiel|gotham)\b|(nyse):\s?pltr' UNION ALL
  SELECT 'QBTS', r'(?i)\b(d-wave|dwave|alan baratz|quantum annealing|advantage quantum|leap quantum)\b|(nyse):\s?qbts' UNION ALL
  SELECT 'RGTI', r'\b(rigetti|rigetti computing|rigetti quantum)\b|(nasdaq):\s?rgti' UNION ALL
  SELECT 'SMR', r'(?i)\b(nuscale|nuscale power|small modular reactor|smr|voygr)\b|(nyse):\s?smr' UNION ALL
  SELECT 'VERU', r'(?i)\b(veru|sabizabulin|enobosarm|mitchell steiner)\b|(nasdaq):\s?veru'
),

-- Step 3: Tag articles with lenient requirements
tagged_articles AS (
  SELECT
    d.ticker,
    g.url,
    g.domain,
    g.DATE,
    EXTRACT(YEAR FROM SAFE.PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(g.DATE AS STRING))) AS year,
    EXTRACT(MONTH FROM SAFE.PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(g.DATE AS STRING))) AS month,
    
    SAFE_CAST(SPLIT(g.V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tone,
    SAFE_CAST(SPLIT(g.V2Tone, ',')[OFFSET(1)] AS FLOAT64) AS positive,
    SAFE_CAST(SPLIT(g.V2Tone, ',')[OFFSET(2)] AS FLOAT64) AS negative,
    SAFE_CAST(SPLIT(g.V2Tone, ',')[OFFSET(6)] AS INT64) AS word_count,
    
    -- Evidence tracking for quality assessment
    CASE WHEN REGEXP_CONTAINS(LOWER(g.url), d.pattern) THEN 1 ELSE 0 END AS url_match,
    CASE WHEN REGEXP_CONTAINS(g.orgs, d.pattern) THEN 1 ELSE 0 END AS org_match,
    CASE WHEN REGEXP_CONTAINS(g.persons, d.pattern) THEN 1 ELSE 0 END AS person_match
    
  FROM gdelt_filtered AS g
  JOIN dictionary AS d ON REGEXP_CONTAINS(
    LOWER(CONCAT(g.url, ' ', g.orgs, ' ', g.persons)),
    d.pattern
  )
  
  WHERE 
    -- Lenient evidence requirements - need only 1 source for unpopular assets
    (CASE WHEN REGEXP_CONTAINS(LOWER(g.url), d.pattern) THEN 1 ELSE 0 END +
     CASE WHEN REGEXP_CONTAINS(g.orgs, d.pattern) THEN 1 ELSE 0 END +
     CASE WHEN REGEXP_CONTAINS(g.persons, d.pattern) THEN 1 ELSE 0 END) >= 1
    AND SAFE_CAST(SPLIT(g.V2Tone, ',')[OFFSET(6)] AS INT64) >= 10
  
  -- Deduplication
  QUALIFY ROW_NUMBER() OVER (PARTITION BY d.ticker, g.url ORDER BY g.DATE DESC) = 1
),

-- Step 4: Monthly aggregation with low thresholds
monthly_sentiment AS (
  SELECT
    ticker,
    year,
    month,
    COUNT(*) AS article_count,
    
    AVG(tone) AS tone_mean,
    STDDEV(tone) AS tone_std,
    AVG(positive) AS positive_mean,
    AVG(negative) AS negative_mean,
    
    (AVG(positive) - AVG(negative)) / 100 AS finbert_sentiment,
    
    CASE 
      WHEN COUNT(*) >= 20 THEN 'high'
      WHEN COUNT(*) >= 10 THEN 'medium'
      WHEN COUNT(*) >= 5 THEN 'low'
      ELSE 'very_low'
    END AS confidence_level,
    
    ARRAY_AGG(DISTINCT url IGNORE NULLS LIMIT 5) AS sample_urls,
    ARRAY_AGG(DISTINCT domain IGNORE NULLS LIMIT 10) AS sources_used
    
  FROM tagged_articles
  WHERE ticker IS NOT NULL
  GROUP BY ticker, year, month
  HAVING COUNT(*) >= 1  -- Accept even single-article months
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