# Enhanced Tickers Configuration
# Combines individual stocks with major market indices and commodities

# Your Specific Stock Tickers
INDIVIDUAL_TICKERS = [
    "AEM","AI","AMD","APP","ARBE","ASML","GOOG","IONQ","MRVL",
    "MSFT","MU","NVDA","PLTR","QBTS","RDDT","RGTI","SMR","VERU"
]

# Benchmark ETF
BENCHMARK_TICKERS = ["QQQ"]

# Combined list for analysis (your tickers + benchmark)
ALL_TICKERS = INDIVIDUAL_TICKERS + BENCHMARK_TICKERS

# Asset keywords mapping for Google News scraping
ASSET_KEYWORDS = {
    # Your Specific Stock Tickers
    "AEM": ["Agnico Eagle", "AEM mining", "gold mining", "precious metals mining", "Canadian mining"],
    "AI": ["artificial intelligence", "AI technology", "AI stocks", "C3.ai", "machine learning", "enterprise AI"],
    "AMD": ["AMD", "Advanced Micro Devices", "semiconductor", "processor", "CPU", "Ryzen", "EPYC"],
    "APP": ["AppLovin", "APP stock", "mobile advertising", "mobile gaming", "ad tech"],
    "ARBE": ["Arbe Robotics", "ARBE", "automotive radar", "autonomous driving", "radar technology"],
    "ASML": ["ASML", "semiconductor equipment", "lithography", "EUV", "chip manufacturing"],
    "GOOG": ["Google", "Alphabet", "GOOGL", "search engine", "YouTube", "Android", "Chrome"],
    "IONQ": ["IonQ", "quantum computing", "quantum technology", "quantum systems", "quantum processors"],
    "MRVL": ["Marvell Technology", "MRVL", "semiconductor", "data infrastructure", "storage controllers"],
    "MSFT": ["Microsoft", "MSFT", "cloud computing", "Azure", "Windows", "Office", "Xbox"],
    "MU": ["Micron Technology", "memory chips", "semiconductor", "DRAM", "NAND flash", "data storage"],
    "NVDA": ["NVIDIA", "NVDA", "graphics cards", "AI chips", "GPU", "gaming", "data center"],
    "PLTR": ["Palantir", "PLTR", "data analytics", "big data", "government contracts", "data mining"],
    "QBTS": ["D-Wave Quantum", "quantum computing", "quantum systems", "quantum annealing"],
    "RDDT": ["Reddit", "RDDT", "social media platform", "online community", "social network"],
    "RGTI": ["Rigetti Computing", "quantum computing", "quantum processors", "quantum cloud"],
    "SMR": ["NuScale Power", "SMR", "small modular reactor", "nuclear energy", "clean energy"],
    "VERU": ["Veru Inc", "pharmaceutical", "biotech", "oncology", "cancer treatment"],
    
    # Benchmark ETF
    "QQQ": ["QQQ", "NASDAQ 100", "Invesco QQQ", "tech ETF", "technology stocks", "NASDAQ ETF"],
}
# Asset categories for organized analysis
ASSET_CATEGORIES = {
    "individual_stocks": INDIVIDUAL_TICKERS,
    "benchmark": BENCHMARK_TICKERS
}
