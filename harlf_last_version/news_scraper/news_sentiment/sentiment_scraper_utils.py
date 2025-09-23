"""
Utility functions for sentiment analysis scraping
Following KISS principle with clean, reusable functions
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
from urllib.parse import urlencode
from datetime import datetime, timedelta
from transformers import pipeline
import logging
from typing import Dict, List, Tuple, Optional
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentScraper:
    """Main class for handling sentiment analysis scraping"""
    
    def __init__(self, output_dir: str = "sentiment_data"):
        self.output_dir = output_dir
        self.finbert_pipeline = None
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        os.makedirs(output_dir, exist_ok=True)
        
    def load_finbert(self):
        """Load FinBERT model for sentiment analysis"""
        if self.finbert_pipeline is None:
            logger.info("Loading FinBERT model...")
            self.finbert_pipeline = pipeline("text-classification", model="ProsusAI/finbert")
            logger.info("FinBERT model loaded successfully")
    
    def generate_google_news_url(self, query: str, start_date: str, end_date: str) -> str:
        """Generate Google News search URL with date filter"""
        base_url = "https://www.google.com/search"
        params = {
            "q": query,
            "tbs": f"cdr:1,cd_min:{start_date},cd_max:{end_date}",
            "tbm": "nws"
        }
        return f"{base_url}?{urlencode(params)}"
    
    def scrape_article_links(self, search_url: str, max_links: int = 10) -> List[str]:
        """Scrape article links from Google News search results"""
        try:
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            links = []
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if "https://" in href and "google.com" not in href and len(links) < max_links:
                    links.append(href)
            
            return list(set(links))[:max_links]  # Remove duplicates and limit
            
        except Exception as e:
            logger.error(f"Failed to scrape links from {search_url}: {e}")
            return []
    
    def extract_article_content(self, url: str) -> Tuple[str, str]:
        """Extract title and content from article URL"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else "No title"
            title = title.strip()
            
            # Extract article text
            paragraphs = soup.find_all('p')
            article_text = " ".join([p.get_text().strip() for p in paragraphs])
            
            # Limit to first 500 words
            words = article_text.split()[:500]
            content = " ".join(words)
            
            return title, content
            
        except Exception as e:
            logger.error(f"Failed to extract content from {url}: {e}")
            return "Error", "Error"
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """Analyze sentiment using FinBERT"""
        if not self.finbert_pipeline:
            self.load_finbert()
        
        try:
            if not isinstance(text, str) or text.strip() == "" or text == "Error":
                return "neutral", 0.0
            
            # Truncate text if too long
            text = text[:512]
            result = self.finbert_pipeline(text)[0]
            return result['label'], result['score']
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return "neutral", 0.0
    
    def calculate_asset_sentiment(self, sentiment_data: List[Tuple[str, float]]) -> float:
        """Calculate asset sentiment score: (P_positive - P_negative) / N"""
        if not sentiment_data:
            return 0.0
        
        positive_scores = [score for label, score in sentiment_data if label == 'positive']
        negative_scores = [score for label, score in sentiment_data if label == 'negative']
        
        sum_positive = sum(positive_scores)
        sum_negative = sum(negative_scores)
        total_articles = len(sentiment_data)
        
        if total_articles == 0:
            return 0.0
        
        return (sum_positive - sum_negative) / total_articles
    
    def scrape_monthly_sentiment(self, asset_name: str, terms: List[str], 
                                year: int, month: int) -> float:
        """Scrape sentiment for an asset in a specific month"""
        logger.info(f"Processing {asset_name} for {month:02d}/{year}")
        
        # Calculate date range for the month
        start_date = f"{month}/1/{year}"
        next_month = month % 12 + 1
        next_year = year if month < 12 else year + 1
        end_date = (datetime(next_year, next_month, 1) - timedelta(days=1)).strftime("%m/%d/%Y")
        
        all_sentiment_data = []
        
        for term in terms:
            logger.info(f"  Searching for term: {term}")
            
            # Generate search URL
            search_url = self.generate_google_news_url(term, start_date, end_date)
            
            # Get article links
            article_links = self.scrape_article_links(search_url, max_links=10)
            logger.info(f"    Found {len(article_links)} articles")
            
            # Process each article
            for url in article_links:
                title, content = self.extract_article_content(url)
                if content != "Error":
                    label, score = self.analyze_sentiment(content)
                    all_sentiment_data.append((label, score))
            
            # Rate limiting
            time.sleep(random.uniform(2, 4))
        
        # Calculate final sentiment score
        sentiment_score = self.calculate_asset_sentiment(all_sentiment_data)
        logger.info(f"  Final sentiment score: {sentiment_score:.4f}")
        
        return sentiment_score
    
    def save_monthly_results(self, results: Dict, year: int, month: int):
        """Save monthly sentiment results to CSV"""
        month_name = datetime(year, month, 1).strftime("%B")
        output_path = os.path.join(self.output_dir, f"{year}_{month:02d}_{month_name}_sentiment.csv")
        
        df = pd.DataFrame([
            {"Asset": asset, "Sentiment_Score": score} 
            for asset, score in results.items()
        ])
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")


def get_asset_definitions() -> Dict[str, List[str]]:
    """Define asset terms for searching"""
    return {
        "SP_500": ["S&P 500", "SP 500", "SPX", "S&P 500 Index", "Standard and Poor's 500"],
        "NASDAQ": ["NASDAQ", "NASDAQ Composite", "NASDAQ Index", "Nasdaq 100"],
        "Dow_Jones": ["Dow Jones", "DJIA", "Dow Jones Industrial Average"],
        "CAC_40": ["CAC 40", "Paris Stock Exchange", "French Stock Market"],
        "FTSE_100": ["FTSE 100", "London Stock Exchange", "UK Stock Market"],
        "EuroStoxx_50": ["EuroStoxx 50", "European Stock Market", "Eurozone Stocks"],
        "Nikkei_225": ["Nikkei 225", "Japanese Stock Market", "Tokyo Stock Exchange"],
        "Hang_Seng": ["Hang Seng", "Hong Kong Stock Market", "HSI"],
        "Shanghai_Composite": ["Shanghai Composite", "Chinese Stock Market", "Shanghai Index"],
        "Gold": ["Gold", "Gold Price", "Gold Market", "Precious Metals"],
        "Silver": ["Silver", "Silver Price", "Silver Market"],
        "Oil": ["Oil", "Crude Oil", "WTI", "Brent Crude", "Oil Price"]
    }
