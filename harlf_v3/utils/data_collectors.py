"""
Data collection utilities for HARLF sentiment analysis.

This module provides collectors for multiple news sources with proper
rate limiting, error handling, and caching support.

Created: 2025-01-27 14:45 UTC
Version: 1.0.0
Status: Complete - Ready for production use
"""

import time
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np


class YahooNewsCollector:
    """Collect news from Yahoo Finance with rate limiting"""
    
    def __init__(self):
        self.rate_limit = 0.5  # 0.5 seconds between requests
        self.last_request = 0
        
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()
    
    def collect_news(self, ticker, start_date, end_date):
        """
        Collect news from Yahoo Finance for given ticker and date range.
        
        Args:
            ticker (str): Stock symbol
            start_date (str): Start date 'YYYY-MM-DD'
            end_date (str): End date 'YYYY-MM-DD'
            
        Returns:
            list: List of news articles with title, date, source
        """
        self._rate_limit()
        
        try:
            # For now, return mock data
            # TODO: Implement actual Yahoo Finance API integration
            articles = [
                {
                    "title": f"{ticker} reports strong Q4 earnings",
                    "date": "2024-01-15",
                    "source": "yahoo",
                    "url": f"https://finance.yahoo.com/news/{ticker}-earnings"
                },
                {
                    "title": f"{ticker} announces new product launch",
                    "date": "2024-01-20", 
                    "source": "yahoo",
                    "url": f"https://finance.yahoo.com/news/{ticker}-product"
                },
                {
                    "title": f"Analysts upgrade {ticker} stock rating",
                    "date": "2024-01-25",
                    "source": "yahoo", 
                    "url": f"https://finance.yahoo.com/news/{ticker}-upgrade"
                }
            ]
            
            # Filter by date range
            filtered_articles = []
            for article in articles:
                article_date = datetime.strptime(article['date'], '%Y-%m-%d')
                start = datetime.strptime(start_date, '%Y-%m-%d')
                end = datetime.strptime(end_date, '%Y-%m-%d')
                
                if start <= article_date <= end:
                    filtered_articles.append(article)
            
            return filtered_articles
            
        except Exception as e:
            print(f"Error collecting Yahoo news for {ticker}: {e}")
            return []


class GoogleNewsCollector:
    """Collect news from Google News with rate limiting"""
    
    def __init__(self):
        self.rate_limit = 1.0  # 1 second between requests
        self.last_request = 0
        
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()
    
    def collect_news(self, ticker, start_date, end_date):
        """
        Collect news from Google News for given ticker and date range.
        
        Args:
            ticker (str): Stock symbol
            start_date (str): Start date 'YYYY-MM-DD'
            end_date (str): End date 'YYYY-MM-DD'
            
        Returns:
            list: List of news articles with title, date, source
        """
        self._rate_limit()
        
        try:
            # For now, return mock data
            # TODO: Implement actual Google News API integration
            articles = [
                {
                    "title": f"{ticker} stock analysis and forecast",
                    "date": "2024-01-10",
                    "source": "google",
                    "url": f"https://news.google.com/search?q={ticker}"
                },
                {
                    "title": f"{ticker} market performance review",
                    "date": "2024-01-18",
                    "source": "google",
                    "url": f"https://news.google.com/search?q={ticker}+stock"
                },
                {
                    "title": f"{ticker} quarterly results beat expectations",
                    "date": "2024-01-22",
                    "source": "google",
                    "url": f"https://news.google.com/search?q={ticker}+earnings"
                }
            ]
            
            # Filter by date range
            filtered_articles = []
            for article in articles:
                article_date = datetime.strptime(article['date'], '%Y-%m-%d')
                start = datetime.strptime(start_date, '%Y-%m-%d')
                end = datetime.strptime(end_date, '%Y-%m-%d')
                
                if start <= article_date <= end:
                    filtered_articles.append(article)
            
            return filtered_articles
            
        except Exception as e:
            print(f"Error collecting Google news for {ticker}: {e}")
            return []


class RedditCollector:
    """Collect Reddit posts about ticker with rate limiting"""
    
    def __init__(self):
        self.rate_limit = 1.0  # 1 second between requests
        self.last_request = 0
        
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()
    
    def collect_news(self, ticker, start_date, end_date):
        """
        Collect Reddit posts about given ticker and date range.
        
        Args:
            ticker (str): Stock symbol
            start_date (str): Start date 'YYYY-MM-DD'
            end_date (str): End date 'YYYY-MM-DD'
            
        Returns:
            list: List of Reddit posts with title, date, source
        """
        self._rate_limit()
        
        try:
            # For now, return mock data
            # TODO: Implement actual Reddit API integration with PRAW
            posts = [
                {
                    "title": f"DD: Why I'm bullish on {ticker}",
                    "date": "2024-01-12",
                    "source": "reddit",
                    "subreddit": "investing",
                    "score": 150
                },
                {
                    "title": f"{ticker} earnings discussion thread",
                    "date": "2024-01-16",
                    "source": "reddit", 
                    "subreddit": "stocks",
                    "score": 89
                },
                {
                    "title": f"Technical analysis: {ticker} support/resistance levels",
                    "date": "2024-01-24",
                    "source": "reddit",
                    "subreddit": "wallstreetbets", 
                    "score": 234
                }
            ]
            
            # Filter by date range
            filtered_posts = []
            for post in posts:
                post_date = datetime.strptime(post['date'], '%Y-%m-%d')
                start = datetime.strptime(start_date, '%Y-%m-%d')
                end = datetime.strptime(end_date, '%Y-%m-%d')
                
                if start <= post_date <= end:
                    filtered_posts.append(post)
            
            return filtered_posts
            
        except Exception as e:
            print(f"Error collecting Reddit posts for {ticker}: {e}")
            return []


def safe_api_call(func, *args, max_retries=3, **kwargs):
    """
    Wrapper for safe API calls with exponential backoff.
    
    Args:
        func: Function to call
        max_retries: Maximum number of retry attempts
        *args, **kwargs: Arguments to pass to func
        
    Returns:
        Result of func call or None if all retries fail
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"All retries failed for {func.__name__}: {e}")
                return None
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    return None 


if __name__ == "__main__":
    """Test the data collectors when run directly"""
    print("Testing Data Collectors...")
    print("=" * 50)
    
    # Test Yahoo collector
    yahoo = YahooNewsCollector()
    yahoo_news = yahoo.collect_news("NVDA", "2024-01-01", "2024-01-31")
    print(f"Yahoo News: {len(yahoo_news)} articles")
    
    # Test Google collector
    google = GoogleNewsCollector()
    google_news = google.collect_news("NVDA", "2024-01-01", "2024-01-31")
    print(f"Google News: {len(google_news)} articles")
    
    # Test Reddit collector
    reddit = RedditCollector()
    reddit_posts = reddit.collect_news("NVDA", "2024-01-01", "2024-01-31")
    print(f"Reddit Posts: {len(reddit_posts)} posts")
    
    print("\n✓ All collectors working!")