from __future__ import annotations
import pandas as pd
from src.utils import ensure_schema

def collect(ticker_to_name: dict[str,str], quality_domains: set[str],
            start_date: str, end_date: str) -> pd.DataFrame:
    # Prefer Pushshift/Reddit API; skip for now to keep key-free.
    return ensure_schema(pd.DataFrame(columns=["ticker","date","title","url","domain","is_quality_source","source"])) 