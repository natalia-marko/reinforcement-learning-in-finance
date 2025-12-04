from __future__ import annotations
import hashlib, datetime as dt
from urllib.parse import urlparse
import pandas as pd

SCHEMA = ["ticker","date","title","url","domain","is_quality_source","source"]

def url_domain(url: str) -> str:
    try:
        d = urlparse(url).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()

def to_dt(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, errors="coerce", utc=True).tz_convert(None)

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    for col in SCHEMA:
        if col not in df.columns:
            df[col] = None
    return df[SCHEMA]

def dedupe_by_url(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: 
        return df
    return df.drop_duplicates(subset=["url"]).reset_index(drop=True)

def month_iter(start_date: str, end_date: str):
    s = pd.Timestamp(start_date).date().replace(day=1)
    e = pd.Timestamp(end_date).date().replace(day=1)
    cur = s
    while cur <= e:
        if cur.month == 12:
            nxt = dt.date(cur.year+1, 1, 1)
        else:
            nxt = dt.date(cur.year, cur.month+1, 1)
        yield cur, (nxt - dt.timedelta(days=1) if nxt <= pd.Timestamp(end_date).date() else pd.Timestamp(end_date).date())
        cur = nxt 