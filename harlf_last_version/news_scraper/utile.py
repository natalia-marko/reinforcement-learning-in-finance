import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import ttest_ind, spearmanr, pearsonr

def analyze_data_coverage(sentiment_data):
    """
    Analyze data coverage for each ticker in the sentiment dataset.
    
    Args:
        sentiment_data: DataFrame with columns ['ticker', 'year', 'month', 'article_count']
    
    Returns:
        DataFrame with coverage statistics for each ticker
    """
    # Create a proper year_month column for accurate date handling
    sentiment_data = sentiment_data.copy()
    sentiment_data['year_month'] = sentiment_data['year'].astype(str) + '-' + sentiment_data['month'].astype(str).str.zfill(2)
    
    print(f"Data range: {sentiment_data['year_month'].min()} to {sentiment_data['year_month'].max()}")
    
    # Get min and max dates from the year_month column
    min_year_month = sentiment_data['year_month'].min()
    max_year_month = sentiment_data['year_month'].max()
    
    # Parse the year_month strings to get year and month values
    min_year, min_month = map(int, min_year_month.split('-'))
    max_year, max_month = map(int, max_year_month.split('-'))
    
    # Generate all possible year-month combinations in the data range
    start_date = datetime(min_year, min_month, 1)
    end_date = datetime(max_year, max_month, 1)
    
    all_months = []
    current = start_date
    while current <= end_date:
        all_months.append(current.strftime('%Y-%m'))
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    print(f"Expected total months in range: {len(all_months)}")
    
    # Analyze missing data by ticker
    missing_analysis = []
    for ticker in sentiment_data['ticker'].unique():
        ticker_data = sentiment_data[sentiment_data['ticker'] == ticker]
        
        # Get actual months for this ticker
        actual_months = set(ticker_data['year_month'])
        expected_months = set(all_months)
        missing_months = expected_months - actual_months
        
        # Calculate months since first record for this ticker
        first_year_month = ticker_data['year_month'].min()
        last_year_month = ticker_data['year_month'].max()
        
        first_year, first_month = map(int, first_year_month.split('-'))
        last_year, last_month = map(int, last_year_month.split('-'))
        
        # Calculate months since first record
        months_since_first = ((last_year - first_year) * 12) + (last_month - first_month) + 1
        
        missing_analysis.append({
            'ticker': ticker,
            'total_months_available': len(actual_months),
            'coverage_percentage_per_existed': (len(actual_months) / months_since_first) * 100,
            'months_since_first': months_since_first,
            'first_record': first_year_month,
            'last_record': last_year_month,
            'total_articles': ticker_data['article_count'].sum()
        })
    
    # Create DataFrame and sort by coverage percentage
    missing_df = pd.DataFrame(missing_analysis)
    missing_df = missing_df.sort_values('coverage_percentage_per_existed', ascending=False).reset_index(drop=True)
    
    print(f"\nData Coverage by Ticker:")
    return missing_df.round(1)


def analyze_missing_months_detailed(df):
    df = df.copy()
    df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
    df = df[['ticker', 'year_month', 'article_count']]
    missing_data = []
    
    for ticker in df['ticker'].unique():
        # Get data for this specific ticker
        ticker_data = df[df['ticker'] == ticker]
        
        # Get first and last record dates for this ticker
        first_year_month = ticker_data['year_month'].min()
        last_year_month = ticker_data['year_month'].max()
        
        # Extract year and month from ticker's date range
        first_year, first_month = map(int, first_year_month.split('-'))
        last_year, last_month = map(int, last_year_month.split('-'))
        
        # Create all expected months for this ticker's range
        expected_months = []
        for year in range(first_year, last_year + 1):
            start_month = first_month if year == first_year else 1
            end_month = last_month if year == last_year else 12
            
            for month in range(start_month, end_month + 1):
                expected_months.append(f"{year}-{month:02d}")
        
        # Get existing months for this ticker
        existing_months = set(ticker_data['year_month'])
        
        # Find missing months within this ticker's date range
        for year_month in expected_months:
            if year_month not in existing_months:
                year, month = map(int, year_month.split('-'))
                missing_data.append({
                    'ticker': ticker,
                    'year': year,
                    'month': month,
                    'year_month': year_month
                })
    
    # Create DataFrame and sort by ticker, then by year_month
    missing_df = pd.DataFrame(missing_data)
    if not missing_df.empty:
        missing_df = missing_df.sort_values(['ticker', 'year_month']).reset_index(drop=True)
    
    # Print summary and return DataFrame
    print(f"Total missing month-ticker combinations: {len(missing_df)}")
    print(f"Sample missing data:")
    
    return missing_df


# IPO dates dictionary for portfolio tickers
IPO_DATES = {
    'aem':  '1978-10',     # Agnico Eagle Mines: Oct 1978 (Toronto; US ADR since 1996)
    'rddt': '2024-03',     # Reddit IPO: March 2024
    'nvda': '1999-01',     # NVIDIA IPO: Jan 1999
    'smr':  '2022-05',     # NuScale Power (SMR): May 2022
    'rgti': '2022-03',     # Rigetti Computing: March 2022 (SPAC close)
    'mu':   '1984-06',     # Micron IPO: June 1984
    'amd':  '1979-09',     # AMD IPO: September 1979
    'app':  '2021-04',     # Applovin IPO: April 2021
    'mrvl': '2000-06',     # Marvell IPO: June 2000
    'msft': '1986-03',     # Microsoft IPO: March 1986
    'qbts': '2022-08',     # D-Wave (QBTS): August 2022
    'pltr': '2020-10',     # Palantir: October 2020
    'asml': '1995-03',     # ASML: March 1995
    'goog': '2004-08',     # Google IPO: August 2004
    'ionq': '2021-10',     # IonQ (SPAC): October 2021
    'ai':   '2020-12',     # C3.ai IPO: December 2020
    'arbe': '2021-10',     # Arbe Robotics: October 2021
    'veru': '1999-04',     # Veru Inc.: April 1999 (as "Urologix", later renamed, but tradable)
    'qqq':  '1999-03',     # QQQ ETF: March 1999
}


def filter_zero_coverage_post_ipo(coverage_binary, ipo_dates):
    """
    Filter zero coverage data to only include months after IPO date.
    
    Args:
        coverage_binary (pd.DataFrame): Binary coverage matrix
        ipo_dates (dict): Dictionary mapping tickers to IPO dates
    
    Returns:
        pd.DataFrame: Filtered zero coverage data
    """
    # Stack zero-coverage locations
    zero_coverage_long = (coverage_binary == 0).stack().reset_index()
    zero_coverage_long.columns = ['ticker', 'date', 'zero_coverage']
    zero_coverage_long = zero_coverage_long[zero_coverage_long['zero_coverage'] == True]
    
    # Filter zero coverage to only include months after IPO date
    filtered_zero_coverage = []
    
    for _, row in zero_coverage_long.iterrows():
        ticker = row['ticker']
        date = row['date']
        
        # Get IPO date for this ticker
        ipo_month = ipo_dates.get(ticker)
        if ipo_month is None:
            continue
        
        # Convert both dates to comparable format for accurate comparison
        try:
            # Convert date to YYYY-MM format for comparison
            if isinstance(date, str):
                date_formatted = date
            else:
                date_formatted = date.strftime('%Y-%m')
            
            # Only include zero coverage if date >= IPO date
            if date_formatted >= ipo_month:
                filtered_zero_coverage.append({
                    'ticker': ticker,
                    'date': date,
                    'zero_coverage': True
                })
        except Exception as e:
            print(f"Error processing date {date} for ticker {ticker}: {str(e)}")
            continue
    
    return pd.DataFrame(filtered_zero_coverage)


def calculate_correlations(df, sentiment_cols, price_col='price_variance'):
    """
    Calculate correlations between sentiment measures and price variance.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment and price data
        sentiment_cols (list): List of sentiment column names
        price_col (str): Name of price variance column
    
    Returns:
        dict: Dictionary with correlation results
    """
    results = {}
    
    for col in sentiment_cols:
        if col in df.columns:
            # Clean data for this column
            clean_data = df.dropna(subset=[col, price_col])
            
            if len(clean_data) > 0:
                # Pearson correlation
                pearson_corr, pearson_p = pearsonr(clean_data[col], clean_data[price_col])
                
                # Spearman correlation
                spearman_corr, spearman_p = spearmanr(clean_data[col], clean_data[price_col])
                
                results[col] = {
                    'pearson_corr': pearson_corr,
                    'pearson_p': pearson_p,
                    'spearman_corr': spearman_corr,
                    'spearman_p': spearman_p,
                    'r_squared': pearson_corr**2,
                    'n_observations': len(clean_data)
                }
    
    return results


def create_sentiment_intensity_categories(df, pos_col='mean_positive', neg_col='mean_negative'):
    """
    Create sentiment intensity categories based on quantiles.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment data
        pos_col (str): Name of positive sentiment column
        neg_col (str): Name of negative sentiment column
    
    Returns:
        pd.DataFrame: DataFrame with intensity categories added
    """
    df = df.copy()
    
    # Use quantiles for more data-driven binning
    pos_quantiles = df[pos_col].quantile([0.25, 0.5, 0.75]).values
    neg_quantiles = df[neg_col].quantile([0.25, 0.5, 0.75]).values
    
    # Create intensity categories
    df['pos_intensity'] = pd.cut(df[pos_col], 
                                bins=[0] + list(pos_quantiles) + [float('inf')], 
                                labels=['Low', 'Medium', 'High', 'Very High'])
    
    df['neg_intensity'] = pd.cut(df[neg_col], 
                                bins=[0] + list(neg_quantiles) + [float('inf')], 
                                labels=['Low', 'Medium', 'High', 'Very High'])
    
    return df


def analyze_asymmetric_effects(df, pos_col='mean_positive', neg_col='mean_negative', 
                             price_col='price_variance', threshold=0.75):
    """
    Analyze asymmetric effects of positive vs negative sentiment on price variance.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment and price data
        pos_col (str): Name of positive sentiment column
        neg_col (str): Name of negative sentiment column
        price_col (str): Name of price variance column
        threshold (float): Quantile threshold for high sentiment periods
    
    Returns:
        dict: Analysis results
    """
    # Get high sentiment thresholds
    high_pos_threshold = df[pos_col].quantile(threshold)
    high_neg_threshold = df[neg_col].quantile(threshold)
    
    # Get high sentiment periods
    high_pos_periods = df[df[pos_col] > high_pos_threshold][price_col]
    high_neg_periods = df[df[neg_col] > high_neg_threshold][price_col]
    
    # Calculate statistics
    results = {
        'high_pos_count': len(high_pos_periods),
        'high_neg_count': len(high_neg_periods),
        'high_pos_mean': high_pos_periods.mean(),
        'high_neg_mean': high_neg_periods.mean(),
        'ratio': high_neg_periods.mean() / high_pos_periods.mean() if high_pos_periods.mean() > 0 else np.nan
    }
    
    # Statistical test
    if len(high_pos_periods) > 1 and len(high_neg_periods) > 1:
        t_stat, t_p = ttest_ind(high_neg_periods, high_pos_periods)
        results.update({
            't_stat': t_stat,
            't_p': t_p,
            'significant': t_p < 0.05
        })
    
    return results
