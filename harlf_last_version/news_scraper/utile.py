import pandas as pd
from datetime import datetime

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
