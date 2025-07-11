"""
Cached API Wrappers for Financial Data
Integrates the TimeSeriesCache with existing financial APIs
Enhanced with batch processing and pre-fetching optimizations
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from .time_series_cache import (
    get_cache, DataType, 
    fetch_ohlcv_with_cache, fetch_news_with_cache, fetch_fundamentals_with_cache
)
from .interface import get_data_in_range
from .googlenews_utils import getNewsData
from .config import get_config, DATA_DIR
from .financialdatasets_market_data import (
    FinancialDatasetsClient,
    get_financialdatasets_historical_prices,
    get_financialdatasets_company_news,
    get_financialdatasets_financials
)

logger = logging.getLogger(__name__)


# ===== BATCH API OPTIMIZATION =====

def batch_fetch_multiple_symbols(symbols: List[str], start_date: datetime, end_date: datetime, 
                                 data_type: str = "ohlcv", max_workers: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple symbols in parallel using ThreadPoolExecutor
    
    Args:
        symbols: List of stock ticker symbols
        start_date: Start date for data
        end_date: End date for data
        data_type: Type of data to fetch ('ohlcv', 'news', 'fundamentals')
        max_workers: Maximum number of concurrent API calls
    
    Returns:
        Dictionary mapping symbols to their DataFrames
    """
    
    def fetch_symbol_data(symbol: str) -> tuple:
        """Fetch data for a single symbol"""
        try:
            if data_type == "ohlcv":
                data = fetch_financialdatasets_prices_cached(symbol, start_date, end_date)
            elif data_type == "news":
                data = fetch_financialdatasets_news_cached(symbol, start_date, end_date)
            elif data_type == "fundamentals":
                data = fetch_financialdatasets_financials_cached(symbol, period='annual')
            else:
                logger.warning(f"Unsupported data type: {data_type}")
                return symbol, pd.DataFrame()
            
            logger.info(f"✅ Batch fetch completed for {symbol}: {len(data)} records")
            return symbol, data
            
        except Exception as e:
            logger.error(f"❌ Batch fetch failed for {symbol}: {e}")
            return symbol, pd.DataFrame()
    
    # Use ThreadPoolExecutor for parallel API calls
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {executor.submit(fetch_symbol_data, symbol): symbol for symbol in symbols}
        
        # Collect results as they complete
        for future in as_completed(future_to_symbol):
            symbol, data = future.result()
            results[symbol] = data
    
    logger.info(f"📊 Batch fetch completed for {len(symbols)} symbols, {data_type} data")
    return results


def prefetch_common_data(symbols: List[str], analysis_dates: List[datetime], 
                        data_types: List[str] = ["ohlcv", "news"]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Pre-fetch commonly used data for multiple symbols and dates
    
    Args:
        symbols: List of stock symbols
        analysis_dates: List of dates for analysis
        data_types: List of data types to prefetch
    
    Returns:
        Nested dictionary: {symbol: {data_type: DataFrame}}
    """
    
    prefetched_data = {}
    
    # Determine date range for prefetching
    min_date = min(analysis_dates) - timedelta(days=30)  # 30 days buffer
    max_date = max(analysis_dates) + timedelta(days=1)
    
    logger.info(f"🔄 Pre-fetching data for {len(symbols)} symbols, {len(data_types)} data types")
    logger.info(f"📅 Date range: {min_date.date()} to {max_date.date()}")
    
    for data_type in data_types:
        logger.info(f"📊 Pre-fetching {data_type} data...")
        batch_results = batch_fetch_multiple_symbols(symbols, min_date, max_date, data_type)
        
        for symbol, data in batch_results.items():
            if symbol not in prefetched_data:
                prefetched_data[symbol] = {}
            prefetched_data[symbol][data_type] = data
    
    logger.info(f"✅ Pre-fetching completed for {len(symbols)} symbols")
    return prefetched_data


# ===== OPTIMIZED API WRAPPERS =====

def fetch_financialdatasets_prices_cached_optimized(symbol: str, start_date: datetime, end_date: datetime, 
                                                   interval: str = 'day', use_cache_only: bool = False) -> pd.DataFrame:
    """
    Optimized version of financialdatasets price fetching with additional performance features
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        interval: Time interval ('day', 'minute', etc.)
        use_cache_only: If True, only return cached data without API calls
    
    Returns:
        DataFrame with OHLCV data
    """
    
    # Check cache first
    cache = get_cache()
    cached_data = cache.get_cached_data(symbol, DataType.OHLCV, start_date, end_date)
    
    if cached_data is not None:
        logger.info(f"🎯 Cache hit for {symbol} OHLCV data")
        return cached_data
    
    if use_cache_only:
        logger.info(f"⚡ Cache-only mode: No API call for {symbol}")
        return pd.DataFrame()
    
    # Fallback to standard cached fetch
    return fetch_financialdatasets_prices_cached(symbol, start_date, end_date, interval)


# ===== ORIGINAL API WRAPPERS (Enhanced) =====

# YFinance OHLCV Data Caching
def fetch_yfinance_data_cached(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch YFinance OHLCV data with intelligent caching
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
    
    Returns:
        DataFrame with OHLCV data
    """
    
    def _fetch_yfinance_api(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Internal function to fetch from YFinance API"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Add one day to end_date to make it inclusive
            end_date_inclusive = end_date + timedelta(days=1)
            
            data = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date_inclusive.strftime('%Y-%m-%d'),
                auto_adjust=True,
                progress=False
            )
            
            if data.empty:
                logger.warning(f"No YFinance data found for {symbol} from {start_date.date()} to {end_date.date()}")
                return pd.DataFrame()
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Standardize column names and add date column
            data['date'] = data['Date']
            data['symbol'] = symbol
            
            # Round numeric columns
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if col in data.columns:
                    data[col] = data[col].round(4)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch YFinance data for {symbol}: {e}")
            return pd.DataFrame()
    
    return fetch_ohlcv_with_cache(symbol, start_date, end_date, _fetch_yfinance_api)


def fetch_yfinance_window_cached(symbol: str, curr_date: datetime, look_back_days: int) -> pd.DataFrame:
    """
    Fetch YFinance data for a window of days before current date with caching
    
    Args:
        symbol: Stock ticker symbol
        curr_date: Current/end date
        look_back_days: Number of days to look back
    
    Returns:
        DataFrame with OHLCV data
    """
    start_date = curr_date - timedelta(days=look_back_days)
    return fetch_yfinance_data_cached(symbol, start_date, curr_date)


# News Data Caching
def fetch_finnhub_news_cached(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch Finnhub news data with caching
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date for news
        end_date: End date for news
    
    Returns:
        DataFrame with news data
    """
    
    def _fetch_finnhub_news_api(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Internal function to fetch Finnhub news from cached files"""
        try:
            # Use existing get_data_in_range function
            data = get_data_in_range(
                symbol, 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d'), 
                "news_data", 
                DATA_DIR
            )
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame format
            news_records = []
            for date_str, news_list in data.items():
                for news_item in news_list:
                    record = {
                        'date': pd.to_datetime(date_str),
                        'symbol': symbol,
                        'headline': news_item.get('headline', ''),
                        'summary': news_item.get('summary', ''),
                        'source': news_item.get('source', ''),
                        'url': news_item.get('url', ''),
                        'datetime': pd.to_datetime(news_item.get('datetime', date_str))
                    }
                    news_records.append(record)
            
            return pd.DataFrame(news_records)
            
        except Exception as e:
            logger.error(f"Failed to fetch Finnhub news for {symbol}: {e}")
            return pd.DataFrame()
    
    return fetch_news_with_cache(symbol, start_date, end_date, _fetch_finnhub_news_api)


def fetch_google_news_cached(query: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch Google News data with caching
    
    Args:
        query: Search query
        start_date: Start date for news
        end_date: End date for news
    
    Returns:
        DataFrame with news data
    """
    
    def _fetch_google_news_api(query: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Internal function to fetch from Google News API"""
        try:
            query_formatted = query.replace(" ", "+")
            news_results = getNewsData(
                query_formatted, 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d')
            )
            
            if not news_results:
                return pd.DataFrame()
            
            # Convert to DataFrame
            news_records = []
            for news_item in news_results:
                record = {
                    'date': pd.to_datetime(news_item.get('date', start_date)),
                    'query': query,
                    'title': news_item.get('title', ''),
                    'snippet': news_item.get('snippet', ''),
                    'source': news_item.get('source', ''),
                    'url': news_item.get('url', ''),
                    'published': pd.to_datetime(news_item.get('published', start_date))
                }
                news_records.append(record)
            
            return pd.DataFrame(news_records)
            
        except Exception as e:
            logger.error(f"Failed to fetch Google News for query '{query}': {e}")
            return pd.DataFrame()
    
    return fetch_news_with_cache(query, start_date, end_date, _fetch_google_news_api)


# Technical Indicators Caching
def fetch_technical_indicators_cached(symbol: str, indicator: str, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
    """
    Fetch technical indicators with caching
    
    Args:
        symbol: Stock ticker symbol
        indicator: Technical indicator name
        start_date: Start date
        end_date: End date
        **kwargs: Additional parameters for indicator calculation
    
    Returns:
        DataFrame with indicator data
    """
    
    def _fetch_indicator_api(symbol: str, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """Internal function to calculate technical indicators"""
        try:
            from .stockstats_utils import StockstatsUtils
            
            # First get the underlying price data
            price_data = fetch_yfinance_data_cached(symbol, start_date, end_date)
            
            if price_data.empty:
                return pd.DataFrame()
            
            # Calculate indicator for each date
            indicator_records = []
            for _, row in price_data.iterrows():
                try:
                    curr_date = row['date'].strftime('%Y-%m-%d')
                    indicator_value = StockstatsUtils.get_stock_stats(
                        symbol,
                        indicator,
                        curr_date,
                        DATA_DIR,
                        online=True
                    )
                    
                    record = {
                        'date': row['date'],
                        'symbol': symbol,
                        'indicator': indicator,
                        'value': float(indicator_value) if indicator_value else None,
                        **kwargs
                    }
                    indicator_records.append(record)
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate {indicator} for {symbol} on {curr_date}: {e}")
                    continue
            
            return pd.DataFrame(indicator_records)
            
        except Exception as e:
            logger.error(f"Failed to fetch indicators for {symbol}: {e}")
            return pd.DataFrame()
    
    cache = get_cache()
    return cache.fetch_with_cache(symbol, DataType.INDICATORS, start_date, end_date, _fetch_indicator_api, indicator=indicator, **kwargs)


# Insider Trading Data Caching
def fetch_insider_data_cached(symbol: str, start_date: datetime, end_date: datetime, data_type: str = "insider_trans") -> pd.DataFrame:
    """
    Fetch insider trading data with caching
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date
        end_date: End date
        data_type: Type of insider data ('insider_trans' or 'insider_senti')
    
    Returns:
        DataFrame with insider data
    """
    
    def _fetch_insider_api(symbol: str, start_date: datetime, end_date: datetime, data_type: str = "insider_trans") -> pd.DataFrame:
        """Internal function to fetch insider data"""
        try:
            data = get_data_in_range(
                symbol,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                data_type,
                DATA_DIR
            )
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            records = []
            for date_str, items in data.items():
                for item in items:
                    record = {
                        'date': pd.to_datetime(date_str),
                        'symbol': symbol,
                        'data_type': data_type,
                        **item  # Include all fields from the insider data
                    }
                    records.append(record)
            
            return pd.DataFrame(records)
            
        except Exception as e:
            logger.error(f"Failed to fetch insider data for {symbol}: {e}")
            return pd.DataFrame()
    
    cache = get_cache()
    cache_data_type = DataType.INSIDER if data_type == "insider_trans" else DataType.SENTIMENT
    return cache.fetch_with_cache(symbol, cache_data_type, start_date, end_date, _fetch_insider_api, data_type=data_type)


# Convenience Functions for Integration
def get_cached_price_data(symbol: str, start_date: str, end_date: str) -> str:
    """
    Get cached price data in string format (compatible with existing interface)
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        Formatted string with price data
    """
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        df = fetch_yfinance_data_cached(symbol, start_dt, end_dt)
        
        if df.empty:
            return f"No data found for {symbol} between {start_date} and {end_date}"
        
        # Format similar to existing interface
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            df_string = df.to_string(index=False)
        
        return f"## Cached Market Data for {symbol} from {start_date} to {end_date}:\n\n{df_string}"
        
    except Exception as e:
        logger.error(f"Failed to get cached price data: {e}")
        return f"Error retrieving cached data for {symbol}: {e}"


def get_cached_news_data(symbol: str, curr_date: str, look_back_days: int = 7) -> str:
    """
    Get cached news data in string format (compatible with existing interface)
    
    Args:
        symbol: Stock ticker symbol
        curr_date: Current date in 'YYYY-MM-DD' format
        look_back_days: Number of days to look back
    
    Returns:
        Formatted string with news data
    """
    try:
        curr_dt = datetime.strptime(curr_date, '%Y-%m-%d')
        start_dt = curr_dt - timedelta(days=look_back_days)
        
        df = fetch_finnhub_news_cached(symbol, start_dt, curr_dt)
        
        if df.empty:
            return f"No cached news found for {symbol}"
        
        # Format similar to existing interface
        news_str = ""
        for _, row in df.iterrows():
            news_str += f"### {row['headline']} ({row['date'].strftime('%Y-%m-%d')})\n{row['summary']}\n\n"
        
        return f"## {symbol} Cached News, from {start_dt.strftime('%Y-%m-%d')} to {curr_date}:\n{news_str}"
        
    except Exception as e:
        logger.error(f"Failed to get cached news data: {e}")
        return f"Error retrieving cached news for {symbol}: {e}"


# ===== FINANCIALDATASETS.AI CACHED WRAPPERS =====

def fetch_financialdatasets_prices_cached(symbol: str, start_date: datetime, end_date: datetime, interval: str = 'day') -> pd.DataFrame:
    """
    Fetch financialdatasets.ai OHLCV data with intelligent caching
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        interval: Time interval ('day', 'minute', etc.)
    
    Returns:
        DataFrame with OHLCV data
    """
    
    def _fetch_financialdatasets_api(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Internal function to fetch from financialdatasets.ai API"""
        try:
            client = FinancialDatasetsClient()
            
            data = client.get_historical_prices(
                ticker=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                interval=interval
            )
            
            if data.empty:
                logger.warning(f"No financialdatasets.ai data found for {symbol}")
                return pd.DataFrame()
            
            # Reset index to make time a column, add symbol
            data = data.reset_index()
            data['symbol'] = symbol
            
            # Rename 'time' to 'date' for consistency
            if 'time' in data.columns:
                data['date'] = data['time']
            
            # Round numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in data.columns:
                    data[col] = data[col].round(4)
            
            logger.info(f"Retrieved {len(data)} records from financialdatasets.ai for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch financialdatasets.ai data for {symbol}: {e}")
            return pd.DataFrame()
    
    return fetch_ohlcv_with_cache(symbol, start_date, end_date, _fetch_financialdatasets_api)


def fetch_financialdatasets_news_cached(symbol: str, start_date: datetime, end_date: datetime, limit: int = 100) -> pd.DataFrame:
    """
    Fetch financialdatasets.ai news data with caching
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date for news
        end_date: End date for news
        limit: Maximum number of articles
    
    Returns:
        DataFrame with news data
    """
    
    def _fetch_financialdatasets_news_api(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Internal function to fetch news from financialdatasets.ai API"""
        try:
            client = FinancialDatasetsClient()
            
            # Get company news
            news_data = client.get_company_news(ticker=symbol, limit=limit)
            
            if news_data.empty:
                logger.warning(f"No financialdatasets.ai news found for {symbol}")
                return pd.DataFrame()
            
            # Filter by date range
            if 'date' in news_data.columns:
                news_data['date'] = pd.to_datetime(news_data['date'])
                mask = (news_data['date'] >= start_date) & (news_data['date'] <= end_date)
                news_data = news_data[mask]
            
            # Add symbol for consistency
            news_data['symbol'] = symbol
            
            logger.info(f"Retrieved {len(news_data)} news articles from financialdatasets.ai for {symbol}")
            return news_data
            
        except Exception as e:
            logger.error(f"Failed to fetch financialdatasets.ai news for {symbol}: {e}")
            return pd.DataFrame()
    
    return fetch_news_with_cache(symbol, start_date, end_date, _fetch_financialdatasets_news_api)


def fetch_financialdatasets_financials_cached(symbol: str, period: str = 'annual', limit: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Fetch financialdatasets.ai financial statements with caching
    
    Args:
        symbol: Stock ticker symbol
        period: 'annual', 'quarterly', or 'ttm'
        limit: Number of statements to return
    
    Returns:
        Dictionary with financial statements DataFrames
    """
    
    def _fetch_financialdatasets_financials_api(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Internal function to fetch financials from financialdatasets.ai API"""
        try:
            client = FinancialDatasetsClient()
            
            # Get all financial statements
            financials = client.get_all_financial_statements(ticker=symbol, period=period, limit=limit)
            
            # Combine all statements into one DataFrame for caching
            combined_data = []
            
            for stmt_type, df in financials.items():
                if not df.empty:
                    df_copy = df.copy()
                    df_copy['statement_type'] = stmt_type
                    df_copy['symbol'] = symbol
                    combined_data.append(df_copy)
            
            if combined_data:
                result = pd.concat(combined_data, ignore_index=True)
                logger.info(f"Retrieved financial statements from financialdatasets.ai for {symbol}")
                return result
            else:
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to fetch financialdatasets.ai financials for {symbol}: {e}")
            return pd.DataFrame()
    
    # For financials, we use a broader date range since they don't have specific dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 5)  # 5 years back
    
    cache = get_cache()
    cached_data = cache.fetch_with_cache(symbol, DataType.FUNDAMENTALS, start_date, end_date, _fetch_financialdatasets_financials_api)
    
    # Split back into separate DataFrames
    result = {}
    if not cached_data.empty and 'statement_type' in cached_data.columns:
        for stmt_type in cached_data['statement_type'].unique():
            stmt_data = cached_data[cached_data['statement_type'] == stmt_type].drop('statement_type', axis=1)
            result[stmt_type] = stmt_data
    
    return result


def fetch_financialdatasets_earnings_cached(symbol: str) -> pd.DataFrame:
    """
    Fetch financialdatasets.ai earnings press releases with caching
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        DataFrame with earnings press releases
    """
    
    def _fetch_financialdatasets_earnings_api(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Internal function to fetch earnings from financialdatasets.ai API"""
        try:
            client = FinancialDatasetsClient()
            
            earnings_data = client.get_earnings_press_releases(ticker=symbol)
            
            if earnings_data.empty:
                logger.warning(f"No financialdatasets.ai earnings found for {symbol}")
                return pd.DataFrame()
            
            # Filter by date range if date column exists
            if 'date' in earnings_data.columns:
                earnings_data['date'] = pd.to_datetime(earnings_data['date'])
                mask = (earnings_data['date'] >= start_date) & (earnings_data['date'] <= end_date)
                earnings_data = earnings_data[mask]
            
            earnings_data['symbol'] = symbol
            
            logger.info(f"Retrieved {len(earnings_data)} earnings releases from financialdatasets.ai for {symbol}")
            return earnings_data
            
        except Exception as e:
            logger.error(f"Failed to fetch financialdatasets.ai earnings for {symbol}: {e}")
            return pd.DataFrame()
    
    # Use a 2-year date range for earnings
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)
    
    return fetch_news_with_cache(symbol, start_date, end_date, _fetch_financialdatasets_earnings_api)


def fetch_financialdatasets_insider_trades_cached(symbol: str, limit: int = 100) -> pd.DataFrame:
    """
    Fetch financialdatasets.ai insider trading data with caching
    
    Args:
        symbol: Stock ticker symbol
        limit: Maximum number of trades to return
    
    Returns:
        DataFrame with insider trading data
    """
    
    def _fetch_financialdatasets_insider_api(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Internal function to fetch insider trades from financialdatasets.ai API"""
        try:
            client = FinancialDatasetsClient()
            
            insider_data = client.get_insider_trades(ticker=symbol, limit=limit)
            
            if insider_data.empty:
                logger.warning(f"No financialdatasets.ai insider trades found for {symbol}")
                return pd.DataFrame()
            
            # Filter by date range if transaction_date exists
            if 'transaction_date' in insider_data.columns:
                insider_data['transaction_date'] = pd.to_datetime(insider_data['transaction_date'])
                insider_data['date'] = insider_data['transaction_date']  # For consistency
                mask = (insider_data['date'] >= start_date) & (insider_data['date'] <= end_date)
                insider_data = insider_data[mask]
            
            insider_data['symbol'] = symbol
            
            logger.info(f"Retrieved {len(insider_data)} insider trades from financialdatasets.ai for {symbol}")
            return insider_data
            
        except Exception as e:
            logger.error(f"Failed to fetch financialdatasets.ai insider trades for {symbol}: {e}")
            return pd.DataFrame()
    
    # Use a 1-year date range for insider trades
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    cache = get_cache()
    return cache.fetch_with_cache(symbol, DataType.INSIDER, start_date, end_date, _fetch_financialdatasets_insider_api)


def fetch_financialdatasets_realtime_quote(symbol: str) -> Dict[str, Any]:
    """
    Get real-time quote from financialdatasets.ai (not cached - real-time data)
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with real-time quote data
    """
    try:
        client = FinancialDatasetsClient()
        quote = client.get_realtime_quote(ticker=symbol)
        
        logger.info(f"Retrieved real-time quote from financialdatasets.ai for {symbol}: ${quote.get('price', 'N/A')}")
        return quote
        
    except Exception as e:
        logger.error(f"Failed to fetch real-time quote for {symbol}: {e}")
        return {}


# Convenience Functions for financialdatasets.ai Integration
def get_financialdatasets_cached_data(symbol: str, start_date: str, end_date: str, data_type: str = 'prices') -> str:
    """
    Get cached financialdatasets.ai data in string format (compatible with existing interface)
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        data_type: Type of data ('prices', 'news', 'financials', 'earnings', 'insider')
    
    Returns:
        Formatted string with data
    """
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        if data_type == 'prices':
            df = fetch_financialdatasets_prices_cached(symbol, start_dt, end_dt)
            title = f"financialdatasets.ai Price Data for {symbol}"
            
        elif data_type == 'news':
            df = fetch_financialdatasets_news_cached(symbol, start_dt, end_dt)
            title = f"financialdatasets.ai News for {symbol}"
            
        elif data_type == 'earnings':
            df = fetch_financialdatasets_earnings_cached(symbol)
            title = f"financialdatasets.ai Earnings Releases for {symbol}"
            
        elif data_type == 'insider':
            df = fetch_financialdatasets_insider_trades_cached(symbol)
            title = f"financialdatasets.ai Insider Trades for {symbol}"
            
        elif data_type == 'financials':
            financials = fetch_financialdatasets_financials_cached(symbol)
            result = f"## {symbol} Financial Statements from financialdatasets.ai:\n\n"
            
            for stmt_type, df in financials.items():
                if not df.empty:
                    result += f"### {stmt_type.replace('_', ' ').title()}:\n"
                    result += df.to_string(index=False)
                    result += "\n\n"
            
            return result if result != f"## {symbol} Financial Statements from financialdatasets.ai:\n\n" else f"No financial data found for {symbol}"
        
        else:
            return f"Unknown data type: {data_type}"
        
        if df.empty:
            return f"No {data_type} data found for {symbol} between {start_date} and {end_date}"
        
        # Format output
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            df_string = df.to_string(index=False)
        
        return f"## {title} from {start_date} to {end_date}:\n\n{df_string}"
        
    except Exception as e:
        logger.error(f"Failed to get cached financialdatasets.ai data: {e}")
        return f"Error retrieving cached data for {symbol}: {e}"


# Cache Management Functions
def get_cache_summary() -> Dict[str, Any]:
    """Get comprehensive cache statistics"""
    cache = get_cache()
    return cache.get_cache_stats()


def clear_old_cache_data(days: int = 30) -> int:
    """Clear cache data older than specified days"""
    cache = get_cache()
    return cache.clear_cache(older_than_days=days)


def clear_symbol_cache(symbol: str) -> int:
    """Clear all cached data for a specific symbol"""
    cache = get_cache()
    total_cleared = 0
    for data_type in DataType:
        cleared = cache.clear_cache(symbol=symbol, data_type=data_type)
        total_cleared += cleared
    return total_cleared


if __name__ == "__main__":
    # Example usage
    print("Testing cached API wrappers...")
    
    # Test OHLCV caching
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"Fetching {symbol} data from {start_date.date()} to {end_date.date()}")
    
    # First call - should fetch from API
    data1 = fetch_yfinance_data_cached(symbol, start_date, end_date)
    print(f"First call: {len(data1)} records")
    
    # Second call - should use cache
    data2 = fetch_yfinance_data_cached(symbol, start_date, end_date)
    print(f"Second call: {len(data2)} records")
    
    # Print cache stats
    stats = get_cache_summary()
    print(f"Cache stats: {stats}") 