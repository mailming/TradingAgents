"""
financialdatasets.ai API Integration for TradingAgents

This module provides professional integration with financialdatasets.ai API
for comprehensive financial data including market data, fundamentals, news,
insider trades, and institutional ownership data.

financialdatasets.ai is specifically designed for AI financial agents with:
- 30,000+ tickers
- 30+ years of historical data  
- Real-time and historical prices
- Company financials and news
- SEC filings and insider data
- Built for developers by developers

Author: TradingAgents Team
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FinancialDatasetsConfig:
    """Configuration for financialdatasets.ai API"""
    base_url: str = "https://api.financialdatasets.ai"
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3

class FinancialDatasetsClient:
    """
    Professional client for financialdatasets.ai API integration
    
    Provides access to:
    - Historical and real-time stock prices
    - Company news and earnings press releases  
    - Financial statements (income, balance sheet, cash flow)
    - SEC filings and insider trades
    - Institutional ownership data
    - Financial metrics and ratios
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize financialdatasets.ai client
        
        Args:
            api_key: API key for financialdatasets.ai (can also use env var FINANCIALDATASETS_API_KEY)
        """
        self.config = FinancialDatasetsConfig()
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('FINANCIALDATASETS_API_KEY')
        
        if not self.api_key:
            logger.warning("No API key provided for financialdatasets.ai. Set FINANCIALDATASETS_API_KEY environment variable.")
        
        self.headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make authenticated request to financialdatasets.ai API"""
        if not self.api_key:
            raise ValueError("API key required for financialdatasets.ai")
        
        url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
        
        try:
            logger.info(f"Making request to financialdatasets.ai: {endpoint}")
            response = requests.get(
                url, 
                headers=self.headers, 
                params=params or {},
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling financialdatasets.ai API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise
    
    # ===== MARKET DATA METHODS =====
    
    def get_historical_prices(
        self, 
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = 'day',
        interval_multiplier: int = 1,
        limit: int = 5000
    ) -> pd.DataFrame:
        """
        Get historical OHLCV price data
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Time interval ('second', 'minute', 'day', 'week', 'month', 'year')
            interval_multiplier: Multiplier for interval (e.g., 5 for 5-minute intervals)
            limit: Maximum number of records (max 5000)
            
        Returns:
            DataFrame with OHLCV data and timestamps
        """
        params = {
            'ticker': ticker.upper(),
            'interval': interval,
            'interval_multiplier': interval_multiplier,
            'start_date': start_date,
            'end_date': end_date,
            'limit': limit
        }
        
        data = self._make_request('prices', params)
        prices = data.get('prices', [])
        
        if not prices:
            logger.warning(f"No price data found for {ticker}")
            return pd.DataFrame()
        
        df = pd.DataFrame(prices)
        
        # Convert time to datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        
        # Ensure numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Retrieved {len(df)} price records for {ticker}")
        return df
    
    def get_realtime_quote(self, ticker: str) -> Dict[str, Any]:
        """
        Get real-time quote for a ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with current price, change, volume, market cap
        """
        params = {'ticker': ticker.upper()}
        data = self._make_request('prices/snapshot', params)
        
        snapshot = data.get('snapshot', {})
        logger.info(f"Retrieved real-time quote for {ticker}: ${snapshot.get('price', 'N/A')}")
        return snapshot
    
    # ===== NEWS AND EARNINGS =====
    
    def get_company_news(
        self, 
        ticker: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get company-specific news articles
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of articles
            
        Returns:
            DataFrame with news articles
        """
        params = {
            'ticker': ticker.upper(),
            'limit': limit
        }
        
        data = self._make_request('news', params)
        news = data.get('news', [])
        
        if not news:
            logger.warning(f"No news found for {ticker}")
            return pd.DataFrame()
        
        df = pd.DataFrame(news)
        
        # Convert date to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', ascending=False, inplace=True)
        
        logger.info(f"Retrieved {len(df)} news articles for {ticker}")
        return df
    
    def get_earnings_press_releases(self, ticker: str) -> pd.DataFrame:
        """
        Get earnings press releases for a company
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with earnings press releases
        """
        params = {'ticker': ticker.upper()}
        
        data = self._make_request('earnings/press-releases', params)
        releases = data.get('press_releases', [])
        
        if not releases:
            logger.warning(f"No earnings press releases found for {ticker}")
            return pd.DataFrame()
        
        df = pd.DataFrame(releases)
        
        # Convert date to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', ascending=False, inplace=True)
        
        logger.info(f"Retrieved {len(df)} earnings press releases for {ticker}")
        return df
    
    # ===== FINANCIAL STATEMENTS =====
    
    def get_income_statements(
        self, 
        ticker: str,
        period: str = 'annual',
        limit: int = 10
    ) -> pd.DataFrame:
        """
        Get income statements for a company
        
        Args:
            ticker: Stock ticker symbol
            period: 'annual', 'quarterly', or 'ttm' (trailing twelve months)
            limit: Number of statements to return
            
        Returns:
            DataFrame with income statement data
        """
        params = {
            'ticker': ticker.upper(),
            'period': period,
            'limit': limit
        }
        
        data = self._make_request('financials/income-statements', params)
        statements = data.get('income_statements', [])
        
        if not statements:
            logger.warning(f"No income statements found for {ticker}")
            return pd.DataFrame()
        
        df = pd.DataFrame(statements)
        
        # Convert report_period to datetime
        if 'report_period' in df.columns:
            df['report_period'] = pd.to_datetime(df['report_period'])
            df.sort_values('report_period', ascending=False, inplace=True)
        
        logger.info(f"Retrieved {len(df)} income statements for {ticker}")
        return df
    
    def get_balance_sheets(
        self, 
        ticker: str,
        period: str = 'annual',
        limit: int = 10
    ) -> pd.DataFrame:
        """
        Get balance sheets for a company
        
        Args:
            ticker: Stock ticker symbol
            period: 'annual', 'quarterly', or 'ttm'
            limit: Number of statements to return
            
        Returns:
            DataFrame with balance sheet data
        """
        params = {
            'ticker': ticker.upper(),
            'period': period,
            'limit': limit
        }
        
        data = self._make_request('financials/balance-sheets', params)
        statements = data.get('balance_sheets', [])
        
        if not statements:
            logger.warning(f"No balance sheets found for {ticker}")
            return pd.DataFrame()
        
        df = pd.DataFrame(statements)
        
        # Convert report_period to datetime
        if 'report_period' in df.columns:
            df['report_period'] = pd.to_datetime(df['report_period'])
            df.sort_values('report_period', ascending=False, inplace=True)
        
        logger.info(f"Retrieved {len(df)} balance sheets for {ticker}")
        return df
    
    def get_cash_flow_statements(
        self, 
        ticker: str,
        period: str = 'annual',
        limit: int = 10
    ) -> pd.DataFrame:
        """
        Get cash flow statements for a company
        
        Args:
            ticker: Stock ticker symbol
            period: 'annual', 'quarterly', or 'ttm'
            limit: Number of statements to return
            
        Returns:
            DataFrame with cash flow data
        """
        params = {
            'ticker': ticker.upper(),
            'period': period,
            'limit': limit
        }
        
        data = self._make_request('financials/cash-flow-statements', params)
        statements = data.get('cash_flow_statements', [])
        
        if not statements:
            logger.warning(f"No cash flow statements found for {ticker}")
            return pd.DataFrame()
        
        df = pd.DataFrame(statements)
        
        # Convert report_period to datetime  
        if 'report_period' in df.columns:
            df['report_period'] = pd.to_datetime(df['report_period'])
            df.sort_values('report_period', ascending=False, inplace=True)
        
        logger.info(f"Retrieved {len(df)} cash flow statements for {ticker}")
        return df
    
    def get_all_financial_statements(
        self, 
        ticker: str,
        period: str = 'annual',
        limit: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Get all financial statements (income, balance sheet, cash flow) for a company
        
        Args:
            ticker: Stock ticker symbol
            period: 'annual', 'quarterly', or 'ttm'
            limit: Number of statements to return for each type
            
        Returns:
            Dictionary with 'income_statements', 'balance_sheets', 'cash_flow_statements'
        """
        logger.info(f"Retrieving all financial statements for {ticker}")
        
        return {
            'income_statements': self.get_income_statements(ticker, period, limit),
            'balance_sheets': self.get_balance_sheets(ticker, period, limit),
            'cash_flow_statements': self.get_cash_flow_statements(ticker, period, limit)
        }
    
    # ===== INSIDER TRADES AND INSTITUTIONAL DATA =====
    
    def get_insider_trades(self, ticker: str, limit: int = 100) -> pd.DataFrame:
        """
        Get insider trading data for a company
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of trades to return
            
        Returns:
            DataFrame with insider trading data
        """
        params = {
            'ticker': ticker.upper(),
            'limit': limit
        }
        
        data = self._make_request('insider-trades', params)
        trades = data.get('insider_trades', [])
        
        if not trades:
            logger.warning(f"No insider trades found for {ticker}")
            return pd.DataFrame()
        
        df = pd.DataFrame(trades)
        
        # Convert transaction_date to datetime
        if 'transaction_date' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            df.sort_values('transaction_date', ascending=False, inplace=True)
        
        logger.info(f"Retrieved {len(df)} insider trades for {ticker}")
        return df
    
    def get_institutional_ownership(self, ticker: str, limit: int = 100) -> pd.DataFrame:
        """
        Get institutional ownership data for a company
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of holdings to return
            
        Returns:
            DataFrame with institutional ownership data
        """
        params = {
            'ticker': ticker.upper(),
            'limit': limit
        }
        
        data = self._make_request('institutional-ownership', params)
        holdings = data.get('institutional_holdings', [])
        
        if not holdings:
            logger.warning(f"No institutional holdings found for {ticker}")
            return pd.DataFrame()
        
        df = pd.DataFrame(holdings)
        
        # Convert period_of_report to datetime
        if 'period_of_report' in df.columns:
            df['period_of_report'] = pd.to_datetime(df['period_of_report'])
            df.sort_values('period_of_report', ascending=False, inplace=True)
        
        logger.info(f"Retrieved {len(df)} institutional holdings for {ticker}")
        return df
    
    # ===== COMPANY AND SECTOR DATA =====
    
    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get basic company information
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with company details
        """
        params = {'ticker': ticker.upper()}
        
        data = self._make_request('company', params)
        company_info = data.get('company', {})
        
        logger.info(f"Retrieved company info for {ticker}")
        return company_info
    
    def get_financial_metrics(
        self, 
        ticker: str,
        period: str = 'annual',
        limit: int = 5
    ) -> pd.DataFrame:
        """
        Get financial metrics and ratios for a company
        
        Args:
            ticker: Stock ticker symbol
            period: 'annual', 'quarterly', or 'ttm'
            limit: Number of periods to return
            
        Returns:
            DataFrame with financial metrics
        """
        params = {
            'ticker': ticker.upper(),
            'period': period,
            'limit': limit
        }
        
        data = self._make_request('financial-metrics', params)
        metrics = data.get('financial_metrics', [])
        
        if not metrics:
            logger.warning(f"No financial metrics found for {ticker}")
            return pd.DataFrame()
        
        df = pd.DataFrame(metrics)
        
        # Convert report_period to datetime
        if 'report_period' in df.columns:
            df['report_period'] = pd.to_datetime(df['report_period'])
            df.sort_values('report_period', ascending=False, inplace=True)
        
        logger.info(f"Retrieved {len(df)} financial metrics for {ticker}")
        return df
    
    # ===== SEARCH AND DISCOVERY =====
    
    def search_companies(self, query: str, limit: int = 20) -> pd.DataFrame:
        """
        Search for companies by name or ticker
        
        Args:
            query: Search query (company name or ticker)
            limit: Maximum number of results
            
        Returns:
            DataFrame with search results
        """
        params = {
            'query': query,
            'limit': limit
        }
        
        data = self._make_request('search', params)
        results = data.get('search_results', [])
        
        if not results:
            logger.warning(f"No companies found for query: {query}")
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        logger.info(f"Found {len(df)} companies for query: {query}")
        return df
    
    # ===== UTILITY METHODS =====
    
    def get_available_tickers(self, endpoint: str = 'prices') -> List[str]:
        """
        Get list of available tickers for a specific endpoint
        
        Args:
            endpoint: API endpoint ('prices', 'news', 'financials', etc.)
            
        Returns:
            List of available ticker symbols
        """
        try:
            data = self._make_request(f'{endpoint}/tickers')
            tickers = data.get('tickers', [])
            logger.info(f"Retrieved {len(tickers)} available tickers for {endpoint}")
            return tickers
        except Exception as e:
            logger.error(f"Error retrieving tickers for {endpoint}: {e}")
            return []
    
    def test_connection(self) -> bool:
        """
        Test API connection and authentication
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test with a simple quote request
            self.get_realtime_quote('AAPL')
            logger.info("financialdatasets.ai API connection successful")
            return True
        except Exception as e:
            logger.error(f"financialdatasets.ai API connection failed: {e}")
            return False

# Convenience functions for direct usage
def get_financialdatasets_historical_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = 'day',
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to get historical prices from financialdatasets.ai
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)  
        end_date: End date (YYYY-MM-DD)
        interval: Time interval ('day', 'minute', etc.)
        api_key: Optional API key
        
    Returns:
        DataFrame with historical price data
    """
    client = FinancialDatasetsClient(api_key)
    return client.get_historical_prices(ticker, start_date, end_date, interval)

def get_financialdatasets_company_news(
    ticker: str,
    limit: int = 50,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to get company news from financialdatasets.ai
    
    Args:
        ticker: Stock ticker symbol
        limit: Maximum number of articles
        api_key: Optional API key
        
    Returns:
        DataFrame with news articles
    """
    client = FinancialDatasetsClient(api_key)
    return client.get_company_news(ticker, limit)

def get_financialdatasets_financials(
    ticker: str,
    period: str = 'annual',
    api_key: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to get all financial statements from financialdatasets.ai
    
    Args:
        ticker: Stock ticker symbol
        period: 'annual', 'quarterly', or 'ttm'
        api_key: Optional API key
        
    Returns:
        Dictionary with all financial statements
    """
    client = FinancialDatasetsClient(api_key)
    return client.get_all_financial_statements(ticker, period)

# Example usage
if __name__ == "__main__":
    # Initialize client
    client = FinancialDatasetsClient()
    
    # Test connection
    if client.test_connection():
        print("✅ financialdatasets.ai connection successful")
        
        # Example: Get TSLA data
        ticker = "TSLA"
        
        # Get real-time quote
        quote = client.get_realtime_quote(ticker)
        print(f"💰 {ticker} Current Price: ${quote.get('price', 'N/A')}")
        
        # Get recent news
        news = client.get_company_news(ticker, limit=5)
        print(f"📰 Retrieved {len(news)} news articles for {ticker}")
        
        # Get financial statements
        financials = client.get_all_financial_statements(ticker, period='annual', limit=3)
        print(f"📊 Retrieved financial statements for {ticker}")
        for stmt_type, df in financials.items():
            print(f"  - {stmt_type}: {len(df)} records")
    else:
        print("❌ financialdatasets.ai connection failed")
        print("Make sure to set FINANCIALDATASETS_API_KEY environment variable") 