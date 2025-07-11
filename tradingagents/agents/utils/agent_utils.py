from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from typing import List
from typing import Annotated
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import RemoveMessage
from langchain_core.tools import tool
from datetime import date, timedelta, datetime
import functools
import pandas as pd
import os
from dateutil.relativedelta import relativedelta
from langchain_openai import ChatOpenAI
import tradingagents.dataflows.interface as interface
from tradingagents.default_config import DEFAULT_CONFIG
from langchain_core.messages import HumanMessage


def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]
        
        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]
        
        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")
        
        return {"messages": removal_operations + [placeholder]}
    
    return delete_messages


class Toolkit:
    _config = DEFAULT_CONFIG.copy()

    @classmethod
    def update_config(cls, config):
        """Update the class-level configuration."""
        cls._config.update(config)

    @property
    def config(self):
        """Access the configuration."""
        return self._config

    def __init__(self, config=None):
        if config:
            self.update_config(config)

    @staticmethod
    @tool
    def get_reddit_news(
        curr_date: Annotated[str, "Date you want to get news for in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve global news from Reddit within a specified time frame.
        Args:
            curr_date (str): Date you want to get news for in yyyy-mm-dd format
        Returns:
            str: A formatted dataframe containing the latest global news from Reddit in the specified time frame.
        """
        
        global_news_result = interface.get_reddit_global_news(curr_date, 7, 5)

        return global_news_result

    @staticmethod
    @tool
    def get_finnhub_news(
        ticker: Annotated[
            str,
            "Search query of a company, e.g. 'AAPL, TSM, etc.",
        ],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ):
        """
        Retrieve the latest news about a given stock from Finnhub within a date range
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: A formatted dataframe containing news about the company within the date range from start_date to end_date
        """

        end_date_str = end_date

        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        look_back_days = (end_date - start_date).days

        finnhub_news_result = interface.get_finnhub_news(
            ticker, end_date_str, look_back_days
        )

        return finnhub_news_result

    @staticmethod
    @tool
    def get_reddit_stock_info(
        ticker: Annotated[
            str,
            "Ticker of a company. e.g. AAPL, TSM",
        ],
        curr_date: Annotated[str, "Current date you want to get news for"],
    ) -> str:
        """
        Retrieve the latest news about a given stock from Reddit, given the current date.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            curr_date (str): current date in yyyy-mm-dd format to get news for
        Returns:
            str: A formatted dataframe containing the latest news about the company on the given date
        """

        stock_news_results = interface.get_reddit_company_news(ticker, curr_date, 7, 5)

        return stock_news_results

    @staticmethod
    @tool
    def get_YFin_data(
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve the stock price data for a given ticker symbol from Yahoo Finance.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
        """

        result_data = interface.get_YFin_data(symbol, start_date, end_date)

        return result_data

    @staticmethod
    @tool
    def get_YFin_data_online(
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve the stock price data for a given ticker symbol from Yahoo Finance.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
        """

        result_data = interface.get_YFin_data_online(symbol, start_date, end_date)

        return result_data

    @staticmethod
    @tool
    def get_stockstats_indicators_report(
        symbol: Annotated[str, "ticker symbol of the company"],
        indicator: Annotated[
            str, "technical indicator to get the analysis and report of"
        ],
        curr_date: Annotated[
            str, "The current trading date you are trading on, YYYY-mm-dd"
        ],
        look_back_days: Annotated[int, "how many days to look back"] = 30,
    ) -> str:
        """
        Retrieve stock stats indicators for a given ticker symbol and indicator.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            indicator (str): Technical indicator to get the analysis and report of
            curr_date (str): The current trading date you are trading on, YYYY-mm-dd
            look_back_days (int): How many days to look back, default is 30
        Returns:
            str: A formatted dataframe containing the stock stats indicators for the specified ticker symbol and indicator.
        """

        result_stockstats = interface.get_stock_stats_indicators_window(
            symbol, indicator, curr_date, look_back_days, False
        )

        return result_stockstats

    @staticmethod
    @tool
    def get_stockstats_indicators_report_online(
        symbol: Annotated[str, "ticker symbol of the company"],
        indicator: Annotated[
            str, "technical indicator to get the analysis and report of"
        ],
        curr_date: Annotated[
            str, "The current trading date you are trading on, YYYY-mm-dd"
        ],
        look_back_days: Annotated[int, "how many days to look back"] = 30,
    ) -> str:
        """
        Retrieve stock stats indicators for a given ticker symbol and indicator.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            indicator (str): Technical indicator to get the analysis and report of
            curr_date (str): The current trading date you are trading on, YYYY-mm-dd
            look_back_days (int): How many days to look back, default is 30
        Returns:
            str: A formatted dataframe containing the stock stats indicators for the specified ticker symbol and indicator.
        """

        result_stockstats = interface.get_stock_stats_indicators_window(
            symbol, indicator, curr_date, look_back_days, True
        )

        return result_stockstats

    @staticmethod
    @tool
    def get_finnhub_company_insider_sentiment(
        ticker: Annotated[str, "ticker symbol for the company"],
        curr_date: Annotated[
            str,
            "current date of you are trading at, yyyy-mm-dd",
        ],
    ):
        """
        Retrieve insider sentiment information about a company (retrieved from public SEC information) for the past 30 days
        Args:
            ticker (str): ticker symbol of the company
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            str: a report of the sentiment in the past 30 days starting at curr_date
        """

        data_sentiment = interface.get_finnhub_company_insider_sentiment(
            ticker, curr_date, 30
        )

        return data_sentiment

    @staticmethod
    @tool
    def get_finnhub_company_insider_transactions(
        ticker: Annotated[str, "ticker symbol"],
        curr_date: Annotated[
            str,
            "current date you are trading at, yyyy-mm-dd",
        ],
    ):
        """
        Retrieve insider transaction information about a company (retrieved from public SEC information) for the past 30 days
        Args:
            ticker (str): ticker symbol of the company
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            str: a report of the company's insider transactions/trading information in the past 30 days
        """

        data_trans = interface.get_finnhub_company_insider_transactions(
            ticker, curr_date, 30
        )

        return data_trans

    @staticmethod
    @tool
    def get_simfin_balance_sheet(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    ):
        """
        Retrieve the most recent balance sheet of a company
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency of the company's financial history: annual / quarterly
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            str: a report of the company's most recent balance sheet
        """

        data_balance_sheet = interface.get_simfin_balance_sheet(ticker, freq, curr_date)

        return data_balance_sheet

    @staticmethod
    @tool
    def get_simfin_cashflow(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    ):
        """
        Retrieve the most recent cash flow statement of a company
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency of the company's financial history: annual / quarterly
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
                str: a report of the company's most recent cash flow statement
        """

        data_cashflow = interface.get_simfin_cashflow(ticker, freq, curr_date)

        return data_cashflow

    @staticmethod
    @tool
    def get_simfin_income_stmt(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    ):
        """
        Retrieve the most recent income statement of a company
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency of the company's financial history: annual / quarterly
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
                str: a report of the company's most recent income statement
        """

        data_income_stmt = interface.get_simfin_income_statements(
            ticker, freq, curr_date
        )

        return data_income_stmt

    @staticmethod
    @tool
    def get_google_news(
        query: Annotated[str, "Query to search with"],
        curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    ):
        """
        Retrieve the latest news from Google News based on a query and date range.
        Args:
            query (str): Query to search with
            curr_date (str): Current date in yyyy-mm-dd format
            look_back_days (int): How many days to look back
        Returns:
            str: A formatted string containing the latest news from Google News based on the query and date range.
        """

        google_news_results = interface.get_google_news(query, curr_date, 7)

        return google_news_results

    # CACHED METHODS FOR IMPROVED PERFORMANCE
    
    @staticmethod
    @tool
    def get_YFin_data_cached(
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve cached stock price data for a given ticker symbol from Yahoo Finance.
        Uses intelligent caching to minimize API calls and improve performance.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSLA
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
        """
        result_data = interface.get_YFin_data_cached(symbol, start_date, end_date)
        return result_data

    @staticmethod
    @tool
    def get_YFin_data_window_cached(
        symbol: Annotated[str, "ticker symbol of the company"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
        look_back_days: Annotated[int, "how many days to look back"],
    ) -> str:
        """
        Retrieve cached stock price data for a window of days with intelligent caching.
        Significantly faster than regular API calls for repeated queries.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSLA
            curr_date (str): Current date in yyyy-mm-dd format
            look_back_days (int): How many days to look back
        Returns:
            str: A formatted dataframe containing the stock price data for the specified window.
        """
        result_data = interface.get_YFin_data_window_cached(symbol, curr_date, look_back_days)
        return result_data

    @staticmethod
    @tool
    def get_stockstats_indicators_cached(
        symbol: Annotated[str, "ticker symbol of the company"],
        indicator: Annotated[str, "technical indicator to get the analysis and report of"],
        curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
        look_back_days: Annotated[int, "how many days to look back"] = 30,
    ) -> str:
        """
        Retrieve cached technical indicators for a given ticker symbol.
        Uses intelligent caching for improved performance over repeated analysis.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSLA
            indicator (str): Technical indicator to get the analysis and report of
            curr_date (str): The current trading date you are trading on, YYYY-mm-dd
            look_back_days (int): How many days to look back, default is 30
        Returns:
            str: A formatted dataframe containing the cached technical indicators.
        """
        result_indicators = interface.get_technical_indicators_cached(symbol, indicator, curr_date, look_back_days)
        return result_indicators

    @staticmethod
    @tool
    def get_finnhub_news_cached(
        ticker: Annotated[str, "ticker symbol for the company"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
        look_back_days: Annotated[int, "how many days to look back"] = 7,
    ) -> str:
        """
        Retrieve cached news about a company from Finnhub.
        Uses intelligent caching to reduce API calls and improve response time.
        Args:
            ticker (str): Ticker symbol for the company
            curr_date (str): Current date in yyyy-mm-dd format
            look_back_days (int): How many days to look back, default is 7
        Returns:
            str: A formatted string containing cached news about the company.
        """
        cached_news = interface.get_finnhub_news_cached(ticker, curr_date, look_back_days)
        return cached_news

    @staticmethod
    @tool
    def get_google_news_cached(
        query: Annotated[str, "Query to search with"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
        look_back_days: Annotated[int, "how many days to look back"] = 7,
    ) -> str:
        """
        Retrieve cached news from Google News based on a query.
        Uses intelligent caching to improve performance and reduce API overhead.
        Args:
            query (str): Query to search with
            curr_date (str): Current date in yyyy-mm-dd format
            look_back_days (int): How many days to look back, default is 7
        Returns:
            str: A formatted string containing cached Google News results.
        """
        cached_google_news = interface.get_google_news_cached(query, curr_date, look_back_days)
        return cached_google_news

    # FINANCIALDATASETS.AI CACHED METHODS
    
    @staticmethod
    @tool
    def get_financialdatasets_prices_cached(
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
        interval: Annotated[str, "Time interval: 'day', 'minute', 'hour', etc."] = 'day',
    ) -> str:
        """
        Retrieve high-quality stock price data from financialdatasets.ai with intelligent caching.
        Provides OHLCV data for 30,000+ tickers with 30+ years of history.
        Uses intelligent caching to minimize API calls and maximize performance.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSLA
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
            interval (str): Time interval: 'day', 'minute', 'hour', etc., default is 'day'
        Returns:
            str: A formatted dataframe containing professional-grade stock price data from financialdatasets.ai.
        """
        from tradingagents.dataflows import get_financialdatasets_cached_data
        result_data = get_financialdatasets_cached_data(symbol, start_date, end_date, 'prices')
        return result_data

    @staticmethod
    @tool
    def get_financialdatasets_news_cached(
        ticker: Annotated[str, "ticker symbol for the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve company-specific news from financialdatasets.ai with intelligent caching.
        Provides real-time and historical news articles designed for AI financial agents.
        Uses intelligent caching to reduce API calls and improve response time.
        Args:
            ticker (str): Ticker symbol for the company, e.g. AAPL, TSLA
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: A formatted string containing professional news data from financialdatasets.ai.
        """
        from tradingagents.dataflows import get_financialdatasets_cached_data
        cached_news = get_financialdatasets_cached_data(ticker, start_date, end_date, 'news')
        return cached_news

    @staticmethod
    @tool
    def get_financialdatasets_financials_cached(
        ticker: Annotated[str, "ticker symbol for the company"],
        period: Annotated[str, "reporting period: 'annual', 'quarterly', or 'ttm'"] = 'annual',
    ) -> str:
        """
        Retrieve comprehensive financial statements from financialdatasets.ai with intelligent caching.
        Includes income statements, balance sheets, and cash flow statements.
        Provides 30+ years of financial data for 30,000+ companies.
        Args:
            ticker (str): Ticker symbol for the company, e.g. AAPL, TSLA
            period (str): Reporting period: 'annual', 'quarterly', or 'ttm' (trailing twelve months)
        Returns:
            str: A formatted string containing comprehensive financial statements from financialdatasets.ai.
        """
        from tradingagents.dataflows import get_financialdatasets_cached_data
        # Use a 5-year range for financials
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        financials = get_financialdatasets_cached_data(ticker, start_date, end_date, 'financials')
        return financials

    @staticmethod
    @tool
    def get_financialdatasets_earnings_cached(
        ticker: Annotated[str, "ticker symbol for the company"],
    ) -> str:
        """
        Retrieve earnings press releases from financialdatasets.ai with intelligent caching.
        Provides earnings-related press releases and announcements.
        Updated instantly when new press releases are published via RSS feeds.
        Args:
            ticker (str): Ticker symbol for the company, e.g. AAPL, TSLA
        Returns:
            str: A formatted string containing earnings press releases from financialdatasets.ai.
        """
        from tradingagents.dataflows import get_financialdatasets_cached_data
        # Use a 2-year range for earnings
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        earnings = get_financialdatasets_cached_data(ticker, start_date, end_date, 'earnings')
        return earnings

    @staticmethod
    @tool
    def get_financialdatasets_insider_trades_cached(
        ticker: Annotated[str, "ticker symbol for the company"],
    ) -> str:
        """
        Retrieve insider trading data from financialdatasets.ai with intelligent caching.
        Provides corporate insider trading activity and holdings data.
        Includes transaction details, insider names, and position changes.
        Args:
            ticker (str): Ticker symbol for the company, e.g. AAPL, TSLA
        Returns:
            str: A formatted string containing insider trading data from financialdatasets.ai.
        """
        from tradingagents.dataflows import get_financialdatasets_cached_data
        # Use a 1-year range for insider trades
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        insider_trades = get_financialdatasets_cached_data(ticker, start_date, end_date, 'insider')
        return insider_trades

    @staticmethod
    @tool
    def get_financialdatasets_realtime_quote(
        ticker: Annotated[str, "ticker symbol for the company"],
    ) -> str:
        """
        Get real-time stock quote from financialdatasets.ai.
        Provides current price, daily change, volume, and market cap data.
        Not cached as this is real-time data that changes continuously.
        Args:
            ticker (str): Ticker symbol for the company, e.g. AAPL, TSLA
        Returns:
            str: A formatted string containing real-time quote data from financialdatasets.ai.
        """
        from tradingagents.dataflows import fetch_financialdatasets_realtime_quote
        try:
            quote = fetch_financialdatasets_realtime_quote(ticker)
            if quote:
                price = quote.get('price', 'N/A')
                change = quote.get('day_change', 'N/A')
                change_pct = quote.get('day_change_percent', 'N/A')
                volume = quote.get('market_cap', 'N/A')
                
                return f"""## Real-time Quote for {ticker} from financialdatasets.ai:

Current Price: ${price}
Daily Change: {change} ({change_pct}%)
Market Cap: {volume}

Data Source: financialdatasets.ai (Professional Financial Data API)"""
            else:
                return f"No real-time quote available for {ticker}"
        except Exception as e:
            return f"Error retrieving real-time quote for {ticker}: {e}"

    @staticmethod
    @tool
    def get_stock_news_openai(
        ticker: Annotated[str, "the company's ticker"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ):
        """
        Retrieve the latest news about a given stock by using OpenAI's news API.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            curr_date (str): Current date in yyyy-mm-dd format
        Returns:
            str: A formatted string containing the latest news about the company on the given date.
        """

        openai_news_results = interface.get_stock_news_openai(ticker, curr_date)

        return openai_news_results

    @staticmethod
    @tool
    def get_global_news_openai(
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ):
        """
        Retrieve the latest macroeconomics news on a given date using OpenAI's macroeconomics news API.
        Args:
            curr_date (str): Current date in yyyy-mm-dd format
        Returns:
            str: A formatted string containing the latest macroeconomic news on the given date.
        """

        openai_news_results = interface.get_global_news_openai(curr_date)

        return openai_news_results

    @staticmethod
    @tool
    def get_fundamentals_openai(
        ticker: Annotated[str, "the company's ticker"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ):
        """
        Retrieve the latest fundamental information about a given stock on a given date by using OpenAI's news API.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            curr_date (str): Current date in yyyy-mm-dd format
        Returns:
            str: A formatted string containing the latest fundamental information about the company on the given date.
        """

        openai_fundamentals_results = interface.get_fundamentals_openai(
            ticker, curr_date
        )

        return openai_fundamentals_results
