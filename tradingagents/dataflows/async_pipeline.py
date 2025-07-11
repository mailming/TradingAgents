"""
Async Data Pipeline for TradingAgents

This module implements asynchronous data fetching and processing:
- Concurrent API calls
- Async data transformation
- Pipeline coordination
- Error handling and retries
- Performance monitoring

Author: TradingAgents Performance Team
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Data pipeline stages"""
    FETCH = "fetch"
    TRANSFORM = "transform"
    VALIDATE = "validate"
    CACHE = "cache"
    COMPLETE = "complete"


class DataType(Enum):
    """Types of data in the pipeline"""
    PRICE_DATA = "price_data"
    NEWS_DATA = "news_data"
    FUNDAMENTAL_DATA = "fundamental_data"
    TECHNICAL_INDICATORS = "technical_indicators"
    MARKET_DATA = "market_data"


@dataclass
class PipelineTask:
    """Represents a task in the data pipeline"""
    task_id: str
    data_type: DataType
    symbol: str
    start_date: datetime
    end_date: datetime
    priority: int = 1
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    stage: PipelineStage = PipelineStage.FETCH
    data: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_fetch_time: float = 0.0
    total_transform_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    concurrent_tasks: int = 0
    max_concurrent: int = 0


class AsyncDataFetcher:
    """Handles asynchronous data fetching from multiple sources"""
    
    def __init__(self, max_concurrent: int = 10, timeout: int = 30):
        """
        Initialize async data fetcher
        
        Args:
            max_concurrent: Maximum concurrent requests
            timeout: Request timeout in seconds
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.session = None
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.info(f"🚀 Async Data Fetcher initialized (max concurrent: {max_concurrent})")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            connector=aiohttp.TCPConnector(limit=self.max_concurrent * 2)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_price_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch price data asynchronously"""
        async with self.semaphore:
            try:
                # In a real implementation, this would make async API calls
                # For now, we'll simulate with the existing sync functions
                loop = asyncio.get_event_loop()
                
                # Run sync function in thread pool
                with ThreadPoolExecutor() as executor:
                    future = loop.run_in_executor(
                        executor,
                        self._fetch_price_data_sync,
                        symbol, start_date, end_date
                    )
                    result = await future
                
                logger.debug(f"📈 Fetched price data for {symbol}")
                return result
                
            except Exception as e:
                logger.error(f"❌ Failed to fetch price data for {symbol}: {e}")
                return pd.DataFrame()
    
    def _fetch_price_data_sync(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Synchronous price data fetching (wrapper for existing functions)"""
        try:
            from .cached_api_wrappers import fetch_financialdatasets_prices_cached
            return fetch_financialdatasets_prices_cached(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"❌ Sync price data fetch failed for {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_news_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch news data asynchronously"""
        async with self.semaphore:
            try:
                loop = asyncio.get_event_loop()
                
                with ThreadPoolExecutor() as executor:
                    future = loop.run_in_executor(
                        executor,
                        self._fetch_news_data_sync,
                        symbol, start_date, end_date
                    )
                    result = await future
                
                logger.debug(f"📰 Fetched news data for {symbol}")
                return result
                
            except Exception as e:
                logger.error(f"❌ Failed to fetch news data for {symbol}: {e}")
                return pd.DataFrame()
    
    def _fetch_news_data_sync(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Synchronous news data fetching"""
        try:
            from .cached_api_wrappers import fetch_financialdatasets_news_cached
            return fetch_financialdatasets_news_cached(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"❌ Sync news data fetch failed for {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_fundamental_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch fundamental data asynchronously"""
        async with self.semaphore:
            try:
                loop = asyncio.get_event_loop()
                
                with ThreadPoolExecutor() as executor:
                    future = loop.run_in_executor(
                        executor,
                        self._fetch_fundamental_data_sync,
                        symbol
                    )
                    result = await future
                
                logger.debug(f"📊 Fetched fundamental data for {symbol}")
                return result
                
            except Exception as e:
                logger.error(f"❌ Failed to fetch fundamental data for {symbol}: {e}")
                return {}
    
    def _fetch_fundamental_data_sync(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Synchronous fundamental data fetching"""
        try:
            from .cached_api_wrappers import fetch_financialdatasets_financials_cached
            return fetch_financialdatasets_financials_cached(symbol)
        except Exception as e:
            logger.error(f"❌ Sync fundamental data fetch failed for {symbol}: {e}")
            return {}


class AsyncDataProcessor:
    """Handles asynchronous data processing and transformation"""
    
    def __init__(self, max_workers: int = 8):
        """
        Initialize async data processor
        
        Args:
            max_workers: Maximum worker threads for CPU-bound tasks
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"⚙️ Async Data Processor initialized (workers: {max_workers})")
    
    async def process_price_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process price data asynchronously"""
        if data.empty:
            return data
        
        try:
            loop = asyncio.get_event_loop()
            
            # Run CPU-intensive processing in thread pool
            future = loop.run_in_executor(
                self.executor,
                self._process_price_data_sync,
                data, symbol
            )
            result = await future
            
            logger.debug(f"⚙️ Processed price data for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to process price data for {symbol}: {e}")
            return data
    
    def _process_price_data_sync(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Synchronous price data processing"""
        try:
            # Add calculated fields
            if 'close' in data.columns:
                data['daily_return'] = data['close'].pct_change()
                data['volatility'] = data['daily_return'].rolling(window=20).std()
                data['sma_20'] = data['close'].rolling(window=20).mean()
                data['sma_50'] = data['close'].rolling(window=50).mean()
            
            # Add metadata
            data['symbol'] = symbol
            data['processed_at'] = datetime.now()
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Price data processing failed for {symbol}: {e}")
            return data
    
    async def process_news_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process news data asynchronously"""
        if data.empty:
            return data
        
        try:
            loop = asyncio.get_event_loop()
            
            future = loop.run_in_executor(
                self.executor,
                self._process_news_data_sync,
                data, symbol
            )
            result = await future
            
            logger.debug(f"⚙️ Processed news data for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to process news data for {symbol}: {e}")
            return data
    
    def _process_news_data_sync(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Synchronous news data processing"""
        try:
            # Add sentiment analysis (simplified)
            if 'headline' in data.columns:
                data['sentiment_score'] = data['headline'].apply(self._calculate_sentiment)
                data['word_count'] = data['headline'].str.split().str.len()
            
            # Add metadata
            data['symbol'] = symbol
            data['processed_at'] = datetime.now()
            
            return data
            
        except Exception as e:
            logger.error(f"❌ News data processing failed for {symbol}: {e}")
            return data
    
    def _calculate_sentiment(self, text: str) -> float:
        """Simple sentiment calculation (would be replaced with proper NLP)"""
        positive_words = ['gain', 'profit', 'growth', 'positive', 'bullish', 'up', 'rise']
        negative_words = ['loss', 'decline', 'negative', 'bearish', 'down', 'fall', 'drop']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)


class AsyncDataPipeline:
    """Main async data pipeline coordinator"""
    
    def __init__(self, max_concurrent_fetches: int = 10, max_workers: int = 8):
        """
        Initialize the async data pipeline
        
        Args:
            max_concurrent_fetches: Maximum concurrent fetch operations
            max_workers: Maximum worker threads for processing
        """
        self.max_concurrent_fetches = max_concurrent_fetches
        self.max_workers = max_workers
        
        # Components
        self.fetcher = None
        self.processor = AsyncDataProcessor(max_workers)
        
        # Pipeline state
        self.tasks: Dict[str, PipelineTask] = {}
        self.metrics = PipelineMetrics()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info(f"🔄 Async Data Pipeline initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.fetcher = AsyncDataFetcher(self.max_concurrent_fetches)
        await self.fetcher.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.fetcher:
            await self.fetcher.__aexit__(exc_type, exc_val, exc_tb)
        self.processor.cleanup()
    
    async def fetch_all_data_for_symbol(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Fetch all types of data for a symbol concurrently
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary with all fetched data types
        """
        logger.info(f"🔄 Starting async data fetch for {symbol}")
        start_time = time.time()
        
        # Create concurrent tasks
        tasks = {
            'price_data': self.fetcher.fetch_price_data(symbol, start_date, end_date),
            'news_data': self.fetcher.fetch_news_data(symbol, start_date, end_date),
            'fundamental_data': self.fetcher.fetch_fundamental_data(symbol)
        }
        
        # Execute all tasks concurrently
        try:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            # Combine results
            data_dict = {}
            for i, (data_type, result) in enumerate(zip(tasks.keys(), results)):
                if isinstance(result, Exception):
                    logger.error(f"❌ Failed to fetch {data_type} for {symbol}: {result}")
                    data_dict[data_type] = pd.DataFrame() if data_type != 'fundamental_data' else {}
                else:
                    data_dict[data_type] = result
            
            fetch_time = time.time() - start_time
            logger.info(f"✅ Completed async fetch for {symbol} in {fetch_time:.2f}s")
            
            return data_dict
            
        except Exception as e:
            logger.error(f"❌ Async fetch failed for {symbol}: {e}")
            return {
                'price_data': pd.DataFrame(),
                'news_data': pd.DataFrame(),
                'fundamental_data': {}
            }
    
    async def process_all_data(self, data_dict: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """
        Process all data types concurrently
        
        Args:
            data_dict: Dictionary with raw data
            symbol: Stock symbol
            
        Returns:
            Dictionary with processed data
        """
        logger.info(f"⚙️ Starting async processing for {symbol}")
        start_time = time.time()
        
        # Create processing tasks
        processing_tasks = []
        
        if 'price_data' in data_dict and not data_dict['price_data'].empty:
            processing_tasks.append(
                self.processor.process_price_data(data_dict['price_data'], symbol)
            )
        else:
            processing_tasks.append(asyncio.sleep(0))  # No-op
        
        if 'news_data' in data_dict and not data_dict['news_data'].empty:
            processing_tasks.append(
                self.processor.process_news_data(data_dict['news_data'], symbol)
            )
        else:
            processing_tasks.append(asyncio.sleep(0))  # No-op
        
        try:
            # Process concurrently
            processed_results = await asyncio.gather(*processing_tasks, return_exceptions=True)
            
            # Combine processed results
            processed_dict = data_dict.copy()
            
            if len(processed_results) > 0 and not isinstance(processed_results[0], Exception):
                processed_dict['price_data'] = processed_results[0]
            
            if len(processed_results) > 1 and not isinstance(processed_results[1], Exception):
                processed_dict['news_data'] = processed_results[1]
            
            process_time = time.time() - start_time
            logger.info(f"✅ Completed async processing for {symbol} in {process_time:.2f}s")
            
            return processed_dict
            
        except Exception as e:
            logger.error(f"❌ Async processing failed for {symbol}: {e}")
            return data_dict
    
    async def run_full_pipeline(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Run the complete async pipeline for a symbol
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary with all processed data
        """
        pipeline_start = time.time()
        
        try:
            # Step 1: Fetch all data concurrently
            raw_data = await self.fetch_all_data_for_symbol(symbol, start_date, end_date)
            
            # Step 2: Process all data concurrently
            processed_data = await self.process_all_data(raw_data, symbol)
            
            # Step 3: Add pipeline metadata
            processed_data['pipeline_metadata'] = {
                'symbol': symbol,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'pipeline_duration': time.time() - pipeline_start,
                'processed_at': datetime.now().isoformat(),
                'pipeline_version': '2.0_async'
            }
            
            total_time = time.time() - pipeline_start
            logger.info(f"🎉 Full async pipeline completed for {symbol} in {total_time:.2f}s")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Full pipeline failed for {symbol}: {e}")
            raise
    
    async def run_batch_pipeline(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, Dict[str, Any]]:
        """
        Run pipeline for multiple symbols concurrently
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary mapping symbols to their processed data
        """
        logger.info(f"🚀 Starting batch async pipeline for {len(symbols)} symbols")
        batch_start = time.time()
        
        # Create tasks for all symbols
        symbol_tasks = {
            symbol: self.run_full_pipeline(symbol, start_date, end_date)
            for symbol in symbols
        }
        
        # Execute all pipelines concurrently
        try:
            results = await asyncio.gather(*symbol_tasks.values(), return_exceptions=True)
            
            # Combine results
            batch_results = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Batch pipeline failed for {symbol}: {result}")
                    batch_results[symbol] = {"error": str(result)}
                else:
                    batch_results[symbol] = result
            
            total_time = time.time() - batch_start
            successful_count = sum(1 for result in results if not isinstance(result, Exception))
            
            logger.info(f"🎉 Batch pipeline completed: {successful_count}/{len(symbols)} successful in {total_time:.2f}s")
            
            return batch_results
            
        except Exception as e:
            logger.error(f"❌ Batch pipeline failed: {e}")
            raise
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics"""
        return {
            "total_tasks": self.metrics.total_tasks,
            "completed_tasks": self.metrics.completed_tasks,
            "failed_tasks": self.metrics.failed_tasks,
            "success_rate": self.metrics.completed_tasks / max(1, self.metrics.total_tasks),
            "average_fetch_time": self.metrics.total_fetch_time / max(1, self.metrics.completed_tasks),
            "average_transform_time": self.metrics.total_transform_time / max(1, self.metrics.completed_tasks),
            "cache_hit_rate": self.metrics.cache_hits / max(1, self.metrics.cache_hits + self.metrics.cache_misses),
            "max_concurrent_tasks": self.metrics.max_concurrent,
            "current_running_tasks": len(self.running_tasks)
        }


# Convenience functions for easy usage
async def fetch_data_async(symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
    """
    Convenience function to fetch data asynchronously
    
    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        
    Returns:
        Processed data dictionary
    """
    async with AsyncDataPipeline() as pipeline:
        return await pipeline.run_full_pipeline(symbol, start_date, end_date)


async def fetch_batch_data_async(symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to fetch batch data asynchronously
    
    Args:
        symbols: List of stock symbols
        start_date: Start date
        end_date: End date
        
    Returns:
        Dictionary mapping symbols to processed data
    """
    async with AsyncDataPipeline() as pipeline:
        return await pipeline.run_batch_pipeline(symbols, start_date, end_date)


if __name__ == "__main__":
    # Test the async pipeline
    async def test_pipeline():
        symbols = ["AAPL", "MSFT", "GOOGL"]
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=30)
        
        print("Testing async data pipeline...")
        
        # Test single symbol
        print("Testing single symbol...")
        single_result = await fetch_data_async("AAPL", start_date, end_date)
        print(f"Single result keys: {list(single_result.keys())}")
        
        # Test batch
        print("Testing batch...")
        batch_results = await fetch_batch_data_async(symbols, start_date, end_date)
        print(f"Batch results: {len(batch_results)} symbols processed")
        
        print("Async pipeline test completed!")
    
    # Run the test
    asyncio.run(test_pipeline()) 