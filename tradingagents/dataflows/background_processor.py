"""
Background Processing System for TradingAgents

This module handles background computation of expensive operations:
- Technical indicators pre-computation
- Market data preparation
- Cache warming
- Predictive data loading

Author: TradingAgents Performance Team
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import queue
import time
import pickle
from dataclasses import dataclass
from enum import Enum
import weakref

from .cached_api_wrappers import fetch_financialdatasets_prices_cached
from .time_series_cache import get_cache, DataType
from .config import get_config

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BackgroundTask:
    """Represents a background processing task"""
    task_id: str
    task_type: str
    priority: TaskPriority
    symbol: str
    params: Dict[str, Any]
    created_at: datetime
    scheduled_for: Optional[datetime] = None
    callback: Optional[Callable] = None
    

class TechnicalIndicatorProcessor:
    """Handles pre-computation of technical indicators"""
    
    def __init__(self, max_workers: int = 4):
        """Initialize the processor"""
        self.max_workers = max_workers
        self.cache = get_cache()
        self.indicators_cache = {}  # In-memory cache for computed indicators
        self.computing_lock = threading.Lock()
        
        # Supported indicators
        self.supported_indicators = [
            'sma_20', 'sma_50', 'sma_200',
            'ema_12', 'ema_26', 'ema_50',
            'rsi_14', 'rsi_30',
            'macd', 'macd_signal', 'macd_histogram',
            'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
            'atr_14', 'atr_30',
            'vwma_20',
            'stoch_k', 'stoch_d',
            'williams_r',
            'cci_14'
        ]
        
        logger.info(f"🔧 Technical Indicator Processor initialized with {max_workers} workers")
    
    def compute_indicators_for_symbol(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Compute all technical indicators for a symbol"""
        logger.info(f"📊 Computing technical indicators for {symbol} ({start_date.date()} to {end_date.date()})")
        
        try:
            # Get price data
            price_data = fetch_financialdatasets_prices_cached(symbol, start_date, end_date)
            
            if price_data.empty:
                logger.warning(f"No price data for {symbol}")
                return {}
            
            # Compute indicators using vectorized operations
            indicators = self._compute_all_indicators(price_data)
            
            # Cache the results
            cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}"
            self.indicators_cache[cache_key] = {
                'timestamp': datetime.now(),
                'indicators': indicators
            }
            
            logger.info(f"✅ Computed {len(indicators)} indicators for {symbol}")
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Failed to compute indicators for {symbol}: {e}")
            return {}
    
    def _compute_all_indicators(self, price_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Compute all technical indicators using vectorized operations"""
        indicators = {}
        
        # Ensure we have the required columns
        if 'close' not in price_data.columns:
            logger.warning("Price data missing 'close' column")
            return indicators
        
        close_prices = price_data['close']
        high_prices = price_data.get('high', close_prices)
        low_prices = price_data.get('low', close_prices)
        volume = price_data.get('volume', pd.Series([1] * len(close_prices)))
        
        try:
            # Simple Moving Averages
            indicators['sma_20'] = self._create_indicator_df(price_data, 'sma_20', close_prices.rolling(20).mean())
            indicators['sma_50'] = self._create_indicator_df(price_data, 'sma_50', close_prices.rolling(50).mean())
            indicators['sma_200'] = self._create_indicator_df(price_data, 'sma_200', close_prices.rolling(200).mean())
            
            # Exponential Moving Averages
            indicators['ema_12'] = self._create_indicator_df(price_data, 'ema_12', close_prices.ewm(span=12).mean())
            indicators['ema_26'] = self._create_indicator_df(price_data, 'ema_26', close_prices.ewm(span=26).mean())
            indicators['ema_50'] = self._create_indicator_df(price_data, 'ema_50', close_prices.ewm(span=50).mean())
            
            # RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi_14'] = self._create_indicator_df(price_data, 'rsi_14', rsi)
            
            # MACD
            ema_12 = close_prices.ewm(span=12).mean()
            ema_26 = close_prices.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            macd_signal = macd_line.ewm(span=9).mean()
            macd_histogram = macd_line - macd_signal
            
            indicators['macd'] = self._create_indicator_df(price_data, 'macd', macd_line)
            indicators['macd_signal'] = self._create_indicator_df(price_data, 'macd_signal', macd_signal)
            indicators['macd_histogram'] = self._create_indicator_df(price_data, 'macd_histogram', macd_histogram)
            
            # Bollinger Bands
            bb_middle = close_prices.rolling(20).mean()
            bb_std = close_prices.rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            indicators['bollinger_middle'] = self._create_indicator_df(price_data, 'bollinger_middle', bb_middle)
            indicators['bollinger_upper'] = self._create_indicator_df(price_data, 'bollinger_upper', bb_upper)
            indicators['bollinger_lower'] = self._create_indicator_df(price_data, 'bollinger_lower', bb_lower)
            
            # ATR (Average True Range)
            high_low = high_prices - low_prices
            high_close = abs(high_prices - close_prices.shift())
            low_close = abs(low_prices - close_prices.shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean()
            indicators['atr_14'] = self._create_indicator_df(price_data, 'atr_14', atr)
            
            # VWMA (Volume Weighted Moving Average)
            vwma = (close_prices * volume).rolling(20).sum() / volume.rolling(20).sum()
            indicators['vwma_20'] = self._create_indicator_df(price_data, 'vwma_20', vwma)
            
            logger.info(f"📈 Computed {len(indicators)} technical indicators")
            
        except Exception as e:
            logger.error(f"❌ Error computing indicators: {e}")
        
        return indicators
    
    def _create_indicator_df(self, price_data: pd.DataFrame, indicator_name: str, values: pd.Series) -> pd.DataFrame:
        """Create a standardized DataFrame for an indicator"""
        return pd.DataFrame({
            'date': price_data.get('date', price_data.index),
            'symbol': price_data.get('symbol', ['UNKNOWN'] * len(values)),
            'indicator': indicator_name,
            'value': values
        })
    
    def get_cached_indicators(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[Dict[str, pd.DataFrame]]:
        """Get cached indicators if available and fresh"""
        cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}"
        
        if cache_key in self.indicators_cache:
            cached_data = self.indicators_cache[cache_key]
            # Check if cache is still fresh (less than 1 hour old)
            if (datetime.now() - cached_data['timestamp']).seconds < 3600:
                logger.info(f"📊 Using cached indicators for {symbol}")
                return cached_data['indicators']
        
        return None
    
    def schedule_indicator_computation(self, symbols: List[str], dates: List[datetime]) -> List[str]:
        """Schedule indicator computation for multiple symbols and dates"""
        task_ids = []
        
        for symbol in symbols:
            for date in dates:
                start_date = date - timedelta(days=200)  # Need historical data for indicators
                end_date = date
                
                task_id = f"indicators_{symbol}_{date.strftime('%Y%m%d')}"
                task_ids.append(task_id)
                
                # Schedule computation (this would be queued in a real background system)
                logger.info(f"📅 Scheduled indicator computation: {task_id}")
        
        return task_ids


class BackgroundJobManager:
    """Manages background processing jobs"""
    
    def __init__(self, max_workers: int = 8):
        """Initialize the background job manager"""
        self.max_workers = max_workers
        self.task_queue = queue.PriorityQueue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running_tasks = {}
        self.completed_tasks = {}
        self.is_running = False
        
        # Processors
        self.indicator_processor = TechnicalIndicatorProcessor()
        
        logger.info(f"🚀 Background Job Manager initialized with {max_workers} workers")
    
    def start(self):
        """Start the background processing"""
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info("▶️ Background job manager started")
    
    def stop(self):
        """Stop the background processing"""
        self.is_running = False
        if hasattr(self, 'worker_thread'):
            self.worker_thread.join(timeout=5)
        
        self.executor.shutdown(wait=True)
        logger.info("⏹️ Background job manager stopped")
    
    def _worker_loop(self):
        """Main worker loop"""
        while self.is_running:
            try:
                # Get next task (blocks for up to 1 second)
                priority, task = self.task_queue.get(timeout=1)
                
                if task:
                    self._execute_task(task)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Worker loop error: {e}")
    
    def _execute_task(self, task: BackgroundTask):
        """Execute a background task"""
        logger.info(f"🔄 Executing task: {task.task_id}")
        
        try:
            self.running_tasks[task.task_id] = task
            
            if task.task_type == "compute_indicators":
                self._execute_indicator_task(task)
            elif task.task_type == "warm_cache":
                self._execute_cache_warming_task(task)
            elif task.task_type == "preload_data":
                self._execute_preload_task(task)
            else:
                logger.warning(f"Unknown task type: {task.task_type}")
            
            # Mark as completed
            self.completed_tasks[task.task_id] = {
                'task': task,
                'completed_at': datetime.now(),
                'status': 'success'
            }
            
            # Call callback if provided
            if task.callback:
                task.callback(task.task_id, 'success', None)
                
            logger.info(f"✅ Task completed: {task.task_id}")
            
        except Exception as e:
            logger.error(f"❌ Task failed: {task.task_id} - {e}")
            
            self.completed_tasks[task.task_id] = {
                'task': task,
                'completed_at': datetime.now(),
                'status': 'failed',
                'error': str(e)
            }
            
            if task.callback:
                task.callback(task.task_id, 'failed', str(e))
        
        finally:
            self.running_tasks.pop(task.task_id, None)
    
    def _execute_indicator_task(self, task: BackgroundTask):
        """Execute technical indicator computation task"""
        symbol = task.symbol
        start_date = task.params['start_date']
        end_date = task.params['end_date']
        
        indicators = self.indicator_processor.compute_indicators_for_symbol(
            symbol, start_date, end_date
        )
        
        # Store results in task for retrieval
        task.params['results'] = indicators
    
    def _execute_cache_warming_task(self, task: BackgroundTask):
        """Execute cache warming task"""
        symbol = task.symbol
        start_date = task.params['start_date']
        end_date = task.params['end_date']
        
        # Warm cache with price data
        fetch_financialdatasets_prices_cached(symbol, start_date, end_date)
        
        logger.info(f"🔥 Cache warmed for {symbol}")
    
    def _execute_preload_task(self, task: BackgroundTask):
        """Execute data preloading task"""
        # This would implement predictive data loading
        logger.info(f"📦 Preloading data for {task.symbol}")
    
    def schedule_task(self, task: BackgroundTask) -> str:
        """Schedule a background task"""
        # Priority queue uses tuple (priority, task)
        priority_value = 5 - task.priority.value  # Invert so higher priority runs first
        self.task_queue.put((priority_value, task))
        
        logger.info(f"📅 Scheduled task: {task.task_id} (Priority: {task.priority.name})")
        return task.task_id
    
    def schedule_indicator_computation(self, symbol: str, start_date: datetime, end_date: datetime, 
                                     priority: TaskPriority = TaskPriority.MEDIUM) -> str:
        """Convenience method to schedule indicator computation"""
        task_id = f"indicators_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        task = BackgroundTask(
            task_id=task_id,
            task_type="compute_indicators",
            priority=priority,
            symbol=symbol,
            params={
                'start_date': start_date,
                'end_date': end_date
            },
            created_at=datetime.now()
        )
        
        return self.schedule_task(task)
    
    def schedule_cache_warming(self, symbols: List[str], dates: List[datetime], 
                              priority: TaskPriority = TaskPriority.LOW) -> List[str]:
        """Schedule cache warming for multiple symbols"""
        task_ids = []
        
        for symbol in symbols:
            for date in dates:
                start_date = date - timedelta(days=30)
                end_date = date
                
                task_id = f"cache_warm_{symbol}_{date.strftime('%Y%m%d')}"
                
                task = BackgroundTask(
                    task_id=task_id,
                    task_type="warm_cache",
                    priority=priority,
                    symbol=symbol,
                    params={
                        'start_date': start_date,
                        'end_date': end_date
                    },
                    created_at=datetime.now()
                )
                
                task_ids.append(self.schedule_task(task))
        
        return task_ids
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a task"""
        if task_id in self.running_tasks:
            return {'status': 'running', 'task': self.running_tasks[task_id]}
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        else:
            return {'status': 'not_found'}
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get background processing statistics"""
        return {
            'queue_size': self.task_queue.qsize(),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'max_workers': self.max_workers,
            'is_running': self.is_running
        }


# Global background job manager instance
_background_manager = None

def get_background_manager() -> BackgroundJobManager:
    """Get or create the global background job manager"""
    global _background_manager
    if _background_manager is None:
        _background_manager = BackgroundJobManager()
        _background_manager.start()
    return _background_manager


def precompute_indicators_for_analysis(symbols: List[str], analysis_dates: List[datetime]) -> Dict[str, str]:
    """
    Precompute technical indicators for upcoming analysis
    
    Args:
        symbols: List of symbols to compute indicators for
        analysis_dates: List of analysis dates
        
    Returns:
        Dictionary mapping symbols to task IDs
    """
    manager = get_background_manager()
    task_ids = {}
    
    for symbol in symbols:
        for date in analysis_dates:
            start_date = date - timedelta(days=200)  # Need historical data
            end_date = date
            
            task_id = manager.schedule_indicator_computation(
                symbol, start_date, end_date, TaskPriority.HIGH
            )
            task_ids[f"{symbol}_{date.strftime('%Y-%m-%d')}"] = task_id
    
    logger.info(f"📊 Scheduled indicator computation for {len(symbols)} symbols, {len(analysis_dates)} dates")
    return task_ids


if __name__ == "__main__":
    # Example usage
    import time
    
    # Test the background processing system
    manager = BackgroundJobManager()
    manager.start()
    
    # Schedule some tasks
    symbols = ["AAPL", "MSFT", "GOOGL"]
    dates = [datetime.now() - timedelta(days=i) for i in range(1, 6)]
    
    # Schedule indicator computation
    task_ids = []
    for symbol in symbols:
        for date in dates:
            start_date = date - timedelta(days=200)
            task_id = manager.schedule_indicator_computation(symbol, start_date, date)
            task_ids.append(task_id)
    
    print(f"Scheduled {len(task_ids)} tasks")
    
    # Wait a bit and check stats
    time.sleep(5)
    stats = manager.get_queue_stats()
    print(f"Queue stats: {stats}")
    
    # Cleanup
    manager.stop() 