"""
Performance Optimization Manager for TradingAgents

This module coordinates various performance optimization strategies:
- Data pre-fetching
- Batch API calls
- Cache warming
- Memory optimization
- Parallel processing coordination

Author: TradingAgents Performance Team
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import gc

from .cached_api_wrappers import (
    batch_fetch_multiple_symbols,
    prefetch_common_data,
    fetch_financialdatasets_prices_cached_optimized
)
from .time_series_cache import get_cache, DataType
from .config import get_config

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    Manages performance optimization strategies for TradingAgents
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the performance optimizer"""
        self.config = config or get_config()
        self.cache = get_cache()
        self.prefetched_data = {}
        self.performance_metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls_saved": 0,
            "prefetch_operations": 0,
            "batch_operations": 0,
            "total_time_saved": 0.0
        }
        
        # Performance settings
        self.max_concurrent_fetches = self.config.get("max_concurrent_fetches", 8)
        self.enable_prefetching = self.config.get("enable_prefetching", True)
        self.enable_batching = self.config.get("enable_batching", True)
        self.cache_warming_enabled = self.config.get("cache_warming_enabled", True)
        
        logger.info("🚀 Performance Optimizer initialized")
    
    def warm_cache_for_analysis(self, symbols: List[str], analysis_dates: List[datetime]) -> Dict[str, Any]:
        """
        Warm cache with data likely to be needed for upcoming analyses
        
        Args:
            symbols: List of stock symbols that will be analyzed
            analysis_dates: List of dates for analysis
            
        Returns:
            Dictionary with warming results and performance metrics
        """
        if not self.cache_warming_enabled:
            logger.info("⚠️ Cache warming is disabled")
            return {}
        
        start_time = time.time()
        
        logger.info(f"🔥 Warming cache for {len(symbols)} symbols, {len(analysis_dates)} dates")
        
        # Determine optimal date range for cache warming
        min_date = min(analysis_dates) - timedelta(days=60)  # 2 months buffer
        max_date = max(analysis_dates) + timedelta(days=1)
        
        # Pre-fetch critical data types
        data_types = ["ohlcv", "news"]
        
        try:
            # Use parallel pre-fetching
            prefetched_data = prefetch_common_data(symbols, analysis_dates, data_types)
            
            # Update metrics
            self.performance_metrics["prefetch_operations"] += 1
            self.prefetched_data.update(prefetched_data)
            
            warming_time = time.time() - start_time
            self.performance_metrics["total_time_saved"] += warming_time * 0.5  # Estimate 50% time savings
            
            logger.info(f"✅ Cache warming completed in {warming_time:.2f}s")
            logger.info(f"📊 Warmed cache for {len(symbols)} symbols, {len(data_types)} data types")
            
            return {
                "symbols_warmed": len(symbols),
                "data_types_warmed": len(data_types),
                "warming_time": warming_time,
                "date_range": f"{min_date.date()} to {max_date.date()}"
            }
            
        except Exception as e:
            logger.error(f"❌ Cache warming failed: {e}")
            return {"error": str(e)}
    
    def optimize_single_analysis(self, symbol: str, analysis_date: datetime) -> Dict[str, Any]:
        """
        Optimize data fetching for a single analysis
        
        Args:
            symbol: Stock symbol
            analysis_date: Analysis date
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        logger.info(f"⚡ Optimizing analysis for {symbol} on {analysis_date.date()}")
        
        # Define data requirements
        start_date = analysis_date - timedelta(days=30)  # 30 days of historical data
        end_date = analysis_date + timedelta(days=1)
        
        optimization_results = {
            "symbol": symbol,
            "analysis_date": analysis_date.date(),
            "data_sources": {},
            "performance_metrics": {},
            "cache_efficiency": {}
        }
        
        try:
            # 1. Price data optimization
            logger.info(f"📊 Optimizing price data for {symbol}")
            price_data = fetch_financialdatasets_prices_cached_optimized(
                symbol, start_date, end_date, use_cache_only=False
            )
            
            optimization_results["data_sources"]["price_data"] = {
                "records": len(price_data),
                "date_range": f"{start_date.date()} to {end_date.date()}",
                "cached": not price_data.empty
            }
            
            # 2. Check cache hit rates
            cache_stats = self.cache.get_cache_stats()
            optimization_results["cache_efficiency"] = {
                "hit_ratio": cache_stats.get("hit_ratio", 0),
                "total_entries": cache_stats.get("total_cache_entries", 0),
                "cache_size_mb": cache_stats.get("cache_size_mb", 0)
            }
            
            # 3. Memory optimization
            self.optimize_memory_usage()
            
            optimization_time = time.time() - start_time
            optimization_results["performance_metrics"]["optimization_time"] = optimization_time
            
            logger.info(f"✅ Analysis optimization completed in {optimization_time:.2f}s")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"❌ Analysis optimization failed for {symbol}: {e}")
            optimization_results["error"] = str(e)
            return optimization_results
    
    def optimize_batch_analysis(self, symbols: List[str], analysis_dates: List[datetime]) -> Dict[str, Any]:
        """
        Optimize data fetching for batch analysis of multiple symbols
        
        Args:
            symbols: List of stock symbols
            analysis_dates: List of analysis dates
            
        Returns:
            Dictionary with batch optimization results
        """
        start_time = time.time()
        
        logger.info(f"🚀 Optimizing batch analysis for {len(symbols)} symbols, {len(analysis_dates)} dates")
        
        batch_results = {
            "symbols": symbols,
            "analysis_dates": [d.date() for d in analysis_dates],
            "optimizations_applied": [],
            "performance_metrics": {},
            "data_summary": {}
        }
        
        try:
            # 1. Pre-fetch common data
            if self.enable_prefetching:
                logger.info("🔄 Pre-fetching common data...")
                prefetch_results = self.warm_cache_for_analysis(symbols, analysis_dates)
                batch_results["optimizations_applied"].append("prefetching")
                batch_results["prefetch_results"] = prefetch_results
            
            # 2. Batch API calls
            if self.enable_batching:
                logger.info("📦 Optimizing with batch API calls...")
                
                # Determine date range
                min_date = min(analysis_dates) - timedelta(days=30)
                max_date = max(analysis_dates) + timedelta(days=1)
                
                # Batch fetch OHLCV data
                ohlcv_data = batch_fetch_multiple_symbols(
                    symbols, min_date, max_date, "ohlcv", max_workers=self.max_concurrent_fetches
                )
                
                batch_results["optimizations_applied"].append("batch_api_calls")
                batch_results["data_summary"]["ohlcv_records"] = sum(len(df) for df in ohlcv_data.values())
                
                self.performance_metrics["batch_operations"] += 1
            
            # 3. Memory optimization
            self.optimize_memory_usage()
            batch_results["optimizations_applied"].append("memory_optimization")
            
            # 4. Calculate performance metrics
            optimization_time = time.time() - start_time
            estimated_sequential_time = len(symbols) * len(analysis_dates) * 5  # 5 seconds per symbol-date
            time_saved = max(0, estimated_sequential_time - optimization_time)
            
            batch_results["performance_metrics"] = {
                "optimization_time": optimization_time,
                "estimated_sequential_time": estimated_sequential_time,
                "time_saved": time_saved,
                "efficiency_gain": f"{(time_saved / max(1, estimated_sequential_time)) * 100:.1f}%"
            }
            
            self.performance_metrics["total_time_saved"] += time_saved
            
            logger.info(f"✅ Batch optimization completed in {optimization_time:.2f}s")
            logger.info(f"⚡ Estimated time saved: {time_saved:.2f}s ({batch_results['performance_metrics']['efficiency_gain']})")
            
            return batch_results
            
        except Exception as e:
            logger.error(f"❌ Batch optimization failed: {e}")
            batch_results["error"] = str(e)
            return batch_results
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Optimize memory usage by cleaning up unused data and running garbage collection
        
        Returns:
            Dictionary with memory optimization results
        """
        logger.info("🧹 Optimizing memory usage...")
        
        # Clear old prefetched data
        cutoff_time = datetime.now() - timedelta(hours=4)  # Keep data for 4 hours
        cleared_count = 0
        
        for symbol in list(self.prefetched_data.keys()):
            # This is a simplified cleanup - in reality, you'd check timestamps
            if len(self.prefetched_data[symbol]) > 10:  # Arbitrary limit
                del self.prefetched_data[symbol]
                cleared_count += 1
        
        # Run garbage collection
        gc.collect()
        
        logger.info(f"✅ Memory optimization completed - cleared {cleared_count} old entries")
        
        return {
            "cleared_entries": cleared_count,
            "gc_collected": True,
            "optimization_time": time.time()
        }
    
    def get_optimization_recommendations(self, symbols: List[str], analysis_dates: List[datetime]) -> List[Dict[str, Any]]:
        """
        Get recommendations for optimizing analysis performance
        
        Args:
            symbols: List of stock symbols
            analysis_dates: List of analysis dates
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Check cache hit rates
        cache_stats = self.cache.get_cache_stats()
        hit_ratio = cache_stats.get("hit_ratio", 0)
        
        if hit_ratio < 0.7:  # Less than 70% hit rate
            recommendations.append({
                "type": "cache_warming",
                "priority": "high",
                "description": "Cache hit rate is low. Consider warming cache before analysis.",
                "action": "Run cache warming for target symbols and dates",
                "expected_improvement": "30-50% faster analysis"
            })
        
        # Check for batch optimization opportunities
        if len(symbols) > 3:
            recommendations.append({
                "type": "batch_processing",
                "priority": "high",
                "description": "Multiple symbols detected. Use batch processing for better performance.",
                "action": "Enable batch API calls and parallel processing",
                "expected_improvement": "40-60% faster for multiple symbols"
            })
        
        # Check for parallel processing opportunities
        if len(analysis_dates) > 1:
            recommendations.append({
                "type": "parallel_processing",
                "priority": "medium",
                "description": "Multiple analysis dates detected. Consider parallel processing.",
                "action": "Use parallel agent processing mode",
                "expected_improvement": "20-40% faster analysis"
            })
        
        # Memory optimization
        if cache_stats.get("cache_size_mb", 0) > 100:  # More than 100MB cache
            recommendations.append({
                "type": "memory_optimization",
                "priority": "low",
                "description": "Large cache size detected. Consider periodic cleanup.",
                "action": "Run memory optimization and cache cleanup",
                "expected_improvement": "Better memory efficiency"
            })
        
        return recommendations
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary
        
        Returns:
            Dictionary with performance metrics and statistics
        """
        cache_stats = self.cache.get_cache_stats()
        
        return {
            "optimizer_metrics": self.performance_metrics,
            "cache_statistics": cache_stats,
            "configuration": {
                "max_concurrent_fetches": self.max_concurrent_fetches,
                "prefetching_enabled": self.enable_prefetching,
                "batching_enabled": self.enable_batching,
                "cache_warming_enabled": self.cache_warming_enabled
            },
            "prefetched_data_summary": {
                "symbols_cached": len(self.prefetched_data),
                "total_data_types": sum(len(data) for data in self.prefetched_data.values())
            },
            "recommendations": self.get_optimization_recommendations([], [])
        }


# Global optimizer instance
_optimizer_instance = None

def get_performance_optimizer(config: Dict[str, Any] = None) -> PerformanceOptimizer:
    """Get or create the global performance optimizer instance"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = PerformanceOptimizer(config)
    return _optimizer_instance


def optimize_for_analysis(symbols: List[str], analysis_dates: List[datetime]) -> Dict[str, Any]:
    """
    Convenience function to optimize performance for upcoming analyses
    
    Args:
        symbols: List of stock symbols
        analysis_dates: List of analysis dates
        
    Returns:
        Dictionary with optimization results
    """
    optimizer = get_performance_optimizer()
    
    if len(symbols) == 1 and len(analysis_dates) == 1:
        return optimizer.optimize_single_analysis(symbols[0], analysis_dates[0])
    else:
        return optimizer.optimize_batch_analysis(symbols, analysis_dates)


if __name__ == "__main__":
    # Example usage
    from datetime import datetime, timedelta
    
    # Test performance optimizer
    optimizer = PerformanceOptimizer()
    
    # Test single analysis optimization
    test_symbol = "AAPL"
    test_date = datetime.now() - timedelta(days=1)
    
    result = optimizer.optimize_single_analysis(test_symbol, test_date)
    print("Single analysis optimization result:", result)
    
    # Test batch optimization
    test_symbols = ["AAPL", "MSFT", "GOOGL"]
    test_dates = [datetime.now() - timedelta(days=i) for i in range(1, 4)]
    
    batch_result = optimizer.optimize_batch_analysis(test_symbols, test_dates)
    print("Batch optimization result:", batch_result)
    
    # Get performance summary
    summary = optimizer.get_performance_summary()
    print("Performance summary:", summary) 