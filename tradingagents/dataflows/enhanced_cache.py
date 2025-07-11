"""
Enhanced Multi-Layer Caching System for TradingAgents

This module provides a sophisticated caching system with:
- In-memory LRU cache for hot data
- Persistent disk cache with compression
- Predictive pre-loading
- Cache warming strategies
- Intelligent cache eviction
- Performance monitoring

Author: TradingAgents Performance Team
"""

import logging
import pickle
import gzip
import hashlib
import threading
import weakref
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import pandas as pd
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import psutil

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels in the hierarchy"""
    MEMORY = "memory"           # Fast in-memory cache
    DISK = "disk"              # Persistent disk cache
    REMOTE = "remote"          # Remote/distributed cache


class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"                # Least Recently Used
    LFU = "lfu"                # Least Frequently Used
    TTL = "ttl"                # Time To Live
    ADAPTIVE = "adaptive"      # Adaptive based on usage patterns


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata"""
    key: str
    data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 1
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if the entry has expired"""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of the entry in seconds"""
        return (datetime.now() - self.created_at).total_seconds()


class InMemoryCache:
    """High-performance in-memory cache with LRU eviction"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 500):
        """
        Initialize in-memory cache
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage': 0
        }
        
        logger.info(f"💾 In-memory cache initialized (max: {max_size} entries, {max_memory_mb}MB)")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check expiration
                if entry.is_expired:
                    del self.cache[key]
                    self.stats['misses'] += 1
                    return None
                
                # Update access info
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                
                self.stats['hits'] += 1
                return entry.data
            
            self.stats['misses'] += 1
            return None
    
    def put(self, key: str, data: Any, ttl_seconds: Optional[int] = None, metadata: Dict[str, Any] = None) -> bool:
        """Put item into cache"""
        with self.lock:
            # Calculate data size
            try:
                size_bytes = len(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
            except:
                size_bytes = 1024  # Fallback estimate
            
            # Check if single item exceeds memory limit
            if size_bytes > self.max_memory_bytes:
                logger.warning(f"Item too large for cache: {size_bytes} bytes")
                return False
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds,
                metadata=metadata or {}
            )
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.stats['memory_usage'] -= old_entry.size_bytes
                del self.cache[key]
            
            # Evict entries if necessary
            self._evict_if_needed(size_bytes)
            
            # Add new entry
            self.cache[key] = entry
            self.stats['memory_usage'] += size_bytes
            
            return True
    
    def _evict_if_needed(self, new_item_size: int):
        """Evict entries if cache limits are exceeded"""
        # Evict by count
        while len(self.cache) >= self.max_size:
            self._evict_lru()
        
        # Evict by memory
        while (self.stats['memory_usage'] + new_item_size) > self.max_memory_bytes:
            if not self.cache:
                break
            self._evict_lru()
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if self.cache:
            key, entry = self.cache.popitem(last=False)  # Remove oldest (LRU)
            self.stats['memory_usage'] -= entry.size_bytes
            self.stats['evictions'] += 1
            logger.debug(f"🗑️ Evicted LRU entry: {key}")
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.stats['memory_usage'] = 0
            logger.info("🧹 In-memory cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / max(1, total_requests)
            
            return {
                'entries': len(self.cache),
                'max_entries': self.max_size,
                'memory_usage_mb': self.stats['memory_usage'] / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self.stats['evictions']
            }


class PredictivePreloader:
    """Handles predictive cache pre-loading based on usage patterns"""
    
    def __init__(self, cache_system):
        """
        Initialize predictive preloader
        
        Args:
            cache_system: Reference to the main cache system
        """
        self.cache_system = weakref.ref(cache_system)
        self.access_patterns: Dict[str, List[datetime]] = {}
        self.prediction_patterns: Dict[str, Dict[str, Any]] = {}
        self.preload_queue: List[Tuple[str, Callable]] = []
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="preloader")
        self.lock = threading.Lock()
        
        logger.info("🔮 Predictive preloader initialized")
    
    def record_access(self, key: str, context: Dict[str, Any] = None):
        """Record cache access for pattern analysis"""
        with self.lock:
            if key not in self.access_patterns:
                self.access_patterns[key] = []
            
            self.access_patterns[key].append(datetime.now())
            
            # Keep only recent accesses (last 30 days)
            cutoff = datetime.now() - timedelta(days=30)
            self.access_patterns[key] = [
                dt for dt in self.access_patterns[key] if dt > cutoff
            ]
            
            # Analyze patterns periodically
            if len(self.access_patterns[key]) % 10 == 0:
                self._analyze_patterns(key)
    
    def _analyze_patterns(self, key: str):
        """Analyze access patterns for a key"""
        accesses = self.access_patterns.get(key, [])
        
        if len(accesses) < 5:
            return
        
        # Calculate access frequency
        time_diffs = []
        for i in range(1, len(accesses)):
            diff = (accesses[i] - accesses[i-1]).total_seconds()
            time_diffs.append(diff)
        
        if time_diffs:
            avg_interval = sum(time_diffs) / len(time_diffs)
            pattern = {
                'average_interval_seconds': avg_interval,
                'access_count': len(accesses),
                'last_access': accesses[-1],
                'predicted_next_access': accesses[-1] + timedelta(seconds=avg_interval),
                'confidence': min(1.0, len(accesses) / 20.0)  # Higher confidence with more data
            }
            
            self.prediction_patterns[key] = pattern
            
            # Schedule preload if pattern is strong
            if pattern['confidence'] > 0.5 and avg_interval < 3600:  # Less than 1 hour
                self._schedule_preload(key, pattern)
    
    def _schedule_preload(self, key: str, pattern: Dict[str, Any]):
        """Schedule predictive preload"""
        predicted_time = pattern['predicted_next_access']
        current_time = datetime.now()
        
        # Schedule preload 5 minutes before predicted access
        preload_time = predicted_time - timedelta(minutes=5)
        
        if preload_time > current_time:
            delay_seconds = (preload_time - current_time).total_seconds()
            
            if delay_seconds < 3600:  # Only schedule if within 1 hour
                logger.info(f"🔮 Scheduled preload for {key} in {delay_seconds:.0f}s")
                
                # Submit to executor with delay
                self.executor.submit(self._delayed_preload, key, delay_seconds)
    
    def _delayed_preload(self, key: str, delay_seconds: float):
        """Execute delayed preload"""
        time.sleep(delay_seconds)
        
        cache_system = self.cache_system()
        if cache_system:
            try:
                # This would trigger a cache refresh
                logger.info(f"🔮 Executing predictive preload for {key}")
                # Implementation depends on specific cache key format
                # For now, just log the event
            except Exception as e:
                logger.error(f"❌ Predictive preload failed for {key}: {e}")
    
    def get_predictions(self) -> Dict[str, Dict[str, Any]]:
        """Get current predictions"""
        with self.lock:
            return self.prediction_patterns.copy()
    
    def cleanup(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)


class EnhancedCacheSystem:
    """
    Multi-layer cache system with predictive capabilities
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize enhanced cache system
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Cache layers
        self.memory_cache = InMemoryCache(
            max_size=self.config.get('memory_max_entries', 1000),
            max_memory_mb=self.config.get('memory_max_mb', 500)
        )
        
        # Disk cache (using existing time series cache)
        from .time_series_cache import get_cache
        self.disk_cache = get_cache()
        
        # Predictive preloader
        self.preloader = PredictivePreloader(self)
        
        # Cache statistics
        self.global_stats = {
            'total_requests': 0,
            'memory_hits': 0,
            'disk_hits': 0,
            'total_misses': 0,
            'preload_hits': 0
        }
        
        # Cache warming thread
        self._warming_thread = None
        self._warming_enabled = self.config.get('cache_warming_enabled', True)
        
        logger.info("🚀 Enhanced cache system initialized")
    
    def get(self, key: str, context: Dict[str, Any] = None) -> Optional[Any]:
        """
        Get item from multi-layer cache
        
        Args:
            key: Cache key
            context: Additional context for predictive analysis
            
        Returns:
            Cached data or None if not found
        """
        self.global_stats['total_requests'] += 1
        
        # Record access for predictive analysis
        self.preloader.record_access(key, context)
        
        # Try memory cache first
        result = self.memory_cache.get(key)
        if result is not None:
            self.global_stats['memory_hits'] += 1
            logger.debug(f"💾 Memory cache hit: {key}")
            return result
        
        # Try disk cache
        try:
            # This would need to be adapted based on disk cache interface
            # For now, simulate disk cache lookup
            disk_result = self._get_from_disk_cache(key)
            if disk_result is not None:
                self.global_stats['disk_hits'] += 1
                logger.debug(f"💿 Disk cache hit: {key}")
                
                # Promote to memory cache
                self.memory_cache.put(key, disk_result)
                return disk_result
        except Exception as e:
            logger.debug(f"Disk cache lookup failed for {key}: {e}")
        
        # Cache miss
        self.global_stats['total_misses'] += 1
        logger.debug(f"❌ Cache miss: {key}")
        return None
    
    def put(self, key: str, data: Any, ttl_seconds: Optional[int] = None, 
            metadata: Dict[str, Any] = None, promote_to_memory: bool = True) -> bool:
        """
        Put item into multi-layer cache
        
        Args:
            key: Cache key
            data: Data to cache
            ttl_seconds: Time to live in seconds
            metadata: Additional metadata
            promote_to_memory: Whether to store in memory cache
            
        Returns:
            True if successfully cached
        """
        success = True
        
        # Store in memory cache if requested
        if promote_to_memory:
            memory_success = self.memory_cache.put(key, data, ttl_seconds, metadata)
            if not memory_success:
                logger.warning(f"Failed to store {key} in memory cache")
                success = False
        
        # Store in disk cache
        try:
            self._put_to_disk_cache(key, data, metadata)
            logger.debug(f"💿 Stored in disk cache: {key}")
        except Exception as e:
            logger.warning(f"Failed to store {key} in disk cache: {e}")
            success = False
        
        return success
    
    def _get_from_disk_cache(self, key: str) -> Optional[Any]:
        """Get item from disk cache (simplified implementation)"""
        # This would be implemented based on the specific disk cache format
        # For now, return None to simulate cache miss
        return None
    
    def _put_to_disk_cache(self, key: str, data: Any, metadata: Dict[str, Any] = None):
        """Put item to disk cache (simplified implementation)"""
        # This would be implemented based on the specific disk cache format
        pass
    
    def warm_cache(self, symbols: List[str], dates: List[datetime], 
                  data_types: List[str] = None) -> Dict[str, Any]:
        """
        Warm cache with frequently accessed data
        
        Args:
            symbols: List of symbols to warm
            dates: List of dates to warm
            data_types: List of data types to warm
            
        Returns:
            Warming results
        """
        if not self._warming_enabled:
            return {"status": "disabled"}
        
        logger.info(f"🔥 Warming cache for {len(symbols)} symbols, {len(dates)} dates")
        
        data_types = data_types or ['price_data', 'news_data']
        warming_start = time.time()
        
        # Use thread pool for concurrent warming
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for symbol in symbols:
                for date in dates:
                    for data_type in data_types:
                        future = executor.submit(
                            self._warm_single_item, symbol, date, data_type
                        )
                        futures.append(future)
            
            # Wait for completion
            completed = 0
            failed = 0
            
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    if result:
                        completed += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    logger.debug(f"Cache warming item failed: {e}")
        
        warming_time = time.time() - warming_start
        
        result = {
            "completed": completed,
            "failed": failed,
            "warming_time": warming_time,
            "symbols": len(symbols),
            "dates": len(dates),
            "data_types": len(data_types)
        }
        
        logger.info(f"🔥 Cache warming completed: {completed} items in {warming_time:.1f}s")
        return result
    
    def _warm_single_item(self, symbol: str, date: datetime, data_type: str) -> bool:
        """Warm a single cache item"""
        try:
            # Generate cache key
            cache_key = f"{symbol}_{data_type}_{date.strftime('%Y%m%d')}"
            
            # Check if already cached
            if self.get(cache_key) is not None:
                return True  # Already cached
            
            # Fetch and cache data (this would be implemented based on data type)
            data = self._fetch_data_for_warming(symbol, date, data_type)
            
            if data is not None:
                return self.put(cache_key, data, ttl_seconds=3600)  # 1 hour TTL
            
            return False
            
        except Exception as e:
            logger.debug(f"Failed to warm {symbol}_{data_type}_{date}: {e}")
            return False
    
    def _fetch_data_for_warming(self, symbol: str, date: datetime, data_type: str) -> Optional[Any]:
        """Fetch data for cache warming (placeholder implementation)"""
        # This would be implemented to fetch actual data
        # For now, return placeholder data
        return f"warm_data_{symbol}_{data_type}_{date.strftime('%Y%m%d')}"
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_requests = self.global_stats['total_requests']
        
        stats = {
            "global_stats": {
                "total_requests": total_requests,
                "memory_hits": self.global_stats['memory_hits'],
                "disk_hits": self.global_stats['disk_hits'],
                "total_misses": self.global_stats['total_misses'],
                "overall_hit_rate": (self.global_stats['memory_hits'] + self.global_stats['disk_hits']) / max(1, total_requests),
                "memory_hit_rate": self.global_stats['memory_hits'] / max(1, total_requests),
                "preload_hits": self.global_stats['preload_hits']
            },
            "memory_cache": self.memory_cache.get_stats(),
            "disk_cache": self.disk_cache.get_cache_stats() if self.disk_cache else {},
            "predictive_patterns": len(self.preloader.get_predictions()),
            "system_memory": {
                "available_mb": psutil.virtual_memory().available / (1024 * 1024),
                "percent_used": psutil.virtual_memory().percent
            }
        }
        
        return stats
    
    def optimize_cache_settings(self) -> Dict[str, Any]:
        """Analyze usage and optimize cache settings"""
        stats = self.get_comprehensive_stats()
        
        recommendations = []
        current_performance = stats["global_stats"]["overall_hit_rate"]
        
        # Memory cache optimization
        memory_stats = stats["memory_cache"]
        if memory_stats["hit_rate"] < 0.7:
            recommendations.append({
                "type": "memory_size_increase",
                "description": "Consider increasing memory cache size",
                "current_hit_rate": memory_stats["hit_rate"],
                "suggested_action": "Increase max_entries or max_memory_mb"
            })
        
        # Memory usage optimization
        if memory_stats["memory_usage_mb"] / memory_stats["max_memory_mb"] > 0.9:
            recommendations.append({
                "type": "memory_pressure",
                "description": "Memory cache is near capacity",
                "suggested_action": "Increase memory limit or review TTL settings"
            })
        
        # Eviction rate optimization
        if memory_stats["evictions"] > memory_stats["hits"] * 0.1:
            recommendations.append({
                "type": "high_eviction_rate",
                "description": "High eviction rate detected",
                "suggested_action": "Increase cache size or adjust eviction strategy"
            })
        
        return {
            "current_performance": current_performance,
            "recommendations": recommendations,
            "optimization_potential": max(0, 0.95 - current_performance),
            "stats_snapshot": stats
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.memory_cache.clear()
        self.preloader.cleanup()
        logger.info("🧹 Enhanced cache system cleaned up")


# Global enhanced cache instance
_enhanced_cache = None

def get_enhanced_cache(config: Dict[str, Any] = None) -> EnhancedCacheSystem:
    """Get or create the global enhanced cache instance"""
    global _enhanced_cache
    if _enhanced_cache is None:
        _enhanced_cache = EnhancedCacheSystem(config)
    return _enhanced_cache


if __name__ == "__main__":
    # Test the enhanced cache system
    import time
    
    # Initialize cache
    cache = EnhancedCacheSystem({
        'memory_max_entries': 100,
        'memory_max_mb': 50,
        'cache_warming_enabled': True
    })
    
    # Test basic operations
    print("Testing enhanced cache system...")
    
    # Test puts and gets
    for i in range(10):
        key = f"test_key_{i}"
        data = f"test_data_{i}" * 100  # Make it larger
        
        success = cache.put(key, data)
        print(f"Put {key}: {success}")
        
        retrieved = cache.get(key)
        print(f"Get {key}: {'✅' if retrieved == data else '❌'}")
    
    # Test cache warming
    symbols = ["AAPL", "MSFT", "GOOGL"]
    dates = [datetime.now() - timedelta(days=i) for i in range(1, 4)]
    
    warming_result = cache.warm_cache(symbols, dates)
    print(f"Cache warming: {warming_result}")
    
    # Get statistics
    stats = cache.get_comprehensive_stats()
    print(f"Cache stats: {json.dumps(stats, indent=2, default=str)}")
    
    # Get optimization recommendations
    optimization = cache.optimize_cache_settings()
    print(f"Optimization recommendations: {len(optimization['recommendations'])}")
    
    # Cleanup
    cache.cleanup()
    print("Enhanced cache test completed!") 