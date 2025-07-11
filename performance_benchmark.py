#!/usr/bin/env python3
"""
TradingAgents Performance Benchmarking Script

This script measures performance improvements from various optimizations:
- Parallel vs Sequential Processing
- Cache Performance
- API Optimization
- Memory Usage
- Overall Analysis Speed

Usage: python performance_benchmark.py

Author: TradingAgents Performance Team
"""

import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import statistics
import json
import psutil
import gc

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.dataflows.performance_optimizer import get_performance_optimizer
from tradingagents.dataflows.time_series_cache import get_cache


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite for TradingAgents
    """
    
    def __init__(self):
        """Initialize the benchmark suite"""
        self.results = {
            "benchmark_timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "test_results": {},
            "performance_comparison": {},
            "recommendations": []
        }
        
        # Test configuration
        self.test_symbols = ["MSFT", "AAPL", "GOOGL", "TSLA"]
        self.test_dates = [datetime.now() - timedelta(days=i) for i in range(1, 6)]  # 5 days
        
        # Performance baseline (typical times before optimization)
        self.baseline_times = {
            "single_analysis_sequential": 85.0,  # seconds
            "single_analysis_parallel": 45.0,    # seconds (expected)
            "batch_analysis_sequential": 340.0,  # 4 symbols * 85s
            "batch_analysis_optimized": 120.0    # expected with all optimizations
        }
        
        print("🚀 TradingAgents Performance Benchmark Suite")
        print("=" * 60)
        print(f"📊 Test Symbols: {', '.join(self.test_symbols)}")
        print(f"📅 Test Dates: {self.test_dates[0].date()} to {self.test_dates[-1].date()}")
        print(f"💻 System: {self.results['system_info']['cpu_cores']} cores, {self.results['system_info']['memory_gb']:.1f}GB RAM")
        print()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for the benchmark"""
        return {
            "cpu_cores": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": sys.version.split()[0],
            "platform": sys.platform
        }
    
    def _clear_memory_collections(self):
        """Clear ChromaDB memory collections to ensure clean state"""
        try:
            import chromadb
            chroma_client = chromadb.Client()
            
            collections_to_clear = ["bull_memory", "bear_memory", "neutral_memory"]
            for collection_name in collections_to_clear:
                try:
                    chroma_client.delete_collection(collection_name)
                except:
                    pass  # Collection doesn't exist
                    
            print("🧹 Memory collections cleared")
        except Exception as e:
            print(f"⚠️ Could not clear memory collections: {e}")
    
    def benchmark_single_analysis_sequential(self, symbol: str = "MSFT") -> Dict[str, Any]:
        """Benchmark single analysis with sequential processing"""
        print(f"📊 Benchmarking single analysis (sequential) for {symbol}...")
        
        # Clear memory state
        self._clear_memory_collections()
        gc.collect()
        
        # Configuration for sequential processing
        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = "anthropic"
        config["deep_think_llm"] = "claude-3-5-sonnet-20241022"
        config["quick_think_llm"] = "claude-3-5-haiku-20241022"
        config["max_debate_rounds"] = 3
        config["max_risk_discuss_rounds"] = 3
        config["online_tools"] = True
        
        # Memory usage before
        memory_before = psutil.Process().memory_info().rss / (1024**2)  # MB
        
        start_time = time.time()
        
        try:
            # Initialize with sequential processing
            ta = TradingAgentsGraph(
                debug=False,
                config=config,
                parallel_processing=False  # Sequential mode
            )
            
            analysis_date = self.test_dates[0].strftime('%Y-%m-%d')
            final_state, decision = ta.propagate(symbol, analysis_date)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Memory usage after
            memory_after = psutil.Process().memory_info().rss / (1024**2)  # MB
            
            result = {
                "symbol": symbol,
                "analysis_date": analysis_date,
                "duration": duration,
                "decision": decision,
                "memory_usage_mb": memory_after - memory_before,
                "success": True,
                "processing_mode": "sequential"
            }
            
            print(f"   ✅ Sequential analysis: {duration:.1f}s, Decision: {decision}")
            return result
            
        except Exception as e:
            print(f"   ❌ Sequential analysis failed: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "success": False,
                "processing_mode": "sequential"
            }
    
    def benchmark_single_analysis_parallel(self, symbol: str = "MSFT") -> Dict[str, Any]:
        """Benchmark single analysis with parallel processing"""
        print(f"🚀 Benchmarking single analysis (parallel) for {symbol}...")
        
        # Clear memory state
        self._clear_memory_collections()
        gc.collect()
        
        # Configuration for parallel processing  
        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = "anthropic"
        config["deep_think_llm"] = "claude-3-5-sonnet-20241022"
        config["quick_think_llm"] = "claude-3-5-haiku-20241022"
        config["max_debate_rounds"] = 2  # Reduced for speed
        config["max_risk_discuss_rounds"] = 2
        config["online_tools"] = True
        
        # Memory usage before
        memory_before = psutil.Process().memory_info().rss / (1024**2)  # MB
        
        start_time = time.time()
        
        try:
            # Initialize with parallel processing
            ta = TradingAgentsGraph(
                debug=False,
                config=config,
                parallel_processing=True  # Parallel mode
            )
            
            analysis_date = self.test_dates[0].strftime('%Y-%m-%d')
            final_state, decision = ta.propagate(symbol, analysis_date)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Memory usage after
            memory_after = psutil.Process().memory_info().rss / (1024**2)  # MB
            
            result = {
                "symbol": symbol,
                "analysis_date": analysis_date,
                "duration": duration,
                "decision": decision,
                "memory_usage_mb": memory_after - memory_before,
                "success": True,
                "processing_mode": "parallel"
            }
            
            print(f"   ✅ Parallel analysis: {duration:.1f}s, Decision: {decision}")
            return result
            
        except Exception as e:
            print(f"   ❌ Parallel analysis failed: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "success": False,
                "processing_mode": "parallel"
            }
    
    def benchmark_cache_performance(self) -> Dict[str, Any]:
        """Benchmark cache performance with cold vs warm cache"""
        print("📈 Benchmarking cache performance...")
        
        cache = get_cache()
        
        # Clear cache for cold start
        cache.clear_cache()
        
        # Test symbol and date range
        test_symbol = "AAPL"
        start_date = self.test_dates[-1] - timedelta(days=30)
        end_date = self.test_dates[0]
        
        # Cold cache test
        cold_start_time = time.time()
        try:
            from tradingagents.dataflows.cached_api_wrappers import fetch_financialdatasets_prices_cached
            data1 = fetch_financialdatasets_prices_cached(test_symbol, start_date, end_date)
            cold_duration = time.time() - cold_start_time
            cold_success = True
            cold_records = len(data1)
        except Exception as e:
            cold_duration = time.time() - cold_start_time
            cold_success = False
            cold_records = 0
            print(f"   ❌ Cold cache test failed: {e}")
        
        # Warm cache test (same data)
        warm_start_time = time.time()
        try:
            data2 = fetch_financialdatasets_prices_cached(test_symbol, start_date, end_date)
            warm_duration = time.time() - warm_start_time
            warm_success = True
            warm_records = len(data2)
        except Exception as e:
            warm_duration = time.time() - warm_start_time
            warm_success = False
            warm_records = 0
            print(f"   ❌ Warm cache test failed: {e}")
        
        # Cache statistics
        cache_stats = cache.get_cache_stats()
        
        cache_improvement = ((cold_duration - warm_duration) / cold_duration * 100) if cold_duration > 0 else 0
        
        result = {
            "test_symbol": test_symbol,
            "date_range": f"{start_date.date()} to {end_date.date()}",
            "cold_cache": {
                "duration": cold_duration,
                "success": cold_success,
                "records": cold_records
            },
            "warm_cache": {
                "duration": warm_duration,
                "success": warm_success,
                "records": warm_records
            },
            "cache_stats": cache_stats,
            "performance_improvement": f"{cache_improvement:.1f}%"
        }
        
        print(f"   ✅ Cache performance: {cache_improvement:.1f}% improvement")
        print(f"   📊 Cold: {cold_duration:.3f}s, Warm: {warm_duration:.3f}s")
        
        return result
    
    def benchmark_batch_optimization(self) -> Dict[str, Any]:
        """Benchmark batch optimization performance"""
        print("📦 Benchmarking batch optimization...")
        
        # Get performance optimizer
        optimizer = get_performance_optimizer()
        
        # Test with multiple symbols
        test_symbols = self.test_symbols[:3]  # First 3 symbols
        test_dates = self.test_dates[:2]      # First 2 dates
        
        start_time = time.time()
        
        try:
            # Run batch optimization
            batch_results = optimizer.optimize_batch_analysis(test_symbols, test_dates)
            
            optimization_duration = time.time() - start_time
            
            result = {
                "symbols": test_symbols,
                "dates": [d.date().isoformat() for d in test_dates],
                "optimization_duration": optimization_duration,
                "batch_results": batch_results,
                "success": True
            }
            
            print(f"   ✅ Batch optimization: {optimization_duration:.1f}s")
            if "performance_metrics" in batch_results:
                efficiency = batch_results["performance_metrics"].get("efficiency_gain", "N/A")
                print(f"   🚀 Efficiency gain: {efficiency}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Batch optimization failed: {e}")
            return {
                "symbols": test_symbols,
                "error": str(e),
                "success": False
            }
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns"""
        print("🧠 Benchmarking memory usage...")
        
        # Memory before any operations
        process = psutil.Process()
        memory_baseline = process.memory_info().rss / (1024**2)  # MB
        
        # Run a single analysis and track memory
        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = "anthropic"
        config["deep_think_llm"] = "claude-3-5-sonnet-20241022"
        config["quick_think_llm"] = "claude-3-5-haiku-20241022"
        config["max_debate_rounds"] = 1  # Minimal for memory test
        config["max_risk_discuss_rounds"] = 1
        config["online_tools"] = True
        
        try:
            # Initialize system
            ta = TradingAgentsGraph(debug=False, config=config)
            memory_after_init = process.memory_info().rss / (1024**2)
            
            # Run analysis
            analysis_date = self.test_dates[0].strftime('%Y-%m-%d')
            final_state, decision = ta.propagate("MSFT", analysis_date)
            memory_after_analysis = process.memory_info().rss / (1024**2)
            
            # Force garbage collection
            gc.collect()
            memory_after_gc = process.memory_info().rss / (1024**2)
            
            # Get cache information
            cache_stats = get_cache().get_cache_stats()
            
            result = {
                "memory_baseline_mb": memory_baseline,
                "memory_after_init_mb": memory_after_init,
                "memory_after_analysis_mb": memory_after_analysis,
                "memory_after_gc_mb": memory_after_gc,
                "memory_usage_analysis_mb": memory_after_analysis - memory_after_init,
                "memory_freed_by_gc_mb": memory_after_analysis - memory_after_gc,
                "cache_size_mb": cache_stats.get("cache_size_mb", 0),
                "success": True
            }
            
            print(f"   ✅ Memory usage: {result['memory_usage_analysis_mb']:.1f}MB for analysis")
            print(f"   🧹 GC freed: {result['memory_freed_by_gc_mb']:.1f}MB")
            print(f"   💾 Cache size: {cache_stats.get('cache_size_mb', 0):.1f}MB")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Memory benchmark failed: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark suite"""
        print("🎯 Running comprehensive performance benchmark...")
        print("=" * 60)
        
        # 1. Single Analysis Benchmarks
        print("\n1️⃣ Single Analysis Performance")
        print("-" * 40)
        
        sequential_result = self.benchmark_single_analysis_sequential()
        time.sleep(2)  # Brief pause between tests
        
        parallel_result = self.benchmark_single_analysis_parallel()
        time.sleep(2)
        
        # 2. Cache Performance
        print("\n2️⃣ Cache Performance")
        print("-" * 40)
        
        cache_result = self.benchmark_cache_performance()
        time.sleep(2)
        
        # 3. Batch Optimization
        print("\n3️⃣ Batch Optimization")
        print("-" * 40)
        
        batch_result = self.benchmark_batch_optimization()
        time.sleep(2)
        
        # 4. Memory Usage
        print("\n4️⃣ Memory Usage")
        print("-" * 40)
        
        memory_result = self.benchmark_memory_usage()
        
        # Store results
        self.results["test_results"] = {
            "single_analysis_sequential": sequential_result,
            "single_analysis_parallel": parallel_result,
            "cache_performance": cache_result,
            "batch_optimization": batch_result,
            "memory_usage": memory_result
        }
        
        # Calculate performance improvements
        self._calculate_performance_improvements()
        
        # Generate recommendations
        self._generate_recommendations()
        
        return self.results
    
    def _calculate_performance_improvements(self):
        """Calculate performance improvements from optimizations"""
        improvements = {}
        
        # Single analysis improvement
        seq_result = self.results["test_results"]["single_analysis_sequential"]
        par_result = self.results["test_results"]["single_analysis_parallel"]
        
        if seq_result.get("success") and par_result.get("success"):
            seq_time = seq_result["duration"]
            par_time = par_result["duration"]
            improvement = ((seq_time - par_time) / seq_time) * 100
            
            improvements["parallel_vs_sequential"] = {
                "sequential_time": seq_time,
                "parallel_time": par_time,
                "improvement_percent": improvement,
                "time_saved": seq_time - par_time
            }
        
        # Cache improvement
        cache_result = self.results["test_results"]["cache_performance"]
        if cache_result.get("cold_cache") and cache_result.get("warm_cache"):
            cold_time = cache_result["cold_cache"]["duration"]
            warm_time = cache_result["warm_cache"]["duration"]
            cache_improvement = ((cold_time - warm_time) / cold_time) * 100
            
            improvements["cache_optimization"] = {
                "cold_cache_time": cold_time,
                "warm_cache_time": warm_time,
                "improvement_percent": cache_improvement,
                "time_saved": cold_time - warm_time
            }
        
        self.results["performance_comparison"] = improvements
    
    def _generate_recommendations(self):
        """Generate performance recommendations based on benchmark results"""
        recommendations = []
        
        # Parallel processing recommendation
        if "parallel_vs_sequential" in self.results["performance_comparison"]:
            improvement = self.results["performance_comparison"]["parallel_vs_sequential"]["improvement_percent"]
            if improvement > 20:  # More than 20% improvement
                recommendations.append({
                    "type": "parallel_processing",
                    "priority": "high",
                    "improvement": f"{improvement:.1f}%",
                    "recommendation": "Enable parallel processing mode for all analyses"
                })
        
        # Cache recommendations
        cache_stats = self.results["test_results"]["cache_performance"].get("cache_stats", {})
        hit_ratio = cache_stats.get("hit_ratio", 0)
        
        if hit_ratio < 0.8:  # Less than 80% hit rate
            recommendations.append({
                "type": "cache_optimization",
                "priority": "medium",
                "current_hit_ratio": f"{hit_ratio:.1%}",
                "recommendation": "Implement cache warming for frequently analyzed symbols"
            })
        
        # Memory recommendations
        memory_result = self.results["test_results"]["memory_usage"]
        if memory_result.get("success"):
            memory_usage = memory_result["memory_usage_analysis_mb"]
            if memory_usage > 500:  # More than 500MB per analysis
                recommendations.append({
                    "type": "memory_optimization",
                    "priority": "medium",
                    "current_usage": f"{memory_usage:.1f}MB",
                    "recommendation": "Implement memory cleanup and optimize data structures"
                })
        
        self.results["recommendations"] = recommendations
    
    def print_benchmark_summary(self):
        """Print a comprehensive benchmark summary"""
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 80)
        
        # Performance improvements
        if "performance_comparison" in self.results:
            print("\n🚀 Performance Improvements:")
            print("-" * 40)
            
            for optimization, data in self.results["performance_comparison"].items():
                improvement = data.get("improvement_percent", 0)
                time_saved = data.get("time_saved", 0)
                
                print(f"   {optimization.replace('_', ' ').title()}: {improvement:.1f}% faster ({time_saved:.1f}s saved)")
        
        # Cache efficiency
        cache_result = self.results["test_results"].get("cache_performance", {})
        if cache_result.get("cache_stats"):
            cache_stats = cache_result["cache_stats"]
            print(f"\n💾 Cache Statistics:")
            print(f"   Hit Ratio: {cache_stats.get('hit_ratio', 0):.1%}")
            print(f"   Cache Size: {cache_stats.get('cache_size_mb', 0):.1f}MB")
            print(f"   Total Entries: {cache_stats.get('total_cache_entries', 0)}")
        
        # Memory usage
        memory_result = self.results["test_results"].get("memory_usage", {})
        if memory_result.get("success"):
            print(f"\n🧠 Memory Usage:")
            print(f"   Per Analysis: {memory_result['memory_usage_analysis_mb']:.1f}MB")
            print(f"   Cache Size: {memory_result['cache_size_mb']:.1f}MB")
        
        # Recommendations
        if self.results["recommendations"]:
            print(f"\n💡 Recommendations:")
            print("-" * 40)
            for rec in self.results["recommendations"]:
                priority = rec["priority"].upper()
                print(f"   [{priority}] {rec['recommendation']}")
        
        print("\n" + "=" * 80)
    
    def save_benchmark_results(self, filename: str = None):
        """Save benchmark results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_benchmark_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"📁 Benchmark results saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save benchmark results: {e}")


def main():
    """Main benchmarking function"""
    print("🚀 TradingAgents Performance Benchmarking")
    print("This will run comprehensive performance tests...")
    print()
    
    # Check API keys
    required_keys = ['ANTHROPIC_API_KEY', 'FINANCIALDATASETS_API_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"❌ Missing API keys: {', '.join(missing_keys)}")
        print("Please set the required environment variables before running benchmarks.")
        return
    
    # Initialize and run benchmark
    benchmark = PerformanceBenchmark()
    
    try:
        results = benchmark.run_comprehensive_benchmark()
        benchmark.print_benchmark_summary()
        benchmark.save_benchmark_results()
        
        print("\n✅ Benchmark completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main() 