"""
Optimized Memory Manager for TradingAgents

This module provides enhanced ChromaDB memory management with:
- Connection pooling
- Persistent collections
- Optimized query patterns
- Memory usage optimization
- Connection lifecycle management

Author: TradingAgents Performance Team
"""

import logging
from typing import Dict, List, Any, Optional, Union
import chromadb
from chromadb.config import Settings
import threading
import weakref
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class CollectionType(Enum):
    """Types of memory collections"""
    BULL_MEMORY = "bull_memory"
    BEAR_MEMORY = "bear_memory"
    TRADER_MEMORY = "trader_memory"
    RISK_MANAGER_MEMORY = "risk_manager_memory"
    INVEST_JUDGE_MEMORY = "invest_judge_memory"


@dataclass
class ConnectionInfo:
    """Information about a ChromaDB connection"""
    client: chromadb.Client
    created_at: datetime
    last_used: datetime
    use_count: int
    thread_id: int


class ChromaDBConnectionPool:
    """Connection pool for ChromaDB clients"""
    
    def __init__(self, max_connections: int = 5, connection_timeout: int = 300):
        """
        Initialize the connection pool
        
        Args:
            max_connections: Maximum number of connections to maintain
            connection_timeout: Connection timeout in seconds
        """
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.connections: Dict[str, ConnectionInfo] = {}
        self.lock = threading.RLock()
        self._cleanup_thread = None
        self._running = False
        
        # ChromaDB settings for optimal performance
        self.chroma_settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db",
            anonymized_telemetry=False,
            allow_reset=True
        )
        
        self._start_cleanup_thread()
        logger.info(f"🔗 ChromaDB Connection Pool initialized (max: {max_connections})")
    
    def _start_cleanup_thread(self):
        """Start background thread for connection cleanup"""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._running = True
            self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self._cleanup_thread.start()
    
    def _cleanup_loop(self):
        """Background cleanup of stale connections"""
        while self._running:
            try:
                time.sleep(60)  # Check every minute
                self._cleanup_stale_connections()
            except Exception as e:
                logger.error(f"❌ Connection cleanup error: {e}")
    
    def _cleanup_stale_connections(self):
        """Remove stale connections"""
        with self.lock:
            current_time = datetime.now()
            stale_keys = []
            
            for key, conn_info in self.connections.items():
                age = (current_time - conn_info.last_used).total_seconds()
                if age > self.connection_timeout:
                    stale_keys.append(key)
            
            for key in stale_keys:
                logger.info(f"🧹 Removing stale connection: {key}")
                del self.connections[key]
    
    def get_connection(self, collection_type: CollectionType) -> chromadb.Client:
        """Get a connection from the pool"""
        with self.lock:
            thread_id = threading.get_ident()
            key = f"{collection_type.value}_{thread_id}"
            
            # Check if we have an existing connection for this thread/collection
            if key in self.connections:
                conn_info = self.connections[key]
                conn_info.last_used = datetime.now()
                conn_info.use_count += 1
                logger.debug(f"♻️ Reusing connection: {key}")
                return conn_info.client
            
            # Create new connection if under limit
            if len(self.connections) < self.max_connections:
                client = chromadb.Client(self.chroma_settings)
                
                conn_info = ConnectionInfo(
                    client=client,
                    created_at=datetime.now(),
                    last_used=datetime.now(),
                    use_count=1,
                    thread_id=thread_id
                )
                
                self.connections[key] = conn_info
                logger.info(f"🔗 Created new connection: {key}")
                return client
            
            # If at limit, find least recently used connection
            lru_key = min(self.connections.keys(), 
                         key=lambda k: self.connections[k].last_used)
            
            logger.info(f"♻️ Recycling LRU connection: {lru_key} -> {key}")
            
            client = self.connections[lru_key].client
            del self.connections[lru_key]
            
            # Update connection info
            conn_info = ConnectionInfo(
                client=client,
                created_at=datetime.now(),
                last_used=datetime.now(),
                use_count=1,
                thread_id=thread_id
            )
            
            self.connections[key] = conn_info
            return client
    
    def release_connection(self, collection_type: CollectionType):
        """Release a connection back to the pool"""
        with self.lock:
            thread_id = threading.get_ident()
            key = f"{collection_type.value}_{thread_id}"
            
            if key in self.connections:
                self.connections[key].last_used = datetime.now()
                logger.debug(f"📤 Released connection: {key}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        with self.lock:
            return {
                "active_connections": len(self.connections),
                "max_connections": self.max_connections,
                "connection_details": {
                    key: {
                        "created_at": info.created_at.isoformat(),
                        "last_used": info.last_used.isoformat(),
                        "use_count": info.use_count,
                        "age_seconds": (datetime.now() - info.created_at).total_seconds()
                    }
                    for key, info in self.connections.items()
                }
            }
    
    def shutdown(self):
        """Shutdown the connection pool"""
        self._running = False
        
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        
        with self.lock:
            self.connections.clear()
        
        logger.info("🔒 ChromaDB Connection Pool shutdown")


class OptimizedFinancialSituationMemory:
    """
    Optimized financial situation memory with connection pooling and persistence
    """
    
    def __init__(self, collection_type: str, config: Dict[str, Any], connection_pool: ChromaDBConnectionPool = None):
        """
        Initialize optimized memory
        
        Args:
            collection_type: Type of collection (e.g., 'bull_memory')
            config: Configuration dictionary
            connection_pool: Optional connection pool (creates new if None)
        """
        self.collection_type = CollectionType(collection_type)
        self.config = config
        self.connection_pool = connection_pool or _get_global_connection_pool()
        
        # Performance settings
        self.max_results = config.get("memory_max_results", 10)
        self.embedding_batch_size = config.get("embedding_batch_size", 50)
        
        # Initialize collection
        self._collection = None
        self._ensure_collection_exists()
        
        logger.info(f"💾 Optimized memory initialized: {collection_type}")
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists with optimal settings"""
        try:
            client = self.connection_pool.get_connection(self.collection_type)
            
            # Try to get existing collection
            try:
                self._collection = client.get_collection(self.collection_type.value)
                logger.debug(f"📂 Using existing collection: {self.collection_type.value}")
            except Exception:
                # Create new collection with metadata
                self._collection = client.create_collection(
                    name=self.collection_type.value,
                    metadata={
                        "description": f"Financial situation memory for {self.collection_type.value}",
                        "created_at": datetime.now().isoformat(),
                        "version": "2.0_optimized"
                    }
                )
                logger.info(f"📁 Created new collection: {self.collection_type.value}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize collection {self.collection_type.value}: {e}")
            raise
        finally:
            self.connection_pool.release_connection(self.collection_type)
    
    def add_memory(self, memory_content: str, metadata: Dict[str, Any] = None) -> str:
        """Add memory with optimized batching"""
        try:
            client = self.connection_pool.get_connection(self.collection_type)
            collection = client.get_collection(self.collection_type.value)
            
            # Generate ID and prepare metadata
            memory_id = f"{self.collection_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            enhanced_metadata = {
                "timestamp": datetime.now().isoformat(),
                "collection_type": self.collection_type.value,
                **(metadata or {})
            }
            
            # Add to collection
            collection.add(
                documents=[memory_content],
                metadatas=[enhanced_metadata],
                ids=[memory_id]
            )
            
            logger.debug(f"💾 Added memory: {memory_id[:20]}...")
            return memory_id
            
        except Exception as e:
            logger.error(f"❌ Failed to add memory: {e}")
            raise
        finally:
            self.connection_pool.release_connection(self.collection_type)
    
    def query_memories(self, query: str, n_results: int = None) -> List[Dict[str, Any]]:
        """Query memories with optimized retrieval"""
        try:
            client = self.connection_pool.get_connection(self.collection_type)
            collection = client.get_collection(self.collection_type.value)
            
            n_results = n_results or self.max_results
            
            # Perform similarity search
            results = collection.query(
                query_texts=[query],
                n_results=min(n_results, collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "similarity": 1 - results['distances'][0][i] if results['distances'] else 0.0
                    })
            
            logger.debug(f"🔍 Retrieved {len(formatted_results)} memories for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Failed to query memories: {e}")
            return []
        finally:
            self.connection_pool.release_connection(self.collection_type)
    
    def get_recent_memories(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent memories efficiently"""
        try:
            client = self.connection_pool.get_connection(self.collection_type)
            collection = client.get_collection(self.collection_type.value)
            
            # Get all memories and sort by timestamp
            all_results = collection.get(include=["documents", "metadatas"])
            
            if not all_results['documents']:
                return []
            
            # Sort by timestamp (newest first)
            memories_with_time = []
            for i, doc in enumerate(all_results['documents']):
                metadata = all_results['metadatas'][i] if all_results['metadatas'] else {}
                timestamp_str = metadata.get('timestamp', '1970-01-01T00:00:00')
                
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except:
                    timestamp = datetime.min
                
                memories_with_time.append({
                    "content": doc,
                    "metadata": metadata,
                    "timestamp": timestamp
                })
            
            # Sort and limit
            memories_with_time.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return memories_with_time[:limit]
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent memories: {e}")
            return []
        finally:
            self.connection_pool.release_connection(self.collection_type)
    
    def clear_old_memories(self, days_to_keep: int = 30) -> int:
        """Clear old memories to optimize performance"""
        try:
            client = self.connection_pool.get_connection(self.collection_type)
            collection = client.get_collection(self.collection_type.value)
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Get all memories
            all_results = collection.get(include=["metadatas"])
            
            if not all_results['metadatas']:
                return 0
            
            # Find old memories
            old_ids = []
            for i, metadata in enumerate(all_results['metadatas']):
                timestamp_str = metadata.get('timestamp', '1970-01-01T00:00:00')
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if timestamp < cutoff_date:
                        old_ids.append(all_results['ids'][i])
                except:
                    continue
            
            # Delete old memories
            if old_ids:
                collection.delete(ids=old_ids)
                logger.info(f"🧹 Cleared {len(old_ids)} old memories from {self.collection_type.value}")
            
            return len(old_ids)
            
        except Exception as e:
            logger.error(f"❌ Failed to clear old memories: {e}")
            return 0
        finally:
            self.connection_pool.release_connection(self.collection_type)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory collection statistics"""
        try:
            client = self.connection_pool.get_connection(self.collection_type)
            collection = client.get_collection(self.collection_type.value)
            
            total_count = collection.count()
            
            # Get recent memory count (last 7 days)
            recent_count = 0
            if total_count > 0:
                recent_cutoff = datetime.now() - timedelta(days=7)
                all_results = collection.get(include=["metadatas"])
                
                for metadata in all_results['metadatas'] or []:
                    timestamp_str = metadata.get('timestamp', '1970-01-01T00:00:00')
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if timestamp >= recent_cutoff:
                            recent_count += 1
                    except:
                        continue
            
            return {
                "collection_name": self.collection_type.value,
                "total_memories": total_count,
                "recent_memories_7d": recent_count,
                "max_results": self.max_results,
                "last_checked": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get memory stats: {e}")
            return {"error": str(e)}
        finally:
            self.connection_pool.release_connection(self.collection_type)


# Global connection pool instance
_global_connection_pool = None

def _get_global_connection_pool() -> ChromaDBConnectionPool:
    """Get or create the global connection pool"""
    global _global_connection_pool
    if _global_connection_pool is None:
        _global_connection_pool = ChromaDBConnectionPool()
    return _global_connection_pool


def create_optimized_memory(collection_type: str, config: Dict[str, Any]) -> OptimizedFinancialSituationMemory:
    """
    Factory function to create optimized memory instances
    
    Args:
        collection_type: Type of memory collection
        config: Configuration dictionary
        
    Returns:
        Optimized memory instance
    """
    return OptimizedFinancialSituationMemory(collection_type, config)


def cleanup_all_memory_collections():
    """Clean up all memory collections"""
    pool = _get_global_connection_pool()
    
    for collection_type in CollectionType:
        try:
            client = pool.get_connection(collection_type)
            try:
                collection = client.get_collection(collection_type.value)
                client.delete_collection(collection_type.value)
                logger.info(f"🗑️ Deleted collection: {collection_type.value}")
            except Exception:
                pass  # Collection doesn't exist
        except Exception as e:
            logger.error(f"❌ Failed to cleanup collection {collection_type.value}: {e}")
        finally:
            pool.release_connection(collection_type)


def get_global_memory_stats() -> Dict[str, Any]:
    """Get statistics for all memory collections"""
    pool = _get_global_connection_pool()
    stats = {
        "connection_pool": pool.get_pool_stats(),
        "collections": {}
    }
    
    for collection_type in CollectionType:
        try:
            memory = create_optimized_memory(collection_type.value, {})
            stats["collections"][collection_type.value] = memory.get_memory_stats()
        except Exception as e:
            stats["collections"][collection_type.value] = {"error": str(e)}
    
    return stats


if __name__ == "__main__":
    # Test the optimized memory system
    import time
    
    # Test connection pool
    pool = ChromaDBConnectionPool(max_connections=3)
    
    # Test memory operations
    config = {"memory_max_results": 5}
    memory = OptimizedFinancialSituationMemory("bull_memory", config, pool)
    
    # Add test memories
    for i in range(5):
        memory.add_memory(f"Test memory {i}", {"test_id": i})
        time.sleep(0.1)
    
    # Query memories
    results = memory.query_memories("test", n_results=3)
    print(f"Query results: {len(results)}")
    
    # Get stats
    stats = memory.get_memory_stats()
    print(f"Memory stats: {stats}")
    
    # Pool stats
    pool_stats = pool.get_pool_stats()
    print(f"Pool stats: {pool_stats}")
    
    # Cleanup
    pool.shutdown() 