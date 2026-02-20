"""
Caching system for expensive operations (translations, embeddings, phonetic codes)
"""

import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Optional, Callable
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """Simple file-based cache with TTL support"""
    
    def __init__(self, cache_dir: Path, ttl_seconds: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl_seconds
        
        # In-memory cache for fast access
        self.memory_cache = {}
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key"""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve from cache if not expired"""
        # Check memory cache first
        if key in self.memory_cache:
            data, timestamp = self.memory_cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.memory_cache[key]
        
        # Check file cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data, timestamp = pickle.load(f)
                
                if time.time() - timestamp < self.ttl:
                    # Restore to memory cache
                    self.memory_cache[key] = (data, timestamp)
                    return data
                else:
                    # Expired, remove file
                    cache_path.unlink()
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
                cache_path.unlink(missing_ok=True)
        
        return None
    
    def set(self, key: str, value: Any):
        """Store in cache"""
        timestamp = time.time()
        
        # Store in memory
        self.memory_cache[key] = (value, timestamp)
        
        # Store in file
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump((value, timestamp), f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def clear(self):
        """Clear all cache"""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Cache cleared")
    
    def clear_expired(self):
        """Remove expired cache files"""
        current_time = time.time()
        removed = 0
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    _, timestamp = pickle.load(f)
                
                if current_time - timestamp >= self.ttl:
                    cache_file.unlink()
                    removed += 1
            except Exception:
                cache_file.unlink()
                removed += 1
        
        if removed > 0:
            logger.info(f"Removed {removed} expired cache files")


def cached(cache_manager: CacheManager, prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{prefix}_{func.__name__}_" + cache_manager._get_cache_key(*args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(key)
            if cached_result is not None:
                return cached_result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache_manager.set(key, result)
            return result
        
        return wrapper
    return decorator
