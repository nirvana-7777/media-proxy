import threading
import time
from typing import Any, Dict, Optional


class LRUCache:
    """Thread-safe LRU cache implementation"""

    def __init__(self, max_size: int = 1000, ttl: int = 300):
        """
        Args:
            max_size: Maximum number of items in cache
            ttl: Time to live in seconds (default: 5 minutes)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        self._cleanup_counter = 0

    def get(self, key: str) -> Optional[Any]:  # Changed to Any for flexibility
        """Get item from cache, return None if not found or expired"""
        with self.lock:
            if key not in self.cache:
                return None

            item = self.cache[key]

            # Check if expired
            if time.time() - item["timestamp"] > self.ttl:
                del self.cache[key]
                return None

            # Update access time
            item["timestamp"] = time.time()
            return item["value"]

    def set(self, key: str, value: Any):  # Changed to Any for flexibility
        """Set item in cache, evicting LRU if necessary"""
        with self.lock:
            # Cleanup every 100 operations
            self._cleanup_counter += 1
            if self._cleanup_counter >= 100:
                self._cleanup_expired()
                self._cleanup_counter = 0

            # Evict LRU if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()

            self.cache[key] = {"value": value, "timestamp": time.time()}

    def _cleanup_expired(self):
        """Remove expired items"""
        current_time = time.time()
        expired_keys = [
            k for k, v in self.cache.items() if current_time - v["timestamp"] > self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]

    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.cache:
            return

        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
        del self.cache[lru_key]

    def clear(self):
        """Clear all items from cache"""
        with self.lock:
            self.cache.clear()

    def size(self) -> int:
        """Get current cache size"""
        with self.lock:
            return len(self.cache)

    def cleanup_expired(self) -> int:
        """Manually trigger cleanup and return count of removed items"""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                k for k, v in self.cache.items() if current_time - v["timestamp"] > self.ttl
            ]
            for key in expired_keys:
                del self.cache[key]
            return len(expired_keys)
