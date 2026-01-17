from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import time
import psutil
import os
from contextlib import asynccontextmanager

from .api import app as api_router
from .services.decryptor import DecryptorService
from .services.cache import LRUCache

# Configuration
MAX_CONCURRENT_DOWNLOADS = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "10"))
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))

# Global state
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    print("Starting MP4 Decryptor API...")

    # Initialize services
    state["decryptor"] = DecryptorService(max_concurrent_downloads=MAX_CONCURRENT_DOWNLOADS)
    state["cache"] = LRUCache(max_size=CACHE_MAX_SIZE, ttl=CACHE_TTL)
    state["start_time"] = time.time()
    state["active_tasks"] = 0
    state["cache_hits"] = 0
    state["cache_misses"] = 0

    print(f"Initialized decryptor with {MAX_CONCURRENT_DOWNLOADS} concurrent downloads")
    print(f"Cache enabled: {CACHE_MAX_SIZE} items, {CACHE_TTL}s TTL")

    yield

    # Shutdown
    print("Shutting down MP4 Decryptor API...")
    if "decryptor" in state:
        await state["decryptor"].close()
    print("Cleanup complete")


# Create FastAPI app
app = FastAPI(
    title="MP4 Segment Decryptor API",
    description="High-performance API for decrypting encrypted MP4 media segments with CENC support",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
app.include_router(api_router)

# Store app state
app.state = state


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "MP4 Segment Decryptor API",
        "version": "2.0.0",
        "description": "High-performance API for decrypting encrypted MP4 segments",
        "endpoints": {
            "health": "/health",
            "decrypt": "/decrypt",
            "decrypt_direct": "/decrypt/direct",
            "batch_decrypt": "/decrypt/batch",
            "async_decrypt": "/decrypt/async",
            "info": "/info",
            "stats": "/stats"
        },
        "features": [
            "AES-128-CTR (CENC) decryption",
            "Subsample encryption support",
            "MP4 box parsing",
            "In-place decryption",
            "Async processing",
            "Caching",
            "Batch operations"
        ]
    }


@app.get("/info")
async def get_info():
    """Get decryptor information and capabilities"""
    return {
        "algorithm_support": ["aes-128-ctr", "aes-128-cbc", "aes-256-cbc"],
        "max_concurrent_downloads": MAX_CONCURRENT_DOWNLOADS,
        "cache_enabled": CACHE_MAX_SIZE > 0,
        "cache_size": CACHE_MAX_SIZE,
        "cache_ttl": CACHE_TTL,
        "performance": {
            "concurrent_downloads": MAX_CONCURRENT_DOWNLOADS,
            "in_memory_processing": True,
            "async_io": True
        }
    }


@app.get("/stats")
async def get_stats():
    """Get application statistics"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    return {
        "uptime": time.time() - state.get("start_time", time.time()),
        "memory_usage_mb": memory_info.rss / 1024 / 1024,
        "active_tasks": state.get("active_tasks", 0),
        "cache_stats": {
            "size": state.get("cache", LRUCache(1)).size() if "cache" in state else 0,
            "hits": state.get("cache_hits", 0),
            "misses": state.get("cache_misses", 0),
            "hit_ratio": (
                    state.get("cache_hits", 0) /
                    max(state.get("cache_hits", 0) + state.get("cache_misses", 0), 1)
            )
        },
        "decryptor": {
            "max_concurrent": MAX_CONCURRENT_DOWNLOADS,
            "active_downloads": state.get("decryptor", DecryptorService(1)).semaphore._value
            if "decryptor" in state else 0
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=int(os.getenv("WORKERS", "4")),
        log_level="info"
    )