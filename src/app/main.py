import logging
import os
import time
import warnings
from contextlib import asynccontextmanager

import psutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from .api import app as api_app
from .api import init_services
from .services.cache import LRUCache
from .services.decryptor import DecryptorService

warnings.filterwarnings(
    "ignore",
    message="An HTTPS request is being sent through an HTTPS proxy",
    category=RuntimeWarning,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
MAX_CONCURRENT_DOWNLOADS = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "10"))
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    logger.info("Starting Media Proxy API...")

    # Initialize services
    decryptor = DecryptorService(max_concurrent_downloads=MAX_CONCURRENT_DOWNLOADS)
    cache_service = LRUCache(max_size=CACHE_MAX_SIZE, ttl=CACHE_TTL)

    # Initialize API services
    init_services(decryptor, cache_service)

    # Setup app state
    app.state.start_time = time.time()
    app.state.active_tasks = 0
    app.state.cache_hits = 0
    app.state.cache_misses = 0
    app.state.decryptor = decryptor
    app.state.cache = cache_service

    logger.info(f"Initialized decryptor with {MAX_CONCURRENT_DOWNLOADS} concurrent downloads")
    logger.info(f"Cache enabled: {CACHE_MAX_SIZE} items, {CACHE_TTL}s TTL")

    yield

    # Shutdown
    logger.info("Shutting down Media Proxy API...")
    await decryptor.close()
    logger.info("Cleanup complete")


# Create FastAPI app
app = FastAPI(
    title="Media Proxy API",
    description="High-performance API for proxying MP4 media segments " "with CENC support",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
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

# Mount API router
app.mount("/api", api_app)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "MP4 Segment Decryptor API",
        "version": "2.0.0",
        "description": "High-performance API for proxying MP4 segments",
        "endpoints": {
            "health": "/api/health",
            "proxy": "/api/proxy/<base64_url>/<template_path>?proxy=...&ua=...",
            "decrypt": "/api/decrypt/<base64_url>/<template_path>?key=...&proxy=...&ua=...",
            "decrypt_json": "/api/decrypt/json",
            "decrypt_stream": "/api/decrypt/stream",
            "batch_decrypt": "/api/decrypt/batch",
            "async_decrypt": "/api/decrypt/async",
            "info": "/info",
            "stats": "/stats",
            "docs": "/docs",
        },
        "features": [
            "AES-128-CTR (CENC) decryption",
            "Subsample encryption support",
            "MP4 box parsing and manipulation",
            "In-place decryption for performance",
            "Template-based URL composition for MPD",
            "Proxy support (HTTP/HTTPS/SOCKS)",
            "Async processing",
            "LRU caching with TTL",
            "Batch operations",
            "Streaming responses",
        ],
    }


@app.get("/info")
async def get_info():
    """Get decryptor information and capabilities"""
    return {
        "algorithm_support": ["aes-128-ctr"],
        "max_concurrent_downloads": MAX_CONCURRENT_DOWNLOADS,
        "cache_enabled": CACHE_MAX_SIZE > 0,
        "cache_size": CACHE_MAX_SIZE,
        "cache_ttl": CACHE_TTL,
        "performance": {
            "concurrent_downloads": MAX_CONCURRENT_DOWNLOADS,
            "in_memory_processing": True,
            "in_place_decryption": True,
            "async_io": True,
        },
        "supported_features": [
            "CENC (Common Encryption)",
            "Subsample encryption",
            "8-byte and 16-byte IVs",
            "Multiple fragments",
            "Protection box removal",
            "Template-based URL composition",
            "HTTP/HTTPS/SOCKS proxy support",
        ],
    }


@app.get("/stats")
async def get_stats():
    """Get application statistics"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    cache_service = getattr(app.state, "cache", None)
    decryptor = getattr(app.state, "decryptor", None)

    cache_hits = getattr(app.state, "cache_hits", 0)
    cache_misses = getattr(app.state, "cache_misses", 0)
    total_requests = cache_hits + cache_misses

    return {
        "uptime": time.time() - getattr(app.state, "start_time", time.time()),
        "memory_usage_mb": memory_info.rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "active_tasks": getattr(app.state, "active_tasks", 0),
        "cache_stats": {
            "size": cache_service.size() if cache_service else 0,
            "max_size": CACHE_MAX_SIZE,
            "hits": cache_hits,
            "misses": cache_misses,
            "total_requests": total_requests,
            "hit_ratio": cache_hits / max(total_requests, 1),
        },
        "decryptor": {
            "max_concurrent": MAX_CONCURRENT_DOWNLOADS,
            "available_slots": decryptor.semaphore._value if decryptor else 0,
        },
    }


@app.get("/cache/clear")
async def clear_cache():
    """Clear the cache"""
    cache_service = getattr(app.state, "cache", None)
    if cache_service:
        cache_service.clear()
        return {"status": "success", "message": "Cache cleared"}
    return {"status": "error", "message": "Cache not available"}


@app.get("/cache/cleanup")
async def cleanup_cache():
    """Remove expired items from cache"""
    cache_service = getattr(app.state, "cache", None)
    if cache_service:
        removed = cache_service.cleanup_expired()
        return {
            "status": "success",
            "message": f"Removed {removed} expired items",
            "removed_count": removed,
        }
    return {"status": "error", "message": "Cache not available"}


if __name__ == "__main__":
    import uvicorn

    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7775"))
    workers = int(os.getenv("WORKERS", "1"))  # Note: workers > 1 requires shared state
    reload = os.getenv("RELOAD", "false").lower() == "true"

    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Workers: {workers}, Reload: {reload}")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info",
    )
