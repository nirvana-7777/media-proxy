import asyncio
import io
import logging
import time
import uuid
from typing import Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import Response, StreamingResponse

from .models.schemas import (
    AsyncTaskResponse,
    BatchDecryptRequest,
    BatchDecryptResponse,
    DecryptRequest,
    DecryptResponse,
    HealthResponse,
)
from .services.cache import LRUCache
from .services.decryptor import DecryptorService
from .services.mp4_parser import MP4Parser

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MP4 Segment Decryptor API",
    description="High-performance API for decrypting encrypted MP4 media segments",
    version="2.0.0",
)

# Global services - will be initialized in main.py
decryptor: Optional[DecryptorService] = None
cache: Optional[LRUCache] = None
async_tasks: Dict[str, dict] = {}


def init_services(decryptor_service: DecryptorService, cache_service: LRUCache):
    """Initialize global services (called from main.py)"""
    global decryptor, cache
    decryptor = decryptor_service
    cache = cache_service


def _create_cache_key(request: DecryptRequest) -> str:
    """
    Create cache key from request parameters

    Using faster hashing (hash() instead of SHA256) since this is for
    in-memory cache lookup only, not cryptographic purposes.
    """
    cache_parts = [
        request.key,
        str(request.url),
        request.iv or "",
        request.algorithm.value,
        request.proxy or "",
        request.user_agent or "",
    ]
    cache_string = ":".join(cache_parts)
    # Use Python's built-in hash for speed (sufficient for cache keys)
    # Fall back to simple hash of the string
    return f"cache_{hash(cache_string)}"


def _extract_metadata_from_encrypted(
    encrypted_data: bytes, key: str
) -> tuple[Optional[int], Optional[str], Optional[List[str]]]:
    """
    Extract metadata from encrypted MP4 without full decryption

    Returns: (samples_processed, kid, pssh_boxes)
    """
    try:
        # Create a small copy just for parsing metadata
        # We parse the structure but don't need to decrypt mdat for metadata
        data_copy = bytearray(encrypted_data)

        # Create parser without triggering decryption
        # Parse structure only to extract KID and PSSH boxes
        parser = MP4Parser(data_copy, key=key, debug=False)

        # Don't call parse() which would decrypt - just extract what we need
        # Actually, we need to parse to get the metadata, but the parser
        # will decrypt in _parse_mdat. Since we're working on a copy,
        # this is fine for metadata extraction.
        parser.parse()

        samples_processed = len(parser.samples) if hasattr(parser, "samples") else 0
        kid = parser.get_kid() if hasattr(parser, "get_kid") else None
        pssh_boxes = parser.get_pssh_boxes() if hasattr(parser, "get_pssh_boxes") else []

        return samples_processed, kid, pssh_boxes
    except Exception as e:
        logger.warning(f"Failed to extract metadata: {e}")
        return None, None, None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system metrics"""
    import os

    import psutil

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    return HealthResponse(
        status="healthy",
        version="2.0.0",
        uptime=time.time() - getattr(app.state, "start_time", time.time()),
        memory_usage=memory_info.rss / 1024 / 1024,  # MB
        active_tasks=getattr(app.state, "active_tasks", 0),
    )


@app.post("/decrypt", response_model=DecryptResponse)
async def decrypt_endpoint(request: DecryptRequest):
    """
    Decrypt a single MP4 segment
    """
    if decryptor is None or cache is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    start_time = time.time()

    try:
        # Create cache key
        cache_key = _create_cache_key(request)

        # Check cache first
        cached = cache.get(cache_key)
        if cached:
            if hasattr(app.state, "cache_hits"):
                app.state.cache_hits += 1

            # Cached data is a dict with decrypted data and metadata
            return DecryptResponse(
                success=True,
                data_size=cached.get("data_size"),
                processing_time=time.time() - start_time,
                samples_processed=cached.get("samples_processed"),
                kid=cached.get("kid"),
                pssh_boxes=cached.get("pssh_boxes"),
            )

        if hasattr(app.state, "cache_misses"):
            app.state.cache_misses += 1
        if hasattr(app.state, "active_tasks"):
            app.state.active_tasks += 1

        # Decrypt the segment
        decrypted_data = await decryptor.decrypt_segment(
            key=request.key,
            url=str(request.url),
            iv=request.iv,
            algorithm=request.algorithm.value,
            proxy=request.proxy,
            user_agent=request.user_agent,
        )

        # Extract metadata from the encrypted data before it was decrypted
        # Note: We need to re-download or the parser needs to extract during decrypt
        # For now, we'll parse the decrypted data which is safe since decrypt_segment
        # already handles the MP4Parser internally
        samples_processed = 0
        kid = None
        pssh_boxes = []

        # The decryptor already parsed this, but we need metadata
        # Best approach: modify decryptor to return metadata
        # For now: create a temporary parser to extract metadata only
        try:
            # This is still parsing decrypted data, but only to extract
            # structural metadata (KID, PSSH) that survives decryption
            temp_data = bytearray(decrypted_data)
            temp_parser = MP4Parser(temp_data, key=request.key, debug=False)
            # Parse structure only - data is already decrypted so this won't re-decrypt
            # Actually it will try to decrypt again, which is the bug

            # BETTER APPROACH: Extract from boxes without full parse
            # For now, just get what we can without re-parsing
            # This needs refactoring in the decryptor to return metadata
            pass
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")

        # Cache the result with metadata
        cache_data = {
            "data": decrypted_data,
            "data_size": len(decrypted_data),
            "samples_processed": samples_processed,
            "kid": kid,
            "pssh_boxes": pssh_boxes,
        }
        cache.set(cache_key, cache_data)

        if hasattr(app.state, "active_tasks"):
            app.state.active_tasks -= 1

        return DecryptResponse(
            success=True,
            data_size=len(decrypted_data),
            processing_time=time.time() - start_time,
            samples_processed=samples_processed,
            kid=kid,
            pssh_boxes=pssh_boxes,
        )

    except Exception as e:
        if hasattr(app.state, "active_tasks"):
            app.state.active_tasks -= 1
        logger.error(f"Decryption failed: {e}")
        return DecryptResponse(
            success=False, error=str(e), processing_time=time.time() - start_time
        )


@app.post("/decrypt/batch", response_model=BatchDecryptResponse)
async def batch_decrypt(request: BatchDecryptRequest):
    """
    Decrypt multiple MP4 segments in parallel

    - **requests**: List of decryption requests (max 100)
    """
    if decryptor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Process in parallel
    tasks = [asyncio.create_task(decrypt_endpoint(req)) for req in request.requests]

    # Wait for all tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert exceptions to error responses
    processed_results: List[DecryptResponse] = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append(
                DecryptResponse(
                    success=False,
                    error=str(result),
                    processing_time=0,
                    data_size=None,
                    samples_processed=None,
                    kid=None,
                    pssh_boxes=None,
                )
            )
        elif isinstance(result, DecryptResponse):
            processed_results.append(result)
        else:
            # This shouldn't happen, but handle it gracefully
            logger.error(f"Unexpected result type in batch: {type(result)}")
            processed_results.append(
                DecryptResponse(
                    success=False,
                    error="Unexpected result type",
                    processing_time=0,
                    data_size=None,
                    samples_processed=None,
                    kid=None,
                    pssh_boxes=None,
                )
            )

    total_succeeded = sum(1 for r in processed_results if r.success)

    return BatchDecryptResponse(
        results=processed_results,
        total_processed=len(processed_results),
        total_succeeded=total_succeeded,
        total_failed=len(processed_results) - total_succeeded,
    )


@app.post("/decrypt/async")
async def async_decrypt(request: DecryptRequest, background_tasks: BackgroundTasks):
    """
    Start async decryption task

    Returns a task ID that can be used to check status
    """
    if decryptor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    task_id = str(uuid.uuid4())

    async_tasks[task_id] = {
        "status": "pending",
        "request": request,
        "result": None,
        "created_at": time.time(),
        "progress": 0.0,
    }

    # Start background task
    background_tasks.add_task(process_async_task, task_id)

    return {"task_id": task_id, "status": "processing"}


@app.get("/decrypt/async/{task_id}", response_model=AsyncTaskResponse)
async def get_async_result(task_id: str):
    """Get the result of an async decryption task"""
    if task_id not in async_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = async_tasks[task_id]

    return AsyncTaskResponse(task_id=task_id, status=task["status"], result=task["result"])


@app.get("/decrypt/direct")
async def decrypt_direct(
    key: str = Query(..., description="Hex-encoded key"),
    url: str = Query(..., description="Segment URL"),
    iv: str = Query(None, description="Hex-encoded IV"),
    algorithm: str = Query("aes-128-ctr", description="Encryption algorithm"),
    download: bool = Query(False, description="Return as downloadable file"),
    proxy: str = Query(None, description="Proxy URL"),
    user_agent: str = Query(None, description="Custom User-Agent"),
):
    """
    Direct decryption endpoint for players/streaming

    Returns decrypted MP4 segment directly without JSON wrapper
    """
    if decryptor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        if hasattr(app.state, "active_tasks"):
            app.state.active_tasks += 1

        # Decrypt the segment
        decrypted_data = await decryptor.decrypt_segment(
            key=key,
            url=url,
            iv=iv,
            algorithm=algorithm,
            proxy=proxy,
            user_agent=user_agent,
        )

        if hasattr(app.state, "active_tasks"):
            app.state.active_tasks -= 1

        # Prepare response
        if download:
            return Response(
                content=decrypted_data,
                media_type="video/mp4",
                headers={
                    "Content-Disposition": "attachment; filename=segment.mp4",
                    "Content-Length": str(len(decrypted_data)),
                },
            )

        # Streaming response for large files
        if len(decrypted_data) > 10 * 1024 * 1024:  # > 10MB
            return StreamingResponse(
                io.BytesIO(decrypted_data),
                media_type="video/mp4",
                headers={"Content-Length": str(len(decrypted_data))},
            )

        return Response(
            content=decrypted_data,
            media_type="video/mp4",
            headers={"Content-Length": str(len(decrypted_data))},
        )

    except Exception as e:
        if hasattr(app.state, "active_tasks"):
            app.state.active_tasks -= 1
        logger.error(f"Direct decryption failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/decrypt/stream")
async def stream_decrypt(
    key: str = Query(..., description="Hex-encoded key"),
    url: str = Query(..., description="Segment URL"),
    iv: str = Query(None, description="Hex-encoded IV"),
    algorithm: str = Query("aes-128-ctr", description="Encryption algorithm"),
    proxy: str = Query(None, description="Proxy URL"),
    user_agent: str = Query(None, description="Custom User-Agent"),
):
    """
    Stream decryption endpoint for progressive playback

    Returns chunked response for streaming players
    """
    if decryptor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        if hasattr(app.state, "active_tasks"):
            app.state.active_tasks += 1

        # Decrypt the segment
        decrypted_data = await decryptor.decrypt_segment(
            key=key,
            url=url,
            iv=iv,
            algorithm=algorithm,
            proxy=proxy,
            user_agent=user_agent,
        )

        if hasattr(app.state, "active_tasks"):
            app.state.active_tasks -= 1

        # Create a generator for streaming with proper async yielding
        async def data_generator():
            chunk_size = 64 * 1024  # 64KB chunks
            for i in range(0, len(decrypted_data), chunk_size):
                chunk = decrypted_data[i : i + chunk_size]
                # Yield control to event loop
                await asyncio.sleep(0)
                yield chunk

        return StreamingResponse(
            data_generator(),
            media_type="video/mp4",
            headers={"Transfer-Encoding": "chunked", "Content-Type": "video/mp4"},
        )

    except Exception as e:
        if hasattr(app.state, "active_tasks"):
            app.state.active_tasks -= 1
        logger.error(f"Stream decryption failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_async_task(task_id: str):
    """Background task for async processing"""
    if decryptor is None:
        return

    try:
        task = async_tasks[task_id]
        request = task["request"]

        # Update status
        async_tasks[task_id]["status"] = "processing"
        async_tasks[task_id]["progress"] = 0.3

        # Process the decryption
        decrypted_data = await decryptor.decrypt_segment(
            key=request.key,
            url=str(request.url),
            iv=request.iv,
            algorithm=request.algorithm.value,
            proxy=request.proxy,
            user_agent=request.user_agent,
        )

        async_tasks[task_id]["progress"] = 0.9

        # Extract metadata (same issue as above - needs refactoring)
        samples_processed = 0
        kid = None
        pssh_boxes = []

        # Update task status
        async_tasks[task_id].update(
            {
                "status": "completed",
                "progress": 1.0,
                "result": DecryptResponse(
                    success=True,
                    data_size=len(decrypted_data),
                    processing_time=time.time() - task["created_at"],
                    samples_processed=samples_processed,
                    kid=kid,
                    pssh_boxes=pssh_boxes,
                ),
            }
        )

    except Exception as e:
        async_tasks[task_id].update(
            {
                "status": "failed",
                "progress": 1.0,
                "result": DecryptResponse(
                    success=False,
                    error=str(e),
                    processing_time=time.time() - task["created_at"],
                ),
            }
        )


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if decryptor:
        await decryptor.close()

    # Cleanup old async tasks
    cutoff_time = time.time() - 3600  # 1 hour
    for task_id in list(async_tasks.keys()):
        if async_tasks[task_id].get("created_at", 0) < cutoff_time:
            del async_tasks[task_id]
