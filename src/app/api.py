from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import Response, StreamingResponse
import base64
import uuid
import time
import hashlib
from typing import Dict, List, Optional
import asyncio
import logging
import io

from .models.schemas import (
    DecryptRequest,
    DecryptResponse,
    BatchDecryptRequest,
    BatchDecryptResponse,
    HealthResponse,
    AsyncTaskResponse,
    DecryptorInfo
)
from .services.decryptor import DecryptorService
from .services.cache import LRUCache
from .services.mp4_parser import MP4Parser

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MP4 Segment Decryptor API",
    description="High-performance API for decrypting encrypted MP4 media segments",
    version="2.0.0"
)

# Initialize services (will be injected by main.py)
decryptor: Optional[DecryptorService] = None
cache: Optional[LRUCache] = None

# Task tracking
async_tasks: Dict[str, dict] = {}


def get_decryptor():
    """Dependency to get decryptor service"""
    if decryptor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return decryptor


def get_cache():
    """Dependency to get cache"""
    if cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    return cache


@app.on_event("startup")
async def startup_event():
    """Initialize services from app state"""
    global decryptor, cache
    if hasattr(app, 'state'):
        decryptor = app.state.get("decryptor")
        cache = app.state.get("cache")

    if decryptor is None:
        decryptor = DecryptorService(max_concurrent_downloads=10)
    if cache is None:
        cache = LRUCache(max_size=1000, ttl=300)

    app.state.start_time = time.time()
    app.state.active_tasks = 0
    app.state.cache_hits = 0
    app.state.cache_misses = 0


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system metrics"""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    return HealthResponse(
        status="healthy",
        version="2.0.0",
        uptime=time.time() - app.state.start_time,
        memory_usage=memory_info.rss / 1024 / 1024,  # MB
        active_tasks=app.state.active_tasks
    )


@app.post("/decrypt", response_model=DecryptResponse)
async def decrypt(
        request: DecryptRequest,
        decryptor_service: DecryptorService = Depends(get_decryptor),
        cache_service: LRUCache = Depends(get_cache)
):
    """
    Decrypt a single MP4 segment

    - **key**: Hex-encoded decryption key (32 chars for AES-128)
    - **url**: URL of the MP4 segment to decrypt
    - **iv**: Optional base64 encoded initialization vector
    - **algorithm**: Decryption algorithm (default: aes-128-ctr)
    - **remove_protection_boxes**: Remove encryption metadata (default: true)
    """
    start_time = time.time()

    try:
        # Create cache key
        cache_key = hashlib.sha256(
            f"{request.key}:{request.url}:{request.iv or ''}:{request.algorithm}".encode()
        ).hexdigest()

        # Check cache first
        cached = cache_service.get(cache_key)
        if cached:
            app.state.cache_hits += 1
            return DecryptResponse(
                success=True,
                data_size=len(cached),
                processing_time=time.time() - start_time,
                samples_processed=0,  # Unknown from cache
                kid=None  # Unknown from cache
            )

        app.state.cache_misses += 1
        app.state.active_tasks += 1

        # Decrypt the segment
        decrypted_data = await decryptor_service.decrypt_segment(
            key=request.key,
            url=str(request.url),
            iv=request.iv,
            algorithm=request.algorithm.value
        )

        # Parse MP4 to get metadata (optional, for response info)
        samples_processed = 0
        kid = None

        if request.remove_protection_boxes:
            # Parse to count samples and get KID
            try:
                data_copy = bytearray(decrypted_data)
                parser = MP4Parser(data_copy, key=request.key, debug=False)
                if parser.parse():
                    samples_processed = len(parser.samples)
                    kid = parser.get_kid()
            except Exception as e:
                logger.warning(f"Failed to parse for metadata: {e}")

        # Cache the result
        cache_service.set(cache_key, decrypted_data)

        app.state.active_tasks -= 1

        return DecryptResponse(
            success=True,
            data_size=len(decrypted_data),
            processing_time=time.time() - start_time,
            samples_processed=samples_processed,
            kid=kid,
            pssh_boxes=[]  # Would need to extract from parser
        )

    except Exception as e:
        app.state.active_tasks -= 1
        logger.error(f"Decryption failed: {e}")
        return DecryptResponse(
            success=False,
            error=str(e),
            processing_time=time.time() - start_time
        )


@app.post("/decrypt/batch", response_model=BatchDecryptResponse)
async def batch_decrypt(
        request: BatchDecryptRequest,
        decryptor_service: DecryptorService = Depends(get_decryptor)
):
    """
    Decrypt multiple MP4 segments in parallel

    - **requests**: List of decryption requests (max 100)
    """
    start_time = time.time()

    # Process in parallel
    tasks = []
    for req in request.requests:
        task = asyncio.create_task(
            decrypt(req, decryptor_service, cache)
        )
        tasks.append(task)

    # Wait for all tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert exceptions to error responses
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append(DecryptResponse(
                success=False,
                error=str(result),
                processing_time=0
            ))
        else:
            processed_results.append(result)

    total_succeeded = sum(1 for r in processed_results if r.success)

    return BatchDecryptResponse(
        results=processed_results,
        total_processed=len(processed_results),
        total_succeeded=total_succeeded,
        total_failed=len(processed_results) - total_succeeded
    )


@app.post("/decrypt/async")
async def async_decrypt(
        request: DecryptRequest,
        background_tasks: BackgroundTasks,
        decryptor_service: DecryptorService = Depends(get_decryptor)
):
    """
    Start async decryption task

    Returns a task ID that can be used to check status
    """
    task_id = str(uuid.uuid4())

    async_tasks[task_id] = {
        "status": "pending",
        "request": request,
        "result": None,
        "created_at": time.time(),
        "progress": 0.0
    }

    # Start background task
    background_tasks.add_task(process_async_task, task_id, decryptor_service)

    return {"task_id": task_id, "status": "processing"}


@app.get("/decrypt/async/{task_id}", response_model=AsyncTaskResponse)
async def get_async_result(task_id: str):
    """Get the result of an async decryption task"""
    if task_id not in async_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = async_tasks[task_id]

    return AsyncTaskResponse(
        task_id=task_id,
        status=task["status"],
        result=task["result"]
    )


@app.get("/decrypt/direct")
async def decrypt_direct(
        key: str = Query(..., description="Hex-encoded key"),
        url: str = Query(..., description="Segment URL"),
        iv: str = Query(None, description="Base64 encoded IV"),
        algorithm: str = Query("aes-128-ctr", description="Encryption algorithm"),
        remove_boxes: bool = Query(True, description="Remove protection boxes"),
        download: bool = Query(False, description="Return as downloadable file"),
        decryptor_service: DecryptorService = Depends(get_decryptor)
):
    """
    Direct decryption endpoint for players/streaming

    Returns the raw decrypted MP4 segment
    """
    try:
        app.state.active_tasks += 1

        # Create request object
        request = DecryptRequest(
            key=key,
            url=url,
            iv=iv,
            algorithm=algorithm,
            remove_protection_boxes=remove_boxes
        )

        # Decrypt the segment
        decrypted_data = await decryptor_service.decrypt_segment(
            key=key,
            url=url,
            iv=iv,
            algorithm=algorithm
        )

        app.state.active_tasks -= 1

        # Prepare response
        if download:
            return Response(
                content=decrypted_data,
                media_type="video/mp4",
                headers={
                    "Content-Disposition": "attachment; filename=segment.mp4",
                    "Content-Length": str(len(decrypted_data))
                }
            )

        # Streaming response for large files
        if len(decrypted_data) > 10 * 1024 * 1024:  # > 10MB
            return StreamingResponse(
                io.BytesIO(decrypted_data),
                media_type="video/mp4",
                headers={"Content-Length": str(len(decrypted_data))}
            )

        return Response(
            content=decrypted_data,
            media_type="video/mp4",
            headers={"Content-Length": str(len(decrypted_data))}
        )

    except Exception as e:
        app.state.active_tasks -= 1
        logger.error(f"Direct decryption failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/decrypt/stream")
async def stream_decrypt(
        key: str = Query(..., description="Hex-encoded key"),
        url: str = Query(..., description="Segment URL"),
        iv: str = Query(None, description="Base64 encoded IV"),
        algorithm: str = Query("aes-128-ctr", description="Encryption algorithm"),
        decryptor_service: DecryptorService = Depends(get_decryptor)
):
    """
    Stream decryption endpoint for progressive playback

    Returns chunked response for streaming players
    """
    try:
        app.state.active_tasks += 1

        # Decrypt the segment
        decrypted_data = await decryptor_service.decrypt_segment(
            key=key,
            url=url,
            iv=iv,
            algorithm=algorithm
        )

        app.state.active_tasks -= 1

        # Create a generator for streaming
        async def data_generator():
            chunk_size = 64 * 1024  # 64KB chunks
            for i in range(0, len(decrypted_data), chunk_size):
                yield decrypted_data[i:i + chunk_size]

        return StreamingResponse(
            data_generator(),
            media_type="video/mp4",
            headers={
                "Transfer-Encoding": "chunked",
                "Content-Type": "video/mp4"
            }
        )

    except Exception as e:
        app.state.active_tasks -= 1
        logger.error(f"Stream decryption failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_async_task(task_id: str, decryptor_service: DecryptorService):
    """Background task for async processing"""
    try:
        task = async_tasks[task_id]
        request = task["request"]

        # Update status
        async_tasks[task_id]["status"] = "processing"
        async_tasks[task_id]["progress"] = 0.3

        # Process the decryption
        decrypted_data = await decryptor_service.decrypt_segment(
            key=request.key,
            url=str(request.url),
            iv=request.iv,
            algorithm=request.algorithm.value
        )

        async_tasks[task_id]["progress"] = 0.9

        # Parse for metadata
        samples_processed = 0
        kid = None
        if request.remove_protection_boxes:
            try:
                data_copy = bytearray(decrypted_data)
                parser = MP4Parser(data_copy, key=request.key, debug=False)
                if parser.parse():
                    samples_processed = len(parser.samples)
                    kid = parser.get_kid()
            except Exception:
                pass

        # Update task status
        async_tasks[task_id].update({
            "status": "completed",
            "progress": 1.0,
            "result": DecryptResponse(
                success=True,
                data_size=len(decrypted_data),
                processing_time=time.time() - task["created_at"],
                samples_processed=samples_processed,
                kid=kid
            )
        })

    except Exception as e:
        async_tasks[task_id].update({
            "status": "failed",
            "progress": 1.0,
            "result": DecryptResponse(
                success=False,
                error=str(e),
                processing_time=time.time() - task["created_at"]
            )
        })


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global decryptor
    if decryptor:
        await decryptor.close()

    # Cleanup old async tasks
    cutoff_time = time.time() - 3600  # 1 hour
    for task_id in list(async_tasks.keys()):
        if async_tasks[task_id].get("created_at", 0) < cutoff_time:
            del async_tasks[task_id]