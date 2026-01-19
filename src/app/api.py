import asyncio
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
from .utils.utils import compose_url_from_template

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
    return f"cache_{hash(cache_string)}"


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


@app.get("/proxy/{encoded_url:path}")
async def proxy_segment(
    encoded_url: str,
    proxy: Optional[str] = Query(None, description="Proxy URL"),
    ua: Optional[str] = Query(None, description="Custom User-Agent"),
):
    """
    Proxy media segments through HTTP request

    URL format: /api/proxy/<base64_encoded_base_url>/<optional_template_path>
    Example: /api/proxy/aHR0cHM6Ly9jZG4uZXhhbXBsZS5jb20vcGF0aA==/segment-123.m4s?
    proxy=http://proxy:8080
    """
    if decryptor is None:
        return Response(
            content='{"error": "Service not initialized"}',
            status_code=503,
            media_type="application/json",
        )

    try:
        # Split the encoded_url into base64 part and optional suffix
        parts = encoded_url.split("/", 1)
        base64_part = parts[0]
        template_suffix = parts[1] if len(parts) > 1 else ""

        # Compose the full URL
        try:
            original_url = compose_url_from_template(base64_part, template_suffix)
        except ValueError as decode_err:
            logger.error(f"Failed to decode proxy URL: {decode_err}")
            return Response(
                content='{"error": "Invalid encoded URL"}',
                status_code=400,
                media_type="application/json",
            )

        logger.debug("Proxy request:")
        logger.debug(f"  Base64 part: {base64_part[:50]}...")
        logger.debug(f"  Template suffix: {template_suffix}")
        logger.debug(f"  Final URL: {original_url}")

        logger.info(f"Fetching media segment: {original_url[:100]}...")

        # Download via decryptor service
        result = await decryptor.download_segment(url=original_url, proxy=proxy, user_agent=ua)

        logger.info(f"Successfully fetched segment, size: {len(result.data)} bytes")

        # Prepare response headers
        response_headers = {}

        # Set Content-Type (pass through from upstream)
        content_type = result.headers.get("Content-Type", "application/octet-stream")

        # Add Content-Length if available
        if "Content-Length" in result.headers:
            response_headers["Content-Length"] = result.headers["Content-Length"]

        # Copy other potentially useful headers
        for header in ["Cache-Control", "ETag", "Last-Modified"]:
            if header in result.headers:
                response_headers[header] = result.headers[header]

        # Return the content directly
        return Response(content=result.data, media_type=content_type, headers=response_headers)

    except Exception as proxy_err:
        logger.error(f"Proxy error: {str(proxy_err)}", exc_info=True)
        return Response(
            content=f'{{"error": "Proxy failed: {str(proxy_err)}"}}',
            status_code=502,
            media_type="application/json",
        )


@app.get("/decrypt/{encoded_url:path}")
async def decrypt_segment_endpoint(
    encoded_url: str,
    key: str = Query(..., description="Hex-encoded decryption key"),
    proxy: Optional[str] = Query(None, description="Proxy URL"),
    ua: Optional[str] = Query(None, description="Custom User-Agent"),
):
    """
    Decrypt media segments on-the-fly

    URL format: /api/decrypt/<base64_encoded_base_url>/<optional_template_path>?key=...
    Example: /api/decrypt/aHR0cHM6Ly9jZG4uZXhhbXBsZS5jb20vcGF0aA==/segment-123.m4s?
    key=0123456789abcdef0123456789abcdef
    """
    if decryptor is None:
        return Response(
            content='{"error": "Service not initialized"}',
            status_code=503,
            media_type="application/json",
        )

    try:
        # Split the encoded_url into base64 part and optional suffix
        parts = encoded_url.split("/", 1)
        base64_part = parts[0]
        template_suffix = parts[1] if len(parts) > 1 else ""

        # Compose the full URL
        try:
            original_url = compose_url_from_template(base64_part, template_suffix)
        except ValueError as decode_err:
            logger.error(f"Failed to decode URL: {decode_err}")
            return Response(
                content='{"error": "Invalid encoded URL"}',
                status_code=400,
                media_type="application/json",
            )

        logger.debug("Decrypt request:")
        logger.debug(f"  Base64 part: {base64_part[:50]}...")
        logger.debug(f"  Template suffix: {template_suffix}")
        logger.debug(f"  Final URL: {original_url}")

        logger.info(f"Decrypting media segment: {original_url[:100]}...")

        if hasattr(app.state, "active_tasks"):
            app.state.active_tasks += 1

        # Download the segment first
        download_result = await decryptor.download_segment(
            url=original_url, proxy=proxy, user_agent=ua
        )

        # Convert to bytearray for in-place decryption
        from .services.mp4_parser import MP4Parser

        data = bytearray(download_result.data)

        # Parse and decrypt MP4 structure
        parser = MP4Parser(data, key=key, debug=False)

        if not parser.parse():
            if hasattr(app.state, "active_tasks"):
                app.state.active_tasks -= 1
            logger.error("Failed to parse/decrypt MP4 structure")
            return Response(
                content='{"error": "Failed to parse/decrypt MP4 structure"}',
                status_code=500,
                media_type="application/json",
            )

        if hasattr(app.state, "active_tasks"):
            app.state.active_tasks -= 1

        logger.info(f"Successfully decrypted segment, size: {len(data)} bytes")

        # Prepare response headers
        response_headers = {}

        # Set Content-Type (pass through from upstream)
        content_type = download_result.headers.get("Content-Type", "video/mp4")

        # Add Content-Length if available (should be same as decryption is in-place)
        if "Content-Length" in download_result.headers:
            response_headers["Content-Length"] = download_result.headers["Content-Length"]

        # Copy other potentially useful headers
        for header in ["Cache-Control", "ETag", "Last-Modified"]:
            if header in download_result.headers:
                response_headers[header] = download_result.headers[header]

        # Return the decrypted content
        return Response(content=bytes(data), media_type=content_type, headers=response_headers)

    except ValueError as e:
        if hasattr(app.state, "active_tasks"):
            app.state.active_tasks -= 1
        logger.error(f"Validation error: {str(e)}")
        return Response(
            content=f'{{"error": "{str(e)}"}}', status_code=400, media_type="application/json"
        )
    except Exception as decrypt_err:
        if hasattr(app.state, "active_tasks"):
            app.state.active_tasks -= 1
        logger.error(f"Decrypt error: {str(decrypt_err)}", exc_info=True)
        return Response(
            content=f'{{"error": "Decryption failed: {str(decrypt_err)}"}}',
            status_code=502,
            media_type="application/json",
        )


@app.post("/decrypt/json", response_model=DecryptResponse)
async def decrypt_json_endpoint(request: DecryptRequest):
    """
    Decrypt a single MP4 segment (JSON request/response)
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

        # Decrypt the segment with metadata extraction
        result = await decryptor.decrypt_segment_with_metadata(
            key=request.key,
            url=str(request.url),
            iv=request.iv,
            algorithm=request.algorithm.value,
            proxy=request.proxy,
            user_agent=request.user_agent,
        )

        # Cache the result with metadata
        cache_data = {
            "data": result.data,
            "data_size": len(result.data),
            "samples_processed": result.samples_processed,
            "kid": result.kid,
            "pssh_boxes": result.pssh_boxes,
        }
        cache.set(cache_key, cache_data)

        if hasattr(app.state, "active_tasks"):
            app.state.active_tasks -= 1

        return DecryptResponse(
            success=True,
            data_size=len(result.data),
            processing_time=time.time() - start_time,
            samples_processed=result.samples_processed,
            kid=result.kid,
            pssh_boxes=result.pssh_boxes,
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
    tasks = [asyncio.create_task(decrypt_json_endpoint(req)) for req in request.requests]

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

        # Process the decryption with metadata
        result = await decryptor.decrypt_segment_with_metadata(
            key=request.key,
            url=str(request.url),
            iv=request.iv,
            algorithm=request.algorithm.value,
            proxy=request.proxy,
            user_agent=request.user_agent,
        )

        async_tasks[task_id]["progress"] = 0.9

        # Update task status
        async_tasks[task_id].update(
            {
                "status": "completed",
                "progress": 1.0,
                "result": DecryptResponse(
                    success=True,
                    data_size=len(result.data),
                    processing_time=time.time() - task["created_at"],
                    samples_processed=result.samples_processed,
                    kid=result.kid,
                    pssh_boxes=result.pssh_boxes,
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
