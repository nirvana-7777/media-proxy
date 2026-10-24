import asyncio
import base64
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from urllib.parse import parse_qs

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
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
from .services.stream_proxy import StreamProxyService
from .utils.utils import decode_base64_url

logger = logging.getLogger(__name__)

# Global services - will be initialized via lifespan
decryptor: Optional[DecryptorService] = None
cache: Optional[LRUCache] = None
stream_proxy: Optional[StreamProxyService] = None
async_tasks: Dict[str, dict] = {}


def init_services(
    decryptor_service: DecryptorService,
    cache_service: LRUCache,
    stream_proxy_service: Optional[StreamProxyService] = None,
) -> None:
    """Initialize global services (called from main.py lifespan)"""
    global decryptor, cache, stream_proxy
    decryptor = decryptor_service
    cache = cache_service
    stream_proxy = stream_proxy_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Lifespan handler — startup is handled by main.py; we own shutdown cleanup here."""
    yield
    # Shutdown: close services and purge stale async tasks
    if stream_proxy:
        await stream_proxy.close()
    if decryptor:
        await decryptor.close()
    cutoff_time = time.time() - 3600  # 1 hour
    for task_id in list(async_tasks.keys()):
        if async_tasks[task_id].get("created_at", 0) < cutoff_time:
            del async_tasks[task_id]


app = FastAPI(
    title="MP4 Segment Decryptor API",
    description="High-performance API for decrypting encrypted MP4 media segments",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_headers_param(value: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Decode a base64url-encoded JSON headers dict (no padding required on input).

    The sending side encodes with:
        base64.urlsafe_b64encode(json.dumps(headers).encode()).decode().rstrip("=")

    We restore the stripped padding before decoding.
    """
    if not value:
        return None
    padded = value + "=" * ((4 - len(value) % 4) % 4)
    data = json.loads(base64.urlsafe_b64decode(padded))
    if not isinstance(data, dict):
        raise ValueError("'headers' must encode a JSON object")
    return {str(k): v for k, v in data.items()}


def _get_request_headers(request: DecryptRequest) -> Optional[Dict[str, str]]:
    """
    Safely retrieve the optional segment_headers field from a DecryptRequest.

    The field is named `segment_headers` on the schema to avoid colliding with
    FastAPI/Starlette's own `.headers` attribute on request objects.
    Returns None if the field doesn't exist yet (backward compat during migration).
    """
    return getattr(request, "segment_headers", None)


def _create_cache_key(request: DecryptRequest) -> str:
    """
    Create a cache key from request parameters.

    Uses Python's built-in hash() — fast enough for in-memory lookup,
    not intended for cryptographic use.
    """
    seg_headers = _get_request_headers(request)
    cache_parts = [
        request.key or "",
        str(request.url),
        request.iv or "",
        request.algorithm.value,
        request.proxy or "",
        request.user_agent or "",
        json.dumps(seg_headers, sort_keys=True) if seg_headers else "",
    ]
    cache_string = ":".join(cache_parts)
    return f"cache_{hash(cache_string)}"


def _parse_encoded_url(
    encoded_url: str,
) -> tuple[str, Optional[str], Optional[str], Optional[Dict[str, str]], str]:
    """
    Decode a base64-encoded parameter block and optional trailing template suffix.

    Returns: (original_url_base, proxy, ua, extra_headers, template_suffix)
    Raises ValueError on any decode/parse failure.
    """
    parts = encoded_url.split("/", 1)
    base64_part = parts[0]
    template_suffix = parts[1] if len(parts) > 1 else ""

    decoded = decode_base64_url(base64_part)
    params = parse_qs(decoded)

    def get_param(param_key: str) -> Optional[str]:
        values = params.get(param_key)
        return values[0] if values else None

    original_url_base = get_param("url")
    if not original_url_base:
        raise ValueError("Missing 'url' parameter in encoded data")

    proxy = get_param("proxy")
    ua = get_param("ua")
    extra_headers = _decode_headers_param(get_param("headers"))

    return original_url_base, proxy, ua, extra_headers, template_suffix


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


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


def _json_error(status_code: int, message: str) -> Response:
    """Tiny JSON error document (json.dumps -- never string interpolation)."""
    return Response(
        content=json.dumps({"error": message}),
        status_code=status_code,
        media_type="application/json",
    )


@app.api_route("/proxy/{encoded_url:path}", methods=["GET", "HEAD"])
async def proxy_segment(request: Request, encoded_url: str):
    """
    Stream one binary media resource from the upstream CDN (HLS phase 2).

    Serves HLS media segments, ``EXT-X-MAP`` init segments and AES-128 key
    files, plus DASH segments on the very same URL contract (template suffix
    included).  The token is the DASH format, unchanged:

        url={original}&proxy={proxy}&ua={ua}&headers={base64_json}

    Behaviour contract (see services/stream_proxy.py):
      * the client's ``Range`` is forwarded upstream, nothing else of the
        client's is; the token's headers are what the CDN sees
      * GET streams the body in chunks; HEAD returns headers only
      * upstream statuses are passed through (403/404/410 reach the player
        as themselves); 502 = connect error, 504 = timeout, nothing else
      * URLs are forwarded byte-for-byte, signed queries included
    """
    if stream_proxy is None:
        return _json_error(503, "Service not initialized")

    try:
        original_url_base, proxy, ua, extra_headers, template_suffix = _parse_encoded_url(
            encoded_url
        )
    except ValueError as decode_err:
        # Never log the token itself -- only the failure reason.
        logger.error("Failed to decode proxy URL: %s", decode_err)
        return _json_error(400, "Invalid encoded URL")

    if not original_url_base.lower().startswith(("http://", "https://")):
        return _json_error(400, "Unsupported URL scheme")

    # DASH template remainder (HLS tokens carry no suffix); identical
    # composition to the previous implementation.
    if template_suffix:
        original_url = original_url_base.rstrip("/") + "/" + template_suffix
    else:
        original_url = original_url_base

    try:
        outcome = await stream_proxy.proxy(
            method=request.method.upper(),
            url=original_url,
            proxy=proxy,
            token_headers=extra_headers,
            token_user_agent=ua,
            client_range=request.headers.get("range"),
        )
    except Exception as exc:
        # Unexpected internal error: 500, no internals leaked.
        logger.error("proxy internal error: %s", type(exc).__name__)
        return _json_error(500, "Internal proxy error")

    if request.method.upper() == "HEAD":
        # Headers only -- Content-Length/Content-Range stay the upstream ones.
        return Response(status_code=outcome.status, headers=outcome.headers)

    if outcome.body is not None:
        return StreamingResponse(
            outcome.body, status_code=outcome.status, headers=outcome.headers
        )

    if outcome.message is not None:
        # Pass-through upstream statuses and proxy-generated 502/504, with a
        # one-line JSON reason.
        return Response(
            content=json.dumps({"error": outcome.message}),
            status_code=outcome.status,
            media_type="application/json",
            headers=outcome.headers,
        )

    return Response(status_code=outcome.status, headers=outcome.headers)


@app.get("/decrypt/{encoded_url:path}")
async def decrypt_segment_endpoint(encoded_url: str):
    """
    Process media segments (with or without decryption).
    All parameters are embedded in the base64-encoded URL block.
    Format: url={original}&key={key}&kid={kid}&proxy={proxy}&ua={ua}&headers={base64_json}
    """
    if decryptor is None:
        return Response(
            content='{"error": "Service not initialized"}',
            status_code=503,
            media_type="application/json",
        )
    try:
        try:
            original_url_base, proxy, ua, extra_headers, template_suffix = _parse_encoded_url(
                encoded_url
            )
            # key/kid live in the same encoded block; re-use the already-parsed params
            decoded = decode_base64_url(encoded_url.split("/", 1)[0])
            params = parse_qs(decoded)

            def get_param(param_key: str) -> Optional[str]:
                values = params.get(param_key)
                return values[0] if values else None

            enc_key = get_param("key")
            kid = get_param("kid")

        except ValueError as decode_err:
            logger.error(f"Failed to decode URL: {decode_err}")
            return Response(
                content='{"error": "Invalid encoded URL"}',
                status_code=400,
                media_type="application/json",
            )

        original_url = (
            original_url_base.rstrip("/") + "/" + template_suffix
            if template_suffix
            else original_url_base
        )

        logger.info(f"Processing segment: {original_url[:100]}...")
        logger.info(f"Parameters - key: {'***' if enc_key else None}, kid: {kid}, proxy: {proxy}")

        if hasattr(app.state, "active_tasks"):
            app.state.active_tasks += 1

        result = await decryptor.decrypt_segment_with_metadata(
            url=original_url,
            key=enc_key,
            kid=kid,
            proxy=proxy,
            user_agent=ua,
            headers=extra_headers,
        )

        if hasattr(app.state, "active_tasks"):
            app.state.active_tasks -= 1

        logger.info(f"Successfully processed segment, size: {len(result.data)} bytes")
        if result.kid:
            logger.info(f"Extracted KID: {result.kid}")

        response_headers: Dict[str, str] = {}
        content_type = "video/mp4"

        if result.kid:
            response_headers["X-Content-KID"] = result.kid

        if result.pssh_boxes:
            response_headers["X-Content-PSSH-Count"] = str(len(result.pssh_boxes))
            for i, pssh in enumerate(result.pssh_boxes[:3]):
                response_headers[f"X-Content-PSSH-{i}"] = pssh[:50]

        response_headers["Content-Length"] = str(len(result.data))

        if enc_key:
            response_headers["X-Content-Decrypted"] = "true"
            if result.samples_processed:
                response_headers["X-Content-Samples"] = str(result.samples_processed)
        else:
            response_headers["X-Content-Decrypted"] = "false"

        return Response(content=result.data, media_type=content_type, headers=response_headers)

    except ValueError as e:
        if hasattr(app.state, "active_tasks"):
            app.state.active_tasks -= 1
        logger.error(f"Validation error: {str(e)}")
        return Response(
            content=f'{{"error": "{str(e)}"}}', status_code=400, media_type="application/json"
        )
    except Exception as err:
        if hasattr(app.state, "active_tasks"):
            app.state.active_tasks -= 1
        logger.error(f"Error: {str(err)}", exc_info=True)
        return Response(
            content=f'{{"error": "Processing failed: {str(err)}"}}',
            status_code=502,
            media_type="application/json",
        )


@app.post("/decrypt/json", response_model=DecryptResponse)
async def decrypt_json_endpoint(request: DecryptRequest):
    """Decrypt a single MP4 segment (JSON request/response)"""
    if decryptor is None or cache is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    start_time = time.time()

    try:
        cache_key = _create_cache_key(request)

        cached = cache.get(cache_key)
        if cached:
            if hasattr(app.state, "cache_hits"):
                app.state.cache_hits += 1
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

        result = await decryptor.decrypt_segment_with_metadata(
            url=str(request.url),
            key=request.key,
            iv=request.iv,
            algorithm=request.algorithm.value,
            proxy=request.proxy,
            user_agent=request.user_agent,
            headers=_get_request_headers(request),
        )

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
    Decrypt multiple MP4 segments in parallel.

    - **requests**: List of decryption requests (max 100)
    """
    if decryptor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Renamed from `tasks` to avoid shadowing the module-level `async_tasks` dict
    batch_tasks = [asyncio.create_task(decrypt_json_endpoint(req)) for req in request.requests]
    results = await asyncio.gather(*batch_tasks, return_exceptions=True)

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
    Start an async decryption task.

    Returns a task ID that can be used to check status.
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

    background_tasks.add_task(process_async_task, task_id)

    return {"task_id": task_id, "status": "processing"}


@app.get("/decrypt/async/{task_id}", response_model=AsyncTaskResponse)
async def get_async_result(task_id: str):
    """Get the result of an async decryption task"""
    if task_id not in async_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    async_task = async_tasks[task_id]
    return AsyncTaskResponse(
        task_id=task_id,
        status=async_task["status"],
        result=async_task["result"],
    )


@app.get("/decrypt/stream")
async def stream_decrypt(
    url: str = Query(..., description="Segment URL"),
    enc_key: str = Query(..., alias="key", description="Hex-encoded key"),
    iv: str = Query(None, description="Hex-encoded IV"),
    algorithm: str = Query("aes-128-ctr", description="Encryption algorithm"),
    proxy: str = Query(None, description="Proxy URL"),
    user_agent: str = Query(None, description="Custom User-Agent"),
    headers_param: str = Query(
        None, alias="headers", description="Base64url-encoded JSON headers dict"
    ),
):
    """
    Stream decryption endpoint for progressive playback.

    Returns a chunked response for streaming players.
    """
    if decryptor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        if hasattr(app.state, "active_tasks"):
            app.state.active_tasks += 1

        extra_headers = _decode_headers_param(headers_param)

        decrypted_data = await decryptor.decrypt_segment(
            url=url,
            key=enc_key,
            iv=iv,
            algorithm=algorithm,
            proxy=proxy,
            user_agent=user_agent,
            headers=extra_headers,
        )

        if hasattr(app.state, "active_tasks"):
            app.state.active_tasks -= 1

        async def data_generator():
            chunk_size = 64 * 1024  # 64 KB chunks
            for i in range(0, len(decrypted_data), chunk_size):
                chunk = decrypted_data[i : i + chunk_size]
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


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------


async def process_async_task(task_id: str) -> None:
    """Background worker for async decryption tasks"""
    if decryptor is None:
        return

    if task_id not in async_tasks:
        logger.error(f"process_async_task: unknown task_id {task_id!r}")
        return

    async_task = async_tasks[task_id]
    request: DecryptRequest = async_task["request"]

    try:
        async_tasks[task_id]["status"] = "processing"
        async_tasks[task_id]["progress"] = 0.3

        result = await decryptor.decrypt_segment_with_metadata(
            url=str(request.url),
            key=request.key,
            iv=request.iv,
            algorithm=request.algorithm.value,
            proxy=request.proxy,
            user_agent=request.user_agent,
            headers=_get_request_headers(request),
        )

        async_tasks[task_id]["progress"] = 0.9
        async_tasks[task_id].update(
            {
                "status": "completed",
                "progress": 1.0,
                "result": DecryptResponse(
                    success=True,
                    data_size=len(result.data),
                    processing_time=time.time() - async_task["created_at"],
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
                    processing_time=time.time() - async_task["created_at"],
                ),
            }
        )