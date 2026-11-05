import asyncio
import contextlib
import errno
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, TypedDict

import aiohttp
import yarl

from .mp4_parser import MP4Parser

logger = logging.getLogger(__name__)

# Timeout configuration for DASH segment downloads.
# Segments are typically 2-10s of media, so a 30s timeout just stalls playback
# on proxy failure. Fail fast and let the caller/player retry or adapt.

# Proxy timeout: short connect so a dead/flaky WARP tunnel is detected quickly.
SEGMENT_TIMEOUT_PROXY = aiohttp.ClientTimeout(
    total=10,
    connect=4,
    sock_read=6,
)

# Direct timeout: slightly more generous since there is no proxy overhead.
SEGMENT_TIMEOUT_DIRECT = aiohttp.ClientTimeout(
    total=10,
    connect=4,
    sock_read=6,
)

# HTTP status codes that indicate a definitive server rejection.
# These are never worth retrying — the server made a decision.
# Exception: 403 via proxy triggers a one-shot manifest-refresh retry
# (see _download_segment_internal) before the error is propagated.
_NO_RETRY_STATUSES = {400, 401, 403, 404, 410}

# Default Chrome User-Agent for Windows
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


# Errors that signal "the pooled keep-alive connection was already dead when we
# used it" (server closed it while our request was in flight).  The request never
# reached the server, so retrying immediately on a fresh connection is safe and
# costs one RTT instead of the regular 0.5 s back-off.
_STALE_CONNECTION_ERRORS = tuple(
    exc
    for exc in (
        aiohttp.ServerDisconnectedError,
        # Added in aiohttp 3.10; absent in older pins such as 3.9.x.
        getattr(aiohttp, "ClientConnectionResetError", None),
    )
    if exc is not None
)


def _is_stale_connection_error(exc: BaseException) -> bool:
    if isinstance(exc, _STALE_CONNECTION_ERRORS):
        return True
    # ECONNRESET / EPIPE on a *used* connection; connect failures (refused,
    # unreachable, DNS) are ClientConnectorError and keep the normal back-off.
    return (
        isinstance(exc, aiohttp.ClientOSError)
        and not isinstance(exc, aiohttp.ClientConnectorError)
        and getattr(exc, "errno", None) in (errno.ECONNRESET, errno.EPIPE)
    )


class _DownloadAttempt(TypedDict):
    proxy: Optional[str]
    timeout: aiohttp.ClientTimeout
    delay: float


# Upstream sessions are pooled (see DecryptorService._acquire_session): at most
# this many distinct SOCKS endpoints + the shared HTTP/direct session.
SESSION_POOL_MAX = 8
SESSION_IDLE_TTL = 120.0


class _PooledSession:
    __slots__ = ("key", "session", "refs", "last_used")

    def __init__(self, key: str, session: aiohttp.ClientSession):
        self.key = key
        self.session = session
        self.refs = 0
        self.last_used = time.monotonic()


async def _silent_close(session: aiohttp.ClientSession) -> None:
    with contextlib.suppress(Exception):
        await session.close()


def _merge_headers(
    user_agent: Optional[str], extra_headers: Optional[Dict[str, str]]
) -> Optional[Dict[str, str]]:
    """Per-request headers.  A User-Agent in ``extra_headers`` wins over ``user_agent``;
    without either, the session default (DEFAULT_USER_AGENT) applies."""
    merged: Dict[str, str] = {}
    if user_agent:
        merged["User-Agent"] = user_agent
    for key, value in (extra_headers or {}).items():
        if key.lower() == "user-agent":
            merged.pop("User-Agent", None)
        merged[key] = value
    return merged or None


def _parse_and_decrypt(
    encrypted_data: bytes, key: Optional[str], kid: Optional[str]
) -> Tuple[bytes, "MP4Parser"]:
    """CPU-bound part of a segment request.  Runs in a worker thread so that
    parsing + AES never block the event loop (and with it /api/proxy streams)."""
    data = bytearray(encrypted_data)
    parser = MP4Parser(data, key=key, kid=kid, debug=False)
    if not parser.parse():
        raise Exception("Failed to parse MP4 structure")
    return bytes(data), parser


class DecryptionResult:
    """Container for decryption results including metadata"""

    def __init__(
        self,
        data: bytes,
        samples_processed: int = 0,
        kid: Optional[str] = None,
        pssh_boxes: Optional[List[str]] = None,
    ):
        self.data = data
        self.samples_processed = samples_processed
        self.kid = kid
        self.pssh_boxes = pssh_boxes or []


class DownloadResult:
    """Container for download results including headers"""

    def __init__(self, data: bytes, headers: Dict[str, str]):
        self.data = data
        self.headers = headers


class DecryptorService:
    """Service for downloading and decrypting CENC-encrypted MP4 segments"""

    def __init__(
        self,
        max_concurrent_downloads: int = 10,
        max_proxy_concurrent: Optional[int] = None,
    ):
        """
        Initialize the decryptor service

        Args:
            max_concurrent_downloads: Maximum number of concurrent downloads
            max_proxy_concurrent: Max concurrent requests through an HTTP/SOCKS proxy
                (default: env MAX_PROXY_CONCURRENT, else 3 -- unchanged behaviour).
                Note it is held while the body is read, so with N tracks per
                stream it caps how many segments can be in flight via the proxy.
        """
        if max_proxy_concurrent is None:
            try:
                max_proxy_concurrent = max(1, int(os.getenv("MAX_PROXY_CONCURRENT", "3")))
            except ValueError:
                max_proxy_concurrent = 3
        self._pool: Dict[str, _PooledSession] = {}
        # Small dedicated pool for the CPU-bound parse+AES step.  More threads than
        # cores only add GIL contention for the event loop (measured: fewer workers
        # = shorter loop stalls at the same throughput).  Env: DECRYPT_WORKERS.
        try:
            workers = max(1, int(os.getenv("DECRYPT_WORKERS", "")))
        except ValueError:
            workers = max(1, min(4, os.cpu_count() or 1))
        self._cpu_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cenc")
        self.semaphore = asyncio.Semaphore(max_concurrent_downloads)
        self.proxy_semaphore = asyncio.Semaphore(max_proxy_concurrent)
        self.max_concurrent = max_concurrent_downloads

    def _acquire_session(self, proxy: Optional[str]) -> _PooledSession:
        """Return a pooled keep-alive session.

        HTTP(S) proxies are applied per request (aiohttp's connector pools by
        proxy + host), so they share the direct session.  SOCKS is baked into
        the connector, so each SOCKS endpoint gets its own session.  The
        User-Agent is a per-request header, never a reason for a new session
        (a new session means a new TCP+TLS handshake for every segment).
        """
        self._sweep_sessions()
        key = proxy if (proxy and proxy.lower().startswith("socks")) else ""
        entry = self._pool.get(key)
        if entry is None or entry.session.closed:
            entry = _PooledSession(key, self._create_session(key or None))
            self._pool[key] = entry
            self._evict_over_cap()
        entry.refs += 1
        entry.last_used = time.monotonic()
        return entry

    @staticmethod
    def _release_session(entry: _PooledSession) -> None:
        entry.refs = max(0, entry.refs - 1)
        entry.last_used = time.monotonic()

    def _close_entry(self, entry: _PooledSession) -> None:
        self._pool.pop(entry.key, None)
        if not entry.session.closed:
            with contextlib.suppress(RuntimeError):
                asyncio.get_running_loop().create_task(_silent_close(entry.session))

    def _sweep_sessions(self) -> None:
        now = time.monotonic()
        for entry in list(self._pool.values()):
            if entry.refs == 0 and now - entry.last_used > SESSION_IDLE_TTL:
                self._close_entry(entry)

    def _evict_over_cap(self) -> None:
        while len(self._pool) > SESSION_POOL_MAX:
            idle = [e for e in self._pool.values() if e.refs == 0]
            if not idle:
                return  # everything busy; stay over cap until the next sweep
            self._close_entry(min(idle, key=lambda e: e.last_used))

    @staticmethod
    def _create_session(socks_proxy: Optional[str] = None) -> aiohttp.ClientSession:
        """Create an aiohttp session (default UA; per-request headers override it)."""
        connector: aiohttp.TCPConnector
        if socks_proxy:
            try:
                from aiohttp_socks import ProxyConnector

                connector = ProxyConnector.from_url(socks_proxy)
            except ImportError:
                raise Exception(
                    "SOCKS proxy support requires aiohttp-socks. "
                    "Install with: pip install aiohttp-socks"
                )
        else:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)

        return aiohttp.ClientSession(
            timeout=SEGMENT_TIMEOUT_DIRECT,  # per-request timeouts override this
            connector=connector,
            headers={"User-Agent": DEFAULT_USER_AGENT},
            trust_env=False,  # Don't use environment proxy settings
        )

    async def download_segment(
        self,
        url: str,
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> DownloadResult:
        """
        Download a segment and return data with headers.

        Args:
            url: URL of the segment to download
            proxy: Optional proxy URL
            user_agent: Optional user agent string
            headers: Optional extra request headers (e.g. token auth headers)

        Returns:
            DownloadResult containing data and headers

        Raises:
            Exception: If download fails
        """
        entry = self._acquire_session(proxy)

        try:
            async with self.semaphore:
                data, response_headers = await self._download_segment_internal(
                    url, entry.session, proxy, extra_headers=_merge_headers(user_agent, headers)
                )

                if not data:
                    raise Exception("Downloaded segment is empty")

                return DownloadResult(data=data, headers=response_headers)

        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.error(f"Network error downloading segment from {url}: {str(e)}")
            if proxy:
                raise Exception(f"Failed to download segment via proxy {proxy}: {str(e)}")
            raise Exception(f"Failed to download segment: {str(e)}")
        finally:
            self._release_session(entry)

    async def decrypt_segment(
        self,
        url: str,  # Required, comes first
        key: Optional[str] = None,
        iv: Optional[str] = None,
        kid: Optional[str] = None,
        algorithm: str = "aes-128-ctr",
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> bytes:
        """
        Download and (optionally) decrypt an MP4 segment (backward compatible version).

        Args:
            url: URL of the segment to process
            key: Hex-encoded decryption key (32 hex chars = 16 bytes) - optional
            iv: Optional hex-encoded initialization vector (for testing)
            kid: Optional hex-encoded Key ID
            algorithm: Encryption algorithm (default: aes-128-ctr)
            proxy: Optional proxy URL
            user_agent: Optional user agent string
            headers: Optional extra request headers

        Returns:
            Processed MP4 segment as bytes (decrypted if key provided)

        Raises:
            ValueError: If key format is invalid (when provided)
            Exception: If download or decryption fails
        """
        result = await self.decrypt_segment_with_metadata(
            url=url,
            key=key,
            iv=iv,
            kid=kid,
            algorithm=algorithm,
            proxy=proxy,
            user_agent=user_agent,
            headers=headers,
        )
        return result.data

    async def decrypt_segment_with_metadata(
        self,
        url: str,  # Required, comes first
        key: Optional[str] = None,
        iv: Optional[str] = None,
        kid: Optional[str] = None,
        algorithm: str = "aes-128-ctr",
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> DecryptionResult:
        """
        Download and (optionally) decrypt an MP4 segment, returning data and metadata.

        Args:
            url: URL of the segment to process
            key: Hex-encoded decryption key (32 hex chars = 16 bytes) - optional
            iv: Optional hex-encoded initialization vector (for testing)
            kid: Optional hex-encoded Key ID
            algorithm: Encryption algorithm (default: aes-128-ctr)
            proxy: Optional proxy URL
            user_agent: Optional user agent string
            headers: Optional extra request headers

        Returns:
            DecryptionResult containing data and metadata

        Raises:
            ValueError: If key format is invalid (when provided)
            Exception: If download or decryption fails
        """
        # Validate key format only if provided
        if key:
            if len(key) != 32:
                raise ValueError(f"Key must be 32 hex characters (16 bytes), got {len(key)}")

            try:
                bytes.fromhex(key)
            except ValueError:
                raise ValueError("Key must be valid hexadecimal")

        entry = self._acquire_session(proxy)

        try:
            async with self.semaphore:
                encrypted_data, _ = await self._download_segment_internal(
                    url,
                    entry.session,
                    proxy,
                    extra_headers=_merge_headers(user_agent, headers),
                )

                if not encrypted_data:
                    raise Exception("Downloaded segment is empty")

                # Parse MP4 structure and decrypt (if key provided) off the
                # event loop; the semaphore still bounds concurrent work/memory.
                decrypted, parser = await asyncio.get_running_loop().run_in_executor(
                    self._cpu_pool, _parse_and_decrypt, encrypted_data, key, kid
                )

                return DecryptionResult(
                    data=decrypted,
                    samples_processed=parser.samples_processed if key else 0,
                    kid=parser.get_kid(),
                    pssh_boxes=parser.get_pssh_boxes(),
                )

        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.error(f"Network error downloading segment from {url}: {str(e)}")
            if proxy:
                raise Exception(f"Failed to download segment via proxy {proxy}: {str(e)}")
            raise Exception(f"Failed to download segment: {str(e)}")
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to process segment from {url}: {str(e)}")
            raise
        finally:
            self._release_session(entry)

    @staticmethod
    def _build_manifest_url(segment_url: str) -> str:
        """
        Derive the MPD manifest URL from a segment URL by appending index.mpd.

        The segment URL base (everything up to and including the last '/') is used
        as the manifest directory.  If the URL already ends with index.mpd we leave
        it unchanged so the caller can safely pass any URL.

        Examples
        --------
        https://svc45.…/DASH/dash          → https://svc45.…/DASH/dash/index.mpd
        https://svc45.…/DASH/dash/         → https://svc45.…/DASH/dash/index.mpd
        https://svc45.…/DASH/dash/seg1.mp4 → https://svc45.…/DASH/dash/index.mpd
        https://svc45.…/DASH/dash/index.mpd → (unchanged)
        """
        if segment_url.endswith("index.mpd"):
            return segment_url
        # Strip any trailing filename (non-slash suffix) to get the directory
        base = segment_url.rstrip("/")
        if "." in base.rsplit("/", 1)[-1]:
            # Last path component looks like a filename — drop it
            base = base.rsplit("/", 1)[0]
        return base.rstrip("/") + "/index.mpd"

    async def _touch_manifest(
        self,
        segment_url: str,
        session: aiohttp.ClientSession,
        proxy_url: Optional[str],
        extra_headers: Optional[Dict[str, str]],
        timeout: aiohttp.ClientTimeout,
    ) -> bool:
        """
        Fire a GET request to the MPD manifest derived from *segment_url*.

        We do not parse or use the response body — the sole purpose is to
        re-authenticate the CDN session so subsequent segment requests succeed.

        Returns True if the manifest responded with HTTP 200, False otherwise.
        """
        manifest_url = self._build_manifest_url(segment_url)
        logger.info(f"Touching manifest to refresh CDN auth: {manifest_url}")
        try:
            async with session.get(
                manifest_url,
                proxy=proxy_url,
                headers=extra_headers,
                timeout=timeout,
            ) as resp:
                ok = resp.status == 200
                logger.info(
                    f"Manifest touch returned HTTP {resp.status} — {'ok' if ok else 'unexpected'}"
                )
                return ok
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            logger.warning(f"Manifest touch failed: {type(exc).__name__}: {exc}")
            return False

    async def _download_segment_internal(
        self,
        url: str,
        session: aiohttp.ClientSession,
        proxy: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> tuple[bytes, Dict[str, str]]:
        """
        Download segment with retry logic and return data + headers.

        Args:
            url: URL to download from
            session: ClientSession to use
            proxy: Optional proxy URL (for HTTP/HTTPS proxies)
            extra_headers: Optional per-request headers to merge into the request
                           (e.g. {"Authorization": "Bearer ..."}).
                           These are sent only for this request and do not mutate
                           the shared session headers.

        Special 403 handling when a proxy is active
        --------------------------------------------
        WARP (and similar egress proxies) can rotate the egress IP between the
        MPD fetch and subsequent segment fetches, causing the CDN to return 403
        because the segment token was issued for a different IP.  On the first
        403 we:
          1. Touch the MPD manifest (same URL base + /index.mpd) to re-establish
             the CDN session for the new egress IP.
          2. Wait 1 s.
          3. Retry the segment once more.
        This whole sequence fires at most once per _download_segment_internal
        call — we never enter an infinite 403-refresh loop.

        Returns:
            Tuple of (downloaded data as bytes, response headers dict)

        Raises:
            aiohttp.ClientError: If all retry attempts fail
        """
        # For HTTP/HTTPS proxies, pass the proxy URL per-request.
        # SOCKS proxies are baked into the connector at session creation time.
        proxy_url = proxy if (proxy and not proxy.startswith("socks")) else None

        last_error: Optional[Exception] = None
        manifest_refresh_attempted = False  # guard: fire at most once
        stale_retry_used = False  # guard: one immediate retry per call

        # Attempt schedule:
        #   With proxy:    proxy (immediate) → proxy (0.5s) → proxy (1.5s)
        #                  All attempts always use the proxy — never fall back to direct.
        #   Without proxy: direct (immediate) → direct (0.5s) → direct (1.5s)
        #
        # 403/4xx responses normally short-circuit immediately — no retries —
        # EXCEPT for the special proxy-403 case handled below.
        # connect=2s on proxy attempts means a dead WARP tunnel fails fast.
        if proxy_url:
            attempts: List[_DownloadAttempt] = [
                {"proxy": proxy_url, "timeout": SEGMENT_TIMEOUT_PROXY, "delay": 0.0},
                {"proxy": proxy_url, "timeout": SEGMENT_TIMEOUT_PROXY, "delay": 0.5},
                {"proxy": proxy_url, "timeout": SEGMENT_TIMEOUT_PROXY, "delay": 1.5},
            ]
        else:
            attempts = [
                {"proxy": None, "timeout": SEGMENT_TIMEOUT_DIRECT, "delay": 0.0},
                {"proxy": None, "timeout": SEGMENT_TIMEOUT_DIRECT, "delay": 0.5},
                {"proxy": None, "timeout": SEGMENT_TIMEOUT_DIRECT, "delay": 1.5},
            ]

        for attempt_num, plan in enumerate(attempts):
            if plan["delay"] > 0:
                await asyncio.sleep(plan["delay"])  # sleep outside semaphore

            via = f"proxy {proxy}" if plan["proxy"] else "direct"

            try:
                proxy_ctx = self.proxy_semaphore if plan["proxy"] else nullcontext()
                async with proxy_ctx:
                    async with session.get(
                        yarl.URL(url, encoded=True),
                        proxy=plan["proxy"],
                        headers=extra_headers,  # merged on top of session-level headers by aiohttp
                        timeout=plan["timeout"],
                    ) as response:

                        # Definitive server rejections — no point retrying.
                        if response.status in _NO_RETRY_STATUSES:
                            logger.error(
                                f"Segment request rejected (HTTP {response.status}) "
                                f"via {via}: {url}"
                            )
                            response.raise_for_status()  # raises ClientResponseError immediately

                        response.raise_for_status()
                        data = await response.read()

                        response_headers = {k: v for k, v in response.headers.items()}

                        if not self._is_valid_mp4(data):
                            logger.warning("Downloaded data doesn't appear to be valid MP4")

                        if attempt_num > 0:
                            logger.info(f"Segment succeeded on attempt {attempt_num + 1} via {via}")

                        return data, response_headers

            except aiohttp.ClientResponseError as e:
                last_error = e
                logger.warning(f"Download attempt {attempt_num + 1} ({via}) HTTP {e.status}: {e}")

                # Special case: 403 via proxy likely means the WARP egress IP
                # rotated since the MPD was fetched.  Touch the manifest once to
                # re-authenticate the CDN session, then retry the segment.
                if e.status == 403 and plan["proxy"] and not manifest_refresh_attempted:
                    manifest_refresh_attempted = True
                    await self._touch_manifest(
                        url, session, proxy_url, extra_headers, plan["timeout"]
                    )
                    logger.info("Waiting 1 s before retrying segment after manifest touch …")
                    await asyncio.sleep(1.0)
                    # Continue to the next loop iteration (which will retry the segment).
                    # We do NOT raise here, so the retry loop carries on.
                    continue

                if e.status in _NO_RETRY_STATUSES:
                    raise  # No retry — server made a decision.

            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                last_error = e
                logger.warning(
                    f"Download attempt {attempt_num + 1} ({via}) failed: {type(e).__name__}: {e}"
                )
                if plan["proxy"]:
                    logger.error(f"Proxy request failed: {e} — will retry via proxy")

                # Stale keep-alive connection: retry right away on a fresh one.
                if attempt_num == 0 and not stale_retry_used and _is_stale_connection_error(e):
                    stale_retry_used = True
                    attempts[1]["delay"] = 0.0

        raise last_error or Exception("Download failed after all retries")

    @staticmethod
    def _is_valid_mp4(data: bytes) -> bool:
        """
        Quick check if data looks like valid MP4 or other expected media format.

        Args:
            data: Data to check

        Returns:
            True if data appears to be a valid media format
        """
        if len(data) < 8:
            return False

        # Check for JPEG/image formats
        if data[:2] == b"\xff\xd8":  # JPEG signature
            return True
        if data[:4] == b"\x89PNG":  # PNG signature
            return True

        # Check for common MP4 box types at start
        common_types = [
            b"ftyp",
            b"styp",
            b"moof",
            b"moov",
            b"mdat",
            b"free",
            b"skip",
            b"wide",
        ]
        box_type = data[4:8]

        if box_type in common_types:
            return True

        # Check for subtitle/text formats
        if data[:6] == b"WEBVTT":
            return True

        if data[:5] == b"<?xml" or data[:5] == b"<tt x":
            return True

        # For unknown formats, check if it at least has valid box structure
        try:
            size = int.from_bytes(data[0:4], "big")
            if 8 <= size <= len(data):
                if all(32 <= b <= 126 for b in box_type):
                    return True
        except (ValueError, OverflowError):
            pass

        return len(data) > 0

    async def decrypt_batch(
        self, segments: List[Dict], max_concurrent: Optional[int] = None
    ) -> List[bytes]:
        """
        Decrypt multiple segments concurrently.

        Args:
            segments: List of dicts with 'url', and optional
                      'key', 'kid', 'iv', 'proxy', 'user_agent', 'headers'
            max_concurrent: Override default concurrency limit

        Returns:
            List of decrypted segment data in same order as input
        """
        original_limit = self.semaphore._value
        if max_concurrent and max_concurrent != self.max_concurrent:
            self.semaphore = asyncio.Semaphore(max_concurrent)

        try:
            tasks = []
            for seg in segments:
                task = self.decrypt_segment(
                    url=seg["url"],
                    key=seg.get("key"),
                    kid=seg.get("kid"),
                    iv=seg.get("iv"),
                    proxy=seg.get("proxy"),
                    user_agent=seg.get("user_agent"),
                    headers=seg.get("headers"),
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            decrypted_segments: List[bytes] = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Segment {i} failed: {str(result)}")
                    raise result
                if isinstance(result, bytes):
                    decrypted_segments.append(result)

            return decrypted_segments

        finally:
            if max_concurrent and max_concurrent != self.max_concurrent:
                self.semaphore = asyncio.Semaphore(original_limit)

    async def decrypt_batch_with_metadata(
        self, segments: List[Dict], max_concurrent: Optional[int] = None
    ) -> List[DecryptionResult]:
        """
        Decrypt multiple segments concurrently with metadata.

        Args:
            segments: List of dicts with 'url', and optional
                      'key', 'kid', 'iv', 'proxy', 'user_agent', 'headers'
            max_concurrent: Override default concurrency limit

        Returns:
            List of DecryptionResult objects in same order as input
        """
        original_limit = self.semaphore._value
        if max_concurrent and max_concurrent != self.max_concurrent:
            self.semaphore = asyncio.Semaphore(max_concurrent)

        try:
            tasks = []
            for seg in segments:
                task = self.decrypt_segment_with_metadata(
                    url=seg["url"],
                    key=seg.get("key"),
                    kid=seg.get("kid"),
                    iv=seg.get("iv"),
                    proxy=seg.get("proxy"),
                    user_agent=seg.get("user_agent"),
                    headers=seg.get("headers"),
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            decrypted_segments: List[DecryptionResult] = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Segment {i} failed: {str(result)}")
                    raise result
                if isinstance(result, DecryptionResult):
                    decrypted_segments.append(result)

            return decrypted_segments

        finally:
            if max_concurrent and max_concurrent != self.max_concurrent:
                self.semaphore = asyncio.Semaphore(original_limit)

    async def close(self):
        """Cleanup resources"""
        self._cpu_pool.shutdown(wait=False)
        entries = list(self._pool.values())
        self._pool.clear()
        for entry in entries:
            await _silent_close(entry.session)
        if entries:
            await asyncio.sleep(0.1)  # let aiohttp tear down SSL transports

    async def __aenter__(self):
        """Context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()
