"""Streaming proxy engine behind ``/api/proxy`` (HLS phase 2 + DASH segments).

The endpoint carries *binary* HLS/DASH resources from an upstream CDN to the
player: media segments, ``EXT-X-MAP`` init segments and AES-128 key files.
Playlists are never fetched or rewritten here -- if one reaches this module
its bytes are passed through unchanged.

Guarantees implemented here
----------------------------
* The upstream URL is sent byte-for-byte (``yarl.URL(..., encoded=True)``);
  signed query strings are never re-encoded or normalised.
* The token's headers are the *only* headers sent upstream, plus the client's
  ``Range`` when present.  When the token carries no User-Agent a constant
  neutral UA is used (never the client's).  Nothing else from the client is
  forwarded.
* Redirects (<= ``redirect_limit``) are followed manually so the proxy applies
  to every hop: ``Referer``/``Origin``/``User-Agent`` survive all hops, while
  ``Cookie``/``Authorization`` are only ever sent to the original host.
* Bodies are streamed in bounded chunks and never buffered.  When the player
  disconnects, the upstream read is stopped and the connection released
  promptly (shielded cleanup inside the stream iterator).
* Upstream statuses are passed through verbatim.  The proxy itself only ever
  produces 502 ("upstream connection failed") for connect errors and 504
  ("upstream timeout") for timeouts -- players must be able to tell an
  expired token (403/410) from a network failure.
* Content types: ``Content-Type``/``ETag``/``Last-Modified``/``Cache-Control``
  are passed through; generic upstream types (``application/octet-stream``,
  ``binary/octet-stream``, ``text/plain`` or missing) are re-derived from the
  upstream path extension.  ``.key`` responses are forced to
  ``Cache-Control: no-store``.
* Logging policy: host, status, size and duration only.  The token, the URL
  (its query can hold signed material), header values and proxy URLs are
  never written to the log.

Pooling: one ``aiohttp.ClientSession`` per distinct proxy (plus one for
direct traffic), each with a bounded connector (``limit_per_host``).  A
stalled upstream trips ``sock_read`` and never holds a worker: no semaphore
is held while a body is being streamed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, Optional

import aiohttp
import yarl

from ..utils.utils import socks_error_types
from .decryptor import DEFAULT_USER_AGENT

logger = logging.getLogger(__name__)

__all__ = ["StreamProxyService", "StreamProxyConfig", "ProxyOutcome"]


# ---------------------------------------------------------------------------
# Policy tables
# ---------------------------------------------------------------------------

#: Upstream types considered "generic" -- trigger re-derivation from the
#: extension of the upstream path.
GENERIC_UPSTREAM_TYPES = frozenset(
    {"application/octet-stream", "binary/octet-stream", "text/plain"}
)

#: Extension -> Content-Type (HLS carrier contract, phase 2).
EXTENSION_CONTENT_TYPES: Dict[str, str] = {
    ".ts": "video/mp2t",
    ".m4s": "video/mp4",
    ".mp4": "video/mp4",
    ".cmfv": "video/mp4",
    ".m4a": "audio/mp4",
    ".cmfa": "audio/mp4",
    ".aac": "audio/aac",
    ".vtt": "text/vtt",
    ".webvtt": "text/vtt",
    ".key": "application/octet-stream",
}

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

#: Never forwarded upstream, whatever the token says (hop-by-hop or framed
#: by the transport itself).
_STRIP_REQUEST_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "te",
        "trailer",
        "upgrade",
        "proxy-authorization",
        "proxy-authenticate",
        "proxy-connection",
    }
)

#: Sent to the original host only -- dropped on cross-host redirect hops.
_ORIGINAL_HOST_ONLY = frozenset({"cookie", "authorization"})

#: aiohttp injects ``Accept: */*`` and ``Accept-Encoding: gzip, deflate, br``
#: on its own.  Neither belongs to the token's header set, and advertising
#: gzip would invite content transformation we must not do -- so both are
#: suppressed.  Values the token explicitly carries are never affected.
_SKIP_AUTO_HEADERS = frozenset({"Accept", "Accept-Encoding"})

#: Response headers copied from upstream onto proxied responses.
_PASSTHROUGH_RESPONSE_HEADERS = (
    "Content-Length",
    "Content-Range",
    "ETag",
    "Last-Modified",
    "Cache-Control",
    "Content-Encoding",
    "Location",
)


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass
class ProxyOutcome:
    """What ``/api/proxy`` should answer with.

    ``body`` is an async iterator of raw upstream bytes (streamed, never
    buffered).  ``body is None`` means "no body": HEAD responses, 204/304,
    and every error -- for those, ``message`` carries a one-line reason that
    the API layer renders as a tiny JSON error document.
    """

    status: int
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[AsyncIterator[bytes]] = None
    message: Optional[str] = None

    @classmethod
    def error(cls, status: int, message: str) -> "ProxyOutcome":
        return cls(status=status, headers={}, body=None, message=message)


@dataclass(frozen=True)
class _ResumeInfo:
    """What is needed to continue a body with ``Range`` after an upstream stall."""

    url: yarl.URL
    headers: Dict[str, str]
    first: int  # absolute offset of the first body byte we were promised
    last: int  # absolute offset of the last body byte (inclusive)


def _resume_info(
    resp: aiohttp.ClientResponse, url: yarl.URL, headers: Dict[str, str]
) -> Optional[_ResumeInfo]:
    """Resumable only for identity-encoded bodies with a known byte span."""
    if resp.headers.get("Content-Encoding", "identity").lower() != "identity":
        return None
    try:
        if resp.status == 206:
            unit, _, rest = (resp.headers.get("Content-Range") or "").partition(" ")
            span = rest.split("/", 1)[0]
            first_s, _, last_s = span.partition("-")
            if unit.lower() != "bytes" or not first_s.isdigit() or not last_s.isdigit():
                return None
            first, last = int(first_s), int(last_s)
        elif resp.status == 200:
            length = int(resp.headers["Content-Length"])
            first, last = 0, length - 1
        else:
            return None
    except (KeyError, ValueError):
        return None
    if last < first:
        return None
    return _ResumeInfo(url=url, headers=dict(headers), first=first, last=last)


class _SetupError(Exception):
    """Failed to create an upstream session (library/config problem)."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "") or default)
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, "") or default)
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class StreamProxyConfig:
    connect_timeout: float = 5.0
    read_timeout: float = 6.0
    redirect_limit: int = 5
    chunk_size: int = 64 * 1024
    pool_max: int = 8
    connect_limit: int = 100
    connect_limit_per_host: int = 10
    session_idle_ttl: float = 120.0
    # Live HLS segments are ~2 s long: a body that stalls for 15 s is useless
    # to the player.  Detect stalls early and recover instead of failing.
    resume_attempts: int = 2  # mid-body Range-resume retries per response
    header_retries: int = 1  # retries when upstream never answers (GET/HEAD)

    @classmethod
    def from_env(cls) -> "StreamProxyConfig":
        return cls(
            connect_timeout=_env_float("MEDIA_PROXY_CONNECT_TIMEOUT", 5.0),
            read_timeout=_env_float("MEDIA_PROXY_READ_TIMEOUT", 6.0),
            redirect_limit=_env_int("MEDIA_PROXY_REDIRECT_LIMIT", 5),
            chunk_size=_env_int("MEDIA_PROXY_CHUNK_SIZE", 64 * 1024),
            pool_max=_env_int("MEDIA_PROXY_POOL_MAX", 8),
            connect_limit=_env_int("MEDIA_PROXY_CONNECT_LIMIT", 100),
            connect_limit_per_host=_env_int("MEDIA_PROXY_CONNECT_LIMIT_PER_HOST", 10),
            session_idle_ttl=_env_float("MEDIA_PROXY_SESSION_IDLE", 120.0),
            resume_attempts=_env_int("MEDIA_PROXY_RESUME_ATTEMPTS", 2),
            header_retries=_env_int("MEDIA_PROXY_HEADER_RETRIES", 1),
        )


# ---------------------------------------------------------------------------
# Session pool: one ClientSession per distinct proxy endpoint
# ---------------------------------------------------------------------------


class _PoolEntry:
    __slots__ = ("key", "session", "per_request_proxy", "refs", "streams", "last_used")

    def __init__(
        self, *, key: str, session: aiohttp.ClientSession, per_request_proxy: Optional[str]
    ):
        self.key = key
        self.session = session
        # HTTP(S) proxies are applied per request; SOCKS is baked into the
        # connector (ProxyConnector) and needs no per-request argument.
        self.per_request_proxy = per_request_proxy
        self.refs = 0
        self.streams = 0  # bodies currently streaming through this session
        self.last_used = time.monotonic()


async def _silent_close(session: aiohttp.ClientSession) -> None:
    with contextlib.suppress(Exception):
        await session.close()


class _SessionPool:
    """Bounded LRU pool of upstream sessions, keyed by proxy URL ('' = direct)."""

    def __init__(self, cfg: StreamProxyConfig) -> None:
        self._cfg = cfg
        self._entries: Dict[str, _PoolEntry] = {}

    # -- lifecycle ---------------------------------------------------------

    def acquire(self, proxy_url: Optional[str]) -> _PoolEntry:
        self._sweep()
        key = proxy_url or ""
        entry = self._entries.get(key)
        if entry is None:
            session, per_request = self._create(proxy_url)
            entry = _PoolEntry(key=key, session=session, per_request_proxy=per_request)
            self._entries[key] = entry
            self._evict_over_cap()
        entry.refs += 1
        entry.last_used = time.monotonic()
        return entry

    def release(self, entry: _PoolEntry) -> None:
        entry.refs = max(0, entry.refs - 1)
        entry.last_used = time.monotonic()

    def begin_stream(self, entry: _PoolEntry) -> None:
        entry.streams += 1

    def end_stream(self, entry: _PoolEntry) -> None:
        entry.streams = max(0, entry.streams - 1)

    async def close_all(self) -> None:
        entries = list(self._entries.values())
        self._entries.clear()
        for entry in entries:
            await _silent_close(entry.session)
        if entries:
            # Give aiohttp a moment to tear down SSL transports cleanly.
            await asyncio.sleep(0.1)

    # -- internals ---------------------------------------------------------

    def _create(self, proxy_url: Optional[str]) -> tuple[aiohttp.ClientSession, Optional[str]]:
        cfg = self._cfg
        timeout = aiohttp.ClientTimeout(
            total=None,  # a slow-but-alive body must not be cut off
            connect=cfg.connect_timeout,
            sock_read=cfg.read_timeout,
        )
        if proxy_url and proxy_url.lower().startswith("socks"):
            try:
                from aiohttp_socks import ProxyConnector
            except ImportError as exc:  # pragma: no cover - depends on deploy
                raise _SetupError(
                    "SOCKS proxy requested but aiohttp-socks is not installed"
                ) from exc
            try:
                connector = ProxyConnector.from_url(proxy_url)
            except Exception as exc:
                raise _SetupError(f"invalid SOCKS proxy URL ({type(exc).__name__})") from exc
            per_request: Optional[str] = None
        else:
            connector = aiohttp.TCPConnector(
                limit=cfg.connect_limit,
                limit_per_host=cfg.connect_limit_per_host,
                enable_cleanup_closed=True,
            )
            per_request = proxy_url or None
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            trust_env=False,  # never consult environment proxies
            auto_decompress=False,  # byte-transparent: pass encoded bytes as-is
        )
        return session, per_request

    def _idle(self, entry: _PoolEntry) -> bool:
        return entry.refs == 0 and entry.streams == 0

    def _sweep(self) -> None:
        now = time.monotonic()
        ttl = self._cfg.session_idle_ttl
        for key in list(self._entries):
            entry = self._entries[key]
            if self._idle(entry) and now - entry.last_used > ttl:
                self._close_entry(entry)

    def _evict_over_cap(self) -> None:
        while len(self._entries) > self._cfg.pool_max:
            victim = self._oldest_idle()
            if victim is None:
                return  # everything busy; stay over cap until the next sweep
            self._close_entry(victim)

    def _oldest_idle(self) -> Optional[_PoolEntry]:
        candidates = [e for e in self._entries.values() if self._idle(e)]
        if not candidates:
            return None
        return min(candidates, key=lambda e: e.last_used)

    def _close_entry(self, entry: _PoolEntry) -> None:
        self._entries.pop(entry.key, None)
        session = entry.session
        if session.closed:
            return
        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().create_task(_silent_close(session))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drop_headers(headers: Dict[str, str], lower_names: frozenset) -> None:
    for key in [k for k in headers if k.lower() in lower_names]:
        del headers[key]


def _extension_of(url: yarl.URL) -> str:
    path = url.path or "/"
    segment = path.rsplit("/", 1)[-1].lower()
    if "." not in segment:
        return ""
    ext = segment[segment.rfind(".") :]
    return ext if 0 < len(ext) <= 16 else ""


def _base_media_type(raw: Optional[str]) -> str:
    if not raw:
        return ""
    return raw.split(";", 1)[0].strip().lower()


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class StreamProxyService:
    """Upstream fetch engine for ``/api/proxy`` (see module docstring)."""

    def __init__(self, config: Optional[StreamProxyConfig] = None) -> None:
        self._cfg = config or StreamProxyConfig()
        self._pool = _SessionPool(self._cfg)

    # -- public API --------------------------------------------------------

    async def proxy(
        self,
        *,
        method: str,
        url: str,
        proxy: Optional[str] = None,
        token_headers: Optional[Dict[str, str]] = None,
        token_user_agent: Optional[str] = None,
        client_range: Optional[str] = None,
    ) -> ProxyOutcome:
        """Fetch one upstream resource.

        Never raises for upstream/network conditions -- failures come back as
        ``ProxyOutcome`` (pass-through status, 502 connect, 504 timeout).
        Only genuinely unexpected bugs propagate to the caller.
        """
        proxy_url = (proxy or "").strip() or None
        if proxy_url is not None:
            scheme = proxy_url.split("://", 1)[0].lower()
            if scheme not in ("http", "https") and not scheme.startswith("socks"):
                return ProxyOutcome.error(400, f"unsupported proxy scheme {scheme!r}")
        try:
            entry = self._pool.acquire(proxy_url)
        except _SetupError as exc:
            logger.warning("proxy session setup failed: %s", exc)
            return ProxyOutcome.error(502, f"upstream session unavailable ({exc})")

        try:
            return await self._fetch(
                entry,
                method=method.upper(),
                url=url,
                token_headers=token_headers,
                token_user_agent=token_user_agent,
                client_range=client_range,
            )
        except Exception:
            # Unexpected internal error -- release and let the caller 500 it.
            self._pool.release(entry)
            raise

    async def close(self) -> None:
        await self._pool.close_all()

    # -- request walking ----------------------------------------------------

    async def _fetch(
        self,
        entry: _PoolEntry,
        *,
        method: str,
        url: str,
        token_headers: Optional[Dict[str, str]],
        token_user_agent: Optional[str],
        client_range: Optional[str],
    ) -> ProxyOutcome:
        base_headers = self._build_request_headers(token_headers, token_user_agent, client_range)

        try:
            original = yarl.URL(url, encoded=True)
        except Exception as exc:
            logger.warning("proxy rejected upstream URL: %s", type(exc).__name__)
            self._pool.release(entry)
            return ProxyOutcome.error(502, "invalid upstream URL")

        if original.scheme not in ("http", "https"):
            self._pool.release(entry)
            return ProxyOutcome.error(400, "unsupported URL scheme")

        original_host = (original.host or "").lower()
        current = original
        started = time.monotonic()
        hop = 0
        header_retries = 0

        while True:
            headers = dict(base_headers)
            if (current.host or "").lower() != original_host:
                _drop_headers(headers, _ORIGINAL_HOST_ONLY)

            try:
                resp = await entry.session.request(
                    method,
                    current,
                    headers=headers,
                    proxy=entry.per_request_proxy,
                    allow_redirects=False,
                    skip_auto_headers=_SKIP_AUTO_HEADERS,
                )
            except (asyncio.TimeoutError, aiohttp.ServerTimeoutError) as exc:
                if header_retries < self._cfg.header_retries:
                    header_retries += 1
                    logger.warning(
                        "proxy upstream timeout before headers host=%s, retry %d/%d: %s",
                        (current.host or "").lower(),
                        header_retries,
                        self._cfg.header_retries,
                        type(exc).__name__,
                    )
                    continue
                return self._network_error(
                    entry, current, 504, "upstream timeout", exc, started, hop
                )
            except (aiohttp.ClientError, OSError, *socks_error_types()) as exc:
                # socks_error_types(): aiohttp-socks 0.12+ wraps connect
                # failures in non-OSError exceptions; without this arm a dead
                # SOCKS endpoint would surface as 500 instead of 502.
                return self._network_error(
                    entry, current, 502, "upstream connection failed", exc, started, hop
                )
            except ValueError as exc:
                # Not a network condition: aiohttp refuses a header name/value in
                # the token (e.g. a newline).  A bad token is a 400, not a 500.
                self._pool.release(entry)
                logger.warning("proxy rejected token headers: %s", type(exc).__name__)
                return ProxyOutcome.error(400, "invalid header in token")

            if resp.status in _REDIRECT_STATUSES:
                location = resp.headers.get("Location", "").strip()
                followable = bool(location) and hop < self._cfg.redirect_limit
                target: Optional[yarl.URL] = None
                if followable:
                    try:
                        target = current.join(yarl.URL(location, encoded=True))
                    except Exception:
                        target = None
                    followable = target is not None and target.scheme in ("http", "https")
                if followable and target is not None:
                    logger.debug(
                        "proxy redirect %d -> host=%s (hop %d)", resp.status, target.host, hop + 1
                    )
                    await resp.release()
                    current = target
                    hop += 1
                    continue
                if location and hop >= self._cfg.redirect_limit:
                    await resp.release()
                    self._pool.release(entry)
                    logger.warning(
                        "proxy gave up after %d redirects host=%s",
                        hop,
                        (current.host or "").lower(),
                    )
                    return ProxyOutcome.error(502, "too many redirects")
                # Un-followable 3xx (no/foreign Location): pass through below.

            return await self._finish(entry, resp, current, method, started, hop, headers)

    async def _finish(
        self,
        entry: _PoolEntry,
        resp: aiohttp.ClientResponse,
        final_url: yarl.URL,
        method: str,
        started: float,
        hop: int,
        request_headers: Dict[str, str],
    ) -> ProxyOutcome:
        status = resp.status
        host = (final_url.host or "").lower()
        elapsed = time.monotonic() - started

        if method == "HEAD":
            headers = self._response_headers(resp.headers, final_url)
            await resp.release()
            self._pool.release(entry)
            logger.info(
                "proxy HEAD host=%s status=%d size=%s redir=%d t=%.3fs",
                host,
                status,
                resp.headers.get("Content-Length") or "-",
                hop,
                elapsed,
            )
            return ProxyOutcome(status=status, headers=headers, body=None)

        if status >= 400:
            # Pass the upstream verdict through untouched; keep a Content-Range
            # (416) if there is one.  No body -- the API layer adds a tiny JSON
            # reason so players can tell expired tokens from network failures.
            error_headers: Dict[str, str] = {}
            content_range = resp.headers.get("Content-Range")
            if content_range:
                error_headers["Content-Range"] = content_range
            await resp.release()
            self._pool.release(entry)
            logger.info(
                "proxy %s host=%s status=%d (upstream) redir=%d t=%.3fs",
                method,
                host,
                status,
                hop,
                elapsed,
            )
            return ProxyOutcome(
                status=status,
                headers=error_headers,
                body=None,
                message=f"upstream returned {status}",
            )

        if status < 200 or status in (204, 304):
            # No body by definition; pass headers through.
            headers = self._response_headers(resp.headers, final_url)
            await resp.release()
            self._pool.release(entry)
            logger.info(
                "proxy %s host=%s status=%d redir=%d t=%.3fs", method, host, status, hop, elapsed
            )
            return ProxyOutcome(status=status, headers=headers, body=None)

        # 2xx/3xx with a body: stream it, never buffer it.
        headers = self._response_headers(resp.headers, final_url)
        body = self._body_iterator(
            entry, resp, host, started, _resume_info(resp, final_url, request_headers)
        )
        logger.info(
            "proxy %s host=%s status=%d size=%s redir=%d t=%.3fs",
            method,
            host,
            status,
            resp.headers.get("Content-Length") or "chunked",
            hop,
            elapsed,
        )
        return ProxyOutcome(status=status, headers=headers, body=body)

    def _network_error(
        self,
        entry: _PoolEntry,
        current: yarl.URL,
        status: int,
        message: str,
        exc: Exception,
        started: float,
        hop: int,
    ) -> ProxyOutcome:
        self._pool.release(entry)
        host = (current.host or "").lower()
        # Never log str(exc): aiohttp connector errors embed the full URL.
        logger.warning(
            "proxy upstream error host=%s status=%d redir=%d t=%.3fs: %s",
            host,
            status,
            hop,
            time.monotonic() - started,
            type(exc).__name__,
        )
        return ProxyOutcome.error(status, f"{message} ({type(exc).__name__})")

    # -- header policy ------------------------------------------------------

    def _build_request_headers(
        self,
        token_headers: Optional[Dict[str, str]],
        token_user_agent: Optional[str],
        client_range: Optional[str],
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        for key, value in (token_headers or {}).items():
            if key.lower() in _STRIP_REQUEST_HEADERS:
                continue
            headers[key] = value if isinstance(value, str) else str(value)

        if not any(k.lower() == "user-agent" for k in headers):
            ua = (token_user_agent or "").strip()
            # Constant neutral UA when the token provides none -- never the
            # client's, never empty.
            headers["User-Agent"] = ua if ua else DEFAULT_USER_AGENT

        if client_range and client_range.strip():
            headers["Range"] = client_range.strip()
        return headers

    def _response_headers(
        self, upstream: aiohttp.abc.Headers, final_url: yarl.URL
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Content-Type": self._content_type(upstream, final_url),
            "Accept-Ranges": upstream.get("Accept-Ranges") or "bytes",
        }
        for name in _PASSTHROUGH_RESPONSE_HEADERS:
            value = upstream.get(name)
            if value is not None:
                headers[name] = value
        if _extension_of(final_url) == ".key":
            # AES-128 key material: never cacheable, anywhere, by anyone.
            headers["Cache-Control"] = "no-store"
        return headers

    def _content_type(self, upstream: aiohttp.abc.Headers, final_url: yarl.URL) -> str:
        raw = upstream.get("Content-Type")
        base = _base_media_type(raw)
        if base and base not in GENERIC_UPSTREAM_TYPES:
            return raw
        derived = EXTENSION_CONTENT_TYPES.get(_extension_of(final_url))
        if derived:
            return derived
        return raw or "application/octet-stream"

    # -- streaming -----------------------------------------------------------

    async def _resume_request(
        self, entry: _PoolEntry, resume: _ResumeInfo, offset: int
    ) -> Optional[aiohttp.ClientResponse]:
        """Re-request the rest of the body (``offset`` bytes already sent)."""
        start = resume.first + offset
        headers = dict(resume.headers)
        headers["Range"] = f"bytes={start}-{resume.last}"
        resp = await entry.session.request(
            "GET",
            resume.url,
            headers=headers,
            proxy=entry.per_request_proxy,
            allow_redirects=False,
            skip_auto_headers=_SKIP_AUTO_HEADERS,
        )
        # Only a 206 that starts exactly where we stopped is safe to splice in.
        if resp.status == 206:
            info = _resume_info(resp, resume.url, headers)
            if info is not None and info.first == start and info.last == resume.last:
                return resp
        resp.close()
        return None

    def _body_iterator(
        self,
        entry: _PoolEntry,
        resp: aiohttp.ClientResponse,
        host: str,
        started: float,
        resume: Optional[_ResumeInfo] = None,
    ) -> AsyncIterator[bytes]:
        cfg = self._cfg
        self._pool.begin_stream(entry)

        async def iterator() -> AsyncIterator[bytes]:
            completed = False
            total = 0
            attempts = 0
            current = resp
            try:
                while True:
                    try:
                        async for chunk in current.content.iter_chunked(cfg.chunk_size):
                            total += len(chunk)
                            yield chunk
                        completed = True
                        return
                    except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                        can_resume = (
                            resume is not None
                            and attempts < cfg.resume_attempts
                            and resume.first + total <= resume.last
                        )
                        logger.warning(
                            "proxy stream stalled host=%s sent=%dB: %s%s",
                            host,
                            total,
                            type(exc).__name__,
                            " -> resuming with Range" if can_resume else " -> aborting",
                        )
                        if not can_resume:
                            raise
                        # Narrow for mypy: ``can_resume`` implies ``resume is not None``.
                        assert resume is not None
                        attempts += 1
                        current.close()
                        try:
                            nxt = await self._resume_request(entry, resume, total)
                        except (asyncio.TimeoutError, aiohttp.ClientError, OSError):
                            nxt = None
                        if nxt is None:
                            raise exc
                        current = nxt
            finally:
                # Sync bookkeeping first so it always runs, even while the
                # task is being cancelled (player disconnected).
                self._pool.end_stream(entry)
                self._pool.release(entry)
                if completed:
                    logger.info(
                        "proxy stream done host=%s bytes=%d t=%.3fs resumes=%d",
                        host,
                        total,
                        time.monotonic() - started,
                        attempts,
                    )
                # Shielded so the upstream release still happens when the
                # response task is cancelled -- this is what stops the
                # upstream read promptly on client disconnect.
                with contextlib.suppress(Exception):
                    if completed:
                        await asyncio.shield(current.release())
                    else:
                        current.close()  # synchronous: runs even while being cancelled

        return iterator()