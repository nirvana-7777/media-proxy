# Media Proxy — HLS phase 2 (proxied HLS carrier)

FastAPI service that carries **binary HLS/DASH resources** from upstream CDNs to
players: media segments, `EXT-X-MAP` init segments, AES-128 key files, and DASH
segments on the very same URL contract. Playlists are never fetched or
rewritten here — if one reaches the proxy its bytes are returned unchanged.
`/api/decrypt` (server-side ClearKey) stays DASH-only, untouched.

The streaming engine lives in `app/services/stream_proxy.py`; the endpoint is
`/api/proxy/<token>` in `app/api.py` (GET **and** HEAD).

## URL contract (unchanged from DASH)

```
{MEDIA_PROXY_URL}/api/proxy/<token>[/template-suffix]
token = base64url, no padding, of a urlencoded query string:
  url      absolute upstream URL (required, http/https only)
  proxy    provider HTTP(S)/SOCKS proxy, scheme://host:port (optional)
  ua       User-Agent (legacy DASH key; optional)
  headers  base64url(no padding) of a JSON object of request headers (optional)
```

HLS tokens carry no path suffix; DASH tokens may carry a template remainder,
composed as before: `url.rstrip("/") + "/" + suffix`.

## Behaviour contract

| Requirement | Behaviour |
|---|---|
| URL fidelity | Upstream URL sent byte-for-byte (`yarl` `encoded=True`); signed queries never re-encoded or normalised, on first hop and through redirects |
| Request headers | Exactly the token's headers, plus the client's `Range` when present. No `Accept`/`Accept-Encoding` is injected. Neutral constant User-Agent when the token has none (never the client's) |
| Proxy | Token's `proxy` applies to the upstream request, every redirect hop and the CDN connection. HTTP(S) per-request, SOCKS via `aiohttp-socks` connectors (clear 502 when the library is absent) |
| Range / HEAD | `Range` forwarded verbatim; 206 + `Content-Range` + `Content-Length` + `Accept-Ranges: bytes` passthrough; open-ended ranges supported; HEAD answers headers only; upstream 416 passed through |
| Redirects | Followed manually, at most `MEDIA_PROXY_REDIRECT_LIMIT` (default 5). `Referer`/`Origin`/`User-Agent` survive every hop; `Cookie`/`Authorization` are sent to the original host only. Exhausted loops → 502 "too many redirects" |
| Streaming | Bodies stream in `MEDIA_PROXY_CHUNK_SIZE` chunks (default 64 KiB), never buffered. Client disconnect stops the upstream read immediately (shielded release) |
| Status mapping | Upstream statuses pass through verbatim. Proxy-generated failures: **502 = connect error only**, **504 = timeout only** — so players can tell an expired token (403/410) from a network failure |
| Content types | `Content-Type`/`ETag`/`Last-Modified`/`Cache-Control` passthrough. Generic upstream types (`application/octet-stream`, `binary/octet-stream`, `text/plain`, missing) are re-derived from the path extension: `.ts`→`video/mp2t`, `.m4s`/`.mp4`/`.cmfv`→`video/mp4`, `.m4a`/`.cmfa`→`audio/mp4`, `.aac`→`audio/aac`, `.vtt`/`.webvtt`→`text/vtt`, `.key`→`application/octet-stream` |
| Keys (`.key`) | `Cache-Control: no-store` is forced; nothing is cached anywhere; the URL is never logged (see below) |
| Logging | `host`, status, size, duration only. Never the token, the URL (its query may be signed), header values or proxy URLs |

## Byte transparency — why GZip is gone

`GZipMiddleware` was removed from `app/main.py`. Compressing proxied bodies
would strip `Content-Length` from 206/HEAD answers (breaking `EXT-X-BYTERANGE`
seeking), waste CPU re-compressing already-compressed media, and — combined
with `Accept-Encoding` — invite exactly the content transformation a carrier
must not do. JSON endpoints are small; nobody misses it.

## Pooling, timeouts, workers

One pooled `ClientSession` per distinct proxy (plus one for direct traffic),
bounded per host (`MEDIA_PROXY_CONNECT_LIMIT_PER_HOST`, default 10) with an
LRU cap (`MEDIA_PROXY_POOL_MAX`, default 8) and idle sweeping. Connect and
read timeouts are separate; `total` is unbounded so a slow-but-alive body is
never cut off. **No semaphore is held while a body streams** — a stalled
upstream trips `sock_read` and never occupies a worker.

```bash
MEDIA_PROXY_CONNECT_TIMEOUT=5          # seconds, connect phase (incl. DNS)
MEDIA_PROXY_READ_TIMEOUT=15            # seconds, per-read (headers + each chunk)
MEDIA_PROXY_REDIRECT_LIMIT=5
MEDIA_PROXY_CHUNK_SIZE=65536
MEDIA_PROXY_POOL_MAX=8
MEDIA_PROXY_CONNECT_LIMIT=100
MEDIA_PROXY_CONNECT_LIMIT_PER_HOST=10
MEDIA_PROXY_SESSION_IDLE=120           # seconds before an idle session is closed
```

The legacy DASH retry ladder and 403-manifest-refresh heuristic were **not**
carried into the streaming path: `/api/proxy` is single-attempt with
deterministic status mapping, which is what the phase-2 contract specifies
(players must see 403 as 403). `/api/decrypt` keeps its original engine,
unchanged.

## Deployment notes

* **Access-log redaction.** uvicorn's access log prints the request path —
  which for `/api/proxy` and `/api/decrypt` *is* the token (upstream URL with
  signed queries, header values, keys). A `logging.Filter` installed at
  startup rewrites those paths to `<redacted>` (query strings included), on
  both `/api/...` (modern stacks) and the prefix-stripped `/proxy/...` form
  older Starlette versions leave in the scope. Method, status and client
  address survive.
* **CORS is off by default.** Players (Kodi, ffmpeg) are not browsers; a
  permissive policy would let any LAN web page read proxied responses. Set
  `CORS_ALLOW_ORIGINS` (comma-separated) only for a browser client you
  control; methods are limited to GET/HEAD/POST and credentials are never
  echoed.
* **Long request paths.** The service accepts request paths of well over 8 KiB
  (validated by the test suite at ~9.5 KiB). Under uvicorn+h11 the whole
  request line *plus* headers must stay under h11's 16 KiB event budget;
  installing `uvicorn[standard]` (httptools) removes that ceiling. If a web
  server sits in front, raise its limits, e.g. nginx:
  `large_client_header_buffers 8 32k;`
* **Streaming through a front server.** Disable response buffering and
  compression for `/api/proxy/`: `proxy_buffering off; gzip off;`
* **Proxy credentials.** The backend sends `scheme://host:port` only. If a
  URL ever embeds `user:pass@`, aiohttp honours it — but such URLs are never
  logged by this service.
* **Dependency range (migration).** The suite is verified end-to-end on
  fastapi 0.104 → 0.162, uvicorn 0.24 → 0.54, aiohttp 3.9.1 → 3.14.5,
  aiohttp-socks 0.8.4 → 0.12.0 (Python 3.12). Two modern-stack changes are
  absorbed in code, so no version has to be held back: (1) aiohttp-socks
  0.12+ raises its own exception types (no longer `OSError` subclasses) on
  proxy connect failures — `utils.socks_error_types()` keeps them mapped to
  502 in the stream proxy and to handled download errors in the decryptor;
  (2) aiohttp 3.14's larger client read buffers can starve best-effort task
  cancellation after a player disconnects (uvicorn's `send()` silently
  returns once disconnected, so the pump loses backpressure) —
  `/api/proxy` bodies are wrapped in `_disconnect_aware_body`, which watches
  for `http.disconnect` itself and closes the upstream within one chunk of
  the client going away. Keep aiohttp/aiohttp-socks bumps coupled
  (aiohttp ≥ 3.10 with aiohttp-socks ≥ 0.9).
* **Security posture (assumption, as decided).** Tokens are not signed; this
  rests on backend and proxy being reachable only by trusted clients (LAN).
  Do not expose either publicly. If that ever changes: add `sig` (HMAC) and
  `exp` keys inside the same payload, restrict targets to http/https and
  non-private ranges, keep the no-token-logging policy that is already in
  place.

## Layout

```
app/
  main.py                     app wiring, lifespan, no GZip (see above)
  api.py                      /api endpoints; /api/proxy is the streaming carrier
  services/stream_proxy.py    the engine: pool, redirects, Range/HEAD, status map
  services/decryptor.py       untouched DASH/decrypt engine (/api/decrypt)
  models/ services/ utils/    unchanged modules from phase 1
tests/
  harness.py                  fake CDNs, raw-TCP CDN, dummy forward proxy
  acceptance_test.py          the handoff's acceptance table (1–11) + extras
```

## Tests

```bash
python tests/acceptance_test.py    # from the project root; exit 0 iff all pass
```

Covers the handoff's table: byte-identical 2 MB segments, derived content
types, token-only headers, bounded/open-ended ranges, HEAD, 416, cross-host
redirects with host-scoped `Cookie`/`Authorization`, status passthrough,
502/504, 700-char signed queries inside ≥8 KiB paths, forward-proxy transit,
client-disconnect propagation, DASH regression (suffix, legacy `ua`,
`/api/decrypt`), the full extension table, `Cache-Control`/`.key` no-store,
chunked passthrough, no-transformation, and redirect loops.
