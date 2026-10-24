#!/usr/bin/env python3
"""Acceptance tests for the media proxy -- HLS phase 2 (proxied HLS).

Implements the handoff's acceptance table (tests 1-11) plus targeted extras
(content-type table, 416, chunked upstream, no-transformation, redirect
extras, SOCKS degradation).  The proxy under test runs in-process (uvicorn)
against controllable fake upstreams; every request the fake CDNs and the
dummy forward proxy receive is logged and asserted on.

Run from the project root:  python tests/acceptance_test.py
Exit code 0 iff every test passed.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

# Fast-fail timeouts for the 502/504 tests (production defaults are 5s/15s).
os.environ.setdefault("MEDIA_PROXY_CONNECT_TIMEOUT", "2")
os.environ.setdefault("MEDIA_PROXY_READ_TIMEOUT", "2.5")
os.environ.setdefault("MEDIA_PROXY_SESSION_IDLE", "60")
os.environ.setdefault("MEDIA_PROXY_CHUNK_SIZE", "16384")

import aiohttp

from app.main import app as proxy_app
from app.services.decryptor import DEFAULT_USER_AGENT
from harness import (
    CHUNKED_BODY,
    FREE_BOX,
    KEY16,
    SEG,
    SEG_SIZE,
    DummyProxy,
    FakeCDN,
    RawCDN,
    Socks5Server,
    make_token,
    start_proxy_server,
)

CLIENT_UA = "Kodi/21.0 (test client; must never reach upstream)"


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------


async def t01_basic_passthrough_and_content_type(ctx):
    """1: 2 MB .m4s, no headers -> 200, same bytes, video/mp4 (upstream octet-stream)."""
    r = await ctx.client.get(ctx.purl(make_token(f"{ctx.cdn1.origin}/cdn/octet/seg.m4s")))
    assert r.status == 200, f"status {r.status}"
    body = await r.read()
    assert body == SEG, f"body differs ({len(body)} bytes)"
    assert r.headers.get("Content-Type") == "video/mp4", r.headers.get("Content-Type")
    assert r.headers.get("Content-Length") == str(SEG_SIZE)
    assert r.headers.get("Accept-Ranges") == "bytes"
    rec = ctx.cdn1.last()
    assert rec["headers"].get("User-Agent") == DEFAULT_USER_AGENT, rec["headers"].get("User-Agent")
    assert CLIENT_UA not in json.dumps(rec["headers"]), "client UA leaked upstream"
    assert "Referer" not in rec["headers"] and "Range" not in rec["headers"]
    return True, "200, 2 MB byte-identical, CT derived, neutral UA upstream"


async def t02_token_headers_only(ctx):
    """2: token headers are exactly what the CDN sees -- none of the client's."""
    hdrs = {"Referer": "https://r.example/", "User-Agent": "UA/1"}
    r = await ctx.client.get(
        ctx.purl(make_token(f"{ctx.cdn1.origin}/cdn/octet/seg.m4s", headers=hdrs))
    )
    assert r.status == 200, r.status
    await r.read()
    h = ctx.cdn1.last()["headers"]
    assert h.get("Referer") == "https://r.example/", h.get("Referer")
    assert h.get("User-Agent") == "UA/1", h.get("User-Agent")
    assert CLIENT_UA not in json.dumps(h), "client UA leaked upstream"
    assert "Range" not in h and "Cookie" not in h and "Accept" not in h, sorted(h)
    return True, "CDN saw exactly the token's Referer+UA, nothing of the client's"


async def t03_range_bounded(ctx):
    """3: Range bytes=100-199 -> 206 + Content-Range + 100 bytes; CDN saw the Range."""
    r = await ctx.client.get(
        ctx.purl(make_token(f"{ctx.cdn1.origin}/cdn/octet/seg.m4s")),
        headers={"Range": "bytes=100-199"},
    )
    assert r.status == 206, f"status {r.status}"
    assert r.headers["Content-Range"] == f"bytes 100-199/{SEG_SIZE}", r.headers.get("Content-Range")
    assert r.headers.get("Content-Length") == "100"
    body = await r.read()
    assert body == SEG[100:200], f"{len(body)} bytes"
    assert ctx.cdn1.last()["headers"].get("Range") == "bytes=100-199"
    return True, "206, Content-Range, exactly 100 bytes, Range reached the CDN"


async def t04_range_open_and_head(ctx):
    """4: open-ended Range plus HEAD (headers only, no body)."""
    r = await ctx.client.get(
        ctx.purl(make_token(f"{ctx.cdn1.origin}/cdn/octet/seg.m4s")),
        headers={"Range": "bytes=100-"},
    )
    assert r.status == 206, r.status
    assert r.headers["Content-Range"] == f"bytes 100-{SEG_SIZE - 1}/{SEG_SIZE}"
    assert await r.read() == SEG[100:]

    h = await ctx.client.head(ctx.purl(make_token(f"{ctx.cdn1.origin}/cdn/octet/seg.m4s")))
    assert h.status == 200, h.status
    assert h.headers.get("Content-Length") == str(SEG_SIZE), h.headers.get("Content-Length")
    assert h.headers.get("Content-Type") == "video/mp4"
    assert h.headers.get("Accept-Ranges") == "bytes"
    assert await h.read() == b""

    h2 = await ctx.client.head(
        ctx.purl(make_token(f"{ctx.cdn1.origin}/cdn/octet/seg.m4s")),
        headers={"Range": "bytes=100-"},
    )
    assert h2.status == 206, h2.status
    assert h2.headers["Content-Range"] == f"bytes 100-{SEG_SIZE - 1}/{SEG_SIZE}"
    assert h2.headers.get("Content-Length") == str(SEG_SIZE - 100)
    assert await h2.read() == b""
    return True, "open Range tail ok; HEAD answers headers only (200 and 206)"


async def t04x_416_passthrough(ctx):
    """4x: unsatisfiable Range -> upstream 416 passed through with Content-Range."""
    r = await ctx.client.get(
        ctx.purl(make_token(f"{ctx.cdn1.origin}/cdn/octet/seg.m4s")),
        headers={"Range": "bytes=999999999-"},
    )
    assert r.status == 416, r.status
    assert r.headers.get("Content-Range") == f"bytes */{SEG_SIZE}", r.headers.get("Content-Range")
    await r.read()
    return True, "416 passed through, Content-Range intact"


async def t05_redirect_cross_host(ctx):
    """5: 302 to a second host; final body returned; Referer/UA travel, Cookie/Authz do not."""
    hdrs = {
        "Referer": "https://r.example/",
        "User-Agent": "UA/1",
        "Cookie": "sid=secret-value",
        "Authorization": "Bearer test-token",
    }
    r = await ctx.client.get(ctx.purl(make_token(f"{ctx.cdn1.origin}/redirect302", headers=hdrs)))
    assert r.status == 200, r.status
    assert await r.read() == SEG
    rec1 = ctx.cdn1.last_for("/redirect302")
    assert rec1["headers"].get("Cookie") == "sid=secret-value"  # original host: yes
    assert rec1["headers"].get("Authorization") == "Bearer test-token"
    rec2 = ctx.cdn2.last()
    assert rec2["path"] == "/cdn/octet/seg.m4s", rec2["path"]
    assert rec2["raw_query"] == "sig=%2Fa%2Bb&h=2", rec2["raw_query"]
    h2 = rec2["headers"]
    assert h2.get("Referer") == "https://r.example/", "Referer lost on redirect"
    assert h2.get("User-Agent") == "UA/1", "User-Agent lost on redirect"
    assert "Cookie" not in h2, "Cookie leaked to the second host"
    assert "Authorization" not in h2, "Authorization leaked to the second host"
    return True, "body served from host2; Referer/UA kept on hop; Cookie/Authz host-scoped"


async def t06_status_passthrough(ctx):
    """6: upstream 403/404/410/500 come back as themselves, not 502."""
    for code in (403, 404, 410, 500):
        r = await ctx.client.get(ctx.purl(make_token(f"{ctx.cdn1.origin}/status/{code}")))
        assert r.status == code, (code, r.status)
        assert r.headers.get("Content-Type") == "application/json"
        data = await r.json()
        assert f"upstream returned {code}" in data["error"], data
    return True, "403/404/410/500 passed through with a JSON reason"


async def t06x_head_error(ctx):
    """6x: HEAD of an upstream error -> status only, no body."""
    h = await ctx.client.head(ctx.purl(make_token(f"{ctx.cdn1.origin}/status/403")))
    assert h.status == 403, h.status
    assert await h.read() == b""
    return True, "HEAD of an error: status only"


async def t07_unreachable_and_timeout(ctx):
    """7: connect refused -> 502; stalled upstream -> 504 (nothing else maps to these)."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    dead_port = s.getsockname()[1]
    s.close()

    r = await ctx.client.get(ctx.purl(make_token(f"http://127.0.0.1:{dead_port}/seg.ts")))
    assert r.status == 502, r.status
    data = await r.json()
    assert "upstream connection failed" in data["error"], data

    t0 = time.monotonic()
    r = await ctx.client.get(ctx.purl(make_token(f"{ctx.cdn1.origin}/slow")))
    elapsed = time.monotonic() - t0
    assert r.status == 504, r.status
    data = await r.json()
    assert "upstream timeout" in data["error"], data
    assert 2.0 <= elapsed <= 8.0, elapsed
    return True, f"refused -> 502; stall -> 504 in {elapsed:.2f}s"


async def t08_long_url_8k(ctx):
    """8: 700-char signed query inside a >=8 KB proxy path, forwarded byte-exact."""
    sig = "x%2Fy%2B" + "a" * 693  # 700 chars incl. %2F / %2B traps
    pad = 6400
    while True:
        target = f"/big?sig={sig}&pad=" + "b" * pad
        tok = make_token(f"{ctx.raw.origin}{target}")
        if len("/api/proxy/") + len(tok) >= 8192:
            break
        pad += 256
    url = ctx.purl(tok)
    path_len = len(url) - len(f"http://127.0.0.1:{ctx.proxy_port}")
    assert path_len >= 8192, path_len

    r = await ctx.client.get(url)
    assert r.status == 200, f"status {r.status} (path {path_len} bytes)"
    assert await r.read() == b"raw-ok"
    got_target = ctx.raw.requests[-1].split(" ", 2)[1]
    assert got_target == target, f"query mutated:\n  want {target[:96]}\n  got  {got_target[:96]}"
    return True, f"{path_len}-byte request path accepted, {len(sig) + len('pad=') + pad}-char query byte-exact"


async def t09_via_forward_proxy(ctx):
    """9: token proxy -> the request transits that proxy and arrives intact."""
    tok = make_token(f"{ctx.cdn1.origin}/cdn/octet/seg.m4s", proxy=ctx.dp.origin)
    r = await ctx.client.get(ctx.purl(tok))
    assert r.status == 200, r.status
    assert await r.read() == SEG
    assert r.headers.get("Content-Type") == "video/mp4"
    entry = ctx.dp.requests[-1]
    assert entry["method"] == "GET"
    assert entry["target"] == f"{ctx.cdn1.origin}/cdn/octet/seg.m4s", entry  # absolute-form at the proxy
    assert ctx.cdn1.last()["headers"].get("User-Agent") == DEFAULT_USER_AGENT
    return True, "request arrived through the forward proxy (absolute-form), headers intact"


async def t09x_socks_tunnel_or_degrade(ctx):
    """9x: SOCKS tokens, both worlds.

    * aiohttp-socks absent  -> socks5:// token degrades to a clear 502 naming
      the missing library.
    * aiohttp-socks present -> the tunnel runs for real: a segment streams
      through the harness SOCKS5 server to CDN1 with the token's headers, and
      a dead SOCKS endpoint maps to a 502 connection failure.
    """
    try:
        import aiohttp_socks  # noqa: F401
    except ImportError:
        r = await ctx.client.get(
            ctx.purl(
                make_token(
                    f"{ctx.cdn1.origin}/cdn/octet/seg.m4s", proxy="socks5://127.0.0.1:9"
                )
            )
        )
        assert r.status == 502, r.status
        data = await r.json()
        assert "aiohttp-socks" in data["error"], data
        return True, "SOCKS without the library degrades to a clear 502"

    # library installed (the production default): ride the tunnel end-to-end
    tunnels_before = len(ctx.socks.connects)
    r = await ctx.client.get(
        ctx.purl(
            make_token(
                f"{ctx.cdn1.origin}/cdn/octet/seg.m4s",
                proxy=ctx.socks.proxy_url,
                headers={"User-Agent": "tokenUA/9", "Referer": "https://cdn.example/hls"},
            )
        )
    )
    assert r.status == 200, f"{r.status}: {await r.text()}"
    assert await r.read() == SEG
    assert len(ctx.socks.connects) == tunnels_before + 1, ctx.socks.connects
    dst_host, dst_port = ctx.socks.connects[-1]
    assert dst_host == "127.0.0.1" and dst_port == ctx.cdn1.port, (dst_host, dst_port)
    rec = ctx.cdn1.last()
    assert rec["headers"].get("User-Agent") == "tokenUA/9", rec["headers"]
    assert rec["headers"].get("Referer") == "https://cdn.example/hls", rec["headers"]
    assert r.headers.get("Content-Type") == "video/mp4", r.headers.get("Content-Type")

    # a dead SOCKS endpoint is a 502 connection failure, not a hang
    r = await ctx.client.get(
        ctx.purl(
            make_token(
                f"{ctx.cdn1.origin}/cdn/octet/seg.m4s", proxy="socks5://127.0.0.1:9"
            )
        )
    )
    assert r.status == 502, r.status
    data = await r.json()
    assert "upstream connection failed" in data["error"], data
    return True, (
        f"tunnel end-to-end (CONNECT {dst_host}:{dst_port}, token headers intact, "
        "octet-stream upgraded to video/mp4); dead SOCKS -> 502"
    )


async def t10_client_disconnect_closes_upstream(ctx):
    """10: client aborts a 100 MB download -> upstream closed promptly."""
    r = await ctx.client.get(ctx.purl(make_token(f"{ctx.raw.origin}/stream")))
    assert r.status == 200, r.status
    received = 0
    async for chunk in r.content.iter_chunked(16384):
        received += len(chunk)
        if received >= 3 * 16384:
            break
    r.close()  # player goes away mid-stream
    assert await asyncio.wait_for(ctx.raw.stream_event.wait(), 5.0), "upstream not closed within 5s"
    assert received < 5_000_000, received
    assert ctx.raw.stream_sent < 50_000_000, ctx.raw.stream_sent
    return True, (
        f"upstream closed promptly (proxy had read {received} B, CDN had sent {ctx.raw.stream_sent} B)"
    )


async def t11_dash_regression(ctx):
    """11: DASH template suffix, legacy ua param, and /api/decrypt all unchanged."""
    tok = make_token(f"{ctx.cdn1.origin}/DASH/dash")
    r = await ctx.client.get(ctx.purl(tok, "video/123456.m4s"))
    assert r.status == 200, r.status
    assert await r.read() == SEG
    assert ctx.cdn1.last()["path"] == "/DASH/dash/video/123456.m4s", ctx.cdn1.last()["path"]

    tok = make_token(f"{ctx.cdn1.origin}/DASH/dash/")  # trailing slash composes identically
    r = await ctx.client.get(ctx.purl(tok, "video/123456.m4s"))
    assert r.status == 200, r.status
    assert ctx.cdn1.last()["path"] == "/DASH/dash/video/123456.m4s"

    r = await ctx.client.get(
        ctx.purl(make_token(f"{ctx.cdn1.origin}/cdn/octet/seg.m4s", ua="LegacyUA/9"))
    )
    assert r.status == 200, r.status
    assert ctx.cdn1.last()["headers"].get("User-Agent") == "LegacyUA/9"

    tok = make_token(f"{ctx.cdn1.origin}/mp4free/seg.m4s")
    r = await ctx.client.get(f"http://127.0.0.1:{ctx.proxy_port}/api/decrypt/{tok}")
    assert r.status == 200, r.status
    assert r.headers.get("X-Content-Decrypted") == "false", dict(r.headers)
    assert await r.read() == FREE_BOX
    return True, "suffix composition, legacy ua, /api/decrypt: all as before"


async def t12_content_type_table(ctx):
    """Content-type derivation table plus generic/missing/specific variants."""
    cases = [
        (".ts", "video/mp2t"),
        (".m4s", "video/mp4"),
        (".mp4", "video/mp4"),
        (".cmfv", "video/mp4"),
        (".m4a", "audio/mp4"),
        (".cmfa", "audio/mp4"),
        (".aac", "audio/aac"),
        (".vtt", "text/vtt"),
        (".webvtt", "text/vtt"),
        (".key", "application/octet-stream"),
    ]
    for ext, expected in cases:
        r = await ctx.client.get(ctx.purl(make_token(f"{ctx.cdn1.origin}/cdn/octet/file{ext}")))
        assert r.status == 200, (ext, r.status)
        got = r.headers.get("Content-Type")
        assert got == expected, (ext, got, expected)
        await r.read()

    for mode, ext, expected in [
        ("textplain", ".ts", "video/mp2t"),
        ("binoct", ".m4s", "video/mp4"),
        ("none", ".vtt", "text/vtt"),
    ]:
        r = await ctx.client.get(ctx.purl(make_token(f"{ctx.cdn1.origin}/cdn/{mode}/file{ext}")))
        got = r.headers.get("Content-Type")
        assert got == expected, (mode, ext, got, expected)
        await r.read()

    r = await ctx.client.get(ctx.purl(make_token(f"{ctx.cdn1.origin}/cdn/typed/file.ts")))
    assert r.headers.get("Content-Type") == "video/mp2t"  # specific types are never overridden
    await r.read()
    r = await ctx.client.get(ctx.purl(make_token(f"{ctx.cdn1.origin}/cdn/typed2/file.ts")))
    assert r.headers.get("Content-Type") == "video/mp4"
    await r.read()
    return True, "10 extensions + generic/none/specific variants all correct"


async def t13_cache_control_and_key_nostore(ctx):
    """Cache-Control/ETag/Last-Modified passthrough; .key forced to no-store."""
    r = await ctx.client.get(ctx.purl(make_token(f"{ctx.cdn1.origin}/cdn/cachey/seg.m4s")))
    assert r.headers.get("Cache-Control") == "public, max-age=99", r.headers.get("Cache-Control")
    assert r.headers.get("ETag") == '"etag-x1"'
    assert r.headers.get("Last-Modified") == "Wed, 21 Oct 2026 07:28:00 GMT"
    await r.read()

    r = await ctx.client.get(ctx.purl(make_token(f"{ctx.cdn1.origin}/cdn/keymode/secret.key")))
    assert r.status == 200, r.status
    assert r.headers.get("Cache-Control") == "no-store", r.headers.get("Cache-Control")
    assert r.headers.get("Content-Type") == "application/octet-stream"
    assert await r.read() == KEY16
    return True, "Cache-Control/ETag/Last-Modified passthrough; .key -> no-store + octet-stream"


async def t14_chunked_and_no_transformation(ctx):
    """Chunked upstream streams through; the proxy never transforms a body."""
    r = await ctx.client.get(ctx.purl(make_token(f"{ctx.cdn1.origin}/cdn/chunked/file.m4s")))
    assert r.status == 200, r.status
    assert "Content-Length" not in r.headers, dict(r.headers)
    assert await r.read() == CHUNKED_BODY

    r = await ctx.client.get(
        ctx.purl(make_token(f"{ctx.cdn1.origin}/cdn/octet/seg.m4s")),
        headers={"Accept-Encoding": "gzip"},
    )
    assert r.status == 200, r.status
    assert "Content-Encoding" not in r.headers, "body was transformed"
    assert await r.read() == SEG
    return True, "chunked passthrough ok; Accept-Encoding: gzip gets identity bytes"


async def t15_redirect_extras(ctx):
    """Relative Location resolution; a redirect loop ends in 502 after <=5 hops."""
    r = await ctx.client.get(ctx.purl(make_token(f"{ctx.cdn1.origin}/redirect-rel")))
    assert r.status == 200, r.status
    assert await r.read() == SEG
    assert ctx.cdn1.last()["path"] == "/cdn/octet/seg.m4s", ctx.cdn1.last()["path"]

    r = await ctx.client.get(ctx.purl(make_token(f"{ctx.cdn1.origin}/redirect-loop")))
    assert r.status == 502, r.status
    data = await r.json()
    assert "too many redirects" in data["error"], data
    hops = sum(1 for e in ctx.cdn1.requests if e["path"] == "/redirect-loop")
    assert hops <= 6, hops
    return True, f"relative redirect resolved; loop stopped after {hops} CDN hits with 502"


TESTS = [
    t01_basic_passthrough_and_content_type,
    t02_token_headers_only,
    t03_range_bounded,
    t04_range_open_and_head,
    t04x_416_passthrough,
    t05_redirect_cross_host,
    t06_status_passthrough,
    t06x_head_error,
    t07_unreachable_and_timeout,
    t08_long_url_8k,
    t09_via_forward_proxy,
    t09x_socks_tunnel_or_degrade,
    t10_client_disconnect_closes_upstream,
    t11_dash_regression,
    t12_content_type_table,
    t13_cache_control_and_key_nostore,
    t14_chunked_and_no_transformation,
    t15_redirect_extras,
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def main():
    cdn2 = FakeCDN("cdn2")  # second host, answers to the name "localhost"
    await cdn2.start()
    cdn1 = FakeCDN("cdn1")
    cdn1.redirect_target = f"http://localhost:{cdn2.port}/cdn/octet/seg.m4s?sig=%2Fa%2Bb&h=2"
    await cdn1.start()

    raw = RawCDN()
    await raw.start()
    dp = DummyProxy()
    await dp.start()
    socks = Socks5Server()
    await socks.start()

    server, server_task, port = await start_proxy_server(proxy_app)
    client = aiohttp.ClientSession(headers={"User-Agent": CLIENT_UA})

    ctx = SimpleNamespace(
        cdn1=cdn1, cdn2=cdn2, raw=raw, dp=dp, socks=socks, client=client, proxy_port=port
    )
    ctx.purl = lambda tok, suffix="": (
        f"http://127.0.0.1:{port}/api/proxy/{tok}" + (f"/{suffix}" if suffix else "")
    )

    width = max(len(t.__name__) for t in TESTS) + 2
    failures = 0
    print(f"\n{'test':{width}}result  detail")
    print("-" * 110)
    for t in TESTS:
        try:
            ok, detail = await t(ctx)
        except Exception as exc:  # noqa: BLE001 - test runner
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        if not ok:
            failures += 1
        print(f"{t.__name__:{width}}{'PASS' if ok else 'FAIL'}    {detail}")

    await client.close()
    server.should_exit = True
    await server_task
    await cdn1.stop()
    await cdn2.stop()
    await raw.stop()
    await dp.stop()
    await socks.stop()

    total = len(TESTS)
    print("-" * 110)
    print(f"MEDIA PROXY ACCEPTANCE: {total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))