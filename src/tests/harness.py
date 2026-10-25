"""Acceptance-test harness: controllable upstream servers for the media proxy.

Everything runs in one event loop on 127.0.0.1 with ephemeral ports:

* ``FakeCDN``   -- aiohttp.web "CDN".  Two instances: the primary one answers
                   to the name ``127.0.0.1``, the second one to ``localhost``
                   (a different host name, same loopback) so cross-host
                   redirect semantics can be tested.  Every request it
                   receives is logged (method, path, raw query, headers).
* ``RawCDN``     -- hand-rolled asyncio TCP server for the tests that need
                   wire-level control: >8 KB request targets and immediate
                   FIN detection when the proxy stops reading (client
                   disconnect during a 100 MB stream).
* ``DummyProxy`` -- minimal HTTP forward proxy (absolute-form requests are
                   rewritten to origin-form and relayed), so "the request
                   arrived through that proxy" is observable.

``make_token`` is byte-identical to the backend's builder (handoff section 3).
"""

from __future__ import annotations

import asyncio
import base64
import json
import socket
import time
from urllib.parse import urlencode, urlsplit

import uvicorn
from aiohttp import web

SEG_SIZE = 2 * 1024 * 1024
SEG = b"0123456789abcdef" * (SEG_SIZE // 16)
KEY16 = bytes(range(16))
FREE_BOX = b"\x00\x00\x00\x08free"  # minimal well-formed MP4 'free' box
CHUNKED_BODY = b"chunk-datum-" * 4096  # 48 KiB, streamed in 16 KiB writes
LAST_MODIFIED = "Wed, 21 Oct 2026 07:28:00 GMT"


# ---------------------------------------------------------------------------
# Token builder (handoff section 3 -- byte-identical to the backend)
# ---------------------------------------------------------------------------


def make_token(url, proxy=None, headers=None, ua=None):
    params = {"url": url}
    if proxy:
        params["proxy"] = proxy
    if headers:
        params["headers"] = (
            base64.urlsafe_b64encode(json.dumps(headers).encode()).decode().rstrip("=")
        )
    if ua:
        params["ua"] = ua
    return base64.urlsafe_b64encode(urlencode(params).encode()).decode().rstrip("=")


def apply_range(spec, total):
    """Fake-CDN Range support.

    Returns ``(start, end)`` inclusive, ``None`` when the range is absent or
    unsupported (serve full body), or the string ``"416"`` when unsatisfiable.
    """
    if not spec or not spec.startswith("bytes=") or "," in spec:
        return None
    body = spec[6:].strip()
    first, _, last = body.partition("-")
    try:
        if first == "":
            n = int(last)
            if n <= 0:
                return "416"
            start, end = max(0, total - n), total - 1
        else:
            start = int(first)
            end = int(last) if last else total - 1
            end = min(end, total - 1)
    except ValueError:
        return None
    if start < 0 or start > end or start >= total:
        return "416"
    return start, end


# ---------------------------------------------------------------------------
# FakeCDN (aiohttp.web)
# ---------------------------------------------------------------------------


class FakeCDN:
    MODE_HEADERS = {
        "octet": {"Content-Type": "application/octet-stream"},
        "none": {},  # no Content-Type at all
        "textplain": {"Content-Type": "text/plain"},
        "binoct": {"Content-Type": "binary/octet-stream"},
        "typed": {"Content-Type": "video/mp2t"},
        "typed2": {"Content-Type": "video/mp4"},
        "cachey": {
            "Content-Type": "application/octet-stream",
            "Cache-Control": "public, max-age=99",
            "ETag": '"etag-x1"',
            "Last-Modified": LAST_MODIFIED,
        },
        "keymode": {
            "Content-Type": "application/octet-stream",
            "Cache-Control": "public, max-age=3600",
        },
    }

    def __init__(self, name):
        self.name = name
        self.requests = []
        self.runner = None
        self.port = None
        self.redirect_target = ""  # set once both CDNs are up

    # -- logging -----------------------------------------------------------

    def _record(self, request: web.Request) -> None:
        self.requests.append(
            {
                "ts": time.monotonic(),
                "method": request.method,
                "path": request.path,
                "raw_query": request.rel_url.raw_query_string,
                "headers": {k: v for k, v in request.headers.items()},
            }
        )

    def last(self):
        return self.requests[-1]

    def last_for(self, path_prefix):
        for entry in reversed(self.requests):
            if entry["path"].startswith(path_prefix):
                return entry
        raise AssertionError(f"no request logged for {path_prefix!r}")

    # -- content -----------------------------------------------------------

    def _payload(self, filename):
        if filename.endswith(".key"):
            return KEY16
        if filename == "seg.m4s":
            return SEG
        return SEG[:65536]

    @staticmethod
    async def _head(request, status, headers, body):
        resp = web.StreamResponse(status=status, headers=headers)
        resp.content_length = len(body)
        await resp.prepare(request)
        await resp.write_eof()
        return resp

    async def handle_cdn(self, request):
        self._record(request)
        mode = request.match_info["mode"]
        filename = request.match_info["filename"]
        payload = self._payload(filename)

        if mode == "chunked":
            resp = web.StreamResponse(headers={"Content-Type": "application/octet-stream"})
            await resp.prepare(request)
            for i in range(0, len(CHUNKED_BODY), 16384):
                await resp.write(CHUNKED_BODY[i : i + 16384])
                await resp.drain()
            await resp.write_eof()
            return resp

        headers = dict(self.MODE_HEADERS.get(mode, {"Content-Type": "application/octet-stream"}))

        rng = apply_range(request.headers.get("Range"), len(payload))
        if rng == "416":
            headers["Content-Range"] = f"bytes */{len(payload)}"
            if request.method == "HEAD":
                return await self._head(request, 416, headers, b"")
            return web.Response(status=416, headers=headers)
        if rng is not None:
            start, end = rng
            body = payload[start : end + 1]
            headers["Content-Range"] = f"bytes {start}-{end}/{len(payload)}"
            status = 206
        else:
            body = payload
            status = 200

        if request.method == "HEAD":
            return await self._head(request, status, headers, body)
        return web.Response(status=status, body=body, headers=headers)

    async def handle_status(self, request):
        self._record(request)
        code = int(request.match_info["code"])
        return web.Response(status=code, text=f"upstream says {code}")

    async def handle_slow(self, request):
        self._record(request)
        await asyncio.sleep(10)
        return web.Response(text="finally")

    async def handle_redirect302(self, request):
        self._record(request)
        # Raw Location header: HTTPFound() would normalize it through
        # yarl (URL(location) decodes %2F / re-quotes +) -- we want the
        # bytes on the wire to be exactly what this "CDN" dictated.
        resp = web.Response(status=302)
        resp.headers["Location"] = self.redirect_target
        return resp

    async def handle_redirect_rel(self, request):
        self._record(request)
        resp = web.Response(status=302)
        resp.headers["Location"] = "/cdn/octet/seg.m4s"
        return resp

    async def handle_redirect_loop(self, request):
        self._record(request)
        resp = web.Response(status=302)
        resp.headers["Location"] = f"{request.scheme}://{request.host}/redirect-loop"
        return resp

    async def handle_dash(self, request):
        self._record(request)
        return web.Response(
            status=200, body=SEG, headers={"Content-Type": "application/octet-stream"}
        )

    async def handle_mp4free(self, request):
        self._record(request)
        return web.Response(status=200, body=FREE_BOX, headers={"Content-Type": "video/mp4"})

    # -- lifecycle ---------------------------------------------------------

    def make_app(self):
        app = web.Application()
        r = app.router
        r.add_route("*", "/cdn/{mode}/{filename}", self.handle_cdn)
        r.add_route("*", "/status/{code}", self.handle_status)
        r.add_route("*", "/slow", self.handle_slow)
        r.add_route("*", "/redirect302", self.handle_redirect302)
        r.add_route("*", "/redirect-rel", self.handle_redirect_rel)
        r.add_route("*", "/redirect-loop", self.handle_redirect_loop)
        r.add_route("*", "/dash/{suffix:.+}", self.handle_dash)
        r.add_route("*", "/DASH/dash/{suffix:.+}", self.handle_dash)
        r.add_route("*", "/mp4free/{name}", self.handle_mp4free)
        return app

    async def start(self):
        self.runner = web.AppRunner(self.make_app(), access_log=None)
        await self.runner.setup()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        site = web.TCPSite(self.runner, "127.0.0.1", self.port)
        await site.start()
        return self.port

    @property
    def origin(self):
        return f"http://127.0.0.1:{self.port}"

    async def stop(self):
        if self.runner:
            await self.runner.cleanup()


# ---------------------------------------------------------------------------
# RawCDN (wire-level control)
# ---------------------------------------------------------------------------


class RawCDN:
    """Hand-rolled TCP server.

    /stream -> declares Connection: close with no length framing (read-to-EOF
               body), writes up to 100 MB, watches for the peer FIN and flags
               it immediately (this is how "client disconnect closes the
               upstream promptly" is observed).
    other    -> canned "raw-ok" response, Connection: close.
    """

    def __init__(self):
        self.requests = []  # first lines, e.g. "GET /big?sig=... HTTP/1.1"
        self.stream_event = None
        self.stream_sent = 0
        self._server = None

    async def _handle(self, reader, writer):
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except (asyncio.IncompleteReadError, ConnectionResetError, OSError):
            return
        text = head.decode("latin-1")
        first_line = text.split("\r\n", 1)[0]
        self.requests.append(first_line)
        target = first_line.split(" ", 2)[1]

        if target.startswith("/stream"):
            event = asyncio.Event()
            self.stream_event = event
            eof = asyncio.Event()

            async def watch_fin():
                try:
                    while True:
                        chunk = await reader.read(64)
                        if not chunk:
                            eof.set()
                            return
                except Exception:
                    eof.set()

            watcher = asyncio.create_task(watch_fin())
            sent = 0
            try:
                writer.write(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: video/mp2t\r\n"
                    b"Connection: close\r\n\r\n"
                )
                while sent < 100_000_000 and not eof.is_set():
                    writer.write(b"\xab" * 65536)
                    await writer.drain()
                    sent += 65536
            except Exception:
                pass
            finally:
                self.stream_sent = sent
                event.set()
                watcher.cancel()
                writer.close()
            return

        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/octet-stream\r\n"
            b"Content-Length: 6\r\n"
            b"Connection: close\r\n\r\nraw-ok"
        )
        await writer.drain()
        writer.close()

    async def start(self):
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self._server.sockets[0].getsockname()[1]

    @property
    def origin(self):
        host, port = self._server.sockets[0].getsockname()[:2]
        return f"http://{host}:{port}"

    async def stop(self):
        if self._server:
            self._server.close()


# ---------------------------------------------------------------------------
# Byte-pump helpers (shared by DummyProxy and Socks5Server)
# ---------------------------------------------------------------------------


async def _pump(src, dst):
    """Copy one direction until EOF or error.  Never raises."""
    try:
        while True:
            chunk = await src.read(65536)
            if not chunk:
                break
            dst.write(chunk)
            await dst.drain()
    except Exception:
        pass


async def _bridge(reader, writer, up_reader, up_writer):
    """Pump both directions; finish as soon as either direction ends.

    When one pump finishes, the sibling task is cancelled and the caller
    closes both transports.  Without this a handler can linger blocked on an
    idle upstream keep-alive read long after the client is gone (observed
    with the aiohttp 3.9 client pool), leaving pending tasks at shutdown.
    """
    t1 = asyncio.create_task(_pump(reader, up_writer))
    t2 = asyncio.create_task(_pump(up_reader, writer))
    _done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


# ---------------------------------------------------------------------------
# DummyProxy (HTTP forward proxy)
# ---------------------------------------------------------------------------


class DummyProxy:
    def __init__(self):
        self.requests = []  # {"method", "target", "headers"}
        self._server = None

    async def _handle(self, reader, writer):
        up_writer = None
        try:
            try:
                head = await reader.readuntil(b"\r\n\r\n")
            except (asyncio.IncompleteReadError, ConnectionResetError, OSError):
                return
            text = head.decode("latin-1")
            lines = text.split("\r\n")
            method, target, _ = lines[0].split(" ", 2)
            headers = {}
            for line in lines[1:]:
                if ": " in line:
                    key, value = line.split(": ", 1)
                    headers[key.strip().lower()] = value.strip()
            self.requests.append({"method": method, "target": target, "headers": headers})

            u = urlsplit(target)
            if u.scheme != "http":
                writer.write(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
                await writer.drain()
                return
            up_reader, up_writer = await asyncio.open_connection(u.hostname, u.port)
            origin = (u.path or "/") + (f"?{u.query}" if u.query else "")
            up_writer.write(
                text.replace(lines[0], f"{method} {origin} HTTP/1.1", 1).encode("latin-1")
            )
            await up_writer.drain()

            await _bridge(reader, writer, up_reader, up_writer)
        except Exception:
            pass
        finally:
            if up_writer is not None:
                up_writer.close()
            writer.close()

    async def start(self):
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self._server.sockets[0].getsockname()[1]

    @property
    def origin(self):
        return f"http://127.0.0.1:{self._server.sockets[0].getsockname()[1]}"

    async def stop(self):
        if self._server:
            self._server.close()


# ---------------------------------------------------------------------------
# Socks5Server (minimal SOCKS5 proxy, RFC 1928 subset)
#
# Speaks just enough SOCKS5 for the client half of aiohttp-socks: NO AUTH
# method negotiation and CONNECT for IPv4/domain targets, then a blind byte
# tunnel via _bridge.  Lets the acceptance suite exercise socks5:// tokens
# end-to-end; ``connects`` records every CONNECT the proxy engine asked for.
# ---------------------------------------------------------------------------


class Socks5Server:
    def __init__(self):
        self._server = None
        self.connects = []  # [(dst_host, dst_port)]

    async def _handle(self, reader, writer):
        up_writer = None
        try:
            # method negotiation: VER NMETHODS METHODS
            head = await reader.readexactly(2)
            if head[0] != 0x05:
                return
            methods = await reader.readexactly(head[1])
            if 0x00 not in methods:
                writer.write(b"\x05\xff")  # NO ACCEPTABLE METHODS
                await writer.drain()
                return
            writer.write(b"\x05\x00")  # select NO AUTHENTICATION
            await writer.drain()

            # request: VER CMD RSV ATYP DST.ADDR DST.PORT
            req = await reader.readexactly(4)
            if req[1] != 0x01:  # CONNECT only
                return
            atyp = req[3]
            if atyp == 0x01:  # IPv4
                host = ".".join(str(b) for b in await reader.readexactly(4))
            elif atyp == 0x03:  # domain name
                length = (await reader.readexactly(1))[0]
                host = (await reader.readexactly(length)).decode("ascii")
            else:  # IPv6: unsupported by this stub
                return
            port = int.from_bytes(await reader.readexactly(2), "big")
            self.connects.append((host, port))

            up_reader, up_writer = await asyncio.open_connection(host, port)
            # success reply: VER REP RSV ATYP BND.ADDR BND.PORT (zero BND is legal)
            writer.write(b"\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00")
            await writer.drain()

            await _bridge(reader, writer, up_reader, up_writer)
        except Exception:
            pass  # client vanished or target dead: drop the tunnel
        finally:
            if up_writer is not None:
                up_writer.close()
            writer.close()

    async def start(self):
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self._server.sockets[0].getsockname()[1]

    @property
    def port(self):
        return self._server.sockets[0].getsockname()[1]

    @property
    def proxy_url(self):
        return f"socks5://127.0.0.1:{self.port}"

    async def stop(self):
        if self._server:
            self._server.close()


# ---------------------------------------------------------------------------
# Proxy under test (in-process uvicorn)
# ---------------------------------------------------------------------------


async def start_proxy_server(app):
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(1000):
        if server.started:
            break
        await asyncio.sleep(0.01)
    if not server.started:
        raise RuntimeError("proxy server did not start")
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, task, port
