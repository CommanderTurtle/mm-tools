#!/usr/bin/env python3
"""HTTP Basic Auth reverse proxy for the Hy-Nomas Demo Gateway."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os

from aiohttp import ClientSession, ClientTimeout, web


BACKEND = os.environ.get("NOMAS_PROXY_BACKEND", "http://127.0.0.1:18084")
HOST = os.environ.get("NOMAS_PROXY_HOST", "0.0.0.0")
PORT = int(os.environ.get("NOMAS_PROXY_PORT", "18085"))
USERNAME = os.environ.get("NOMAS_DEMO_USERNAME", "demo")
EXPECTED = os.environ.get("NOMAS_DEMO_CREDENTIAL_SHA256", "").strip()
PUBLIC_PATH_PREFIXES = ("/share/", "/public/", "/fonts/")
PUBLIC_PATHS = {"/hunyuan-logo.png"}
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def authorized(request: web.Request) -> bool:
    value = request.headers.get("Authorization", "")
    if not EXPECTED or not value.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(value[6:], validate=True)
    except Exception:
        return False
    digest = hashlib.sha256(decoded).hexdigest()
    return hmac.compare_digest(digest, EXPECTED)


def public_request(request: web.Request) -> bool:
    return request.method in {"GET", "HEAD"} and (
        request.path.startswith(PUBLIC_PATH_PREFIXES) or request.path in PUBLIC_PATHS
    )


async def proxy(request: web.Request) -> web.StreamResponse:
    if not public_request(request) and not authorized(request):
        return web.Response(
            status=401,
            text="Authentication required\n",
            headers={
                "WWW-Authenticate": ('Basic realm="Hy-Nomas Demo", charset="UTF-8"'),
                "Cache-Control": "no-store",
            },
        )

    target = BACKEND + request.rel_url.raw_path
    if request.rel_url.raw_query_string:
        target += "?" + request.rel_url.raw_query_string
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP
        and key.lower() not in {"host", "authorization"}
    }
    peer = request.remote or ""
    prior = request.headers.get("X-Forwarded-For")
    headers["X-Forwarded-For"] = f"{prior}, {peer}" if prior else peer
    headers["X-Forwarded-Proto"] = request.scheme
    headers["X-Forwarded-Host"] = request.host

    session: ClientSession = request.app["session"]
    try:
        async with session.request(
            request.method,
            target,
            headers=headers,
            data=request.content.iter_chunked(1024 * 1024),
            allow_redirects=False,
        ) as upstream:
            response_headers = {
                key: value
                for key, value in upstream.headers.items()
                if key.lower() not in HOP_BY_HOP
            }
            response_headers["X-Content-Type-Options"] = "nosniff"
            response_headers["Referrer-Policy"] = "same-origin"
            response = web.StreamResponse(
                status=upstream.status,
                reason=upstream.reason,
                headers=response_headers,
            )
            await response.prepare(request)
            if request.method != "HEAD":
                async for chunk in upstream.content.iter_chunked(1024 * 1024):
                    await response.write(chunk)
            await response.write_eof()
            return response
    except Exception as exc:  # noqa: BLE001
        return web.Response(
            status=502,
            text=f"Upstream unavailable: {type(exc).__name__}\n",
        )


async def session_context(app: web.Application):
    timeout = ClientTimeout(
        total=None,
        connect=10,
        sock_connect=10,
        sock_read=None,
    )
    async with ClientSession(
        timeout=timeout,
        auto_decompress=False,
    ) as session:
        app["session"] = session
        yield


def main() -> None:
    if not EXPECTED:
        raise RuntimeError(
            "NOMAS_DEMO_CREDENTIAL_SHA256 is required; hash the literal "
            "'username:password' with SHA-256"
        )
    app = web.Application(client_max_size=20 * 1024**3)
    app.cleanup_ctx.append(session_context)
    app.router.add_route("*", "/{tail:.*}", proxy)
    web.run_app(
        app,
        host=HOST,
        port=PORT,
        access_log_format='%a %t "%r" %s %b',
    )


if __name__ == "__main__":
    main()
