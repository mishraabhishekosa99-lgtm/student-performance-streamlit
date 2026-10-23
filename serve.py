from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from aiohttp import ClientError, ClientSession, WSMsgType, web
from multidict import CIMultiDict

PROJECT_DIR = Path(__file__).resolve().parent
UPSTREAM = "http://127.0.0.1:8501"
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "sec-websocket-protocol",
}


def wait_for_streamlit(process: subprocess.Popen, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    health_url = f"{UPSTREAM}/_stcore/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Streamlit exited early with status {process.returncode}.")
        try:
            with urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            time.sleep(0.4)
    raise RuntimeError("Streamlit did not become ready before the startup timeout.")


async def create_session(app: web.Application):
    app["client"] = ClientSession()
    yield
    await app["client"].close()


async def route_manifest(_: web.Request) -> web.Response:
    path = PROJECT_DIR / "manus-routes.json"
    return web.Response(
        text=path.read_text(encoding="utf-8"),
        content_type="application/json",
        headers={"Cache-Control": "no-cache"},
    )


def filtered_request_headers(request: web.Request) -> CIMultiDict:
    headers = CIMultiDict()
    for name, value in request.headers.items():
        if name.casefold() not in HOP_BY_HOP and name.casefold() not in {"host", "content-length"}:
            headers.add(name, value)
    if request.remote:
        headers["X-Forwarded-For"] = request.remote
    headers["X-Forwarded-Proto"] = request.headers.get("X-Forwarded-Proto", request.scheme)
    return headers


async def proxy_websocket(request: web.Request) -> web.StreamResponse:
    local_socket = web.WebSocketResponse(protocols=("streamlit",), heartbeat=30)
    await local_socket.prepare(request)
    client = request.app["client"]
    upstream_url = f"{UPSTREAM}{request.rel_url}"
    headers = filtered_request_headers(request)
    origin = request.headers.get("Origin")
    try:
        remote_socket = await client.ws_connect(
            upstream_url,
            headers=headers,
            protocols=("streamlit",),
            origin=origin,
            heartbeat=30,
            max_msg_size=0,
        )
    except (ClientError, OSError) as exc:
        await local_socket.close(message=f"Upstream websocket unavailable: {exc}".encode())
        return local_socket

    async def local_to_remote() -> None:
        async for message in local_socket:
            if message.type == WSMsgType.TEXT:
                await remote_socket.send_str(message.data)
            elif message.type == WSMsgType.BINARY:
                await remote_socket.send_bytes(message.data)
            elif message.type == WSMsgType.PING:
                await remote_socket.ping(message.data)
            elif message.type == WSMsgType.PONG:
                await remote_socket.pong(message.data)
            elif message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                break

    async def remote_to_local() -> None:
        async for message in remote_socket:
            if message.type == WSMsgType.TEXT:
                await local_socket.send_str(message.data)
            elif message.type == WSMsgType.BINARY:
                await local_socket.send_bytes(message.data)
            elif message.type == WSMsgType.PING:
                await local_socket.ping(message.data)
            elif message.type == WSMsgType.PONG:
                await local_socket.pong(message.data)
            elif message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                break

    try:
        await asyncio.gather(local_to_remote(), remote_to_local(), return_exceptions=True)
    finally:
        await remote_socket.close()
        await local_socket.close()
    return local_socket


async def proxy_request(request: web.Request) -> web.StreamResponse:
    if request.headers.get("Upgrade", "").casefold() == "websocket":
        return await proxy_websocket(request)

    client = request.app["client"]
    url = f"{UPSTREAM}{request.rel_url}"
    body = await request.read()
    headers = filtered_request_headers(request)
    try:
        async with client.request(
            request.method,
            url,
            headers=headers,
            data=body if body else None,
            allow_redirects=False,
        ) as remote:
            response_headers = CIMultiDict()
            for name, value in remote.headers.items():
                folded = name.casefold()
                if folded not in HOP_BY_HOP and folded not in {"content-length", "content-encoding"}:
                    response_headers.add(name, value)
            response_body = await remote.read()
            return web.Response(status=remote.status, body=response_body, headers=response_headers)
    except (ClientError, OSError) as exc:
        raise web.HTTPBadGateway(text=f"Streamlit upstream is unavailable: {exc}") from exc


def make_application() -> web.Application:
    app = web.Application(client_max_size=200 * 1024 * 1024)
    app.cleanup_ctx.append(create_session)
    app.router.add_get("/manus-routes.json", route_manifest, allow_head=True)
    app.router.add_route("*", "/{tail:.*}", proxy_request)
    return app


def main() -> None:
    environment = os.environ.copy()
    environment.setdefault("PORT", "3000")
    streamlit = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.address=127.0.0.1",
            "--server.port=8501",
            "--server.headless=true",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false",
            "--browser.gatherUsageStats=false",
        ],
        cwd=PROJECT_DIR,
        env=environment,
    )
    try:
        wait_for_streamlit(streamlit)
        web.run_app(
            make_application(),
            host="0.0.0.0",
            port=int(environment["PORT"]),
            print=None,
            handle_signals=True,
        )
    finally:
        if streamlit.poll() is None:
            streamlit.terminate()
            try:
                streamlit.wait(timeout=10)
            except subprocess.TimeoutExpired:
                streamlit.kill()
                streamlit.wait(timeout=5)


if __name__ == "__main__":
    main()
