from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlsplit


MAX_REQUEST_BYTES = 64 * 1024


def is_loopback_host(host: str) -> bool:
    value = host.strip().strip("[]").lower()
    if value == "localhost":
        return True
    try:
        return ip_address(value).is_loopback
    except ValueError:
        return False


def split_host_header(value: str) -> tuple[str, int | None]:
    parsed = urlsplit(f"//{value}")
    if not parsed.hostname:
        raise ValueError("missing Host header")
    try:
        return parsed.hostname, parsed.port
    except ValueError as exc:
        raise ValueError("invalid Host header") from exc


def request_is_local(host_header: str, origin: str | None, server_port: int) -> bool:
    try:
        host, port = split_host_header(host_header)
    except ValueError:
        return False
    if not is_loopback_host(host) or (port is not None and port != server_port):
        return False
    if not origin:
        return True
    try:
        parsed = urlsplit(origin)
        origin_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return False
    return (
        parsed.scheme == "http"
        and parsed.hostname is not None
        and is_loopback_host(parsed.hostname)
        and origin_port == server_port
    )


SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; style-src 'self'; "
        "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; "
        "base-uri 'none'; form-action 'none'"
    ),
    "Cross-Origin-Opener-Policy": "same-origin",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}
