"""Per-client rate limiting (slowapi) configured only from Settings.

The limiter keeps its counters in process memory: limits apply per worker process, so with
N uvicorn workers or N replicas a client can make up to N times the configured rate. That
is acceptable for the current single-process deployment; scaling out needs a shared
storage backend (e.g. Redis via `storage_uri`).

Client IPs come from the direct peer unless that peer is a configured trusted proxy (see
`resolve_client_ip`), so the limit cannot be dodged by sending a fake X-Forwarded-For.
"""

import ipaddress
import os
import warnings
from collections.abc import Callable, Sequence

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import IPvAnyNetwork
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

RATE_LIMITED_DETAIL = "Rate limit exceeded"
UNKNOWN_CLIENT = "unknown"
FORWARDED_FOR_HEADER = "x-forwarded-for"

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def _parse_ip(value: str) -> IPAddress | None:
    """Parse an IP address, returning None for anything that is not one."""
    try:
        return ipaddress.ip_address(value.strip())
    except ValueError:
        return None


def _is_trusted(address: IPAddress, trusted: Sequence[IPvAnyNetwork]) -> bool:
    """True if `address` lies in any trusted network of the same IP version."""
    return any(address.version == net.version and address in net for net in trusted)


def resolve_client_ip(
    peer: str | None, forwarded_for: Sequence[str], trusted: Sequence[IPvAnyNetwork]
) -> str:
    """Return the IP to rate-limit on.

    If the direct `peer` is a trusted proxy, X-Forwarded-For entries are walked from the
    right (closest hop first) and the first address that is not itself a trusted proxy
    is the client. The leftmost entry is never trusted blindly: it is only reached when
    every hop to its right is trusted. Unparsable entries stop the walk and the peer is
    used instead (conservative: shared proxy bucket rather than a spoofable value).

    Args:
        peer: IP of the direct TCP peer (`request.client.host`), if known.
        forwarded_for: Every X-Forwarded-For header value, in received order.
        trusted: Trusted proxy networks from Settings.
    """
    if peer is None:
        return UNKNOWN_CLIENT
    peer_ip = _parse_ip(peer)
    if peer_ip is None or not _is_trusted(peer_ip, trusted):
        return peer
    hops = [hop.strip() for value in forwarded_for for hop in value.split(",") if hop.strip()]
    client = peer
    for hop in reversed(hops):
        hop_ip = _parse_ip(hop)
        if hop_ip is None:
            return peer
        client = str(hop_ip)
        if not _is_trusted(hop_ip, trusted):
            return client
    return client


def client_ip_key(trusted: Sequence[IPvAnyNetwork]) -> Callable[[Request], str]:
    """Build a slowapi key function that resolves client IPs with `trusted` proxies."""
    trusted_networks = tuple(trusted)

    def key(request: Request) -> str:
        peer = request.client.host if request.client else None
        return resolve_client_ip(
            peer, request.headers.getlist(FORWARDED_FOR_HEADER), trusted_networks
        )

    return key


def build_limiter(trusted_proxies: Sequence[IPvAnyNetwork] = ()) -> Limiter:
    """Create an in-memory (per-process) limiter keyed on the resolved client IP.

    slowapi would otherwise load a `.env` from the working directory; pointing it at the
    null device keeps every limit and toggle under `Settings` control.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Config file .* not found")
        return Limiter(
            key_func=client_ip_key(trusted_proxies),
            storage_uri="memory://",
            strategy="fixed-window",
            config_filename=os.devnull,
        )


def rate_limit_exceeded_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a generic 429 with a Retry-After header when the limit window is known."""
    headers: dict[str, str] = {}
    if isinstance(exc, RateLimitExceeded) and exc.limit is not None:
        headers["Retry-After"] = str(exc.limit.limit.get_expiry())
    return JSONResponse({"detail": RATE_LIMITED_DETAIL}, status_code=429, headers=headers)
