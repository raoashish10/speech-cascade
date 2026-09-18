"""API-key authentication for the streaming gateway.

Replaces the single shared `GATEWAY_AUTH_TOKEN` with named, individually
revocable keys. The shared token still works -- the bare-metal deployment
behind Vast.ai's Caddy edge uses it and nothing there needs to change -- but
new deployments should prefer keys, for three reasons the old scheme could
not give:

  revocation    drop one client without rotating the credential every other
                client is using.
  attribution   logs say which client opened a session, instead of "someone
                holding the token".
  rate limiting  a per-key session cap is only possible once sessions have an
                identity. The global cap in server.py protects the GPU from
                the fleet as a whole, but cannot stop one client starving the
                rest -- see MAX_CONCURRENT_SESSIONS' own comment about VRAM
                headroom.

KEY FORMAT. A key is `sc_<key_id>_<secret>`; the client sends the whole thing.
Only its SHA-256 is configured on the server, so the deployment environment
never holds a usable credential:

    GATEWAY_API_KEYS=<key_id>:<sha256-of-whole-key>:<name>,...

The key_id is a lookup handle, not a secret -- it exists so verification is a
dict hit rather than a scan over every configured key, and so logs can name a
client without printing anything sensitive. Comparison is constant-time.

FAIL-CLOSED. Starting with no credentials configured used to silently disable
authentication entirely, so a single missing environment variable published an
open endpoint with nothing in the logs to say so. That is now a startup error.
Running genuinely unauthenticated -- which is correct when something upstream
(Caddy, your own reverse proxy) already authenticates -- has to be stated
explicitly with GATEWAY_ALLOW_ANONYMOUS=1.

TRANSPORT. Send keys as `Authorization: Bearer <key>`. The `?token=` query
parameter is still accepted because browsers cannot set headers on a
WebSocket upgrade, but a key in a URL ends up in proxy access logs, so
service-to-service clients should always use the header.

Minting a key:

    python3 -m streaming_gateway.auth --new alice
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass

KEY_PREFIX = "sc"


@dataclass(frozen=True)
class Principal:
    """Who authenticated. `key_id` is safe to log; the key itself is not."""

    key_id: str
    name: str

    def __str__(self) -> str:  # for log lines
        return f"{self.name} ({self.key_id})"


ANONYMOUS = Principal(key_id="-", name="anonymous")
SHARED_TOKEN_PRINCIPAL = Principal(key_id="shared", name="shared-token")


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def generate_key(name: str) -> tuple[str, str]:
    """Mint a key. Returns (key, env_entry); only the entry is stored."""
    key_id = secrets.token_hex(4)
    key = f"{KEY_PREFIX}_{key_id}_{secrets.token_urlsafe(32)}"
    return key, f"{key_id}:{hash_key(key)}:{name}"


def parse_api_keys(raw: str) -> dict[str, tuple[str, str]]:
    """Parse GATEWAY_API_KEYS into {key_id: (sha256, name)}."""
    table: dict[str, tuple[str, str]] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"GATEWAY_API_KEYS entry {entry!r} is malformed; expected "
                "<key_id>:<sha256>:<name>"
            )
        key_id, digest, name = (p.strip() for p in parts)
        if len(digest) != 64:
            raise ValueError(
                f"GATEWAY_API_KEYS entry for {key_id!r} does not look like a "
                "SHA-256 hex digest -- store the hash, not the key itself"
            )
        table[key_id] = (digest.lower(), name)
    return table


class Authenticator:
    """Verifies presented credentials against configured keys."""

    def __init__(
        self,
        api_keys: str | None = None,
        shared_token: str | None = None,
        allow_anonymous: bool | None = None,
    ) -> None:
        api_keys = os.environ.get("GATEWAY_API_KEYS", "") if api_keys is None else api_keys
        shared_token = (
            os.environ.get("GATEWAY_AUTH_TOKEN", "") if shared_token is None else shared_token
        )
        if allow_anonymous is None:
            allow_anonymous = os.environ.get("GATEWAY_ALLOW_ANONYMOUS", "") == "1"

        self.keys = parse_api_keys(api_keys)
        self.shared_token = shared_token
        self.allow_anonymous = allow_anonymous

        if not self.keys and not self.shared_token and not self.allow_anonymous:
            raise RuntimeError(
                "No gateway credentials configured. Set GATEWAY_API_KEYS (preferred) "
                "or GATEWAY_AUTH_TOKEN. If this gateway really should accept "
                "unauthenticated connections -- which is only correct when something "
                "upstream already authenticates them -- set GATEWAY_ALLOW_ANONYMOUS=1 "
                "to say so explicitly. Refusing to start rather than silently "
                "publishing an open endpoint."
            )

    @property
    def enabled(self) -> bool:
        return bool(self.keys or self.shared_token)

    def authenticate(self, presented: str | None) -> Principal | None:
        """Return the Principal for a credential, or None to reject."""
        if not self.enabled:
            return ANONYMOUS
        if not presented:
            return None

        # API key: parse the id out to pick which digest to compare against.
        parts = presented.split("_", 2)
        if len(parts) == 3 and parts[0] == KEY_PREFIX:
            entry = self.keys.get(parts[1])
            if entry is None:
                return None
            digest, name = entry
            if hmac.compare_digest(hash_key(presented), digest):
                return Principal(key_id=parts[1], name=name)
            return None

        # Shared token (legacy). Constant-time, unlike the original ==.
        if self.shared_token and hmac.compare_digest(presented, self.shared_token):
            return SHARED_TOKEN_PRINCIPAL
        return None


def _main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Mint a gateway API key.")
    p.add_argument("--new", metavar="NAME", required=True, help="client name")
    args = p.parse_args()

    key, entry = generate_key(args.new)
    print("Give this to the client (shown once, not recoverable):\n")
    print(f"    {key}\n")
    print("Add this to the gateway's environment:\n")
    print(f"    GATEWAY_API_KEYS={entry}\n")
    print("Append further keys comma-separated. Revoke by deleting an entry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
