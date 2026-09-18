# Streaming gateway

FastAPI WebSocket service fronting the Triton voice pipeline. A client
streams raw 16kHz float32 PCM mic audio in over `/ws/stream`; VAD segments
it into utterances, and each finalized utterance runs ASR -> streamed LLM ->
per-sentence TTS against Triton, with transcript / LLM-delta / TTS-audio
messages streamed back as they're produced. See `server.py` and
`session.py` for the message protocol and turn state machine.

Run with:
```bash
uvicorn streaming_gateway.server:app --host 127.0.0.1 --port 18010 --ws-max-size 20971520
```
(already wired up as the `speech-cascade-gateway` supervisor service —
`supervisorctl restart speech-cascade-gateway`, logs at
`/var/log/portal/speech-cascade-gateway.log`).

## External access

The gateway process itself only ever binds `127.0.0.1:18010`. Reachability
from outside the box comes entirely from the Vast.ai instance's Caddy
reverse-proxy edge, wired up via `/etc/portal.yaml` (system-level config,
not part of this repo — see "System-level wiring" below).

**External URL:**
```
ws://<PUBLIC_IPADDR>:<VAST_TCP_PORT_10100>/ws/stream?token=<instance token>
```
(`wss://` instead of `ws://` if the instance has `ENABLE_HTTPS=true` — check
before assuming; this instance does not, so `ws://`.) `<PUBLIC_IPADDR>` and
`<VAST_TCP_PORT_10100>` are instance env vars; `vast-capabilities | jq
'.services[] | select(.name=="Streaming Gateway")'` prints the ready-made
`direct_url` too (Caddy's manifest doesn't distinguish `ws`/`wss` from
`http`/`https` in that field — swap the scheme by hand as above).

**Auth.** The instance has one shared token (`$OPEN_BUTTON_TOKEN` /
`$WEB_PASSWORD` — both work, they're the same value), not a per-user
credential. Caddy checks it on every request that reaches the external
port, WebSocket upgrades included, and rejects with HTTP 401 *before* the
upgrade completes if it's missing or wrong — confirmed empirically (not
just assumed from the docs) while wiring this up: an unauthenticated
handshake against the external port gets `InvalidStatus: HTTP 401`, a wrong
token gets the same, and a correct token completes the upgrade and a full
turn end-to-end. Two ways to send it, both accepted by Caddy for the
WebSocket upgrade specifically:

- **Query param** — `?token=<token>`. This is the one to use from a
  browser: the native `WebSocket` API can't set custom headers on the
  upgrade request, so this is the only option available to real web/mobile
  clients.
- **Header** — `Authorization: Bearer <token>`. Works for non-browser
  clients (this repo's test script supports both via `--auth-mode`).

A request straight to `127.0.0.1:18010` (the internal port, from
*inside* the box) bypasses Caddy entirely and needs no token — that's
expected (localhost-to-internal-port is the same-box admin path, not the
external one) and is how the supervisor service itself, and any other
process on the box, talk to the gateway.

**Copy-pasteable example**, streaming a wav file from outside the box
through the authed edge and saving the returned TTS audio locally:
```bash
python3 scripts/test_streaming_client.py --wav your_clip.wav \
  --gateway-url ws://<PUBLIC_IPADDR>:<VAST_TCP_PORT_10100>/ws/stream \
  --token "$OPEN_BUTTON_TOKEN" --voice Sofia
```
Add `--auth-mode header` to send the token as `Authorization: Bearer …`
instead of the default `?token=` query param.

## API keys (deployments without an upstream auth edge)

Everything above describes the bare-metal instance, where Caddy authenticates
and the gateway trusts it. In the container deployments there is no Caddy, so
the gateway checks credentials itself — see `streaming_gateway/auth.py`.

**It refuses to start with no credentials configured.** The earlier behavior
was to accept every connection when `GATEWAY_AUTH_TOKEN` was unset, which
turned one missing environment variable into an open WebSocket endpoint with
nothing in the logs to indicate it. Running unauthenticated is still possible
— it is correct when something upstream already authenticates — but it now has
to be said out loud with `GATEWAY_ALLOW_ANONYMOUS=1`.

**Prefer named API keys over the shared token.** Mint one per client:

```bash
python3 -m streaming_gateway.auth --new alice
```

That prints the key to hand the client (once — it is not recoverable) and the
entry for the gateway's environment:

```
GATEWAY_API_KEYS=a7401936:af1e7cac...b533:alice
```

Only the SHA-256 is stored, so the environment never holds a usable
credential. Comma-separate further keys; revoke one by deleting its entry and
redeploying. Clients authenticate exactly as before:

```
Authorization: Bearer sc_a7401936_e5SFd0g-...
```

Three things this buys that a single shared token cannot:

- **revocation** — drop one client without rotating the credential every other
  client is using.
- **attribution** — log lines name the client (`session opened for alice
  (a7401936)`) instead of "someone holding the token".
- **per-key limits** — `GATEWAY_MAX_SESSIONS_PER_KEY` (default 2) caps each
  client separately. The global `GATEWAY_MAX_SESSIONS` protects the GPU from
  the fleet as a whole but cannot stop one client taking every slot, and a cap
  per client is only expressible once connections have an identity.

`GATEWAY_AUTH_TOKEN` still works and can coexist with keys, so nothing about
the bare-metal deployment has to change.

**Send keys in the header, not the URL.** `?token=` is still accepted because
a browser's `WebSocket` API cannot set headers, but a credential in a query
string ends up in proxy access logs. For service-to-service clients — the only
kind here today — always use `Authorization: Bearer`.

**Why Triton itself stays internal-only.** Triton's HTTP/gRPC/metrics ports
(18000/18001/18002) are not exposed externally, and that's deliberate, not
an oversight: this gateway already gives an external caller everything
Triton would (transcript, LLM text, TTS audio) end-to-end, without also
handing them the ability to load/unload/reload arbitrary models on shared,
VRAM-constrained hardware. Exposing Triton too would only widen the attack
surface for no added external capability. If a future consumer genuinely
needs raw Triton access from outside the box, that's a deliberate separate
decision, not a default to reach for.

**System-level wiring** (lives on the instance in `/etc/portal.yaml`, not
tracked in this repo — reproduce on a fresh instance by re-running this):
```python
python3 -c "
import yaml
d = yaml.safe_load(open('/etc/portal.yaml')) or {'applications': {}}
d['applications']['Streaming Gateway'] = {
    'hostname': 'localhost',
    'external_port': 10100,
    'internal_port': 18010,
    'open_path': '/ws/stream',
    'name': 'Streaming Gateway',
}
yaml.safe_dump(d, open('/etc/portal.yaml', 'w'), sort_keys=False)
"
supervisorctl restart caddy
```
Port `10100` was picked because it was the free external port
(`vast-capabilities | jq '.instance.open_ports[]|select(.in_use==false)'`)
at the time this was wired up; a different instance may need a different
free port. Triton's ports are deliberately *not* added to `/etc/portal.yaml`
(see above).

## Safeguards for external reachability

Exposing the gateway externally means anyone holding the shared instance
token can open a session — there's no per-user scoping. What's here, and
what's deliberately not, for a small research/demo deployment:

- **Concurrent-session cap** (`MAX_CONCURRENT_SESSIONS`, default 4, env
  `GATEWAY_MAX_SESSIONS`, in `server.py`). The real risk on this box isn't
  abuse of the gateway process — it's every open session queuing work onto
  the *same* GPU-resident Triton models (`whisper_asr`, `qwen_llm`,
  `chatterbox_tts`) with limited VRAM headroom at steady state (the ~5.5GB
  figure here was measured against kokoro_tts; not yet re-measured against
  chatterbox_tts's different footprint — measured standalone at ~3.0-3.4GB,
  see docs/tts-replacement-investigation.md, but not yet under this
  gateway's own concurrent-session load). A hard
  cap on concurrent WebSocket sessions is the cheapest guard against that
  specific failure mode: past the cap, new connections are accepted (so the
  client gets a clean WS close, not a raw TCP-level failure) and
  immediately closed with code `1013` ("Try Again Later") and a reason
  string, rather than being left to pile into the same inference queue.
  Verified by opening 6 concurrent connections against the cap of 4: the
  first 4 stayed open, the 5th and 6th were closed with `1013 / gateway at
  capacity, try again shortly`.
- **Message-size cap** — already present before this PR
  (`--ws-max-size 20971520`, i.e. 20MB, set because the default 1MB WS frame
  limit is too small for a base64-encoded multi-second float32 TTS chunk).
  Left as-is; still the right number for what a session's audio_chunk/
  tts_chunk messages actually need.
- **Deliberately not added: per-IP rate limiting.** The trust boundary here
  is the shared token, not the source IP — anyone with the token is
  equally "trusted" (or not) regardless of which IP they connect from, so
  IP-based limiting wouldn't add a meaningful guarantee, just complexity.
  If this ever grows real per-user identity, rate limiting keyed on that
  identity would be the right follow-up, not IP.
- **Deliberately not added: a session/connection duration cap beyond the
  existing per-utterance `MAX_UTTERANCE_SEC` (15s) in `session.py`.** That
  already bounds the worst case for a single stuck utterance; capping total
  *connection* lifetime would just cut off a legitimately long conversation
  for no real protective benefit, since idle time between utterances costs
  nothing (no inference runs until VAD confirms speech).
- **Deliberately not added: token rotation / per-session tokens / auth at
  the application layer.** Caddy's edge auth is the instance-wide security
  boundary already used by every other exposed service on this box
  (Tensorboard, Syncthing, etc.); building a second, gateway-specific auth
  layer on top would be inconsistent with how the rest of the instance
  works and is more than this demo scope calls for.

## Testing

`scripts/test_streaming_client.py` streams a wav file in real-time-simulated
chunks and prints/saves the transcript, LLM text, and TTS audio as they
stream back. Works both against the local internal port (no `--token`
needed) and, with `--token`, against the external Caddy-authed port — see
the "Copy-pasteable example" above.
