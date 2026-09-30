"""Unit tests for the streaming gateway's API-key authentication.

No server, no GPU, no Triton -- streaming_gateway/auth.py deliberately has no
dependencies beyond the standard library so the credential logic can be tested
here rather than only against a live endpoint.

Written to cover the specific failures this module exists to prevent:

  - the previous scheme FAILED OPEN. An unset GATEWAY_AUTH_TOKEN disabled
    authentication entirely, so one missing environment variable published an
    open WebSocket endpoint with nothing in the logs to say so. That is the
    first test below, and it is the reason the rest of this module exists.
  - a single shared token cannot be revoked for one client without rotating
    the credential every other client is using.
  - the old comparison was `token == GATEWAY_AUTH_TOKEN`, not constant-time.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest  # noqa: E402

from streaming_gateway.auth import (  # noqa: E402
    Authenticator,
    generate_key,
    hash_key,
    parse_api_keys,
)


def _auth(api_keys="", shared_token="", allow_anonymous=False):
    return Authenticator(
        api_keys=api_keys, shared_token=shared_token, allow_anonymous=allow_anonymous
    )


def test_refuses_to_start_with_no_credentials():
    """The whole point: a missing env var must stop the process, not open it."""
    with pytest.raises(RuntimeError, match="No gateway credentials configured"):
        _auth()


def test_anonymous_requires_an_explicit_opt_out():
    """Running unauthenticated stays possible, but has to be stated."""
    a = _auth(allow_anonymous=True)
    assert a.authenticate(None).name == "anonymous"
    assert a.enabled is False


def test_valid_key_is_accepted_and_named():
    key, entry = generate_key("alice")
    principal = _auth(api_keys=entry).authenticate(key)
    assert principal is not None
    assert principal.name == "alice"


@pytest.mark.parametrize(
    "presented",
    [
        None,
        "",
        "sc_deadbeef_whatever",          # unknown key id
        "not-a-key",                     # not even the right shape
        "sc_short",                      # malformed
    ],
)
def test_bad_credentials_are_rejected(presented):
    _, entry = generate_key("alice")
    assert _auth(api_keys=entry).authenticate(presented) is None


def test_tampered_secret_is_rejected():
    """Right key id, wrong secret -- the id is a lookup handle, not a credential."""
    key, entry = generate_key("alice")
    tampered = key[:-1] + ("X" if key[-1] != "X" else "Y")
    assert _auth(api_keys=entry).authenticate(tampered) is None


def test_revoking_one_key_leaves_the_others_working():
    k_alice, e_alice = generate_key("alice")
    k_bob, e_bob = generate_key("bob")

    both = _auth(api_keys=f"{e_alice},{e_bob}")
    assert both.authenticate(k_alice).name == "alice"
    assert both.authenticate(k_bob).name == "bob"

    # Revoke alice by dropping her entry; bob is untouched.
    after = _auth(api_keys=e_bob)
    assert after.authenticate(k_alice) is None
    assert after.authenticate(k_bob).name == "bob"


def test_environment_never_holds_a_usable_credential():
    key, entry = generate_key("alice")
    assert key not in entry
    assert hash_key(key) in entry


def test_legacy_shared_token_still_works():
    """The bare-metal deployment behind Caddy uses this; it must not break."""
    a = _auth(shared_token="s3cret")
    assert a.authenticate("s3cret").name == "shared-token"
    assert a.authenticate("wrong") is None


def test_keys_and_shared_token_can_coexist():
    key, entry = generate_key("alice")
    a = _auth(api_keys=entry, shared_token="s3cret")
    assert a.authenticate(key).name == "alice"
    assert a.authenticate("s3cret").name == "shared-token"


@pytest.mark.parametrize(
    "bad",
    [
        "nocolons",
        "a:b:c:d",
        "id:tooshort:name",     # not a sha256
    ],
)
def test_malformed_config_fails_loudly(bad):
    """A typo in GATEWAY_API_KEYS should not silently drop a key."""
    with pytest.raises(ValueError):
        parse_api_keys(bad)


def test_blank_entries_are_ignored():
    """Trailing commas and whitespace are a normal env-var accident."""
    _, entry = generate_key("alice")
    assert len(parse_api_keys(f" {entry} , ")) == 1
