"""
OpenRouter provisioning-API wrapper.

Creates and deletes throwaway inference keys using a long-lived provisioning
key stored in the ``OPENROUTER_PROVISIONING_KEY`` environment variable.

The provisioning key is NEVER hard-coded.  To run:

    setx OPENROUTER_PROVISIONING_KEY "sk-or-v1-..."     # Windows persistent
    # then in a fresh shell:
    python scripts/openrouter_keys.py create --label fant2-distill --budget 0  # budget=0 -> free-tier only
    python scripts/openrouter_keys.py list
    python scripts/openrouter_keys.py delete --hash <key-hash>

Endpoints used (per OpenRouter docs):
    POST   /api/v1/keys              create
    GET    /api/v1/keys              list
    GET    /api/v1/keys/{hash}       inspect
    PATCH  /api/v1/keys/{hash}       update (disable / change limit)
    DELETE /api/v1/keys/{hash}       remove

The response for POST includes a one-time secret ``key`` (starts with
``sk-or-v1-``) and a permanent ``data.hash``.  Keep the hash for later
deletion; the secret is never returned again.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import urllib.request
import urllib.error


OR_BASE = "https://openrouter.ai/api/v1"
ENV_VAR = "OPENROUTER_PROVISIONING_KEY"


# --------------------------------------------------------------------------- #
# Low-level HTTP                                                              #
# --------------------------------------------------------------------------- #

def _auth_header() -> Dict[str, str]:
    prov = os.environ.get(ENV_VAR, "").strip()
    if not prov:
        raise RuntimeError(
            f"{ENV_VAR} not set.  Run:\n"
            f'    setx {ENV_VAR} "sk-or-v1-..."   (Windows)\n'
            f"    export {ENV_VAR}=sk-or-v1-...    (Linux/macOS)\n"
            f"Then restart the shell."
        )
    return {
        "Authorization": f"Bearer {prov}",
        "Content-Type":  "application/json",
        "User-Agent":    "fant2-distill/1.0",
    }


def _request(method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{OR_BASE}{path}"
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers=_auth_header())
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} {e.reason} on {method} {path}: {err_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error on {method} {path}: {e.reason}") from e
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}


# --------------------------------------------------------------------------- #
# High-level API                                                              #
# --------------------------------------------------------------------------- #

def create_key(
    label: str,
    *,
    limit_usd: Optional[float] = None,
    limit_reset: Optional[str] = None,
    include_byok_in_limit: bool = False,
    expires_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Create a new inference API key.

    Args:
        label:        Human-readable label (shown in the dashboard).
        limit_usd:    Max spend in USD; ``None`` -> unlimited (subject to the
                      master credit balance).  Use ``0`` to force free-tier-
                      only behaviour (spend refused for paid models).
        limit_reset:  One of ``"daily"``, ``"weekly"``, ``"monthly"`` or None.
        include_byok_in_limit: Count BYOK usage against the limit.
        expires_at:   When the key expires (UTC).  Defaults to now+30d.

    Returns:
        Dict with ``key`` (the one-time secret) and ``data`` (meta w/ ``hash``).
    """
    if expires_at is None:
        expires_at = datetime.now(timezone.utc) + timedelta(days=30)
    iso = expires_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    body: Dict[str, Any] = {
        "name": label,
        "expires_at": iso,
        "include_byok_in_limit": include_byok_in_limit,
    }
    if limit_usd is not None:
        body["limit"] = float(limit_usd)
    if limit_reset:
        body["limit_reset"] = limit_reset

    return _request("POST", "/keys", body)


def list_keys() -> List[Dict[str, Any]]:
    """List all keys owned by this provisioning key."""
    resp = _request("GET", "/keys")
    if isinstance(resp, dict) and "data" in resp:
        data = resp["data"]
        if isinstance(data, list):
            return data
    return []


def inspect_key(key_hash: str) -> Dict[str, Any]:
    return _request("GET", f"/keys/{key_hash}")


def update_key(key_hash: str, **fields) -> Dict[str, Any]:
    """Patch a key (e.g. ``disabled=True`` or ``limit=5.0``)."""
    return _request("PATCH", f"/keys/{key_hash}", fields)


def delete_key(key_hash: str) -> Dict[str, Any]:
    return _request("DELETE", f"/keys/{key_hash}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _cmd_create(args) -> None:
    expires = None
    if args.expires_hours:
        expires = datetime.now(timezone.utc) + timedelta(hours=args.expires_hours)

    resp = create_key(
        args.label,
        limit_usd=args.budget,
        limit_reset=args.reset or None,
        include_byok_in_limit=False,
        expires_at=expires,
    )
    key = resp.get("key", "")
    meta = resp.get("data", {}) or {}
    key_hash = meta.get("hash", "")

    print("=" * 60)
    print("  FANT 2 distillation inference key")
    print("=" * 60)
    print(f"  label:    {args.label}")
    print(f"  hash:     {key_hash}")
    print(f"  expires:  {meta.get('expires_at', '?')}")
    print(f"  limit:    {meta.get('limit', 'unlimited')} ({meta.get('limit_reset', 'no reset')})")
    print()
    print("  SECRET (save it now, never shown again):")
    print(f"     {key}")
    print()

    # Write the secret to a file the teacher client will read.
    out_dir = os.path.dirname(os.path.abspath(args.out_file)) if args.out_file else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    target = args.out_file or ".openrouter_key"
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "key":        key,
            "hash":       key_hash,
            "label":      args.label,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": meta.get("expires_at"),
        }, indent=2))
    print(f"  written:  {target}")
    print()
    print("  Delete when done:")
    print(f"    python scripts/openrouter_keys.py delete --hash {key_hash}")
    print()


def _cmd_list(args) -> None:
    for k in list_keys():
        print(f"  {k.get('hash','?'):<48}  "
              f"{k.get('name','?'):<24}  "
              f"usage={k.get('usage', 0):<8}  "
              f"limit={k.get('limit', '-')!s:<8}  "
              f"disabled={k.get('disabled', False)}")


def _cmd_inspect(args) -> None:
    print(json.dumps(inspect_key(args.hash), indent=2))


def _cmd_delete(args) -> None:
    r = delete_key(args.hash)
    print(f"  deleted {args.hash}: {r}")


def _cmd_disable(args) -> None:
    r = update_key(args.hash, disabled=True)
    print(f"  disabled {args.hash}: {r}")


def main() -> None:
    p = argparse.ArgumentParser(description="OpenRouter provisioning-API helper")
    sub = p.add_subparsers(dest="cmd", required=True)

    cr = sub.add_parser("create", help="create a throwaway key")
    cr.add_argument("--label", default=f"fant2-distill-{int(time.time())}")
    cr.add_argument("--budget", type=float, default=None,
                    help="USD spend cap; 0 = free-tier-only; omit = unlimited")
    cr.add_argument("--reset", choices=["daily", "weekly", "monthly"], default=None,
                    help="auto-reset period for the budget")
    cr.add_argument("--expires-hours", type=int, default=None,
                    help="expire the key after N hours (default: 30 days)")
    cr.add_argument("--out-file", default=".openrouter_key",
                    help="where to write the returned secret (JSON)")
    cr.set_defaults(func=_cmd_create)

    ls = sub.add_parser("list", help="list all keys")
    ls.set_defaults(func=_cmd_list)

    ins = sub.add_parser("inspect", help="show one key")
    ins.add_argument("--hash", required=True)
    ins.set_defaults(func=_cmd_inspect)

    de = sub.add_parser("delete", help="delete one key")
    de.add_argument("--hash", required=True)
    de.set_defaults(func=_cmd_delete)

    di = sub.add_parser("disable", help="disable (soft-delete) one key")
    di.add_argument("--hash", required=True)
    di.set_defaults(func=_cmd_disable)

    args = p.parse_args()
    try:
        args.func(args)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
