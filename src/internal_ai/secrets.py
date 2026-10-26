"""Operator keys for one platform home.

The file sits next to the database, mode 0600, so a key is not an environment
variable that a job or a prompt can inherit. When the file is absent, the
existing environment variables still work and the test suite needs no store.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SECRETS_NAME = "secrets.json"

CONTROL_PLANE_KEY = "control_plane_key"
SERVING_KEYS = "serving_keys"
ANTHROPIC_API_KEY = "anthropic_api_key"

_ENV_FALLBACK = {
    CONTROL_PLANE_KEY: "INTERNAL_AI_API_KEY",
    SERVING_KEYS: "INTERNAL_AI_SERVING_KEYS",
    ANTHROPIC_API_KEY: "ANTHROPIC_API_KEY",
}


def secrets_path(home: Path) -> Path:
    return home / SECRETS_NAME


def load_operator_secrets(home: Path) -> dict[str, str]:
    path = secrets_path(home)
    if not path.is_file():
        return {}
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise PermissionError(f"{path} must be readable only by its owner")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object")
    return {str(key): str(value) for key, value in payload.items() if value}


def write_operator_secrets(home: Path, values: dict[str, str]) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    path = secrets_path(home)
    path.write_text(json.dumps(values, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def operator_secret(home: Path, name: str, explicit: str | None = None) -> str:
    """A key from the file, else its environment variable, else empty.

    An explicit argument wins so a caller that already holds the value does
    not read the file twice. The value is never written to a log here.
    """
    if explicit:
        return explicit
    stored = load_operator_secrets(home).get(name, "")
    if stored:
        return stored
    env_name = _ENV_FALLBACK.get(name, "")
    return os.environ.get(env_name, "") if env_name else ""


def refuse_secret_in_prompt(secret: str, prompt: str, system: str = "") -> None:
    """A model prompt is not a place for an operator key."""
    if secret and (secret in prompt or secret in system):
        raise RuntimeError("refusing to send a prompt that contains an operator secret")
