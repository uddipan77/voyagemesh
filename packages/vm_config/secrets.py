"""Runtime secret loading.

Secrets are *always* read from a file at runtime and are never embedded in source,
images, logs, traces, or API responses. Values are wrapped in :class:`SecretStr` so an
accidental ``str()``/``repr()``/log interpolation renders ``**********`` instead of the
plaintext.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr

__all__ = ["SecretLoadError", "read_secret_file", "try_read_secret_file"]

# A secret file larger than this is almost certainly the wrong file (e.g. a cert bundle
# or an accidental log dump); refuse it rather than loading megabytes into memory.
_MAX_SECRET_BYTES = 8192


class SecretLoadError(RuntimeError):
    """Raised when a secret file is missing, unreadable, empty, or implausible.

    The message deliberately contains the *path* but never any file content, so it is
    safe to surface in logs and startup diagnostics.
    """


def read_secret_file(path: str | Path, *, name: str = "secret") -> SecretStr:
    """Read a single-value secret from ``path``.

    Whitespace and trailing newlines are stripped — editors and ``echo`` habitually append
    a newline, and a stray ``\\n`` in an ``Authorization`` header produces a confusing
    401 rather than an obvious error.

    Args:
        path: Filesystem location of the secret.
        name: Human-readable name used in error messages (never the value).

    Returns:
        The stripped secret wrapped in :class:`SecretStr`.

    Raises:
        SecretLoadError: If the file is missing, not a file, unreadable, empty, or
            larger than 8 KiB.
    """
    resolved = Path(path).expanduser()

    if not resolved.exists():
        raise SecretLoadError(
            f"{name}: file not found at {resolved}. "
            f"Create it, or point the corresponding *_FILE environment variable at the "
            f"correct location. The file must contain only the secret value."
        )
    if not resolved.is_file():
        raise SecretLoadError(f"{name}: {resolved} exists but is not a regular file.")

    try:
        size = resolved.stat().st_size
    except OSError as exc:  # pragma: no cover - platform dependent
        raise SecretLoadError(f"{name}: cannot stat {resolved} ({exc.strerror}).") from exc

    if size > _MAX_SECRET_BYTES:
        raise SecretLoadError(
            f"{name}: {resolved} is {size} bytes, which exceeds the {_MAX_SECRET_BYTES} byte "
            f"limit. This is unlikely to be a single API key — check the path."
        )

    try:
        raw = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise SecretLoadError(f"{name}: cannot read {resolved} ({exc.strerror}).") from exc
    except UnicodeDecodeError as exc:
        raise SecretLoadError(
            f"{name}: {resolved} is not valid UTF-8 text ({exc.reason})."
        ) from exc

    value = raw.strip()
    if not value:
        raise SecretLoadError(f"{name}: {resolved} is empty (or contains only whitespace).")

    return SecretStr(value)


def try_read_secret_file(path: str | Path | None, *, name: str = "secret") -> SecretStr | None:
    """Best-effort variant of :func:`read_secret_file`.

    Returns ``None`` instead of raising when ``path`` is ``None`` or the secret cannot be
    loaded. Used by optional integrations so that, for example, a missing Groq key
    degrades the service to mock-LLM mode rather than preventing startup.
    """
    if path is None:
        return None
    try:
        return read_secret_file(path, name=name)
    except SecretLoadError:
        return None
