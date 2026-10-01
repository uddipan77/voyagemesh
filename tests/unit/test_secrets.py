"""Secret loading must fail safely and must never leak the value."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from vm_config.secrets import SecretLoadError, read_secret_file, try_read_secret_file

pytestmark = pytest.mark.unit

SAMPLE_KEY = "gsk_TESTKEY_abcdefghijklmnopqrstuvwxyz0123456789"


@pytest.fixture()
def key_file(tmp_path):
    path = tmp_path / "api.key"
    path.write_text(f"  {SAMPLE_KEY}\n\n", encoding="utf-8")
    return path


def test_reads_and_strips_whitespace(key_file):
    secret = read_secret_file(key_file, name="groq api key")
    assert isinstance(secret, SecretStr)
    assert secret.get_secret_value() == SAMPLE_KEY


def test_secret_is_masked_in_repr_and_str(key_file):
    secret = read_secret_file(key_file)
    assert SAMPLE_KEY not in repr(secret)
    assert SAMPLE_KEY not in str(secret)
    assert SAMPLE_KEY not in f"{secret}"


def test_missing_file_raises_actionable_error(tmp_path):
    missing = tmp_path / "nope.key"
    with pytest.raises(SecretLoadError) as exc:
        read_secret_file(missing, name="groq api key")
    assert "groq api key" in str(exc.value)
    assert str(missing) in str(exc.value)


def test_empty_file_raises(tmp_path):
    empty = tmp_path / "empty.key"
    empty.write_text("   \n\t\n", encoding="utf-8")
    with pytest.raises(SecretLoadError, match="empty"):
        read_secret_file(empty)


def test_directory_is_rejected(tmp_path):
    with pytest.raises(SecretLoadError, match="not a regular file"):
        read_secret_file(tmp_path)


def test_oversized_file_is_rejected(tmp_path):
    big = tmp_path / "big.key"
    big.write_text("x" * 9000, encoding="utf-8")
    with pytest.raises(SecretLoadError, match="exceeds"):
        read_secret_file(big)


def test_error_message_never_contains_secret_content(tmp_path):
    """A malformed secret file must not have its contents echoed into the exception."""
    binary = tmp_path / "binary.key"
    binary.write_bytes(b"\xff\xfe" + SAMPLE_KEY.encode())
    with pytest.raises(SecretLoadError) as exc:
        read_secret_file(binary)
    assert SAMPLE_KEY not in str(exc.value)


def test_try_read_returns_none_instead_of_raising(tmp_path):
    assert try_read_secret_file(None) is None
    assert try_read_secret_file(tmp_path / "absent.key") is None


def test_try_read_returns_value_when_present(key_file):
    secret = try_read_secret_file(key_file)
    assert secret is not None
    assert secret.get_secret_value() == SAMPLE_KEY
