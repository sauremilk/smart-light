"""Tests for session-log privacy helpers."""

from core.session_log import pseudonymize_identity


def test_pseudonymize_identity_is_deterministic_for_same_input():
    a = pseudonymize_identity("participant-123", "salt-A")
    b = pseudonymize_identity("participant-123", "salt-A")

    assert a == b
    assert isinstance(a, str)
    assert len(a) == 16


def test_pseudonymize_identity_changes_with_salt_and_handles_none():
    x = pseudonymize_identity("participant-123", "salt-A")
    y = pseudonymize_identity("participant-123", "salt-B")

    assert x != y
    assert pseudonymize_identity(None, "salt-A") is None
