"""Tests for services/twilio_service.py

Covers:
  - _validate_phone (E.164 format enforcement)
  - _truncate_body (character limit)
  - _is_configured (missing credential detection)
  - send_sms (success, invalid phone, unconfigured, empty body, timeout)
  - send_guardian_alert (retry logic, raises on exhaustion, invalid phone)

No real Twilio calls.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.twilio_service import (
    _MAX_SMS_BODY_CHARS,
    _is_configured,
    _reset_client,
    _truncate_body,
    _validate_phone,
    send_guardian_alert,
    send_sms,
)


@pytest.fixture(autouse=True)
def reset_twilio_client():
    _reset_client()
    yield
    _reset_client()


# ─────────────────────────────────────────────────────────────────────────────
# _validate_phone
# ─────────────────────────────────────────────────────────────────────────────


class TestValidatePhone:
    def test_valid_e164(self):
        assert _validate_phone("+94771234567") == "+94771234567"

    def test_strips_whitespace(self):
        assert _validate_phone("  +94771234567  ") == "+94771234567"

    def test_missing_plus_returns_none(self):
        assert _validate_phone("94771234567") is None

    def test_too_short_returns_none(self):
        assert _validate_phone("+12345") is None

    def test_too_long_returns_none(self):
        assert _validate_phone("+1234567890123456") is None

    def test_empty_string_returns_none(self):
        assert _validate_phone("") is None

    def test_letters_returns_none(self):
        assert _validate_phone("+1800FLOWERS") is None


# ─────────────────────────────────────────────────────────────────────────────
# _truncate_body
# ─────────────────────────────────────────────────────────────────────────────


class TestTruncateBody:
    def test_short_body_unchanged(self):
        body = "Hello"
        assert _truncate_body(body) == body

    def test_body_at_limit_unchanged(self):
        body = "a" * _MAX_SMS_BODY_CHARS
        assert _truncate_body(body) == body

    def test_body_over_limit_truncated(self):
        body = "a" * (_MAX_SMS_BODY_CHARS + 50)
        result = _truncate_body(body)
        assert len(result) == _MAX_SMS_BODY_CHARS
        assert result.endswith("…")


# ─────────────────────────────────────────────────────────────────────────────
# _is_configured
# ─────────────────────────────────────────────────────────────────────────────


class TestIsConfigured:
    def test_all_configured(self):
        with patch("services.twilio_service.get_settings") as mock_settings:
            mock_settings.return_value.twilio_account_sid = "ACtest"
            mock_settings.return_value.twilio_auth_token = "token"
            mock_settings.return_value.twilio_from_number = "+10000000000"
            ok, reason = _is_configured()
        assert ok is True
        assert reason == ""

    def test_missing_sid(self):
        with patch("services.twilio_service.get_settings") as mock_settings:
            mock_settings.return_value.twilio_account_sid = ""
            mock_settings.return_value.twilio_auth_token = "token"
            mock_settings.return_value.twilio_from_number = "+10000000000"
            ok, reason = _is_configured()
        assert ok is False
        assert "ACCOUNT_SID" in reason

    def test_missing_auth_token(self):
        with patch("services.twilio_service.get_settings") as mock_settings:
            mock_settings.return_value.twilio_account_sid = "ACtest"
            mock_settings.return_value.twilio_auth_token = ""
            mock_settings.return_value.twilio_from_number = "+10000000000"
            ok, reason = _is_configured()
        assert ok is False
        assert "AUTH_TOKEN" in reason


# ─────────────────────────────────────────────────────────────────────────────
# send_sms
# ─────────────────────────────────────────────────────────────────────────────


class TestSendSms:
    @pytest.mark.asyncio
    async def test_unconfigured_returns_false(self):
        with patch("services.twilio_service._is_configured", return_value=(False, "missing SID")):
            result = await send_sms("+94771234567", "Hello")
        assert result is False

    @pytest.mark.asyncio
    async def test_invalid_phone_returns_false(self):
        with patch("services.twilio_service._is_configured", return_value=(True, "")):
            result = await send_sms("not-a-phone", "Hello")
        assert result is False

    @pytest.mark.asyncio
    async def test_empty_body_returns_false(self):
        with patch("services.twilio_service._is_configured", return_value=(True, "")):
            result = await send_sms("+94771234567", "")
        assert result is False

    @pytest.mark.asyncio
    async def test_successful_send_returns_true(self):
        mock_msg = MagicMock()
        mock_msg.sid = "SM123"
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg

        with patch("services.twilio_service._is_configured", return_value=(True, "")):
            with patch("services.twilio_service._get_client", return_value=mock_client):
                with patch("services.twilio_service.get_settings") as mock_settings:
                    mock_settings.return_value.twilio_from_number = "+10000000000"
                    result = await send_sms("+94771234567", "Hello!")

        assert result is True

    @pytest.mark.asyncio
    async def test_twilio_timeout_returns_false(self):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = asyncio.TimeoutError()

        with patch("services.twilio_service._is_configured", return_value=(True, "")):
            with patch("services.twilio_service._get_client", return_value=mock_client):
                with patch("services.twilio_service.get_settings") as mock_settings:
                    mock_settings.return_value.twilio_from_number = "+10000000000"
                    with patch("asyncio.to_thread", side_effect=asyncio.TimeoutError()):
                        result = await send_sms("+94771234567", "Hello")

        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# send_guardian_alert
# ─────────────────────────────────────────────────────────────────────────────


class TestSendGuardianAlert:
    @pytest.mark.asyncio
    async def test_raises_if_not_configured(self):
        with patch("services.twilio_service._is_configured", return_value=(False, "missing SID")):
            with pytest.raises(RuntimeError, match="not configured"):
                await send_guardian_alert(
                    guardian_phone="+94771234567",
                    user_display_name="Isara",
                    alert_level="high",
                    session_id="sess-1",
                )

    @pytest.mark.asyncio
    async def test_raises_on_invalid_phone(self):
        with patch("services.twilio_service._is_configured", return_value=(True, "")):
            with pytest.raises(ValueError, match="invalid phone"):
                await send_guardian_alert(
                    guardian_phone="not-a-phone",
                    user_display_name="Isara",
                    alert_level="high",
                    session_id="sess-1",
                )

    @pytest.mark.asyncio
    async def test_successful_send_returns_sid(self):
        mock_msg = MagicMock()
        mock_msg.sid = "SMguardian123"
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg

        with patch("services.twilio_service._is_configured", return_value=(True, "")):
            with patch("services.twilio_service._get_client", return_value=mock_client):
                with patch("services.twilio_service.get_settings") as mock_settings:
                    mock_settings.return_value.twilio_from_number = "+10000000000"
                    sid = await send_guardian_alert(
                        guardian_phone="+94771234567",
                        user_display_name="Isara",
                        alert_level="crisis",
                        session_id="sess-1",
                    )

        assert sid == "SMguardian123"

    @pytest.mark.asyncio
    async def test_raises_after_all_retries_fail(self):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("network error")

        with patch("services.twilio_service._is_configured", return_value=(True, "")):
            with patch("services.twilio_service._get_client", return_value=mock_client):
                with patch("services.twilio_service.get_settings") as mock_settings:
                    mock_settings.return_value.twilio_from_number = "+10000000000"
                    with patch("asyncio.sleep", new_callable=AsyncMock):
                        with patch("asyncio.to_thread", side_effect=RuntimeError("network error")):
                            with pytest.raises(RuntimeError, match="all.*attempts failed"):
                                await send_guardian_alert(
                                    guardian_phone="+94771234567",
                                    user_display_name="Isara",
                                    alert_level="high",
                                    session_id="sess-1",
                                )

    @pytest.mark.asyncio
    async def test_crisis_body_contains_urgent(self):
        """Crisis alerts should use the urgent wording."""
        bodies_sent = []
        mock_client = MagicMock()

        def capture_create(**kwargs):
            bodies_sent.append(kwargs.get("body", ""))
            msg = MagicMock()
            msg.sid = "SM123"
            return msg

        mock_client.messages.create.side_effect = capture_create

        with patch("services.twilio_service._is_configured", return_value=(True, "")):
            with patch("services.twilio_service._get_client", return_value=mock_client):
                with patch("services.twilio_service.get_settings") as mock_settings:
                    mock_settings.return_value.twilio_from_number = "+10000000000"
                    await send_guardian_alert(
                        guardian_phone="+94771234567",
                        user_display_name="Isara",
                        alert_level="crisis",
                        session_id="sess-1",
                    )

        assert any("URGENT" in b or "urgent" in b.lower() for b in bodies_sent)
