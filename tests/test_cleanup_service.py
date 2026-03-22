"""Tests for services/cleanup_service.py"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.cleanup_service import clean_expired_rate_limits


class TestCleanExpiredRateLimits:
    @pytest.mark.asyncio
    async def test_returns_count_of_deleted_rows(self):
        db = MagicMock()
        db.table.return_value.delete.return_value.lt.return_value.execute = AsyncMock(
            return_value=MagicMock(data=[{"id": "1"}, {"id": "2"}])
        )
        async def fake_client():
            return db
        with patch("services.cleanup_service.get_service_client", side_effect=fake_client):
            count = await clean_expired_rate_limits()
        assert count == 2

    @pytest.mark.asyncio
    async def test_returns_zero_on_no_deleted_rows(self):
        db = MagicMock()
        db.table.return_value.delete.return_value.lt.return_value.execute = AsyncMock(
            return_value=MagicMock(data=[])
        )
        async def fake_client():
            return db
        with patch("services.cleanup_service.get_service_client", side_effect=fake_client):
            count = await clean_expired_rate_limits()
        assert count == 0

    @pytest.mark.asyncio
    async def test_never_raises_on_db_failure(self):
        db = MagicMock()
        db.table.return_value.delete.return_value.lt.return_value.execute = AsyncMock(
            side_effect=RuntimeError("DB down")
        )
        async def fake_client():
            return db
        with patch("services.cleanup_service.get_service_client", side_effect=fake_client):
            count = await clean_expired_rate_limits()
        assert count == 0

    @pytest.mark.asyncio
    async def test_returns_zero_when_data_is_none(self):
        db = MagicMock()
        db.table.return_value.delete.return_value.lt.return_value.execute = AsyncMock(
            return_value=MagicMock(data=None)
        )
        async def fake_client():
            return db
        with patch("services.cleanup_service.get_service_client", side_effect=fake_client):
            count = await clean_expired_rate_limits()
        assert count == 0