"""Tests for services/cleanup_service.py"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.cleanup_service import clean_expired_rate_limits


class TestCleanExpiredRateLimits:
    @pytest.mark.asyncio
    async def test_returns_count_of_deleted_rows(self):
        with patch("services.cleanup_service.get_service_client", new_callable=AsyncMock) as mock_get:
            mock_get.return_value.table.return_value.delete.return_value.lt.return_value.execute = AsyncMock(
                return_value=MagicMock(data=[{"id": "1"}, {"id": "2"}])
            )
            count = await clean_expired_rate_limits()

        assert count == 2

    @pytest.mark.asyncio
    async def test_returns_zero_on_no_deleted_rows(self):
        with patch("services.cleanup_service.get_service_client", new_callable=AsyncMock) as mock_get:
            mock_get.return_value.table.return_value.delete.return_value.lt.return_value.execute = AsyncMock(
                return_value=MagicMock(data=[])
            )
            count = await clean_expired_rate_limits()

        assert count == 0

    @pytest.mark.asyncio
    async def test_never_raises_on_db_failure(self):
        with patch("services.cleanup_service.get_service_client", new_callable=AsyncMock) as mock_get:
            mock_get.return_value.table.return_value.delete.return_value.lt.return_value.execute = AsyncMock(
                side_effect=RuntimeError("DB down")
            )
            count = await clean_expired_rate_limits()

        assert count == 0

    @pytest.mark.asyncio
    async def test_returns_zero_when_data_is_none(self):
        with patch("services.cleanup_service.get_service_client", new_callable=AsyncMock) as mock_get:
            mock_get.return_value.table.return_value.delete.return_value.lt.return_value.execute = AsyncMock(
                return_value=MagicMock(data=None)
            )
            count = await clean_expired_rate_limits()

        assert count == 0