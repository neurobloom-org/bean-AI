"""Tests for shared/config.py

Covers:
  - Settings defaults
  - cors_origins_list property
  - is_production property
  - validate_required_for_production (missing fields, localhost check)
  - get_settings singleton
  - reset_settings
"""

from __future__ import annotations

import pytest

from shared.config import Settings, get_settings, reset_settings


@pytest.fixture(autouse=True)
def reset_cfg():
    reset_settings()
    yield
    reset_settings()


class TestSettingsDefaults:
    def test_default_environment_is_development(self):
        s = Settings()
        assert s.environment == "development"

    def test_default_port_is_8080(self):
        s = Settings()
        assert s.port == 8080

    def test_default_alert_threshold(self):
        s = Settings()
        assert s.alert_threshold == 3

    def test_default_minor_alert_threshold(self):
        s = Settings()
        assert s.minor_alert_threshold == 2

    def test_minor_threshold_less_than_adult_threshold(self):
        s = Settings()
        assert s.minor_alert_threshold < s.alert_threshold


class TestCorsOriginsListProperty:
    def test_single_origin(self):
        s = Settings(cors_allowed_origins="http://localhost:3000")
        assert s.cors_origins_list == ["http://localhost:3000"]

    def test_multiple_origins(self):
        s = Settings(cors_allowed_origins="http://a.com,http://b.com")
        assert s.cors_origins_list == ["http://a.com", "http://b.com"]

    def test_strips_whitespace(self):
        s = Settings(cors_allowed_origins="  http://a.com  ,  http://b.com  ")
        assert "http://a.com" in s.cors_origins_list
        assert "http://b.com" in s.cors_origins_list


class TestIsProduction:
    def test_development_is_not_production(self):
        s = Settings(environment="development")
        assert s.is_production is False

    def test_production_is_production(self):
        s = Settings(
            environment="production",
            supabase_url="https://proj.supabase.co",
            supabase_service_role_key="svc-key",
            supabase_jwt_secret="jwt-secret",
            google_api_key="google-key",
            openai_api_key="openai-key",
            deepgram_api_key="deepgram-key",
            elevenlabs_api_key="elevenlabs-key",
            rate_limit_hash_salt="salt",
            oauth_state_secret="oauth-secret",
            internal_api_key="internal-key",
            cors_allowed_origins="https://app.example.com",
            frontend_base_url="https://app.example.com",
            google_oauth_redirect_uri="https://api.example.com/auth/google/callback",
        )
        assert s.is_production is True


class TestProductionValidation:
    def test_missing_required_fields_raises_value_error(self):
        with pytest.raises(ValueError, match="Missing required"):
            Settings(
                environment="production",
                supabase_url="",  # missing
                supabase_service_role_key="svc",
                supabase_jwt_secret="jwt",
                google_api_key="gk",
                openai_api_key="ok",
                deepgram_api_key="dk",
                elevenlabs_api_key="ek",
                rate_limit_hash_salt="salt",
                oauth_state_secret="oauth",
                internal_api_key="internal",
                cors_allowed_origins="https://app.example.com",
                frontend_base_url="https://app.example.com",
                google_oauth_redirect_uri="https://api.example.com/callback",
            )

    def test_localhost_in_production_raises_value_error(self):
        with pytest.raises(ValueError, match="localhost"):
            Settings(
                environment="production",
                supabase_url="https://proj.supabase.co",
                supabase_service_role_key="svc-key",
                supabase_jwt_secret="jwt-secret",
                google_api_key="google-key",
                openai_api_key="openai-key",
                deepgram_api_key="deepgram-key",
                elevenlabs_api_key="elevenlabs-key",
                rate_limit_hash_salt="salt",
                oauth_state_secret="oauth-secret",
                internal_api_key="internal-key",
                cors_allowed_origins="http://localhost:3000",  # should fail
                frontend_base_url="https://app.example.com",
                google_oauth_redirect_uri="https://api.example.com/callback",
            )


class TestGetSettingsSingleton:
    def test_returns_same_instance(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reset_creates_new_instance(self):
        s1 = get_settings()
        reset_settings()
        s2 = get_settings()
        assert s1 is not s2
