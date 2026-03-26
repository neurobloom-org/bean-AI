-- ============================================================================
-- BEAN AI v1 — Migration 010: Device Registry
-- Maps ESP32 device IDs to user accounts.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.devices (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id       TEXT        NOT NULL UNIQUE,   -- hardcoded on ESP32
    user_id         UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    display_name    TEXT,                          -- e.g. "Bean #1"
    last_seen_at    TIMESTAMPTZ,
    registered_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_devices_device_id ON public.devices(device_id);
CREATE INDEX IF NOT EXISTS idx_devices_user_id   ON public.devices(user_id);

ALTER TABLE public.devices ENABLE ROW LEVEL SECURITY;

-- Service role only — device registration is admin/backend only
DROP POLICY IF EXISTS "devices: service role full access" ON public.devices;
CREATE POLICY "devices: service role full access"
    ON public.devices FOR ALL TO service_role
    USING (TRUE) WITH CHECK (TRUE);

COMMENT ON TABLE public.devices IS
    'ESP32 device registry. device_id is hardcoded on the hardware.';