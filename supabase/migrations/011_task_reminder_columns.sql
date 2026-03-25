-- ============================================================================
-- BEAN AI v1 — Migration 011: Task In-Conversation Reminder Tracking
-- ============================================================================
-- Tracks when a reminder was delivered in-conversation and whether a
-- 30-minute follow-up is still pending.

ALTER TABLE public.tasks
    ADD COLUMN IF NOT EXISTS reminder_delivered_at  TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS followup_reminder_at   TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS reminder_acknowledged  BOOLEAN NOT NULL DEFAULT FALSE;

COMMENT ON COLUMN public.tasks.reminder_delivered_at IS
    'When BEAN first slipped the reminder into conversation.';
COMMENT ON COLUMN public.tasks.followup_reminder_at IS
    'Scheduled time for the 30-min follow-up if first reminder was ignored.';
COMMENT ON COLUMN public.tasks.reminder_acknowledged IS
    'Set to TRUE when user responds to the reminder.';

CREATE INDEX IF NOT EXISTS idx_tasks_followup
    ON public.tasks(followup_reminder_at)
    WHERE status IN ('pending', 'snoozed')
      AND reminder_acknowledged = FALSE
      AND followup_reminder_at IS NOT NULL;