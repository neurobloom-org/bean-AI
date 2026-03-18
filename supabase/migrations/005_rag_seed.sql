-- ============================================================================
-- BEAN AI v5 — RAG Seed Migration 005
-- CBT/DBT technique library for therapy agent
-- NOTE: embeddings are populated by a one-time seed script after migration.
--       Run: python scripts/seed_rag_embeddings.py
-- ============================================================================

INSERT INTO public.rag_techniques (name, description, example, category) VALUES

-- ── CBT Techniques ───────────────────────────────────────────────────────────
(
    'Reflective Listening',
    'Mirror back what the user said to show you truly heard them. Use phrases like "It sounds like you''re feeling..." or "What I''m hearing is...". Do NOT add advice or silver linings — just reflect.',
    'User: "Nobody cares about me." BEAN: "It sounds like you''re feeling really invisible and alone right now. That must be such a heavy feeling."',
    'cbt'
),
(
    'Validation',
    'Acknowledge that their feelings make complete sense given their situation. Avoid minimising, dismissing, or jumping straight to problem-solving. Make them feel truly understood first.',
    'User: "I''m so overwhelmed." BEAN: "It makes complete sense you''d feel overwhelmed — you have so much going on at once. Anyone would feel that way."',
    'cbt'
),
(
    'Open-ended Questions',
    'Ask questions that invite the user to explore and express their experience further. Avoid yes/no questions. Start with "What", "How", or "Tell me more about...".',
    '"What has that been like for you?" or "Tell me more about what happened this afternoon."',
    'cbt'
),
(
    'Normalisation',
    'Help the user understand that their emotional response is a normal, human reaction — not a sign that something is wrong with them. Reduces shame and isolation.',
    '"Feeling anxious before a big test is something almost everyone experiences — it actually shows how much you care."',
    'cbt'
),
(
    'Cognitive Defusion',
    'Help the user notice they are having a thought, rather than the thought being a fact. Use phrases like "I''m noticing I''m having the thought that..." to create distance from unhelpful beliefs.',
    'User: "I''m a failure." BEAN: "It sounds like your mind is really telling you that story right now. What''s been happening today that brought that thought up?"',
    'cbt'
),
(
    'Behavioural Activation',
    'Gently encourage the user to engage in small, meaningful activities — even when motivation is low. Frame it as an experiment, not a demand. Small actions can lift mood.',
    '"Even a 5-minute walk or texting one friend can sometimes shift how you''re feeling. Is there one tiny thing you feel like you could try?"',
    'cbt'
),
(
    'Thought Record (Gentle)',
    'Help the user examine a specific situation, the emotions it triggered, and the thoughts behind them — without judging or pressuring them to "fix" the thought.',
    '"When that happened, what was going through your mind right at that moment?"',
    'cbt'
),

-- ── DBT Techniques ───────────────────────────────────────────────────────────
(
    'TIPP (Temperature)',
    'For intense distress, suggest changing body temperature: splashing cold water on the face or holding an ice cube activates the dive reflex and quickly reduces physiological arousal.',
    '"When feelings get really overwhelming, sometimes something physical can help — like splashing cold water on your face. Has anything like that ever helped you?"',
    'dbt'
),
(
    'TIPP (Intense Exercise)',
    'Suggest brief intense movement to burn off emotional energy when distress is very high. Even 2-3 minutes of jumping jacks or fast walking can reduce emotional intensity.',
    '"Sometimes when the feelings are really big, moving your body fast for even just 2 minutes can help. Even jumping in place counts."',
    'dbt'
),
(
    'Radical Acceptance',
    'Gently invite the user to acknowledge reality as it is, without fighting it. Not the same as approval — it''s accepting what can''t be changed in this moment to reduce suffering.',
    '"Some things are outside our control, and fighting against that can make the pain even heavier. What if, just for right now, we let it be what it is?"',
    'dbt'
),
(
    'Distress Tolerance — ACCEPTS',
    'Help the user use distraction as a coping tool when they need to tolerate distress in the short term. Activities, Contributing, Comparisons, Emotions, Pushing away, Thoughts, Sensations.',
    '"Until this feeling passes a bit, is there something you could do to give your mind something else to focus on — even just for 10 minutes?"',
    'dbt'
),
(
    'Self-Soothe with Five Senses',
    'Encourage the user to engage each of the five senses in a calming way. Vision, hearing, smell, taste, touch. This grounds them in the present moment.',
    '"Is there something nearby you could pick up and just hold — feeling its texture? Or a smell you really like? Sometimes our senses can pull us back to the present."',
    'dbt'
),
(
    'Check the Facts',
    'Gently explore whether the interpretation of an event fits the actual facts. Not to dismiss feelings, but to see if the story the mind is telling matches reality.',
    '"What do you know for sure happened? And what might your mind be adding on top of that?"',
    'dbt'
),

-- ── General / Supportive ─────────────────────────────────────────────────────
(
    'Active Presence',
    'Sometimes the most powerful thing is simply being present with someone in pain — not fixing, not advising. Sit with them. Let silence be okay. Let them feel genuinely accompanied.',
    '"I''m right here with you. You don''t have to figure anything out right now."',
    'general'
),
(
    'Strengths Spotting',
    'When appropriate, gently reflect back a strength or resilience you''ve noticed in what they''ve shared — without minimising their pain or forcing positivity.',
    '"I notice that even though things have been really hard, you''re still here, still talking about it. That takes real strength."',
    'general'
),
(
    'Empathic Curiosity',
    'Express genuine, warm interest in the details of the user''s experience. This communicates care and helps the user feel worth knowing.',
    '"I really want to understand what this has been like for you. Can you tell me a bit more about what''s been going on?"',
    'general'
)

ON CONFLICT DO NOTHING;
