# Combined Record mood entries

Stage 1 extends the existing action and entry API. No data rewrite is required.
Existing mood values, scale metadata, labels, dates, and revision history remain readable.

## Action settings

Supply `settings.what_to_record` when activating a reviewed candidate. It defaults to
`["mood", "energy"]`. Supported values are `mood`, `energy`, `sleep_quality`, and
`motivation`. Mood is required and duplicates are rejected. Existing stored actions
without this field receive the default when read. Linking to an ongoing action with
different measurements returns a conflict rather than silently losing the selection.
Session extraction and the setup interface now support this selection (Stage 2).

## Session extraction and setup (Stage 2)

Prompt v0.5.0 produces at most one combined candidate. Mood and Energy are defaults;
Sleep quality and Motivation require explicit tracking agreement in the cited session.
The user can review all four choices before activation, with Mood required. Candidate
`what_to_record` defaults preserve readability of older saved mood candidates.

The new mobile setup sends `duration_kind: "unscheduled"` with null dates, duration,
and preferred check-in time/timezone. There is no implicit 14-day expiry. Candidate
`duration_days` remains a legacy schema field and is not applied by this setup.
Existing fixed and ongoing actions remain supported. A compatible active ongoing or
unscheduled action is linked rather than duplicated; differing measurements return
a conflict. Users can select the existing measurements to link the session, without
changing the original action. Repeating activation with changed measurements also
returns a conflict.

Mobile parsers now use backend `record_mood`, `mood_value`, and `mood_label` fields.
Earlier on-device `record_rating` session summaries remain readable, with a notice
to regenerate the summary before using the new setup. The old scheduling setup card
has been replaced. Entry wizard and Today/Journey changes remain subsequent stages.

## Entry values

Keep sending the existing date/time fields and `mood_value` / `scale_id` pair.
Additional fields are:

- `energy_value`, `sleep_quality_value`, `motivation_value`: nullable integers 0–10;
  an answered measurement must be enabled on the action.
- `feelings`: a list drawn from `Calm`, `Tired`, `Relieved`, `Anxious`.
- `custom_feeling`: optional text up to 500 characters.
- `feeling_intensity`: optional integer 0–10, requiring at least one feeling.
- `context`: optional text up to 2,000 characters, now available for every action.
- `current_step`: `mood`, `additional_ratings`, `feelings`, `intensity`, `context`,
  or `review`. Defaults to `mood` for older records.

Unanswered ratings are null, never a default slider number; zero is a valid answer.
Optional answers can be omitted. Confirmation rejects entries with no answers.

## Save and resume

Use one client-generated entry UUID throughout the wizard:

1. `PUT /actions/{action_id}/entries/{entry_id}/draft` with `values` and, on
   subsequent saves, the returned `expected_revision`.
2. `GET /actions/{action_id}/entries` returns drafts and confirmed entries. Resume
   a draft using its values, current step, ID, and revision.
3. `POST /actions/{action_id}/entries/{entry_id}/confirm` with the full values,
   expected revision, and a stable `idempotency_key`. Retrying the same confirmation
   for the same entry returns the saved result.
4. `PATCH /actions/{action_id}/entries/{entry_id}` amends a confirmed entry with
   full values and a required expected revision. Previous answers remain in
   `revision_history`; the original `confirmed_at` is preserved.

Drafts must be filtered out of charts and completed check-ins. Export already includes
only confirmed entries. Saves use the existing atomic filesystem storage; mutations
are serialized within one service process. Multi-process storage coordination is not
provided by this filesystem backend.

Legacy scheduling and scale fields remain supported for current clients. The new
mobile entry wizard is a subsequent stage.
