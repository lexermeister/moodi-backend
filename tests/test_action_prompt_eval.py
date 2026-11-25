from types import SimpleNamespace

from scripts.evaluate_action_prompt import evaluate_case

CASE = {
    "name": "mood_only",
    "turns": [{"role": "client", "text": "I will log my mood each evening."}],
    "expected_types": ["record_mood"],
}


def test_raw_prompt_evaluation_catches_wrong_type_and_duplicates():
    wrong = SimpleNamespace(
        type="practise_relaxation_breathing",
        source_ids=["S10"],
        source_excerpt="I will log my mood each evening.",
    )
    generator = SimpleNamespace(
        generate=lambda *_: SimpleNamespace(action_candidates=[wrong, wrong])
    )
    result = evaluate_case(generator, "synthetic instructions", CASE, 1)
    assert result["passed"] is False
    assert "wrong_types_or_duplicate" in result["issues"]
    assert result["types"] == ["practise_relaxation_breathing", "practise_relaxation_breathing"]


def test_prompt_evaluation_checks_source_ids_and_quote_without_repairing_output():
    wrong = SimpleNamespace(type="record_mood", source_ids=["S1"], source_excerpt="invented")
    generator = SimpleNamespace(generate=lambda *_: SimpleNamespace(action_candidates=[wrong]))
    result = evaluate_case(generator, "", CASE, 1)
    assert result["passed"] is False
    assert result["issues"] == ["invalid_source_ids"]


def test_prompt_evaluation_accepts_supported_raw_output():
    correct = SimpleNamespace(
        type="record_mood", source_ids=["S10"], source_excerpt="I will log my mood each evening."
    )
    generator = SimpleNamespace(generate=lambda *_: SimpleNamespace(action_candidates=[correct]))
    assert evaluate_case(generator, "", CASE, 1)["passed"] is True


def test_complete_prompt_examples_match_the_actual_candidate_schemas():
    import json
    from pathlib import Path

    from pydantic import TypeAdapter

    from app.schemas.session_insights import (
        EnjoyableActivityCandidate,
        PracticeCandidate,
        RecordMoodCandidate,
    )

    prompt = (
        Path(__file__).resolve().parents[1] / "app/prompts/session_insights_v0_6.txt"
    ).read_text()
    candidates = TypeAdapter(
        list[RecordMoodCandidate | EnjoyableActivityCandidate | PracticeCandidate]
    )
    for label, expected in (
        ("Mood-recording agreement only:", "record_mood"),
        ("Breathing-practice agreement only:", "practise_relaxation_breathing"),
        ("Enjoyable-activity agreement only:", "do_enjoyable_or_meaningful_activity"),
    ):
        example, _ = json.JSONDecoder().raw_decode(prompt.split(label, 1)[1].lstrip())
        parsed = candidates.validate_python(example)
        assert len(parsed) == 1
        assert parsed[0].type == expected
