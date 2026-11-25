"""Evaluate raw model action extraction on fictional inputs; never read/write saved sessions.

Run explicitly with: uv run python -m scripts.evaluate_action_prompt --repeats 2
This calls the configured OpenAI model and may incur API charges. Ordinary pytest tests mock it.
"""

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic
from uuid import uuid4

from app.core.config import Settings
from app.services.transcription import OpenAISessionInsightsGenerator

ROOT = Path(__file__).resolve().parents[1]


def evaluate_case(generator, prompt: str, case: dict, repetition: int) -> dict:
    recording_id = uuid4()
    segments = [
        {"id": f"S{index}", "origin": "transcript", **turn}
        for index, turn in enumerate(case["turns"], start=10)
    ]
    data = {
        "session_id": str(recording_id),
        "source_version": f"synthetic:{recording_id}",
        "session_date": "2026-11-25",
        "coverage": "complete_for_input",
        "preferred_variant": None,
        "output_language": "en",
        "role_attribution": "model_inferred",
        "segments": segments,
    }
    started = monotonic()
    try:
        # Deliberately bypass SessionInsightsService's recovery/filtering and persistence.
        result = generator.generate(recording_id, prompt, data)
        types = [item.type for item in result.action_candidates]
        issues = []
        if Counter(types) != Counter(case["expected_types"]):
            issues.append("wrong_types_or_duplicate")
        source = {turn["id"]: turn["text"] for turn in segments}
        for item in result.action_candidates:
            if not item.source_ids or not set(item.source_ids) <= source.keys():
                issues.append("invalid_source_ids")
            elif not any(item.source_excerpt in source[key] for key in item.source_ids):
                issues.append("unquoted_evidence")
            if (
                item.type == "record_mood"
                and "expected_measurements" in case
                and set(item.what_to_record) != set(case["expected_measurements"])
            ):
                issues.append("wrong_measurements")
            if (
                item.type == "practise_relaxation_breathing"
                and "expected_practice_minutes" in case
                and item.duration_minutes != case["expected_practice_minutes"]
            ):
                issues.append("wrong_practice_duration")
        return {
            "case": case["name"],
            "repetition": repetition,
            "passed": not issues,
            "types": types,
            "issues": issues,
            "duration_ms": round((monotonic() - started) * 1000),
        }
    except Exception as error:
        return {
            "case": case["name"],
            "repetition": repetition,
            "passed": False,
            "error_type": getattr(error, "provider_error_type", type(error).__name__),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 5))
    args = parser.parse_args()
    settings = Settings()
    prompt = (ROOT / "app/prompts/session_insights_v0_6.txt").read_text()
    cases = json.loads((ROOT / "examples/action-prompt-eval.json").read_text())
    generator = OpenAISessionInsightsGenerator(
        settings.openai_api_key,
        settings.openai_insights_model,
        settings.openai_insights_reasoning_effort,
    )
    # Initialize once before sharing the read-only client across evaluation workers.
    client = generator._get_client()
    results = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(evaluate_case, generator, prompt, case, repetition)
                for repetition in range(1, args.repeats + 1)
                for case in cases
            ]
            for future in futures:
                result = future.result()
                results.append(result)
                print(json.dumps(result), flush=True)
    finally:
        client.close()
    passed = sum(item["passed"] for item in results)
    print(
        json.dumps(
            {"passed": passed, "total": len(results), "model": settings.openai_insights_model}
        )
    )
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
