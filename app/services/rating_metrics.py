from dataclasses import dataclass
from typing import Literal

type RatingMetric = Literal["mood", "motivation", "sleep_quality"]
type RatingConstruct = Literal[
    "overall_day_mood",
    "daily_motivation",
    "sleep_quality",
]


@dataclass(frozen=True)
class RatingMetricDefinition:
    metric: RatingMetric
    construct: RatingConstruct
    display_name: str
    question: str
    scale_id: str
    minimum_label: str
    maximum_label: str
    reference_period: str


RATING_METRICS: dict[RatingMetric, RatingMetricDefinition] = {
    "mood": RatingMetricDefinition(
        metric="mood",
        construct="overall_day_mood",
        display_name="Mood",
        question="How was your overall mood today?",
        scale_id="overall-day-mood-0-10-v1",
        minimum_label="worst",
        maximum_label="best",
        reference_period="whole day",
    ),
    "motivation": RatingMetricDefinition(
        metric="motivation",
        construct="daily_motivation",
        display_name="Motivation",
        question="How motivated did you feel today?",
        scale_id="motivation-0-10-v1",
        minimum_label="not motivated",
        maximum_label="very motivated",
        reference_period="whole day",
    ),
    "sleep_quality": RatingMetricDefinition(
        metric="sleep_quality",
        construct="sleep_quality",
        display_name="Sleep quality",
        question="How would you rate last night's sleep?",
        scale_id="sleep-quality-0-10-v1",
        minimum_label="very poor",
        maximum_label="excellent",
        reference_period="previous night",
    ),
}


def get_rating_metric(metric: RatingMetric) -> RatingMetricDefinition:
    return RATING_METRICS[metric]
