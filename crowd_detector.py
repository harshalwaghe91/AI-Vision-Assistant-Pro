"""Crowd counting and density analysis."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from utils import crowd_density_level


def count_persons(detections: List[dict]) -> int:
    """Count detections whose class is person."""
    return sum(1 for det in detections if det.get("class_name") == "person")


def object_counts(detections: List[dict]) -> Dict[str, int]:
    """Return same-object counts such as person=25 and chair=10."""
    counts: Dict[str, int] = {}
    for det in detections:
        class_name = det.get("class_name", "unknown")
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts


def different_object_count(detections: List[dict]) -> int:
    """Return number of different object classes."""
    return len({det.get("class_name") for det in detections})


def crowd_alert(person_count: int, threshold: int) -> str:
    """Return alert status for crowd threshold."""
    return "Alert" if person_count >= threshold else "Normal"


def crowd_metrics(crowd_counts: List[int]) -> dict:
    """Calculate min, max, average and current crowd statistics."""
    if not crowd_counts:
        return {
            "current": 0,
            "maximum": 0,
            "minimum": 0,
            "average": 0.0,
            "density": "No Crowd",
        }
    return {
        "current": crowd_counts[-1],
        "maximum": max(crowd_counts),
        "minimum": min(crowd_counts),
        "average": round(sum(crowd_counts) / len(crowd_counts), 2),
        "density": crowd_density_level(crowd_counts[-1]),
    }


def build_crowd_dataframe(crowd_counts: List[int]) -> pd.DataFrame:
    """Build chart-friendly crowd analytics data."""
    rows = []
    for index, count in enumerate(crowd_counts, start=1):
        rows.append(
            {
                "Frame": index,
                "Total Persons": count,
                "Crowd Density Level": crowd_density_level(count),
            }
        )
    return pd.DataFrame(rows)
