"""Utility helpers for AI Vision Assistant Pro."""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pandas as pd
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
DATABASE_DIR = BASE_DIR / "database"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
APP_TIMEZONE = ZoneInfo("Asia/Kolkata")


def local_now() -> datetime:
    """Return the current application time in Asia/Kolkata."""
    return datetime.now(APP_TIMEZONE)


def ensure_directories() -> None:
    """Create required project directories if they do not exist."""
    for directory in [ASSETS_DIR, OUTPUTS_DIR, REPORTS_DIR, DATABASE_DIR, SAMPLE_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def create_session_id(prefix: str = "session") -> str:
    """Return a readable unique session id."""
    stamp = local_now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:6]}"


def current_timestamp() -> str:
    """Return timestamp text suitable for database rows and reports."""
    return local_now().strftime("%Y-%m-%d %H:%M:%S IST")


def timestamp_to_ist(value: object) -> str:
    """Convert stored UTC/legacy timestamps to readable India time."""
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "nat"}:
        return ""
    if text.endswith(" IST"):
        return text

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return text
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed.tz_convert(APP_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S IST")


def dataframe_timestamps_to_ist(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with all known timestamp columns normalized to IST."""
    converted = df.copy()
    for column in ("timestamp", "started_at", "ended_at"):
        if column in converted.columns:
            converted[column] = converted[column].map(timestamp_to_ist)
    return converted


def image_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to OpenCV BGR format."""
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR frame to a PIL image."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def save_uploaded_file(uploaded_file, destination_dir: Path) -> Path:
    """Save a Streamlit uploaded file and return its local path."""
    ensure_directories()
    safe_name = uploaded_file.name.replace(" ", "_")
    path = destination_dir / f"{local_now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
    with open(path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return path


def crowd_density_level(person_count: int) -> str:
    """Classify crowd density using the final-year project rules."""
    if person_count <= 0:
        return "No Crowd"
    if 1 <= person_count <= 5:
        return "Low Crowd"
    if 6 <= person_count <= 15:
        return "Medium Crowd"
    if 16 <= person_count <= 30:
        return "High Crowd"
    return "Very High Crowd"


def generate_scene_summary(object_counts: Dict[str, int], crowd_level: str) -> str:
    """Generate a rule-based scene summary without paid APIs."""
    if not object_counts:
        return "No major objects were detected in the scene. Try lowering the confidence threshold or using a clearer image."

    sorted_items = sorted(object_counts.items(), key=lambda item: item[1], reverse=True)
    parts = []
    for name, count in sorted_items[:6]:
        label = name if count == 1 else f"{name}s"
        parts.append(f"{count} {label}")

    if len(parts) == 1:
        object_text = parts[0]
    else:
        object_text = ", ".join(parts[:-1]) + f", and {parts[-1]}"

    persons = object_counts.get("person", 0)
    environment = infer_environment(object_counts)
    crowd_text = crowd_level.lower()

    if persons > 0:
        return (
            f"The scene contains {object_text}. The crowd level is {crowd_text}. "
            f"This looks like {environment}."
        )
    return f"The scene contains {object_text}. No crowd is visible. This looks like {environment}."


def infer_environment(object_counts: Dict[str, int]) -> str:
    """Infer a simple scene type from detected object names."""
    indoor_objects = {"chair", "couch", "tv", "laptop", "keyboard", "mouse", "book", "clock", "dining table"}
    road_objects = {"car", "bus", "truck", "motorcycle", "bicycle", "traffic light", "stop sign"}
    food_objects = {"cup", "bottle", "fork", "knife", "spoon", "bowl", "banana", "apple", "pizza"}

    names = set(object_counts)
    if names & road_objects:
        return "an outdoor traffic or street environment"
    if names & indoor_objects:
        return "an indoor classroom, office, or room environment"
    if names & food_objects:
        return "a cafe, kitchen, or dining environment"
    return "a general real-world environment"


def format_counts(counts: Dict[str, int]) -> str:
    """Convert a count dictionary to compact readable text."""
    if not counts:
        return "No objects"
    return ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))


def path_to_download_name(path: os.PathLike) -> str:
    """Return a neat download filename for Streamlit download buttons."""
    return Path(path).name
