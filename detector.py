"""YOLO detector wrapper for images, webcam frames, and videos."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from crowd_detector import count_persons, different_object_count, object_counts
from utils import OUTPUTS_DIR, current_timestamp, generate_scene_summary


COCO_CLASS_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag",
    27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
    32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
    36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
    45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
    55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
    70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}


MODEL_OPTIONS = {
    "YOLOv8 Nano - fastest": "yolov8n.pt",
    "YOLOv8 Small - balanced": "yolov8s.pt",
    "YOLOv11 Nano - latest": "yolo11n.pt",
}


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(model_name: str) -> YOLO:
    """Load YOLO once per model choice."""
    return YOLO(model_name)


def run_detection(frame: np.ndarray, model_name: str, confidence: float) -> Tuple[List[dict], np.ndarray]:
    """Run YOLO and return detection dictionaries plus annotated frame."""
    model = load_model(model_name)
    results = model.predict(frame, conf=confidence, verbose=False)
    detections = parse_results(results, model)
    annotated = draw_detections(frame.copy(), detections)
    return detections, annotated


def parse_results(results, model) -> List[dict]:
    """Convert Ultralytics results into plain dictionaries."""
    detections: List[dict] = []
    names = getattr(model, "names", COCO_CLASS_NAMES)
    if not results:
        return detections

    boxes = results[0].boxes
    if boxes is None:
        return detections

    for box in boxes:
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = names.get(class_id, COCO_CLASS_NAMES.get(class_id, str(class_id)))
        detections.append(
            {
                "timestamp": current_timestamp(),
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "box": (x1, y1, x2, y2),
                "object_id": None,
            }
        )
    return detections


def draw_detections(frame: np.ndarray, detections: List[dict]) -> np.ndarray:
    """Draw bounding boxes, labels, and tracking IDs."""
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        class_name = det["class_name"]
        confidence = det["confidence"]
        object_id = det.get("object_id")
        color = color_for_class(class_name)
        label = f"{class_name} {confidence:.2f}"
        if object_id is not None:
            label = f"ID {object_id} | {label}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(y1, label_size[1] + 12)
        cv2.rectangle(frame, (x1, label_y - label_size[1] - 10), (x1 + label_size[0] + 8, label_y + 4), color, -1)
        cv2.putText(frame, label, (x1 + 4, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def color_for_class(class_name: str) -> Tuple[int, int, int]:
    """Generate a stable BGR color for a class name."""
    seed = abs(hash(class_name)) % 255
    return int((seed * 3) % 255), int((seed * 7) % 255), int((seed * 11) % 255)


def summarize_detections(detections: List[dict], crowd_level: str) -> Dict[str, object]:
    """Create reusable detection summary values."""
    counts = object_counts(detections)
    persons = count_persons(detections)
    return {
        "same_object_counts": counts,
        "different_object_count": different_object_count(detections),
        "total_persons": persons,
        "total_objects": len(detections),
        "scene_summary": generate_scene_summary(counts, crowd_level),
    }


def save_detected_image(frame: np.ndarray, prefix: str = "detected_image") -> Path:
    """Save an annotated image to outputs."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / f"{prefix}_{current_timestamp().replace(':', '-').replace(' ', '_')}.jpg"
    cv2.imwrite(str(path), frame)
    return path
