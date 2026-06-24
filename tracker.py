"""Simple centroid-based object tracker with unique IDs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Track:
    object_id: int
    centroid: Tuple[int, int]
    class_name: str
    box: Tuple[int, int, int, int]
    missing_frames: int = 0


class CentroidTracker:
    """Assign stable IDs to detections across nearby frames.

    This beginner-friendly tracker avoids heavyweight dependencies while still
    demonstrating object tracking logic for final-year presentations.
    """

    def __init__(
        self,
        max_distance: int = 180,
        max_missing: int = 60,
        minimum_iou: float = 0.05,
    ):
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.minimum_iou = minimum_iou
        self.next_object_id = 1
        self.tracks: Dict[int, Track] = {}

    def reset(self) -> None:
        self.next_object_id = 1
        self.tracks.clear()

    def update(self, detections: List[dict]) -> List[dict]:
        """Update tracks and attach object_id to each detection."""
        if not detections:
            for track in self.tracks.values():
                track.missing_frames += 1
            self._remove_lost_tracks()
            return []

        centroids = [self._centroid(det["box"]) for det in detections]
        assigned_detections = set()
        assigned_tracks = set()

        if self.tracks:
            track_ids = list(self.tracks.keys())
            cost_matrix = np.full((len(track_ids), len(centroids)), np.inf, dtype=float)

            for row, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                for col, centroid in enumerate(centroids):
                    if track.class_name != detections[col]["class_name"]:
                        continue

                    distance = self._distance(track.centroid, centroid)
                    overlap = self._iou(track.box, detections[col]["box"])
                    recovery_distance = self.max_distance * (
                        1.0 + min(track.missing_frames, 6) * 0.12
                    )
                    if distance > recovery_distance and overlap < self.minimum_iou:
                        continue

                    # Lower is better. Overlap helps keep the same ID when a
                    # detection box changes size around a mostly stationary object.
                    cost_matrix[row, col] = (
                        distance / max(recovery_distance, 1.0)
                        + (1.0 - overlap) * 0.65
                    )

            while cost_matrix.size and np.isfinite(cost_matrix).any():
                row, col = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                track_id = track_ids[row]

                if row in assigned_tracks or col in assigned_detections:
                    cost_matrix[row, col] = np.inf
                    continue

                self.tracks[track_id].centroid = centroids[col]
                self.tracks[track_id].class_name = detections[col]["class_name"]
                self.tracks[track_id].box = detections[col]["box"]
                self.tracks[track_id].missing_frames = 0
                detections[col]["object_id"] = track_id
                assigned_tracks.add(row)
                assigned_detections.add(col)
                cost_matrix[row, :] = np.inf
                cost_matrix[:, col] = np.inf

        for index, detection in enumerate(detections):
            if index not in assigned_detections:
                object_id = self.next_object_id
                self.next_object_id += 1
                self.tracks[object_id] = Track(
                    object_id,
                    centroids[index],
                    detection["class_name"],
                    detection["box"],
                )
                detection["object_id"] = object_id

        for track_id, track in list(self.tracks.items()):
            if track_id not in [det.get("object_id") for det in detections]:
                track.missing_frames += 1

        self._remove_lost_tracks()
        return detections

    def _remove_lost_tracks(self) -> None:
        lost_ids = [track_id for track_id, track in self.tracks.items() if track.missing_frames > self.max_missing]
        for track_id in lost_ids:
            del self.tracks[track_id]

    @staticmethod
    def _centroid(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    @staticmethod
    def _distance(point_a: Tuple[int, int], point_b: Tuple[int, int]) -> float:
        return float(np.linalg.norm(np.array(point_a) - np.array(point_b)))

    @staticmethod
    def _iou(
        box_a: Tuple[int, int, int, int],
        box_b: Tuple[int, int, int, int],
    ) -> float:
        """Return intersection-over-union for two boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
        intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
        intersection = intersection_width * intersection_height
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0
