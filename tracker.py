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
    missing_frames: int = 0


class CentroidTracker:
    """Assign stable IDs to detections across nearby frames.

    This beginner-friendly tracker avoids heavyweight dependencies while still
    demonstrating object tracking logic for final-year presentations.
    """

    def __init__(self, max_distance: int = 120, max_missing: int = 18):
        self.max_distance = max_distance
        self.max_missing = max_missing
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
            distance_matrix = np.zeros((len(track_ids), len(centroids)), dtype=float)

            for row, track_id in enumerate(track_ids):
                for col, centroid in enumerate(centroids):
                    if self.tracks[track_id].class_name != detections[col]["class_name"]:
                        distance_matrix[row, col] = np.inf
                    else:
                        distance_matrix[row, col] = self._distance(self.tracks[track_id].centroid, centroid)

            while distance_matrix.size:
                row, col = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
                distance = distance_matrix[row, col]
                track_id = track_ids[row]

                if distance > self.max_distance:
                    break
                if row in assigned_tracks or col in assigned_detections:
                    distance_matrix[row, col] = np.inf
                    if np.isinf(distance_matrix).all():
                        break
                    continue

                self.tracks[track_id].centroid = centroids[col]
                self.tracks[track_id].class_name = detections[col]["class_name"]
                self.tracks[track_id].missing_frames = 0
                detections[col]["object_id"] = track_id
                assigned_tracks.add(row)
                assigned_detections.add(col)
                distance_matrix[row, :] = np.inf
                distance_matrix[:, col] = np.inf

                if np.isinf(distance_matrix).all():
                    break

        for index, detection in enumerate(detections):
            if index not in assigned_detections:
                object_id = self.next_object_id
                self.next_object_id += 1
                self.tracks[object_id] = Track(object_id, centroids[index], detection["class_name"])
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
