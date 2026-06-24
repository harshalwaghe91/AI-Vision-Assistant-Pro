"""SQLite storage for detections, crowd logs, and sessions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from utils import DATABASE_DIR, current_timestamp, dataframe_timestamps_to_ist


DB_PATH = DATABASE_DIR / "vision_assistant.db"


def get_connection() -> sqlite3.Connection:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db() -> None:
    """Create application tables."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                session_type TEXT,
                started_at TEXT,
                ended_at TEXT,
                report_path TEXT,
                notes TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                session_id TEXT,
                frame_number INTEGER,
                object_name TEXT,
                object_id INTEGER,
                confidence REAL,
                x1 INTEGER,
                y1 INTEGER,
                x2 INTEGER,
                y2 INTEGER
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS crowd_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                session_id TEXT,
                frame_number INTEGER,
                total_persons INTEGER,
                same_object_counts TEXT,
                different_object_count INTEGER,
                crowd_density_level TEXT,
                alert_status TEXT
            )
            """
        )
        conn.commit()


def create_session(session_id: str, session_type: str, notes: str = "") -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions (session_id, session_type, started_at, ended_at, report_path, notes)
            VALUES (?, ?, ?, NULL, NULL, ?)
            """,
            (session_id, session_type, current_timestamp(), notes),
        )
        conn.commit()


def end_session(session_id: str, report_path: Optional[str] = None) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = ?, report_path = COALESCE(?, report_path) WHERE session_id = ?",
            (current_timestamp(), report_path, session_id),
        )
        conn.commit()


def insert_detections(session_id: str, frame_number: int, detections: Iterable[dict]) -> None:
    rows = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        rows.append(
            (
                current_timestamp(),
                session_id,
                frame_number,
                det.get("class_name"),
                det.get("object_id"),
                det.get("confidence"),
                x1,
                y1,
                x2,
                y2,
            )
        )
    if not rows:
        return
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO detections
            (timestamp, session_id, frame_number, object_name, object_id, confidence, x1, y1, x2, y2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def insert_crowd_log(
    session_id: str,
    frame_number: int,
    total_persons: int,
    same_object_counts: Dict[str, int],
    different_object_count: int,
    crowd_density_level: str,
    alert_status: str,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO crowd_logs
            (timestamp, session_id, frame_number, total_persons, same_object_counts, different_object_count, crowd_density_level, alert_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                current_timestamp(),
                session_id,
                frame_number,
                total_persons,
                json.dumps(same_object_counts),
                different_object_count,
                crowd_density_level,
                alert_status,
            ),
        )
        conn.commit()


def fetch_dataframe(table_name: str, limit: int = 5000) -> pd.DataFrame:
    allowed = {"detections", "crowd_logs", "sessions"}
    if table_name not in allowed:
        raise ValueError("Unknown table requested.")
    with get_connection() as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT ?", conn, params=(limit,))
    return dataframe_timestamps_to_ist(df)


def fetch_sessions() -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM sessions ORDER BY started_at DESC", conn)
    return dataframe_timestamps_to_ist(df)


def fetch_detection_history(limit: int = 1000) -> pd.DataFrame:
    return fetch_dataframe("detections", limit)


def fetch_crowd_history(limit: int = 1000) -> pd.DataFrame:
    return fetch_dataframe("crowd_logs", limit)


def fetch_detections_by_session(session_id: str) -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM detections WHERE session_id = ? ORDER BY frame_number, id",
            conn,
            params=(session_id,),
        )
    return dataframe_timestamps_to_ist(df)


def fetch_crowd_logs_by_session(session_id: str) -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM crowd_logs WHERE session_id = ? ORDER BY frame_number, id",
            conn,
            params=(session_id,),
        )
    return dataframe_timestamps_to_ist(df)


def fetch_latest_open_session_id(session_type: str) -> Optional[str]:
    """Return the newest unfinished session for a workflow."""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT session_id
            FROM sessions
            WHERE session_type = ? AND ended_at IS NULL
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (session_type,),
        ).fetchone()
    return str(row[0]) if row else None


def total_sessions() -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
    return int(row[0]) if row else 0
