"""Analytics helpers and Plotly dashboard charts."""

from __future__ import annotations

import json
from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def object_count_chart(counts: Dict[str, int]):
    if not counts:
        return empty_chart("No objects detected yet")
    df = pd.DataFrame({"Object": list(counts.keys()), "Count": list(counts.values())}).sort_values("Count", ascending=False)
    return px.bar(df, x="Object", y="Count", color="Object", title="Object Count")


def object_distribution_chart(counts: Dict[str, int]):
    if not counts:
        return empty_chart("No distribution available")
    df = pd.DataFrame({"Object": list(counts.keys()), "Count": list(counts.values())})
    return px.pie(df, names="Object", values="Count", title="Object Distribution", hole=0.35)


def crowd_line_chart(crowd_df: pd.DataFrame):
    if crowd_df.empty:
        return empty_chart("No crowd data available")
    x_col = "Frame" if "Frame" in crowd_df.columns else "frame_number"
    y_col = "Total Persons" if "Total Persons" in crowd_df.columns else "total_persons"
    return px.line(crowd_df, x=x_col, y=y_col, markers=True, title="Crowd Count Over Time")


def density_trend_chart(crowd_df: pd.DataFrame):
    if crowd_df.empty:
        return empty_chart("No density trend available")
    density_col = "Crowd Density Level" if "Crowd Density Level" in crowd_df.columns else "crowd_density_level"
    counts = crowd_df[density_col].value_counts().reset_index()
    counts.columns = ["Density", "Frames"]
    return px.bar(counts, x="Density", y="Frames", color="Density", title="Crowd Density Trend")


def empty_chart(title: str):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": title,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 16},
            }
        ],
    )
    return fig


def aggregate_object_counts_from_history(detections_df: pd.DataFrame) -> Dict[str, int]:
    if detections_df.empty or "object_name" not in detections_df:
        return {}
    return detections_df["object_name"].value_counts().to_dict()


def parse_same_object_counts(value: str) -> Dict[str, int]:
    try:
        return json.loads(value) if value else {}
    except json.JSONDecodeError:
        return {}


def dashboard_metrics(detections_df: pd.DataFrame, crowd_df: pd.DataFrame, session_count: int) -> dict:
    total_objects = 0 if detections_df.empty else len(detections_df)
    unique_classes = 0 if detections_df.empty else detections_df["object_name"].nunique()
    total_persons = int(crowd_df["total_persons"].max()) if not crowd_df.empty and "total_persons" in crowd_df else 0
    max_crowd = int(crowd_df["total_persons"].max()) if not crowd_df.empty and "total_persons" in crowd_df else 0
    avg_crowd = round(float(crowd_df["total_persons"].mean()), 2) if not crowd_df.empty and "total_persons" in crowd_df else 0.0
    return {
        "Total Objects": total_objects,
        "Unique Object Classes": unique_classes,
        "Total Persons": total_persons,
        "Max Crowd Count": max_crowd,
        "Average Crowd Count": avg_crowd,
        "Total Sessions": session_count,
    }
