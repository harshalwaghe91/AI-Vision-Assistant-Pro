"""CSV, Excel, and PDF report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from utils import REPORTS_DIR, current_timestamp


def generate_csv_report(df: pd.DataFrame, filename_prefix: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{filename_prefix}_{safe_timestamp()}.csv"
    df.to_csv(path, index=False)
    return path


def generate_excel_report(
    detections_df: pd.DataFrame,
    crowd_df: pd.DataFrame,
    filename_prefix: str = "vision_report",
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{filename_prefix}_{safe_timestamp()}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        detections_df.to_excel(writer, index=False, sheet_name="Detections")
        crowd_df.to_excel(writer, index=False, sheet_name="Crowd Logs")
    return path


def generate_pdf_report(
    detections_df: pd.DataFrame,
    crowd_df: pd.DataFrame,
    summary: str,
    metrics: Optional[Dict[str, object]] = None,
    filename_prefix: str = "vision_report",
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{filename_prefix}_{safe_timestamp()}.pdf"

    doc = SimpleDocTemplate(str(path), pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AI Vision Assistant Pro - Detection Report", styles["Title"]))
    story.append(Paragraph(f"Generated at: {current_timestamp()}", styles["Normal"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("AI Scene Summary", styles["Heading2"]))
    story.append(Paragraph(summary or "No summary available.", styles["BodyText"]))
    story.append(Spacer(1, 12))

    if metrics:
        story.append(Paragraph("Key Metrics", styles["Heading2"]))
        metric_data = [["Metric", "Value"]] + [[key, str(value)] for key, value in metrics.items()]
        story.append(make_table(metric_data))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Recent Detection Rows", styles["Heading2"]))
    story.append(make_table(dataframe_preview(detections_df, 8)))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Recent Crowd Rows", styles["Heading2"]))
    story.append(make_table(dataframe_preview(crowd_df, 8)))

    doc.build(story)
    return path


def dataframe_preview(df: pd.DataFrame, limit: int) -> list:
    if df.empty:
        return [["Status"], ["No data available"]]
    preview = df.head(limit).copy()
    preview = preview.astype(str)
    return [preview.columns.tolist()] + preview.values.tolist()


def make_table(data: list) -> Table:
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def safe_timestamp() -> str:
    return current_timestamp().replace(":", "-").replace(" ", "_")
