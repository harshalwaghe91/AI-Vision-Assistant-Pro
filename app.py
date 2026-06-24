"""AI Vision Assistant Pro Streamlit application."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / "database" / "ultralytics"))

import cv2
import pandas as pd
import plotly.io as pio
import streamlit as st
from av import VideoFrame
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from analytics import (
    aggregate_object_counts_from_history,
    crowd_line_chart,
    dashboard_metrics,
    density_trend_chart,
    object_count_chart,
    object_distribution_chart,
)
from crowd_detector import (
    build_crowd_dataframe,
    crowd_alert,
    crowd_metrics,
    different_object_count,
    object_counts,
)
from database import (
    create_session,
    end_session,
    fetch_crowd_history,
    fetch_crowd_logs_by_session,
    fetch_detections_by_session,
    fetch_detection_history,
    fetch_latest_open_session_id,
    fetch_sessions,
    init_db,
    insert_crowd_log,
    insert_detections,
    total_sessions,
)
from detector import MODEL_OPTIONS, draw_detections, run_detection, save_detected_image, summarize_detections
from report_generator import generate_csv_report, generate_excel_report, generate_pdf_report
from tracker import CentroidTracker
from utils import (
    OUTPUTS_DIR,
    create_session_id,
    crowd_density_level,
    cv2_to_pil,
    ensure_directories,
    format_counts,
    generate_scene_summary,
    image_to_cv2,
    path_to_download_name,
    save_uploaded_file,
)


st.set_page_config(
    page_title="AI Vision Assistant Pro",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

pio.templates.default = "plotly_dark"
ensure_directories()
init_db()
LOGO_PATH = "assets/ai_vision_logo.png"
YOLO_INFERENCE_LOCK = threading.Lock()
APP_BUILD = "Stable person tracking v7"
LIVE_WEBRTC_KEY = "ai-vision-browser-camera"


def apply_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #070a12;
            --panel: #101827;
            --panel-2: #162033;
            --ink: #edf4ff;
            --muted: #9fafc5;
            --line: rgba(148, 163, 184, 0.18);
            --cyan: #38bdf8;
            --green: #34d399;
            --amber: #f59e0b;
            --rose: #fb7185;
        }
        .stApp {
            background:
                radial-gradient(circle at 18% 5%, rgba(56, 189, 248, 0.16), transparent 26%),
                radial-gradient(circle at 78% 0%, rgba(52, 211, 153, 0.13), transparent 24%),
                linear-gradient(135deg, #050812 0%, #0d1320 48%, #070a12 100%);
            color: var(--ink);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #060914 0%, #0b1020 100%);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] img {
            border: 1px solid rgba(56, 189, 248, 0.22);
            border-radius: 8px;
        }
        .block-container {
            padding-top: 1.4rem;
            max-width: 1400px;
        }
        .hero {
            position: relative;
            overflow: hidden;
            padding: 34px 34px 28px;
            border: 1px solid rgba(56, 189, 248, 0.24);
            background:
                linear-gradient(120deg, rgba(16, 24, 39, 0.96), rgba(22, 32, 51, 0.82)),
                repeating-linear-gradient(90deg, rgba(148, 163, 184, 0.08) 0 1px, transparent 1px 76px);
            border-radius: 8px;
            margin: 10px 0 18px;
            box-shadow: 0 22px 80px rgba(0, 0, 0, 0.28);
        }
        .hero:after {
            content: "";
            position: absolute;
            width: 360px;
            height: 360px;
            right: -120px;
            top: -130px;
            border: 1px solid rgba(56, 189, 248, 0.22);
            transform: rotate(20deg);
            opacity: 0.8;
        }
        .eyebrow {
            color: var(--green);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 12px;
        }
        .hero h1 {
            margin: 0;
            font-size: clamp(2rem, 4vw, 4.15rem);
            line-height: 1.02;
            letter-spacing: 0;
            max-width: 940px;
        }
        .hero p {
            color: var(--muted);
            font-size: 1.08rem;
            margin: 16px 0 0;
            max-width: 900px;
            line-height: 1.65;
        }
        .hero-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 22px;
        }
        .hero-pill {
            border: 1px solid rgba(148, 163, 184, 0.20);
            background: rgba(2, 6, 23, 0.35);
            border-radius: 999px;
            padding: 9px 13px;
            color: #dbeafe;
            font-size: 0.9rem;
        }
        .section-title {
            margin: 26px 0 12px;
        }
        .section-title h2 {
            margin: 0;
            font-size: 1.45rem;
            letter-spacing: 0;
        }
        .section-title p {
            margin: 6px 0 0;
            color: var(--muted);
        }
        .card {
            border: 1px solid var(--line);
            background: linear-gradient(180deg, rgba(16, 24, 39, 0.94), rgba(12, 18, 31, 0.9));
            border-radius: 8px;
            padding: 18px;
            min-height: 148px;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }
        .card h3 {
            font-size: 1.03rem;
            margin: 0 0 8px 0;
            color: #f8fafc;
        }
        .card p {
            color: var(--muted);
            margin: 0;
            line-height: 1.5;
        }
        .feature-index {
            display: inline-flex;
            width: 30px;
            height: 30px;
            align-items: center;
            justify-content: center;
            color: #07111f;
            background: linear-gradient(135deg, var(--green), var(--cyan));
            border-radius: 8px;
            font-weight: 900;
            margin-bottom: 12px;
        }
        .cockpit {
            border: 1px solid rgba(56, 189, 248, 0.24);
            background: #07111f;
            border-radius: 8px;
            overflow: hidden;
        }
        .cockpit-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 14px;
            background: #0e1727;
            border-bottom: 1px solid var(--line);
            color: #dbeafe;
            font-weight: 800;
        }
        .dot {
            display: inline-block;
            width: 9px;
            height: 9px;
            border-radius: 50%;
            margin-right: 7px;
            background: var(--green);
        }
        .vision-frame {
            position: relative;
            min-height: 350px;
            background:
                linear-gradient(rgba(56, 189, 248, 0.08) 1px, transparent 1px),
                linear-gradient(90deg, rgba(56, 189, 248, 0.08) 1px, transparent 1px),
                linear-gradient(135deg, #0a1322, #111827);
            background-size: 34px 34px, 34px 34px, auto;
        }
        .bbox {
            position: absolute;
            border: 2px solid var(--green);
            box-shadow: 0 0 20px rgba(52, 211, 153, 0.24);
        }
        .bbox b {
            position: absolute;
            top: -28px;
            left: -2px;
            background: var(--green);
            color: #06101c;
            padding: 4px 8px;
            font-size: 0.76rem;
        }
        .bbox.cyan {
            border-color: var(--cyan);
        }
        .bbox.cyan b {
            background: var(--cyan);
        }
        .bbox.amber {
            border-color: var(--amber);
        }
        .bbox.amber b {
            background: var(--amber);
        }
        .hud {
            position: absolute;
            right: 16px;
            bottom: 16px;
            width: min(300px, 58%);
            border: 1px solid rgba(148, 163, 184, 0.24);
            background: rgba(2, 6, 23, 0.76);
            border-radius: 8px;
            padding: 14px;
        }
        .hud-row {
            display: flex;
            justify-content: space-between;
            color: #dbeafe;
            font-size: 0.88rem;
            padding: 6px 0;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        }
        .hud-row:last-child {
            border-bottom: 0;
        }
        .pipeline {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
        }
        .pipeline-step {
            border: 1px solid var(--line);
            background: rgba(16, 24, 39, 0.82);
            border-radius: 8px;
            padding: 14px;
        }
        .pipeline-step small {
            color: var(--green);
            font-weight: 800;
        }
        .pipeline-step h4 {
            margin: 8px 0 6px;
        }
        .pipeline-step p {
            color: var(--muted);
            margin: 0;
            line-height: 1.45;
        }
        .status-ok {
            color: #34d399;
            font-weight: 700;
        }
        .status-alert {
            color: #fb7185;
            font-weight: 800;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(16, 24, 39, 0.95), rgba(10, 15, 26, 0.92));
            border: 1px solid var(--line);
            padding: 14px;
            border-radius: 8px;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }
        .small-muted {
            color: var(--muted);
            font-size: 0.92rem;
        }
        @media (max-width: 900px) {
            .pipeline {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .vision-frame {
                min-height: 280px;
            }
        }
        @media (max-width: 640px) {
            .pipeline {
                grid-template-columns: 1fr;
            }
            .hero {
                padding: 24px 18px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_css()

    if Path(LOGO_PATH).exists():
        st.sidebar.image(LOGO_PATH, use_column_width=True)
    st.sidebar.title("AI Vision Assistant Pro")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "Live Detection",
            "Image Detection",
            "Video Detection",
            "Crowd Analytics",
            "Reports",
            "Detection History",
            "About Project",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("YOLO + OpenCV + Streamlit + SQLite")
    st.sidebar.caption(APP_BUILD)

    if page == "Home":
        home_page()
    elif page == "Live Detection":
        live_detection_page()
    elif page == "Image Detection":
        image_detection_page()
    elif page == "Video Detection":
        video_detection_page()
    elif page == "Crowd Analytics":
        crowd_analytics_page()
    elif page == "Reports":
        reports_page()
    elif page == "Detection History":
        detection_history_page()
    else:
        about_page()


def home_page() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Computer Vision Command Center</div>
          <h1>AI Vision Assistant Pro</h1>
          <p>Real-time YOLO detection, crowd intelligence, object tracking, evidence logs, and automated reports in one polished dashboard built for final-year project demonstrations.</p>
          <div class="hero-actions">
            <span class="hero-pill">YOLOv8 / YOLOv11 Ready</span>
            <span class="hero-pill">Live Camera Analytics</span>
            <span class="hero-pill">Crowd Density Alerts</span>
            <span class="hero-pill">CSV, Excel, PDF Reports</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    detections_df = fetch_detection_history(5000)
    crowd_df = fetch_crowd_history(5000)
    has_real_data = not detections_df.empty or not crowd_df.empty
    demo_counts = sample_object_counts()
    demo_crowd_df = sample_crowd_dataframe()

    metrics = dashboard_metrics(detections_df, crowd_df, total_sessions()) if has_real_data else {
        "Total Objects": 184,
        "Unique Object Classes": 8,
        "Total Persons": 42,
        "Max Crowd Count": 31,
        "Average Crowd Count": 17.8,
        "Total Sessions": 6,
    }
    cols = st.columns(6)
    for index, (label, value) in enumerate(metrics.items()):
        cols[index].metric(label, value)

    left, right = st.columns([1.1, 0.9])
    with left:
        render_vision_cockpit()
    with right:
        render_section_title("Operational Snapshot", "What your evaluator sees at a glance.")
        st.markdown(
            """
            <div class="card">
              <div class="hud-row"><span>Inference Engine</span><strong>Ultralytics YOLO</strong></div>
              <div class="hud-row"><span>Tracking Mode</span><strong>Centroid IDs</strong></div>
              <div class="hud-row"><span>Crowd Policy</span><strong>Threshold Alerts</strong></div>
              <div class="hud-row"><span>Report Layer</span><strong>CSV / Excel / PDF</strong></div>
              <div class="hud-row"><span>Storage</span><strong>SQLite Audit Trail</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.success("Presentation mode: dashboard uses live data when available and rich demo analytics before the first detection run.")

    render_section_title("AI Workflow", "A complete pipeline from camera input to professional reporting.")
    st.markdown(
        """
        <div class="pipeline">
          <div class="pipeline-step"><small>01 INPUT</small><h4>Camera, Image, Video</h4><p>Capture frames from webcam, uploaded images, or uploaded videos.</p></div>
          <div class="pipeline-step"><small>02 DETECT</small><h4>YOLO Inference</h4><p>Detect people, vehicles, devices, furniture, and common COCO classes.</p></div>
          <div class="pipeline-step"><small>03 ANALYZE</small><h4>Counts & Density</h4><p>Measure same object counts, different classes, crowd level, and alert state.</p></div>
          <div class="pipeline-step"><small>04 REPORT</small><h4>Smart Evidence</h4><p>Store history in SQLite and export CSV, Excel, and PDF summaries.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_title("Core Capabilities", "Feature set designed for practical demonstrations and project evaluation.")
    cards = [
        ("Real-Time Detection", "Webcam-based YOLO inference with FPS, confidence control, object totals, person count, and crowd alert status."),
        ("Image Intelligence", "Upload images, compare original vs detected output, count object classes, summarize the scene, and export CSV."),
        ("Video Processing", "Frame-by-frame video analytics with processed MP4 output, progress tracking, and crowd trend graphs."),
        ("Crowd Analytics", "Person-only crowd counting with low, medium, high, and very high density classification."),
        ("Object Tracking IDs", "Lightweight centroid tracking adds unique IDs to detections for a clear tracking story."),
        ("Smart Reporting", "SQLite-backed evidence history with downloadable CSV, Excel, and PDF reports for presentation handover."),
    ]
    card_cols = st.columns(3)
    for index, (title, body) in enumerate(cards):
        with card_cols[index % 3]:
            st.markdown(
                f'<div class="card"><div class="feature-index">{index + 1}</div><h3>{title}</h3><p>{body}</p></div>',
                unsafe_allow_html=True,
            )

    render_section_title("Detection Analytics Dashboard", "Live charts switch from demo data to real detection history automatically.")
    counts = aggregate_object_counts_from_history(detections_df) if has_real_data else demo_counts
    chart_crowd_df = crowd_df if has_real_data else demo_crowd_df
    left, right = st.columns(2)
    left.plotly_chart(object_count_chart(counts), use_container_width=True)
    right.plotly_chart(object_distribution_chart(counts), use_container_width=True)
    st.plotly_chart(crowd_line_chart(chart_crowd_df), use_container_width=True)


def render_section_title(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_vision_cockpit() -> None:
    st.markdown(
        """
        <div class="cockpit">
          <div class="cockpit-top">
            <span><span class="dot"></span>Live Vision Preview</span>
            <span>FPS 28.4 | ALERT NORMAL</span>
          </div>
          <div class="vision-frame">
            <div class="bbox" style="left: 9%; top: 23%; width: 17%; height: 44%;"><b>ID 12 person 0.91</b></div>
            <div class="bbox cyan" style="left: 35%; top: 31%; width: 22%; height: 32%;"><b>ID 18 laptop 0.86</b></div>
            <div class="bbox amber" style="left: 66%; top: 20%; width: 18%; height: 48%;"><b>ID 21 person 0.94</b></div>
            <div class="hud">
              <div class="hud-row"><span>Objects</span><strong>37</strong></div>
              <div class="hud-row"><span>Classes</span><strong>8</strong></div>
              <div class="hud-row"><span>Persons</span><strong>14</strong></div>
              <div class="hud-row"><span>Density</span><strong>Medium Crowd</strong></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sample_object_counts() -> dict:
    return {
        "person": 42,
        "chair": 28,
        "laptop": 17,
        "bottle": 14,
        "cell phone": 11,
        "backpack": 9,
        "book": 7,
        "cup": 6,
    }


def sample_crowd_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Frame": list(range(1, 13)),
            "Total Persons": [3, 6, 9, 13, 18, 23, 31, 27, 21, 14, 10, 7],
            "Crowd Density Level": [
                "Low Crowd",
                "Medium Crowd",
                "Medium Crowd",
                "Medium Crowd",
                "High Crowd",
                "High Crowd",
                "Very High Crowd",
                "High Crowd",
                "High Crowd",
                "Medium Crowd",
                "Medium Crowd",
                "Medium Crowd",
            ],
        }
    )


def model_controls(prefix: str):
    left, right = st.columns(2)
    model_label = left.selectbox("Model", list(MODEL_OPTIONS.keys()), key=f"{prefix}_model")
    confidence = right.slider("Confidence Threshold", 0.10, 0.90, 0.35, 0.05, key=f"{prefix}_conf")
    return MODEL_OPTIONS[model_label], confidence


class BrowserYOLOProcessor(VideoProcessorBase):
    """Process a device camera stream received through the browser."""

    def __init__(self, model_name: str, confidence: float, crowd_threshold: int, session_id: str):
        self.model_name = model_name
        self.confidence = confidence
        self.crowd_threshold = crowd_threshold
        self.tracker = CentroidTracker()
        self.session_id = session_id
        self.frame_number = 0
        self.reported_object_ids = set()
        self.object_sightings = {}
        self.started_at = time.time()
        self.state_lock = threading.Lock()
        self.latest_metrics = {
            "fps": 0.0,
            "objects": 0,
            "classes": 0,
            "persons": 0,
            "density": "No Crowd",
            "alert": "Normal",
        }

    def recv(self, frame: VideoFrame) -> VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        self.frame_number += 1

        # Cached YOLO models are shared by Streamlit sessions, so inference is
        # serialized to keep concurrent browser streams reliable.
        with YOLO_INFERENCE_LOCK:
            detections, _ = run_detection(image, self.model_name, self.confidence)

        detections = self.tracker.update(detections)
        annotated = draw_detections(image.copy(), detections)
        counts = object_counts(detections)
        persons = counts.get("person", 0)
        density = crowd_density_level(persons)
        alert = crowd_alert(persons, self.crowd_threshold)

        new_detections = []
        for detection in detections:
            object_id = detection.get("object_id")
            if object_id is None:
                continue
            self.object_sightings[object_id] = self.object_sightings.get(object_id, 0) + 1
            if (
                object_id not in self.reported_object_ids
                and self.object_sightings[object_id] >= 5
            ):
                new_detections.append(detection)
                self.reported_object_ids.add(object_id)

        if new_detections:
            insert_detections(self.session_id, self.frame_number, new_detections)
        if self.frame_number == 1 or self.frame_number % 10 == 0:
            insert_crowd_log(
                self.session_id,
                self.frame_number,
                persons,
                counts,
                different_object_count(detections),
                density,
                alert,
            )

        elapsed = max(time.time() - self.started_at, 0.01)
        with self.state_lock:
            self.latest_metrics = {
                "fps": self.frame_number / elapsed,
                "objects": len(detections),
                "classes": different_object_count(detections),
                "persons": persons,
                "density": density,
                "alert": alert,
            }

        status_color = (40, 40, 220) if alert == "Alert" else (20, 170, 90)
        cv2.rectangle(annotated, (12, 12), (360, 54), (8, 12, 24), -1)
        cv2.putText(
            annotated,
            f"Persons: {persons} | {density} | {alert}",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
        )
        return VideoFrame.from_ndarray(annotated, format="bgr24")

    def get_metrics(self) -> dict:
        with self.state_lock:
            return self.latest_metrics.copy()

    def on_ended(self) -> None:
        pass


def handle_live_camera_change() -> None:
    """Create/finalize sessions from the WebRTC component's real state."""
    context = st.session_state.get(LIVE_WEBRTC_KEY)
    if context is None:
        return

    is_playing = bool(context.state.playing)
    was_playing = bool(st.session_state.get("browser_live_was_playing", False))

    if is_playing and not was_playing:
        session_id = create_session_id("browser_live")
        st.session_state.browser_live_session_id = session_id
        st.session_state.last_live_report_paths = {}
        create_session(session_id, "Browser Live Detection")
    elif was_playing and not is_playing:
        session_id = (
            st.session_state.get("browser_live_session_id")
            or fetch_latest_open_session_id("Browser Live Detection")
        )
        if session_id:
            st.session_state.pending_live_report_session_id = session_id

    st.session_state.browser_live_was_playing = is_playing


def request_native_webrtc_stop() -> None:
    """Tell the browser component to stop and preserve the session for reports."""
    session_id = (
        st.session_state.get("browser_live_session_id")
        or fetch_latest_open_session_id("Browser Live Detection")
    )
    if session_id:
        st.session_state.pending_live_report_session_id = session_id
    st.session_state.force_live_camera_stop = True


def live_detection_page() -> None:
    st.title("Live Detection")
    st.caption("Works on phones, tablets, and computers. Allow camera access when your browser asks.")
    model_name, confidence = model_controls("live")
    threshold = st.slider("Crowd Alert Threshold", 1, 50, 15)
    st.info(
        "Start with the camera control below. The permanent Stop Live Detection "
        "button remains available above the video throughout the session."
    )
    if st.button(
        "Stop Live Detection",
        type="primary",
        use_container_width=True,
        key="persistent_stop_live_detection",
    ):
        request_native_webrtc_stop()
        st.rerun()

    session_id = st.session_state.get("browser_live_session_id")
    processor_factory = lambda: BrowserYOLOProcessor(model_name, confidence, threshold, session_id)
    force_stop = bool(st.session_state.get("force_live_camera_stop", False))
    context = webrtc_streamer(
        key=LIVE_WEBRTC_KEY,
        mode=WebRtcMode.SENDRECV,
        desired_playing_state=False if force_stop else None,
        video_processor_factory=processor_factory,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ]
        },
        media_stream_constraints={
            "video": {
                "facingMode": {"ideal": "environment"},
                "width": {"ideal": 960, "max": 960},
                "height": {"ideal": 540, "max": 540},
            },
            "audio": False,
        },
        async_processing=True,
        video_html_attrs={
            "autoPlay": True,
            "controls": False,
            "muted": True,
            "style": {
                "width": "100%",
                "maxHeight": "540px",
                "aspectRatio": "16 / 9",
                "objectFit": "contain",
                "backgroundColor": "#050812",
            },
        },
        translations={
            "start": "Start Live Detection",
            "stop": "Stop Live Detection",
            "select_device": "Select Camera",
        },
        on_change=handle_live_camera_change,
    )

    pending_session_id = st.session_state.get("pending_live_report_session_id")
    if pending_session_id and not context.state.playing:
        time.sleep(0.3)
        with st.spinner("Finalizing live detection reports..."):
            finalize_live_session(pending_session_id)
        st.session_state.pop("pending_live_report_session_id", None)
        st.session_state.browser_live_session_id = None
        st.session_state.force_live_camera_stop = False
        st.session_state.browser_live_was_playing = False
        st.success("Live detection stopped. CSV, Excel, and PDF reports are ready below.")
    elif force_stop and not context.state.playing:
        st.session_state.force_live_camera_stop = False

    if context.state.playing:
        st.success("Live detection is active. Counts and crowd status appear on the video.")
    else:
        st.caption("Start a new camera session above. Completed sessions are saved in Detection History.")

    show_last_live_report_downloads()


def filter_new_live_detections(detections: list) -> list:
    """Return detections whose tracking ID has not been saved in this live session."""
    new_detections = []
    reported_ids = st.session_state.setdefault("live_reported_object_ids", set())
    for detection in detections:
        object_id = detection.get("object_id")
        if object_id is None:
            new_detections.append(detection)
            continue
        if object_id not in reported_ids:
            new_detections.append(detection)
            reported_ids.add(object_id)
    return new_detections


def finalize_live_session(session_id: str) -> None:
    """End a live session and create downloadable reports for that run."""
    detections_df = fetch_detections_by_session(session_id)
    crowd_df = fetch_crowd_logs_by_session(session_id)

    if detections_df.empty and crowd_df.empty:
        end_session(session_id)
        st.session_state.last_live_report_paths = {}
        return

    csv_source = detections_df if not detections_df.empty else crowd_df
    csv_path = generate_csv_report(csv_source, f"live_report_{session_id}")
    excel_path = generate_excel_report(detections_df, crowd_df, f"live_report_{session_id}")

    object_counts_for_summary = {}
    if not detections_df.empty and "object_name" in detections_df:
        object_counts_for_summary = detections_df["object_name"].value_counts().to_dict()
    max_crowd = int(crowd_df["total_persons"].max()) if not crowd_df.empty else 0
    summary = generate_scene_summary(object_counts_for_summary, crowd_density_level(max_crowd))
    pdf_path = generate_pdf_report(
        detections_df,
        crowd_df,
        summary,
        {
            "Unique Saved Objects": len(detections_df),
            "Crowd Log Frames": len(crowd_df),
            "Max Crowd Count": max_crowd,
            "Session ID": session_id,
        },
        f"live_report_{session_id}",
    )

    end_session(session_id, str(pdf_path))
    st.session_state.last_live_report_paths = {
        "CSV": csv_path,
        "Excel": excel_path,
        "PDF": pdf_path,
    }


def show_last_live_report_downloads() -> None:
    paths = st.session_state.get("last_live_report_paths", {})
    if not paths:
        return
    st.subheader("Latest Live Detection Reports")
    cols = st.columns(len(paths))
    mime_types = {
        "CSV": "text/csv",
        "Excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "PDF": "application/pdf",
    }
    for index, (label, path) in enumerate(paths.items()):
        with open(path, "rb") as file:
            cols[index].download_button(
                f"Download {label}",
                file,
                file_name=path_to_download_name(path),
                mime=mime_types[label],
            )


def image_detection_page() -> None:
    st.title("Image Detection")
    model_name, confidence = model_controls("image")
    threshold = st.slider("Crowd Alert Threshold", 1, 50, 15, key="image_threshold")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded is None:
        st.info("Upload an image to run object detection.")
        return

    from PIL import Image

    pil_image = Image.open(uploaded)
    frame = image_to_cv2(pil_image)

    if st.button("Run YOLO Detection", type="primary"):
        session_id = create_session_id("image")
        create_session(session_id, "Image Detection", uploaded.name)
        with st.spinner("Detecting objects..."):
            detections, _ = run_detection(frame, model_name, confidence)
            tracker = CentroidTracker()
            detections = tracker.update(detections)
            annotated = draw_detections(frame.copy(), detections)
            counts = object_counts(detections)
            persons = counts.get("person", 0)
            density = crowd_density_level(persons)
            alert = crowd_alert(persons, threshold)
            summary = summarize_detections(detections, density)

            insert_detections(session_id, 1, detections)
            insert_crowd_log(session_id, 1, persons, counts, different_object_count(detections), density, alert)
            detected_path = save_detected_image(annotated, "image_detection")

            detection_rows = detection_rows_dataframe(session_id, 1, detections)
            csv_path = generate_csv_report(detection_rows, "image_detection_report")
            end_session(session_id, str(csv_path))

        col1, col2 = st.columns(2)
        col1.image(pil_image, caption="Original Image", use_column_width=True)
        col2.image(cv2_to_pil(annotated), caption="Detected Image", use_column_width=True)

        metric_cols = st.columns(5)
        metric_cols[0].metric("Total Objects", len(detections))
        metric_cols[1].metric("Different Classes", different_object_count(detections))
        metric_cols[2].metric("Total Persons", persons)
        metric_cols[3].metric("Crowd Density", density)
        metric_cols[4].metric("Alert Status", alert)

        st.subheader("Same Object Counting")
        st.dataframe(pd.DataFrame(list(counts.items()), columns=["Object Name", "Count"]), use_container_width=True)
        st.subheader("AI Scene Summary")
        st.success(summary["scene_summary"])

        with open(detected_path, "rb") as file:
            st.download_button("Download Detected Image", file, file_name=path_to_download_name(detected_path), mime="image/jpeg")
        with open(csv_path, "rb") as file:
            st.download_button("Download CSV Report", file, file_name=path_to_download_name(csv_path), mime="text/csv")


def video_detection_page() -> None:
    st.title("Video Detection")
    model_name, confidence = model_controls("video")
    threshold = st.slider("Crowd Alert Threshold", 1, 50, 15, key="video_threshold")
    frame_skip = st.slider("Process Every Nth Frame", 1, 10, 2)
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded is None:
        st.info("Upload a video to process it frame-by-frame.")
        return

    input_path = save_uploaded_file(uploaded, OUTPUTS_DIR)
    if st.button("Process Video", type="primary"):
        session_id = create_session_id("video")
        create_session(session_id, "Video Detection", uploaded.name)
        output_path = OUTPUTS_DIR / f"processed_{session_id}.mp4"
        tracker = CentroidTracker()
        crowd_counts = []
        detection_rows = []
        crowd_rows = []

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            st.error("Could not read the uploaded video.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        progress = st.progress(0)
        preview = st.empty()

        frame_number = 0
        last_annotated = None
        with st.spinner("Processing video..."):
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_number += 1
                if frame_number % frame_skip == 0:
                    detections, _ = run_detection(frame, model_name, confidence)
                    detections = tracker.update(detections)
                    last_annotated = draw_detections(frame.copy(), detections)
                    counts = object_counts(detections)
                    persons = counts.get("person", 0)
                    density = crowd_density_level(persons)
                    alert = crowd_alert(persons, threshold)
                    crowd_counts.append(persons)

                    insert_detections(session_id, frame_number, detections)
                    insert_crowd_log(session_id, frame_number, persons, counts, different_object_count(detections), density, alert)
                    detection_rows.extend(detection_row_dicts(session_id, frame_number, detections))
                    crowd_rows.append(
                        {
                            "Timestamp": pd.Timestamp.now(),
                            "Session ID": session_id,
                            "Frame Number": frame_number,
                            "Total Persons": persons,
                            "Same Object Counts": json.dumps(counts),
                            "Different Object Count": different_object_count(detections),
                            "Crowd Density Level": density,
                            "Alert Status": alert,
                        }
                    )
                else:
                    last_annotated = frame

                writer.write(last_annotated)
                if frame_number % 10 == 0:
                    preview.image(cv2_to_pil(last_annotated), caption=f"Processing frame {frame_number}", use_column_width=True)
                progress.progress(min(frame_number / total_frames, 1.0))

        cap.release()
        writer.release()
        progress.progress(1.0)

        detections_df = pd.DataFrame(detection_rows)
        crowd_df = pd.DataFrame(crowd_rows)
        csv_path = generate_csv_report(crowd_df, "video_crowd_report")
        end_session(session_id, str(csv_path))

        st.success("Video processing complete.")
        st.video(str(output_path))
        st.subheader("Frame-wise Crowd Count")
        st.plotly_chart(crowd_line_chart(build_crowd_dataframe(crowd_counts)), use_container_width=True)
        st.dataframe(crowd_df, use_container_width=True)

        with open(output_path, "rb") as file:
            st.download_button("Download Processed Video", file, file_name=path_to_download_name(output_path), mime="video/mp4")
        with open(csv_path, "rb") as file:
            st.download_button("Download CSV Report", file, file_name=path_to_download_name(csv_path), mime="text/csv")


def crowd_analytics_page() -> None:
    st.title("Crowd Analytics")
    st.caption("This page focuses on person counting, density classification, and threshold alerts.")
    threshold = st.slider("Crowd Threshold", 1, 80, 15)
    crowd_df = fetch_crowd_history(5000).sort_values("frame_number")

    if crowd_df.empty:
        st.info("No crowd records yet. Run image, video, or live detection first.")
        return

    counts = crowd_df["total_persons"].fillna(0).astype(int).tolist()
    stats = crowd_metrics(counts)
    cols = st.columns(5)
    cols[0].metric("Current Crowd", stats["current"])
    cols[1].metric("Maximum Crowd", stats["maximum"])
    cols[2].metric("Average Crowd", stats["average"])
    cols[3].metric("Minimum Crowd", stats["minimum"])
    cols[4].metric("Density", stats["density"])

    status = crowd_alert(stats["current"], threshold)
    if status == "Alert":
        st.error(f"Crowd alert: current count {stats['current']} crossed threshold {threshold}.")
    else:
        st.success(f"Crowd status normal: current count {stats['current']} is below threshold {threshold}.")

    left, right = st.columns(2)
    left.plotly_chart(crowd_line_chart(crowd_df), use_container_width=True)
    right.plotly_chart(density_trend_chart(crowd_df), use_container_width=True)
    st.dataframe(crowd_df, use_container_width=True)


def reports_page() -> None:
    st.title("Reports")
    detections_df = fetch_detection_history(5000)
    crowd_df = fetch_crowd_history(5000)
    metrics = dashboard_metrics(detections_df, crowd_df, total_sessions())
    counts = aggregate_object_counts_from_history(detections_df)
    summary = generate_scene_summary(counts, crowd_density_level(metrics["Max Crowd Count"]))

    st.subheader("Generate Downloadable Reports")
    col1, col2, col3 = st.columns(3)

    if col1.button("Generate CSV Report"):
        csv_path = generate_csv_report(detections_df, "detections_report")
        with open(csv_path, "rb") as file:
            st.download_button("Download CSV", file, file_name=path_to_download_name(csv_path), mime="text/csv")

    if col2.button("Generate Excel Report"):
        excel_path = generate_excel_report(detections_df, crowd_df)
        with open(excel_path, "rb") as file:
            st.download_button(
                "Download Excel",
                file,
                file_name=path_to_download_name(excel_path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    if col3.button("Generate PDF Report"):
        pdf_path = generate_pdf_report(detections_df, crowd_df, summary, metrics)
        with open(pdf_path, "rb") as file:
            st.download_button("Download PDF", file, file_name=path_to_download_name(pdf_path), mime="application/pdf")

    st.subheader("Dashboard")
    metric_cols = st.columns(6)
    for index, (label, value) in enumerate(metrics.items()):
        metric_cols[index].metric(label, value)

    left, right = st.columns(2)
    left.plotly_chart(object_count_chart(counts), use_container_width=True)
    right.plotly_chart(object_distribution_chart(counts), use_container_width=True)
    st.plotly_chart(crowd_line_chart(crowd_df), use_container_width=True)


def detection_history_page() -> None:
    st.title("Detection History")
    st.caption("All new timestamps are displayed in India Standard Time (IST, UTC+05:30).")
    history_view = st.radio(
        "History View",
        ["Detections", "Crowd Logs", "Sessions"],
        horizontal=True,
        key="history_view_selector",
    )

    if history_view == "Detections":
        df = fetch_detection_history(5000)
        show_history_table(df, "No detection records yet. Run Image, Video, or Live Detection first.")
    elif history_view == "Crowd Logs":
        df = fetch_crowd_history(5000)
        show_history_table(df, "No crowd records yet. Run a detection workflow first.")
    else:
        df = fetch_sessions()
        show_history_table(df, "No sessions yet. Start a detection workflow to create a session.")


def show_history_table(df: pd.DataFrame, empty_message: str) -> None:
    if df.empty:
        st.info(empty_message)
        return
    cleaned = df.copy()
    for column in cleaned.columns:
        cleaned[column] = cleaned[column].astype(str)
    st.dataframe(cleaned, use_container_width=True, hide_index=True)


def about_page() -> None:
    st.title("About Project")
    st.markdown(
        """
        AI Vision Assistant Pro is a final-year computer vision dashboard built with Python, YOLO, OpenCV,
        Streamlit, Plotly, Pandas, SQLite, and ReportLab.

        The system detects objects in real time, counts same and different object classes, tracks objects
        with unique IDs, analyzes crowd density, stores detection history, and generates professional reports.
        """
    )

    st.subheader("Architecture")
    st.code(
        """
        Streamlit UI
          -> detector.py: YOLO inference and annotation
          -> tracker.py: unique object IDs
          -> crowd_detector.py: person counting and density levels
          -> database.py: SQLite persistence
          -> analytics.py: Plotly charts and dashboard metrics
          -> report_generator.py: CSV, Excel, PDF exports
        """,
        language="text",
    )


def detection_row_dicts(session_id: str, frame_number: int, detections: list) -> list:
    rows = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        rows.append(
            {
                "Timestamp": pd.Timestamp.now(),
                "Session ID": session_id,
                "Frame Number": frame_number,
                "Object Name": det.get("class_name"),
                "Object ID": det.get("object_id"),
                "Confidence": det.get("confidence"),
                "X1": x1,
                "Y1": y1,
                "X2": x2,
                "Y2": y2,
            }
        )
    return rows


def detection_rows_dataframe(session_id: str, frame_number: int, detections: list) -> pd.DataFrame:
    return pd.DataFrame(detection_row_dicts(session_id, frame_number, detections))


if __name__ == "__main__":
    main()
