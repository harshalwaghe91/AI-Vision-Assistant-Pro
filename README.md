# AI Vision Assistant Pro

Real-Time Object Detection, Crowd Analytics & Smart Reporting Dashboard

## Overview

AI Vision Assistant Pro is a professional final-year computer vision web app built with YOLO, OpenCV, Streamlit, WebRTC, SQLite, Pandas, Plotly, and ReportLab. It supports browser-camera detection, image detection, video detection, crowd density analysis, object tracking with unique IDs, detection history, and downloadable CSV, Excel, and PDF reports.

The project is designed for final-year demonstrations, GitHub portfolios, and practical computer vision learning.

## Features

- Real-time browser-camera object detection on phones, tablets, and computers
- Image upload detection with original and annotated output views
- Video upload detection with frame-by-frame processing
- Crowd detection by counting only the `person` class
- Same object counting, for example `person = 25`, `chair = 10`
- Different object counting, for example `person`, `chair`, `laptop`, `bottle` = 4 object types
- Object tracking with unique IDs using a centroid tracker
- Crowd density levels:
  - Low Crowd = 1 to 5 persons
  - Medium Crowd = 6 to 15 persons
  - High Crowd = 16 to 30 persons
  - Very High Crowd = above 30 persons
- Crowd threshold slider and alert status
- Detection analytics dashboard using Plotly
- CSV, Excel, and PDF report generation
- SQLite database for sessions, detections, and crowd logs
- Rule-based AI scene summary without paid API keys
- Streamlit Cloud, GitHub, Docker, and local machine ready

## Tech Stack

- Python 3.11
- Streamlit
- OpenCV
- Ultralytics YOLOv8 / YOLOv11
- Pandas
- NumPy
- Plotly
- SQLite
- ReportLab
- Pillow

## Architecture

```text
AI-Vision-Assistant-Pro/
|-- app.py                  Streamlit multi-page dashboard
|-- detector.py             YOLO loading, inference, annotation
|-- tracker.py              Centroid object tracker with unique IDs
|-- crowd_detector.py       Person counting, density, alerts
|-- analytics.py            Plotly charts and dashboard metrics
|-- database.py             SQLite schema and persistence helpers
|-- report_generator.py     CSV, Excel, PDF report generation
|-- utils.py                Shared helpers
|-- requirements.txt        Python dependencies
|-- Dockerfile              Container deployment
|-- assets/                 Static assets
|-- outputs/                Processed images and videos
|-- reports/                Generated reports
|-- database/               SQLite database file
|-- sample_data/            Demo media folder
```

## Installation

1. Create and activate a Python 3.11 virtual environment.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Run the app.

```bash
streamlit run app.py
```

The app will open at:

```text
http://localhost:8501
```

## Streamlit Pages

- Home
- Live Detection
- Image Detection
- Video Detection
- Crowd Analytics
- Reports
- Detection History
- About Project

## Database Tables

### detections

- Timestamp
- Session ID
- Frame Number
- Object Name
- Object ID
- Confidence
- X1
- Y1
- X2
- Y2

### crowd_logs

- Timestamp
- Session ID
- Frame Number
- Total Persons
- Same Object Counts
- Different Object Count
- Crowd Density Level
- Alert Status

### sessions

- Session ID
- Session Type
- Started At
- Ended At
- Report Path
- Notes

## Streamlit Cloud Deployment

1. Push this folder to a GitHub repository.
2. Go to Streamlit Cloud.
3. Create a new app from the repository.
4. Set the main file path to:

```text
app.py
```

5. Use Python 3.11 if the platform asks for a runtime.
6. Deploy.

The Live Detection page uses WebRTC and HTTPS, so each device can securely provide its own camera stream. Users must allow camera access in their browser. A STUN server is configured for remote connections; restrictive corporate or institutional networks may additionally require a TURN server.

## Docker Run

```bash
docker build -t ai-vision-assistant-pro .
docker run -p 8501:8501 ai-vision-assistant-pro
```

## Screenshots Placeholder

Add screenshots after running the project:

- Home dashboard
- Live detection
- Image detection result
- Video analytics result
- Crowd analytics
- PDF report

## Future Scope

- Add DeepSORT or ByteTrack for stronger tracking
- Add face blur for privacy-aware analytics
- Add restricted-area intrusion detection
- Add email report scheduling
- Add optional Gemini/OpenAI scene summary when an API key is configured
- Add role-based admin login
- Add REST API endpoints for external camera feeds
