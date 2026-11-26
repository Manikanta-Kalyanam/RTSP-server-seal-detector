# RTSP Server Seal Detector

Dockerized CUDA-accelerated FastAPI inference service and GStreamer RTSP camera server for real-time cargo seal detection using MMDetection / MMYOLO (`PP-YOLOE`).

## Overview

This repository provides a containerized real-time video inference pipeline that ingests live RTSP, MJPEG, or local camera streams, runs `PP-YOLOE` object detection on GPU/CPU, and streams annotated MJPEG frames to a web frontend.

- **`main` branch:** FastAPI `/predict/rtsp` MJPEG inference endpoint, `Dockerfile` (`nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`), and `rtsp_server.py` (`GstRtspServer` H.264/RTP camera publisher).
- **`camera-proof` branch:** Updated universal video capture handler and pinned `mmcv==2.0.1` + Conda `environment.yml` dependencies (later evolved into `seal-detector-pro`).

## Architecture

```
.
├── app.py                  # FastAPI server exposing /predict/rtsp MJPEG stream and UI templates
├── rtsp_server.py          # GStreamer (GstRtspServer) v4l2src -> x264enc RTSP server on port 8554
├── frontend/
│   ├── index.html          # Live stream viewer and RTSP configuration UI
│   └── results.html        # Detection results view
├── Dockerfile              # CUDA 11.8 + cuDNN 8 + FFmpeg + PyTorch 2.0.1 + mmcv 2.0.1 image
├── requirements.txt        # Python dependencies (PyTorch CUDA 11.8, OpenMMLab, FastAPI)
└── environment.yml         # Conda environment definition (det-env)
```

## Tech Stack

- **Inference & CV:** PyTorch 2.0.1 (CUDA 11.8), MMCV 2.0.1, MMDetection 3.3.0, MMYOLO 0.6.0, MMEngine 0.10.7, OpenCV
- **API & Streaming:** FastAPI, Uvicorn, Jinja2, GStreamer (`GstRtspServer` + `x264enc`)
- **Containerization:** Docker (`nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`), FFmpeg

## Key Features

- **Live MJPEG Inference Stream (`/predict/rtsp`):** Reads frames from an RTSP feed with reduced buffer latency (`CAP_PROP_BUFFERSIZE`), runs `PP-YOLOE` inference, draws bounding boxes (`score >= 0.3`), and yields `multipart/x-mixed-replace` JPEG frames.
- **GStreamer RTSP Camera Publisher (`rtsp_server.py`):** Exposes local V4L2 camera devices (`/dev/video0` or `/dev/video2`) as a zero-latency H.264 RTSP stream at `rtsp://127.0.0.1:8554/stream`.
- **Reproducible CUDA Docker Build:** Pre-installs PyTorch 2.0.1 CUDA 11.8 wheels, `mmcv==2.0.1`, and FFmpeg dependencies in a single container image.

## Setup & Usage

1. **Place model config and weights:**
   - Config: `configs/ppyoloe_localised/ppyoloeloc.py`
   - Checkpoint: `models/ppyoloe_localised/epoch_100.pth`

2. **Run with Docker:**
   ```bash
   docker build -t rtsp-seal-detector .
   docker run --gpus all -p 5000:5000 rtsp-seal-detector
   ```

3. **Optional — Publish a local webcam over RTSP:**
   ```bash
   python3 rtsp_server.py
   ```
   Then open `http://localhost:5000` and connect to `rtsp://127.0.0.1:8554/stream`.
