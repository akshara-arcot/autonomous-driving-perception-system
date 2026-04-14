# Autonomous Driving Perception System

A production-level machine learning system built to handle the full perception stack for autonomous driving — from raw video input to scene-level understanding.


## Overview

This project brings together some of the most effective tools in modern computer vision to build a complete, real-world autonomous driving perception pipeline. At its core, YOLOv8 handles object detection, ByteTrack keeps track of those objects across frames, and a scene reasoning module ties everything together into a coherent understanding of the driving environment.

To make this practical and fast, the inference is optimized with TensorRT using both FP16 and INT8 precision. The whole system is packaged for deployment using Docker and NVIDIA Triton, making it straightforward to scale. The goal was to build something that reflects how a real production ML system is designed and optimized.


## Key Features

- YOLOv8 object detection pipeline built for driving scenarios
- Fine-tuned on the BDD100K dataset for domain-specific accuracy
- TensorRT FP16 and INT8 optimization for low-latency inference
- FPS benchmarking across multiple backends to measure real performance gains
- Multi-backend support across PyTorch, ONNX, and TensorRT
- Fully Dockerized pipeline for reproducible, portable deployment
- NVIDIA Triton Inference Server integration for scalable serving
- Modular ML system design that makes each component easy to swap or extend


## System Architecture

```
Video Input
    |
    v
YOLOv8 Detection
    |
    v
ByteTrack Tracking
    |
    v
Lane Detection
    |
    v
Fusion Layer
    |
    v
Scene Reasoning
```





## Model Optimization

Getting a model to run accurately and getting it to run fast enough for real-time driving are two challenging tasks. TensorRT was used to close that gap, with both FP16 and INT8 quantization tested to understand the trade-off between precision and speed.

| Backend            | Precision | FPS     |
|--------------------|-----------|---------|
| PyTorch (Baseline) | FP32      | ~35 FPS |
| TensorRT           | FP16      | ~38 FPS |
| TensorRT           | INT8      | ~41 FPS |

The jump from FP16 to INT8 delivered around an 8% improvement in throughput — a meaningful gain in a latency-sensitive system — with only a minor reduction in numerical precision.


## Fine-Tuning

Instead of relying on a generic pretrained model, YOLOv8 was fine-tuned on the [BDD100K dataset](https://bdd-data.berkeley.edu/). It is a dataset built from real driving footage across different times of day, weather conditions, and road environments. 

## Results

| Metric              | Value |
|---------------------|-------|
| Baseline YOLOv8 FPS | ~35   |
| TensorRT FP16 FPS   | ~38   |
| TensorRT INT8 FPS   | ~41   |
| Fine-tuned mAP50    | ~0.4  |



## Deployment

### Docker

The pipeline is containerized using ONNX as the inference backend, which is compatible with CPU, making it easy to test, share, and deploy across different environments. Everything needed to run the system is captured in the container.

```bash
# Build the Docker image
docker build -t perception-app .

# Run the container
docker run --rm -e OMP_NUM_THREADS=6  -v "$(pwd)/outputs:/app/outputs"  perception-app
```

### NVIDIA Triton Inference Server

For GPU-accelerated serving at scale, the model is packaged in the Triton model repository format. This makes it ready for deployment in cloud environments like AWS or any NVIDIA-powered infrastructure, and sets up a clean path toward serving multiple models simultaneously in production.

```bash
# Start the Triton Inference Server
docker run --rm -it -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models
```

> **Note:** TensorRT requires an NVIDIA GPU. For CPU-only environments, the ONNX backend via Docker is recommended.Triton execution was validated at server level.


## Tech Stack

| Component              | Technology                      |
|------------------------|---------------------------------|
| Object Detection       | YOLOv8 (Ultralytics)            |
| Multi-Object Tracking  | ByteTrack                       |
| Inference Optimization | TensorRT (FP16, INT8), ONNX     |
| Model Serving          | NVIDIA Triton Inference Server  |
| Containerization       | Docker                          |
| Training Dataset       | BDD100K                         |



## License

This project is intended for research and demonstration purposes.

