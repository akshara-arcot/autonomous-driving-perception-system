from ultralytics import YOLO
import numpy as np
import time
import os
import cv2

from src.pipeline import process_frame
from src.lane_detection import lane_detection,lane_estimation,estimate_lane_center
from src.fusion import classify_vehicles,estimate_vehicle_center
from src.vehicle_tracking import vehicle_tracking

# Load model
model = YOLO("models/yolov8n.onnx")


# video loop
video_capture = cv2.VideoCapture('data/project_video.mp4')
frame_count=0
total_time = 0
out_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = video_capture.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('outputs/output_video_detection.mp4', out_fourcc,fps_input ,(width,height))

while True:
  ret, frame = video_capture.read()
  if not ret:
    break
  start = time.time()
  result = process_frame(frame,model)
  end = time.time()
  total_time +=(end-start)
  fps = 1/(end-start)
  frame_count += 1
  cv2.putText(result,
              f"FPS: {fps:.2f}",
              (30,200),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.8,
              (0,255,255),
              2)
  out.write(result)
  #if frame_count % 20 == 0:
    #cv2_imshow(result)
  #if frame_count == 150:
  #  break
  #cv2_imshow(result)
avg_fps = frame_count/(total_time+1e-6)
print("Average FPS:", avg_fps)
video_capture.release()
out.release()
print("Saved as output_video_detection.mp4")



