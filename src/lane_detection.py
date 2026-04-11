import cv2
import numpy as np
def lane_detection(frame):
  # gray scale
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # Gaussian blurring to reduce noise
  gaussian_blur = cv2.GaussianBlur(src=gray, ksize=(5,5), sigmaX=0, sigmaY=0)
  # Canny Edge Detection
  edges = cv2.Canny(image=gaussian_blur, threshold1=75, threshold2=200)
  # Region of interest
  #height = frame.shape[0]
  #width = frame.shape[1]
  height,width = edges.shape
  polygons = np.array([[
      (0,height),
      (width,height),
      (int(width*0.6),int(height*0.6)),
      (int(width*0.4),int(height*0.6))]])
  mask = np.zeros_like(edges)
  cv2.fillPoly(img=mask, pts=polygons, color=255)

  cropped = cv2.bitwise_and(edges, mask)
  lines = cv2.HoughLinesP(cropped, rho=2, theta=np.pi/180, threshold=50, lines=np.array([]), minLineLength=40, maxLineGap=50)
  lines_image = np.zeros_like(frame)
  if lines is not None:
    for line in lines:
      x1, y1, x2, y2 = line.reshape(4)
      cv2.line(lines_image, (x1, y1), (x2, y2), (0,255,0),5)

  combo = cv2.addWeighted(frame, 0.8, lines_image, 1, 1)
  return combo,lines

  # Estimate left-right lanes
def lane_estimation(lines):
  left_x = []
  right_x = []
  for line in lines:
    x1,y1,x2,y2 = line.reshape(4)
    slope = (y2-y1)/(x2-x1)
    if slope < 0:
      left_x.extend([x1,x2])
    else:
      right_x.extend([x1,x2])
  return left_x,right_x

 # Estimate the lane center
def estimate_lane_center(left_x,right_x):
  if len(left_x)==0 or len(right_x) ==0:
    return None
  left_mean = np.mean(left_x)
  right_mean = np.mean(right_x)
  lane_center = (left_mean + right_mean)/2
  return lane_center




