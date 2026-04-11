from src.lane_detection import lane_detection,lane_estimation,estimate_lane_center
from src.vehicle_tracking import vehicle_tracking
from src.fusion import estimate_vehicle_center,classify_vehicles
import cv2

def process_frame(frame,model):
  original_frame = frame.copy()
  output = original_frame.copy()
  # 1. Lane detection
  image,lines = lane_detection(frame)
  left_x,right_x = lane_estimation(lines)
  lane_center = estimate_lane_center(left_x,right_x)

  # 2. Vehicle detection
  boxes,ids = vehicle_tracking(frame,model)
  vehicle_centers = estimate_vehicle_center(boxes)

  ego_count = 0
  other_count = 0

  closest_y =0
  closest_id = None


  # 3. Classification of ego lane and other lane
  ego_lane, other_lane = classify_vehicles(vehicle_centers,lane_center)

  # 4. Drawing
  for box,tracking_id in zip(boxes,ids):
    x1,y1,x2,y2 = box
    center = (x1+x2)/2
    if abs(center-lane_center) < 300:
      color = (0,0,255) # RED - ego lane
      ego_count += 1
      if tracking_id is None:
        label = "EGO"
      else:
        label =f"ID {int(tracking_id)} | EGO"
      if y2> closest_y:
        closest_y = y2
        closest_id = tracking_id
    else:
      color = (0,255,0) # GREEN - other lane
      other_count += 1
      if tracking_id is None:
        label = "OTHER"
      else:
        label = f"ID {int(tracking_id)} | OTHER"
    cv2.rectangle(output,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
    cv2.putText(output,
              label,
              (int(x1),int(y1)-10),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.5,
              (0,255,255),
              2)
  # 5. Scene Summary
  y0 = 40
  dy = 40
  cv2.putText(output,
              f"Ego: {ego_count} | Other: {other_count}",
              (30,y0),
              cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
  cv2.putText(output,
              f"Closest Vehicle: ID {int(closest_id) if closest_id else 'None'}",
              (30,y0+dy),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

  if lane_center is None:
    lane_text = "Lane Center not detected"
  else:
    lane_text = f"Lane Center: {int(lane_center)}"
  cv2.putText(output,
              lane_text,
              (30,y0+2*dy),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.8,
              (0,255,255),
              2)
  if ego_count > 0:
    status = "RISK: VEHICLE AHEAD"
  else:
    status = "PATH CLEAR"
  cv2.putText(output,
              status,
              (30,y0+3*dy),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

  cv2.line(output,(int(lane_center),0),(int(lane_center),720),(255,0,0),2)
  #cv2_imshow(output)
  return output