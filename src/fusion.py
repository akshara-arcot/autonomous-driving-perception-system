# Estimate vehicle center
def estimate_vehicle_center(bounding_boxes):
  vehicle_centers = []
  for box in bounding_boxes:
    x1,y1,x2,y2 = box
    center = (x1+x2)/2
    vehicle_centers.append(center)
  return vehicle_centers

# Estimate  ego lane vehicle  and draw separately
def classify_vehicles(vehicle_centers,lane_center):
  ego_lane = []
  other_lane = []
  if lane_center is None:
    return ego_lane,other_lane
  threshold = 300
  for center in vehicle_centers:
    if abs(center-lane_center) < threshold:
      ego_lane.append(center)
    else:
      other_lane.append(center)
  return ego_lane,other_lane

