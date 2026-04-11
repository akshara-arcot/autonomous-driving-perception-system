# Vehicle detection and tracking function
def vehicle_tracking(frame,model):
  results = model.track(frame,persist=True,tracker = "bytetrack.yaml",conf=0.25,classes=[0,1,2,3,5,7,9,10,11,12])
  boxes= []
  ids = []
  if results[0].boxes is not None:
    xyxy = results[0].boxes.xyxy.cpu().numpy()
    if results[0].boxes.id is not None:
      tracking_id = results[0].boxes.id.cpu().numpy()
    else:
      tracking_id = [None]*len(xyxy)
    for box,id in zip(xyxy,tracking_id):
      x1,y1,x2,y2 = box
      boxes.append([x1,y1,x2,y2])
      ids.append(id)

  return boxes,ids