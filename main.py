import cv2
import numpy as np
import torch

# Load YOLOv5 model (YOLOv5 from Ultralytics is widely used)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Define the class names (COCO dataset)
class_names = model.names


# Detect animals in video frames
def detect_animals(frame):
    # Convert frame to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Use YOLO model to perform detection
    results = model(img)
    
    # Extract detection results
    detections = results.xyxy[0].cpu().numpy()
    return detections


# Track animals
def initialize_trackers(frame, detections):
    trackers = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if class_names[int(cls)] in ['cat', 'dog']:  # Adjust based on your target animals
            if hasattr(cv2, 'TrackerCSRT_create'):
                tracker = cv2.TrackerCSRT_create()
            else:
                tracker = cv2.legacy.TrackerCSRT.create()
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            tracker.init(frame, bbox)
            trackers.append(tracker)
    return trackers

def update_trackers(trackers, frame):
    boxes = []
    for tracker in trackers:
        success, box = tracker.update(frame)
        if success:
            boxes.append(box)
    return boxes




# Pet Tracking
def main(video_path):
    cap = cv2.VideoCapture(video_path)
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    detections = detect_animals(frame)
    trackers = initialize_trackers(frame, detections)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        boxes = update_trackers(trackers, frame)
        
        for box in boxes:
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        
        # Display the frame with the tracked objects
        cv2.imshow('Pet Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = 'video.mp4'
    main(video_path)

