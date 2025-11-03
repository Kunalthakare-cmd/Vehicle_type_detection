import cv2
import time
from ultralytics import YOLO
from collections import defaultdict

# Load your trained YOLOv8 model
model = YOLO("D:/Kunal Files/Sem 5/AI/project_new/best.pt")

# Define colors for each class
colors = {
    "car": (255, 0, 0),     # Blue
    "bus": (0, 0, 255),     # Red
    "truck": (0, 255, 0),   # Green
    "bicycle": (0, 165, 255)   # Orange
}

# Store previous positions for speed estimation
prev_positions = defaultdict(lambda: None)

# Open webcam
cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 with tracking (gives IDs)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    # Vehicle count dictionary
    vehicle_count = {"car": 0, "bus": 0, "truck": 0, "bicycle": 0}

    # Loop through tracked objects
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            class_name = model.names[class_id].lower()

            if class_name in vehicle_count:
                vehicle_count[class_name] += 1

            # Pick color for class
            color = colors.get(class_name, (255, 255, 255))

            # Draw rectangle + ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name.upper()} ID:{track_id} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # --- SPEED CALCULATION ---
            cx = int((x1 + x2) / 2)   # center point
            cy = int((y1 + y2) / 2)

            if prev_positions[track_id] is not None:
                prev_x, prev_y, prev_time_id = prev_positions[track_id]
                curr_time = time.time()
                dt = curr_time - prev_time_id

                # Distance in pixels
                dist_px = ((cx - prev_x) ** 2 + (cy - prev_y) ** 2) ** 0.5

                # Convert pixels to meters (assume 10px = 1m → adjust as needed)
                dist_m = dist_px / 10.0
                if dt > 0:
                    speed = (dist_m / dt) * 3.6  # km/h
                else:
                     speed = 0

                # cv2.putText(frame, f"Speed: {speed:.1f} km/h", (x1, y2 + 20),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            prev_positions[track_id] = (cx, cy, time.time())

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display vehicle counts
    y_offset = 60
    for v_type, count in vehicle_count.items():
        cv2.putText(frame, f"{v_type.upper()}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors.get(v_type, (255, 255, 255)), 2)
        y_offset += 30

    # Show frame
    cv2.imshow("Real-Time Vehicle Detection + Tracking + Speed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
