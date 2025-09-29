import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import time
from collections import defaultdict
from datetime import datetime
import threading

# Set page configuration - simplified
st.set_page_config(
    page_title="Vehicle Detection",
    page_icon="🚗",
    layout="wide"
)

# Minimal CSS for better performance
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state - simplified
if 'total_counts' not in st.session_state:
    st.session_state.total_counts = {"car": 0, "bus": 0, "truck": 0, "auto": 0}
if 'detection_active' not in st.session_state:
    st.session_state.detection_active = False
if 'crossed_vehicles' not in st.session_state:
    st.session_state.crossed_vehicles = set()

# Simple vehicle tracking
class SimpleTracker:
    def __init__(self):
        self.next_id = 0
        self.tracked_objects = {}
        self.max_distance = 50
        
    def update(self, detections, detection_line_y):
        current_centroids = []
        new_counts = {"car": 0, "bus": 0, "truck": 0, "auto": 0}
        
        for detection in detections:
            x1, y1, x2, y2, class_name = detection
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            current_centroids.append((center_x, center_y, class_name, (x1, y1, x2, y2)))
        
        # Simple tracking and counting
        for center_x, center_y, class_name, bbox in current_centroids:
            # Check if vehicle crosses detection line
            if abs(center_y - detection_line_y) < 30:  # Near detection line
                vehicle_key = f"{center_x}_{center_y}_{class_name}"
                if vehicle_key not in st.session_state.crossed_vehicles:
                    st.session_state.crossed_vehicles.add(vehicle_key)
                    st.session_state.total_counts[class_name] += 1
                    
        return current_centroids

# Initialize tracker
if 'tracker' not in st.session_state:
    st.session_state.tracker = SimpleTracker()

# Header - simplified
st.title("🚗 Vehicle Detection System")
st.write("Real-time vehicle detection with counting")

# Sidebar - essential settings only
st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Model Path", value="D:/Kunal Files/Sem 5/AI/project_new/best.pt")
confidence_threshold = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5, 0.1)
detection_line_y = st.sidebar.slider("Detection Line", 0.3, 0.7, 0.5, 0.1)

# Load model - cached for performance
@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(model_path)
if model is None:
    st.stop()

# Colors for different vehicle types
colors = {
    "car": (0, 255, 0),     # Green
    "bus": (255, 0, 0),     # Red  
    "truck": (0, 0, 255),   # Blue
    "auto": (255, 255, 0),  # Yellow
}

# Layout - simplified
col1, col2 = st.columns([3, 1])

with col1:
    frame_placeholder = st.empty()
    
with col2:
    st.subheader("Vehicle Counts")
    counts_placeholder = st.empty()

# Control buttons
col_btn1, col_btn2, col_btn3 = st.columns(3)
with col_btn1:
    start_btn = st.button("▶️ Start", type="primary")
with col_btn2:
    stop_btn = st.button("⏹️ Stop")
with col_btn3:
    reset_btn = st.button("🔄 Reset")

# Status
status_placeholder = st.empty()

# Main processing function - optimized
def process_frame(frame, detection_line_y):
    """Lightweight frame processing"""
    height, width = frame.shape[:2]
    line_y = int(height * detection_line_y)
    
    # Draw detection line
    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)
    cv2.putText(frame, "DETECTION LINE", (10, line_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # YOLO detection
    results = model(frame, conf=confidence_threshold, verbose=False)
    
    # Process detections
    current_counts = {"car": 0, "bus": 0, "truck": 0, "auto": 0}
    detections = []
    
    for r in results:
        boxes = r.boxes.xyxy
        classes = r.boxes.cls
        confidences = r.boxes.conf

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(classes[i])
            class_name = model.names[class_id].lower()
            conf = float(confidences[i])

            if class_name in current_counts:
                current_counts[class_name] += 1
                detections.append((x1, y1, x2, y2, class_name))
                
                # Draw bounding box - simplified
                color = colors.get(class_name, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Simple label
                label = f"{class_name.upper()}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Update tracker
    st.session_state.tracker.update(detections, line_y)
    
    return frame, current_counts

# Event handlers
if start_btn:
    st.session_state.detection_active = True

if stop_btn:
    st.session_state.detection_active = False

if reset_btn:
    st.session_state.total_counts = {"car": 0, "bus": 0, "truck": 0, "auto": 0}
    st.session_state.crossed_vehicles.clear()
    st.success("Counts reset!")

# Main detection loop - optimized for performance
if st.session_state.detection_active:
    # Initialize camera with optimized settings
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS to prevent freezing
    
    if not cap.isOpened():
        st.error("Cannot open camera")
        st.stop()
    
    status_placeholder.success("🟢 Detection Active")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while st.session_state.detection_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break
            
            # Process every frame to prevent lag
            processed_frame, current_counts = process_frame(frame, detection_line_y)
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Add FPS to frame
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Total: {sum(st.session_state.total_counts.values())}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display frame
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Update counts display - simple format
            with counts_placeholder.container():
                total_vehicles = sum(st.session_state.total_counts.values())
                st.metric("Total Vehicles", total_vehicles)
                
                # Individual counts
                for vehicle, count in st.session_state.total_counts.items():
                    if count > 0:  # Only show vehicles that have been detected
                        st.write(f"**{vehicle.upper()}:** {count}")
            
            # Reduced delay for smoother video
            time.sleep(0.05)  # 20 FPS max
            
    except Exception as e:
        st.error(f"Detection error: {e}")
    finally:
        cap.release()
        st.session_state.detection_active = False
        status_placeholder.error("🔴 Detection Stopped")

# Simple summary
if sum(st.session_state.total_counts.values()) > 0:
    st.subheader("Session Summary")
    total = sum(st.session_state.total_counts.values())
    st.write(f"**Total vehicles counted:** {total}")
    
    for vehicle, count in st.session_state.total_counts.items():
        if count > 0:
            percentage = (count / total) * 100
            st.write(f"- {vehicle.upper()}: {count} ({percentage:.1f}%)")