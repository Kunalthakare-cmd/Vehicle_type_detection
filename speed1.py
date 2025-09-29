# import cv2
# import time
# import numpy as np
# from ultralytics import YOLO
# from collections import defaultdict, deque
# import csv
# from datetime import datetime
# import os

# class VehicleTracker:
#     def __init__(self, model_path):
#         self.model = YOLO(model_path)
        
#         # Enhanced color scheme for better visibility
#         self.colors = {
#             "car": (255, 100, 50),      # Orange-Blue
#             "bus": (50, 50, 255),       # Red
#             "truck": (50, 255, 100),    # Green
#             "bicycle": (0, 255, 255),   # Yellow
#             "motorcycle": (255, 100, 255)  # Magenta
#         }
        
#         # Speed tracking
#         self.prev_positions = defaultdict(lambda: None)
#         self.vehicle_speeds = defaultdict(lambda: deque(maxlen=15))
#         self.track_history = defaultdict(lambda: deque(maxlen=30))
        
#         # Data storage for CSV export
#         self.vehicle_data = {}  # {track_id: {'class': '', 'max_speed': 0, 'avg_speed': [], 'first_seen': '', 'last_seen': ''}}
#         self.session_start_time = datetime.now()
        
#         # Configuration
#         self.px_to_meter = 0.05  # Pixel to meter ratio (adjust based on camera)
#         self.speed_limit = 60    # km/h
        
#         # FPS calculation
#         self.frame_times = deque(maxlen=30)
#         self.prev_time = time.time()
        
#     def calculate_speed(self, track_id, cx, cy, current_time):
#         """Calculate smooth vehicle speed"""
#         if self.prev_positions[track_id] is not None:
#             prev_x, prev_y, prev_time = self.prev_positions[track_id]
#             dt = current_time - prev_time
            
#             if dt > 0.05:  # Minimum time threshold for stable calculation
#                 # Calculate distance in pixels
#                 dist_px = np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
                
#                 # Convert to meters and then km/h
#                 dist_m = dist_px * self.px_to_meter
#                 speed_mps = dist_m / dt
#                 speed_kmh = speed_mps * 3.6
                
#                 # Only add valid speeds (filter out noise)
#                 if 0 < speed_kmh < 200:  # Reasonable speed range
#                     self.vehicle_speeds[track_id].append(speed_kmh)
                
#                 # Return smoothed average
#                 if len(self.vehicle_speeds[track_id]) > 0:
#                     return np.mean(self.vehicle_speeds[track_id])
        
#         return 0.0
    
#     def draw_trail(self, frame, track_id, color):
#         """Draw vehicle movement trail"""
#         points = list(self.track_history[track_id])
        
#         if len(points) > 1:
#             for i in range(1, len(points)):
#                 # Fade effect: thicker and more opaque for recent points
#                 alpha = i / len(points)
#                 thickness = int(2 + alpha * 3)
                
#                 # Draw line segment
#                 cv2.line(frame, points[i-1], points[i], color, thickness)
    
#     def draw_modern_bbox(self, frame, x1, y1, x2, y2, class_name, track_id, conf, speed, color):
#         """Draw modern-styled bounding box with info"""
        
#         # Calculate dimensions
#         width = x2 - x1
#         height = y2 - y1
#         corner_length = min(width, height) // 4
        
#         # Draw corner brackets (modern look)
#         thickness = 3
        
#         # Top-left corner
#         cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
#         cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness)
        
#         # Top-right corner
#         cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness)
#         cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness)
        
#         # Bottom-left corner
#         cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
#         cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness)
        
#         # Bottom-right corner
#         cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness)
#         cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness)
        
#         # Speed violation - full box in red
#         if speed > self.speed_limit:
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
#         # Create info background panel
#         info_texts = [
#             f"{class_name.upper()}",
#             f"ID: {track_id}",
#             f"Conf: {conf:.2f}",
#             f"Speed: {speed:.1f} km/h"
#         ]
        
#         # Calculate panel size
#         max_text_width = 0
#         text_height = 20
#         padding = 8
        
#         for text in info_texts:
#             (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#             max_text_width = max(max_text_width, tw)
        
#         panel_width = max_text_width + 2 * padding
#         panel_height = len(info_texts) * text_height + 2 * padding
        
#         # Position panel above bbox (or below if not enough space)
#         panel_x1 = x1
#         panel_y1 = y1 - panel_height - 5
        
#         if panel_y1 < 0:
#             panel_y1 = y2 + 5
        
#         panel_x2 = panel_x1 + panel_width
#         panel_y2 = panel_y1 + panel_height
        
#         # Ensure panel is within frame
#         if panel_x2 > frame.shape[1]:
#             panel_x1 = frame.shape[1] - panel_width
#             panel_x2 = frame.shape[1]
        
#         if panel_y2 > frame.shape[0]:
#             panel_y1 = frame.shape[0] - panel_height
#             panel_y2 = frame.shape[0]
        
#         # Draw semi-transparent background
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (40, 40, 40), -1)
#         cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), color, 2)
#         cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
#         # Draw text information
#         text_y = panel_y1 + padding + 15
#         for i, text in enumerate(info_texts):
#             text_color = (255, 255, 255)
            
#             # Color code speed based on limit
#             if i == 3:  # Speed line
#                 if speed > self.speed_limit:
#                     text_color = (0, 100, 255)  # Red for overspeed
#                 elif speed > self.speed_limit * 0.8:
#                     text_color = (0, 200, 255)  # Orange for warning
#                 else:
#                     text_color = (100, 255, 100)  # Green for normal
#             elif i == 0:  # Class name
#                 text_color = color
            
#             cv2.putText(frame, text, (panel_x1 + padding, text_y),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
#             text_y += text_height
        
#         # Overspeed warning badge
#         if speed > self.speed_limit:
#             badge_text = "OVERSPEED!"
#             (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
#             badge_x = x1
#             badge_y = y1 - 35
            
#             if badge_y < 25:
#                 badge_y = y2 + 25
            
#             # Warning background
#             cv2.rectangle(frame, (badge_x - 5, badge_y - th - 5), 
#                          (badge_x + tw + 5, badge_y + 5), (0, 0, 255), -1)
#             cv2.putText(frame, badge_text, (badge_x, badge_y),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
#     def save_data_to_csv(self):
#         """Save all tracked vehicle data to CSV file"""
#         if not self.vehicle_data:
#             print("\n⚠️  No vehicle data to save yet!")
#             return
        
#         # Create filename with timestamp
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"vehicle_data_{timestamp}.csv"
        
#         # Prepare data for CSV
#         csv_data = []
        
#         for track_id, data in sorted(self.vehicle_data.items()):
#             avg_speed = np.mean(data['avg_speed']) if data['avg_speed'] else 0
            
#             csv_data.append({
#                 'Vehicle_ID': track_id,
#                 'Class': data['class'].upper(),
#                 'Max_Speed_kmh': round(data['max_speed'], 2),
#                 'Avg_Speed_kmh': round(avg_speed, 2),
#                 'First_Detected': data['first_seen'],
#                 'Last_Detected': data['last_seen'],
#                 'Speed_Violations': 'YES' if data['max_speed'] > self.speed_limit else 'NO'
#             })
        
#         # Write to CSV
#         try:
#             with open(filename, 'w', newline='') as csvfile:
#                 fieldnames = ['Vehicle_ID', 'Class', 'Max_Speed_kmh', 'Avg_Speed_kmh', 
#                              'First_Detected', 'Last_Detected', 'Speed_Violations']
#                 writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
#                 writer.writeheader()
#                 writer.writerows(csv_data)
            
#             # Print summary
#             print("\n" + "="*70)
#             print(f"✅ DATA SAVED SUCCESSFULLY: {filename}")
#             print("="*70)
#             print(f"\n📊 SESSION SUMMARY:")
#             print(f"   • Total Unique Vehicles Tracked: {len(self.vehicle_data)}")
#             print(f"   • Session Duration: {datetime.now() - self.session_start_time}")
            
#             # Count by class
#             class_counts = defaultdict(int)
#             violations = 0
#             for data in self.vehicle_data.values():
#                 class_counts[data['class']] += 1
#                 if data['max_speed'] > self.speed_limit:
#                     violations += 1
            
#             print(f"\n📈 VEHICLE BREAKDOWN:")
#             for vehicle_class, count in sorted(class_counts.items()):
#                 print(f"   • {vehicle_class.upper()}: {count}")
            
#             print(f"\n⚠️  SPEED VIOLATIONS: {violations}")
#             print(f"   • Speed Limit: {self.speed_limit} km/h")
#             print("\n" + "="*70 + "\n")
            
#         except Exception as e:
#             print(f"\n❌ Error saving CSV: {e}\n")
    
#     def update_vehicle_data(self, track_id, class_name, speed):
#         """Update stored vehicle data"""
#         current_time = datetime.now().strftime("%H:%M:%S")
        
#         if track_id not in self.vehicle_data:
#             # New vehicle detected
#             self.vehicle_data[track_id] = {
#                 'class': class_name,
#                 'max_speed': speed,
#                 'avg_speed': [speed] if speed > 0 else [],
#                 'first_seen': current_time,
#                 'last_seen': current_time
#             }
#         else:
#             # Update existing vehicle data
#             self.vehicle_data[track_id]['last_seen'] = current_time
#             if speed > 0:
#                 self.vehicle_data[track_id]['avg_speed'].append(speed)
#                 if speed > self.vehicle_data[track_id]['max_speed']:
#                     self.vehicle_data[track_id]['max_speed'] = speed
#         """Draw minimal performance dashboard"""
#         h, w = frame.shape[:2]
        
#         # Create dashboard background
#         dashboard_height = 100
#         dashboard_width = 280
#         dash_x = w - dashboard_width - 10
#         dash_y = 10
        
#         # Semi-transparent background
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (dash_x, dash_y), 
#                      (dash_x + dashboard_width, dash_y + dashboard_height),
#                      (30, 30, 30), -1)
#         cv2.rectangle(overlay, (dash_x, dash_y), 
#                      (dash_x + dashboard_width, dash_y + dashboard_height),
#                      (0, 255, 255), 2)
#         cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
#         # Title
#         cv2.putText(frame, "SYSTEM STATUS", (dash_x + 10, dash_y + 25),
#                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
#         # FPS with color coding
#         fps_color = (100, 255, 100) if fps > 20 else (0, 200, 255) if fps > 10 else (0, 100, 255)
#         cv2.putText(frame, f"FPS: {fps:.1f}", (dash_x + 10, dash_y + 55),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2, cv2.LINE_AA)
        
#         # Active vehicles being tracked
#         cv2.putText(frame, f"Tracking: {active_vehicles} vehicles", 
#                    (dash_x + 10, dash_y + 85),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
#     def process_frame(self, frame):
#         """Main processing function"""
#         current_time = time.time()
        
#         # Run YOLO tracking
#         results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", 
#                                    verbose=False, conf=0.3)
        
#         active_vehicles = 0
        
#         # Process each detection
#         for result in results:
#             if result.boxes is None or len(result.boxes) == 0:
#                 continue
            
#             for box in result.boxes:
#                 # Extract box information
#                 x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
#                 class_id = int(box.cls[0].cpu().numpy())
#                 conf = float(box.conf[0].cpu().numpy())
                
#                 # Get tracking ID
#                 if box.id is not None:
#                     track_id = int(box.id[0].cpu().numpy())
#                 else:
#                     continue
                
#                 class_name = self.model.names[class_id].lower()
                
#                 # Filter for vehicle classes
#                 if class_name not in self.colors:
#                     continue
                
#                 active_vehicles += 1
                
#                 # Calculate center point
#                 cx = (x1 + x2) // 2
#                 cy = (y1 + y2) // 2
                
#                 # Store track history for trail
#                 self.track_history[track_id].append((cx, cy))
                
#                 # Get color for this class
#                 color = self.colors[class_name]
                
#                 # Draw movement trail
#                 self.draw_trail(frame, track_id, color)
                
#                 # Calculate speed
#                 speed = self.calculate_speed(track_id, cx, cy, current_time)
                
#                 # Update position for next frame
#                 self.prev_positions[track_id] = (cx, cy, current_time)
                
#                 # Store vehicle data for CSV export
#                 self.update_vehicle_data(track_id, class_name, speed)
                
#                 # Draw detection box with info
#                 self.draw_modern_bbox(frame, x1, y1, x2, y2, 
#                                      class_name, track_id, conf, speed, color)
        
#         # Calculate FPS
#         dt = current_time - self.prev_time
#         self.frame_times.append(dt)
#         self.prev_time = current_time
        
#         fps = 1.0 / np.mean(self.frame_times) if len(self.frame_times) > 0 else 0
        
#         # Draw dashboard
#         self.draw_dashboard(frame, fps, active_vehicles)
        
#         return frame

# def main():
#     print("=" * 60)
#     print("  VEHICLE CLASSIFICATION & SPEED DETECTION SYSTEM")
#     print("=" * 60)
#     print("\n  Controls:")
#     print("    Q - Quit Application")
#     print("    S - Save Data to CSV")
#     print("\n" + "=" * 60 + "\n")
    
#     # Initialize tracker
#     tracker = VehicleTracker("D:/Kunal Files/Sem 5/AI/project_new/best.pt")
    
#     # Open webcam
#     cap = cv2.VideoCapture(0)
    
#     # Set camera properties for better quality
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#     cap.set(cv2.CAP_PROP_FPS, 30)
    
#     if not cap.isOpened():
#         print("Error: Cannot open webcam")
#         return
    
#     print("System initialized successfully!")
#     print("Press 'Q' to quit | Press 'S' to save data\n")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Cannot read frame")
#             break
        
#         # Process frame
#         processed_frame = tracker.process_frame(frame)
        
#         # Display
#         cv2.imshow("Vehicle Classification & Speed Detection", processed_frame)
        
#         # Handle keyboard input
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q') or key == ord('Q'):
#             print("\nShutting down...")
#             break
    
#     # Cleanup
#     cap.release()
#     cv2.destroyAllWindows()
#     print("System closed successfully!")

# if __name__ == "__main__":
#     main()