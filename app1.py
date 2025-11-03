import streamlit as st
import cv2
import numpy as np
import time
import threading
import queue
from collections import defaultdict
from datetime import datetime
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import noisereduce as nr
import pickle
import joblib
from ultralytics import YOLO

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Multimodal Vehicle Detection",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .detection-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .fusion-result {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'detection_active' not in st.session_state:
    st.session_state.detection_active = False
if 'total_counts' not in st.session_state:
    st.session_state.total_counts = defaultdict(int)
if 'crossed_vehicles' not in st.session_state:
    st.session_state.crossed_vehicles = set()
if 'yolo_prediction' not in st.session_state:
    st.session_state.yolo_prediction = {"class": None, "confidence": 0.0}
if 'audio_prediction' not in st.session_state:
    st.session_state.audio_prediction = {"class": None, "confidence": 0.0}
if 'fused_prediction' not in st.session_state:
    st.session_state.fused_prediction = {"class": None, "source": None}

# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_yolo_model(model_path):
    """Load YOLO model"""
    try:
        model = YOLO(model_path)
        st.success(f"✅ YOLO model loaded successfully from {model_path}")
        return model
    except Exception as e:
        st.error(f"❌ Error loading YOLO model: {e}")
        return None

@st.cache_resource
def load_yamnet_model():
    """Load YAMNet model from TensorFlow Hub"""
    try:
        with st.spinner("Loading YAMNet model from TensorFlow Hub..."):
            model = hub.load('https://tfhub.dev/google/yamnet/1')
        st.success("✅ YAMNet model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Error loading YAMNet model: {e}")
        return None

@st.cache_resource
def load_audio_classifier(model_path, encoder_path):
    """Load trained audio classifier and label encoder with multiple methods"""
    try:
        # Load classifier model
        classifier = tf.keras.models.load_model(model_path, compile=False)
        st.success(f"✅ Audio classifier loaded from {model_path}")
        
        # Try multiple methods to load label encoder
        label_encoder = None
        error_messages = []
        
        # Method 1: Try pickle with different protocols
        try:
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f, encoding='latin1')
            st.success(f"✅ Label encoder loaded with pickle from {encoder_path}")
        except Exception as e1:
            error_messages.append(f"Pickle method 1: {str(e1)}")
            
            # Method 2: Try joblib
            try:
                label_encoder = joblib.load(encoder_path)
                st.success(f"✅ Label encoder loaded with joblib from {encoder_path}")
            except Exception as e2:
                error_messages.append(f"Joblib method: {str(e2)}")
                
                # Method 3: Try pickle with no encoding
                try:
                    with open(encoder_path, 'rb') as f:
                        label_encoder = pickle.load(f)
                    st.success(f"✅ Label encoder loaded with standard pickle from {encoder_path}")
                except Exception as e3:
                    error_messages.append(f"Pickle method 2: {str(e3)}")
        
        # If all methods failed, create a default encoder
        if label_encoder is None:
            st.warning("⚠️ Could not load label encoder. Using default vehicle classes.")
            st.warning("Error details: " + "; ".join(error_messages))
            
            # Create a simple mock label encoder
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.array(['bus', 'car', 'truck', 'train', 'motorcycle', 'bicycle'])
            st.info(f"Using default classes: {list(label_encoder.classes_)}")
        
        return classifier, label_encoder
        
    except Exception as e:
        st.error(f"❌ Error loading audio classifier: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

# ============================================================================
# AUDIO PROCESSING CLASS
# ============================================================================
class AudioProcessor:
    def __init__(self, yamnet_model, classifier, label_encoder, 
                 sample_rate=16000, duration=1.5):
        self.yamnet_model = yamnet_model
        self.classifier = classifier
        self.label_encoder = label_encoder
        self.sample_rate = sample_rate
        self.duration = duration
        self.audio_queue = queue.Queue(maxsize=5)
        self.running = False
        self.thread = None
        self.stream = None
        
    def extract_yamnet_embeddings(self, audio):
        """Extract embeddings from audio using YAMNet"""
        try:
            # Ensure audio is float32 and in range [-1, 1]
            audio = audio.astype(np.float32)
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()  # Normalize to [-1, 1]
            
            # Get YAMNet embeddings
            scores, embeddings, spectrogram = self.yamnet_model(audio)
            
            # Average embeddings across time
            avg_embedding = np.mean(embeddings.numpy(), axis=0)
            return avg_embedding
        except Exception as e:
            print(f"Error extracting embeddings: {e}")
            return None
    
    def process_audio_chunk(self, audio_data):
        """Process audio chunk with noise reduction and classification"""
        try:
            # Apply noise reduction
            reduced_audio = nr.reduce_noise(
                y=audio_data, 
                sr=self.sample_rate,
                stationary=True,
                prop_decrease=0.8
            )
            
            # Extract YAMNet embeddings
            embeddings = self.extract_yamnet_embeddings(reduced_audio)
            
            if embeddings is not None:
                # Reshape for classifier input
                embeddings = embeddings.reshape(1, -1)
                
                # Predict vehicle class
                predictions = self.classifier.predict(embeddings, verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class_idx])
                
                # Get class name
                if hasattr(self.label_encoder, 'inverse_transform'):
                    predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
                else:
                    # Fallback to direct class access
                    predicted_class = self.label_encoder.classes_[predicted_class_idx]
                
                return {
                    "class": predicted_class.lower(),
                    "confidence": confidence
                }
        except Exception as e:
            print(f"Error processing audio: {e}")
        
        return {"class": None, "confidence": 0.0}
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio status: {status}")
        
        # Put audio data in queue (non-blocking)
        try:
            self.audio_queue.put_nowait(indata.copy().flatten())
        except queue.Full:
            pass  # Skip if queue is full
    
    def audio_processing_loop(self):
        """Main audio processing loop running in separate thread"""
        while self.running:
            try:
                # Get audio data from queue (with timeout)
                audio_data = self.audio_queue.get(timeout=0.5)
                
                # Process audio
                result = self.process_audio_chunk(audio_data)
                
                # Update session state
                st.session_state.audio_prediction = result
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing loop: {e}")
    
    def start(self):
        """Start audio processing"""
        if not self.running:
            self.running = True
            
            try:
                # Start audio stream
                self.stream = sd.InputStream(
                    channels=1,
                    samplerate=self.sample_rate,
                    blocksize=int(self.sample_rate * self.duration),
                    callback=self.audio_callback
                )
                self.stream.start()
                
                # Start processing thread
                self.thread = threading.Thread(target=self.audio_processing_loop, daemon=True)
                self.thread.start()
                
                print("Audio processing started successfully")
            except Exception as e:
                print(f"Error starting audio processing: {e}")
                st.warning(f"⚠️ Could not start audio capture: {e}")
    
    def stop(self):
        """Stop audio processing"""
        if self.running:
            self.running = False
            try:
                if self.stream:
                    self.stream.stop()
                    self.stream.close()
                if self.thread:
                    self.thread.join(timeout=2.0)
                print("Audio processing stopped")
            except Exception as e:
                print(f"Error stopping audio: {e}")

# ============================================================================
# VEHICLE TRACKER
# ============================================================================
class VehicleTracker:
    def __init__(self):
        self.tracked_objects = {}
        self.next_id = 0
        self.max_distance = 50
        
    def update(self, detections, detection_line_y):
        """Update tracker with new detections and count vehicles crossing line"""
        for detection in detections:
            x1, y1, x2, y2, class_name, conf = detection
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Check if vehicle crosses detection line
            if abs(center_y - detection_line_y) < 30:
                vehicle_key = f"{center_x}_{center_y}_{class_name}"
                if vehicle_key not in st.session_state.crossed_vehicles:
                    st.session_state.crossed_vehicles.add(vehicle_key)
                    st.session_state.total_counts[class_name] += 1

# ============================================================================
# FUSION LOGIC
# ============================================================================
def fuse_predictions(yolo_pred, audio_pred, confidence_threshold=0.5):
    """
    Intelligent fusion of YOLO and audio predictions
    
    Rules:
    1. If YOLO confidence >= threshold, use YOLO
    2. If YOLO confidence < threshold or no detection, use audio
    3. Return fused prediction with source information
    """
    yolo_conf = yolo_pred.get("confidence", 0.0)
    yolo_class = yolo_pred.get("class")
    audio_conf = audio_pred.get("confidence", 0.0)
    audio_class = audio_pred.get("class")
    
    # Rule 1: YOLO has high confidence
    if yolo_class and yolo_conf >= confidence_threshold:
        return {
            "class": yolo_class,
            "confidence": yolo_conf,
            "source": "Visual (YOLO)",
            "color": "green"
        }
    
    # Rule 2: Fall back to audio
    elif audio_class and audio_conf > 0.3:
        return {
            "class": audio_class,
            "confidence": audio_conf,
            "source": "Audio (YAMNet)",
            "color": "blue"
        }
    
    # Rule 3: No confident prediction
    else:
        return {
            "class": "Unknown",
            "confidence": 0.0,
            "source": "No Detection",
            "color": "gray"
        }

# ============================================================================
# VIDEO PROCESSING
# ============================================================================
def process_video_frame(frame, model, detection_line_y, confidence_threshold, colors):
    """Process single video frame with YOLO detection"""
    height, width = frame.shape[:2]
    line_y = int(height * detection_line_y)
    
    # Draw detection line
    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 3)
    cv2.putText(frame, "DETECTION LINE", (10, line_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # YOLO detection
    results = model(frame, conf=confidence_threshold, verbose=False)
    
    detections = []
    best_detection = {"class": None, "confidence": 0.0}
    
    for r in results:
        boxes = r.boxes.xyxy
        classes = r.boxes.cls
        confidences = r.boxes.conf

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(classes[i])
            class_name = model.names[class_id].lower()
            conf = float(confidences[i])
            
            # Map motorcycle to match audio classes
            if class_name == "motorbike":
                class_name = "motorcycle"
            
            detections.append((x1, y1, x2, y2, class_name, conf))
            
            # Track best detection
            if conf > best_detection["confidence"]:
                best_detection = {"class": class_name, "confidence": conf}
            
            # Draw bounding box
            color = colors.get(class_name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            label = f"{class_name.upper()} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Update YOLO prediction in session state
    st.session_state.yolo_prediction = best_detection
    
    return frame, detections

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.title("🚗🎧 Multimodal Vehicle Detection System")
    st.markdown("**Real-time Visual + Audio Vehicle Classification**")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Model paths - updated with your Windows paths
    yolo_model_path = st.sidebar.text_input(
        "YOLO Model Path", 
        value=r"D:\Kunal Files\Sem 5\AI\project_new\best.pt"
    )
    audio_model_path = st.sidebar.text_input(
        "Audio Classifier Path", 
        value=r"D:\Kunal Files\Sem 5\AI\project_new\vehicle_sound_classifier.keras"
    )
    encoder_path = st.sidebar.text_input(
        "Label Encoder Path", 
        value=r"D:\Kunal Files\Sem 5\AI\project_new\label_encoder.pkl"
    )
    
    # Detection parameters
    st.sidebar.subheader("Detection Parameters")
    yolo_confidence = st.sidebar.slider(
        "YOLO Confidence Threshold", 
        0.1, 1.0, 0.5, 0.05
    )
    fusion_threshold = st.sidebar.slider(
        "Fusion Threshold", 
        0.1, 1.0, 0.5, 0.05,
        help="Minimum YOLO confidence to use visual prediction"
    )
    detection_line_y = st.sidebar.slider(
        "Detection Line Position", 
        0.3, 0.7, 0.5, 0.05
    )
    
    # Camera settings
    st.sidebar.subheader("Camera Settings")
    camera_source = st.sidebar.text_input(
        "Camera Source (0 for webcam, URL for IP camera)", 
        value="0"
    )
    
    # Audio settings
    st.sidebar.subheader("Audio Settings")
    enable_audio = st.sidebar.checkbox("Enable Audio Detection", value=True)
    
    # Load models
    st.sidebar.markdown("---")
    st.sidebar.subheader("📦 Model Loading")
    
    with st.spinner("🔄 Loading models..."):
        yolo_model = load_yolo_model(yolo_model_path)
        
        if enable_audio:
            yamnet_model = load_yamnet_model()
            audio_classifier, label_encoder = load_audio_classifier(
                audio_model_path, 
                encoder_path
            )
        else:
            yamnet_model = None
            audio_classifier = None
            label_encoder = None
            st.sidebar.info("ℹ️ Audio detection disabled")
    
    # Check if minimum required models are loaded
    if not yolo_model:
        st.error("❌ YOLO model is required. Please check the path.")
        st.stop()
    
    if enable_audio and not all([yamnet_model, audio_classifier, label_encoder]):
        st.warning("⚠️ Audio models failed to load. Continuing with visual detection only.")
        enable_audio = False
    
    st.sidebar.success("✅ Required models loaded successfully!")
    
    # Initialize audio processor
    if enable_audio and 'audio_processor' not in st.session_state:
        st.session_state.audio_processor = AudioProcessor(
            yamnet_model, 
            audio_classifier, 
            label_encoder
        )
    
    # Initialize tracker
    if 'tracker' not in st.session_state:
        st.session_state.tracker = VehicleTracker()
    
    # Vehicle colors
    colors = {
        "car": (0, 255, 0),
        "bus": (255, 0, 0),
        "truck": (0, 0, 255),
        "motorcycle": (255, 0, 255),
        "bicycle": (0, 255, 255),
        "train": (255, 255, 0),
        "auto": (255, 128, 0),
    }
    
    # Control buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        start_btn = st.button("▶️ Start Detection", type="primary", use_container_width=True)
    with col_btn2:
        stop_btn = st.button("⏹️ Stop Detection", use_container_width=True)
    with col_btn3:
        reset_btn = st.button("🔄 Reset Counts", use_container_width=True)
    
    # Handle button actions
    if start_btn:
        st.session_state.detection_active = True
        if enable_audio and hasattr(st.session_state, 'audio_processor'):
            st.session_state.audio_processor.start()
        st.rerun()
    
    if stop_btn:
        st.session_state.detection_active = False
        if enable_audio and hasattr(st.session_state, 'audio_processor'):
            st.session_state.audio_processor.stop()
        st.rerun()
    
    if reset_btn:
        st.session_state.total_counts = defaultdict(int)
        st.session_state.crossed_vehicles.clear()
        st.success("✅ Counts reset!")
        time.sleep(1)
        st.rerun()
    
    # Status indicator
    status_placeholder = st.empty()
    if st.session_state.detection_active:
        audio_status = " and audio" if enable_audio else ""
        status_placeholder.success(f"🟢 **Detection Active** - Processing video{audio_status} streams")
    else:
        status_placeholder.info("🔴 **Detection Stopped** - Click 'Start Detection' to begin")
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Video Feed")
        video_placeholder = st.empty()
    
    with col2:
        st.subheader("🎯 Detection Results")
        
        # Fused prediction
        fusion_placeholder = st.empty()
        
        # Individual predictions
        st.markdown("#### Visual Detection (YOLO)")
        yolo_placeholder = st.empty()
        
        if enable_audio:
            st.markdown("#### Audio Detection (YAMNet)")
            audio_placeholder = st.empty()
        
        st.markdown("---")
        st.subheader("📊 Vehicle Counts")
        counts_placeholder = st.empty()
    
    # Main detection loop
    if st.session_state.detection_active:
        # Initialize camera
        try:
            cam_source = int(camera_source) if camera_source.isdigit() else camera_source
        except:
            cam_source = camera_source
        
        cap = cv2.VideoCapture(cam_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            st.error("❌ Cannot open camera. Please check camera source.")
            st.stop()
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while st.session_state.detection_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to capture frame")
                    break
                
                # Process frame with YOLO
                processed_frame, detections = process_video_frame(
                    frame, yolo_model, detection_line_y, 
                    yolo_confidence, colors
                )
                
                # Update tracker
                height = frame.shape[0]
                line_y = int(height * detection_line_y)
                st.session_state.tracker.update(detections, line_y)
                
                # Get predictions
                yolo_pred = st.session_state.yolo_prediction
                audio_pred = st.session_state.audio_prediction if enable_audio else {"class": None, "confidence": 0.0}
                
                # Fuse predictions
                fused = fuse_predictions(yolo_pred, audio_pred, fusion_threshold)
                st.session_state.fused_prediction = fused
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Add info overlay to frame
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Fused: {fused['class'] or 'None'}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Display frame
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Update fusion result
                with fusion_placeholder.container():
                    if fused['class'] and fused['class'] != 'Unknown':
                        st.markdown(f"""
                        <div class="fusion-result">
                            🎯 DETECTED: {fused['class'].upper()}<br>
                            <small>Source: {fused['source']} | Confidence: {fused['confidence']:.2%}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("⏳ Waiting for detection...")
                
                # Update YOLO prediction display
                with yolo_placeholder.container():
                    if yolo_pred['class']:
                        conf_class = "confidence-high" if yolo_pred['confidence'] >= 0.7 else \
                                    "confidence-medium" if yolo_pred['confidence'] >= 0.4 else \
                                    "confidence-low"
                        st.markdown(f"""
                        <div class="detection-box">
                            🚗 <strong>{yolo_pred['class'].upper()}</strong><br>
                            <span class="{conf_class}">Confidence: {yolo_pred['confidence']:.2%}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("No visual detection")
                
                # Update audio prediction display
                if enable_audio:
                    with audio_placeholder.container():
                        if audio_pred['class']:
                            conf_class = "confidence-high" if audio_pred['confidence'] >= 0.7 else \
                                        "confidence-medium" if audio_pred['confidence'] >= 0.4 else \
                                        "confidence-low"
                            st.markdown(f"""
                            <div class="detection-box">
                                🎧 <strong>{audio_pred['class'].upper()}</strong><br>
                                <span class="{conf_class}">Confidence: {audio_pred['confidence']:.2%}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("No audio detection")
                
                # Update counts
                with counts_placeholder.container():
                    total = sum(st.session_state.total_counts.values())
                    st.metric("Total Vehicles Counted", total)
                    
                    if total > 0:
                        for vehicle, count in sorted(
                            st.session_state.total_counts.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        ):
                            if count > 0:
                                pct = (count / total) * 100
                                st.write(f"**{vehicle.upper()}:** {count} ({pct:.1f}%)")
                
                # Small delay to prevent UI freezing
                time.sleep(0.01)
                
        except Exception as e:
            st.error(f"❌ Error during detection: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
        finally:
            cap.release()
            if enable_audio and hasattr(st.session_state, 'audio_processor'):
                st.session_state.audio_processor.stop()
            st.session_state.detection_active = False
    
    # Summary section
    if sum(st.session_state.total_counts.values()) > 0:
        st.markdown("---")
        st.subheader("📈 Session Summary")
        
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        
        total_vehicles = sum(st.session_state.total_counts.values())
        
        with col_sum1:
            st.metric("Total Vehicles", total_vehicles)
        
        with col_sum2:
            most_common = max(st.session_state.total_counts.items(), 
                            key=lambda x: x[1], default=("None", 0))
            st.metric("Most Common", most_common[0].upper() if most_common[1] > 0 else "N/A")
        
        with col_sum3:
            unique_types = len([c for c in st.session_state.total_counts.values() if c > 0])
            st.metric("Unique Types", unique_types)

if __name__ == "__main__":
    main()