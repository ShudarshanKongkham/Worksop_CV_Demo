import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import yt_dlp as youtube_dl
import time
import colorsys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import queue

# Set logging level to ERROR to suppress YOLO detection outputs
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Load COCO class names
with open('coco.names', 'r') as f:
    class_names = f.read().splitlines()

class AnalyticsTracker:
    def __init__(self, max_history_length=100):
        self.max_history_length = max_history_length
        self.reset_analytics()
    
    def reset_analytics(self):
        self.detection_counts = defaultdict(int)
        self.confidence_scores = defaultdict(list)
        self.detection_history = deque(maxlen=self.max_history_length)
        self.fps_history = deque(maxlen=50)
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = time.time()
        self.total_objects_detected = 0
        self.alerts = []
        self.detection_timestamps = defaultdict(list)
    
    def update(self, detections, fps):
        current_time = time.time()
        timestamp = datetime.now()
        
        # Update frame statistics
        self.frame_count += 1
        self.fps_history.append(fps)
        
        # Count detections per class
        frame_detections = defaultdict(int)
        frame_confidences = defaultdict(list)
        
        for detection in detections:
            class_name = detection['class']
            confidence = detection['confidence']
            
            self.detection_counts[class_name] += 1
            frame_detections[class_name] += 1
            self.confidence_scores[class_name].append(confidence)
            frame_confidences[class_name].append(confidence)
            self.detection_timestamps[class_name].append(timestamp)
            self.total_objects_detected += 1
        
        # Store frame-level data
        self.detection_history.append({
            'timestamp': timestamp,
            'detections': dict(frame_detections),
            'total_objects': sum(frame_detections.values()),
            'fps': fps
        })
        
        # Check for alerts
        self._check_alerts(frame_detections, frame_confidences)
    
    def _check_alerts(self, frame_detections, frame_confidences):
        # Alert for high object density
        total_objects_in_frame = sum(frame_detections.values())
        if total_objects_in_frame > 10:
            self.alerts.append({
                'type': 'High Density',
                'message': f'{total_objects_in_frame} objects detected in single frame',
                'timestamp': datetime.now(),
                'severity': 'warning'
            })
        
        # Alert for low confidence detections
        for class_name, confidences in frame_confidences.items():
            avg_confidence = np.mean(confidences) if confidences else 0
            if avg_confidence < 0.5 and len(confidences) > 0:
                self.alerts.append({
                    'type': 'Low Confidence',
                    'message': f'{class_name} detected with low confidence ({avg_confidence:.2f})',
                    'timestamp': datetime.now(),
                    'severity': 'info'
                })
        
        # Keep only recent alerts (last 10)
        self.alerts = self.alerts[-10:]
    
    def get_summary_stats(self):
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        runtime = time.time() - self.start_time
        
        return {
            'total_frames': self.frame_count,
            'runtime': runtime,
            'avg_fps': avg_fps,
            'total_objects': self.total_objects_detected,
            'objects_per_minute': (self.total_objects_detected / (runtime / 60)) if runtime > 0 else 0,
            'unique_classes': len(self.detection_counts)
        }

def rounded_rectangle(image, pt1, pt2, color, thickness, radius, lineType=cv2.LINE_AA):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, lineType)
    cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, lineType)
    cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, lineType)
    cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, lineType)
    cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness, lineType)
    cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness, lineType)
    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness, lineType)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness, lineType)

def generate_distinct_colors(num_colors):
    """Generates a list of distinct colors using HSV space."""
    colors = []
    for i in range(num_colors):
        hue = i / num_colors  # Divide the hue circle into equal parts
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # Full saturation and value
        colors.append(tuple(int(c * 255) for c in rgb))  # Convert to 0-255 range
    return colors

def visualize_detections_and_segmentation(frame, results, model, selected_classes, class_colors, show_analytics=True):
    # Create a copy of the frame
    frame_copy = frame.copy()
    detections = []

    for result in results:
        if hasattr(result, "masks") and result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()

            for i in range(len(masks)):
                mask = masks[i]
                box = boxes[i]
                x1, y1, x2, y2 = map(int, box[:4])
                conf = float(box[4])
                cls = int(box[5])

                if class_names[cls] not in selected_classes:
                    continue

                # Store detection data for analytics
                detections.append({
                    'class': class_names[cls],
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })

                # Get color from pre-generated list based on selected classes
                class_name = class_names[cls]
                color = class_colors[selected_classes.index(class_name)]

                alpha = 0.4
                resized_mask = cv2.resize(mask, (frame_copy.shape[1], frame_copy.shape[0]))
                colored_mask = np.zeros_like(frame_copy, dtype=np.uint8)
                colored_mask[resized_mask > 0.5] = color
                frame_copy = cv2.addWeighted(frame_copy, 1, colored_mask, alpha, 0)

                thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                corner_radius = 10
                label = f"{class_names[cls]} {conf:.2f}"

                rounded_rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness, corner_radius)

                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                text_x = x1
                text_y = y1 - 10
                if text_y < text_height + baseline:
                    text_y = y2 + baseline + text_height
                padding = 5

                cv2.rectangle(frame_copy, (text_x - padding, text_y - text_height - padding - baseline),
                              (text_x + text_width + padding, text_y + padding), color, -1)
                cv2.putText(frame_copy, label, (text_x, text_y - baseline), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # Add frame-level analytics overlay
    if show_analytics and detections:
        # Count objects in current frame
        object_count = len(detections)
        avg_confidence = np.mean([d['confidence'] for d in detections])
        
        # Add overlay text
        overlay_text = f"Objects: {object_count} | Avg Conf: {avg_confidence:.2f}"
        cv2.rectangle(frame_copy, (10, 10), (400, 50), (0, 0, 0), -1)
        cv2.putText(frame_copy, overlay_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame_copy, detections

def create_analytics_dashboard(analytics):
    """Create real-time analytics dashboard"""
    stats = analytics.get_summary_stats()
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Objects", int(stats['total_objects']))
    with col2:
        st.metric("Avg FPS", f"{stats['avg_fps']:.1f}")
    with col3:
        st.metric("Runtime", f"{stats['runtime']:.1f}s")
    with col4:
        st.metric("Objects/Min", f"{stats['objects_per_minute']:.1f}")
    
    # Detection counts chart
    if analytics.detection_counts:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Detection Counts")
            counts_df = pd.DataFrame(list(analytics.detection_counts.items()), 
                                   columns=['Class', 'Count'])
            fig_bar = px.bar(counts_df, x='Class', y='Count', 
                           title="Objects Detected by Class")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.subheader("Confidence Distribution")
            conf_data = []
            for class_name, confidences in analytics.confidence_scores.items():
                for conf in confidences[-20:]:  # Last 20 detections
                    conf_data.append({'Class': class_name, 'Confidence': conf})
            
            if conf_data:
                conf_df = pd.DataFrame(conf_data)
                fig_box = px.box(conf_df, x='Class', y='Confidence',
                               title="Confidence Score Distribution")
                st.plotly_chart(fig_box, use_container_width=True)
    
    # Time series charts
    if len(analytics.detection_history) > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Objects Over Time")
            time_data = []
            for entry in list(analytics.detection_history)[-50:]:  # Last 50 frames
                time_data.append({
                    'Time': entry['timestamp'],
                    'Objects': entry['total_objects']
                })
            
            if time_data:
                time_df = pd.DataFrame(time_data)
                fig_line = px.line(time_df, x='Time', y='Objects',
                                 title="Objects Detected Over Time")
                st.plotly_chart(fig_line, use_container_width=True)
        
        with col2:
            st.subheader("FPS Monitor")
            fps_data = []
            for i, fps in enumerate(list(analytics.fps_history)[-30:]):  # Last 30 FPS readings
                fps_data.append({'Frame': i, 'FPS': fps})
            
            if fps_data:
                fps_df = pd.DataFrame(fps_data)
                fig_fps = px.line(fps_df, x='Frame', y='FPS',
                                title="Real-time FPS")
                st.plotly_chart(fig_fps, use_container_width=True)

def display_alerts(analytics):
    """Display recent alerts"""
    if analytics.alerts:
        st.subheader("🔔 Recent Alerts")
        for alert in reversed(analytics.alerts[-5:]):  # Show last 5 alerts
            timestamp = alert['timestamp'].strftime("%H:%M:%S")
            if alert['severity'] == 'warning':
                st.warning(f"**{alert['type']}** ({timestamp}): {alert['message']}")
            else:
                st.info(f"**{alert['type']}** ({timestamp}): {alert['message']}")

def get_best_video_url(url):
    """Gets the best video URL using yt-dlp."""
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'noplaylist': True,
        'quiet': True,
        'extract_flat': True,
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
             info_dict = ydl.extract_info(url, download=False)
             if 'entries' in info_dict:
                 video_url = info_dict['entries'][0].get('url')
             else:
                 video_url = info_dict.get('url')
        return video_url

    except youtube_dl.utils.DownloadError as e:
        st.error(f"Error downloading video: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def main():
    st.set_page_config(page_title="AI Object Detection & Analytics", layout="wide")
    
    st.title("🎯 Advanced Object Detection & Real-time Analytics")
    st.markdown("*Powered by YOLO11 with intelligent monitoring and insights*")

    # Initialize analytics tracker
    if 'analytics' not in st.session_state:
        st.session_state.analytics = AnalyticsTracker()

    model_name = "yolo11n-seg.pt"  # Change this to the desired model {n: nano, s: small, m: medium, l: large, xl: extra large}
    model = YOLO(model_name)
    print(f"Running model: {model_name}")

    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Class selection
        selected_classes = st.multiselect("Select Classes to Detect", class_names, default=[class_names[0]])
        class_colors = generate_distinct_colors(len(selected_classes))

        # Source selection
        source_type = st.radio("Select Input Source:", ["Webcam", "Stream URL", "YouTube URL"])

        # Analytics options
        st.header("📊 Analytics Options")
        show_live_analytics = st.checkbox("Show Live Analytics", value=True)
        show_frame_overlay = st.checkbox("Show Frame Overlay", value=True)
        
        # Reset analytics button
        if st.button("🔄 Reset Analytics"):
            st.session_state.analytics.reset_analytics()
            st.success("Analytics reset!")

        if source_type == "Stream URL":
            stream_url = st.text_input("Enter Stream URL:")
        elif source_type == "YouTube URL":
            youtube_url = st.text_input("Enter YouTube URL:")
            if youtube_url:
                with st.spinner("Getting video stream..."):
                    stream_url = get_best_video_url(youtube_url)
                    if not stream_url:
                        return
            else:
                stream_url = None
        else:
            stream_url = 0

    # Main layout - Video on top, analytics below
    st.header("📹 Live Detection")
    frame_placeholder = st.empty()
    stop_button_pressed = st.button("⏹️ Stop Detection", type="primary")
    
    # Analytics section below video
    st.header("📈 Live Analytics")
    analytics_placeholder = st.empty()
    alerts_placeholder = st.empty()

    # Video capture setup
    if source_type in ("Stream URL", "YouTube URL") and stream_url:
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            st.error(f"Could not open video source: {stream_url}")
            return
    elif source_type == "Webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam.")
            return
    else:
        cap = None

    # Main detection loop
    if cap:
        fps_start_time = time.time()
        frame_counter = 0
        
        while cap.isOpened() and not stop_button_pressed:
            ret, frame = cap.read()
            if not ret:
                if source_type in ("Stream URL", "YouTube URL"):
                    st.error("Stream ended or encountered an error.")
                else:
                    st.error("Could not read frame from webcam.")
                break

            # Calculate FPS
            frame_counter += 1
            if frame_counter % 10 == 0:  # Update FPS every 10 frames
                fps = 10 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            else:
                fps = 0

            # Process frame
            results = model(frame, stream=True)
            frame_processed, detections = visualize_detections_and_segmentation(
                frame, results, model, selected_classes, class_colors, show_frame_overlay
            )
            
            # Update analytics
            if fps > 0:  # Only update when we have a valid FPS reading
                st.session_state.analytics.update(detections, fps)
            
            # Display frame at full width (let Streamlit handle the sizing)
            frame_placeholder.image(cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

            # Update analytics dashboard
            if show_live_analytics and frame_counter % 5 == 0:  # Update analytics every 5 frames
                with analytics_placeholder.container():
                    create_analytics_dashboard(st.session_state.analytics)
                
                with alerts_placeholder.container():
                    display_alerts(st.session_state.analytics)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    cv2.destroyAllWindows()

    # Final analytics summary
    if st.session_state.analytics.frame_count > 0:
        st.header("📊 Session Summary")
        create_analytics_dashboard(st.session_state.analytics)

if __name__ == "__main__":
    main()