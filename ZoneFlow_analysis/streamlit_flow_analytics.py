import streamlit as st
import cv2
import numpy as np
import time
import torch
import yt_dlp as youtube_dl
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
from PIL import Image
import json

# Import YOLO components
try:
    from ultralytics import YOLO
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    st.error("Please ensure YOLO dependencies are installed. Check requirements.txt")
    st.stop()

# Try to import drawable canvas, provide fallback if not available
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False
    st.warning("⚠️ streamlit-drawable-canvas not available. Using manual coordinate input only.")
    with st.expander("📋 Installation Instructions"):
        st.code("""
# To fix the canvas issue, try:
pip install streamlit==1.29.0
pip install streamlit-drawable-canvas==0.9.3

# Or if that doesn't work:
pip install --force-reinstall streamlit-drawable-canvas==0.9.3
        """, language="bash")

# Set logging level to reduce noise
logging.getLogger().setLevel(logging.ERROR)

class ZoneFlowTracker:
    """Handles zone tracking and analytics"""
    def __init__(self):
        self.zones = []  # Store zones as polygons
        self.zone_counts = {}  # Track entry/exit counts for each zone
        self.object_paths = {}  # Track object paths
        self.zone_analytics = defaultdict(lambda: {'entered': 0, 'exited': 0, 'current': 0, 'history': []})
        self.frame_count = 0
        self.start_time = time.time()
        
    def add_zone(self, polygon):
        """Add a new zone polygon"""
        zone_id = len(self.zones)
        self.zones.append(polygon)
        self.zone_counts[zone_id] = {"entered": 0, "exited": 0}
        self.zone_analytics[zone_id] = {'entered': 0, 'exited': 0, 'current': 0, 'history': []}
        return zone_id
    
    def clear_zones(self):
        """Clear all zones"""
        self.zones = []
        self.zone_counts = {}
        self.zone_analytics = defaultdict(lambda: {'entered': 0, 'exited': 0, 'current': 0, 'history': []})
        self.object_paths = {}
    
    def is_inside_polygon(self, point, polygon, zone_id=None, frame_width=None, frame_height=None):
        """Check if point is inside polygon - SIMPLIFIED without coordinate transformation"""
        # Use original polygon coordinates directly - NO TRANSFORMATION
        poly = np.array(polygon, np.int32)
        return cv2.pointPolygonTest(poly, point, False) >= 0
    
    def analyze_crossing(self, track_id, path, frame_width=None, frame_height=None):
        """Analyze if object crossed zone boundaries"""
        if len(path) < 2:
            return
            
        prev_center, curr_center = path[-2], path[-1]
        
        for zone_id, zone in enumerate(self.zones):
            was_inside = self.is_inside_polygon(prev_center, zone, zone_id, frame_width, frame_height)
            is_inside = self.is_inside_polygon(curr_center, zone, zone_id, frame_width, frame_height)
            
            if is_inside and not was_inside:  # Entered
                self.zone_counts[zone_id]["entered"] += 1
                self.zone_analytics[zone_id]['entered'] += 1
                self.zone_analytics[zone_id]['current'] += 1
                self.zone_analytics[zone_id]['history'].append({
                    'timestamp': datetime.now(),
                    'action': 'entered',
                    'track_id': track_id
                })
            elif was_inside and not is_inside:  # Exited
                self.zone_counts[zone_id]["exited"] += 1
                self.zone_analytics[zone_id]['exited'] += 1
                self.zone_analytics[zone_id]['current'] = max(0, self.zone_analytics[zone_id]['current'] - 1)
                self.zone_analytics[zone_id]['history'].append({
                    'timestamp': datetime.now(),
                    'action': 'exited',
                    'track_id': track_id
                })
    
    def update_object_path(self, track_id, center, frame_width=None, frame_height=None):
        """Update object tracking path"""
        if track_id not in self.object_paths:
            self.object_paths[track_id] = []
        self.object_paths[track_id].append(center)
        
        # Keep only recent path points (for performance)
        if len(self.object_paths[track_id]) > 50:
            self.object_paths[track_id] = self.object_paths[track_id][-50:]
        
        self.analyze_crossing(track_id, self.object_paths[track_id], frame_width, frame_height)

def load_yolo_model(model_name, device):
    """Load YOLO model using Ultralytics"""
    try:
        model = YOLO(model_name)
        if device == "cuda":
            model.to("cuda")
        # CPU is default, no need to explicitly set
        return model, model.names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def generate_unique_color(track_id):
    """Generate unique color for track ID"""
    np.random.seed(int(track_id) % 1000)
    return tuple(np.random.randint(0, 255, 3).tolist())

def generate_zone_color(zone_id, alpha=0.4):
    """Generate unique color for zone"""
    np.random.seed(zone_id * 42)  # Different seed for zones
    color = np.random.randint(0, 255, 3).tolist()
    return (color[0], color[1], color[2], int(alpha * 255))

def draw_zones(frame, zones, zone_tracker=None, debug=False):
    """Draw zones on frame with transparency - SIMPLIFIED without coordinate transformation"""
    overlay = frame.copy()
    frame_height, frame_width = frame.shape[:2]
    
    if debug:
        print(f"🔍 Frame dimensions: {frame_width}x{frame_height}")
        print(f"🔍 Number of zones to draw: {len(zones)}")
    
    for i, zone in enumerate(zones):
        if len(zone) < 3:
            continue
        
        # Use original zone coordinates directly - NO TRANSFORMATION
        transformed_zone = zone
        
        if debug:
            print(f"🎯 Drawing Zone {i+1}: {zone[:3]}... (first 3 points)")
            
        poly = np.array(transformed_zone, np.int32)
        color = generate_zone_color(i)
        
        # Fill polygon with transparent color
        cv2.fillPoly(overlay, [poly], color[:3])
        
        # Draw boundary
        cv2.polylines(frame, [poly], isClosed=True, color=(0, 0, 0), thickness=3)
        cv2.polylines(frame, [poly], isClosed=True, color=(255, 255, 255), thickness=1)
        
        # Add zone label
        if len(transformed_zone) > 0:
            label_pos = (int(transformed_zone[0][0]), int(transformed_zone[0][1]) - 10)
            cv2.putText(frame, f'Zone {i + 1}', label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Blend overlay
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

def draw_object_paths(frame, object_paths):
    """Draw tracked object paths"""
    for track_id, path in object_paths.items():
        if len(path) < 2:
            continue
        color = generate_unique_color(track_id)
        for i in range(1, len(path)):
            cv2.line(frame, path[i-1], path[i], color, 2)

def perform_inference(frame, model, names, tracker, zone_tracker, conf_threshold=0.5, iou_threshold=0.45):
    """Perform YOLO inference and tracking using Ultralytics"""
    # Get frame dimensions for coordinate transformation
    frame_height, frame_width = frame.shape[:2]
    
    # Run YOLO inference
    results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
    
    # Prepare detections for tracker
    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf = boxes.conf[i].cpu().numpy()
            cls = int(boxes.cls[i].cpu().numpy())
            
            # Convert to format expected by tracker: [x, y, w, h]
            x1, y1, x2, y2 = xyxy
            w, h = x2 - x1, y2 - y1
            
            detections.append((
                [int(x1), int(y1), int(w), int(h)],
                float(conf),
                names[cls]
            ))
    
    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # Draw detections and tracks
    annotated_frame = frame.copy()
    
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
            
        track_id = track.track_id
        bbox = track.to_ltrb()
        center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
        
        # Update zone tracker with frame dimensions
        zone_tracker.update_object_path(track_id, center, frame_width, frame_height)
        
        # Draw bounding box and track ID
        color = generate_unique_color(track_id)
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw track ID label
        label = f'ID: {track_id}'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated_frame

def display_zone_analytics(zone_tracker):
    """Display simple zone analytics in sidebar"""
    if not zone_tracker.zones:
        st.sidebar.info("No zones defined yet")
        return
    
    st.sidebar.header("📊 Quick Stats")
    
    # Summary metrics
    try:
        total_entries = sum(analytics['entered'] for analytics in zone_tracker.zone_analytics.values())
        total_exits = sum(analytics['exited'] for analytics in zone_tracker.zone_analytics.values())
        total_current = sum(analytics['current'] for analytics in zone_tracker.zone_analytics.values())
        
        st.sidebar.metric("🟢 Total Entries", total_entries)
        st.sidebar.metric("🔴 Total Exits", total_exits)
        st.sidebar.metric("👥 Current Inside", total_current)
        
        # Simple zone list
        st.sidebar.subheader("🎯 Zone Status")
        for zone_id, analytics in zone_tracker.zone_analytics.items():
            if zone_id < len(zone_tracker.zones):
                st.sidebar.write(f"**Zone {zone_id + 1}:** {analytics['current']} inside")
                
    except Exception as e:
        st.sidebar.error(f"Analytics error: {str(e)}")
        st.sidebar.write("Analytics temporarily unavailable")

def create_analytics_dashboard(zone_tracker):
    """Create simple analytics dashboard"""
    if not zone_tracker.zones:
        st.info("Define zones to see analytics")
        return
    
    # Overall summary
    st.subheader("📈 Overall Summary")
    
    # Calculate totals
    total_entries = sum(analytics['entered'] for analytics in zone_tracker.zone_analytics.values())
    total_exits = sum(analytics['exited'] for analytics in zone_tracker.zone_analytics.values())
    total_current = sum(analytics['current'] for analytics in zone_tracker.zone_analytics.values())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🟢 Total Entries", total_entries)
    with col2:
        st.metric("🔴 Total Exits", total_exits)
    with col3:
        st.metric("👥 Currently Inside", total_current)
    
    # Per-zone breakdown
    st.subheader("🎯 Zone Breakdown")
    
    for zone_id, analytics in zone_tracker.zone_analytics.items():
        if zone_id < len(zone_tracker.zones):
            with st.expander(f"📍 Zone {zone_id + 1} Analytics", expanded=True):
                # Zone metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Entered", analytics['entered'])
                with col2:
                    st.metric("Exited", analytics['exited'])
                with col3:
                    st.metric("Current Count", analytics['current'])
                
                # Simple progress bars
                if analytics['entered'] > 0 or analytics['exited'] > 0:
                    max_activity = max(analytics['entered'], analytics['exited'], 1)
                    st.write("**Activity Level:**")
                    st.progress(analytics['entered'] / max_activity, text=f"Entries: {analytics['entered']}")
                    st.progress(analytics['exited'] / max_activity, text=f"Exits: {analytics['exited']}")
                
                # Recent activity (simplified)
                if analytics['history']:
                    st.write("**Recent Activity (Last 5 events):**")
                    recent_events = analytics['history'][-5:]
                    for event in reversed(recent_events):
                        timestamp = event['timestamp'].strftime("%H:%M:%S")
                        action_emoji = "🟢" if event['action'] == 'entered' else "🔴"
                        track_id = event['track_id']
                        st.write(f"{action_emoji} {event['action']} at {timestamp} (ID: {track_id})")
                else:
                    st.write("*No activity recorded yet*")

def transform_canvas_coordinates(points, canvas_width, canvas_height, target_width, target_height):
    """Transform coordinates from canvas space to video frame space"""
    if canvas_width == target_width and canvas_height == target_height:
        return points  # No transformation needed
    
    scale_x = target_width / canvas_width
    scale_y = target_height / canvas_height
    
    transformed_points = []
    for x, y in points:
        new_x = int(x * scale_x)
        new_y = int(y * scale_y)
        transformed_points.append((new_x, new_y))
    
    return transformed_points

def extract_zones_from_canvas(canvas_data):
    """Extract polygon zones from canvas data"""
    zones = []
    try:
        if canvas_data and "objects" in canvas_data:
            for obj in canvas_data["objects"]:
                print(f"🔍 Processing canvas object: {obj.get('type', 'unknown')}")
                
                if obj.get("type") == "path":
                    # Extract points from path object (polygons)
                    if "path" in obj:
                        points = parse_svg_path(obj["path"])
                        if len(points) >= 3:
                            # For fabric.js path objects, coordinates are already absolute
                            # Just convert to integers
                            zone_points = [(int(x), int(y)) for x, y in points]
                            zones.append(zone_points)
                            print(f"📍 Extracted path zone: {zone_points[:3]}...")
                
                elif obj.get("type") == "rect":
                    # Handle rectangle objects
                    left = obj.get("left", 0)
                    top = obj.get("top", 0)
                    width = obj.get("width", 0)
                    height = obj.get("height", 0)
                    
                    # Convert rectangle to polygon points (clockwise)
                    rect_points = [
                        (int(left), int(top)),
                        (int(left + width), int(top)),
                        (int(left + width), int(top + height)),
                        (int(left), int(top + height))
                    ]
                    zones.append(rect_points)
                    print(f"📍 Extracted rect zone: {rect_points}")
                
                elif obj.get("type") == "polygon":
                    # Handle polygon objects directly
                    if "points" in obj:
                        points = [(int(p["x"]), int(p["y"])) for p in obj["points"]]
                        if len(points) >= 3:
                            zones.append(points)
                            print(f"📍 Extracted polygon zone: {points[:3]}...")
        
        print(f"🎯 Total zones extracted: {len(zones)}")
        return zones
    except Exception as e:
        st.error(f"Error extracting zones from canvas: {e}")
        print(f"❌ Error in extract_zones_from_canvas: {e}")
        return []

def parse_svg_path(path_data):
    """Parse SVG path data to extract coordinates"""
    points = []
    try:
        if isinstance(path_data, list):
            for cmd in path_data:
                if isinstance(cmd, list) and len(cmd) >= 3:
                    # SVG path commands: ['M', x, y] or ['L', x, y]
                    if cmd[0] in ['M', 'L']:
                        points.append((cmd[1], cmd[2]))
        return points
    except Exception as e:
        return []

def get_youtube_stream_url(url):
    """Extract stream URL from YouTube using yt-dlp"""
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'noplaylist': True,
        'quiet': True,
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            return info_dict.get('url')
    except Exception as e:
        st.error(f"Error extracting YouTube URL: {e}")
        return None



def main():
    st.set_page_config(
        page_title="Zone Flow Analytics",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Zone Flow Analytics Dashboard")
    st.markdown("*Real-time object tracking and zone-based flow analysis*")
    
    # Initialize session state
    if 'zone_tracker' not in st.session_state:
        st.session_state.zone_tracker = ZoneFlowTracker()
    if 'zones_defined' not in st.session_state:
        st.session_state.zones_defined = False
    if 'temp_zone_points' not in st.session_state:
        st.session_state.temp_zone_points = []
    if 'zone_definition_mode' not in st.session_state:
        st.session_state.zone_definition_mode = True
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Model selection
        st.subheader("Model Configuration")
        
        model_series = st.radio(
            "YOLO Series",
            ["YOLO11", "YOLO12"],
            index=0,
            help="Select YOLO model series. YOLO12 may not be available yet."
        )
        
        if model_series == "YOLO11":
            model_options = [
                "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt", 
                "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt"
            ]
        else:
            model_options = [
                "yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt",
                "yolo12n-seg.pt", "yolo12s-seg.pt", "yolo12m-seg.pt", "yolo12l-seg.pt", "yolo12x-seg.pt"
            ]
        
        model_name = st.selectbox(
            "Model Size",
            model_options,
            index=0,
            help="n=nano (fastest), s=small, m=medium, l=large, x=extra large (most accurate). Segmentation models (-seg) provide pixel-level masks."
        )
        
        device = st.selectbox("Device", ["cpu", "cuda"], index=0, help="Select CPU for compatibility or CUDA for GPU acceleration")
        
        # Debug: Check model_name type and value
        st.write(f"Debug - model_name type: {type(model_name)}, value: {model_name}")
        
        # Detection parameters
        st.subheader("Detection Parameters")
        conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
        iou_threshold = st.slider("IoU Threshold", 0.1, 0.9, 0.45, 0.05)
        
        # Model info
        with st.expander("Model Information"):
            st.write(f"**Selected Model:** {model_name}")
            if "seg" in str(model_name):
                st.info("🎯 Segmentation model selected - provides pixel-level object masks")
            else:
                st.info("📦 Detection model selected - provides bounding boxes only")
            
            st.write("**Model Sizes:**")
            st.write("- **Nano (n):** Fastest, lowest accuracy")
            st.write("- **Small (s):** Good balance of speed and accuracy")
            st.write("- **Medium (m):** Better accuracy, moderate speed")
            st.write("- **Large (l):** High accuracy, slower")
            st.write("- **Extra Large (x):** Highest accuracy, slowest")
        
        # Source selection
        source_type = st.radio(
            "Video Source",
            ["Webcam", "Video File", "Stream URL", "YouTube URL"]
        )
        
        video_source = None
        if source_type == "Webcam":
            video_source = 0
        elif source_type == "Video File":
            uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
            if uploaded_file:
                # Save uploaded file temporarily
                with open(f"temp_{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.read())
                video_source = f"temp_{uploaded_file.name}"
        elif source_type == "Stream URL":
            video_source = st.text_input("Stream URL")
        elif source_type == "YouTube URL":
            youtube_url = st.text_input("YouTube URL")
            if youtube_url:
                with st.spinner("Extracting stream URL..."):
                    video_source = get_youtube_stream_url(youtube_url)
        
        st.divider()
        
        # Mode selection
        st.header("🎮 Mode Selection")
        mode = st.radio(
            "Select Mode:",
            ["Zone Definition", "Live Analysis"],
            index=0 if st.session_state.zone_definition_mode else 1
        )
        
        st.session_state.zone_definition_mode = (mode == "Zone Definition")
        
        # Zone management
        st.header("🏁 Zone Management")
        
        st.write(f"**Defined Zones:** {len(st.session_state.zone_tracker.zones)}")
        
        # Zone definition help
        if st.session_state.zone_definition_mode:
            with st.expander("How to define zones", expanded=True):
                st.write("""
                **Zone Definition Steps:**
                1. Select a video source above
                2. Click 'Load Video Frame' to get a frame
                3. Input coordinates manually or use the coordinate helper
                4. Click 'Complete Zone' when you have at least 3 points
                5. Repeat for multiple zones
                6. Switch to 'Live Analysis' mode to begin tracking
                """)
        
        # Zone drawing method
        if st.session_state.zone_definition_mode:
            st.subheader("Zone Drawing Options")
            
            # Check if canvas is available
            if CANVAS_AVAILABLE:
                drawing_options = ["Interactive Canvas (Polygons)", "Interactive Canvas (Rectangles)", "Manual Coordinates"]
                default_index = 0
            else:
                drawing_options = ["Manual Coordinates", "Demo Zones"]
                default_index = 0
                st.warning("⚠️ Interactive canvas not available. Using manual coordinate input.")
            
            drawing_mode = st.radio(
                "Drawing Method:",
                drawing_options,
                index=default_index,
                help="Choose how to define zones"
            )
            
            # Manual coordinate input (fallback option)
            if drawing_mode == "Manual Coordinates":
                with st.expander("Manual Coordinate Input"):
                    col_x, col_y, col_add = st.columns([1, 1, 1])
                    with col_x:
                        point_x = st.number_input("X coordinate", min_value=0, value=100)
                    with col_y:
                        point_y = st.number_input("Y coordinate", min_value=0, value=100)
                    with col_add:
                        st.write("")  # Space
                        if st.button("➕ Add Point"):
                            st.session_state.temp_zone_points.append((int(point_x), int(point_y)))
                            st.rerun()
        
        # Zone controls
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Clear All Zones"):
                st.session_state.zone_tracker.clear_zones()
                st.session_state.zones_defined = False
                st.session_state.temp_zone_points = []
                st.success("All zones cleared")
        
        with col2:
            if st.button("🔄 Reset Analytics"):
                st.session_state.zone_tracker.zone_analytics = defaultdict(
                    lambda: {'entered': 0, 'exited': 0, 'current': 0, 'history': []}
                )
                st.session_state.zone_tracker.zone_counts = {}
                st.session_state.zone_tracker.object_paths = {}
                st.success("Analytics reset")
        
        with col3:
            if st.button("🎯 Demo Zones"):
                # Create demo zones for testing
                st.session_state.zone_tracker.clear_zones()
                demo_zones = [
                    [(100, 100), (300, 100), (300, 200), (100, 200)],  # Rectangle zone
                    [(400, 150), (500, 100), (600, 150), (500, 250)]   # Diamond zone
                ]
                for zone in demo_zones:
                    st.session_state.zone_tracker.add_zone(zone)
                st.success("Demo zones created!")
        
        # Display zone analytics
        display_zone_analytics(st.session_state.zone_tracker)
    
    # Main content area
    if not video_source:
        st.info("Please select a video source to begin")
        return
    
    # Load model
    with st.spinner(f"Loading YOLO model: {str(model_name)}..."):
        model, names = load_yolo_model(str(model_name), device)
        if not model:
            st.error("Failed to load model")
            return
    
    # Initialize tracker
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=None, embedder_gpu=False)
    
    # Video processing section - full width
    st.header("📹 Video Processing")
    
    if st.session_state.zone_definition_mode:
        st.subheader("🎯 Zone Definition")
        
        # Load a frame for zone definition
        if st.button("📸 Load Video Frame"):
            cap = cv2.VideoCapture(video_source)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # IMPORTANT: Use the same resize factor as live analysis (0.6)
                    frame = cv2.resize(frame, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_LINEAR)
                    st.session_state.current_frame = frame.copy()
                    st.session_state.frame_width = frame.shape[1]
                    st.session_state.frame_height = frame.shape[0]
                    st.info(f"📐 Frame loaded: {st.session_state.frame_width}x{st.session_state.frame_height}")
                cap.release()
            else:
                st.error("Cannot open video source")
        
        # Zone definition interface
        if 'current_frame' in st.session_state:
            frame_copy = st.session_state.current_frame.copy()
            
            # Check drawing mode from sidebar
            if 'drawing_mode' not in st.session_state:
                st.session_state.drawing_mode = "Interactive Canvas"
            
            if drawing_mode.startswith("Interactive Canvas") and CANVAS_AVAILABLE:
                # Interactive canvas drawing
                st.subheader("🎨 Draw Zones on Frame")
                
                try:
                    if "Polygons" in drawing_mode:
                        st.info("🔹 Draw polygons by clicking points. Double-click to close the polygon.")
                        canvas_drawing_mode = "polygon"
                    else:
                        st.info("🔲 Draw rectangles by clicking and dragging.")
                        canvas_drawing_mode = "rect"
                    
                    # Convert frame to PIL Image for canvas
                    frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Store frame dimensions for coordinate mapping
                    st.session_state.canvas_height = pil_image.height
                    st.session_state.canvas_width = pil_image.width
                    
                    # Create canvas
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.3)",  # Semi-transparent orange fill
                        stroke_width=3,
                        stroke_color="#FF4500",  # Orange stroke
                        background_image=pil_image,
                        update_streamlit=True,
                        height=pil_image.height,
                        width=pil_image.width,
                        drawing_mode=canvas_drawing_mode,
                        point_display_radius=6,
                        key="zone_canvas",
                    )
                except Exception as e:
                    st.error(f"❌ Canvas error: {e}")
                    st.error("Please try the manual coordinate input method instead.")
                    canvas_result = None
                
                # Extract zones from canvas
                if canvas_result and canvas_result.json_data is not None:
                    zones_from_canvas = extract_zones_from_canvas(canvas_result.json_data)
                    
                    # Debug: Show coordinate information
                    if zones_from_canvas:
                        st.write("🔍 **Debug Info:**")
                        st.write(f"Canvas dimensions: {st.session_state.canvas_width}x{st.session_state.canvas_height}")
                        for i, zone in enumerate(zones_from_canvas):
                            st.write(f"Zone {i+1} coordinates: {zone[:3]}...")  # Show first 3 points
                    
                    # Update zones if new ones are drawn
                    if zones_from_canvas and len(zones_from_canvas) != len(st.session_state.zone_tracker.zones):
                        st.session_state.zone_tracker.clear_zones()
                        for zone_points in zones_from_canvas:
                            # Store zones directly without coordinate transformation metadata
                            st.session_state.zone_tracker.add_zone(zone_points)
                        st.rerun()  # Refresh to show updated zone count
                
                # Canvas controls and info
                canvas_col1, canvas_col2, canvas_col3 = st.columns(3)
                with canvas_col1:
                    if st.button("💾 Save Drawn Zones", type="primary"):
                        if canvas_result and canvas_result.json_data and canvas_result.json_data["objects"]:
                            zones_count = len(st.session_state.zone_tracker.zones)
                            st.success(f"Saved {zones_count} zones!")
                            if zones_count > 0:
                                st.info("Switch to 'Live Analysis' mode to start tracking!")
                                # Show saved zone coordinates for debugging
                                with st.expander("🔍 Saved Zone Coordinates (Debug)"):
                                    for i, zone in enumerate(st.session_state.zone_tracker.zones):
                                        st.write(f"**Zone {i+1}:** {zone}")
                        else:
                            st.warning("No zones drawn yet!")
                
                with canvas_col2:
                    if st.button("🔄 Clear Canvas"):
                        st.session_state.zone_tracker.clear_zones()
                        st.rerun()
                
                with canvas_col3:
                    zones_count = len(st.session_state.zone_tracker.zones)
                    st.metric("Zones Created", zones_count)
                
                # Show zone details if any exist
                if st.session_state.zone_tracker.zones:
                    with st.expander("📋 Zone Details", expanded=False):
                        for i, zone in enumerate(st.session_state.zone_tracker.zones):
                            st.write(f"**Zone {i+1}:** {len(zone)} points")
                            if len(zone) <= 6:  # Show points for simple shapes
                                for j, point in enumerate(zone):
                                    st.write(f"  Point {j+1}: ({point[0]}, {point[1]})")
                            else:
                                st.write(f"  Complex polygon with {len(zone)} vertices")
                
                # Drawing tips
                with st.expander("💡 Drawing Tips"):
                    if "Polygons" in drawing_mode:
                        st.write("""
                        **Polygon Drawing:**
                        - Click to add points for your polygon
                        - Double-click to complete the polygon
                        - Create complex shapes like triangles, pentagons, etc.
                        - Best for irregular monitoring areas
                        """)
                    else:
                        st.write("""
                        **Rectangle Drawing:**
                        - Click and drag to create rectangles
                        - Perfect for doorways, windows, or square areas
                        - Faster for simple rectangular zones
                        - Hold shift for squares
                        """)
            
            elif drawing_mode == "Demo Zones":
                # Demo zones option
                st.subheader("🎯 Demo Zones")
                st.info("Click the button below to create sample zones for testing.")
                
                if st.button("Create Demo Zones", type="primary"):
                    st.session_state.zone_tracker.clear_zones()
                    # Create demo zones based on frame dimensions
                    frame_h, frame_w = frame_copy.shape[:2]
                    demo_zones = [
                        [(int(frame_w*0.1), int(frame_h*0.2)), (int(frame_w*0.4), int(frame_h*0.2)), 
                         (int(frame_w*0.4), int(frame_h*0.6)), (int(frame_w*0.1), int(frame_h*0.6))],  # Left rectangle
                        [(int(frame_w*0.6), int(frame_h*0.3)), (int(frame_w*0.9), int(frame_h*0.3)), 
                         (int(frame_w*0.9), int(frame_h*0.7)), (int(frame_w*0.6), int(frame_h*0.7))]   # Right rectangle
                    ]
                    for zone in demo_zones:
                        st.session_state.zone_tracker.add_zone(zone)
                    st.success(f"Created {len(demo_zones)} demo zones!")
                    st.rerun()
                
                # Show frame with existing zones
                draw_zones(frame_copy, st.session_state.zone_tracker.zones, st.session_state.zone_tracker)
                st.image(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB), channels="RGB")
            
            else:
                # Manual coordinate mode (existing functionality)
                # Draw existing zones and current zone on frame
                draw_zones(frame_copy, st.session_state.zone_tracker.zones, st.session_state.zone_tracker)
                
                # Draw current zone being defined
                if len(st.session_state.temp_zone_points) > 0:
                    points = np.array(st.session_state.temp_zone_points, np.int32)
                    if len(points) > 1:
                        cv2.polylines(frame_copy, [points], isClosed=False, color=(0, 255, 255), thickness=3)
                    
                    # Draw points
                    for i, point in enumerate(st.session_state.temp_zone_points):
                        cv2.circle(frame_copy, point, 8, (0, 255, 255), -1)
                        cv2.putText(frame_copy, str(i+1), (point[0]+10, point[1]-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                video_placeholder = st.empty()
                video_placeholder.image(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB), channels="RGB")
                
                # Display current zone points info
                if st.session_state.temp_zone_points:
                    st.info(f"Current zone has {len(st.session_state.temp_zone_points)} points. Need at least 3 to complete.")
                    with st.expander("Current Zone Points"):
                        for i, point in enumerate(st.session_state.temp_zone_points):
                            st.write(f"Point {i+1}: ({point[0]}, {point[1]})")
                
                # Zone completion controls
                zone_col1, zone_col2, zone_col3 = st.columns(3)
                with zone_col1:
                    if st.button("✅ Complete Current Zone", type="primary") and len(st.session_state.temp_zone_points) >= 3:
                        st.session_state.zone_tracker.add_zone(st.session_state.temp_zone_points.copy())
                        st.session_state.temp_zone_points = []
                        st.success(f"Zone {len(st.session_state.zone_tracker.zones)} created!")
                        st.rerun()
                
                with zone_col2:
                    if st.button("↶ Undo Last Point") and st.session_state.temp_zone_points:
                        st.session_state.temp_zone_points.pop()
                        st.rerun()
                
                with zone_col3:
                    if st.button("🗑️ Clear Current Zone"):
                        st.session_state.temp_zone_points = []
                        st.rerun()
        else:
            st.info("Click 'Load Video Frame' to begin zone definition")
    
    else:  # Live Analysis Mode
        st.subheader("📹 Live Video Analysis")
        video_placeholder = st.empty()
        
        # Control buttons
        button_col1, button_col2 = st.columns(2)
        with button_col1:
            start_button = st.button("▶️ Start Analysis", type="primary", 
                                    disabled=len(st.session_state.zone_tracker.zones) == 0)
        with button_col2:
            stop_button = st.button("⏹️ Stop Analysis")
        
        if len(st.session_state.zone_tracker.zones) == 0:
            st.warning("⚠️ Please define at least one zone before starting analysis")
        
        # Analysis processing loop
        if start_button or st.session_state.get('analysis_running', False):
            st.session_state.analysis_running = True
            
            # Open video capture
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                st.error(f"Cannot open video source: {video_source}")
                return
            
            # Processing loop
            frame_count = 0
            fps_start_time = time.time()
            
            while st.session_state.analysis_running and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.warning("End of video or cannot read frame")
                    break
                
                # Resize frame for better performance - SAME as zone definition (fx=0.6, fy=0.6)
                frame = cv2.resize(frame, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_LINEAR)
                
                # Process frame with object detection and tracking
                frame = perform_inference(frame, model, names, tracker, st.session_state.zone_tracker, 
                                        conf_threshold, iou_threshold)
                draw_object_paths(frame, st.session_state.zone_tracker.object_paths)
                draw_zones(frame, st.session_state.zone_tracker.zones, st.session_state.zone_tracker, debug=True)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    fps = 10 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    
                    # Add FPS and info to frame
                    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f'Zones: {len(st.session_state.zone_tracker.zones)}', (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f'Frame: {frame.shape[1]}x{frame.shape[0]}', (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Display frame
                video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                
                # Small delay to prevent overwhelming
                time.sleep(0.03)
            
            cap.release()
            st.session_state.analysis_running = False
    
    # Analytics section - moved below video
    st.header("📊 Analytics Dashboard")
    try:
        if not st.session_state.zone_definition_mode and st.session_state.zone_tracker.zones:
            create_analytics_dashboard(st.session_state.zone_tracker)
        elif st.session_state.zone_definition_mode:
            st.info("Switch to 'Live Analysis' mode to see real-time analytics")
        else:
            st.info("Define zones and start analysis to see analytics")
    except Exception as e:
        st.error(f"Analytics dashboard error: {str(e)}")
        st.write("Please try refreshing the page or redefining zones")

if __name__ == "__main__":
    main() 