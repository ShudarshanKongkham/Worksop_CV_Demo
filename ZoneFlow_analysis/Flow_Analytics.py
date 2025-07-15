import time
import numpy as np
import torch
import cv2
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.plots import Annotator
from utils.torch_utils import select_device
from deep_sort_realtime.deepsort_tracker import DeepSort

# Global Variables
zones = []  # Store zones as polygons
current_polygon = []  # Vertices of the current polygon
drawing_polygon = False  # Track if a polygon is being drawn
analysis_started = False  # Track if video analysis has started

object_paths = {}  # Track object paths
zone_counts = {}  # Track entry/exit counts for each zone

### 1. Load YOLO Model
def load_model(weights, device):
    """Load the YOLO model."""
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, fp16=True)
    return model, model.names

### 2. Generate Unique Colors
def generate_unique_color(track_id):
    np.random.seed(int(track_id) % 1000)
    return tuple(np.random.randint(0, 255, 3).tolist())

def generate_zone_color(zone_id, alpha=0.4):
    np.random.seed(zone_id)
    color = np.random.randint(0, 255, 3).tolist()
    return (color[0], color[1], color[2], int(alpha * 255))

### 3. Mouse Callback for Zone Definition
def mouse_callback(event, x, y, flags, param):
    global drawing_polygon, current_polygon

    if event == cv2.EVENT_LBUTTONDOWN:  # Add point to the polygon
        drawing_polygon = True
        current_polygon.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN and len(current_polygon) > 0:  # Undo last point
        current_polygon.pop()

### 4. Draw Zones on Frame
def draw_zones(frame):
    """Draw all defined zones with transparent filling and boundary."""
    overlay = frame.copy()
    for i, zone in enumerate(zones):
        poly = np.array(zone, np.int32)

        # Fill polygon with a transparent color
        color = generate_zone_color(i)
        cv2.fillPoly(overlay, [poly], color[:3])

        # Draw dark boundary
        cv2.polylines(frame, [poly], isClosed=True, color=(0, 0, 0), thickness=2)

        # Label the zone
        cv2.putText(frame, f'Zone {i + 1}', zone[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Blend overlay with the frame
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

### 5. Analyze Crossings within ROI Polygons
def is_inside_polygon(point, polygon):
    """Check if the object's center is inside a polygon."""
    poly = np.array(polygon, np.int32)
    return cv2.pointPolygonTest(poly, point, False) >= 0

def analyze_crossing(track_id, path):
    """Analyze if the object entered/exited a polygon zone."""
    if len(path) < 2:
        return

    prev_center, curr_center = path[-2], path[-1]

    for zone_id, zone in enumerate(zones):
        entered = is_inside_polygon(curr_center, zone) and not is_inside_polygon(prev_center, zone)
        exited = is_inside_polygon(prev_center, zone) and not is_inside_polygon(curr_center, zone)

        if entered:
            zone_counts[zone_id]["entered"] += 1
        if exited:
            zone_counts[zone_id]["exited"] += 1

### 6. Perform Inference
def inference(frame, model, names, tracker, path_frame):
    """Perform inference and return annotated frames."""
    img = np.ascontiguousarray(frame[..., ::-1].transpose(2, 0, 1))
    img = torch.from_numpy(img).to(model.device).float() / 255.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    img = img.half() if model.fp16 else img

    with torch.no_grad():
        predictions = model(img)
        pred = non_max_suppression(predictions[0], conf_thres=0.5, iou_thres=0.45, max_det=100)

    annotator = Annotator(frame, line_width=2, example=str(names))

    detections = [
        ([int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])],
         conf.item(), names[int(cls)])
        for *xyxy, conf, cls in pred[0]
    ] if len(pred[0]) > 0 else []

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        track_id = track.track_id
        bbox = track.to_ltrb()
        center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

        if track_id not in object_paths:
            object_paths[track_id] = []
        object_paths[track_id].append(center)

        analyze_crossing(track_id, object_paths[track_id])

        color = generate_unique_color(track_id)
        draw_object_path(path_frame, object_paths[track_id], color)
        annotator.box_label(bbox, f'ID: {track_id}', color=color)

    return annotator.result()

def draw_object_path(frame, path, color):
    """Draw the tracked object's path."""
    for i in range(1, len(path)):
        if path[i - 1] and path[i]:
            cv2.line(frame, path[i - 1], path[i], color, 2)

def display_zone_counts(frame, zone_counts):
    """Display the zone counts in a neat overlay box."""
    # Define the starting position for the text
    start_x, start_y = 10, 20
    padding = 40  # Space between text lines

    # Draw a semi-transparent black rectangle as a background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, len(zone_counts) * padding + 20), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)  # Blend overlay with frame

    # Display zone counts neatly with padding
    for i, counts in zone_counts.items():
        # Define the text for the current zone
        entered_text = f'Zone {i + 1} Entered: {counts["entered"]}'
        exited_text = f'Exited: {counts["exited"]}'

        # Display "Entered" in green
        cv2.putText(frame, entered_text, (start_x, start_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display "Exited" in red, below the entered text
        cv2.putText(frame, exited_text, (start_x, start_y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Update y position for the next zone
        start_y += padding

### 7. Main Run Function with Recording Capability
def run(weights='yolov9-c.pt', device=0):
    """Run the YOLO model with ROI zone tracking and recording."""
    global current_polygon, analysis_started

    model, names = load_model(weights, device)
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=None, embedder_gpu=True, half=True)

    cap = cv2.VideoCapture("G:/UTS/2024/Spring_2024/Image Processing/Assignment/Video-Analytics-/data_/traffic_1.mp4")

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)

    if not ret:
        print("Error: Could not read video frame.")
        return

    # Initialize video writers for recording
    frame_height, frame_width = frame.shape[:2]
    output_yolo = cv2.VideoWriter('yolo_detection_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))
    output_zone_analysis = cv2.VideoWriter('zone_analysis_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))

    cv2.namedWindow('Define Zones')
    cv2.setMouseCallback('Define Zones', mouse_callback)

    while not analysis_started:
        frame_copy = frame.copy()

        # Draw the current polygon being defined
        if len(current_polygon) > 0:
            cv2.polylines(frame_copy, [np.array(current_polygon)], isClosed=False, color=(0, 255, 255), thickness=2)

        draw_zones(frame_copy)
        cv2.imshow('Define Zones', frame_copy)

        key = cv2.waitKey(1)
        if key == 13 and len(current_polygon) > 2:  # Enter finishes current polygon
            zones.append(current_polygon[:])
            zone_counts[len(zones) - 1] = {"entered": 0, "exited": 0}
            current_polygon = []
        elif key == ord('s'):  # 's' starts the analysis
            analysis_started = True

    path_frame = np.zeros_like(frame)

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)

        if not ret:
            print("End of video.")
            break

        # Run inference and overlay zones on YOLO detection frame
        annotated_frame = inference(frame, model, names, tracker, path_frame)
        draw_zones(annotated_frame)  # Overlay zones directly on YOLO detection frame

        # Draw zones and display zone counts on zone analysis frame
        draw_zones(path_frame)
        display_zone_counts(path_frame, zone_counts)

        # Write frames to respective video files
        output_yolo.write(annotated_frame)
        output_zone_analysis.write(path_frame)

        # Show the frames
        cv2.imshow('YOLOv9 Detection', annotated_frame)
        cv2.imshow('Zone Analysis', path_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    output_yolo.release()
    output_zone_analysis.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
