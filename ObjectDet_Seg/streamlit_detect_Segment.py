import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import yt_dlp as youtube_dl
import time
import colorsys  # Import colorsys

# Set logging level to ERROR to suppress YOLO detection outputs
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Load COCO class names
with open('coco.names', 'r') as f:
    class_names = f.read().splitlines()

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


def visualize_detections_and_segmentation(frame, results, model, selected_classes, class_colors):
    # Create a copy of the frame
    frame_copy = frame.copy()

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

    return frame_copy

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
    st.title("Object Detection and Segmentation")

    model_name = "yolo11n-seg.pt"  # Change this to the desired model {n: nano, s: small, m: medium, l: large, xl: extra large}
    model = YOLO(model_name)
    # st.write(f"Running model: {model_name}")
    print(f"Running model: {model_name}")

    with st.sidebar:
        selected_classes = st.multiselect("Select Classes to Detect", class_names, default=[class_names[0]])
        # Generate distinct colors *only* for the selected classes
        class_colors = generate_distinct_colors(len(selected_classes))

        source_type = st.radio("Select Input Source:", ["Webcam", "Stream URL", "YouTube URL"])

        if source_type == "Stream URL":
            stream_url = st.text_input("Enter Stream URL:")
        elif source_type == "YouTube URL":
            youtube_url = st.text_input("Enter YouTube URL:")
            if youtube_url:
                stream_url = get_best_video_url(youtube_url)
                if not stream_url:
                    return
            else:
                stream_url = None
        else:
            stream_url = 0

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

    frame_placeholder = st.empty()
    stop_button_pressed = st.sidebar.button("Stop")

    if cap:
        while cap.isOpened() and not stop_button_pressed:
            ret, frame = cap.read()
            if not ret:
                if source_type in ("Stream URL", "YouTube URL"):
                    st.error("Stream ended or encountered an error.")
                else:
                    st.error("Could not read frame from webcam.")
                break

            height, width = frame.shape[:2]
            new_width = width * 2
            new_height = height * 2

            results = model(frame, stream=True)
            # Pass the class_colors to the visualization function
            frame = visualize_detections_and_segmentation(frame, results, model, selected_classes, class_colors)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()