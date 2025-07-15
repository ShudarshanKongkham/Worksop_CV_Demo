# Zone Flow Analytics - Streamlit App

A comprehensive real-time object tracking and zone-based flow analysis application powered by YOLO11/12 and Streamlit.

## Features

- 🎯 **Multi-Zone Definition**: Create custom polygonal zones for area-specific tracking
- 📊 **Real-time Analytics**: Live tracking of object entry/exit counts and current occupancy
- 🚀 **YOLO11/12 Integration**: Latest Ultralytics YOLO models with segmentation support
- 📹 **Multiple Video Sources**: Support for webcam, video files, stream URLs, and YouTube videos
- 🎮 **Interactive Interface**: Easy zone definition with visual feedback
- 📈 **Rich Visualizations**: Plotly-powered analytics dashboards and charts

## Installation

1. **Clone the repository** (if not already done)
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_flow_analytics.py
   ```

2. **Configure the application**:
   - Select YOLO model (YOLO11 or YOLO12)
   - Choose model size (nano to extra large)
   - Set detection parameters (confidence and IoU thresholds)
   - Select video source (webcam, file, stream, or YouTube)

3. **Define zones** (Zone Definition mode):
   - Click "Load Video Frame" to capture a frame
   - Add zone points manually using coordinates or use the demo zones
   - Complete zones with at least 3 points
   - Switch to "Live Analysis" mode

4. **Start analysis** (Live Analysis mode):
   - Click "Start Analysis" to begin real-time tracking
   - Monitor zone entry/exit counts in real-time
   - View analytics dashboard with charts and metrics

## Model Options

### YOLO11 Models
- **Detection Models**: `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`
- **Segmentation Models**: `yolo11n-seg.pt`, `yolo11s-seg.pt`, `yolo11m-seg.pt`, `yolo11l-seg.pt`, `yolo11x-seg.pt`

### YOLO12 Models (if available)
- **Detection Models**: `yolo12n.pt`, `yolo12s.pt`, `yolo12m.pt`, `yolo12l.pt`, `yolo12x.pt`
- **Segmentation Models**: `yolo12n-seg.pt`, `yolo12s-seg.pt`, `yolo12m-seg.pt`, `yolo12l-seg.pt`, `yolo12x-seg.pt`

**Model Size Guide:**
- **n (nano)**: Fastest inference, lowest accuracy
- **s (small)**: Good balance of speed and accuracy
- **m (medium)**: Better accuracy, moderate speed  
- **l (large)**: High accuracy, slower inference
- **x (extra large)**: Highest accuracy, slowest inference

## Video Sources

- **Webcam**: Uses default camera (index 0)
- **Video File**: Upload MP4, AVI, or MOV files
- **Stream URL**: Direct video stream URLs (RTSP, HTTP, etc.)
- **YouTube URL**: Automatically extracts video stream from YouTube links

## Zone Definition

1. **Interactive Canvas Drawing**: 
   - **Polygons**: Click points to create custom polygon shapes, double-click to complete
   - **Rectangles**: Click and drag to create rectangular zones (perfect for doors, windows)
   - Real-time visual feedback on the video frame
   
2. **Manual Coordinate Input**: Enter X,Y coordinates manually (fallback option)
3. **Demo Zones**: Use pre-defined zones for quick testing
4. **Visual Zone Preview**: See all zones overlaid on the video with transparency

## Analytics Features

- **Real-time Metrics**: Total entries, exits, current occupancy per zone
- **Interactive Charts**: Entry/exit comparisons, occupancy heatmaps
- **Activity Timeline**: Chronological view of zone activities
- **Recent Activity Log**: Track individual object movements

## Performance Tips

- Use smaller models (nano/small) for real-time performance
- Reduce video resolution for better FPS
- Use CPU for testing, GPU for production workloads
- Adjust confidence thresholds based on your use case

## Troubleshooting

- **Model Loading Issues**: Ensure internet connection for first-time model download
- **Performance Issues**: Try smaller model sizes or reduce video resolution
- **Zone Definition Problems**: Use demo zones to test before creating custom zones
- **Video Source Issues**: Verify video source accessibility and format support

## Requirements

See `requirements.txt` for complete list of dependencies. Key requirements:
- Python 3.8+
- Streamlit 1.28+
- Ultralytics 8.1+
- OpenCV 4.8+
- PyTorch 2.0+

## License

This project is part of the AI Workshop CV Demo repository. 