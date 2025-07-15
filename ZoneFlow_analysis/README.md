# Zone Flow Analytics - Streamlit App

A comprehensive real-time object tracking and zone-based flow analysis application powered by YOLO11/12 and Streamlit.

## ✨ Key Features

- 🎯 **Smart Zone Definition**: Interactive canvas drawing with polygons/rectangles or manual coordinates
- 📊 **Live Analytics**: Real-time metrics that update during analysis + detailed charts after stopping
- 🚀 **YOLO11/12 Integration**: Latest Ultralytics YOLO models with detection and segmentation support
- 📹 **Multiple Video Sources**: Webcam, video files, stream URLs, and YouTube videos
- 🎮 **User-Friendly Interface**: Simplified controls with visual feedback and error handling
- 📈 **Dual Analytics Mode**: Live updating values during analysis + comprehensive dashboard after stopping

## 🚀 Quick Start

### Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   streamlit run streamlit_flow_analytics.py
   ```

### Basic Workflow

1. **Configure Model**: Select YOLO11/12 model and size in the sidebar
2. **Choose Video Source**: Webcam, file upload, stream URL, or YouTube
3. **Define Zones**: Use interactive canvas or manual coordinates to create tracking zones
4. **Start Analysis**: Watch live metrics update in real-time
5. **View Results**: Stop analysis to see detailed charts and history

## 🎯 Zone Definition Options

### Interactive Canvas Drawing (Recommended)
- **Polygon Mode**: Click points to create custom shapes, double-click to finish
- **Rectangle Mode**: Click and drag to create rectangular zones
- **Visual Feedback**: See zones overlaid on actual video frame
- **Auto-coordinate Matching**: Zones drawn match exactly with live analysis

### Alternative Methods
- **Manual Coordinates**: Enter X,Y points manually for precise control
- **Demo Zones**: Quick test zones for immediate setup
- **Fallback Support**: Graceful handling if canvas is unavailable

### Zone Features
- ✅ Unlimited zones per video
- ✅ Visual zone previews with transparency
- ✅ Zone validation (minimum 3 points)
- ✅ Easy zone management (clear, reset, demo)

## 📊 Analytics Dashboard

### Live Analytics (During Analysis)
- **Real-time Updates**: Metrics update every 10 frames while running
- **Current Counts**: 🟢 Entries | 🔴 Exits | 👥 Currently Inside
- **Zone Status**: Individual zone metrics in compact layout
- **Performance Info**: FPS, frame dimensions, zone count

### Detailed Analytics (After Stopping)
- **Overall Summary**: Total metrics across all zones
- **Zone Breakdown**: Individual zone performance with expandable sections
- **Activity Progress**: Visual progress bars showing relative zone activity
- **Event History**: Recent entries/exits with timestamps and track IDs
- **Trends Analysis**: Compare zone performance and occupancy patterns

## 🤖 Model Configuration

### YOLO11 Models
| Model | Speed | Accuracy | Use Case |
|-------|--------|----------|----------|
| `yolo11n.pt` | ⚡⚡⚡ | ⭐⭐ | Real-time, low-power devices |
| `yolo11s.pt` | ⚡⚡ | ⭐⭐⭐ | Balanced performance |
| `yolo11m.pt` | ⚡ | ⭐⭐⭐⭐ | Good accuracy, moderate speed |
| `yolo11l.pt` | 🐌 | ⭐⭐⭐⭐⭐ | High accuracy applications |
| `yolo11x.pt` | 🐌🐌 | ⭐⭐⭐⭐⭐⭐ | Maximum accuracy |

### YOLO12 Models
- Same size options available: `yolo12n.pt`, `yolo12s.pt`, `yolo12m.pt`, `yolo12l.pt`, `yolo12x.pt`

### Segmentation Models
- Add `-seg` suffix for pixel-level object masks: `yolo11n-seg.pt`, `yolo11s-seg.pt`, etc.
- Provides detailed object boundaries instead of just bounding boxes

### Device Options
- **CPU**: Compatible with all systems, slower processing
- **CUDA**: GPU acceleration for faster inference (requires NVIDIA GPU)

## 📹 Video Source Support

### Supported Sources
- **📷 Webcam**: Default camera or specify camera index
- **📁 Video Files**: MP4, AVI, MOV formats via file upload
- **🌐 Stream URLs**: RTSP, HTTP video streams
- **📺 YouTube**: Automatic stream extraction from YouTube URLs

### Performance Tips
- **For Real-time**: Use webcam with nano/small models
- **For Accuracy**: Use large/extra-large models with video files
- **For Testing**: Use demo zones with any video source

## ⚙️ Configuration Options

### Detection Parameters
- **Confidence Threshold**: 0.1-0.9 (default: 0.5) - Higher = fewer false positives
- **IoU Threshold**: 0.1-0.9 (default: 0.45) - Controls detection overlap handling

### Interface Settings
- **Two Modes**: Zone Definition ↔ Live Analysis
- **Visual Feedback**: Real-time zone overlays and object tracking
- **Error Handling**: Graceful fallbacks and helpful error messages

## 🎮 User Interface

### Sidebar Controls
- Model selection and configuration
- Video source setup
- Mode switching (Definition ↔ Analysis)
- Quick analytics summary
- Zone management tools

### Main Interface
- **Zone Definition Mode**: Interactive drawing canvas with video frame
- **Live Analysis Mode**: Video stream with real-time object tracking
- **Analytics Section**: Live metrics + detailed dashboard

## 🔧 Troubleshooting

### Common Issues
- **Canvas Drawing Problems**: Use manual coordinates or demo zones as fallback
- **Model Loading Slow**: First download takes time, subsequent loads are faster
- **Performance Issues**: Try smaller models (nano/small) or reduce video quality
- **Video Source Failed**: Check file format, URL validity, or camera permissions

### Performance Optimization
- **CPU Usage**: Use nano models, reduce video resolution
- **GPU Usage**: Enable CUDA, use larger models for better accuracy
- **Memory**: Close other applications, use shorter video clips for testing

### Dependencies
- **Streamlit**: >=1.28.0,<1.30.0 (for canvas compatibility)
- **OpenCV**: Computer vision operations
- **Ultralytics**: YOLO model inference
- **DeepSort**: Object tracking across frames

## 📝 Technical Details

### Object Tracking Pipeline
1. **YOLO Detection**: Identify objects in each frame
2. **DeepSort Tracking**: Maintain object identities across frames
3. **Zone Analysis**: Check object positions against defined zones
4. **Analytics Update**: Track entries, exits, and current occupancy
5. **Visual Output**: Render results with bounding boxes and zone overlays

### Zone Coordinate System
- Zones defined in frame coordinates (0.6 scale factor applied consistently)
- Direct coordinate mapping between definition and analysis modes
- No complex transformations - what you draw is what you get

### Performance Metrics
- **FPS Display**: Real-time processing speed
- **Frame Info**: Current video dimensions
- **Zone Count**: Number of active tracking zones
- **Object Counts**: Live tracking statistics

## 📄 License

This project is part of an AI Workshop CV Demo. Please refer to the main repository for licensing information.

## 🤝 Contributing

This application was developed as part of a computer vision workshop. For improvements or bug reports, please refer to the main repository or create issues with detailed descriptions. 