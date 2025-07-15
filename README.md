# 🚀 AI Workshop CV Demo

**Dive into the world of AI!** This repository is a playground of computer vision projects and apps, designed to spark curiosity and empower creators to build their own amazing AI adventures.

## 🎯 Projects Overview

This repository contains two powerful computer vision applications showcasing different aspects of AI-powered video analysis:

### 1. 📦 Object Detection & Segmentation (`ObjectDet_Seg/`)
Real-time object detection and segmentation using YOLOv11 with Streamlit interface.

**Key Features:**
- 🎯 Real-time object detection and segmentation
- 📹 Multiple input sources (webcam, streams, YouTube)
- 🤖 YOLOv11 model integration
- 🎮 Interactive Streamlit interface
- 📊 Live detection visualization

### 2. 🏁 Zone Flow Analytics (`ZoneFlow_analysis/`)
Advanced zone-based tracking and analytics system for monitoring object movement through defined areas.

**Key Features:**
- 🎯 Interactive zone definition (polygons/rectangles)
- 📊 Live analytics with real-time updates
- 🚀 YOLO11/12 integration with object tracking
- 📈 Comprehensive analytics dashboard
- 🎮 Dual-mode interface (Definition + Analysis)

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ 
- Git for cloning
- Conda (recommended for environment management)

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ShudarshanKongkham/Worksop_CV_Demo.git
   cd Worksop_CV_Demo
   ```

2. **Choose Your Project:**
   
   **Option A: Object Detection & Segmentation**
   ```bash
   cd ObjectDet_Seg
   conda create -n yolo-detection python=3.10
   conda activate yolo-detection
   pip install -r requirements.txt
   streamlit run streamlit_detect_Segment.py
   ```

   **Option B: Zone Flow Analytics**
   ```bash
   cd ZoneFlow_analysis
   conda create -n zone-analytics python=3.10
   conda activate zone-analytics
   pip install -r requirements.txt
   streamlit run streamlit_flow_analytics.py
   ```

## 📦 Object Detection & Segmentation

### Features
- **Real-time Processing**: Live object detection and segmentation
- **Multiple Sources**: Webcam, stream URLs, YouTube videos
- **YOLOv11 Integration**: Latest object detection models
- **Segmentation Support**: Pixel-level object boundaries
- **Interactive Controls**: Real-time parameter adjustment

### Usage
1. Select input source (webcam/stream/YouTube)
2. Adjust detection parameters
3. Watch real-time detection results
4. Use stop button to terminate processing

### Supported Formats
- **Video Files**: MP4, AVI, MOV
- **Streams**: RTSP, HTTP video streams  
- **YouTube**: Direct URL processing
- **Webcam**: Default camera access

## 🏁 Zone Flow Analytics

### Features
- **Smart Zone Definition**: Interactive canvas drawing with visual feedback
- **Live Analytics**: Real-time metrics updating during analysis
- **Advanced Tracking**: YOLO + DeepSort for object persistence
- **Comprehensive Dashboard**: Detailed analytics after stopping analysis
- **Multiple Models**: YOLO11/12 with various sizes and segmentation support

### Workflow
1. **Configure Model**: Select YOLO model and parameters
2. **Choose Video Source**: Webcam, file, stream, or YouTube
3. **Define Zones**: Use interactive canvas or manual coordinates
4. **Start Analysis**: Monitor live metrics and object tracking
5. **Review Results**: Access detailed analytics and event history

### Analytics Features
- **Live Updates**: Entry/exit counts update every 10 frames
- **Zone Management**: Unlimited custom zones with visual overlays
- **Event Tracking**: Complete history with timestamps and track IDs
- **Performance Metrics**: FPS monitoring and system information

## 🎮 User Interfaces

### Object Detection Interface
- **Sidebar Controls**: Source selection, model parameters
- **Main Display**: Live video stream with detection overlays
- **Real-time Info**: Detection counts, processing speed

### Zone Analytics Interface
- **Dual Mode System**: 
  - **Zone Definition Mode**: Interactive drawing and setup
  - **Live Analysis Mode**: Real-time tracking and analytics
- **Live Metrics**: Current counts updating during analysis
- **Detailed Dashboard**: Comprehensive analytics after stopping

## 🤖 AI Models Supported

### YOLO11 Models
| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| `yolo11n.pt` | ⚡⚡⚡ | ⭐⭐ | Real-time applications |
| `yolo11s.pt` | ⚡⚡ | ⭐⭐⭐ | Balanced performance |
| `yolo11m.pt` | ⚡ | ⭐⭐⭐⭐ | High accuracy needs |
| `yolo11l.pt` | 🐌 | ⭐⭐⭐⭐⭐ | Production quality |
| `yolo11x.pt` | 🐌🐌 | ⭐⭐⭐⭐⭐⭐ | Maximum accuracy |

### YOLO12 Models
- Same performance tiers available
- Enhanced accuracy and efficiency
- Segmentation variants with `-seg` suffix

### Device Support
- **CPU**: Universal compatibility, good for testing
- **CUDA**: GPU acceleration for production workloads

## 📹 Video Source Support

Both applications support multiple input sources:

- **📷 Webcam**: Default camera or specific camera index
- **📁 Video Files**: Upload MP4, AVI, MOV files
- **🌐 Stream URLs**: RTSP, HTTP video streams
- **📺 YouTube**: Automatic stream extraction
- **🔄 Real-time Processing**: Live analysis capabilities

## 🔧 Technical Requirements

### System Requirements
- **Python**: 3.9 or higher
- **Memory**: 4GB+ RAM recommended
- **GPU**: Optional NVIDIA GPU for CUDA acceleration
- **Storage**: 2GB+ for models and dependencies

### Key Dependencies
- **Streamlit**: Web interface framework
- **Ultralytics**: YOLO model implementation
- **OpenCV**: Computer vision operations
- **PyTorch**: Deep learning backend
- **DeepSort**: Object tracking (Zone Analytics)

## 🎯 Use Cases

### Object Detection & Segmentation
- **Security Monitoring**: Real-time surveillance analysis
- **Traffic Analysis**: Vehicle and pedestrian detection
- **Retail Analytics**: Customer behavior analysis
- **Quality Control**: Manufacturing defect detection

### Zone Flow Analytics
- **Crowd Management**: Monitor area occupancy and flow
- **Retail Footfall**: Track customer movement patterns  
- **Security Zones**: Perimeter monitoring and alerts
- **Traffic Flow**: Intersection and lane analysis
- **Event Analytics**: Venue capacity and movement analysis

## 🚀 Performance Tips

### For Real-time Performance
- Use smaller models (nano/small) for live processing
- Enable GPU acceleration when available
- Reduce video resolution for better FPS
- Optimize zone complexity for tracking applications

### For Maximum Accuracy
- Use larger models (large/extra-large) for detailed analysis
- Process video files instead of live streams
- Use segmentation models for precise boundaries
- Fine-tune detection thresholds for specific use cases

## 🔧 Troubleshooting

### Common Issues
- **Model Download**: First run requires internet for model download
- **Performance**: Try smaller models or reduce video quality
- **Video Sources**: Verify format support and source accessibility
- **Dependencies**: Use conda environments to avoid conflicts

### Getting Help
- Check individual project READMEs for specific guidance
- Verify all dependencies are installed correctly
- Test with demo data before using custom sources
- Use CPU mode if GPU setup is problematic

## 🤝 Contributing

This repository was developed as part of an AI workshop to demonstrate practical computer vision applications. 

### Development Setup
1. Fork the repository
2. Create feature branches for improvements
3. Test changes with both applications
4. Submit pull requests with detailed descriptions

### Areas for Improvement
- Additional model support (YOLOv10, custom models)
- Enhanced analytics features
- Mobile-optimized interfaces
- Cloud deployment options
- Integration with external databases

## 📄 License

This project is part of an AI Workshop CV Demo. Please refer to individual project licenses and dependencies for specific terms.

## 🎓 Learning Resources

This repository serves as a practical introduction to:
- **Computer Vision**: Object detection and tracking
- **Deep Learning**: YOLO model implementation
- **Web Development**: Streamlit application development
- **Data Analytics**: Real-time metrics and visualization
- **Software Engineering**: Project structure and deployment

Perfect for students, developers, and anyone interested in AI-powered computer vision applications!

---

**Ready to explore AI-powered computer vision?** Choose your adventure and start building amazing applications! 🚀
