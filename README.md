# Worksop_CV_Demo
Dive into the world of AI! This repository is a playground of mini-projects and apps, designed to spark curiosity and empower young creators to build their own amazing AI adventures.

# Object Detection and Segmentation with Streamlit

This project demonstrates object detection and segmentation using YOLOv8 and Streamlit. It allows you to process video streams from webcams, stream URLs, and YouTube videos, displaying detections and segmentations in real-time.

## To Run Locally

Clone the repository and follow the instructions below:

### Prerequisites

-   Python 3.9+
-   Git (for cloning the repository)

### Steps

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/ShudarshanKongkham/Worksop_CV_Demo.git](https://github.com/ShudarshanKongkham/Worksop_CV_Demo.git)
    ```
2.  **Change Directory:**
    From the repository directory change directory

    ```bash
    cd ObjectDet_Seg
    ```
3.  **Set up a Conda Environment (Recommended):**

    It's strongly recommended to use a Conda environment to manage dependencies. This ensures that your project's dependencies are isolated and don't conflict with other Python projects.

    *   **Create the environment:**

        ```bash
        conda create -n my-yolo-env python=3.10
        ```

        *(You can change 'my-yolo-env' to the environment name you prefer.)*

    *   **Activate the environment:**

        ```bash
        conda activate my-yolo-env
        ```
        
4.  **Install Requirements:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Application:**

    ```bash
    streamlit run streamlit_detect_Segment.py
    ```
     *(Ensure your Conda environment is activated before running this command)*
    *Note: The YOLOv11 model (`yolo11n-seg.pt`) will be downloaded automatically on the first run.*

## Input Sources

The application supports the following input sources, selectable via the sidebar:

-   **Webcam:** Uses your computer's default webcam.
-   **Stream URL:**  Enter a video stream URL (e.g., an RTSP URL).
-   **YouTube URL:** Enter a YouTube video URL.

## Notes
- The pre-trained YOLOv8 model will be automatically downloaded the first time you run the application.
- A "Stop" button in the sidebar allows you to terminate the video processing.
