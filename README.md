# 👤 Face Detection & Face Embedding API (YOLO + Dlib + FastAPI)

This project provides a production-ready **FastAPI backend** for detecting faces in images, extracting facial landmarks, and generating **128-dimensional face embeddings** using **YOLOv3** and **Dlib's face recognition model**.

The system also includes optional **image/video processing pipelines** that send frames to the API and visualize detections (bounding boxes + landmarks + embeddings).

---

## 🚀 Features

- **Face Detection** using YOLOv3 (WIDER FACE pretrained model)
- **Facial Landmark Detection** using Dlib’s 68-point shape predictor
- **Face Embeddings** using Dlib ResNet (128-dim descriptor)
- **FastAPI Endpoint**: `/process_frame/`
- **JSON Output** including:
  - Bounding boxes  
  - Confidence scores  
  - Landmarks  
  - Embedding vectors  
- **Video & Image Client Scripts**
- **Processes 3 frames per second from video**
- **Clean modular structure** for real-world use

---

## 📁 Project Structure

├── app.py # FastAPI backend
├── main.py # Image/video client processor
├── video_processing.py # Alternative video processing script
├── models.py # YOLO + Dlib detection & embedding logic
├── config.py # Paths & configuration
├── yolov3-wider_16000.weights
├── yolov3-face.cfg
├── shape_predictor_68_face_landmarks.dat
├── dlib_face_recognition_resnet_model_v1.dat
└── README.md

---

## 🛠️ Installation

### **1. Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### **2. Install dependencies**

```bash
pip install fastapi uvicorn opencv-python numpy dlib requests matplotlib
```

### **3. Download required model files**

Place these files in the project folder:

-yolov3-wider_16000.weights
-yolov3-face.cfg
-shape_predictor_68_face_landmarks.dat
-dlib_face_recognition_resnet_model_v1.dat

(These are large files; add them to .gitignore so =they don’t get pushed to GitHub.)

---

## ▶️ Running the API

Start your FastAPI server:

```bash
uvicorn app:app --reload
```

The API will start at:

```bash
http://127.0.0.1:8000
```

Interactive documentation available at:

```bash
http://127.0.0.1:8000/docs
```

---

## 📡 API Endpoint

**POST /process_frame/**

Send an image frame as *multipart/form-data*.

Example request (Python)

```bash
files = {'file': ('frame.jpg', img_bytes, 'image/jpeg')}
response = requests.post("http://127.0.0.1:8000/process_frame/", files=files)
```

Example JSON response:

```bash
{
  "face_count": 1,
  "faces": [
    {
      "face_id": 0,
      "bbox": [120, 55, 80, 80],
      "confidence": 0.94,
      "landmarks": [[x1, y1], [x2, y2], ...],
      "embedding": [0.123, -0.452, ...128 dims...]
    }
  ]
}
```

---

## 🖼️ Image Processing Client

You can process a single image and visualize the detections:

```bash
python main.py
```

This script will:

-Send the image to the API
-Draw bounding boxes
-Draw 68 facial landmark points
-Show the output using Matplotlib

## 🎥 Video Processing Client

Use the API with video frames:

```bash
video_processing(config.video_path, config.api_url)
```

The script processes 3 frames per second, extracts:

-Face count
-Bounding boxes
-Landmarks
-Embeddings

and prints structured JSON.

---

## 🧠 Core Logic Overview

**YOLO Face Detection**

*models.py* loads YOLOv3 and performs:

-Forward pass through the network
-Non-max suppression
-Confidence filtering

**Facial Landmark Prediction**

Using *shape_predictor_68_face_landmarks.dat*

Embedding Extraction

Using:

```bash
face_rec_model.compute_face_descriptor(...)
```

Outputs a 128-dim embedding vector, used for:

F-ace recognition
-Re-identification
-Person tracking
-Feature clustering

---

## ⚙️ Configuration (config.py)

```bash
weights_path = "yolov3-wider_16000.weights"
config_path = "yolov3-face.cfg"

video_path = "crowd.mp4"
image_path = "group.jfif"

api_url = "http://127.0.0.1:8000/process_frame/"

shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
```

---

👨‍💻 Author

Mustafa Tariq

Master In Applied Artificial Intelligence

AI Engineer • Computer Vision • FastAPI
