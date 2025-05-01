👁️ Real-Time Object Detection using YOLOv4
My First Computer Vision Project | Detecting and Counting People in Video 🎥

📽️ Demo
![image](https://github.com/user-attachments/assets/c226b170-a87d-4f6f-97fa-ba83c791d226)


🧠 About the Project
This is my first hands-on project in Computer Vision, where I’ve implemented real-time object detection using the YOLOv4 model. The system detects and counts the number of persons appearing in each frame of a video.

🔧 Technologies Used
Python

OpenCV

YOLOv4

Deep Learning (Darknet)

Pre-trained weights (on COCO dataset)

🚀 Features
Detects multiple objects in real-time (focus: persons)

Displays bounding boxes with confidence scores

Counts and displays the number of persons in each frame

Supports video input and real-time processing

📁 Files
yolov4-p6.cfg: YOLOv4 configuration file

yolov4-p6.weights: Pre-trained weights

coco.names: Label names for COCO dataset

main.py: Main Python script

sample3.mp4: Input video file

🛠️ Areas for Improvement
⏱️ Reduce frame processing time

🎯 Increase detection accuracy

📦 Detect more object classes

🧠 Upgrade to YOLOv5/YOLOv8 for better performance

🔄 Add object tracking for smoother results

▶️ How to Run
Install dependencies:

pip install opencv-python numpy
Place yolov4-p6.weights, yolov4-p6.cfg, and coco.names in the same directory.

Run the script:
python main.py
📌 Output Sample
Detected Persons: 3
Detected Persons: 1
Detected Persons: 5
