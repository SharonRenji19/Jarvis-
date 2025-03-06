# Jarvis-
Jarvis for Image detection and Speech which we might add in future 
Run the code in any suitable platform , VScode is recommended

Install the required libraried like
pip install opencv-python , Its is an open library for the Computer Vision
pip install ultralytics ( This imports the YOLO (You Only Look Once) model from the Ultralytics library, which is a popular implementation of the YOLO object detection algorithm)
import cv2: This imports the OpenCV library, which is used for computer vision tasks, including capturing video from the camera and processing images.
import math: This imports the math library, which is used for mathematical operations (in this case, for rounding confidence scores).

Initialize Webcam:
cap = cv2.VideoCapture(0): This initializes the video capture object. The argument 0 typically refers to the default camera (your laptop's integrated camera).
cap.set(3, 640): This sets the width of the video frame to 640 pixels.
cap.set(4, 480): This sets the height of the video frame to 480 pixels.
Load the YOLO Model:
This line loads a pre-trained YOLO model from the specified path. The model file (yolov8n.pt) contains the weights and architecture of the YOLO network. You need to ensure that this file is available in the specified directory.

Start Detection Loop:
while True:: This creates an infinite loop that will continuously capture frames from the webcam.
success, img = cap.read(): This reads a frame from the webcam. success is a boolean indicating if the frame was captured successfully, and img is the captured image.

Object Detection:
results = model(img, stream=True): This line passes the captured image to the YOLO model for object detection. The stream=True argument allows for processing video streams.

Process Detection Results:
for r in results:: This iterates over the results returned by the model.
boxes = r.boxes: This extracts the bounding boxes of detected objects from the result.
