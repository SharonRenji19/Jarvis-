from ultralytics import YOLO
import cv2
import math
import time

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Load the YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", 
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
              "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
              "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", 
              "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
              "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
              "toothbrush"]

# Initialize last_capture_time
last_capture_time = time.time()

while True:
    success, img = cap.read()
    
    # Check if the image was captured successfully
    if not success:
        print("Failed to capture image")
        break  # Exit the loop if the image capture fails

    # Process the image only every 5 seconds
    current_time = time.time()
    if current_time - last_capture_time >= 5:  # Check if 5 seconds have passed
        results = model(img, stream=True)  # Process the image with the model
        
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values
                
                # Draw bounding box and label on the image
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, f"{classNames[int(box.cls[0])]} {math.ceil(box.conf[0] * 100)}%", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        last_capture_time = current_time  # Update the last capture time

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()