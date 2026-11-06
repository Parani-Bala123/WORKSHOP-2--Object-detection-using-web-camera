# WORKSHOP 2 – OBJECT DETECTION USING WEB CAMERA

**Name:** Parani Bala M  
**Register Number:** 212224230192

## AIM

To develop a real-time object detection system using the YOLOv8 model and a webcam to identify and detect objects from live video frames.

## OBJECTIVES

1. To understand the working of the YOLOv8 object detection model.
2. To access the laptop webcam using OpenCV.
3. To detect multiple objects in real time.
4. To display detected objects with bounding boxes, labels and confidence scores.
5. To understand real-time image processing using Python.

## ALGORITHM

1. Start the program.
2. Import the required libraries such as YOLO, OpenCV and Matplotlib.
3. Load the pretrained YOLOv8 Nano model (`yolov8n.pt`).
4. Open the laptop webcam using `cv2.VideoCapture(0)`.
5. Check whether the webcam is opened successfully.
6. Capture video frames continuously from the webcam.
7. Pass each captured frame to the YOLO model for object detection.
8. Set the confidence threshold to 0.60.
9. Draw bounding boxes, object labels and confidence scores on the detected objects.
10. Resize the annotated frame and convert it into JPEG format.
11. Display the processed frame in the Jupyter Notebook.
12. Continue the process until the execution is interrupted.
13. Release the webcam and stop the program.

## PROGRAM

```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from IPython.display import display, Image, clear_output

# Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")

else:
    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame.")
                break

            # Perform object detection
            results = model(
                frame,
                conf=0.60,
                imgsz=640,
                verbose=False
            )

            # Draw bounding boxes and labels
            annotated_frame = results[0].plot()

            # Resize frame
            annotated_frame = cv2.resize(
                annotated_frame, (640, 480)
            )

            # Convert frame to JPEG
            ret, buffer = cv2.imencode(
                '.jpg', annotated_frame
            )

            # Display output in Jupyter Notebook
            if ret:
                clear_output(wait=True)
                display(Image(data=buffer.tobytes()))

    except KeyboardInterrupt:
        print("Detection stopped.")

    finally:
        # Release webcam
        cap.release()
        print("Webcam released successfully.")
```

## OUTPUT

<img width="1372" height="611" alt="image" src="https://github.com/user-attachments/assets/848450bf-5bb0-4b65-a6a8-b4068b866cd3" />



## RESULT

Thus, real-time object detection using the YOLOv8 Nano model and laptop webcam was successfully implemented using Python and OpenCV. The system detects objects from live video frames and displays their labels, bounding boxes and confidence scores.
